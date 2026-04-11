# SFT and Margin-aware DPO (MDPO) trainers with optional FSDP support.

import torch
import torch.nn as nn
import torch.nn.functional as F
from prompt import format_schema
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import os
import wandb
from tqdm import tqdm
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)


@dataclass
class TrainerConfig:
    """Configuration for training."""
    # Model settings
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B"
    max_length: int = 2048
    
    # Training settings
    learning_rate: float = 2e-5
    dpo_learning_rate: float = 2e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # DPO settings
    beta: float = 0.2  # KL penalty coefficient
    label_smoothing: float = 0.0
    reference_free: bool = False
    
    # MDPO settings
    use_margin: bool = True
    
    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    
    # Paths
    output_dir: str = "./outputs"
    wandb_project: str = "learnat"
    

def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100
) -> torch.Tensor:
    """Per-sequence sum (or mean) of log-probs for the given labels."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape except for vocab dimension")
    
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    # Create loss mask
    loss_mask = labels != label_pad_token_id
    
    # Compute per-token log probabilities
    labels[labels == label_pad_token_id] = 0  # Dummy value for gather
    per_token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(2)
    
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "dpo",
    label_smoothing: float = 0.0,
    margin: Optional[torch.Tensor] = None,
    reference_free: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO / IPO / MDPO preference loss. Returns (loss, chosen_reward, rejected_reward)."""
    if reference_free:
        # No reference model - just use policy log probs
        chosen_rewards = beta * policy_chosen_logps
        rejected_rewards = beta * policy_rejected_logps
    else:
        # Standard implicit reward formulation
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    if loss_type == "dpo":
        # Standard DPO loss: -log(sigmoid(r_w - r_l))
        logits = chosen_rewards - rejected_rewards
        
        if label_smoothing > 0:
            # Smooth labels
            losses = (
                -F.logsigmoid(logits) * (1 - label_smoothing)
                - F.logsigmoid(-logits) * label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
    
    elif loss_type == "ipo":
        # IPO loss: (logits - 1/2β)^2
        logits = chosen_rewards - rejected_rewards
        losses = (logits - 1 / (2 * beta)) ** 2
    
    elif loss_type == "mdpo":
        # MDPO loss with AST margin
        # L_MDPO = -log(sigmoid(r_w - r_l - margin))
        logits = chosen_rewards - rejected_rewards
        
        if margin is not None:
            # Subtract margin offset
            logits = logits - margin
        
        if label_smoothing > 0:
            losses = (
                -F.logsigmoid(logits) * (1 - label_smoothing)
                - F.logsigmoid(-logits) * label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class NL2SQLDataset(Dataset):
    """Tokenized dataset for SFT or DPO training."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        mode: str = "sft"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.mode == "sft":
            return self._process_sft_item(item)
        else:
            return self._process_dpo_item(item)
    
    def _process_sft_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        # Build input prompt
        schema = item.get("schema", "")
        if isinstance(schema, dict):
            schema = format_schema(schema)
        prompt = self._build_prompt(
            item["question"],
            schema,
            item.get("evidence", "")
        )
        
        # Build target (decomposition sequence)
        target = self._build_target(item["decomposition"])
        
        # Tokenize
        full_text = prompt + target
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask prompt part)
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels = encoded["input_ids"].clone()
        labels[0, :prompt_len] = -100  # Mask prompt
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def _process_dpo_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        # Build context
        schema = item.get("schema", "")
        if isinstance(schema, dict):
            schema = format_schema(schema)
        context = self._build_prompt(
            item["question"],
            schema,
            item.get("evidence", "")
        )
        
        if "previous_steps" in item:
            for step in item["previous_steps"]:
                context += f"\nSubtask: {step[0]}\nSQL: {step[1]}"
        
        # Build chosen and rejected responses
        chosen = f"\nSubtask: {item['chosen'][0]}\nSQL: {item['chosen'][1]}"
        rejected = f"\nSubtask: {item['rejected'][0]}\nSQL: {item['rejected'][1]}"
        
        # Tokenize
        chosen_encoded = self.tokenizer(
            context + chosen,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        rejected_encoded = self.tokenizer(
            context + rejected,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels
        context_len = len(self.tokenizer.encode(context, add_special_tokens=False))
        
        chosen_labels = chosen_encoded["input_ids"].clone()
        chosen_labels[0, :context_len] = -100
        
        rejected_labels = rejected_encoded["input_ids"].clone()
        rejected_labels[0, :context_len] = -100
        
        result = {
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        }
        
        # Add margin if available
        if "margin" in item:
            result["margin"] = torch.tensor(item["margin"])
        
        return result
    
    def _build_prompt(self, question: str, schema: str, evidence: str) -> str:
        prompt = f"""You are a data science expert. Given the following database schema and question, decompose the query into simpler subtasks.

Database Schema:
{schema}

Evidence:
{evidence}

Question: {question}

Please decompose this query into subtasks, providing both the natural language description and the corresponding SQL for each subtask.
"""
        return prompt
    
    def _build_target(self, decomposition: List[Tuple[str, str]]) -> str:
        target = ""
        for i, (query, sql) in enumerate(decomposition):
            target += f"\nSubtask {i+1}: {query}\nSQL: {sql}"
        return target


class BasicTrainer:
    """Single-GPU trainer supporting SFT, DPO, and MDPO."""
    
    def __init__(
        self,
        policy: nn.Module,
        config: TrainerConfig,
        seed: int = 42,
        output_dir: str = "./outputs",
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        self.policy = policy
        self.config = config
        self.seed = seed
        self.output_dir = output_dir
        self.reference_model = reference_model
        self.rank = rank
        self.world_size = world_size
        
        # Setup
        torch.manual_seed(seed)
        os.makedirs(output_dir, exist_ok=True)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            padding_side="right",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Optimizer will be created in train()
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.example_counter = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def train_sft(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """Standard cross-entropy SFT on decomposition trajectories."""
        self.logger.info("Starting SFT training...")
        
        # Create dataset and dataloader
        train_dataset = NL2SQLDataset(
            train_data,
            self.tokenizer,
            self.config.max_length,
            mode="sft"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.policy.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=self.rank != 0
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.policy.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.policy(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Logging
                if self.global_step % self.config.log_every == 0 and self.rank == 0:
                    progress_bar.set_postfix({
                        "loss": loss.item() * self.config.gradient_accumulation_steps,
                        "lr": self.scheduler.get_last_lr()[0]
                    })
                    
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": self.global_step
                        })
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            total_loss += avg_epoch_loss
            
            self.logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if self.rank == 0 and (epoch + 1) % 1 == 0:
                self.save(f"checkpoint-epoch-{epoch + 1}")
        
        return {"sft_loss": total_loss / self.config.num_epochs}
    
    def train_dpo(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        loss_type: str = "mdpo"
    ) -> Dict[str, float]:
        """DPO / IPO / MDPO preference training."""
        self.logger.info(f"Starting {loss_type.upper()} training...")
        
        # Freeze reference model
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # Create dataset and dataloader
        train_dataset = NL2SQLDataset(
            train_data,
            self.tokenizer,
            self.config.max_length,
            mode="dpo"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer with DPO learning rate
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.dpo_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.policy.train()
        metrics = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_accuracy": []
        }
        
        for epoch in range(self.config.num_epochs):
            epoch_metrics = {k: 0.0 for k in metrics}
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=self.rank != 0
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                device = self.policy.device
                
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)
                
                margin = batch.get("margin")
                if margin is not None:
                    margin = margin.to(device)
                
                # Get policy log probs
                chosen_outputs = self.policy(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                rejected_outputs = self.policy(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )
                
                policy_chosen_logps = get_batch_logps(
                    chosen_outputs.logits,
                    chosen_labels
                )
                policy_rejected_logps = get_batch_logps(
                    rejected_outputs.logits,
                    rejected_labels
                )
                
                # Get reference log probs
                if self.reference_model is not None:
                    with torch.no_grad():
                        # Run reference model on CPU to save GPU memory
                        ref_chosen_outputs = self.reference_model(
                            input_ids=chosen_input_ids.cpu(),
                            attention_mask=chosen_attention_mask.cpu()
                        )
                        ref_rejected_outputs = self.reference_model(
                            input_ids=rejected_input_ids.cpu(),
                            attention_mask=rejected_attention_mask.cpu()
                        )

                        ref_chosen_logps = get_batch_logps(
                            ref_chosen_outputs.logits.to(device),
                            chosen_labels
                        )
                        ref_rejected_logps = get_batch_logps(
                            ref_rejected_outputs.logits.to(device),
                            rejected_labels
                        )
                else:
                    ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                
                # Compute loss
                loss, chosen_reward, rejected_reward = preference_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.config.beta,
                    loss_type=loss_type,
                    label_smoothing=self.config.label_smoothing,
                    margin=margin if loss_type == "mdpo" else None,
                    reference_free=self.config.reference_free
                )
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Update metrics
                actual_loss = loss.item() * self.config.gradient_accumulation_steps
                epoch_metrics["loss"] += actual_loss
                epoch_metrics["chosen_reward"] += chosen_reward.item()
                epoch_metrics["rejected_reward"] += rejected_reward.item()
                
                reward_accuracy = (chosen_reward > rejected_reward).float().mean().item()
                epoch_metrics["reward_accuracy"] += reward_accuracy
                
                # Logging
                if self.global_step % self.config.log_every == 0 and self.rank == 0:
                    progress_bar.set_postfix({
                        "loss": actual_loss,
                        "reward_acc": reward_accuracy
                    })
                    
                    if wandb.run is not None:
                        wandb.log({
                            f"train/{loss_type}_loss": actual_loss,
                            "train/chosen_reward": chosen_reward.item(),
                            "train/rejected_reward": rejected_reward.item(),
                            "train/reward_accuracy": reward_accuracy,
                            "train/global_step": self.global_step
                        })
            
            # Average epoch metrics
            num_batches = len(train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
                metrics[key].append(epoch_metrics[key])
            
            self.logger.info(
                f"Epoch {epoch + 1} - Loss: {epoch_metrics['loss']:.4f}, "
                f"Reward Acc: {epoch_metrics['reward_accuracy']:.4f}"
            )
            
            # Save checkpoint
            if self.rank == 0:
                self.save(f"checkpoint-{loss_type}-epoch-{epoch + 1}")
        
        return {k: sum(v) / len(v) for k, v in metrics.items() if v}
    
    def save(self, name: str = "final") -> None:
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "config": vars(self.config)
        }
        
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        
        torch.save(state, os.path.join(save_path, "training_state.pt"))
        
        self.logger.info(f"Saved checkpoint to {save_path}")
    
    def load(self, path: str) -> None:
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            self.global_step = state.get("global_step", 0)
            
            if "optimizer" in state and self.optimizer is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state and self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
        
        self.logger.info(f"Loaded checkpoint from {path}")


class FSDPTrainer(BasicTrainer):
    """BasicTrainer extended with FSDP for multi-GPU training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Wrap model with FSDP
        if self.world_size > 1:
            self.policy = FSDP(
                self.policy,
                use_orig_params=True,
                device_id=self.rank
            )
            
            if self.reference_model is not None:
                self.reference_model = FSDP(
                    self.reference_model,
                    use_orig_params=True,
                    device_id=self.rank
                )
    
    def save(self, name: str = "final") -> None:
        """Save FSDP checkpoint."""
        if self.rank != 0:
            return
        
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save with FSDP state dict
        with FSDP.state_dict_type(
            self.policy,
            state_dict_type=torch.distributed.fsdp.StateDictType.FULL_STATE_DICT
        ):
            state_dict = self.policy.state_dict()
        
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"Saved FSDP checkpoint to {save_path}")


def disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


if __name__ == "__main__":
    # Test code
    print("=" * 60)
    print("Testing Trainer Module")
    print("=" * 60)
    
    # Create mock data
    sft_data = [
        {
            "question": "Find users over 18",
            "schema": "users(id, name, age)",
            "evidence": "age > 18 means adult",
            "decomposition": [
                ("Get all users", "SELECT * FROM users"),
                ("Filter adults", "SELECT * FROM users WHERE age > 18")
            ]
        }
    ]
    
    dpo_data = [
        {
            "question": "Find users over 18",
            "schema": "users(id, name, age)",
            "evidence": "",
            "previous_steps": [],
            "chosen": ("Get adult users", "SELECT * FROM users WHERE age > 18"),
            "rejected": ("Get all data", "SELECT * FROM users"),
            "margin": 0.5
        }
    ]
    
    print("Test data created successfully!")
    print(f"SFT samples: {len(sft_data)}")
    print(f"DPO samples: {len(dpo_data)}")
    
    print("\n" + "=" * 60)
    print("Module test completed!")
    print("=" * 60)
