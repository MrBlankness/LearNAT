# Main entry point for SFT and MDPO training.
#
# Usage:
#   python train.py --mode sft --model_name Qwen/Qwen2.5-Coder-7B --data_path ./data/sft_data.json
#   python train.py --mode mdpo --model_name ./outputs/sft_checkpoint --data_path ./data/dpo_data.json

import os
import sys
import json
import logging
import argparse
from typing import Optional, Dict, Any

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf, DictConfig
import wandb

from trainers import BasicTrainer, FSDPTrainer, TrainerConfig, disable_dropout
from utils import (
    get_local_dir, get_local_run_dir, get_open_port,
    init_distributed, rank0_print, set_seed
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True


def load_training_data(data_path: str) -> list:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} training samples from {data_path}")
    return data


def worker_main(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    policy: torch.nn.Module,
    reference_model: Optional[torch.nn.Module] = None
):
    """Training loop executed by each worker process."""
    # Initialize distributed if using FSDP
    if args.use_fsdp:
        init_distributed(rank, world_size, port=args.fsdp_port)
    
    # Initialize wandb on rank 0
    if rank == 0 and args.wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}_{args.mode}",
            config=vars(args)
        )
    
    # Create trainer config
    config = TrainerConfig(
        model_name_or_path=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        dpo_learning_rate=args.dpo_learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        use_margin=args.use_margin,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project
    )
    
    # Create trainer
    TrainerClass = FSDPTrainer if args.use_fsdp else BasicTrainer
    trainer = TrainerClass(
        policy=policy,
        config=config,
        seed=args.seed,
        output_dir=args.output_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size
    )
    
    # Load training data
    train_data = load_training_data(args.data_path)
    
    # Run training
    if args.mode == 'sft':
        logger.info("Starting SFT training...")
        metrics = trainer.train_sft(train_data)
    elif args.mode in ['dpo', 'ipo', 'mdpo']:
        logger.info(f"Starting {args.mode.upper()} training...")
        metrics = trainer.train_dpo(train_data, loss_type=args.mode)
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")
    
    # Save final model
    if rank == 0:
        trainer.save("final")
        logger.info(f"Training completed. Final metrics: {metrics}")
    
    # Close wandb
    if rank == 0 and args.wandb_enabled:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train LearNAT model")
    
    # Mode
    parser.add_argument("--mode", type=str, default="sft",
                       choices=["sft", "dpo", "ipo", "mdpo"],
                       help="Training mode")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B",
                       help="Model name or path")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save GPU memory")
    
    # Data settings
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data JSON file")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate for SFT")
    parser.add_argument("--dpo_learning_rate", type=float, default=2e-6,
                       help="Learning rate for DPO")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    
    # DPO settings
    parser.add_argument("--beta", type=float, default=0.2,
                       help="Beta for DPO/MDPO")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="Label smoothing factor")
    parser.add_argument("--use_margin", action="store_true",
                       help="Use AST margin in MDPO")
    
    # Distributed training
    parser.add_argument("--use_fsdp", action="store_true",
                       help="Use FSDP for distributed training")
    parser.add_argument("--fsdp_port", type=int, default=None,
                       help="Port for FSDP")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=500,
                       help="Save checkpoint every N steps")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--exp_name", type=str, default="learnat",
                       help="Experiment name")
    
    # Wandb
    parser.add_argument("--wandb_enabled", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="learnat",
                       help="Wandb project name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("LearNAT Training Script")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading policy model...")
    model_kwargs = {}
    if not args.use_fsdp:
        model_kwargs['device_map'] = 'auto'

    policy_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=policy_dtype,
        trust_remote_code=True,
        **model_kwargs
    )
    disable_dropout(policy)

    # Enable gradient checkpointing to save GPU memory
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for policy model...")
        policy.gradient_checkpointing_enable()

    # Load reference model for DPO - load to CPU to save GPU memory
    reference_model = None
    if args.mode in ['dpo', 'ipo', 'mdpo']:
        logger.info("Loading reference model onto CPU to save GPU memory...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=policy_dtype,
            trust_remote_code=True,
            # Note: no device_map='auto', load to CPU
        )
        disable_dropout(reference_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        logger.info("Reference model loaded on CPU.")
    
    # Start training
    if args.use_fsdp:
        # Get open port if not specified
        if args.fsdp_port is None:
            args.fsdp_port = get_open_port()
            logger.info(f"Using FSDP port: {args.fsdp_port}")
        
        # Start multiple processes
        world_size = torch.cuda.device_count()
        logger.info(f"Starting {world_size} processes for FSDP training")
        
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, args, policy, reference_model),
            join=True
        )
    else:
        # Single process training
        worker_main(0, 1, args, policy, reference_model)


if __name__ == "__main__":
    main()
