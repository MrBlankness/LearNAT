# Distributed training helpers, seed management, and misc utilities.

import os
import getpass
import socket
import random
from datetime import datetime
from typing import Dict, List, Optional, Union, Type

import torch
import torch.distributed as dist
import numpy as np


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return a per-user cache directory under the first existing prefix."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    
    # Create the last prefix if none exist
    os.makedirs(prefixes_to_resolve[-1], exist_ok=True)
    return f"{prefixes_to_resolve[-1]}/{getpass.getuser()}"


def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a timestamped run directory."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = 'localhost',
    port: int = 12355,
    backend: str = 'nccl'
) -> None:
    """Set up torch.distributed."""
    print(f"Rank {rank}: initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def rank0_print(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def slice_and_move_batch_for_device(
    batch: Dict,
    rank: int,
    world_size: int,
    device: str
) -> Dict:
    """Shard a batch across workers and move to device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in sliced.items()
    }
    return on_device


def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1
) -> torch.Tensor:
    """Pad or truncate *tensor* along *dim* to exactly *length*."""
    if tensor.size(dim) >= length:
        return tensor.narrow(dim, 0, length)
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],
            dim=dim
        )


def all_gather_if_needed(
    values: torch.Tensor,
    rank: int,
    world_size: int
) -> torch.Tensor:
    """all_gather when distributed, identity otherwise."""
    if world_size == 1:
        return values
    
    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    return {
        k: (f"{v:.5g}" if isinstance(v, float) else v)
        for k, v in d.items()
    }


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: Optional[int] = None, message: str = '') -> None:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank}] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
            print('*' * 40)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporarilySeededRandom:
    """Context manager that temporarily overrides the RNG state."""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None
        self.stored_torch_state = None

    def __enter__(self):
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()
        self.stored_torch_state = torch.get_rng_state()
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
        torch.set_rng_state(self.stored_torch_state)


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Return param counts and estimated memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size / (1024 ** 2),
        "buffer_size_mb": buffer_size / (1024 ** 2),
        "total_size_mb": (param_size + buffer_size) / (1024 ** 2)
    }


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_text(text: str, max_tokens: int, tokenizer) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


if __name__ == "__main__":
    # Test utilities
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test seed setting
    print("\n1. Testing set_seed...")
    set_seed(42)
    print(f"Random: {random.random():.4f}")
    print(f"NumPy: {np.random.random():.4f}")
    print(f"PyTorch: {torch.rand(1).item():.4f}")
    
    # Test temporary seeding
    print("\n2. Testing TemporarilySeededRandom...")
    set_seed(42)
    val1 = random.random()
    with TemporarilySeededRandom(0):
        val_temp = random.random()
    val2 = random.random()
    print(f"Before context: {val1:.4f}")
    print(f"Inside context: {val_temp:.4f}")
    print(f"After context: {val2:.4f}")
    
    # Test port finding
    print("\n3. Testing get_open_port...")
    port = get_open_port()
    print(f"Found open port: {port}")
    
    # Test formatted dict
    print("\n4. Testing formatted_dict...")
    d = {"loss": 0.123456789, "accuracy": 0.98765, "epoch": 1}
    print(f"Original: {d}")
    print(f"Formatted: {formatted_dict(d)}")
    
    print("\n" + "=" * 60)
    print("All utility tests completed!")
    print("=" * 60)
