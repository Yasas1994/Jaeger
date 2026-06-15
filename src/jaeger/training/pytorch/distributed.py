import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl") -> bool:
    """Initialize process group if multi-GPU / SLURM environment is detected."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return True

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True

    return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device() -> torch.device:
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
