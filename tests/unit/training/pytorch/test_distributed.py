import os

from jaeger.training.pytorch.distributed import (
    get_device,
    is_main_process,
    setup_distributed,
)


def test_get_device_without_distributed():
    device = get_device()
    assert device.type in ("cpu", "cuda")


def test_is_main_process_without_distributed():
    assert is_main_process() is True


def test_setup_distributed_returns_false_outside_slurm():
    # Ensure no SLURM / torchrun env vars are present
    for key in ["RANK", "WORLD_SIZE", "SLURM_PROCID", "SLURM_NTASKS"]:
        os.environ.pop(key, None)
    assert setup_distributed() is False
