import numpy as np


def test_torch():
    """Smoke test: run a small PyTorch matmul on the CPU."""
    try:
        import torch

        matrix_size = 100
        matrix_a = torch.from_numpy(
            np.random.rand(matrix_size, matrix_size).astype(np.float32)
        )
        matrix_b = torch.from_numpy(
            np.random.rand(matrix_size, matrix_size).astype(np.float32)
        )
        result = torch.matmul(matrix_a, matrix_b)
    except Exception as e:
        return e
    else:
        return result
