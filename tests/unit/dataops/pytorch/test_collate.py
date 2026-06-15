import torch

from jaeger.dataops.pytorch.collate import pad_collate


def test_pad_collate_variable_length_2d():
    x1 = torch.randn(6, 50)
    y1 = torch.tensor([1.0, 0.0, 0.0])
    mask1 = torch.ones(6, 50, dtype=torch.bool)
    mask1[:, -10:] = False

    x2 = torch.randn(6, 30)
    y2 = torch.tensor([0.0, 1.0, 0.0])
    mask2 = torch.ones(6, 30, dtype=torch.bool)

    batch = [(x1, y1, mask1), (x2, y2, mask2)]
    batch_x, batch_y, batch_mask = pad_collate(batch)

    assert batch_x.shape == (2, 6, 50)
    assert batch_y.shape == (2, 3)
    assert batch_mask.shape == (2, 6, 50)

    # First sample should be unchanged except for stacking.
    assert torch.equal(batch_x[0], x1)
    assert torch.equal(batch_mask[0], mask1)

    # Second sample should be padded to length 50.
    assert torch.equal(batch_x[1, :, :30], x2)
    assert torch.equal(batch_x[1, :, 30:], torch.zeros(6, 20))
    assert torch.equal(batch_mask[1, :, :30], mask2)
    assert not batch_mask[1, :, 30:].any()


def test_pad_collate_variable_length_3d():
    x1 = torch.randn(6, 50, 4)
    y1 = torch.tensor([1.0, 0.0, 0.0])
    mask1 = torch.ones(6, 50, dtype=torch.bool)
    mask1[:, -5:] = False

    x2 = torch.randn(6, 30, 4)
    y2 = torch.tensor([0.0, 1.0, 0.0])
    mask2 = torch.ones(6, 30, dtype=torch.bool)

    batch = [(x1, y1, mask1), (x2, y2, mask2)]
    batch_x, batch_y, batch_mask = pad_collate(batch)

    assert batch_x.shape == (2, 6, 50, 4)
    assert batch_y.shape == (2, 3)
    assert batch_mask.shape == (2, 6, 50)

    assert torch.equal(batch_x[0], x1)
    assert torch.equal(batch_mask[0], mask1)

    assert torch.equal(batch_x[1, :, :30, :], x2)
    assert torch.equal(batch_x[1, :, 30:, :], torch.zeros(6, 20, 4))
    assert torch.equal(batch_mask[1, :, :30], mask2)
    assert not batch_mask[1, :, 30:].any()


def test_pad_collate_no_padding_needed():
    x1 = torch.randn(6, 40)
    y1 = torch.tensor([1.0, 0.0, 0.0])
    mask1 = torch.ones(6, 40, dtype=torch.bool)
    x2 = torch.randn(6, 40)
    y2 = torch.tensor([0.0, 1.0, 0.0])
    mask2 = torch.ones(6, 40, dtype=torch.bool)

    batch = [(x1, y1, mask1), (x2, y2, mask2)]
    batch_x, batch_y, batch_mask = pad_collate(batch)

    assert torch.equal(batch_x, torch.stack([x1, x2]))
    assert torch.equal(batch_y, torch.stack([y1, y2]))
    assert torch.equal(batch_mask, torch.stack([mask1, mask2]))
