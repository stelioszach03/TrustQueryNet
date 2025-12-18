import torch

from trustquerynet.training.reproducibility import set_seed


def test_deterministic_flag_enables_torch_setting():
    set_seed(7, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
