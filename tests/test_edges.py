import torch
from posetlm import build_edges

def test_build_edges_shapes():
    T, K, W = 32, 4, 16
    src, dst, delta, ptr, slot = build_edges(T, K, W)
    assert src.shape == dst.shape == slot.shape
    assert delta.dtype in (torch.int16, torch.long)
    assert ptr.shape[0] == T + 1
    assert torch.all(ptr[1:] >= ptr[:-1])
    assert ptr[-1].item() == src.numel()
