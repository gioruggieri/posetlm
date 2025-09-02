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
tests/test_poset_forward.py
import torch
from posetlm import PosetLM, build_edges

def test_forward_shapes():
    V = 256
    model = PosetLM(vocab_size=V, d_model=64, n_layers=2, n_poset_heads=2, k_parents=4)
    B, T = 2, 16
    x = torch.randint(0, V, (B, T))
    edges = tuple(t for t in build_edges(T, 4, 8))
    with torch.no_grad():
        feats = model(x, edges)
    assert feats.shape == (B, T, 64)
