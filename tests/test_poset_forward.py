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
