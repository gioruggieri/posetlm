# PosetLM (Low‑VRAM, research‑ready)


This repository contains a **single, academic** implementation of **PosetLM** (no Transformer baseline), a sparse DAG language model with edge‑wise `sigmoid^(1/τ)` gating and dynamic Top‑K. The script `posetlm.py` supports BYTE/WORD/BPE tokenization and both local `.tokens` files and HuggingFace **WikiText‑103 (raw)** with explicit **test** split.


## Why PosetLM (vs Transformer)?
Transformers compute dense attention over all past positions (`O(T²)`). PosetLM restricts each token *j* to up to `K` parents within a sliding window `W`, forming a sparse **DAG**. Messages from parents are gated with a logistic score and iteratively aggregated:


- For each edge *(i→j)* and head *h*: `a_ij = sigmoid(q_j · k_i / √d_h + b_Δ)^(1/τ)` with relative bias `b_Δ` and temperature `τ`.
- Optional **dynamic Top‑K** keeps only the strongest edges per destination token.
- We accumulate messages `S_B(j)` and normalizers `S_Z(j)` across parents and (optionally) across a few **poset iters**; final state is a normalized average `h_j = S_B(j) / S_Z(j)`.


**Benefits**
- **Lower VRAM / better scaling**: complexity ≈ `O(B·H·T·K·d_h)` vs `O(B·H·T²·d_h)`.
- **Longer context on small GPUs**: fixed `K` keeps cost predictable; `W` controls receptive field.
- **Structured sparsity**: Top‑K and edge‑dropout regularize; relative bias rewards short‑range deps without forbidding long ones.
- **Stable eval**: train with multiple `poset_iters` but use `poset_iters_eval=1` for faster evaluation.


**Trade‑offs**
- Global interactions are approximated via multi‑hop propagation across iters/windows; very long‑range, single‑hop patterns can be harder than with dense attention.


## Install
```bash
python -m venv .venv && source .venv/bin/activate # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
