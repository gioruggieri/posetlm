# PosetLM (Low-VRAM, research-ready)

**PosetLM** is a *Transformer-alternative* language model that builds a sparse **poset/DAG** over the context: each token connects to at most `K` parents within a sliding window `W`. Messages flow along edges with a logistic gate and are iteratively aggregated. This yields **predictable memory/compute** and longer contexts on small GPUs — while keeping strong perplexity.

This repository is intentionally **academic and minimal**: it contains **only PosetLM** (no Transformer baseline) in a single script, `posetlm.py`. It supports BYTE/WORD/BPE tokenization and both:
- local `*.tokens` files (Merity’s WikiText-103 format), and
- HuggingFace datasets (`wikitext-103-raw-v1`) with **validation and test** splits.

---

## Why PosetLM (vs Transformer)?

**Transformers** compute dense causal attention (`O(T²)`) across all past positions. **PosetLM** instead restricts each position *j* to at most `K` parents among the previous `W` tokens, forming a **sparse DAG**.

**Benefits**
- **Lower VRAM & predictable scaling:** cost ≈ `O(B·H·T·K·d_h)` (fixed `K`), not `O(T²)`.
- **Longer contexts on small GPUs:** increase `W` without quadratic blow-up.
- **Regularization:** structured sparsity (Top-K) + edge dropout.
- **Fast evaluation:** train with `poset_iters>1`, evaluate with `poset_iters_eval=1`.

**Trade-offs**
- Single-hop, very long-range interactions may require multiple iterations or windows (multi-hop propagation), whereas dense attention gets them in one hop.

---

## Method in a nutshell

For each edge *(i → j)* and head *h*:
- score: `logit_ij = (q_j · k_i) / √d_h + b_Δ`, where `Δ = j − i` and `b_Δ` is a learnable relative bias;
- gate: `I_ij = sigmoid(logit_ij)`, weight: `a_ij = I_ij^(1/τ)` with temperature `τ`;
- **dynamic Top-K** (optional): per destination *j*, keep only the top-`K' ≤ K` edges by `a_ij`.

Aggregate messages and normalizers over parents (and over a few **poset iterations** during training):
```
S_B(j) = Σ_i a_ij · v_i
S_Z(j) = Σ_i a_ij
h_j    = S_B(j) / max(S_Z(j), ε)
```
Repeat a small number of iterations at train time; switch to `poset_iters_eval=1` at eval for speed.

---

## Install

> Python **3.10+** recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart

### A) WikiText-103 (raw) via HuggingFace

**WORD (word-level PPL)**

```bash
python posetlm.py --dataset hf_wikitext103_raw --tokenizer word \
  --seq_len 512 --batch_size 6 --grad_accum 2 --steps 100000 \
  --scheduler cosine --warmup 4000 --min_lr 3e-5 --lr 2e-4 \
  --k_parents 24 --window 256 --poset_iters 3 --poset_iters_eval 1 \
  --dynamic_topk --topk 12 --fp16_cache --amp --dropout 0.1 \
  --adaptive_softmax --cutoffs "2000,10000,50000"
```

**BPE (subword-level, vocab 16k)**

```bash
python posetlm.py --dataset hf_wikitext103_raw --tokenizer bpe \
  --bpe_vocab_size 16000 --bpe_model ./bpe_wt103_16k.json \
  --seq_len 512 --batch_size 6 --grad_accum 2 --steps 100000 \
  --scheduler cosine --warmup 4000 --min_lr 3e-5 --lr 2e-4 \
  --k_parents 24 --window 256 --poset_iters 3 --poset_iters_eval 1 \
  --dynamic_topk --topk 12 --fp16_cache --amp --dropout 0.1 \
  --adaptive_softmax --cutoffs "2000,8000,14000"
```

### B) Local `.tokens` files (Merity)

```bash
python posetlm.py --dataset text --tokenizer word \
  --text_file "D:/path/to/wiki.train.tokens" \
  --val_file  "D:/path/to/wiki.valid.tokens" \
  --word_vocab_size 0 --append_eos
```

> **Note:** for `*.tokens` files the script automatically disables extra `<eos>` insertion.  
> **Adaptive Softmax cutoffs** are sanitized automatically (sorted, unique, and `< V`).

---

## CLI (common options)

```
--dataset {text,hf_wikitext2,hf_wikitext103_raw}
--tokenizer {byte,word,bpe}
--text_file --val_file --test_file        # for dataset=text
--bpe_vocab_size --bpe_model              # for tokenizer=bpe
--word_vocab_size 0                       # 0 = use entire train vocab (WT103-compat)
--seq_len --batch_size --steps --grad_accum
--k_parents --window --poset_iters --poset_iters_eval
--dynamic_topk --topk
--dropout --edge_dropout --fp16_cache --node_chunk
--lr --scheduler cosine --warmup --min_lr --weight_decay
--grad_clip --amp
--compile --grad_checkpoint
--save_dir --out_csv
--seed --deterministic_final --append_eos
--adaptive_softmax --cutoffs "2000,10000,50000"
```

---

## Tips (VRAM ↔ quality)

### Reduce VRAM
- Decrease `--k_parents`, enable `--dynamic_topk --topk <smaller>`.
- Use `--node_chunk 128` or `256`.
- Enable `--fp16_cache` and `--amp`.
- Lower `--d_model` / `--n_layers`.

### Improve quality
- Increase `--k_parents`, `--window`, and train with `--poset_iters 2–3` (keep `--poset_iters_eval 1`).
- Tune `--tau` (~0.05–0.1) and `--dropout` (0.1–0.3).
- Use Adaptive Softmax for large vocabularies.

---

## Reproducibility & logging
- Set `--seed` (default `1337`) and keep `--deterministic_final` for reproducible final eval.
- Metrics are appended to `metrics.csv` with timestamps, loss, PPL, tokens/s, peak memory, and LR.
- Checkpoints saved to `checkpoints/` (`last.pt` and best-by-val `best.pt`).

---

## Project layout

```
posetlm.py         # the model/training script (PosetLM only)
README.md
requirements.txt
LICENSE
tests/
  test_smoke.py    # tiny CPU smoke test
.github/workflows/
  ci.yml           # CPU-only CI (pytest) on push/PR
```

---

## License

[MIT](./LICENSE) © 2025 **Giovanni Ruggieri**

---

## Contact

Please open an Issue on this repository.  
Maintainer: **Giovanni Ruggieri** — 21227260+gioruggieri@users.noreply.github.com (noreply)

