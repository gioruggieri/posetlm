# -*- coding: utf-8 -*-
"""
PosetLM — WT103-ready (PosetLM-only) — BYTE/WORD/BPE
- Dataset: HuggingFace "wikitext-103-raw-v1" (val/test included) or local .tokens files
- Tokenizer: byte | word | bpe (with optional <eos>; auto-off for *.tokens)
- Training: dropout, proper weight decay, grad checkpoint (opt.), torch.compile (opt.)
- VRAM: edge sparsity (K, W), dynamic Top-K, fp16 cache, node chunking
- Adaptive Softmax with cutoff sanitization; PPL computed for the chosen tokenizer unit

Designed for ~8GB GPUs; scales to larger cards.
"""

import math, os, time, argparse, random, csv, json, contextlib
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ---------------------------- Utils ----------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self.value = None
    def update(self, x: float) -> float:
        x = float(x)
        self.value = x if self.value is None else self.beta * self.value + (1 - self.beta) * x
        return self.value

# ---------------------------- Tokenization helpers ----------------------------

def _read_bytes(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        data = f.read()
    return torch.tensor(list(data), dtype=torch.uint8)

def _iter_word_tokens(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            for tok in line.strip().split():
                if tok:
                    yield tok

def build_word_vocab(train_file: str, vocab_size: Optional[int] = None) -> Dict[str, int]:
    """
    If `vocab_size` is None or <= 0: include *all* tokens seen in train (WT103-compatible).
    Otherwise, keep top-(vocab_size-4) + 4 special tokens.
    """
    cnt = Counter()
    for tok in _iter_word_tokens(train_file):
        cnt[tok] += 1
    most = cnt.most_common() if (vocab_size is None or vocab_size <= 0) else cnt.most_common(max(1, vocab_size - 4))
    stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
    i = 4
    for t,_ in most:
        if t not in stoi:
            stoi[t] = i; i += 1
    return stoi

def save_vocab(stoi: Dict[str,int], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False)

def load_vocab(path: str) -> Dict[str,int]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_word_file(path: str, stoi: Dict[str,int], *, append_eos: bool = True) -> torch.Tensor:
    unk = stoi.get("<unk>", 1); eos_id = stoi.get("<eos>", 3)
    ids: List[int] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if not toks: continue
            for t in toks: ids.append(stoi.get(t, unk))
            if append_eos: ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)

class BPEWrap:
    def __init__(self, model_path: Optional[str], vocab_size: int, train_file: Optional[str]):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
        except Exception:
            raise RuntimeError("For --tokenizer bpe you need 'tokenizers' (pip install tokenizers)")
        if model_path and os.path.isfile(model_path):
            self.tok = Tokenizer.from_file(model_path)
        else:
            if not train_file:
                raise RuntimeError("BPE: provide an existing --bpe_model or a --text_file for training")
            tok = Tokenizer(BPE(unk_token="<unk>"))
            tok.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>","<unk>","<bos>","<eos>"])
            tok.train([train_file], trainer)
            self.tok = tok
            if model_path: tok.save(model_path)
    def encode_file(self, path: str, *, append_eos: bool = True) -> torch.Tensor:
        ids: List[int] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                out = self.tok.encode(line.strip())
                ids.extend(out.ids)
                if append_eos:
                    eos_id = self.tok.token_to_id("<eos>")
                    if eos_id is not None: ids.append(eos_id)
        return torch.tensor(ids, dtype=torch.long)
    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

# ---------------------------- Data containers & loaders ----------------------------

@dataclass
class SeqData:
    train: torch.Tensor  # CPU
    val: torch.Tensor    # CPU
    test: torch.Tensor   # CPU
    vocab_size: int

def load_dataset(kind: str, tokenizer: str, *,
                 text_file: Optional[str] = None, val_file: Optional[str] = None, test_file: Optional[str] = None,
                 val_split: float = 0.1, word_vocab_size: Optional[int] = None,
                 vocab_file: Optional[str] = None, save_vocab_path: Optional[str] = None,
                 bpe_vocab_size: int = 30000, bpe_model: Optional[str] = None,
                 append_eos: bool = True) -> SeqData:
    assert kind in {"text","hf_wikitext2","hf_wikitext103_raw"}
    assert tokenizer in {"byte","word","bpe"}

    if kind.startswith("hf_wikitext"):
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("HuggingFace 'datasets' is not available. Use --dataset text.") from e
        if kind == "hf_wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        else:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_txt = "".join(ds["train"]["text"]) ; valid_txt = "".join(ds["validation"]["text"]) ; test_txt = "".join(ds["test"]["text"]) 
        tmp_tr, tmp_va, tmp_te = "_tmp_wt_train.txt", "_tmp_wt_valid.txt", "_tmp_wt_test.txt"
        for p, s in [(tmp_tr, train_txt), (tmp_va, valid_txt), (tmp_te, test_txt)]:
            with open(p,'w',encoding='utf-8') as f: f.write(s)
        text_file, val_file, test_file = tmp_tr, tmp_va, tmp_te

    # Heuristic: if the user provides Merity's *.tokens files, don't add extra <eos> by default
    if text_file and text_file.endswith('.tokens') and append_eos is True:
        append_eos = False

    if tokenizer == "byte":
        if text_file is None:
            raise ValueError("For tokenizer=byte you must provide --text_file (and optionally --val_file/--test_file)")
        tr = _read_bytes(text_file)
        if val_file is None:
            n = len(tr); cut = int((1.0 - val_split) * n)
            va = tr[cut:].clone(); tr = tr[:cut].clone()
        else:
            va = _read_bytes(val_file)
        te = _read_bytes(test_file) if test_file else va.clone()
        return SeqData(train=tr, val=va, test=te, vocab_size=256)

    if tokenizer == "word":
        if vocab_file and os.path.isfile(vocab_file):
            stoi = load_vocab(vocab_file)
        else:
            if text_file is None:
                raise ValueError("For tokenizer=word you must provide --text_file (train) to build the vocab")
            vseff = None if (word_vocab_size is None or word_vocab_size <= 0) else word_vocab_size
            stoi = build_word_vocab(text_file, vocab_size=vseff)
            if save_vocab_path: save_vocab(stoi, save_vocab_path)
        tr = encode_word_file(text_file, stoi, append_eos=append_eos)
        va = encode_word_file(val_file if val_file else text_file, stoi, append_eos=append_eos)
        te = encode_word_file(test_file if test_file else (val_file if val_file else text_file), stoi, append_eos=append_eos)
        return SeqData(train=tr, val=va, test=te, vocab_size=max(stoi.values())+1)

    bpe = BPEWrap(model_path=bpe_model, vocab_size=bpe_vocab_size, train_file=text_file)
    tr = bpe.encode_file(text_file, append_eos=append_eos)
    va = bpe.encode_file(val_file if val_file else text_file, append_eos=append_eos)
    te = bpe.encode_file(test_file if test_file else (val_file if val_file else text_file), append_eos=append_eos)
    return SeqData(train=tr, val=va, test=te, vocab_size=bpe.vocab_size)

class StreamLoader:
    def __init__(self, data: torch.Tensor, seq_len: int, batch_size: int):
        assert data.dim() == 1 and data.dtype in (torch.uint8, torch.long)
        self.data = data; self.seq_len = seq_len; self.batch_size = batch_size
        self.ptrs = torch.randint(0, max(1, len(data) - seq_len - 1), (batch_size,))
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.empty((self.batch_size, self.seq_len), dtype=torch.long)
        Y = torch.empty((self.batch_size, self.seq_len), dtype=torch.long)
        N = len(self.data)
        for b in range(self.batch_size):
            p = int(self.ptrs[b].item()) if N > self.seq_len + 1 else 0
            x = self.data[p:p+self.seq_len].to(torch.long)
            if x.numel() < self.seq_len:
                r = self.seq_len - x.numel()
                x = torch.cat([x, self.data[:r].to(torch.long)])
                y = torch.cat([self.data[p+1:].to(torch.long), self.data[:r+1].to(torch.long)])
            else:
                y = self.data[p+1:p+self.seq_len+1].to(torch.long)
            X[b], Y[b] = x, y
            if N > self.seq_len + 1:
                p = (p + self.seq_len) % (N - self.seq_len - 1)
                self.ptrs[b] = p
        return X, Y

def iter_val(val: torch.Tensor, seq_len: int, batch_size: int, batches: int):
    N = len(val); ptr = 0
    for _ in range(batches):
        X = torch.empty((batch_size, seq_len), dtype=torch.long)
        Y = torch.empty((batch_size, seq_len), dtype=torch.long)
        for b in range(batch_size):
            if ptr + seq_len + 1 > N:
                over = ptr + seq_len + 1 - N
                x = torch.cat([val[ptr:N].to(torch.long), val[0:max(0, over-1)].to(torch.long)])
                y = torch.cat([val[ptr+1:N].to(torch.long), val[0:over].to(torch.long)])
                ptr = (ptr + seq_len) % (N - 1)
            else:
                x = val[ptr:ptr+seq_len].to(torch.long)
                y = val[ptr+1:ptr+seq_len+1].to(torch.long)
                ptr += seq_len
            X[b], Y[b] = x, y
        yield X, Y

# ---------------------------- Edge builder (Poset) ----------------------------

def build_edges(T: int, K: int, W: int):
    src, dst, delta, slot = [], [], [], []
    ptr = [0]
    for j in range(T):
        i0 = max(0, j - W)
        cand = list(range(i0, j))
        pj = cand[-K:] if len(cand) > K else cand
        for s, i in enumerate(pj):
            src.append(i); dst.append(j); delta.append(j - i); slot.append(s)
        ptr.append(len(src))
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(delta, dtype=torch.int16),
            torch.tensor(ptr, dtype=torch.long),
            torch.tensor(slot, dtype=torch.long))

# ---------------------------- Poset blocks & model ----------------------------

class PosetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, k_parents: int = 16, tau: float = 0.07,
                 window: int = 128, poset_iters: int = 3, poset_iters_eval: Optional[int] = None,
                 dynamic_topk: bool = False, topk: Optional[int] = None, edge_dropout: float = 0.0,
                 fp16_cache: bool = False, node_chunk: int = 0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.ln = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.tau = tau
        self.k_parents = k_parents
        self.poset_iters = poset_iters
        self.poset_iters_eval = poset_iters_eval if poset_iters_eval is not None else poset_iters
        self.rel_bias = nn.Embedding(max(2, window+1), 1)
        nn.init.zeros_(self.rel_bias.weight)
        self.dynamic_topk = dynamic_topk
        self.topk = topk if topk is not None else k_parents
        self.edge_dropout = edge_dropout
        self.fp16_cache = fp16_cache
        self.node_chunk = node_chunk

    def _apply_dynamic_topk_chunk(self, a, dst_idx, slot, j0, Tchunk, K):
        if not self.dynamic_topk or self.topk is None or self.topk >= K:
            return a
        B, H, E = a.shape; Bz = B*H
        a_flat = a.reshape(Bz, E)
        dst_local = (dst_idx - j0)
        idx_e = (dst_local * K + slot)
        scores = torch.full((Bz, Tchunk*K), float('-inf'), device=a.device, dtype=a.dtype)
        idx_e_expand = idx_e.unsqueeze(0).expand(Bz, -1)
        scores.scatter_(1, idx_e_expand, a_flat)
        keep3 = torch.zeros_like(scores.view(Bz, Tchunk, K)).bool()
        kidx = torch.topk(scores.view(Bz, Tchunk, K), k=self.topk, dim=-1).indices
        keep3.scatter_(-1, kidx, True)
        keep = keep3.view(Bz, Tchunk*K).gather(1, idx_e_expand)
        return a * keep.view(B, H, E).float()

    def _aggregate_chunks(self, a_flat, B_state, Zb, src_idx, dst_idx, ptr, j0, j1, Bz, T, dh):
        s = int(ptr[j0].item()); e = int(ptr[j1].item()); E = e - s
        device = a_flat.device
        S_B_flat = torch.zeros((Bz*T, dh), device=device, dtype=a_flat.dtype)
        S_Z_flat = torch.zeros((Bz*T,), device=device, dtype=a_flat.dtype)
        if E <= 0:
            return S_B_flat.view(Bz, T, dh), S_Z_flat.view(Bz, T)
        src_e = src_idx[s:e]; dst_e = dst_idx[s:e]
        Bs = B_state[:, src_e, :].to(a_flat.dtype)
        Zs = Zb[:, src_e]
        a_e = a_flat[:, :E]
        offset = torch.arange(Bz, device=device).unsqueeze(1) * T
        dst_flat = (dst_e.unsqueeze(0) + offset).reshape(-1)
        S_B_flat.index_add_(0, dst_flat, (a_e.unsqueeze(-1) * Bs).reshape(-1, dh))
        S_Z_flat.index_add_(0, dst_flat, (a_e * Zs).reshape(-1))
        return S_B_flat.view(Bz, T, dh), S_Z_flat.view(Bz, T)

    def forward(self, x: torch.Tensor, edges):
        B, T, d = x.shape
        (src_idx, dst_idx_all, delta_all, ptr, slot_all) = edges
        H, dh = self.n_heads, self.dh
        device = x.device
        x_ln = self.ln(x)
        dtype_cache = torch.float16 if (self.fp16_cache and self.training) else x_ln.dtype
        iters = self.poset_iters if self.training else self.poset_iters_eval

        q = self.q(x_ln).view(B, T, H, dh).transpose(1, 2).to(dtype_cache)
        k = self.k(x_ln).view(B, T, H, dh).transpose(1, 2).to(dtype_cache)
        v = self.v(x_ln).view(B, T, H, dh).transpose(1, 2).to(dtype_cache)

        Bz = B * H
        B_state = v.reshape(Bz, T, dh)
        Zb = torch.ones((Bz, T), device=device, dtype=x_ln.dtype)

        def compute_a_for_chunk(j0: int, j1: int):
            s = int(ptr[j0].item()); e = int(ptr[j1].item()); E = e - s
            if E <= 0: return None, None, None
            q_dst = q[:, :, dst_idx_all[s:e], :].to(x_ln.dtype)
            k_src = k[:, :, src_idx[s:e], :].to(x_ln.dtype)
            logits = (q_dst * k_src).sum(-1) / math.sqrt(dh)
            dd = delta_all[s:e].clamp(min=0, max=self.rel_bias.num_embeddings - 1).to(torch.long)
            logits = logits + self.rel_bias(dd).squeeze(-1).to(x_ln.dtype)
            I = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            a = I.pow(1.0 / self.tau)
            if self.training and self.edge_dropout > 0.0:
                keep = (torch.rand_like(a) >= self.edge_dropout).float()
                a = a * keep / (1.0 - self.edge_dropout)
            if self.dynamic_topk and self.topk is not None and self.topk < self.k_parents:
                a = self._apply_dynamic_topk_chunk(a, dst_idx_all[s:e], slot_all[s:e], j0, j1-j0, self.k_parents)
            return a, s, e

        node_chunk = self.node_chunk if self.node_chunk and self.node_chunk > 0 else T

        for _ in range(iters):
            S_B_tot = torch.zeros((Bz, T, dh), device=device, dtype=x_ln.dtype)
            S_Z_tot = torch.zeros((Bz, T), device=device, dtype=x_ln.dtype)
            for j0 in range(0, T, node_chunk):
                j1 = min(T, j0 + node_chunk)
                out = compute_a_for_chunk(j0, j1)
                if out[0] is None: continue
                a_chunk, s, e = out
                a_flat = a_chunk.reshape(Bz, -1)
                SBc, SZc = self._aggregate_chunks(a_flat, B_state, Zb, src_idx, dst_idx_all, ptr, j0, j1, Bz, T, dh)
                S_B_tot += SBc; S_Z_tot += SZc
            B_state = (v.reshape(Bz, T, dh).to(x_ln.dtype) + S_B_tot).to(dtype_cache)
            Zb = 1.0 + S_Z_tot

        S_B_tot = torch.zeros((Bz, T, dh), device=device, dtype=x_ln.dtype)
        S_Z_tot = torch.zeros((Bz, T), device=device, dtype=x_ln.dtype)
        for j0 in range(0, T, node_chunk):
            j1 = min(T, j0 + node_chunk)
            out = compute_a_for_chunk(j0, j1)
            if out[0] is None: continue
            a_chunk, s, e = out
            a_flat = a_chunk.reshape(Bz, -1)
            SBc, SZc = self._aggregate_chunks(a_flat, B_state, Zb, src_idx, dst_idx_all, ptr, j0, j1, Bz, T, dh)
            S_B_tot += SBc; S_Z_tot += SZc

        h = (S_B_tot / S_Z_tot.clamp_min(1e-9).unsqueeze(-1)).view(B, H, T, dh)
        h = h.transpose(1, 2).contiguous().view(B, T, d)
        return x + self.drop(self.out(h))

class PosetLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 n_poset_heads: int = 4, k_parents: int = 16, tau: float = 0.07,
                 poset_iters: int = 3, poset_iters_eval: Optional[int] = 1,
                 dynamic_topk: bool = False, topk: Optional[int] = None,
                 edge_dropout: float = 0.0, fp16_cache: bool = False, node_chunk: int = 0, window: int = 128,
                 tie_weights: bool = True, dropout: float = 0.0, grad_checkpoint: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.grad_checkpoint = grad_checkpoint
        self.blocks = nn.ModuleList([
            PosetBlock(d_model, n_poset_heads, k_parents, tau, window,
                       poset_iters, poset_iters_eval, dynamic_topk, topk, edge_dropout,
                       fp16_cache, node_chunk, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.tie_weights = tie_weights
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02); 
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx: torch.Tensor, edges):
        x = self.drop(self.embed(idx))
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(lambda t, b=blk: b(t, edges), x)
            else:
                x = blk(x, edges)
        x = self.ln_f(x)
        return x  # features

# ---------------------------- Config ----------------------------

@dataclass
class Config:
    tokenizer: str = "word"
    dataset: str = "hf_wikitext103_raw"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    d_model: int = 256
    n_layers: int = 6
    n_poset_heads: int = 4
    k_parents: int = 16
    tau: float = 0.07
    poset_iters: int = 3
    poset_iters_eval: int = 1
    dynamic_topk: bool = False
    topk: Optional[int] = None
    window: int = 128
    edge_dropout: float = 0.0
    fp16_cache: bool = False
    node_chunk: int = 0
    tie_weights: bool = True
    dropout: float = 0.1
    grad_checkpoint: bool = False
    seq_len: int = 512
    batch_size: int = 8
    steps: int = 800
    eval_interval: int = 200
    eval_batches: int = 20
    final_eval_batches: int = 200
    deterministic_final: bool = True
    append_eos: bool = True
    lr: float = 3e-4
    min_lr: Optional[float] = None
    scheduler: str = "none"
    warmup: int = 0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    amp: bool = False
    out_csv: str = "metrics.csv"
    seed: int = 1337
    compile: bool = False
    text_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    val_split: float = 0.1
    word_vocab_size: int = 0
    vocab_file: Optional[str] = None
    save_vocab_path: Optional[str] = "vocab.json"
    bpe_vocab_size: int = 16000
    bpe_model: Optional[str] = "bpe_wt103.json"
    adaptive_softmax: bool = False
    cutoffs: str = "2000,10000,50000"
    save_dir: str = "checkpoints"
    save_every: int = 0
    keep_last_n: int = 3
    resume: Optional[str] = None
    quiet_ckpt: bool = True
    word_ppl_from_bytes: bool = False

# ---------------------------- Eval helpers ----------------------------

def _eval_ce(model, X: torch.Tensor, Y: torch.Tensor, edges_dev, head: Optional[nn.Linear],
             criterion_adap: Optional[nn.AdaptiveLogSoftmaxWithLoss]):
    feats = model(X, edges_dev)
    if criterion_adap is not None:
        with torch.no_grad():
            inp = feats.reshape(-1, feats.size(-1)).to(torch.float32)
            tgt = Y.view(-1)
            loss = criterion_adap(inp, tgt).loss
        return loss
    else:
        logits = head(feats)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

def evaluate_loader(model, cfg: Config, loader: "StreamLoader", edges_dev=None, batches: int = 20,
                    head=None, criterion_adap=None):
    model.eval(); vals = []
    with torch.no_grad():
        for _ in range(batches):
            X, Y = loader.next_batch()
            X = X.to(cfg.device); Y = Y.to(cfg.device)
            cast_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
            with cast_ctx:
                loss = _eval_ce(model, X, Y, edges_dev, head, criterion_adap)
            li = float(loss.item())
            if math.isfinite(li) and abs(li) < 1e6: vals.append(li)
    model.train()
    if not vals: return float('nan'), float('nan')
    mean_loss = sum(vals)/len(vals)
    ppl = math.exp(mean_loss) if mean_loss < 100 else float('inf')
    return mean_loss, ppl

def evaluate_sequence(model, cfg: Config, seq: torch.Tensor, edges_dev=None, batches: int = 100,
                      head=None, criterion_adap=None):
    model.eval(); vals = []
    with torch.no_grad():
        for X, Y in iter_val(seq, cfg.seq_len, cfg.batch_size, batches):
            X = X.to(cfg.device); Y = Y.to(cfg.device)
            cast_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
            with cast_ctx:
                loss = _eval_ce(model, X, Y, edges_dev, head, criterion_adap)
            li = float(loss.item())
            if math.isfinite(li) and abs(li) < 1e6: vals.append(li)
    model.train()
    if not vals: return float('nan'), float('nan')
    mean_loss = sum(vals)/len(vals)
    ppl = math.exp(mean_loss) if mean_loss < 100 else float('inf')
    return mean_loss, ppl

# ---------------------------- Optim helpers ----------------------------

def wd_groups(named_params, weight_decay: float):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad: continue
        if p.dim() == 1 or n.endswith('.bias') or ('ln' in n) or ('norm' in n) or ('embed' in n):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def _sanitize_cutoffs(cutoffs_str: str, V: int):
    try:
        raw = [int(x) for x in cutoffs_str.split(',') if x.strip()]
    except Exception:
        raw = []
    clean = sorted(set(c for c in raw if 0 < c < V))
    if not clean:
        cand = [max(1, V//8), max(2, V//2), max(3, (7*V)//8)]
        clean = sorted(set(c for c in cand if 0 < c < V))
    return clean

# ---------------------------- Train ----------------------------

def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = load_dataset(cfg.dataset, cfg.tokenizer,
                        text_file=cfg.text_file, val_file=cfg.val_file, test_file=cfg.test_file,
                        val_split=cfg.val_split, word_vocab_size=cfg.word_vocab_size,
                        vocab_file=cfg.vocab_file, save_vocab_path=cfg.save_vocab_path,
                        bpe_vocab_size=cfg.bpe_vocab_size, bpe_model=cfg.bpe_model,
                        append_eos=cfg.append_eos)

    train_loader = StreamLoader(data.train, cfg.seq_len, cfg.batch_size)
    val_loader = StreamLoader(data.val, cfg.seq_len, cfg.batch_size)
    V = data.vocab_size

    model = PosetLM(vocab_size=V, d_model=cfg.d_model, n_layers=cfg.n_layers,
                    n_poset_heads=cfg.n_poset_heads, k_parents=cfg.k_parents, tau=cfg.tau,
                    poset_iters=cfg.poset_iters, poset_iters_eval=cfg.poset_iters_eval,
                    dynamic_topk=cfg.dynamic_topk, topk=cfg.topk,
                    edge_dropout=cfg.edge_dropout, fp16_cache=cfg.fp16_cache, node_chunk=cfg.node_chunk, window=cfg.window,
                    tie_weights=cfg.tie_weights, dropout=cfg.dropout, grad_checkpoint=cfg.grad_checkpoint).to(device)

    if cfg.compile:
        try:
            from torch import _dynamo as dynamo
            dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile not enabled: {e}")

    head = None; criterion_adap = None
    if cfg.adaptive_softmax:
        cutoffs = _sanitize_cutoffs(cfg.cutoffs, V)
        if V > 10000:
            criterion_adap = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=cfg.d_model, n_classes=V, cutoffs=cutoffs, div_value=4.0
            ).to(device)
            print(f"[INFO] Adaptive Softmax (V={V}, cutoffs={cutoffs})")
        else:
            print(f"[INFO] V={V} ≤ 10k: using dense head.")
            head = model.head.to(device)
    else:
        head = model.head.to(device)

    edges_cpu = build_edges(cfg.seq_len, cfg.k_parents, cfg.window)
    edges_dev = tuple(t.to(device) for t in edges_cpu)
    print(f"[INFO] Poset edges: E={edges_cpu[0].numel()} for T={cfg.seq_len}, K={cfg.k_parents}, W={cfg.window}")

    params = wd_groups(model.named_parameters(), cfg.weight_decay)
    if criterion_adap is not None:
        params += wd_groups(criterion_adap.named_parameters(), cfg.weight_decay)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95))

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == 'cuda'))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

    csv_f = open(cfg.out_csv, 'a', newline=''); csv_w = csv.writer(csv_f)
    header = ["time","model","tokenizer","step","val_loss","val_ppl","bpb","tok_s","mem_MB","params_M","lr"]
    try:
        if os.path.getsize(cfg.out_csv) == 0: csv_w.writerow(header)
    except FileNotFoundError:
        csv_w.writerow(header)

    pbar_iter = range(cfg.steps)
    pbar = tqdm(pbar_iter, ncols=120, dynamic_ncols=True, leave=True) if TQDM else pbar_iter

    ln2 = math.log(2.0)
    best_val = float('inf')
    tokens_seen = 0; t0 = time.time(); ema_tps = EMA(0.9)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for step in pbar:
        X, Y = train_loader.next_batch(); X = X.to(device); Y = Y.to(device)

        if cfg.scheduler == 'cosine':
            warmup = max(0, cfg.warmup)
            base = cfg.lr; min_lr = cfg.min_lr if (cfg.min_lr is not None) else (0.1*cfg.lr)
            if step < warmup:
                lr_cur = base * (step + 1) / max(1, warmup)
            else:
                prog = (step + 1 - warmup) / max(1, cfg.steps - warmup)
                prog = min(max(prog, 0.0), 1.0)
                lr_cur = min_lr + 0.5*(base-min_lr)*(1.0+math.cos(math.pi*prog))
            for pg in optimizer.param_groups: pg['lr'] = lr_cur
        else:
            lr_cur = cfg.lr

        with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == 'cuda')):
            feats = model(X, edges_dev)
            if criterion_adap is not None:
                inp = feats.reshape(-1, feats.size(-1)).to(torch.float32)
                tgt = Y.view(-1)
                loss = criterion_adap(inp, tgt).loss / cfg.grad_accum
            else:
                logits = head(feats)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) / cfg.grad_accum

        scaler.scale(loss).backward()
        if (step + 1) % cfg.grad_accum == 0:
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

        tokens_seen += X.numel(); dt = time.time() - t0
        tps = tokens_seen / max(1e-6, dt); tps_ema = ema_tps.update(tps)

        if (step + 1) % cfg.eval_interval == 0 or step == 0:
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
            val_loss, val_ppl = evaluate_loader(model, cfg, val_loader, edges_dev, batches=cfg.eval_batches, head=head, criterion_adap=criterion_adap)
            max_mem = torch.cuda.max_memory_allocated()/(1024**2) if device.type=='cuda' else 0.0
            bpb = (val_loss/ln2) if cfg.tokenizer=='byte' else float('nan')

            if cfg.save_dir:
                os.makedirs(cfg.save_dir, exist_ok=True)
                step1 = step + 1
                torch.save({"model":model.state_dict(),"step":step1,"cfg":asdict(cfg)}, os.path.join(cfg.save_dir,'last.pt'))
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model":model.state_dict(),"step":step1,"cfg":asdict(cfg),"val":val_loss}, os.path.join(cfg.save_dir,'best.pt'))

            msg = (f"posetlm/{cfg.tokenizer} step {step+1} | loss {loss.item()*cfg.grad_accum:.3f} | val {val_loss:.3f} | ppl {val_ppl:.2f}"
                   + (f" | bpb {bpb:.3f}" if cfg.tokenizer=='byte' else "")
                   + f" | tok/s {tps_ema:,.0f} | mem {max_mem:.0f}MB | params ~{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M | lr {lr_cur:.2e}")
            if TQDM: pbar.set_description(msg)
            else: print(msg, flush=True)
            csv_w.writerow([int(time.time()), "posetlm", cfg.tokenizer, step+1, f"{val_loss:.6f}", f"{val_ppl:.6f}",
                            f"{bpb if bpb==bpb else 0.0:.6f}", f"{tps_ema:.2f}",
                            f"{max_mem:.2f}", f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}", f"{lr_cur:.6e}"])
            csv_f.flush()

    print("=== FINAL RESULTS ===")
    if cfg.deterministic_final:
        val_loss, val_ppl = evaluate_sequence(model, cfg, data.val, edges_dev, batches=cfg.final_eval_batches, head=head, criterion_adap=criterion_adap)
        test_loss, test_ppl = evaluate_sequence(model, cfg, data.test, edges_dev, batches=cfg.final_eval_batches, head=head, criterion_adap=criterion_adap)
    else:
        val_loss, val_ppl = evaluate_loader(model, cfg, val_loader, edges_dev, batches=cfg.final_eval_batches, head=head, criterion_adap=criterion_adap)
        test_loader = StreamLoader(data.test, cfg.seq_len, cfg.batch_size)
        test_loss, test_ppl = evaluate_loader(model, cfg, test_loader, edges_dev, batches=cfg.final_eval_batches, head=head, criterion_adap=criterion_adap)

    ln2 = math.log(2.0)
    print(f"Final VAL:  loss {val_loss:.4f} | PPL {val_ppl:.2f}" + (f" | bpb {val_loss/ln2:.3f}" if cfg.tokenizer=='byte' else ""))
    print(f"Final TEST: loss {test_loss:.4f} | PPL {test_ppl:.2f}" + (f" | bpb {test_loss/ln2:.3f}" if cfg.tokenizer=='byte' else ""))

# ---------------------------- CLI ----------------------------

def parse_args() -> 'Config':
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', type=str, default='word', choices=['byte','word','bpe'])
    p.add_argument('--dataset', type=str, default='hf_wikitext103_raw', choices=['text','hf_wikitext2','hf_wikitext103_raw'])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_poset_heads', type=int, default=4)
    p.add_argument('--k_parents', type=int, default=16)
    p.add_argument('--tau', type=float, default=0.07)
    p.add_argument('--poset_iters', type=int, default=3)
    p.add_argument('--poset_iters_eval', type=int, default=1)
    p.add_argument('--dynamic_topk', action='store_true')
    p.add_argument('--topk', type=int, default=None)
    p.add_argument('--window', type=int, default=128)
    p.add_argument('--edge_dropout', type=float, default=0.0)
    p.add_argument('--fp16_cache', action='store_true')
    p.add_argument('--node_chunk', type=int, default=0)
    p.add_argument('--tie_weights', action='store_true')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--grad_checkpoint', action='store_true')
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_batches', type=int, default=20)
    p.add_argument('--final_eval_batches', type=int, default=200)
    p.add_argument('--deterministic_final', action='store_true')
    p.add_argument('--append_eos', action='store_true')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=None)
    p.add_argument('--scheduler', type=str, default='none', choices=['none','cosine'])
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--out_csv', type=str, default='metrics.csv')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--text_file', type=str, default=None)
    p.add_argument('--val_file', type=str, default=None)
    p.add_argument('--test_file', type=str, default=None)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--word_vocab_size', type=int, default=0)
    p.add_argument('--vocab_file', type=str, default=None)
    p.add_argument('--save_vocab_path', type=str, default='vocab.json')
    p.add_argument('--bpe_vocab_size', type=int, default=16000)
    p.add_argument('--bpe_model', type=str, default='bpe_wt103.json')
    p.add_argument('--adaptive_softmax', action='store_true')
    p.add_argument('--cutoffs', type=str, default='2000,10000,50000')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--save_every', type=int, default=0)
    p.add_argument('--keep_last_n', type=int, default=3)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--quiet_ckpt', action='store_true')
    p.add_argument('--word_ppl_from_bytes', action='store_true')

    a = p.parse_args()
    cfg = Config(**vars(a))
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
