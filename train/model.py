"""tiny char GPT for 1.6. attention = our dense_attention + mask per pattern"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for sparse_attn

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_attn.dense import dense_attention
from sparse_attn.masks import causal_mask, sliding_window_mask, bigbird_mask


@dataclass
class Config:
    vocab: int = 65
    ctx: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    attn: str = "dense"     # dense | sliding | bigbird
    window: int = 16
    block: int = 16
    n_global: int = 2
    n_random: int = 1


def build_mask(cfg):
    T = cfg.ctx
    if cfg.attn == "dense":
        return causal_mask(T)
    if cfg.attn == "sliding":
        return sliding_window_mask(T, cfg.window)
    if cfg.attn == "bigbird":
        return bigbird_mask(T, cfg.block, cfg.n_global, cfg.n_random)
    raise ValueError(f"unknown attn {cfg.attn}")


class SelfAttention(nn.Module):
    def __init__(self, cfg, mask):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)       # q, k, v in one go
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.register_buffer("mask", mask, persistent=False)    # moves to gpu w/ model

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, H, T, C/H), shape dense_attention wants
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        out = dense_attention(q, k, v, self.mask[:T, :T])
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # heads back together
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg, mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg, mask)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # residual, token keeps itself even if attn = 0
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mask = build_mask(cfg)            # same pattern both layers
        self.tok = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.pos = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, mask) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab)
        if targets is None:
            return logits, None
        # cross_entropy wants (N, classes) + (N,), so flatten batch x time
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
