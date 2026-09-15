"""Sparse attention impls.

gather_sparse_attention: only allowed pairs, loop over queries. slow, test-only cross-check.
sliding_window_attention: chunked, never builds N x N. memory linear in N. used in benchmark.
"""
import math
import torch


def gather_sparse_attention(q, k, v, mask):
    """
    q, k, v : (B, H, N, d)
    mask    : bool (N, N), same pattern for all batch/heads
    empty rows -> zeros (same as safe dense)
    """
    assert mask.dim() == 2, "expects one (N, N) pattern"
    B, H, N, d = q.shape
    out = q.new_zeros(B, H, N, v.size(-1))              # empty rows just stay 0

    for i in range(N):
        idx = mask[i].nonzero(as_tuple=True)[0]         # which keys query i can see
        if idx.numel() == 0:
            continue                                    # nothing allowed, skip
        ki = k[:, :, idx, :]                            # (B, H, m, d), allowed keys only
        vi = v[:, :, idx, :]
        s = q[:, :, i:i+1, :] @ ki.transpose(-2, -1) / math.sqrt(d)   # (B, H, 1, m), no -inf anywhere
        w = torch.softmax(s, dim=-1)
        out[:, :, i, :] = (w @ vi).squeeze(-2)          # drop the len-1 dim -> (B, H, d)

    return out


def sliding_window_attention(q, k, v, w, causal=True, chunk=None):
    """
    same result as dense_attention(q, k, v, sliding_window_mask(N, w, causal))
    but scores per chunk are (c, c+w-1) instead of (N, N)
    q, k, v : (B, H, N, d)
    w       : window size (self included)
    chunk   : queries per chunk, defaults to w
    """
    B, H, N, d = q.shape
    c = chunk or w
    out = q.new_empty(B, H, N, v.size(-1))              # filled chunk by chunk

    for s in range(0, N, c):
        e = min(s + c, N)                               # this chunk's queries: [s, e)
        ks = max(0, s - w + 1)                          # earliest key any query here can see
        ke = e if causal else min(N, e + w - 1)         # latest key (+1), future only if non-causal

        qi = q[:, :, s:e]
        ki = k[:, :, ks:ke]
        vi = v[:, :, ks:ke]
        scores = qi @ ki.transpose(-2, -1) / math.sqrt(d)    # (B, H, e-s, ke-ks), small

        # window mask for just this slice, using real positions
        i = torch.arange(s, e, device=q.device)[:, None]
        j = torch.arange(ks, ke, device=q.device)[None, :]
        diff = i - j
        allowed = (diff >= 0) & (diff < w) if causal else diff.abs() < w
        scores = scores.masked_fill(~allowed, float("-inf"))

        # no empty rows possible here: self always inside the window
        out[:, :, s:e] = torch.softmax(scores, dim=-1) @ vi

    return out
