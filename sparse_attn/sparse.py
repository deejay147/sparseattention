"""Sparse attention impls.

gather_sparse_attention: only allowed pairs, loop over queries.
slow, but independent of dense_attention -> real cross-check for the harness.
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
