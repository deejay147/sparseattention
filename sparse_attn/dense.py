"""Reference dense attention (1.1). No F.scaled_dot_product_attention.
mask: bool, True = allowed.
"""
import math
import torch


def dense_attention(q, k, v, mask=None, safe=True):
    """
    q, k, v : (B, H, N, d)
    mask    : bool, broadcastable to (B, H, N, N)
    safe    : empty rows -> zeros instead of NaN
    returns : (B, H, N, d)
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)      # (B, H, N, N)

    if mask is None:
        return torch.softmax(scores, dim=-1) @ v

    scores = scores.masked_fill(~mask, float("-inf"))    # blocked pairs -> -inf

    if not safe:
        # old behaviour, all-masked row = NaN
        return torch.softmax(scores, dim=-1) @ v

    empty = ~mask.any(dim=-1, keepdim=True)              # (N, 1), True if row has nothing allowed

    scores = scores.masked_fill(empty, 0.0)              # empty rows finite first, no 0/0
    weights = torch.softmax(scores, dim=-1).masked_fill(empty, 0.0)   # then kill them

    return weights @ v
