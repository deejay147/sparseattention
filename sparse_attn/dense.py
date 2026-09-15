"""Reference dense attention (item 1.1). No F.scaled_dot_product_attention.

Mask convention: boolean, True = allowed, False = masked out.
"""
import math
import torch


def dense_attention(q, k, v, mask=None):
    """
    q, k, v : (B, H, N, d) tensors
    mask    : bool tensor broadcastable to (B, H, N, N), True = allowed
    returns : (B, H, N, d)
    """
    d = q.size(-1)

    # similarity of every query with every key -> (B, H, N, N)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    if mask is not None:
        # forbidden pairs -> -inf, so softmax gives them weight exactly 0
        scores = scores.masked_fill(~mask, float("-inf"))

    # each row becomes a probability distribution over keys
    weights = torch.softmax(scores, dim=-1)

    # weighted average of values for every query -> (B, H, N, d)
    return weights @ v
