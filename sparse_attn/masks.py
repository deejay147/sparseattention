"""Attention masks (item 1.2). All masks: bool (N, N), True = allowed."""
import torch

def causal_mask(N, device=None):
    """Token i may attend to j <= i (lower triangle, diagonal included)."""
    return torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))

def sliding_window_mask(N, w, causal=True, device=None):
    """
    causal=True : i attends to the w most recent tokens, itself included (0 <= i-j < w)
    causal=False: i attends to tokens with |i-j| < w
    """
    i = torch.arange(N, device=device)[:, None]   # column vector (N, 1)
    j = torch.arange(N, device=device)[None, :]   # row vector    (1, N)
    diff = i - j                                  # broadcasts to (N, N): diff[i, j] = i - j
    if causal:
        return (diff >= 0) & (diff < w)
    return diff.abs() < w

def bigbird_mask(N, block_size, num_global=1, num_random=1,
                 causal=True, seed=0, device=None):
    """
    BigBird-style block-sparse mask.
      local : each query block sees its own block and both neighbours
      random: each query block sees `num_random` random blocks (fixed by seed)
      global: first `num_global` tokens see everything and are seen by everything
    If N isn't divisible by block_size, the last block is partial.
    """
    b = block_size
    nb = (N + b - 1) // b                          # number of blocks, rounded up

    # block-level pattern: blk[p, q] = True means block p may attend to block q
    blk = torch.zeros(nb, nb, dtype=torch.bool)
    idx = torch.arange(nb)
    for offset in (-1, 0, 1):                      # local: previous, same, next block
        nbr = idx + offset
        ok = (nbr >= 0) & (nbr < nb)               # skip neighbours past the edges
        blk[idx[ok], nbr[ok]] = True

    gen = torch.Generator().manual_seed(seed)      # private RNG: same pattern every run
    for p in range(nb):
        picks = torch.randperm(nb, generator=gen)[:num_random]
        blk[p, picks] = True

    # expand blocks to tokens: each block entry becomes a b x b square, then trim to N
    mask = blk.repeat_interleave(b, dim=0).repeat_interleave(b, dim=1)[:N, :N]

    mask[:num_global, :] = True                    # global tokens see everything
    mask[:, :num_global] = True                    # everything sees global tokens

    if causal:
        mask &= causal_mask(N)                     # then remove all future positions

    return mask.to(device)
