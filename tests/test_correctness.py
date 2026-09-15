"""Correctness harness (1.3, 1.4).

run from repo root:  python tests/test_correctness.py
exit 0 = all pass, 1 = something failed
"""
import math
import sys
from pathlib import Path

# repo root on path so "import sparse_attn" works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sparse_attn.dense import dense_attention
from sparse_attn.masks import causal_mask, sliding_window_mask, bigbird_mask
from sparse_attn.sparse import gather_sparse_attention, sliding_window_attention

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def rand_qkv(B, H, N, d, dtype, qk_scale=1.0, seed=0):
    """random q, k, v. qk_scale blows up q, k so softmax saturates"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, N, d, generator=g, dtype=dtype) * qk_scale
    k = torch.randn(B, H, N, d, generator=g, dtype=dtype) * qk_scale
    v = torch.randn(B, H, N, d, generator=g, dtype=dtype)
    return q, k, v


def patterns(N, causal):
    return {
        "full": causal_mask(N) if causal else torch.ones(N, N, dtype=torch.bool),
        "sliding_w4": sliding_window_mask(N, w=4, causal=causal),
        "bigbird_b4_g1_r1": bigbird_mask(N, block_size=4, num_global=1,
                                         num_random=1, causal=causal),
    }


def empty_row_patterns(N, block=4):
    """test-only patterns that actually leave some rows empty"""
    no_self = causal_mask(N) & ~torch.eye(N, dtype=torch.bool)   # token 0 has nothing
    blk = torch.arange(N) // block                                # block id per token
    prev_block = (blk[:, None] - blk[None, :]) == 1               # block 0 has no prev block
    return {"causal_no_self": no_self, "prev_block_only": prev_block}


# --- 1.3 ---
# fixed after 83/84 error: fixed atol 1e-4 failed bigbird N=64 non-causal scale=100 float32
# (err 1.4e-4, same case float64 = 1.1e-16 -> rounding on near-tied huge logits, not a bug)
# tolerance now scales with dtype precision and score size instead of a guessed constant
def tolerance(q, k, dtype):
    """rounding bound for comparing two impls: 10 * eps * biggest logit, plus a floor"""
    d = q.size(-1)
    max_score = (q @ k.transpose(-2, -1) / math.sqrt(d)).abs().max().item()
    eps = torch.finfo(dtype).eps                       # float32 ~1.2e-7, float64 ~2.2e-16
    floor = 1e-5 if dtype == torch.float32 else 1e-12  # tiny scores -> don't go absurdly strict
    return max(floor, 10 * eps * max_score)


def test_sparse_matches_dense():
    print("\n== 1.3 sparse (gather) vs dense ==")
    for dtype in (torch.float32, torch.float64):
        for N in (7, 16, 64):                 # 7: not divisible by block size
            for causal in (True, False):
                for qk_scale in (1.0, 100.0):  # 100: huge scores, stress test
                    q, k, v = rand_qkv(2, 2, N, 16, dtype, qk_scale)
                    atol = tolerance(q, k, dtype)   # per case, from actual score size
                    for name, m in patterns(N, causal).items():
                        ref = dense_attention(q, k, v, m)
                        out = gather_sparse_attention(q, k, v, m)
                        rows = m.any(-1)       # only rows with >=1 allowed key
                        err = (out - ref)[:, :, rows].abs().max().item()
                        report(f"{name:17s} N={N:<3d} causal={causal!s:5s} "
                               f"scale={qk_scale:<5} {str(dtype)[6:]}",
                               err <= atol, f"max_err={err:.1e} tol={atol:.1e}")


# added phase 3: chunked sliding window (the fast one) vs dense + same mask
def test_sliding_window_matches_dense():
    print("\n== 1.3 chunked sliding window vs dense ==")
    for dtype in (torch.float32, torch.float64):
        for N in (7, 64, 100):
            q, k, v = rand_qkv(2, 2, N, 16, dtype)
            atol = tolerance(q, k, dtype)
            for w in (1, 4, 16, 200):             # 1 = self only, 200 > N = whole context
                for chunk in (None, 3):           # None -> chunk = w; 3 -> uneven chunks
                    for causal in (True, False):
                        ref = dense_attention(q, k, v, sliding_window_mask(N, w, causal=causal))
                        out = sliding_window_attention(q, k, v, w, causal=causal, chunk=chunk)
                        err = (out - ref).abs().max().item()   # no empty rows, compare all
                        report(f"chunked N={N:<3d} w={w:<3d} chunk={str(chunk):4s} "
                               f"causal={causal!s:5s} {str(dtype)[6:]}",
                               err <= atol, f"max_err={err:.1e} tol={atol:.1e}")


# ---- 1.4 ----
def test_nan_handling():
    print("\n== 1.4 NaN handling ==")
    N = 16
    q, k, v = rand_qkv(1, 2, N, 8, torch.float64)
    for name, m in empty_row_patterns(N).items():
        empty = ~m.any(-1)
        tag = f"[{name}, {int(empty.sum())} empty rows]"

        # without fix: should be NaN
        unsafe = dense_attention(q, k, v, m, safe=False)
        report(f"bug reproduced: NaN without fix {tag}",
               torch.isnan(unsafe[:, :, empty]).all())

        # with fix: clean output
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = dense_attention(qq, kk, vv, m)
        report(f"no NaN in output with fix {tag}", not torch.isnan(out).any())
        report(f"empty rows output exactly zero {tag}", (out[:, :, empty] == 0).all())

        # grads clean too
        out.sum().backward()
        report(f"all gradients finite {tag}",
               all(torch.isfinite(t.grad).all() for t in (qq, kk, vv)))

        # matches gather on every row, empty ones included
        err = (out.detach() - gather_sparse_attention(q, k, v, m)).abs().max().item()
        report(f"safe dense == gather on ALL rows {tag}", err <= 1e-10, f"max_err={err:.1e}")


def test_gradient_traps():
    print("\n== 1.4 torch.where-after-softmax trap ==")
    N, d = 16, 8
    m = empty_row_patterns(N)["causal_no_self"]
    empty = ~m.any(-1, keepdim=True)

    def run(style):
        q, k, v = (t.requires_grad_(True) for t in rand_qkv(1, 2, N, d, torch.float64))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)
        if style == "additive":   # scores + bias, how lots of real code does it
            bias = torch.zeros(N, N, dtype=scores.dtype).masked_fill(~m, float("-inf"))
            scores = scores + bias
        else:                     # masked_fill, like dense.py
            scores = scores.masked_fill(~m, float("-inf"))
        w = torch.softmax(scores, dim=-1)                    # NaN in empty rows
        w = torch.where(empty, torch.zeros_like(w), w)       # "fix" after the fact
        out = w @ v
        out.sum().backward()
        return bool(torch.isnan(out).any()), bool(torch.isnan(k.grad).any())

    fwd_nan, grad_nan = run("additive")
    report("additive mask + where-after-softmax: clean fwd, NaN grads (trap reproduced)",
           (not fwd_nan) and grad_nan)
    fwd_nan, grad_nan = run("masked_fill")
    print(f"INFO  masked_fill + where-after-softmax: fwd NaN={fwd_nan}, grad NaN={grad_nan} "
          f"(masked_fill backward zeroes it, fragile, not relied on)")


def test_finite_fill_leaks():
    print("\n== 1.4 why not -1e9 instead of -inf ==")
    N, d = 16, 8
    m = empty_row_patterns(N)["causal_no_self"]
    q, k, v = rand_qkv(1, 2, N, d, torch.float64)
    scores = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(~m, -1e9)
    out = torch.softmax(scores, dim=-1) @ v
    # row 0 all -1e9 -> uniform over ALL tokens, future included
    leaked = torch.allclose(out[:, :, 0], v.mean(dim=2))
    report("-1e9 fill: empty row averages all tokens incl. future (leak reproduced)", leaked)


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    test_sparse_matches_dense()
    test_sliding_window_matches_dense()
    test_nan_handling()
    test_gradient_traps()
    test_finite_fill_leaks()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
