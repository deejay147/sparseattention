"""train tiny GPT on TinyShakespeare (1.6)
python train/train.py --attn dense|sliding|bigbird [--iters 3000]
saves results/train_<name>.json
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))   # so "from model import" works

import torch
from model import GPT, Config

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_data():
    """download once into data/ (gitignored), chars -> ids, 90/10 split"""
    path = ROOT / "data" / "input.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    text = path.read_text()
    chars = sorted(set(text))                       # every distinct char, fixed order
    stoi = {c: i for i, c in enumerate(chars)}      # char -> id
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, B, T, gen, device):
    """B random windows of T chars. y = x shifted by one (next char)"""
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, B, T, device, n_batches=50):
    """avg loss on the SAME val batches every time so points on the curve compare"""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total = sum(model(*get_batch(data, B, T, gen, device))[1].item() for _ in range(n_batches))
    model.train()
    return total / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="dense", choices=["dense", "sliding", "bigbird"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--n_global", type=int, default=2)
    ap.add_argument("--n_random", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, vocab = load_data()

    cfg = Config(vocab=vocab, ctx=args.ctx, attn=args.attn, window=args.window,
                 block=args.block, n_global=args.n_global, n_random=args.n_random)
    torch.manual_seed(args.seed)                          # same init every variant
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)        # same batch order every variant

    name = {"dense": "dense",
            "sliding": f"sliding_w{args.window}",
            "bigbird": f"bigbird_b{args.block}_g{args.n_global}_r{args.n_random}"}[args.attn]
    n_params = sum(prm.numel() for prm in model.parameters())
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"{name}: {n_params/1e6:.2f}M params, {gpu}, iters={args.iters}")

    log, t0 = [], time.time()
    for step in range(args.iters + 1):
        if step % args.eval_every == 0 or step == args.iters:
            val = eval_loss(model, val_data, args.batch, args.ctx, device)
            log.append(dict(step=step, val_loss=val, elapsed_s=round(time.time() - t0, 1)))
            print(f"step {step:5d}  val {val:.4f}  {time.time() - t0:6.1f}s")
        if step == args.iters:
            break

        x, y = get_batch(train_data, args.batch, args.ctx, gen, device)
        _, loss = model(x, y)
        if not math.isfinite(loss.item()):                # NaN -> stop now, don't burn the run
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)                   # clear last step's grads
        loss.backward()
        opt.step()

    out = dict(name=name, config=asdict(cfg), args=vars(args), gpu=gpu,
               n_params=n_params, final_val=log[-1]["val_loss"],
               train_time_s=round(time.time() - t0, 1), log=log)
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / f"train_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
