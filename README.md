# Sparse Attention from Scratch

Postman AI/ML recruitment task (Task 1). Manual dense attention, sliding-window
and BigBird-style sparse attention, correctness harness, benchmarks, and a
2-layer char-level GPT comparison on TinyShakespeare.

## Setup
    pip install -r requirements.txt

## Run
| What | Command |
|---|---|
| Correctness harness | `python tests/test_correctness.py` |
| Benchmark (GPU) | `python bench/benchmark.py` |
| Training (GPU) | `python train/train.py --attn dense` |

## Layout
- `sparse_attn/` — attention implementations and masks
- `tests/` — correctness harness (items 1.3, 1.4)
- `bench/` — benchmark script (1.5), outputs in `results/` and `plots/`
- `train/` — tiny GPT and training loop (1.6)

## Status
- [X] 1.1 Dense attention
- [X] 1.2 Sparsity patterns
- [X] 1.3 Correctness harness
- [X] 1.4 NaN handling
- [x] 1.5 Benchmark
- [x] 1.6 Quality evaluation
- [ ] 1.7 Writeup
