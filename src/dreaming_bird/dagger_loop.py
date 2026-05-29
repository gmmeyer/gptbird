"""Iterative DAgger: repeatedly roll out the latest model, relabel with the geometry oracle,
aggregate, and fine-tune — driving the model's RAW-status phantom rate down round over round.

Each round generates fresh data from the *current* model (so it targets that model's remaining
failures), adds it to the aggregate, and fine-tunes the deployed checkpoint on train + aggregate.
Prints the phantom-rate trajectory. Run in-process so the model stays resident on the GPU.

    uv run python -m dreaming_bird.dagger_loop --rounds 2 --episodes 3000
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch

from .config import EngineConfig
from .dagger import generate_dagger
from .model import DreamGPT
from .rollout import collision_audit
from .tokenizer import Tokenizer
from .train import train as train_fn


def _load(path: str, cap: int = 256):
    ck = torch.load(path, weights_only=False)
    m = DreamGPT(ck["model_cfg"], ck["vocab_size"]).cuda()
    m.load_state_dict(ck["model"])
    m.eval()
    m.cfg = dataclasses.replace(m.cfg, block_size=cap)
    return m


def _audit(path, tok, cfg):
    m = _load(path)
    a = collision_audit(m, tok, cfg=cfg, n=40, derive_status=False)
    del m
    torch.cuda.empty_cache()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2, help="additional rounds beyond round 1")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--deployed", type=str, default="checkpoints/small_pipes.pt")
    args = ap.parse_args()

    tok = Tokenizer()
    cfg = EngineConfig()
    train_tokens = np.fromfile("data/train.bin", dtype=np.uint16)
    agg = [np.fromfile("data/dagger.bin", dtype=np.uint16)]      # round-1 DAgger data
    latest = "checkpoints/small_dagger.pt"                       # round-1 model

    a1 = _audit(latest, tok, cfg)
    print(f"ROUND 1: raw_phantom={a1['phantom_rate']:.2f} antigap={a1['antigap_survival_mean']:.0f}",
          flush=True)

    for r in range(2, 2 + args.rounds):
        gen_model = _load(latest)
        toks, meta = generate_dagger(gen_model, tok, cfg=cfg, n_episodes=args.episodes,
                                     seed=r, batch=256)
        del gen_model
        torch.cuda.empty_cache()
        agg.append(toks)
        combined = np.concatenate([train_tokens, *agg])
        combined.tofile("data/combined.bin")
        out = f"checkpoints/small_dagger{r}.pt"
        print(f"--- round {r}: dagger {meta['n_tokens']} tok (agg {sum(t.size for t in agg)}), "
              f"fine-tuning {out} ---", flush=True)
        train_fn("data/combined.bin", tier="small", iters=args.iters, lr=1e-3, dead_weight=2,
                 init_from=args.deployed, out=out, eval_every=1000)
        a = _audit(out, tok, cfg)
        print(f"ROUND {r}: raw_phantom={a['phantom_rate']:.2f} antigap={a['antigap_survival_mean']:.0f}",
              flush=True)
        latest = out

    print("LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
