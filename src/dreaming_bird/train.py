"""Train a DreamGPT on packed token streams.

Autoregressive next-token loss, MASKED OFF the player-supplied action tokens (they are
conditioning, not predictions). Optional context-frame noise augmentation (``--noise-aug``,
off by default per the drift-mitigation ladder) perturbs INPUT state tokens by +/-1 bin so the
model learns to absorb its own rollout-sized errors; targets stay clean.

Example:
    uv run python -m dreaming_bird.train --data data/nopipes --tier nano --iters 3000 \
        --out checkpoints/nano_nopipes.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import NANO, SMALL, ModelConfig, TokenizerConfig
from .model import DreamGPT
from .tokenizer import Tokenizer

TIERS = {"nano": NANO, "small": SMALL}


class Batcher:
    """Samples random fixed-length windows from the concatenated token stream."""

    def __init__(self, tokens: np.ndarray, block_size: int, tok: Tokenizer,
                 device: str, seed: int = 0, dead_weight: float = 1.0):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        # per-token loss weight: 0 on action targets (conditioning), dead_weight on the rare
        # DEAD status token (~0.1% of tokens -> up-weight so collisions are actually learned).
        w = np.ones(len(tokens), dtype=np.float32)
        w[(tokens >= tok.act_off) & (tokens < tok.act_off + 2)] = 0.0
        w[tokens == tok.status_token(False)] = dead_weight
        self.weight = torch.from_numpy(w)
        self.block = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.n = len(tokens)

    def batch(self, bs: int):
        ix = self.rng.integers(0, self.n - self.block - 1, size=bs)
        ix_t = torch.from_numpy(ix)
        offs = torch.arange(self.block)
        xi = ix_t[:, None] + offs[None, :]
        x = self.tokens[xi]
        y = self.tokens[xi + 1]
        w = self.weight[xi + 1]                     # weight aligns with targets
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
                w.to(self.device, non_blocking=True))


def noise_augment(x: torch.Tensor, tok: Tokenizer, p: float, rng: torch.Generator) -> torch.Tensor:
    """Perturb INPUT state tokens (bird_y / pipe_dx / gap_y) by +/-1 bin with probability ``p``,
    clamped to stay within each field's legal id range. Status/action/specials untouched."""
    if p <= 0:
        return x
    x = x.clone()
    spans = [(tok.by_off, tok.dx_off), (tok.dx_off, tok.gap_off), (tok.gap_off, tok.st_off)]
    for lo, hi in spans:
        field = (x >= lo) & (x < hi)
        hit = field & (torch.rand(x.shape, generator=rng, device=x.device) < p)
        delta = torch.where(torch.rand(x.shape, generator=rng, device=x.device) < 0.5,
                            -1, 1).to(x.dtype)
        xp = (x + delta).clamp(lo, hi - 1)
        x = torch.where(hit, xp, x)
    return x


@torch.no_grad()
def one_step_accuracy(model: DreamGPT, batcher: Batcher, tok: Tokenizer,
                      n_batches: int = 8, bs: int = 64) -> dict:
    """Per-field exact-bin accuracy (and bird_y within +/-1) on supervised targets."""
    model.eval()
    fields = {"bird_y": (tok.by_off, tok.dx_off), "pipe_dx": (tok.dx_off, tok.gap_off),
              "gap_y": (tok.gap_off, tok.st_off), "status": (tok.st_off, tok.act_off)}
    correct = {k: 0 for k in fields}
    within1 = 0
    total = {k: 0 for k in fields}
    by_lo, by_hi = fields["bird_y"]
    for _ in range(n_batches):
        x, y, _ = batcher.batch(bs)
        logits, _ = model(x)
        pred = logits.argmax(-1)
        for k, (lo, hi) in fields.items():
            sel = (y >= lo) & (y < hi)
            total[k] += int(sel.sum())
            correct[k] += int(((pred == y) & sel).sum())
            if k == "bird_y":
                in_field = (pred >= by_lo) & (pred < by_hi)
                within1 += int((sel & in_field & ((pred - y).abs() <= 1)).sum())
    model.train()
    out = {f"{k}_acc": (correct[k] / total[k] if total[k] else float("nan")) for k in fields}
    out["bird_y_within1"] = within1 / total["bird_y"] if total["bird_y"] else float("nan")
    return out


def lr_at(it: int, lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= total:
        return lr * min_ratio
    prog = (it - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def train(data_path: str, tier: str = "nano", iters: int = 3000, bs: int = 64,
          lr: float = 3e-3, warmup: int = 100, noise_aug: float = 0.0, dead_weight: float = 1.0,
          out: str = "checkpoints/model.pt", eval_every: int = 250, seed: int = 0,
          tok_cfg: TokenizerConfig | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    tok = Tokenizer(tok_cfg=tok_cfg) if tok_cfg else Tokenizer()
    cfg: ModelConfig = TIERS[tier]

    tokens = np.fromfile(data_path, dtype=np.uint16)
    batcher = Batcher(tokens, cfg.block_size, tok, device, seed=seed, dead_weight=dead_weight)
    model = DreamGPT(cfg, tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)

    print(f"device={device} tier={tier} params={model.num_params()/1e6:.2f}M "
          f"vocab={tok.vocab_size} block={cfg.block_size} tokens={tokens.size:,} "
          f"noise_aug={noise_aug} dead_weight={dead_weight}")

    t0 = time.perf_counter()
    model.train()
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, lr, warmup, iters)
        x, y, mask = batcher.batch(bs)
        if noise_aug > 0:
            x = noise_augment(x, tok, noise_aug, aug_gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, targets=y, loss_mask=mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % eval_every == 0 or it == iters - 1:
            acc = one_step_accuracy(model, batcher, tok)
            dt = time.perf_counter() - t0
            print(f"it {it:5d} | loss {loss.item():.4f} | bird_y {acc['bird_y_acc']:.3f} "
                  f"(±1 {acc['bird_y_within1']:.3f}) | status {acc['status_acc']:.3f} | "
                  f"{(it+1)/dt:.0f} it/s")

    final = one_step_accuracy(model, batcher, tok)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": cfg, "tier": tier,
                "vocab_size": tok.vocab_size, "tok_cfg": tok.tcfg, "engine_cfg": tok.ecfg,
                "final_acc": final}, out_p)
    print(f"saved {out_p}  final one-step: {final}")
    return final


def _main() -> None:
    ap = argparse.ArgumentParser(description="Train a DreamGPT.")
    ap.add_argument("--data", type=str, required=True, help="path to a .bin token stream")
    ap.add_argument("--tier", choices=list(TIERS), default="nano")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--noise-aug", type=float, default=0.0, help="per-token ±1-bin perturb prob")
    ap.add_argument("--dead-weight", type=float, default=1.0,
                    help="loss weight on the rare DEAD status token (collision learning)")
    ap.add_argument("--out", type=str, default="checkpoints/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.data, tier=args.tier, iters=args.iters, bs=args.batch_size, lr=args.lr,
          noise_aug=args.noise_aug, dead_weight=args.dead_weight, out=args.out, seed=args.seed)


if __name__ == "__main__":
    _main()
