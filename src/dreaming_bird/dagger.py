"""DAgger / scheduled-sampling data: the model's OWN rollout states, relabeled by the oracle.

The model phantom-dies only in its self-generated free-rollout distribution, which teacher-forced
engine data never covers. So we roll the model forward (batched, for speed), let it sample its own
frames, and label each frame's status with the geometry oracle (:func:`engine.collides`) — which we
know is correct. Fine-tuning on these (self-generated state -> correct status) examples is the
principled fix for the exposure bias. The world tokens (bird_y/pipe_dx/gap_y) are the model's own,
so it isn't taught to change its (already good) world model — only to call death correctly where it
actually visits.
"""

from __future__ import annotations

import numpy as np
import torch

from .engine import ACTION_FLAP, ACTION_NOFLAP, FlappyEngine, Obs, collides
from .policies import (lazy_center_policy, noisy_scripted_policy, random_policy,
                       random_start, scripted_policy)
from .tokenizer import BOS, EOS, Tokenizer


def _assign_policy(rng, center):
    r = rng.random()
    if r < 0.28:
        return scripted_policy()
    if r < 0.52:
        return lazy_center_policy(center=center)
    if r < 0.70:
        return noisy_scripted_policy(seed=int(rng.integers(2**31 - 1)))
    if r < 0.86:
        return random_policy(p_flap=float(rng.uniform(0.05, 0.3)), seed=int(rng.integers(2**31 - 1)))
    return "anti"   # anti-gap: deliberately fly into pipes (death coverage)


@torch.no_grad()
def generate_dagger(model, tok: Tokenizer, cfg=None, device: str = "cuda",
                    n_episodes: int = 4000, max_frames: int = 220, cap: int = 256,
                    seed: int = 0, batch: int = 256) -> tuple[np.ndarray, dict]:
    cfg = cfg or tok.ecfg
    rng = np.random.default_rng(seed)
    masks = [torch.from_numpy(tok.legal_mask(s)).to(device) for s in range(3)]
    center = cfg.height / 2.0
    r, H, MAXDX, gh2 = cfg.bird_radius, cfg.height, cfg.max_dx, cfg.gap_height / 2.0
    nb, nd, ng = tok.tcfg.bird_y_bins, tok.tcfg.pipe_dx_bins, tok.tcfg.gap_y_bins
    DEAD, ALIVE = tok.status_token(False), tok.status_token(True)
    streams: list[np.ndarray] = []
    n_deaths = 0

    done_eps = 0
    while done_eps < n_episodes:
        B = min(batch, n_episodes - done_eps)
        pols = [_assign_policy(rng, center) for _ in range(B)]
        rec: list[list[int]] = []
        cur: list[list[float]] = []
        for _ in range(B):
            e = FlappyEngine(cfg, seed=int(rng.integers(2**31 - 1)))
            if rng.random() < 0.2:
                sy, svy = random_start(cfg, rng)
                o = e.reset(seed=int(rng.integers(2**31 - 1)), start_y=sy, start_vy=svy)
            else:
                o = e.reset(seed=int(rng.integers(2**31 - 1)))
            rec.append([BOS] + tok.encode_obs(o))
            cur.append([o.bird_y, o.pipe_dx, o.gap_y])
        ctx = torch.tensor(rec, device=device, dtype=torch.long)   # (B, 5), equal length
        done = [False] * B

        for _ in range(max_frames):
            acts = []
            for i in range(B):
                by, dx, gp = cur[i]
                p = pols[i]
                if p == "anti":
                    a = ACTION_FLAP if by > (cfg.height - gp) else ACTION_NOFLAP
                else:
                    a = p(Obs(by, dx, gp, True))
                acts.append(a)
            ctx = torch.cat([ctx, torch.tensor([[tok.action_token(a)] for a in acts],
                                               device=device, dtype=torch.long)], dim=1)
            cols = []
            dev_type = "cuda" if device == "cuda" else "cpu"
            for slot in range(3):
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                    logits = model(ctx[:, -cap:])[0][:, -1, :]
                logits = logits.float()                       # fp32 for stable mask/sample
                logits[:, ~masks[slot]] = float("-inf")
                col = (torch.multinomial(torch.softmax(logits, dim=-1), 1) if slot == 2
                       else logits.argmax(dim=-1, keepdim=True))
                ctx = torch.cat([ctx, col], dim=1)
                cols.append(col.squeeze(1))
            byb, dxb, gpb = cols                                  # (B,) bin tokens
            by = (byb - tok.by_off + 0.5) / nb * H                # vectorized dequant + collide
            dx = (dxb - tok.dx_off + 0.5) / nd * MAXDX
            gp = (gpb - tok.gap_off + 0.5) / ng * H
            dead = ((by - r <= 0) | (by + r >= H) |
                    ((dx <= r) & ((by - r < gp - gh2) | (by + r > gp + gh2))))
            status = torch.where(dead, torch.full_like(byb, DEAD),
                                 torch.full_like(byb, ALIVE)).unsqueeze(1)
            ctx = torch.cat([ctx, status], dim=1)
            byb_l, dxb_l, gpb_l = byb.tolist(), dxb.tolist(), gpb.tolist()
            by_l, dx_l, gp_l, dead_l = by.tolist(), dx.tolist(), gp.tolist(), dead.tolist()
            for i in range(B):
                cur[i] = (by_l[i], dx_l[i], gp_l[i])
                if not done[i]:
                    rec[i].append(tok.action_token(acts[i]))
                    rec[i].extend([byb_l[i], dxb_l[i], gpb_l[i], DEAD if dead_l[i] else ALIVE])
                    if dead_l[i]:
                        done[i] = True
                        n_deaths += 1
            if all(done):
                break

        for i in range(B):
            rec[i].append(EOS)
            streams.append(np.array(rec[i], dtype=np.uint16))
        done_eps += B
        print(f"  dagger: {done_eps}/{n_episodes} episodes", flush=True)

    tokens = np.concatenate(streams)
    meta = {"source": "dagger", "n_episodes": n_episodes, "n_tokens": int(tokens.size),
            "death_rate": round(n_deaths / n_episodes, 3), "cap": cap}
    return tokens, meta


def _main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .config import EngineConfig
    from .model import DreamGPT

    ap = argparse.ArgumentParser(description="Generate DAgger data from a trained model.")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/small_pipes.pt")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--out", type=str, default="data/dagger")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    tok = Tokenizer(engine_cfg=ck.get("engine_cfg"), tok_cfg=ck.get("tok_cfg"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamGPT(ck["model_cfg"], ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tokens, meta = generate_dagger(model, tok, cfg=ck.get("engine_cfg"), device=device,
                                   n_episodes=args.episodes, batch=args.batch, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(out.with_suffix(".bin"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}.bin")


if __name__ == "__main__":
    _main()
