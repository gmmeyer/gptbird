"""Play the dream: a pygame front-end over the learned world model.

The engine is gone — every frame comes from the network. Spacebar flaps. Game over when the
model itself predicts DEAD. With ``--shadow`` a translucent ghost shows where a real physics
engine (no pipes) would put the bird under the *same* action sequence, making any bird_y drift
visible. ``--autopilot``/``--headless`` run the loop with a scripted controller and no window
(used to smoke-test the pipeline and measure fps).

    uv run python -m dreaming_bird.play --checkpoint checkpoints/small_pipes.pt --shadow
"""

from __future__ import annotations

import argparse
import os


def run(checkpoint: str, fps: int = 30, scale: float = 1.6, shadow: bool = False,
        autopilot: bool = False, headless: bool = False, max_frames: int | None = None,
        seed: int = 0) -> dict:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    import torch

    from .config import EngineConfig
    from .engine import ACTION_FLAP, ACTION_NOFLAP, FlappyEngine
    from .model import DreamGPT
    from .policies import scripted_policy
    from .rollout import DreamStepper
    from .tokenizer import Tokenizer

    ck = torch.load(checkpoint, weights_only=False)
    ecfg: EngineConfig = ck.get("engine_cfg") or EngineConfig()
    tok = Tokenizer(engine_cfg=ecfg, tok_cfg=ck.get("tok_cfg"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamGPT(ck["model_cfg"], ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stepper = DreamStepper(model, tok, device=device, sample_slots=(2,), cfg=ecfg, seed=seed)
    autopolicy = scripted_policy()

    W, H = int(ecfg.width * scale), int(ecfg.height * scale)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dreaming Bird")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(12, int(16 * scale)))

    run_seed = seed

    def new_episode(s: int):
        obs = stepper.reset(seed=s)
        shadow_eng = None
        if shadow:
            shadow_eng = FlappyEngine(EngineConfig(pipes_enabled=False), seed=s)
            shadow_eng.reset(seed=s, start_y=obs.bird_y, start_vy=0.0)
        return obs, shadow_eng

    obs, shadow_eng = new_episode(run_seed)
    frame = pipes = ticks = 0
    prev_dx = obs.pipe_dx
    alive = True
    running = True

    def sx(x):
        return int(x * scale)

    while running:
        flap = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE and alive:
                    flap = True
                elif e.key == pygame.K_r and not alive:
                    run_seed += 1
                    obs, shadow_eng = new_episode(run_seed)
                    frame = pipes = 0
                    prev_dx = obs.pipe_dx
                    alive = True
        if autopilot and alive:
            flap = autopolicy(obs) == ACTION_FLAP

        if alive:
            action = ACTION_FLAP if flap else ACTION_NOFLAP
            obs = stepper.step(action)
            if shadow_eng is not None:
                shadow_eng.step(action)
            if obs.pipe_dx > prev_dx + ecfg.width * 0.3:   # next-pipe pointer jumped -> passed one
                pipes += 1
            prev_dx = obs.pipe_dx
            alive = obs.alive
            frame += 1
        elif autopilot:                                    # autopilot can't press R -> auto-restart
            run_seed += 1
            obs, shadow_eng = new_episode(run_seed)
            frame = pipes = 0
            prev_dx = obs.pipe_dx
            alive = True

        screen.fill((18, 22, 30))
        # next pipe (top + bottom segments) at x = bird_x + dx
        px = sx(ecfg.bird_x + obs.pipe_dx)
        pw = sx(ecfg.pipe_width)
        gap_top = sx(obs.gap_y - ecfg.gap_height / 2)
        gap_bot = sx(obs.gap_y + ecfg.gap_height / 2)
        pygame.draw.rect(screen, (60, 180, 90), (px, 0, pw, gap_top))
        pygame.draw.rect(screen, (60, 180, 90), (px, gap_bot, pw, H - gap_bot))
        # oracle-shadow ghost (true physics under identical actions)
        if shadow_eng is not None:
            pygame.draw.circle(screen, (120, 120, 150),
                               (sx(ecfg.bird_x), sx(shadow_eng.bird_y)),
                               sx(ecfg.bird_radius), width=2)
        # the dreamed bird
        pygame.draw.circle(screen, (240, 210, 70) if alive else (210, 70, 70),
                           (sx(ecfg.bird_x), sx(obs.bird_y)), sx(ecfg.bird_radius))
        hud = f"frame {frame}   pipes {pipes}"
        if not alive:
            hud += "    GAME OVER — R to restart, Esc to quit"
        screen.blit(font.render(hud, True, (225, 225, 235)), (8, 8))
        pygame.display.flip()
        if fps > 0:
            clock.tick(fps)
        ticks += 1
        if max_frames is not None and ticks >= max_frames:   # total iterations, not survived frames
            running = False

    pygame.quit()
    return {"frames": frame, "pipes_passed": pipes, "alive": alive}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Play the Dreaming Bird world model.")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/small_pipes.pt")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--shadow", action="store_true", help="draw the true-physics ghost bird")
    ap.add_argument("--autopilot", action="store_true", help="scripted controller plays")
    ap.add_argument("--headless", action="store_true", help="no window (pipeline smoke test)")
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(run(args.checkpoint, fps=args.fps, scale=args.scale, shadow=args.shadow,
              autopilot=args.autopilot, headless=args.headless, max_frames=args.frames,
              seed=args.seed))


if __name__ == "__main__":
    _main()
