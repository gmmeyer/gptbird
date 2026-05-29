"""Export a trained DreamGPT to ONNX for client-side (in-browser) inference.

Produces ``web/model.onnx`` (dynamic sequence length, returns last-position logits) and
``web/config.json`` (tokenizer offsets, quantization bins, and engine geometry) so the
JavaScript side can build legal masks, decode tokens to physical values, and render. Validates
the exported graph against PyTorch.

    uv run --with onnx --with onnxruntime python -m dreaming_bird.export_onnx \
        --checkpoint checkpoints/small_pipes.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import DreamGPT
from .tokenizer import BOS, EOS, PAD, SEP, Tokenizer


class _LastLogits(torch.nn.Module):
    """Wraps the model to return only the last position's logits (all the decode loop needs)."""

    def __init__(self, model: DreamGPT):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits[:, -1, :]


class _ExplicitRMSNorm(torch.nn.Module):
    """Drop-in for nn.RMSNorm using primitive ops (the legacy ONNX exporter can't emit
    aten::rms_norm at opset 17). Numerically identical given the same weight and eps."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def _swap_rmsnorm(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.RMSNorm):
            eps = child.eps if child.eps is not None else torch.finfo(torch.float32).eps
            setattr(module, name, _ExplicitRMSNorm(child.weight.data, eps))
        else:
            _swap_rmsnorm(child)


def build_config(tok: Tokenizer, block_size: int, web_context_cap: int = 256) -> dict:
    e, t = tok.ecfg, tok.tcfg
    return {
        "vocab_size": tok.vocab_size,
        "specials": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "SEP": SEP},
        "offsets": {"by": tok.by_off, "dx": tok.dx_off, "gap": tok.gap_off,
                    "status": tok.st_off, "action": tok.act_off},
        "bins": {"bird_y": t.bird_y_bins, "pipe_dx": t.pipe_dx_bins, "gap_y": t.gap_y_bins},
        "tokens": {
            "status_alive": tok.status_token(True), "status_dead": tok.status_token(False),
            "action_noflap": tok.act_off + 0, "action_flap": tok.act_off + 1,
        },
        "engine": {
            "height": e.height, "width": e.width, "bird_x": e.bird_x,
            "bird_radius": e.bird_radius, "gap_height": e.gap_height,
            "pipe_width": e.pipe_width, "max_dx": e.max_dx,
            "gravity": e.gravity, "flap_impulse": e.flap_impulse,  # for the JS shadow ghost
            "start_y": e.start_y,
        },
        "block_size": block_size,
        "web_context_cap": web_context_cap,   # JS sliding-window cap (drift is low; small=fast)
        "state_tokens_per_frame": 4,
        "slot_order": ["bird_y", "pipe_dx", "gap_y", "status"],
    }


def export(checkpoint: str, out_dir: str = "web", opset: int = 17,
           web_context_cap: int = 256) -> dict:
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    tok = Tokenizer(engine_cfg=ck.get("engine_cfg"), tok_cfg=ck.get("tok_cfg"))
    model = DreamGPT(ck["model_cfg"], ck["vocab_size"])
    model.load_state_dict(ck["model"])
    model.eval()
    _swap_rmsnorm(model)        # export-friendly norm (identical math, primitive ops)
    wrap = _LastLogits(model).eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"

    dummy = torch.tensor([[BOS, tok.by_token(64), tok.dx_token(30),
                           tok.gap_token(16), tok.status_token(True)]], dtype=torch.long)
    torch.onnx.export(
        wrap, (dummy,), str(onnx_path), input_names=["idx"], output_names=["logits"],
        dynamic_axes={"idx": {1: "T"}, "logits": {0: "B"}}, opset_version=opset, dynamo=False)

    cfg = build_config(tok, block_size=ck["model_cfg"].block_size, web_context_cap=web_context_cap)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- validate against PyTorch on a few variable-length contexts ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for T in (5, 17, 60, 123):
        ctx = [BOS]
        for _ in range(T):
            ctx.append(int(rng.integers(tok.by_off, tok.vocab_size)))
        idx = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            ref = wrap(idx).numpy()
        got = sess.run(None, {"idx": idx.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        # argmax must agree (what greedy decoding depends on)
        assert int(ref.argmax()) == int(got.argmax()), f"argmax mismatch at T={T}"

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB), vocab={tok.vocab_size}, "
          f"block_size={cfg['block_size']}, web_context_cap={web_context_cap}")
    print(f"onnx-vs-torch max logit diff = {max_diff:.2e} (argmax agrees on all tested lengths)")
    print(f"wrote {out/'config.json'}")
    return {"onnx": str(onnx_path), "max_diff": max_diff, "size_mb": size_mb}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Export DreamGPT to ONNX for the web demo.")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/small_pipes.pt")
    ap.add_argument("--out-dir", type=str, default="web")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--web-context-cap", type=int, default=256)
    args = ap.parse_args()
    export(args.checkpoint, out_dir=args.out_dir, opset=args.opset,
           web_context_cap=args.web_context_cap)


if __name__ == "__main__":
    _main()
