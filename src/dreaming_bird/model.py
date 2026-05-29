"""A small nanoGPT-style decoder over the frame/action token stream.

Next-token prediction with causal self-attention (flash via SDPA). Includes a slot-constrained
generation method so a generated frame is always 4 valid field tokens — malformed frames are
impossible by construction. Training uses the full-sequence forward; generation here is
cacheless (fine for eval). A KV-cached real-time loop is added in Phase 4 (rollout.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .tokenizer import STATE_TOKENS_PER_FRAME, Tokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        if torch.onnx.is_in_onnx_export():
            # explicit attention with a dynamic causal mask — exports to plain ONNX ops
            # (MatMul/Softmax/Where) that onnxruntime-web's WebGPU backend supports, and keeps
            # the sequence length dynamic (SDPA's is_causal can bake in a fixed mask size).
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            ar = torch.arange(T, device=x.device)
            att = att + (ar[None, :] > ar[:, None]).to(att.dtype) * (-1e9)
            y = F.softmax(att, dim=-1) @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class DreamGPT(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())   # tied weight counted once

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1),
                                 reduction="none").view(B, T)
            if loss_mask is not None:
                loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            else:
                loss = ce.mean()
        return logits, loss

    @torch.no_grad()
    def generate_state(self, context: torch.Tensor, tok: Tokenizer,
                       temperature: float = 1.0, sample_slots: tuple[int, ...] = (2,),
                       legal_masks: list[torch.Tensor] | None = None,
                       n_slots: int = STATE_TOKENS_PER_FRAME) -> list[int]:
        """Generate the next frame's first ``n_slots`` state tokens with per-slot legal masking.

        ``context`` is a 1-D LongTensor of token ids on the model's device. ``sample_slots`` are
        the slots to sample (default: slot 2 = gap_y, which is stochastic at pipe spawns); other
        slots are greedy. ``n_slots`` < 4 stops early (the playable rollout generates only
        bird_y/pipe_dx/gap_y and derives the status from geometry). Cacheless.
        """
        was_training = self.training
        self.eval()
        if legal_masks is None:
            legal_masks = [torch.from_numpy(tok.legal_mask(s)).to(context.device)
                           for s in range(n_slots)]
        ctx = context
        out: list[int] = []
        for slot in range(n_slots):
            win = ctx[-self.cfg.block_size:].unsqueeze(0)
            logits, _ = self(win)
            logits = logits[0, -1].masked_fill(~legal_masks[slot], float("-inf"))
            if slot in sample_slots:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                token = int(torch.multinomial(probs, 1).item())
            else:
                token = int(logits.argmax().item())
            out.append(token)
            ctx = torch.cat([ctx, torch.tensor([token], device=ctx.device, dtype=ctx.dtype)])
        if was_training:
            self.train()
        return out
