from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class SGDMMode(str, Enum):
    ANALYTIC = "analytic"
    LEARNED = "learned"


@dataclass(frozen=True)
class AndersonAccelerationConfig:
    memory: int = 5
    beta: float = 1.0
    ridge: float = 1e-4
    max_iter: int = 50
    tol: float = 1e-4
    eps: float = 1e-5
    residual_reduction: str = "mean"


def _reduce_scalar(v: torch.Tensor, mode: str) -> float:
    if mode == "max":
        return float(v.max().item())
    return float(v.mean().item())


def anderson_acceleration(
    fixed_point_map: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    config: AndersonAccelerationConfig,
) -> Tuple[torch.Tensor, List[float], int]:
    batch_size, channels, height, width = x0.shape
    flat_dim = channels * height * width

    m = int(config.memory)
    m = max(2, m)
    m = min(m, int(config.max_iter))

    x_hist = torch.zeros(batch_size, m, flat_dim, dtype=x0.dtype, device=x0.device)
    f_hist = torch.zeros_like(x_hist)

    x_hist[:, 0] = x0.reshape(batch_size, -1)
    f0 = fixed_point_map(x0)
    f_hist[:, 0] = f0.reshape(batch_size, -1)

    slot1 = 1 % m
    x_hist[:, slot1] = f_hist[:, 0]
    f1 = fixed_point_map(f_hist[:, 0].view_as(x0))
    f_hist[:, slot1] = f1.reshape(batch_size, -1)

    H = torch.zeros(batch_size, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = 1
    H[:, 1:, 0] = 1
    y = torch.zeros(batch_size, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    residual_trace: List[float] = []
    eye_cache: dict[int, torch.Tensor] = {}

    k = 1
    for k in range(2, int(config.max_iter)):
        n = min(k, m)
        G = f_hist[:, :n] - x_hist[:, :n]

        if n not in eye_cache:
            eye_cache[n] = torch.eye(n, dtype=x0.dtype, device=x0.device)[None]

        H[:, 1 : n + 1, 1 : n + 1] = torch.bmm(G, G.transpose(1, 2)) + float(config.ridge) * eye_cache[n]
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]

        mix = float(config.beta) * (alpha[:, None] @ f_hist[:, :n])[:, 0] + (1.0 - float(config.beta)) * (
            alpha[:, None] @ x_hist[:, :n]
        )[:, 0]

        slot = k % m
        x_hist[:, slot] = mix
        f_mix = fixed_point_map(mix.view_as(x0))
        f_hist[:, slot] = f_mix.reshape(batch_size, -1)

        num = (f_hist[:, slot] - x_hist[:, slot]).norm(dim=1)
        den = float(config.eps) + f_hist[:, slot].norm(dim=1)
        rel = num / den
        r = _reduce_scalar(rel, config.residual_reduction)
        residual_trace.append(r)
        if r < float(config.tol):
            break

    x_star = x_hist[:, k % m].view_as(x0)
    return x_star, residual_trace, int(k + 1)


def to_3d(x_bchw: torch.Tensor) -> torch.Tensor:
    return rearrange(x_bchw, "b c h w -> b (h w) c")


def to_4d(x_bnc: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x_bnc, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | torch.Size) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("BiasFree_LayerNorm expects a single normalized dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sigma = tokens.var(-1, keepdim=True, unbiased=False)
        return tokens / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | torch.Size) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBias_LayerNorm expects a single normalized dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mu = tokens.mean(-1, keepdim=True)
        sigma = tokens.var(-1, keepdim=True, unbiased=False)
        return (tokens - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.core = BiasFree_LayerNorm(dim)
        else:
            self.core = WithBias_LayerNorm(dim)

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        h, w = x_bchw.shape[-2:]
        return to_4d(self.core(to_3d(x_bchw)), h=h, w=w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * float(expansion))
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x_bchw)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.proj_out(x)


def local_conv(dim: int) -> nn.Conv2d:
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)


def window_partition(x: torch.Tensor, window: int, H: int, W: int) -> torch.Tensor:
    B, heads, N, C = x.shape
    x = x.contiguous().view(B * heads, N, C).contiguous().view(B * heads, H, W, C)
    B2, H2, W2, C2 = x.shape
    x = x.view(B2, H2 // window, window, W2 // window, window, C2)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, C2).view(-1, window * window, C2)
    return windows


def window_reverse(windows: torch.Tensor, window: int, H: int, W: int, heads: int) -> torch.Tensor:
    B_heads = int(windows.shape[0] / (H * W / window / window))
    x = windows.view(B_heads, H // window, W // window, window, window, -1)
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B_heads, H, W, -1)
        .view(B_heads // heads, heads, H, W, -1)
        .contiguous()
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(B_heads // heads, H, W, -1)
        .view(B_heads // heads, H * W, -1)
    )
    return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        mask_guided: bool = False,
        sr_ratio: int = 2,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        linear: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}.")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = float(qk_scale) if qk_scale is not None else head_dim**-0.5
        self.sr_ratio = int(sr_ratio)
        self.mask_guided = bool(mask_guided)

        if self.sr_ratio > 1:
            if self.mask_guided:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio == 8:
                    f1, f2, f3 = 8 * 8, 16, 8
                elif self.sr_ratio == 4:
                    f1, f2, f3 = 4 * 4, 8, 4
                elif self.sr_ratio == 2:
                    f1, f2, f3 = 2, 1, None
                else:
                    f1, f2, f3 = 2, 1, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
                else:
                    self.f3 = None
            else:
                self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)

        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

        self.linear = bool(linear)

        self.sigma = nn.Parameter(torch.tensor([0.0002], dtype=torch.float32), requires_grad=True)
        self.magnitude = nn.Parameter(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _fam(self, B: int, device: torch.device, dtype: torch.dtype, side: int = 64) -> torch.Tensor:
        coord = torch.arange(-(side / 2) + 0.5, (side / 2) + 0.5, 1, device=device, dtype=dtype)
        try:
            Xg, Yg = torch.meshgrid(coord, coord, indexing="ij")
        except TypeError:
            Xg, Yg = torch.meshgrid(coord, coord)
        Z = self.magnitude * self.sigma / (2.0 * math.pi) * torch.exp((-0.5 * self.sigma) * (Xg * Xg + Yg * Yg)) + 1.0
        Z = Z.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        return Z

    def forward(self, x_bchw: torch.Tensor, se_state):
        B, C, H, W = x_bchw.shape
        tokens = x_bchw.flatten(2).transpose(1, 2)
        B, N, C = tokens.shape

        lepe = (
            self.lepe_conv(self.lepe_linear(tokens).transpose(1, 2).view(B, C, H, W))
            .view(B, C, -1)
            .transpose(-1, -2)
        )

        if self.sr_ratio > 1:
            if se_state is None:
                q_local = self.q1(tokens).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)

                feat = tokens.permute(0, 2, 1).reshape(B, C, H, W)
                feat_reduced = self.sr(feat).reshape(B, C, -1).permute(0, 2, 1)
                feat_reduced = self.act(self.norm(feat_reduced))

                kv_global = (
                    self.kv1(feat_reduced)
                    .reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                k_global, v_global = kv_global[0], kv_global[1]

                attn_global = (q_local @ k_global.transpose(-2, -1)) * self.scale
                attn_global = attn_global.softmax(dim=-1)
                attn_global = self.attn_drop(attn_global)
                x_global = (attn_global @ v_global).transpose(1, 2).reshape(B, N, C // 2)

                global_sig = torch.mean(attn_global.detach().mean(1), dim=2).reshape(-1, H, W)

                q_nonlocal = self.q2(tokens).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)
                kv_local = (
                    self.kv2(feat.reshape(B, C, -1).permute(0, 2, 1))
                    .reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                k_local, v_local = kv_local[0], kv_local[1]

                q_window = 4
                window = 4
                q_nonlocal_w = window_partition(q_nonlocal, q_window, H, W)
                k_local_w = window_partition(k_local, window, H, W)
                v_local_w = window_partition(v_local, window, H, W)

                attn_local = (q_nonlocal_w @ k_local_w.transpose(-2, -1)) * self.scale
                attn_local = attn_local.softmax(dim=-1)
                attn_local = self.attn_drop(attn_local)

                x_local = attn_local @ v_local_w
                x_local = window_reverse(x_local, q_window, H, W, self.num_heads // 2)

                local_sig = torch.mean(
                    attn_local.detach()
                    .view(
                        B,
                        self.num_heads // 2,
                        (H // window) * (W // window),
                        window * window,
                        window * window,
                    )
                    .mean(1),
                    dim=2,
                )
                local_sig = local_sig.view(B, H // window, W // window, window, window)
                local_sig = local_sig.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)

                fused = torch.cat([x_global, x_local], dim=-1)
                fused = self.proj(fused + lepe)
                fused = self.proj_drop(fused)

                se_map = (local_sig + global_sig) / 2.0

                if H // self.sr_ratio == 16:
                    fam = self._fam(B=B, device=x_bchw.device, dtype=x_bchw.dtype, side=64)
                    se_map = (se_map.unsqueeze(1) * fam).squeeze(1)

                se_flat_1 = se_map.view(B, H * W)
                se_flat_2 = se_map.permute(0, 2, 1).reshape(B, H * W)
                se_state = [se_flat_1, se_flat_2]

            else:
                q = self.q(tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                se_1, se_2 = se_state
                _, sort_idx_1 = torch.sort(se_1, dim=1)
                _, sort_idx_2 = torch.sort(se_2, dim=1)

                if self.sr_ratio == 8:
                    t1, t2, t3 = H * W // (8 * 8), H * W // 16, H * W // 8
                    t1, t2, t3 = t1 // 4, t2 // 2, t3 // 4
                elif self.sr_ratio == 4:
                    t1, t2, t3 = H * W // (4 * 4), H * W // 8, H * W // 4
                    t1, t2, t3 = t1 // 4, t2 // 2, t3 // 4
                elif self.sr_ratio == 2:
                    t1, t2 = H * W // 2, H * W // 1
                    t1, t2 = t1 // 2, t2 // 2

                if self.sr_ratio in (4, 8):
                    p1 = torch.gather(tokens, 1, sort_idx_1[:, : H * W // 4].unsqueeze(-1).repeat(1, 1, C))
                    p2 = torch.gather(tokens, 1, sort_idx_1[:, H * W // 4 : H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3 = torch.gather(tokens, 1, sort_idx_1[:, H * W // 4 * 3 :].unsqueeze(-1).repeat(1, 1, C))

                    seq1 = torch.cat(
                        [
                            self.f1(p1.permute(0, 2, 1).reshape(B, C, t1, -1)).squeeze(-1),
                            self.f2(p2.permute(0, 2, 1).reshape(B, C, t2, -1)).squeeze(-1),
                            self.f3(p3.permute(0, 2, 1).reshape(B, C, t3, -1)).squeeze(-1),
                        ],
                        dim=-1,
                    ).permute(0, 2, 1)

                    tokens_swapped = tokens.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(tokens_swapped, 1, sort_idx_2[:, : H * W // 4].unsqueeze(-1).repeat(1, 1, C))
                    p2_ = torch.gather(tokens_swapped, 1, sort_idx_2[:, H * W // 4 : H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(tokens_swapped, 1, sort_idx_2[:, H * W // 4 * 3 :].unsqueeze(-1).repeat(1, 1, C))

                    seq2 = torch.cat(
                        [
                            self.f1(p1_.permute(0, 2, 1).reshape(B, C, t1, -1)).squeeze(-1),
                            self.f2(p2_.permute(0, 2, 1).reshape(B, C, t2, -1)).squeeze(-1),
                            self.f3(p3_.permute(0, 2, 1).reshape(B, C, t3, -1)).squeeze(-1),
                        ],
                        dim=-1,
                    ).permute(0, 2, 1)

                else:
                    p1 = torch.gather(tokens, 1, sort_idx_1[:, : H * W // 2].unsqueeze(-1).repeat(1, 1, C))
                    p2 = torch.gather(tokens, 1, sort_idx_1[:, H * W // 2 :].unsqueeze(-1).repeat(1, 1, C))

                    seq1 = torch.cat(
                        [
                            self.f1(p1.permute(0, 2, 1).reshape(B, C, t1, -1)).squeeze(-1),
                            self.f2(p2.permute(0, 2, 1).reshape(B, C, t2, -1)).squeeze(-1),
                        ],
                        dim=-1,
                    ).permute(0, 2, 1)

                    tokens_swapped = tokens.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(tokens_swapped, 1, sort_idx_2[:, : H * W // 2].unsqueeze(-1).repeat(1, 1, C))
                    p2_ = torch.gather(tokens_swapped, 1, sort_idx_2[:, H * W // 2 :].unsqueeze(-1).repeat(1, 1, C))

                    seq2 = torch.cat(
                        [
                            self.f1(p1_.permute(0, 2, 1).reshape(B, C, t1, -1)).squeeze(-1),
                            self.f2(p2_.permute(0, 2, 1).reshape(B, C, t2, -1)).squeeze(-1),
                        ],
                        dim=-1,
                    ).permute(0, 2, 1)

                kv1 = (
                    self.kv1(seq1)
                    .reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                kv2 = (
                    self.kv2(seq2)
                    .reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )

                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                fused = (attn @ v).transpose(1, 2).reshape(B, N, C)
                fused = self.proj(fused + lepe)
                fused = self.proj_drop(fused)
                se_state = None

        else:
            q = self.q(tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(tokens).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            fused = (attn @ v).transpose(1, 2).reshape(B, N, C)
            fused = self.proj(fused + lepe)
            fused = self.proj_drop(fused)
            se_state = None

        return fused.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous(), se_state


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float, bias: bool, norm_type: str, mask_guided: bool, sr_ratio: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attn = Attention(dim, num_heads, bias, mask_guided, sr_ratio)
        self.norm2 = LayerNorm(dim, norm_type)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x_bchw: torch.Tensor, se_state):
        attn_out, se_state = self.attn(self.norm1(x_bchw), se_state)
        x = x_bchw + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, se_state


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DEQFixedPoint(nn.Module):
    def __init__(self, f_map: nn.Module, forward_cfg: AndersonAccelerationConfig, backward_cfg: Optional[AndersonAccelerationConfig] = None) -> None:
        super().__init__()
        self.f_map = f_map
        self.forward_cfg = forward_cfg
        self.backward_cfg = backward_cfg if backward_cfg is not None else forward_cfg
        self.forward_residuals: Optional[List[float]] = None
        self.forward_iterations: Optional[int] = None

    def forward(self, phi_boundary: torch.Tensor, x_lap_init: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_star, res, iters = anderson_acceleration(
                lambda z: self.f_map(z, phi_boundary, x_lap_init, False),
                x0,
                self.forward_cfg,
            )
        self.forward_residuals = res
        self.forward_iterations = iters

        params = tuple(p for p in self.f_map.parameters() if p.requires_grad)
        f_map = self.f_map
        backward_cfg = self.backward_cfg

        class _ImplicitBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, equilibrium: torch.Tensor, phi: torch.Tensor, lap: torch.Tensor, *flat_params: torch.Tensor) -> torch.Tensor:
                ctx.f_map = f_map
                ctx.backward_cfg = backward_cfg
                ctx.save_for_backward(equilibrium, phi, lap, *flat_params)
                return equilibrium

            @staticmethod
            def backward(ctx, grad_equilibrium: torch.Tensor):
                equilibrium, phi, lap, *flat_params = ctx.saved_tensors
                f_map = ctx.f_map
                backward_cfg = ctx.backward_cfg

                with torch.enable_grad():
                    z = equilibrium.detach().requires_grad_(True)
                    phi_req = phi.detach().requires_grad_(phi.requires_grad)
                    lap_req = lap.detach().requires_grad_(lap.requires_grad)
                    f_z = f_map(z, phi_req, lap_req, False)

                rhs = grad_equilibrium.to(dtype=z.dtype)

                def j_tf(v: torch.Tensor) -> torch.Tensor:
                    return torch.autograd.grad(f_z, z, v, retain_graph=True, create_graph=False)[0]

                def lin_fp(v: torch.Tensor) -> torch.Tensor:
                    return rhs + j_tf(v)

                v_star, _, _ = anderson_acceleration(lin_fp, rhs, backward_cfg)

                need_phi = phi.requires_grad
                need_lap = lap.requires_grad

                targets: List[torch.Tensor] = []
                if need_phi:
                    targets.append(phi_req)
                if need_lap:
                    targets.append(lap_req)
                targets.extend(flat_params)

                grads = torch.autograd.grad(f_z, targets, v_star, allow_unused=True, retain_graph=False)

                grad_phi = None
                grad_lap = None
                offset = 0
                if need_phi:
                    grad_phi = grads[offset]
                    offset += 1
                if need_lap:
                    grad_lap = grads[offset]
                    offset += 1
                grad_params = grads[offset:]

                return (None, grad_phi, grad_lap, *grad_params)

        return _ImplicitBackward.apply(x_star, phi_boundary, x_lap_init, *params)


class TranIUNet(nn.Module):
    def __init__(
        self,
        Phi: torch.Tensor,
        mask_ET: torch.Tensor,
        iterations: int,
        sgdm_mode: str = "analytic",
    ) -> None:
        super().__init__()
        self.Phi = Phi
        self.mask_ET = mask_ET
        self.iterations = int(iterations)
        self.f = TranBase(self.Phi, self.mask_ET, sgdm_mode=sgdm_mode)

        fwd_cfg = AndersonAccelerationConfig(tol=1e-4, max_iter=self.iterations, memory=5, beta=1.0, ridge=1e-4)
        bwd_cfg = AndersonAccelerationConfig(tol=1e-4, max_iter=self.iterations, memory=5, beta=1.0, ridge=1e-4)
        self.DEQ = DEQFixedPoint(self.f, forward_cfg=fwd_cfg, backward_cfg=bwd_cfg)

    def forward(self, x_lap_init: torch.Tensor, phi_boundary: torch.Tensor) -> torch.Tensor:
        x0 = self.f(x_lap_init, phi_boundary, x_lap_init, True)
        x_star = self.DEQ(phi_boundary, x_lap_init, x0)
        x_hat = self.f(x_star, phi_boundary, x_lap_init, False)
        return x_hat


class TranBase(nn.Module):
    def __init__(
        self,
        Phi: torch.Tensor,
        mask_ET: torch.Tensor,
        inp_channels: int = 1,
        out_channels: int = 1,
        dim: int = 16,
        num_blocks: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [4, 4, 4, 4],
        num_refinement_blocks: int = 2,
        heads: List[int] = [2, 4, 4, 2],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
        sgdm_mode: str = "analytic",
        learned_width: int = 512,
        learned_depth: int = 4,
    ) -> None:
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.ModuleList(
            [
                TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[0])
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**1), heads[1], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[1])
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**2), heads[2], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[2])
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**3), heads[3], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[3])
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**2), heads[2], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[2])
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**1), heads[1], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[1])
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**1), heads[0], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[0])
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.ModuleList(
            [
                TransformerBlock(int(dim * 2**1), heads[0], ffn_expansion_factor, bias, norm_type, True if i % 2 == 1 else False, sr_ratios[0])
                for i in range(num_refinement_blocks)
            ]
        )

        self.dual_pixel_task = bool(dual_pixel_task)
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.register_buffer("S_sens", Phi, persistent=True)
        self.register_buffer("M_mask", mask_ET, persistent=True)

        self._cached_S_t: Optional[torch.Tensor] = None
        self._cached_S_tS: Optional[torch.Tensor] = None

        self.mu_logit = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)

        self.sgdm_mode = SGDMMode(sgdm_mode)
        if self.sgdm_mode == SGDMMode.LEARNED:
            n_meas, n_state = int(self.S_sens.shape[0]), int(self.S_sens.shape[1])
            dims = [n_state] + [int(learned_width)] * (int(learned_depth) - 1) + [n_meas]
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.NS = nn.Sequential(*layers) # A lightweight U-net can also be used.

            dims_t = [n_meas] + [int(learned_width)] * (int(learned_depth) - 1) + [n_state]
            layers_t = []
            for i in range(len(dims_t) - 2):
                layers_t.append(nn.Linear(dims_t[i], dims_t[i + 1]))
                layers_t.append(nn.GELU())
                layers_t.append(nn.LayerNorm(dims_t[i + 1]))
            layers_t.append(nn.Linear(dims_t[-2], dims_t[-1]))
            self.NST = nn.Sequential(*layers_t)
        else:
            self.NS = None
            self.NST = None

    def _S_t(self) -> torch.Tensor:
        if self._cached_S_t is None or self._cached_S_t.device != self.S_sens.device or self._cached_S_t.dtype != self.S_sens.dtype:
            self._cached_S_t = self.S_sens.transpose(0, 1).contiguous()
        return self._cached_S_t

    def _S_tS(self) -> torch.Tensor:
        if self._cached_S_tS is None or self._cached_S_tS.device != self.S_sens.device or self._cached_S_tS.dtype != self.S_sens.dtype:
            S_t = self._S_t()
            self._cached_S_tS = S_t @ self.S_sens
        return self._cached_S_tS

    def forward(self, x_k: torch.Tensor, phi_boundary: torch.Tensor, x_lap_init: torch.Tensor, use_lap_init: bool):
        phi_mb = phi_boundary.squeeze(1).squeeze(-1).t().contiguous()

        if use_lap_init:
            x_k = x_lap_init

        side = int(x_k.size(2))
        x_full = x_k.contiguous().view(x_k.size(0), x_k.size(1), side * side, -1)
        x_full = x_full.squeeze(1).squeeze(2).t().contiguous()

        x_state = self.M_mask @ x_full

        mu = torch.sigmoid(self.mu_logit)

        if self.sgdm_mode == SGDMMode.ANALYTIC:
            S_tS = self._S_tS()
            S_t = self._S_t()
            S_t_phi = S_t @ phi_mb
            x_state = x_state - mu * (S_tS @ x_state) + mu * S_t_phi
        else:
            x_bn = x_state.t().contiguous()
            phi_bn = phi_mb.t().contiguous()
            resid_bn = self.NS(x_bn) - phi_bn
            grad_bn = self.NST(resid_bn)
            x_state = x_state - mu * grad_bn.t().contiguous()

        x_full = self.M_mask.transpose(0, 1) @ x_state
        x_full = x_full.view(side, side, -1).unsqueeze(0)
        r_k = x_full.permute(3, 0, 1, 2).contiguous()

        se_state = None

        enc1 = self.patch_embed(r_k)
        for block in self.encoder_level1:
            enc1, se_state = block(enc1, se_state)

        enc2_in = self.down1_2(enc1)
        for block in self.encoder_level2:
            enc2, se_state = block(enc2_in, se_state)
            enc2_in = enc2

        enc3_in = self.down2_3(enc2_in)
        for block in self.encoder_level3:
            enc3, se_state = block(enc3_in, se_state)
            enc3_in = enc3

        enc4_in = self.down3_4(enc3_in)
        latent = enc4_in
        for block in self.latent:
            latent, se_state = block(latent, se_state)

        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3_in], dim=1)
        dec3 = self.reduce_chan_level3(dec3)
        for block in self.decoder_level3:
            dec3, se_state = block(dec3, se_state)

        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2_in], dim=1)
        dec2 = self.reduce_chan_level2(dec2)
        for block in self.decoder_level2:
            dec2, se_state = block(dec2, se_state)

        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        for block in self.decoder_level1:
            dec1, se_state = block(dec1, se_state)

        for block in self.refinement:
            dec1, se_state = block(dec1, se_state)

        if self.dual_pixel_task:
            dec1 = dec1 + self.skip_conv(enc1)
            out = self.output(dec1)
        else:
            out = self.output(dec1) + r_k

        return out
