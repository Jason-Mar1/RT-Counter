"""Weaformer, Weaver-Attention, ConvAttention and the feature enhancer.

Sec. 3.3 and Fig. 3(b)(c) of the paper; ConvAttention is detailed in the
supplementary material (Fig. 7).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FFN, CrossAttentionLayer, DropPath, MultiHeadAttention, sine_position_embedding_2d


class ConvAttention(nn.Module):
    """Local attention restricted to a ``k x k`` window (the local path of Weaver-Attention).

    Q, K and V come from depth-wise convolutions. Unfold gathers the ``k x k``
    neighbourhood of every position; the softmax-normalised Q-K similarities act
    as a data-dependent convolution kernel that re-weights the V neighbourhood.
    The cost is linear in the number of tokens.
    """

    def __init__(self, dim, num_heads=6, kernel_size=3, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.scale = self.head_dim ** -0.5
        self.q = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.k = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.v = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size // 2)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        """x: ``[B, C, H, W]`` -> ``[B, C, H, W]``."""
        B, C, H, W = x.shape
        h, d, kk, L = self.num_heads, self.head_dim, self.kernel_size ** 2, H * W
        q = self.q(x).reshape(B, h, d, 1, L).permute(0, 1, 4, 3, 2)                   # [B, h, L, 1, d]
        k = self.unfold(self.k(x)).reshape(B, h, d, kk, L).permute(0, 1, 4, 2, 3)     # [B, h, L, d, kk]
        v = self.unfold(self.v(x)).reshape(B, h, d, kk, L).permute(0, 1, 4, 3, 2)     # [B, h, L, kk, d]
        attn = self.attn_drop(((q @ k) * self.scale).softmax(dim=-1))                 # [B, h, L, 1, kk]
        out = (attn @ v).squeeze(3)                                                   # [B, h, L, d]
        return out.permute(0, 1, 3, 2).reshape(B, C, H, W)


class WeaverAttention(nn.Module):
    """Weaves a global and a local path on downsampled tokens.

    Downsampling (rate S on the token count) -> channel split ->
    [self-attention on ``global_ratio`` of the channels | ConvAttention on the rest]
    -> concat -> upsampling. Complexity ~ O((N/S) C + (N/S)^2 C/4).

    ``global_ratio=0`` drops the self-attention path, ``global_ratio=1`` drops
    ConvAttention and ``downsample_rate=1`` disables downsampling (the Weaformer
    ablations in the paper).

    Args:
        dim: channel dimension C.
        num_heads: total heads, shared between the two paths in proportion to their channels.
        downsample_rate: S, the reduction of the token count (S = 4 halves each side).
        global_ratio: fraction of channels routed to the global path (0.25 in the paper).
        kernel_size: local window k of ConvAttention.
    """

    def __init__(self, dim, num_heads=8, downsample_rate=4, global_ratio=0.25, kernel_size=3,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        side = math.isqrt(downsample_rate)
        assert side * side == downsample_rate, "downsample_rate must be a perfect square (1, 4, 16, ...)"
        assert 0.0 <= global_ratio <= 1.0
        self.side = side
        head_dim = dim // num_heads
        self.global_dim = int(round(dim * global_ratio / head_dim)) * head_dim
        self.local_dim = dim - self.global_dim
        self.global_attn = None
        self.local_attn = None
        if self.global_dim > 0:
            self.global_attn = MultiHeadAttention(self.global_dim, self.global_dim // head_dim,
                                                  attn_drop=attn_drop, proj_drop=proj_drop)
        if self.local_dim > 0:
            self.local_attn = ConvAttention(self.local_dim, self.local_dim // head_dim, kernel_size, attn_drop)

    def forward(self, x, H, W):
        """x: ``[B, H*W, C]`` -> ``[B, H*W, C]``."""
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        Hs, Ws = max(1, H // self.side), max(1, W // self.side)
        if (Hs, Ws) != (H, W):
            x = F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)
        x_global, x_local = x.split([self.global_dim, self.local_dim], dim=1)

        outs = []
        if self.global_attn is not None:
            tokens = x_global.flatten(2).transpose(1, 2)
            pos = sine_position_embedding_2d(Hs, Ws, self.global_dim, device=x.device).to(tokens.dtype)
            out = self.global_attn(tokens, tokens, tokens, query_pos=pos, key_pos=pos)
            outs.append(out.transpose(1, 2).reshape(B, self.global_dim, Hs, Ws))
        if self.local_attn is not None:
            outs.append(self.local_attn(x_local))
        x = torch.cat(outs, dim=1)

        if (Hs, Ws) != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.flatten(2).transpose(1, 2)


class Weaformer(nn.Module):
    """A Transformer layer whose self-attention is replaced by Weaver-Attention (Fig. 3c)."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, downsample_rate=4, global_ratio=0.25,
                 kernel_size=3, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WeaverAttention(dim, num_heads, downsample_rate, global_ratio, kernel_size,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.ffn(self.norm2(x)))


class FeatureEnhancer(nn.Module):
    """Alternates Weaformer and prototype cross-attention layers (Fig. 3b).

    With ``depth=2`` (the paper setting):
        F1 = Weaformer(E_v);  F2 = CrossAttn(F1, P_VPT, P_VPT)
        F3 = Weaformer(F2);   F_en = CrossAttn(F3, P_VPT, P_VPT)
    """

    def __init__(self, dim=512, depth=2, num_heads=8, mlp_ratio=4.0, downsample_rate=4, global_ratio=0.25,
                 kernel_size=3, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.weaformers = nn.ModuleList([
            Weaformer(dim, num_heads, mlp_ratio, downsample_rate, global_ratio, kernel_size,
                      drop=drop, attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, prototypes, H, W):
        """
        Args:
            x: E_v, ``[B, H*W, D]``.
            prototypes: P_VPT, ``[B, M, D]``.

        Returns:
            F_en, ``[B, H*W, D]``.
        """
        for weaformer, cross in zip(self.weaformers, self.cross_layers):
            x = cross(weaformer(x, H, W), prototypes)
        return self.norm(x)
