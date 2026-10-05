"""Feature Interaction Module ported from the user-supplied reference code.

This module deliberately preserves the reference FIM's full-token visual and
context self-attention path.  It is separate from the legacy spatial FIM and
the earlier V2 single-context-token FIM retained elsewhere in this package.
"""
import collections.abc
import math
from itertools import repeat

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Reference RT-DETR-style two-dimensional sine position encoding."""

    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (stochastic depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP used by the reference FIM."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Reference multi-head self-attention, with optional Q/K position encoding."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pos_embed=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos_embed is not None:
            pos_embed_heads = pos_embed.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q + pos_embed_heads
            k = k + pos_embed_heads

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-65504.0, max=65504.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Reference multi-head cross-attention, with visual queries over context."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, Nx, C = x.shape
        Ny = y.shape[1]

        q = self.wq(x).reshape(B, Nx, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-65504.0, max=65504.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Reference FIM block: x self-attention, y self-attention, x-to-y attention, FFN."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_x = norm_layer(dim)
        self.selfattn_x = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path_x = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_y = norm_layer(dim)
        self.selfattn_y = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path_y = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path_cross = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_ffn = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, pos_embed=None):
        x = x + self.drop_path_x(self.selfattn_x(self.norm_x(x), pos_embed=pos_embed))
        y = y + self.drop_path_y(self.selfattn_y(self.norm_y(y), pos_embed=None))
        x = x + self.drop_path_cross(self.cross_attn(self.norm_cross(x), y))
        x = x + self.drop_path_ffn(self.mlp(self.norm_ffn(x)))
        return x, y


class ReferenceFIM(nn.Module):
    """Two reference FIM blocks with dynamic two-dimensional sine positions."""

    def __init__(
        self,
        dim=512,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
    ):
        super().__init__()
        self.fim_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(depth)
        ])
        self.fim_norm = nn.LayerNorm(dim)
        self.fim_pos_encoding = PositionEmbeddingSine(num_pos_feats=dim // 2, temperature=10000)

    def forward(self, visual, context, H, W):
        B, _, _ = visual.shape
        pos_mask = torch.ones(B, H, W, dtype=torch.float32, device=visual.device)
        pos_embed = self.fim_pos_encoding(pos_mask).flatten(1, 2)

        x = visual
        y = context
        for block in self.fim_blocks:
            x, y = block(x, y, pos_embed=pos_embed)
        return self.fim_norm(x)
