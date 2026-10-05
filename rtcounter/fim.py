"""Historical Feature Interaction Module (FIM).

This module preserves the FIM used by the archived RT-Counter research
implementation.  Its visual branch downsamples each spatial side by two,
routes half of the channels through global attention and half through
multi-scale dilated local attention, then upsamples before the residual
connection.  It is intentionally separate from the paper-facing Weaformer
implementation in :mod:`rtcounter.weaformer`.
"""
import collections.abc
import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """The RT-DETR-style 2D sine position encoding used by the historical FIM."""

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
    """Stochastic depth, matching the archived implementation."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths per sample on a residual branch."""

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
    """The two-layer MLP used by the archived FIM block."""

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
        return self.drop2(x)


class Attention(nn.Module):
    """Historical multi-head self-attention with position added only to Q and K."""

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
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class DilateAttention(nn.Module):
    """Historical local attention for one dilation rate."""

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            stride=1,
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape(B, d // self.head_dim, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(
            B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 2, 3)
        attn = self.attn_drop(((q @ k) * self.scale).softmax(dim=-1))
        v = self.unfold(v).reshape(
            B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 3, 2)
        return (attn @ v).transpose(1, 2).reshape(B, H, W, d)


class MultiDilatelocalAttention(nn.Module):
    """Historical multi-scale dilated local-attention path."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        kernel_size=3,
        dilation=(1, 3),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = list(dilation)
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, (
            f"num_heads {num_heads} must be divisible by num_dilation {self.num_dilation}!"
        )

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x_chw = x.permute(0, 3, 1, 2)
        qkv = self.qkv(x_chw).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(
            2, 1, 0, 3, 4, 5
        )
        # The archived source assigned each result into a view of ``x_split``.
        # Constructing the same [dilation, B, H, W, C_per_dilation] tensor out
        # of place preserves its forward values and avoids an autograd version
        # counter error during training.
        x_split = torch.stack([
            self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            for i in range(self.num_dilation)
        ], dim=0)
        x = x_split.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    """Historical multi-head cross-attention from visual tokens to context tokens."""

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
        attn = self.attn_drop(torch.clamp(attn, min=-65504.0, max=65504.0).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        return self.proj_drop(self.proj(x))


class CrossAttentionBlock(nn.Module):
    """Historical FIM block with its 1:1 global/local visual interaction path."""

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
        downsample_ratio=2,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.dim = dim

        self.norm_x = norm_layer(dim)
        self.selfattn_x = Attention(
            dim // 2,
            num_heads=num_heads // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.multi_dilate_attn = MultiDilatelocalAttention(
            dim // 2,
            num_heads=num_heads // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            kernel_size=3,
            dilation=(1, 3),
        )
        self.drop_path_x = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_y = norm_layer(dim)
        self.selfattn_y = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path_y = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path_cross = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_ffn = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, pos_embed=None, H=None, W=None):
        if self.downsample_ratio < 2:
            raise ValueError("historical FIM requires downsample_ratio >= 2; it has no full-token fallback")
        if H is None or W is None:
            raise ValueError("historical FIM requires H and W for its spatial downsample/upsample path")
        if H < self.downsample_ratio or W < self.downsample_ratio:
            raise ValueError(
                f"H and W must both be at least downsample_ratio={self.downsample_ratio}; got H={H}, W={W}"
            )

        x_residual = x
        B, _, C = x.shape
        ratio = self.downsample_ratio
        x_norm = self.norm_x(x)
        x_spatial = x_norm.transpose(1, 2).reshape(B, C, H, W)
        H_down, W_down = H // ratio, W // ratio
        x_down_spatial = F.interpolate(x_spatial, size=(H_down, W_down), mode="bilinear", align_corners=False)
        x_down = x_down_spatial.flatten(2).transpose(1, 2)

        pos_embed_down = None
        if pos_embed is not None:
            pos_spatial = pos_embed.transpose(1, 2).reshape(B, C, H, W)
            pos_down_spatial = F.interpolate(
                pos_spatial, size=(H_down, W_down), mode="bilinear", align_corners=False
            )
            pos_embed_down = pos_down_spatial.flatten(2).transpose(1, 2)

        C_half = C // 2
        x_down_part1 = x_down[:, :, :C_half]
        x_down_part2 = x_down[:, :, C_half:]
        if pos_embed_down is not None:
            pos_embed_down_part1 = pos_embed_down[:, :, :C_half]
        else:
            pos_embed_down_part1 = None

        x_attn_part1 = self.selfattn_x(x_down_part1, pos_embed=pos_embed_down_part1)
        x_down_part2_hwc = x_down_part2.transpose(1, 2).reshape(B, C_half, H_down, W_down).permute(0, 2, 3, 1)
        x_attn_part2_hwc = self.multi_dilate_attn(x_down_part2_hwc)
        x_attn_part2 = x_attn_part2_hwc.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        x_attn = torch.cat([x_attn_part1, x_attn_part2], dim=2)
        x_attn_spatial = x_attn.transpose(1, 2).reshape(B, C, H_down, W_down)
        x_attn_upsampled = F.interpolate(x_attn_spatial, size=(H, W), mode="bilinear", align_corners=False)
        x_attn = x_attn_upsampled.flatten(2).transpose(1, 2)
        x = x_residual + self.drop_path_x(x_attn)

        y = y + self.drop_path_y(self.selfattn_y(self.norm_y(y), pos_embed=None))
        x = x + self.drop_path_cross(self.cross_attn(self.norm_cross(x), y))
        x = x + self.drop_path_ffn(self.mlp(self.norm_ffn(x)))
        return x, y


class HistoricalFIM(nn.Module):
    """Stacked historical FIM blocks with the original 2D position encoding."""

    def __init__(
        self,
        dim=512,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        downsample_ratio=2,
    ):
        super().__init__()
        if not isinstance(downsample_ratio, int) or downsample_ratio < 2:
            raise ValueError(
                "historical FIM requires an integer fim_downsample_ratio >= 2; "
                "it always uses the archived downsampled spatial path"
            )
        self.blocks = nn.ModuleList([
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
                downsample_ratio=downsample_ratio,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=dim // 2, temperature=10000)

    def forward(self, x, context, H, W):
        """Enhance visual ``x`` with text/prototype ``context`` and return visual tokens."""
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"x has {N} tokens but H*W is {H * W}")
        if C != self.norm.normalized_shape[0]:
            raise ValueError(f"x has dim {C}; expected {self.norm.normalized_shape[0]}")
        mask = torch.ones(B, H, W, dtype=torch.float32, device=x.device)
        pos_embed = self.position_embedding(mask).flatten(1, 2)
        y = context
        for block in self.blocks:
            x, y = block(x, y, pos_embed=pos_embed, H=H, W=W)
        return self.norm(x)
