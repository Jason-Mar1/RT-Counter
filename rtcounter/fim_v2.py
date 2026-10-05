"""FIM ported from ``models/yoloe_count_v2.py``.

The V2 FIM is deliberately distinct from the later spatial historical FIM:
it applies full-token visual self-attention, then visual-to-single-text
cross-attention, then an MLP.  It has a learned absolute 48 x 48 position
table that is added to visual tokens once before the two FIM blocks.
"""
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth used by the V2 residual branches."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """V2's two-layer MLP helper."""

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
    """V2 full-token visual self-attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """V2 visual-query cross-attention over the enhanced text token."""

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
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """The exact three-residual V2 FIM block."""

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
        self.norm0 = norm_layer(dim)
        self.selfattn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path0 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y):
        x = x + self.drop_path0(self.selfattn(self.norm0(x)))
        x = x + self.drop_path1(self.attn(self.norm1(x), y))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class V2FIM(nn.Module):
    """Two V2 blocks and their learned 48 x 48 absolute position table."""

    max_h = 48
    max_w = 48

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
        self.fim_pos_embed = nn.Parameter(torch.zeros(1, self.max_h * self.max_w, dim))
        torch.nn.init.trunc_normal_(self.fim_pos_embed, std=0.02)

    def forward(self, visual, context, H, W):
        """Return V2-enhanced visual tokens.

        ``context`` must be the one enhanced text token produced by ``V2VPT``.
        V2 has no interpolation for its learned position table, so inputs with
        more than 2,304 visual tokens are rejected instead of silently changing
        the calculation.
        """
        B, N, C = visual.shape
        if N != H * W:
            raise ValueError(f"visual has {N} tokens but H*W is {H * W}")
        if C != self.fim_pos_embed.shape[-1]:
            raise ValueError(f"visual has dim {C}; expected {self.fim_pos_embed.shape[-1]}")
        if N > self.fim_pos_embed.shape[1]:
            raise ValueError(
                f"V2 FIM supports at most {self.fim_pos_embed.shape[1]} visual tokens (48x48); got {N}"
            )
        if context.ndim != 3 or context.shape[0] != B or context.shape[1] != 1 or context.shape[2] != C:
            raise ValueError(
                f"V2 FIM requires context [B, 1, D] matching visual; got {tuple(context.shape)}"
            )

        x = visual + self.fim_pos_embed[:, :H * W, :]
        for block in self.fim_blocks:
            x = block(x, context)
        return self.fim_norm(x)
