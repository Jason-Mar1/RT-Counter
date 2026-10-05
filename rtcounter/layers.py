"""Transformer building blocks shared by VPT, Weaformer and the feature enhancer."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth applied to the residual branch."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class FFN(nn.Module):
    """Two-layer feed-forward network (Linear-GELU-Linear)."""

    def __init__(self, dim, hidden_dim=None, out_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MultiHeadAttention(nn.Module):
    """Multi-head attention; Q comes from ``query`` and K/V from ``key``/``value``.

    Positional embeddings, when given, are added to the inputs of the Q and K
    projections only, so V keeps pure content (DETR convention).
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, query_pos=None, key_pos=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        h = self.num_heads
        q = self.q(query if query_pos is None else query + query_pos)
        k = self.k(key if key_pos is None else key + key_pos)
        v = self.v(value)
        q = q.reshape(B, Nq, h, C // h).transpose(1, 2)
        k = k.reshape(B, Nk, h, C // h).transpose(1, 2)
        v = v.reshape(B, Nk, h, C // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x))


class CrossAttentionLayer(nn.Module):
    """Pre-norm Transformer layer whose attention reads from an external memory.

    x = x + CrossAttn(LN(x), memory, memory)
    x = x + FFN(LN(x))
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, memory):
        x = x + self.drop_path(self.attn(self.norm1(x), memory, memory))
        return x + self.drop_path(self.ffn(self.norm2(x)))


def sine_position_embedding_2d(h, w, dim, device=None, temperature=10000.0):
    """2D sine/cosine embedding for an ``h x w`` grid, returned as ``[1, h*w, dim]``."""
    assert dim % 4 == 0, "2D sine embedding needs dim divisible by 4"
    num_feats = dim // 2
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device) / h * 2 * math.pi
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device) / w * 2 * math.pi
    y, x = torch.meshgrid(y, x, indexing="ij")
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos_x = x[..., None] / dim_t
    pos_y = y[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1).reshape(1, h * w, dim)


def sine_position_embedding_1d(n, dim, temperature=10000.0):
    """1D sine/cosine embedding for ``n`` token indices, returned as ``[1, n, dim]``."""
    assert dim % 2 == 0, "1D sine embedding needs an even dim"
    position = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(temperature) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe[None]
