"""Prototype context path from the user-supplied ``YOLOECountModel`` reference.

The reference extracts 16 visual prototypes, feeds each prototype from the
text token through a residual cross-attention and MLP, then places the text
token before the prototypes for FIM.  The reference also contains a separate
context self-attention after that concatenation.  This release intentionally
omits that extra layer under the user's no-extra-SelfAttn direction; the
Reference FIM still retains its source x/y self-attention internally.
"""
import torch
import torch.nn as nn

from .fim_reference import CrossAttention, Mlp


class ReferenceVPT(nn.Module):
    """Produce the reference text-plus-prototype context for ReferenceFIM.

    Args:
        visual_tokens: visual embedding tokens ``[B, N, D]``.
        text_tokens: one encoded text token ``[B, 1, D]``.

    Returns:
        ``[B, 1 + N_p, D]`` in source order: text first, then visual
        prototypes.  With the default 16 queries this is ``[B, 17, D]``.
    """

    def __init__(
        self,
        dim=512,
        num_prototypes=16,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_queries = nn.Parameter(torch.zeros(1, num_prototypes, dim))
        nn.init.trunc_normal_(self.prototype_queries, std=0.02)

        self.prototype_extraction = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.prototype_text_interaction = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.prototype_norm1 = nn.LayerNorm(dim)
        self.prototype_norm2 = nn.LayerNorm(dim)
        self.prototype_mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, visual_tokens, text_tokens):
        """Return source-order text-plus-prototype context without extra self-attention."""
        prototype_queries = self.prototype_queries.expand(visual_tokens.shape[0], -1, -1)
        visual_prototypes = self.prototype_extraction(prototype_queries, visual_tokens)
        visual_prototypes = self.prototype_norm1(visual_prototypes)

        prototypes_text_enhanced = self.prototype_text_interaction(visual_prototypes, text_tokens)
        visual_prototypes = self.prototype_norm2(visual_prototypes + prototypes_text_enhanced)
        visual_prototypes = visual_prototypes + self.prototype_mlp(visual_prototypes)
        return torch.cat([text_tokens, visual_prototypes], dim=1)
