"""Visual Prototype Textualization (VPT), Sec. 3.2 and Fig. 3(a) of the paper."""
import torch
import torch.nn as nn

from .layers import FFN, MultiHeadAttention, sine_position_embedding_1d


class VPT(nn.Module):
    """Projects learned visual prototypes into the text space and fuses them with the text.

    Stages (paper notation):
        P_dot  = FFN(Norm(CrossAttn(P_ddot, E_v, E_v)))   # extraction + projection to text space
        P_v    = CrossAttn(P_dot, E_t, E_t)               # filtering / enhancement by the text
        P_VPT  = SelfAttn(Concat(P_v, E_t))               # prototype fusion

    The text cross-attention keeps its residual path (it is a Transformer layer in
    Fig. 3a): with a single prompt token the softmax over keys is identically 1, so
    without the residual every prototype would collapse to the same vector.

    Args:
        dim: feature dimension D.
        num_prototypes: number of prototype queries N_p (16 in the paper).
    """

    def __init__(self, dim=512, num_prototypes=16, num_heads=8, mlp_ratio=4.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_queries = nn.Parameter(torch.zeros(1, num_prototypes, dim))
        nn.init.trunc_normal_(self.prototype_queries, std=0.02)
        self.register_buffer("query_pos", sine_position_embedding_1d(num_prototypes, dim), persistent=False)

        # Visual prototype extraction and projection.
        self.extract_attn = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.extract_norm = nn.LayerNorm(dim)
        self.project_ffn = FFN(dim, int(dim * mlp_ratio), drop=drop)

        # Visual prototype filtering and enhancement.
        self.text_attn = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.text_norm = nn.LayerNorm(dim)

        # Prototype fusion.
        self.fusion_attn = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.fusion_norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens, text_tokens):
        """
        Args:
            visual_tokens: E_v, ``[B, HW, D]``.
            text_tokens: E_t, ``[B, K, D]`` (K = 1 for a single prompt).

        Returns:
            P_VPT, ``[B, N_p + K, D]``.
        """
        queries = self.prototype_queries.expand(visual_tokens.shape[0], -1, -1)
        prototypes = self.extract_attn(queries, visual_tokens, visual_tokens, query_pos=self.query_pos)
        prototypes = self.project_ffn(self.extract_norm(prototypes))
        prototypes = self.text_norm(prototypes + self.text_attn(prototypes, text_tokens, text_tokens))
        tokens = torch.cat([prototypes, text_tokens], dim=1)
        return self.fusion_norm(self.fusion_attn(tokens, tokens, tokens))
