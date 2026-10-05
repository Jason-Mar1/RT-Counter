"""V2 visual-prototype extraction and gated prototype-text fusion.

This is a direct module-level port of ``models/yoloe_count_v2.py``.  Unlike
the legacy :mod:`rtcounter.vpt` path, it does not concatenate prototype and
text tokens for a final self-attention layer.  It returns one enhanced text
token for the V2 FIM path.
"""
import torch
import torch.nn as nn


class VisualPrototypeExtractor(nn.Module):
    """Extract visual prototypes with learned queries and cross-attention."""

    def __init__(self, visual_dim=512, num_prototypes=16, num_heads=8):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.visual_dim = visual_dim

        self.prototype_queries = nn.Parameter(torch.randn(num_prototypes, visual_dim))
        nn.init.xavier_uniform_(self.prototype_queries)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, visual_feat):
        """Return ``[B, N_p, C]`` prototypes and their attention weights."""
        B, C, H, W = visual_feat.shape
        visual_tokens = visual_feat.flatten(2).transpose(1, 2)
        queries = self.prototype_queries.unsqueeze(0).expand(B, -1, -1)
        prototypes, attn_weights = self.cross_attention(queries, visual_tokens, visual_tokens)
        prototypes = self.norm(prototypes)
        prototypes = self.dropout(prototypes)
        return prototypes, attn_weights


class PrototypeFusion(nn.Module):
    """Fuse visual prototypes into one text token with V2's learned gate."""

    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, prototypes, text_features):
        """Return one gated, residual-normalised text token and attention weights."""
        enhanced_text, attn_weights = self.cross_attention(text_features, prototypes, prototypes)
        gate_input = torch.cat([text_features, enhanced_text], dim=-1)
        gate_weights = self.gate(gate_input)
        fused_text = gate_weights * enhanced_text + (1 - gate_weights) * text_features
        fused_text = self.layer_norm(fused_text + text_features)
        fused_text = self.dropout(fused_text)
        return fused_text, attn_weights


class V2VPT(nn.Module):
    """V2 wrapper from visual tokens and one text token to enhanced text.

    The source extractor accepts a ``[B, C, H, W]`` map.  The public release
    forwards ``[B, N, D]`` tokens, so this wrapper reconstructs a lossless
    ``[B, D, N, 1]`` view before calling the unchanged extractor.
    """

    def __init__(self, dim=512, num_prototypes=16, num_heads=8):
        super().__init__()
        self.visual_prototype_extractor = VisualPrototypeExtractor(
            visual_dim=dim,
            num_prototypes=num_prototypes,
            num_heads=num_heads,
        )
        self.prototype_fusion = PrototypeFusion(embed_dim=dim, num_heads=num_heads)

    def forward(self, visual_tokens, text_tokens):
        """Return V2's single enhanced text token, ``[B, 1, D]``."""
        visual_feat = visual_tokens.transpose(1, 2).unsqueeze(-1)
        prototypes, _ = self.visual_prototype_extractor(visual_feat)
        enhanced_text, _ = self.prototype_fusion(prototypes, text_tokens)
        return enhanced_text
