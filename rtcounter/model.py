"""RT-Counter: Real-Time Text-Guided Open-Vocabulary Object Counting (Fig. 2)."""
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from .backbone import TextEncoder, YOLOEBackbone
from .embedding import VisualEmbedding
from .fim import HistoricalFIM
from .fim_reference import ReferenceFIM
from .fim_v2 import V2FIM
from .head import PointHead
from .vpt import VPT
from .vpt_reference import ReferenceVPT
from .vpt_v2 import V2VPT
from .weaformer import FeatureEnhancer


@dataclass
class RTCounterConfig:
    # YOLOE image backbone + PAN (initialised from the YOLOE-11s weights).
    yoloe_cfg: str = "yoloe-11s-seg.yaml"
    yoloe_weights: Optional[str] = None
    freeze_backbone: bool = False
    # Frozen MobileCLIP-B(LT) text encoder.
    text_model: str = "mobileclip:blt"
    text_weights: str = "mobileclip_blt.pt"
    # Embedding module.
    dim: int = 512
    num_experts: int = 4
    # Visual prototype-to-text context.
    use_vpt: bool = True
    vpt_type: str = "reference"       # User-supplied reference prototype extraction; "v2" and "legacy" remain explicit.
    num_prototypes: int = 16          # N_p
    # Feature enhancer.  The user-supplied reference FIM is the default source-selected path.
    enhancer_type: str = "reference_fim"
    enhancer_depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    fim_downsample_ratio: int = 2     # each spatial side; historical_fim only
    # The following fields apply only when enhancer_type == "weaformer".
    downsample_rate: int = 4          # Weaformer token-count reduction S
    global_ratio: float = 0.25        # Weaformer global-channel fraction
    local_kernel: int = 3             # Weaformer ConvAttention kernel size
    drop: float = 0.1
    attn_drop: float = 0.1
    drop_path: float = 0.1
    # Prediction.
    threshold: float = 0.5            # phi


class RTCounter(nn.Module):
    """Image backbone -> PAN -> embedding -> (VPT) -> feature enhancer -> point head."""

    def __init__(self, cfg: RTCounterConfig = None, **kwargs):
        super().__init__()
        cfg = cfg or RTCounterConfig(**kwargs)
        self.cfg = cfg
        self.backbone = YOLOEBackbone(cfg.yoloe_cfg, cfg.yoloe_weights)
        assert self.backbone.embed_dim == cfg.dim, "dim must match the YOLOE text embedding size (512)"
        self.text_encoder = TextEncoder(cfg.text_model, cfg.text_weights)
        self.embedding = VisualEmbedding(self.backbone.channels, cfg.dim, cfg.num_experts)
        if cfg.vpt_type not in {"reference", "v2", "legacy"}:
            raise ValueError(
                f"Unsupported vpt_type {cfg.vpt_type!r}; expected 'reference', 'v2', or 'legacy'"
            )
        if cfg.use_vpt:
            if cfg.vpt_type == "reference":
                self.vpt = ReferenceVPT(
                    cfg.dim, cfg.num_prototypes, cfg.num_heads, cfg.mlp_ratio, cfg.drop, cfg.attn_drop
                )
            elif cfg.vpt_type == "v2":
                self.vpt = V2VPT(cfg.dim, cfg.num_prototypes, cfg.num_heads)
            else:
                self.vpt = VPT(cfg.dim, cfg.num_prototypes, cfg.num_heads, cfg.mlp_ratio,
                               cfg.drop, cfg.attn_drop)
        else:
            self.vpt = None

        if cfg.enhancer_type == "reference_fim":
            if cfg.use_vpt and cfg.vpt_type != "reference":
                raise ValueError(
                    "enhancer_type='reference_fim' requires vpt_type='reference' when use_vpt=True because "
                    "the reference FIM consumes ReferenceVPT's text-plus-prototype context [B, 1+N_p, D]"
                )
            self.enhancer = ReferenceFIM(
                cfg.dim,
                cfg.enhancer_depth,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.drop,
                cfg.attn_drop,
                cfg.drop_path,
            )
        elif cfg.enhancer_type == "v2_fim":
            if cfg.use_vpt and cfg.vpt_type != "v2":
                raise ValueError(
                    "enhancer_type='v2_fim' requires vpt_type='v2' when use_vpt=True because V2 FIM "
                    "accepts V2VPT's single enhanced text token [B, 1, D]"
                )
            self.enhancer = V2FIM(
                cfg.dim,
                cfg.enhancer_depth,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.drop,
                cfg.attn_drop,
                cfg.drop_path,
            )
        elif cfg.enhancer_type == "historical_fim":
            self.enhancer = HistoricalFIM(
                cfg.dim,
                cfg.enhancer_depth,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.drop,
                cfg.attn_drop,
                cfg.drop_path,
                cfg.fim_downsample_ratio,
            )
        elif cfg.enhancer_type == "weaformer":
            self.enhancer = FeatureEnhancer(
                cfg.dim,
                cfg.enhancer_depth,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.downsample_rate,
                cfg.global_ratio,
                cfg.local_kernel,
                cfg.drop,
                cfg.attn_drop,
                cfg.drop_path,
            )
        else:
            raise ValueError(
                f"Unsupported enhancer_type {cfg.enhancer_type!r}; expected 'reference_fim', 'v2_fim', "
                "'historical_fim', or 'weaformer'"
            )
        self.head = PointHead(cfg.dim)
        if cfg.freeze_backbone:
            self.backbone.layers.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_backbone:
            # Frozen BatchNorm statistics must not drift.
            self.backbone.layers.eval()
        return self

    def forward(self, images, prompts=None, text_embeddings=None):
        """
        Args:
            images: ``[B, 3, H, W]`` in [0, 1]; H and W must be multiples of 32.
            prompts: list of B text prompts, or
            text_embeddings: pre-computed MobileCLIP embeddings E_t', ``[B, D]`` or ``[B, 1, D]``.

        Returns:
            dict with ``pred_points`` ``[B, N, 2]`` (pixels), ``pred_logits`` ``[B, N, 2]``
            (background, object) and the embedding gate weights.
        """
        if text_embeddings is None:
            text_embeddings = self.text_encoder(prompts, images.device)
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings[:, None]
        text = self.backbone.encode_text(text_embeddings)                     # E_t  [B, 1, D]

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_backbone):
            feats = self.backbone(images)                                     # F_p
        visual_map, gate = self.embedding(feats)                              # E_v  [B, D, H/16, W/16]
        B, D, H, W = visual_map.shape
        visual = visual_map.flatten(2).transpose(1, 2)

        context = self.vpt(visual, text) if self.vpt is not None else text
        enhanced = self.enhancer(visual, context, H, W)                       # F_en
        enhanced = enhanced.transpose(1, 2).reshape(B, D, H, W)
        points, logits = self.head(enhanced, images.shape[-2:])
        return {"pred_points": points, "pred_logits": logits, "gate_weights": gate}

    @torch.no_grad()
    def count(self, images, prompts=None, text_embeddings=None, threshold=None):
        """N_Pred = sum_i 1(c_i > phi). Returns counts ``[B]`` and the kept points per image."""
        threshold = self.cfg.threshold if threshold is None else threshold
        out = self(images, prompts, text_embeddings)
        scores = out["pred_logits"].softmax(-1)[..., 1]
        keep = scores > threshold
        return keep.sum(dim=1), [p[k] for p, k in zip(out["pred_points"], keep)]

    def config_dict(self):
        return asdict(self.cfg)
