"""YOLOE image backbone + PAN, the auxiliary text network, and the frozen text encoder."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_yoloe(cfg="yoloe-11s-seg.yaml", weights=None):
    """Builds a YOLOE segmentation model and optionally loads its pretrained weights."""
    try:
        from ultralytics.nn.tasks import YOLOESegModel
    except ImportError as e:
        raise ImportError(
            "RT-Counter is built on the YOLOE fork of ultralytics. Install it with "
            "`pip install git+https://github.com/THU-MIG/yoloe.git`."
        ) from e
    model = YOLOESegModel(cfg, verbose=False)
    if weights:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        src = (ckpt.get("ema") or ckpt["model"]) if isinstance(ckpt, dict) else ckpt
        state = src.float().state_dict() if isinstance(src, nn.Module) else src
        model.load_state_dict(state, strict=True)
    return model


class YOLOEBackbone(nn.Module):
    """Image backbone and PAN of YOLOE (outputs F_p1..F_p3 at strides 8/16/32).

    The detection/segmentation branches of the YOLOE head are dropped; only its
    text re-parameterisation network (``reprta``, a residual SwiGLU FFN) is kept as
    the auxiliary network that maps E_t' to E_t.
    """

    def __init__(self, cfg="yoloe-11s-seg.yaml", weights=None):
        super().__init__()
        yoloe = build_yoloe(cfg, weights)
        head = yoloe.model[-1]
        self.layers = yoloe.model[:-1]
        self.save = set(yoloe.save)
        self.out_indices = list(head.f)
        self.channels = [branch[0].conv.in_channels for branch in head.cv2]
        self.auxiliary = head.reprta
        self.embed_dim = self.auxiliary.m.w3.out_features

    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return [y[i] for i in self.out_indices]

    def encode_text(self, text_embeddings):
        """Auxiliary network: E_t = E_t' + SwiGLU(FFN(E_t')), L2-normalised as in YOLOE."""
        return F.normalize(self.auxiliary(text_embeddings), dim=-1, p=2)


class TextEncoder:
    """Frozen MobileCLIP text encoder with a per-prompt embedding cache.

    It is deliberately not an ``nn.Module`` so that its weights never enter
    RT-Counter checkpoints or parameter counts. As in the paper, embeddings of
    known categories can be pre-computed once (``save_cache``) and loaded at
    inference time (``load_cache``) to skip text encoding entirely.
    """

    SIZES = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, variant="mobileclip:blt", weights="mobileclip_blt.pt"):
        base, size = variant.split(":")
        if base != "mobileclip":
            raise ValueError(f"Unsupported text encoder {variant!r}; RT-Counter uses MobileCLIP.")
        self.arch = f"mobileclip_{self.SIZES[size]}"
        self.weights = weights
        self.model = None
        self.tokenizer = None
        self.cache = {}

    def _build(self, device):
        import mobileclip

        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"MobileCLIP weights not found at {self.weights}. Download them from "
                "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
            )
        model = mobileclip.create_model_and_transforms(self.arch, pretrained=self.weights, device=device)[0]
        del model.image_encoder
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = mobileclip.get_tokenizer(self.arch)

    @torch.no_grad()
    def __call__(self, prompts, device):
        """Returns E_t' of shape ``[B, 1, D]`` for a list of B prompts."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if missing:
            if self.model is None:
                self._build(device)
            model_device = next(self.model.parameters()).device
            feats = self.model.encode_text(self.tokenizer(missing).to(model_device)).float()
            feats = F.normalize(feats, dim=-1, p=2).cpu()
            self.cache.update(zip(missing, feats))
        return torch.stack([self.cache[p] for p in prompts]).to(device)[:, None]

    def save_cache(self, path):
        torch.save(self.cache, path)

    def load_cache(self, path):
        self.cache.update(torch.load(path, map_location="cpu"))
