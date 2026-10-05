"""Evaluation, checkpointing and optimiser helpers shared by the tools."""
import math
from dataclasses import fields

import torch

from .model import RTCounter, RTCounterConfig


@torch.no_grad()
def evaluate(model, loader, device, threshold=None, amp=True):
    """Counts every image in ``loader`` and returns (MAE, RMSE, per-image records)."""
    model.eval()
    records = []
    for images, points, prompts, ids in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            counts, _ = model.count(images.to(device, non_blocking=True), prompts, threshold=threshold)
        for im_id, prompt, count, gt in zip(ids, prompts, counts.tolist(), points):
            records.append({"id": im_id, "prompt": prompt, "pred": int(count), "gt": len(gt)})
    errors = [r["pred"] - r["gt"] for r in records]
    mae = sum(abs(e) for e in errors) / max(len(errors), 1)
    rmse = math.sqrt(sum(e * e for e in errors) / max(len(errors), 1))
    return mae, rmse, records


def param_groups(model, lr, backbone_lr):
    """The YOLOE image backbone + PAN use ``backbone_lr``; every other trainable module uses ``lr``."""
    backbone = [p for p in model.backbone.layers.parameters() if p.requires_grad]
    ids = {id(p) for p in backbone}
    others = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups = [{"params": others, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


def save_checkpoint(path, model, **extra):
    torch.save({"config": model.config_dict(), "model": model.state_dict(), **extra}, path)


def _has_state_prefix(state_dict, prefix):
    return isinstance(state_dict, dict) and any(key.startswith(prefix) for key in state_dict)


def _has_state_component(state_dict, prefix, component):
    """Whether a state key has a backend prefix and an architecture-specific component."""
    return isinstance(state_dict, dict) and any(
        key.startswith(prefix) and component in key for key in state_dict
    )


def _infer_enhancer_type(checkpoint_config, state_dict, overrides):
    """Infer only known pre-selector enhancer state layouts."""
    if "enhancer_type" in checkpoint_config or "enhancer_type" in overrides:
        return None
    # Reference FIM and V2 FIM both use ``fim_blocks``.  Check their unique
    # block member names before falling through to older independent layouts.
    reference_components = (".norm_x.", ".selfattn_x.", ".norm_y.", ".selfattn_y.", ".norm_cross.")
    if any(_has_state_component(state_dict, "enhancer.fim_blocks.", component) for component in reference_components):
        return "reference_fim"
    v2_components = (".norm0.", ".selfattn.", ".norm1.", ".attn.", ".norm2.")
    if _has_state_prefix(state_dict, "enhancer.fim_pos_embed") or any(
        _has_state_component(state_dict, "enhancer.fim_blocks.", component) for component in v2_components
    ):
        return "v2_fim"
    if _has_state_prefix(state_dict, "enhancer.blocks."):
        return "historical_fim"
    if _has_state_prefix(state_dict, "enhancer.weaformers."):
        return "weaformer"
    raise ValueError(
        "Checkpoint config has no enhancer_type and its state_dict is not a recognised reference FIM, "
        "V2 FIM, historical FIM, or legacy Weaformer layout. Refusing to guess enhancer compatibility; "
        "pass an explicit enhancer_type override only after verifying the checkpoint architecture."
    )


def _infer_vpt_type(checkpoint_config, state_dict, overrides):
    """Infer VPT type only when state keys identify one of the supported layouts."""
    if "vpt_type" in checkpoint_config or "vpt_type" in overrides:
        return None
    if not overrides.get("use_vpt", checkpoint_config.get("use_vpt", True)):
        return None
    # Both ReferenceVPT and legacy VPT own ``prototype_queries``.  Reference
    # selection must therefore use its unique source member names first.
    reference_prefixes = (
        "vpt.prototype_extraction.",
        "vpt.prototype_text_interaction.",
        "vpt.prototype_norm1.",
        "vpt.prototype_norm2.",
        "vpt.prototype_mlp.",
    )
    if any(_has_state_prefix(state_dict, prefix) for prefix in reference_prefixes):
        return "reference"
    if _has_state_prefix(state_dict, "vpt.visual_prototype_extractor.") or _has_state_prefix(
        state_dict, "vpt.prototype_fusion."
    ):
        return "v2"
    legacy_prefixes = (
        "vpt.prototype_queries",
        "vpt.extract_attn.",
        "vpt.extract_norm.",
        "vpt.project_ffn.",
        "vpt.text_attn.",
        "vpt.text_norm.",
        "vpt.fusion_attn.",
        "vpt.fusion_norm.",
    )
    if any(_has_state_prefix(state_dict, prefix) for prefix in legacy_prefixes):
        return "legacy"
    raise ValueError(
        "Checkpoint config enables VPT but has no vpt_type and its state_dict is not a recognised "
        "reference, V2, or legacy VPT layout. Refusing to guess VPT compatibility; pass an explicit vpt_type "
        "override only after verifying the checkpoint architecture."
    )


def load_model(path, device="cpu", **overrides):
    """Rebuild RT-Counter from a checkpoint; explicit ``overrides`` replace saved config entries."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        # ``weights_only`` is unavailable in older supported PyTorch releases.
        if "weights_only" not in str(error):
            raise
        ckpt = torch.load(path, map_location="cpu")
    checkpoint_config = ckpt["config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint config must be a dictionary.")

    state_dict = ckpt.get("model")
    inferred_enhancer_type = _infer_enhancer_type(checkpoint_config, state_dict, overrides)
    inferred_vpt_type = _infer_vpt_type(checkpoint_config, state_dict, overrides)

    names = {f.name for f in fields(RTCounterConfig)}
    cfg = {k: v for k, v in checkpoint_config.items() if k in names}
    if inferred_enhancer_type is not None:
        cfg["enhancer_type"] = inferred_enhancer_type
    if inferred_vpt_type is not None:
        cfg["vpt_type"] = inferred_vpt_type
    cfg.update(yoloe_weights=None, **overrides)
    model = RTCounter(RTCounterConfig(**cfg))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), ckpt
