"""Portable tests for the restored historical FIM.

Set ``FIM_REFERENCE_ARCHIVE`` to an archive containing
``models/yoloe_count_real_sota.py`` to additionally run the original-source
numerical oracle.  The normal test suite intentionally has no machine-local
archive dependency.
"""
import os
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.fim import CrossAttentionBlock, HistoricalFIM  # noqa: E402
from rtcounter.fim_reference import ReferenceFIM  # noqa: E402
from rtcounter.fim_v2 import V2FIM  # noqa: E402
from rtcounter.model import RTCounter, RTCounterConfig  # noqa: E402
from rtcounter.vpt_reference import ReferenceVPT  # noqa: E402
from rtcounter.vpt_v2 import V2VPT  # noqa: E402
from rtcounter.weaformer import FeatureEnhancer  # noqa: E402


def load_reference_cross_attention_block(archive_path):
    """Load only the historical FIM helpers directly from an explicit archive."""
    with zipfile.ZipFile(archive_path) as archive:
        source = archive.read("models/yoloe_count_real_sota.py").decode("utf-8")
    lines = source.splitlines()
    # This is the complete self-contained historical FIM helper region.
    helper_source = "\n".join(lines[102:615])
    namespace = {}
    exec(
        "import collections\nimport collections.abc\nimport math\nimport torch\n"
        "import torch.nn as nn\nimport torch.nn.functional as F\n"
        "from itertools import repeat\n" + helper_source,
        namespace,
    )
    return namespace["CrossAttentionBlock"]


class TestHistoricalFIM(unittest.TestCase):
    def test_shape_and_backpropagation(self):
        torch.manual_seed(9)
        module = HistoricalFIM(
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
            downsample_ratio=2,
        )
        x = torch.randn(2, 7 * 10, 64, requires_grad=True)
        context = torch.randn(2, 17, 64)
        out = module(x, context, 7, 10)
        self.assertEqual(out.shape, x.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_topology_is_fixed_historical_one_to_one(self):
        module = HistoricalFIM(dim=64, depth=1, num_heads=4, drop=0.0, attn_drop=0.0, drop_path=0.0)
        block = module.blocks[0]
        self.assertEqual(block.downsample_ratio, 2)
        self.assertEqual(block.selfattn_x.qkv.in_features, 32)
        self.assertEqual(block.selfattn_x.num_heads, 2)
        self.assertEqual(block.multi_dilate_attn.dim, 32)
        self.assertEqual(block.multi_dilate_attn.num_heads, 2)
        self.assertEqual(block.multi_dilate_attn.dilation, [1, 3])

    def test_rejects_invalid_or_nonspatial_fim_configurations(self):
        with self.assertRaisesRegex(ValueError, "fim_downsample_ratio >= 2"):
            HistoricalFIM(dim=64, depth=1, num_heads=4, downsample_ratio=1)

        block = CrossAttentionBlock(64, 4, downsample_ratio=2)
        x, y = torch.randn(1, 6 * 8, 64), torch.randn(1, 17, 64)
        with self.assertRaisesRegex(ValueError, "requires H and W"):
            block(x, y)
        with self.assertRaisesRegex(ValueError, "at least downsample_ratio"):
            block(torch.randn(1, 1 * 8, 64), y, H=1, W=8)
        with self.assertRaisesRegex(ValueError, "requires downsample_ratio >= 2"):
            CrossAttentionBlock(64, 4, downsample_ratio=1)(x, y, H=6, W=8)

    def test_model_selects_reference_fim_by_default_and_keeps_other_backends_explicit(self):
        class StubBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = (16, 24, 32)
                self.embed_dim = 64

        common = dict(dim=64, num_heads=4, enhancer_depth=1, mlp_ratio=2.0, num_experts=1)
        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            reference = RTCounter(RTCounterConfig(**common))
            self.assertIsInstance(reference.enhancer, ReferenceFIM)
            self.assertIsInstance(reference.vpt, ReferenceVPT)

        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            v2 = RTCounter(RTCounterConfig(enhancer_type="v2_fim", vpt_type="v2", **common))
            self.assertIsInstance(v2.enhancer, V2FIM)
            self.assertIsInstance(v2.vpt, V2VPT)

        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            historical = RTCounter(RTCounterConfig(enhancer_type="historical_fim", vpt_type="legacy", **common))
            self.assertIsInstance(historical.enhancer, HistoricalFIM)
            self.assertEqual(historical.enhancer.blocks[0].downsample_ratio, 2)

        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            weaver = RTCounter(RTCounterConfig(enhancer_type="weaformer", vpt_type="legacy", **common))
            self.assertIsInstance(weaver.enhancer, FeatureEnhancer)

    def test_reference_cross_attention_block_equivalence_when_archive_is_supplied(self):
        archive_path = os.environ.get("FIM_REFERENCE_ARCHIVE")
        if not archive_path:
            self.skipTest("set FIM_REFERENCE_ARCHIVE to run the archived-source numerical oracle")

        reference_cls = load_reference_cross_attention_block(archive_path)
        kwargs = dict(
            dim=64,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            downsample_ratio=2,
        )
        torch.manual_seed(20261005)
        reference = reference_cls(**kwargs).eval()
        candidate = CrossAttentionBlock(**kwargs).eval()
        candidate.load_state_dict(reference.state_dict(), strict=True)

        x = torch.randn(2, 6 * 10, 64)
        y = torch.randn(2, 17, 64)
        pos = torch.randn(2, 6 * 10, 64)
        expected_x, expected_y = reference(x.clone(), y.clone(), pos_embed=pos.clone(), H=6, W=10)
        actual_x, actual_y = candidate(x.clone(), y.clone(), pos_embed=pos.clone(), H=6, W=10)
        torch.testing.assert_close(actual_x, expected_x, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_y, expected_y, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
