"""Unit contracts for the V2 source-selected FIM path."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.fim_v2 import V2FIM  # noqa: E402
from rtcounter.model import RTCounter, RTCounterConfig  # noqa: E402
from rtcounter.vpt_v2 import V2VPT  # noqa: E402


class TestV2FIM(unittest.TestCase):
    def test_shape_and_backpropagation(self):
        torch.manual_seed(12)
        module = V2FIM(
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
        )
        visual = torch.randn(2, 7 * 10, 64, requires_grad=True)
        context = torch.randn(2, 1, 64, requires_grad=True)
        out = module(visual, context, 7, 10)
        self.assertEqual(out.shape, visual.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_v2_topology_uses_full_token_self_attention_and_learned_position_table(self):
        module = V2FIM(dim=64, depth=1, num_heads=4, drop=0.0, attn_drop=0.0, drop_path=0.0)
        block = module.fim_blocks[0]
        self.assertEqual(tuple(module.fim_pos_embed.shape), (1, 48 * 48, 64))
        self.assertTrue(module.fim_pos_embed.requires_grad)
        self.assertTrue(torch.count_nonzero(module.fim_pos_embed).item())
        self.assertEqual(block.selfattn.qkv.in_features, 64)
        self.assertEqual(block.selfattn.num_heads, 4)
        self.assertTrue(hasattr(block, "norm0"))
        self.assertTrue(hasattr(block, "norm1"))
        self.assertTrue(hasattr(block, "norm2"))
        self.assertFalse(hasattr(block, "multi_dilate_attn"))
        self.assertFalse(hasattr(block, "selfattn_y"))

    def test_v2_fim_requires_one_text_token_and_original_position_capacity(self):
        module = V2FIM(dim=64, depth=1, num_heads=4)
        visual = torch.randn(1, 3 * 5, 64)
        with self.assertRaisesRegex(ValueError, "context \[B, 1, D\]"):
            module(visual, torch.randn(1, 17, 64), 3, 5)
        with self.assertRaisesRegex(ValueError, "48x48"):
            module(torch.randn(1, 48 * 48 + 1, 64), torch.randn(1, 1, 64), 1, 48 * 48 + 1)

    def test_model_selects_v2_fim_and_v2_vpt_explicitly(self):
        class StubBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = (16, 24, 32)
                self.embed_dim = 64

        common = dict(dim=64, num_heads=4, enhancer_depth=1, mlp_ratio=2.0, num_experts=1)
        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            model = RTCounter(RTCounterConfig(enhancer_type="v2_fim", vpt_type="v2", **common))
        self.assertIsInstance(model.enhancer, V2FIM)
        self.assertIsInstance(model.vpt, V2VPT)
        self.assertEqual(tuple(model.enhancer.fim_pos_embed.shape), (1, 48 * 48, 64))

    def test_legacy_vpt_cannot_be_silently_connected_to_v2_fim(self):
        class StubBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = (16, 24, 32)
                self.embed_dim = 64

        cfg = RTCounterConfig(dim=64, num_heads=4, num_experts=1, enhancer_type="v2_fim", vpt_type="legacy")
        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            with self.assertRaisesRegex(ValueError, "requires vpt_type='v2'"):
                RTCounter(cfg)


if __name__ == "__main__":
    unittest.main()
