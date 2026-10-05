"""Contracts for the FIM path restored from the user-supplied reference code."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.fim_reference import Attention, ReferenceFIM  # noqa: E402
from rtcounter.model import RTCounter, RTCounterConfig  # noqa: E402
from rtcounter.vpt_reference import ReferenceVPT  # noqa: E402


class TestReferenceFIM(unittest.TestCase):
    def test_shape_context_gradient_and_context_update(self):
        torch.manual_seed(31)
        module = ReferenceFIM(
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
        )
        visual = torch.randn(2, 5 * 7, 64, requires_grad=True)
        context = torch.randn(2, 17, 64, requires_grad=True)
        out = module(visual, context, 5, 7)

        self.assertEqual(out.shape, visual.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

        block = module.fim_blocks[0].eval()
        with torch.no_grad():
            _, updated_context = block(visual.detach(), context.detach(), pos_embed=torch.zeros_like(visual))
        self.assertFalse(torch.equal(updated_context, context.detach()))

    def test_reference_topology_and_dynamic_sine_position(self):
        module = ReferenceFIM(dim=64, depth=1, num_heads=4, drop=0.0, attn_drop=0.0, drop_path=0.0)
        block = module.fim_blocks[0]

        self.assertEqual(block.selfattn_x.qkv.in_features, 64)
        self.assertEqual(block.selfattn_x.num_heads, 4)
        self.assertEqual(block.selfattn_y.qkv.in_features, 64)
        self.assertEqual(block.selfattn_y.num_heads, 4)
        self.assertTrue(hasattr(block, "cross_attn"))
        self.assertFalse(hasattr(block, "multi_dilate_attn"))
        self.assertFalse(hasattr(block, "downsample_ratio"))
        self.assertFalse(hasattr(module, "fim_pos_embed"))
        self.assertEqual(list(module.fim_pos_encoding.parameters()), [])

        pos_3x5 = module.fim_pos_encoding(torch.ones(1, 3, 5))
        pos_4x8 = module.fim_pos_encoding(torch.ones(1, 4, 8))
        self.assertEqual(tuple(pos_3x5.shape), (1, 3, 5, 64))
        self.assertEqual(tuple(pos_4x8.shape), (1, 4, 8, 64))

        # Unlike the earlier learned-48x48 V2 path, the reference's dynamic
        # sine encoding has no artificial 2,304-token capacity gate.
        uncapped = ReferenceFIM(dim=64, depth=0, num_heads=4)
        uncapped_out = uncapped(torch.randn(1, 48 * 48 + 1, 64), torch.randn(1, 1, 64), 1, 48 * 48 + 1)
        self.assertEqual(tuple(uncapped_out.shape), (1, 48 * 48 + 1, 64))

    def test_position_is_added_to_q_and_k_but_not_v(self):
        attention = Attention(dim=4, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.0).eval()
        with torch.no_grad():
            identity = torch.eye(4)
            attention.qkv.weight.copy_(torch.cat([identity, identity, identity]))
            attention.proj.weight.copy_(identity)
            attention.proj.bias.zero_()

        x = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
        pos = torch.tensor([[[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        expected_attention = torch.softmax(((x + pos) @ (x + pos).transpose(-2, -1)) * 0.5, dim=-1)
        expected = expected_attention @ x
        torch.testing.assert_close(attention(x, pos_embed=pos), expected)

    def test_model_defaults_to_reference_fim_and_reference_vpt(self):
        class StubBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = (16, 24, 32)
                self.embed_dim = 64

        common = dict(dim=64, num_heads=4, enhancer_depth=1, mlp_ratio=2.0, num_experts=1)
        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            model = RTCounter(RTCounterConfig(**common))
        self.assertIsInstance(model.enhancer, ReferenceFIM)
        self.assertIsInstance(model.vpt, ReferenceVPT)

    def test_reference_fim_rejects_nonreference_vpt_but_allows_no_vpt_ablation(self):
        class StubBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.channels = (16, 24, 32)
                self.embed_dim = 64

        with patch("rtcounter.model.YOLOEBackbone", return_value=StubBackbone()):
            for vpt_type in ("v2", "legacy"):
                with self.assertRaisesRegex(ValueError, "requires vpt_type='reference'"):
                    RTCounter(RTCounterConfig(dim=64, num_heads=4, num_experts=1, vpt_type=vpt_type))

            ablation = RTCounter(
                RTCounterConfig(dim=64, num_heads=4, num_experts=1, use_vpt=False, vpt_type="v2")
            )
        self.assertIsNone(ablation.vpt)
        self.assertIsInstance(ablation.enhancer, ReferenceFIM)


if __name__ == "__main__":
    unittest.main()
