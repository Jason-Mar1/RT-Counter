"""Focused contracts for the V2 prototype extraction and gated fusion path."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.vpt_v2 import V2VPT  # noqa: E402


class TestV2VPT(unittest.TestCase):
    def test_returns_one_gated_text_token(self):
        module = V2VPT(dim=64, num_prototypes=16, num_heads=4).eval()
        visual_tokens = torch.randn(2, 7 * 10, 64)
        text_tokens = torch.randn(2, 1, 64)

        enhanced_text = module(visual_tokens, text_tokens)

        self.assertEqual(enhanced_text.shape, (2, 1, 64))
        self.assertTrue(hasattr(module, "visual_prototype_extractor"))
        self.assertTrue(hasattr(module, "prototype_fusion"))
        self.assertFalse(hasattr(module, "fusion_attn"))

    def test_lossless_token_view_matches_source_style_map_path(self):
        torch.manual_seed(20261005)
        module = V2VPT(dim=64, num_prototypes=8, num_heads=4).eval()
        visual_tokens = torch.randn(2, 6 * 10, 64)
        text_tokens = torch.randn(2, 1, 64)

        expected_map = visual_tokens.transpose(1, 2).reshape(2, 64, 6, 10)
        prototypes, _ = module.visual_prototype_extractor(expected_map)
        expected, _ = module.prototype_fusion(prototypes, text_tokens)
        actual = module(visual_tokens, text_tokens)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_gradients_flow_through_both_v2_submodules(self):
        module = V2VPT(dim=64, num_prototypes=8, num_heads=4)
        visual_tokens = torch.randn(2, 30, 64, requires_grad=True)
        text_tokens = torch.randn(2, 1, 64, requires_grad=True)

        module(visual_tokens, text_tokens).square().mean().backward()

        self.assertTrue(torch.isfinite(visual_tokens.grad).all())
        self.assertTrue(torch.isfinite(text_tokens.grad).all())
        self.assertIsNotNone(module.visual_prototype_extractor.prototype_queries.grad)
        self.assertIsNotNone(module.prototype_fusion.gate[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
