"""Focused contracts for the user-reference prototype context path."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.vpt_reference import ReferenceVPT  # noqa: E402


class TestReferenceVPT(unittest.TestCase):
    def test_source_ordered_text_and_prototypes_without_extra_context_self_attention(self):
        torch.manual_seed(20261005)
        module = ReferenceVPT(dim=64, num_prototypes=16, num_heads=4, mlp_ratio=2.0).eval()
        visual_tokens = torch.randn(2, 7 * 10, 64)
        text_tokens = torch.randn(2, 1, 64)

        queries = module.prototype_queries.expand(visual_tokens.shape[0], -1, -1)
        prototypes = module.prototype_norm1(module.prototype_extraction(queries, visual_tokens))
        prototypes = module.prototype_norm2(
            prototypes + module.prototype_text_interaction(prototypes, text_tokens)
        )
        prototypes = prototypes + module.prototype_mlp(prototypes)
        expected = torch.cat([text_tokens, prototypes], dim=1)

        actual = module(visual_tokens, text_tokens)

        self.assertEqual(actual.shape, (2, 17, 64))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[:, :1], text_tokens, rtol=0.0, atol=0.0)
        self.assertFalse(hasattr(module, "context_self_attention"))
        self.assertFalse(hasattr(module, "context_norm"))

    def test_uses_reference_parameter_layout_not_v2_gate_or_legacy_concat_attention(self):
        module = ReferenceVPT(dim=64, num_prototypes=16, num_heads=4, mlp_ratio=2.0)
        state_keys = set(module.state_dict())

        self.assertIn("prototype_queries", state_keys)
        self.assertIn("prototype_extraction.wq.weight", state_keys)
        self.assertIn("prototype_text_interaction.wk.weight", state_keys)
        self.assertIn("prototype_norm1.weight", state_keys)
        self.assertIn("prototype_norm2.weight", state_keys)
        self.assertIn("prototype_mlp.fc1.weight", state_keys)
        self.assertTrue(module.prototype_extraction.wq.bias is not None)
        self.assertEqual(module.prototype_mlp.fc1.out_features, 128)
        for forbidden_name in ("gate", "query_pos", "extract_attn", "fusion_attn", "context_self_attention"):
            self.assertFalse(hasattr(module, forbidden_name))

    def test_gradients_reach_queries_and_both_cross_attention_stages(self):
        module = ReferenceVPT(dim=64, num_prototypes=8, num_heads=4, mlp_ratio=2.0)
        visual_tokens = torch.randn(2, 30, 64, requires_grad=True)
        text_tokens = torch.randn(2, 1, 64, requires_grad=True)

        module(visual_tokens, text_tokens).square().mean().backward()

        self.assertTrue(torch.isfinite(visual_tokens.grad).all())
        self.assertTrue(torch.isfinite(text_tokens.grad).all())
        self.assertIsNotNone(module.prototype_queries.grad)
        self.assertIsNotNone(module.prototype_extraction.wq.weight.grad)
        self.assertIsNotNone(module.prototype_text_interaction.wq.weight.grad)


if __name__ == "__main__":
    unittest.main()
