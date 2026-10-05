"""CPU tests for the RT-Counter modules (no pretrained weights needed)."""
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.embedding import VisualEmbedding  # noqa: E402
from rtcounter.head import PointHead, make_anchor_points  # noqa: E402
from rtcounter.loss import RTCounterLoss, smooth_ln  # noqa: E402
from rtcounter.vpt import VPT  # noqa: E402
from rtcounter.weaformer import ConvAttention, FeatureEnhancer, WeaverAttention  # noqa: E402


def naive_conv_attention(module, x):
    """Explicit per-position k x k window attention with zero padding."""
    B, C, H, W = x.shape
    h, d, k = module.num_heads, module.head_dim, module.kernel_size
    q, key, v = module.q(x), F.pad(module.k(x), [k // 2] * 4), F.pad(module.v(x), [k // 2] * 4)
    out = torch.zeros_like(x)
    for i in range(H):
        for j in range(W):
            kw = key[:, :, i:i + k, j:j + k].reshape(B, h, d, k * k)
            vw = v[:, :, i:i + k, j:j + k].reshape(B, h, d, k * k)
            qi = q[:, :, i, j].reshape(B, h, d, 1)
            attn = ((qi * kw).sum(2) * module.scale).softmax(-1)           # [B, h, k*k]
            out[:, :, i, j] = (vw * attn[:, :, None]).sum(-1).reshape(B, C)
    return out


class TestWeaformer(unittest.TestCase):
    def test_conv_attention_matches_naive_window_attention(self):
        torch.manual_seed(0)
        m = ConvAttention(dim=12, num_heads=3, kernel_size=3).eval()
        x = torch.randn(2, 12, 5, 7)
        torch.testing.assert_close(m(x), naive_conv_attention(m, x), rtol=1e-4, atol=1e-5)

    def test_weaver_channel_split_follows_global_ratio(self):
        m = WeaverAttention(512, num_heads=8, global_ratio=0.25)
        self.assertEqual((m.global_dim, m.local_dim), (128, 384))
        self.assertEqual(m.global_attn.num_heads, 2)
        self.assertEqual(m.local_attn.num_heads, 6)

    def test_weaver_shapes_and_ablations(self):
        x = torch.randn(2, 24 * 32, 64)
        for ratio in (0.0, 0.25, 0.5, 1.0):
            for rate in (1, 4):
                m = WeaverAttention(64, num_heads=4, downsample_rate=rate, global_ratio=ratio).eval()
                self.assertEqual(m(x, 24, 32).shape, x.shape)
        self.assertIsNone(WeaverAttention(64, 4, global_ratio=0.0).global_attn)
        self.assertIsNone(WeaverAttention(64, 4, global_ratio=1.0).local_attn)

    def test_weaver_handles_odd_sizes(self):
        m = WeaverAttention(64, num_heads=4).eval()
        x = torch.randn(1, 5 * 7, 64)
        self.assertEqual(m(x, 5, 7).shape, x.shape)

    def test_feature_enhancer_shape(self):
        m = FeatureEnhancer(dim=64, depth=2, num_heads=4).eval()
        x, p = torch.randn(2, 12 * 16, 64), torch.randn(2, 17, 64)
        self.assertEqual(m(x, p, 12, 16).shape, x.shape)


class TestVPT(unittest.TestCase):
    def test_output_shape_and_no_prototype_collapse(self):
        torch.manual_seed(0)
        m = VPT(dim=64, num_prototypes=16, num_heads=4).eval()
        out = m(torch.randn(2, 100, 64), torch.randn(2, 1, 64))
        self.assertEqual(out.shape, (2, 17, 64))
        prototypes = out[:, :16]
        self.assertGreater((prototypes - prototypes.mean(1, keepdim=True)).abs().max().item(), 1e-3)


class TestEmbeddingAndHead(unittest.TestCase):
    def test_visual_embedding_aligns_to_stride_16(self):
        m = VisualEmbedding((32, 64, 128), dim=64, num_experts=4).eval()
        feats = [torch.randn(2, 32, 48, 64), torch.randn(2, 64, 24, 32), torch.randn(2, 128, 12, 16)]
        out, gate = m(feats)
        self.assertEqual(out.shape, (2, 64, 24, 32))
        torch.testing.assert_close(gate.sum(-1), torch.ones(2))

    def test_anchor_points_form_a_stride_8_grid(self):
        anchors = make_anchor_points(2, 3, (32, 48))
        self.assertEqual(anchors.shape, (2 * 3 * 4, 2))
        torch.testing.assert_close(anchors[:4], torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]]))
        xs, ys = anchors[:, 0].unique(), anchors[:, 1].unique()
        torch.testing.assert_close(xs, torch.arange(4.0, 48, 8))
        torch.testing.assert_close(ys, torch.arange(4.0, 32, 8))

    def test_point_head_shapes(self):
        m = PointHead(dim=64).eval()
        points, logits = m(torch.randn(2, 64, 24, 32), (384, 512))
        self.assertEqual(points.shape, (2, 24 * 32 * 4, 2))
        self.assertEqual(logits.shape, (2, 24 * 32 * 4, 2))


class TestLoss(unittest.TestCase):
    def test_smooth_ln_is_continuous_and_matches_definition(self):
        sigma = 0.5
        z = torch.tensor([0.0, 0.25, sigma, 0.75, 10.0])
        expected = torch.tensor([0.0, -math.log(0.75), -math.log(0.5),
                                 0.25 / 0.5 - math.log(0.5), 9.5 / 0.5 - math.log(0.5)])
        torch.testing.assert_close(smooth_ln(z, sigma), expected)

    def test_loss_terms(self):
        crit = RTCounterLoss(lambda_reg=0.25, lambda_cls=5.0, neg_weight=0.5)
        points = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]])
        logits = torch.zeros(1, 3, 2)                       # c = 0.5 everywhere
        target = [torch.tensor([[10.0, 10.0]])]
        out = crit({"pred_points": points, "pred_logits": logits}, target)
        self.assertAlmostEqual(out["loss_reg"].item(), 0.0, places=6)          # exact match -> Smooth_ln(0) = 0
        expected_cls = -(math.log(0.5) + 0.5 * 2 * math.log(0.5)) / 3
        self.assertAlmostEqual(out["loss_cls"].item(), expected_cls, places=5)
        self.assertAlmostEqual(out["loss"].item(), 5.0 * expected_cls, places=5)

    def test_empty_targets(self):
        crit = RTCounterLoss()
        points = torch.randn(2, 8, 2, requires_grad=True)
        logits = torch.randn(2, 8, 2, requires_grad=True)
        out = crit({"pred_points": points, "pred_logits": logits}, [torch.zeros(0, 2), torch.zeros(0, 2)])
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))


class TestFullModel(unittest.TestCase):
    def test_forward_with_precomputed_text(self):
        try:
            from rtcounter import RTCounter, RTCounterConfig
            model = RTCounter(RTCounterConfig()).eval()
        except ImportError as e:
            self.skipTest(f"YOLOE fork not installed: {e}")
        images = torch.rand(1, 3, 128, 160)
        text = F.normalize(torch.randn(1, 512), dim=-1)
        with torch.no_grad():
            out = model(images, text_embeddings=text)
        self.assertEqual(out["pred_points"].shape, (1, 8 * 10 * 4, 2))
        counts, points = model.count(images, text_embeddings=text)
        self.assertEqual(int(counts[0]), len(points[0]))


if __name__ == "__main__":
    unittest.main()
