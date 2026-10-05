"""Visual embedding module (supplementary material, Fig. 8).

The three PAN outputs are brought to H/16 (strided conv for F_p1, bilinear
upsampling for F_p3), refined by depth-wise convolutions, concatenated, and fused
by a Mixture-of-Experts whose gate weights the outputs of all experts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )


def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


class InceptionExpert(nn.Module):
    """Stacked depth-wise convolutions with growing kernels (3x3 to 11x11)."""

    def __init__(self, c1, c2):
        super().__init__()
        hidden = _make_divisible(c2)
        self.pre = nn.Sequential(nn.Conv2d(c1, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.dw = nn.ModuleList([
            nn.Conv2d(hidden, hidden, k, 1, k // 2, groups=hidden, bias=False) for k in (3, 5, 7, 9, 11)
        ])
        self.pw = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.post = nn.Sequential(nn.Conv2d(hidden, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.dw[0](self.pre(x))
        for conv in self.dw[1:]:
            y = y + conv(y)
        return self.post(self.pw(y))


class DepthwiseExpert(nn.Sequential):
    """Depth-wise separable convolution."""

    def __init__(self, c1, c2):
        super().__init__(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
        )


class ContextAnchorExpert(nn.Module):
    """Context anchor attention: strip convolutions (1x11, 11x1) produce a spatial gate."""

    def __init__(self, c1, c2, kernel=11):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Sequential(nn.Conv2d(c2, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.h_conv = nn.Conv2d(c2, c2, (1, kernel), 1, (0, kernel // 2), groups=c2, bias=False)
        self.v_conv = nn.Conv2d(c2, c2, (kernel, 1), 1, (kernel // 2, 0), groups=c2, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(c2, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proj(x)
        attn = self.conv2(self.v_conv(self.h_conv(self.conv1(self.pool(x)))))
        return x * attn.sigmoid()


class PointwiseExpert(nn.Sequential):
    """1x1 projection."""

    def __init__(self, c1, c2):
        super().__init__(nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))


EXPERTS = (InceptionExpert, DepthwiseExpert, ContextAnchorExpert, PointwiseExpert)


class VisualEmbedding(nn.Module):
    """Fuses PAN features [F_p1 (H/8), F_p2 (H/16), F_p3 (H/32)] into E_v at H/16.

    Args:
        in_channels: channels of (F_p1, F_p2, F_p3), e.g. (128, 256, 512) for YOLOE-11s.
        dim: output dimension D.
        num_experts: number of MoE experts (cycles through the four expert types).
    """

    def __init__(self, in_channels=(128, 256, 512), dim=512, num_experts=4):
        super().__init__()
        assert len(in_channels) == 3
        c1, _, _ = in_channels
        self.downsample = ConvBNAct(c1, c1, 3, 2)
        self.refine = nn.ModuleList([ConvBNAct(c, c, 3, groups=c) for c in in_channels])
        concat = sum(in_channels)
        self.experts = nn.ModuleList([EXPERTS[i % len(EXPERTS)](concat, dim) for i in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(concat, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, num_experts),
        )

    def forward(self, feats):
        """
        Args:
            feats: list of the three PAN feature maps.

        Returns:
            E_v as a map ``[B, D, H/16, W/16]`` and the gate weights ``[B, num_experts]``.
        """
        p1, p2, p3 = feats
        size = p2.shape[-2:]
        p1 = self.downsample(p1)
        if p1.shape[-2:] != size:
            p1 = F.interpolate(p1, size=size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
        x = torch.cat([refine(p) for refine, p in zip(self.refine, (p1, p2, p3))], dim=1)

        weights = self.gate(F.adaptive_avg_pool2d(x, 1).flatten(1)).softmax(dim=-1)
        out = torch.stack([expert(x) for expert in self.experts], dim=1)   # [B, E, D, H, W]
        out = (out * weights[:, :, None, None, None]).sum(dim=1)
        return out, weights
