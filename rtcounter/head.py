"""Point prediction head (supplementary material) and anchor points."""
import torch
import torch.nn as nn


def _branch(dim, out_channels):
    return nn.Sequential(
        nn.Conv2d(dim, dim // 2, 3, padding=1), nn.BatchNorm2d(dim // 2), nn.ReLU(inplace=True),
        nn.Conv2d(dim // 2, dim // 4, 3, padding=1), nn.BatchNorm2d(dim // 4), nn.ReLU(inplace=True),
        nn.Conv2d(dim // 4, out_channels, 1),
    )


def make_anchor_points(h, w, image_hw, device=None):
    """Anchor points for an ``h x w`` feature map, a 2x2 sub-grid per cell.

    Every H/16 cell holds four anchors at the centres of its four quadrants, so the
    anchors form a regular grid with stride 8 on the input image.

    Returns:
        ``[h*w*4, 2]`` absolute (x, y) coordinates, ordered (row, col, anchor).
    """
    stride_y = image_hw[0] / h
    stride_x = image_hw[1] / w
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride_y
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride_x
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xs.flatten(), ys.flatten()], dim=1)                     # [h*w, 2]
    dx, dy = stride_x / 4, stride_y / 4
    offsets = torch.tensor([[-dx, -dy], [dx, -dy], [-dx, dy], [dx, dy]], device=device)
    return (centers[:, None, :] + offsets[None]).reshape(-1, 2)


class PointHead(nn.Module):
    """Regression and classification branches of three conv layers each.

    Every location predicts ``num_anchors`` points: an (x, y) offset from its
    anchor and a 2-way (background, object) logit.
    """

    def __init__(self, dim=512, num_anchors=4):
        super().__init__()
        self.num_anchors = num_anchors
        self.regression = _branch(dim, num_anchors * 2)
        self.classification = _branch(dim, num_anchors * 2)

    def forward(self, x, image_hw):
        """
        Args:
            x: F_en as a map ``[B, D, H, W]``.
            image_hw: (height, width) of the network input.

        Returns:
            pred_points ``[B, H*W*A, 2]`` (absolute pixels) and pred_logits ``[B, H*W*A, 2]``.
        """
        B, _, H, W = x.shape
        A = self.num_anchors
        offsets = self.regression(x).view(B, A, 2, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W * A, 2)
        logits = self.classification(x).view(B, A, 2, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W * A, 2)
        anchors = make_anchor_points(H, W, image_hw, device=x.device).to(offsets.dtype)
        return offsets + anchors[None], logits
