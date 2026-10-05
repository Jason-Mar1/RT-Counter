"""Hungarian matching and the training objective of Sec. 3.4."""
import math

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """One-to-one matching between predicted and ground-truth points (P2PNet style).

    cost = cost_point * ||p_hat - p||_2 - cost_class * c_hat
    """

    def __init__(self, cost_class=1.0, cost_point=0.05):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point

    @torch.no_grad()
    def forward(self, pred_points, pred_logits, targets):
        """
        Args:
            pred_points: ``[B, N, 2]``; pred_logits: ``[B, N, 2]``;
            targets: list of B tensors ``[N_GT_b, 2]``.

        Returns:
            list of (pred_idx, gt_idx) int64 tensors.
        """
        scores = pred_logits.float().softmax(-1)[..., 1]
        indices = []
        for points, score, gt in zip(pred_points.float(), scores, targets):
            if len(gt) == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                continue
            gt = gt.to(points.device, torch.float32)
            cost = self.cost_point * torch.cdist(points, gt) - self.cost_class * score[:, None]
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)))
        return indices


def smooth_ln(z, sigma=0.5):
    """Smooth_ln(z) = -ln(1 - z) if z <= sigma else (z - sigma) / (1 - sigma) - ln(1 - sigma)."""
    small = -torch.log1p(-z.clamp(max=sigma))
    large = (z - sigma) / (1 - sigma) - math.log(1 - sigma)
    return torch.where(z <= sigma, small, large)


class RTCounterLoss(nn.Module):
    """L_total = lambda_1 * L_reg + lambda_2 * L_cls.

    L_reg = 1/N_GT * sum_i Smooth_ln(d(p_hat_xi(i), p_i)), d in pixels.
    L_cls = -1/N_Grid * (sum_matched log c + neg_weight * sum_unmatched log(1 - c)).
    """

    def __init__(self, lambda_reg=0.25, lambda_cls=5.0, neg_weight=0.5, sigma=0.5,
                 cost_class=1.0, cost_point=0.05):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.neg_weight = neg_weight
        self.sigma = sigma
        self.matcher = HungarianMatcher(cost_class, cost_point)

    def forward(self, outputs, targets):
        points = outputs["pred_points"].float()
        logits = outputs["pred_logits"].float()
        B, N, _ = logits.shape
        indices = self.matcher(points, logits, targets)

        log_prob = logits.log_softmax(-1)                     # [..., 0] = log(1 - c), [..., 1] = log c
        positive = torch.zeros(B, N, dtype=torch.bool, device=logits.device)
        for b, (src, _) in enumerate(indices):
            positive[b, src.to(logits.device)] = True
        cls_terms = torch.where(positive, log_prob[..., 1], self.neg_weight * log_prob[..., 0])
        loss_cls = -cls_terms.sum(dim=1).div(N).mean()

        matched = [points[b, src.to(points.device)] for b, (src, _) in enumerate(indices)]
        gts = [gt.to(points.device)[dst.to(points.device)] for gt, (_, dst) in zip(targets, indices)]
        num_gt = max(sum(len(g) for g in gts), 1)
        if sum(len(m) for m in matched) > 0:
            dist = (torch.cat(matched) - torch.cat(gts).float()).norm(dim=-1)
            loss_reg = smooth_ln(dist, self.sigma).sum() / num_gt
        else:
            loss_reg = points.sum() * 0.0

        loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls
        return {"loss": loss, "loss_reg": loss_reg.detach(), "loss_cls": loss_cls.detach()}
