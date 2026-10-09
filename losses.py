from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class LossComponents:
    total: torch.Tensor
    event: torch.Tensor
    duration: torch.Tensor
    gap: torch.Tensor
    event_per_token: torch.Tensor
    duration_per_token: torch.Tensor
    gap_per_token: torch.Tensor


def masked_focal_ce(logits, target, mask, gamma=1.5, focal=True):
    safe = target.masked_fill(~mask.bool(), 0)
    ce = F.cross_entropy(logits.transpose(1, 2), safe, reduction="none")
    per = ce * ((1 - torch.exp(-ce)) ** gamma if focal else 1.0)
    per = per * mask
    return per.sum() / mask.sum().clamp_min(1), ce * mask


def sequence_loss(output, event, duration_bin, gap_bin, mask, gap_mask, cfg):
    le, pe = masked_focal_ce(output.event_logits, event, mask, cfg.focal_gamma, cfg.use_focal_event_loss)
    ld, pd = masked_focal_ce(output.duration_logits, duration_bin, mask, focal=False)
    if output.gap_logits is not None and cfg.use_gap:
        lg, pg = masked_focal_ce(
            output.gap_logits, gap_bin, gap_mask, focal=False
        )
    else:
        lg = le.new_zeros(())
        pg = torch.zeros_like(pe)
    return LossComponents(le + ld + lg, le, ld, lg, pe, pd, pg)


def gaussian_nll(output, target, mask, variance_regularization=0.0):
    residual = (target - output.mean).square() * torch.exp(-output.log_variance)
    per_feature = 0.5 * (output.log_variance + residual + math.log(2.0 * math.pi))
    per_token = per_feature.mean(-1) * mask
    loss = per_token.sum() / mask.sum().clamp_min(1)
    if variance_regularization:
        loss = loss + variance_regularization * output.log_variance.square().mean()
    return loss, per_token, residual.mean(-1) * mask, output.log_variance.mean(-1) * mask
