from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2


def combined_distillation_loss(
    ce_loss: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha_ce: float = 0.7,
    alpha_kl: float = 0.3,
    temperature: float = 2.0,
) -> torch.Tensor:
    kl_loss = distillation_kl_loss(student_logits, teacher_logits, temperature)
    return alpha_ce * ce_loss + alpha_kl * kl_loss
