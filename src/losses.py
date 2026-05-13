from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = targets != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)
    probs = F.softmax(logits, dim=1)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    present = one_hot.sum(dims) > 0
    if not present.any():
        return logits.new_tensor(0.0)
    return 1.0 - dice[present].mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    ignore_index: int,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = targets != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss[valid].mean()


def supervised_loss(
    outputs: Dict,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    loss_cfg: Dict,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    ce = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore_index)
    dice = dice_loss(logits, targets, num_classes, ignore_index)
    focal = focal_loss(logits, targets, loss_cfg.get("focal_gamma", 2.0), ignore_index, class_weights)
    total = loss_cfg.get("ce", 1.0) * ce + loss_cfg.get("dice", 0.0) * dice + loss_cfg.get("focal", 0.0) * focal
    items = {"ce": float(ce.detach()), "dice": float(dice.detach()), "focal": float(focal.detach())}
    return total, items


def logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    t_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
    pixels = student_logits.shape[2] * student_logits.shape[3]
    return kd / max(pixels, 1)


def feature_kd_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    teacher_feature = teacher_feature.detach()
    if student_feature.shape[2:] != teacher_feature.shape[2:]:
        teacher_feature = F.interpolate(teacher_feature, size=student_feature.shape[2:], mode="bilinear", align_corners=False)
    return F.mse_loss(F.normalize(student_feature, dim=1), F.normalize(teacher_feature, dim=1))


def prototype_kd_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    teacher_feature = teacher_feature.detach()
    if student_feature.shape[2:] != teacher_feature.shape[2:]:
        teacher_feature = F.interpolate(teacher_feature, size=student_feature.shape[2:], mode="bilinear", align_corners=False)
    s_feat = F.normalize(student_feature, dim=1).permute(0, 2, 3, 1)
    t_feat = F.normalize(teacher_feature, dim=1).permute(0, 2, 3, 1)

    losses = []
    for cls in range(num_classes):
        mask = targets == cls
        if mask.any():
            s_proto = s_feat[mask].mean(dim=0)
            t_proto = t_feat[mask].mean(dim=0)
            losses.append(F.mse_loss(s_proto, t_proto))
    if not losses:
        return student_feature.new_tensor(0.0)
    return torch.stack(losses).mean()


def router_alignment_loss(student_outputs: Dict, teacher_outputs: Dict) -> torch.Tensor:
    student_routers = student_outputs.get("router_probs", [])
    teacher_routers = teacher_outputs.get("router_probs", [])
    if not student_routers or not teacher_routers:
        return student_outputs["logits"].new_tensor(0.0)
    losses = []
    for s_probs, t_probs in zip(student_routers, teacher_routers):
        t_probs = t_probs.detach()
        if s_probs.shape[2:] != t_probs.shape[2:]:
            t_probs = F.interpolate(t_probs, size=s_probs.shape[2:], mode="nearest")
        losses.append(F.kl_div((s_probs + 1e-6).log(), t_probs, reduction="batchmean") / max(s_probs.shape[2] * s_probs.shape[3], 1))
    return torch.stack(losses).mean()


def distillation_loss(
    student_outputs: Dict,
    teacher_outputs: Dict,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    distill_cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    temperature = distill_cfg.get("temperature", 4.0)
    logit = logit_kd_loss(student_outputs["logits"], teacher_outputs["logits"], temperature)
    feature = feature_kd_loss(student_outputs["feature"], teacher_outputs["feature"])
    prototype = prototype_kd_loss(
        student_outputs["feature"], teacher_outputs["feature"], targets, num_classes, ignore_index
    )
    router = router_alignment_loss(student_outputs, teacher_outputs)
    total = (
        distill_cfg.get("logit", 1.0) * logit
        + distill_cfg.get("feature", 1.0) * feature
        + distill_cfg.get("prototype", 0.0) * prototype
        + distill_cfg.get("router", 0.0) * router
    )
    items = {
        "kd_logit": float(logit.detach()),
        "kd_feature": float(feature.detach()),
        "kd_prototype": float(prototype.detach()),
        "kd_router": float(router.detach()),
    }
    return total, items
