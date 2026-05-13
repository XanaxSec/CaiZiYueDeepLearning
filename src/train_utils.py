from typing import Dict, Iterable, Optional, Tuple

import torch

from .data import map_labels, predict_full_image
from .metrics import compute_metrics, format_metrics


def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    optim_cfg = config["optim"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict):
    epochs = config["train"].get("epochs", 300)
    min_lr = config["optim"].get("min_lr", 1e-6)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def maybe_clip_gradients(parameters: Iterable[torch.nn.Parameter], config: Dict) -> None:
    clip_norm: Optional[float] = config["train"].get("grad_clip_norm")
    if clip_norm and clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm)


def validation_enabled(config: Dict) -> bool:
    return bool(config.get("validation", {}).get("enabled", False))


def validation_metric_name(config: Dict) -> str:
    return str(config.get("validation", {}).get("metric", "mIoU"))


def should_validate_epoch(epoch: int, total_epochs: int, config: Dict) -> bool:
    return ValidationScheduler(config, total_epochs).should_validate(epoch)


class ValidationScheduler:
    def __init__(self, config: Dict, total_epochs: int) -> None:
        val_cfg = config.get("validation", {})
        self.enabled = bool(val_cfg.get("enabled", False))
        self.strategy = str(val_cfg.get("strategy", "interval"))
        self.metric = str(val_cfg.get("metric", "mIoU"))
        self.trigger_metric = str(val_cfg.get("trigger_metric", "OA"))
        threshold = float(val_cfg.get("trigger_threshold", 0.70))
        self.trigger_threshold = threshold / 100.0 if threshold > 1.0 else threshold
        self.burst_epochs = int(val_cfg.get("burst_epochs", 4))
        self.total_epochs = int(total_epochs)
        self.interval = max(int(val_cfg.get("interval", 1)), 1)
        self.milestones = {int(epoch) for epoch in val_cfg.get("milestones", [])}
        self.burst_queue = set()

    def should_validate(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        if self.strategy == "milestone_oa_burst":
            return epoch in self.milestones or epoch in self.burst_queue or epoch == self.total_epochs
        return epoch == 1 or epoch % self.interval == 0 or epoch == self.total_epochs

    def update(self, epoch: int, metrics: Dict) -> None:
        if not self.enabled or self.strategy != "milestone_oa_burst":
            return
        if epoch not in self.milestones:
            return
        trigger_score = float(metrics.get(self.trigger_metric, float("-inf")))
        if trigger_score <= self.trigger_threshold:
            return
        for offset in range(1, self.burst_epochs + 1):
            next_epoch = epoch + offset
            if next_epoch <= self.total_epochs:
                self.burst_queue.add(next_epoch)


def evaluate_on_te(
    model: torch.nn.Module,
    arrays: Dict,
    config: Dict,
    device: torch.device,
    model_kind: str,
) -> Tuple[Dict, float]:
    inference_cfg = config.get("inference", {})
    hs = arrays["hs"] if model_kind == "teacher" else None
    pred = predict_full_image(
        model,
        arrays["ms"],
        device=device,
        hs=hs,
        tile_size=inference_cfg.get("tile_size", 256),
        stride=inference_cfg.get("stride", 192),
    )
    num_classes = config["data"]["num_classes"]
    ignore_index = config["data"]["ignore_index"]
    target = map_labels(arrays["test_label"], num_classes, ignore_index)
    metrics = compute_metrics(pred, target, num_classes, ignore_index)
    metric_name = validation_metric_name(config)
    if metric_name not in metrics:
        raise KeyError(f"Validation metric '{metric_name}' is unavailable. Metrics: {list(metrics.keys())}")
    score = float(metrics[metric_name])
    print(format_metrics(metrics))
    return metrics, score
