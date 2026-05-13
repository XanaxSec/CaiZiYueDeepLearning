from typing import Dict, List, Optional

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _safe_list(values: np.ndarray) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for value in values:
        out.append(None if np.isnan(value) else float(value))
    return out


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> Dict:
    hist = confusion_matrix(pred, target, num_classes, ignore_index)
    total = hist.sum()
    tp = np.diag(hist).astype(np.float64)
    gt = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)

    oa = float(tp.sum() / total) if total > 0 else 0.0
    per_class_acc = np.divide(tp, gt, out=np.full(num_classes, np.nan), where=gt > 0)
    iou = np.divide(tp, gt + pred_sum - tp, out=np.full(num_classes, np.nan), where=(gt + pred_sum - tp) > 0)
    aa = float(np.nanmean(per_class_acc)) if np.isfinite(per_class_acc).any() else 0.0
    miou = float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0

    pe = float((gt * pred_sum).sum() / (total ** 2)) if total > 0 else 0.0
    kappa = float((oa - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-12 else 0.0
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "mIoU": miou,
        "per_class_accuracy": _safe_list(per_class_acc),
        "per_class_iou": _safe_list(iou),
        "confusion_matrix": hist.tolist(),
    }


def format_metrics(metrics: Dict) -> str:
    lines = [
        f"OA: {metrics['OA']:.6f}",
        f"AA: {metrics['AA']:.6f}",
        f"Kappa: {metrics['Kappa']:.6f}",
        f"mIoU: {metrics['mIoU']:.6f}",
    ]
    acc = metrics.get("per_class_accuracy", [])
    lines.append("Per-class accuracy: " + ", ".join("nan" if v is None else f"{v:.4f}" for v in acc))
    return "\n".join(lines)
