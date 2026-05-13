import random
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    train_loss: float,
    config: Dict[str, Any],
    model_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "model_name": model_name,
        "config": config,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_state(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "", 1)] = value
    model.load_state_dict(cleaned)
    return ckpt


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
