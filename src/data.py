import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import resolve_path


def _read_mat_variable(path: str, key: str) -> np.ndarray:
    mat = sio.loadmat(path)
    if key not in mat:
        visible = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Variable '{key}' not found in {path}. Available variables: {visible}")
    return np.asarray(mat[key])


def load_houston_arrays(config: Dict) -> Dict[str, np.ndarray]:
    data_cfg = config["data"]
    root = os.path.abspath(data_cfg.get("root", "."))
    hs = _read_mat_variable(resolve_path(root, data_cfg["hs_path"]), data_cfg["hs_key"]).astype(
        np.float32, copy=False
    )
    ms = _read_mat_variable(resolve_path(root, data_cfg["ms_path"]), data_cfg["ms_key"]).astype(
        np.float32, copy=False
    )
    train_label = _read_mat_variable(
        resolve_path(root, data_cfg["train_label_path"]), data_cfg["train_label_key"]
    ).astype(np.int64, copy=False)
    test_label = _read_mat_variable(
        resolve_path(root, data_cfg["test_label_path"]), data_cfg["test_label_key"]
    ).astype(np.int64, copy=False)

    if hs.shape[:2] != ms.shape[:2] or hs.shape[:2] != train_label.shape or hs.shape[:2] != test_label.shape:
        raise ValueError(
            "HS, MS, TR, and TE spatial sizes must match. "
            f"Got HS={hs.shape}, MS={ms.shape}, TR={train_label.shape}, TE={test_label.shape}."
        )
    return {"hs": hs, "ms": ms, "train_label": train_label, "test_label": test_label}


def map_labels(raw_label: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    mapped = np.full(raw_label.shape, ignore_index, dtype=np.int64)
    valid = (raw_label >= 1) & (raw_label <= num_classes)
    mapped[valid] = raw_label[valid] - 1
    return mapped


def compute_class_weights(raw_label: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.array([(raw_label == cls).sum() for cls in range(1, num_classes + 1)], dtype=np.float64)
    nonzero = counts[counts > 0]
    weights = np.ones(num_classes, dtype=np.float32)
    if len(nonzero) > 0:
        median = np.median(nonzero)
        weights = median / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


class HoustonPatchDataset(Dataset):
    def __init__(
        self,
        hs: np.ndarray,
        ms: np.ndarray,
        raw_label: np.ndarray,
        num_classes: int,
        ignore_index: int,
        patch_size: int = 31,
        samples_per_epoch: int = 15000,
        balanced_sampling: bool = True,
        augment: bool = True,
    ) -> None:
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd so every patch has a center pixel.")
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.samples_per_epoch = samples_per_epoch
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.balanced_sampling = balanced_sampling
        self.augment = augment

        self.hs = np.pad(hs, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")
        self.ms = np.pad(ms, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")
        mapped_label = map_labels(raw_label, num_classes, ignore_index)
        self.label = np.pad(
            mapped_label,
            ((self.pad, self.pad), (self.pad, self.pad)),
            mode="constant",
            constant_values=ignore_index,
        )

        self.class_coords: List[np.ndarray] = []
        for cls in range(1, num_classes + 1):
            coords = np.argwhere(raw_label == cls)
            if coords.size > 0:
                self.class_coords.append(coords)
        if not self.class_coords:
            raise ValueError("No labeled training pixels found.")
        self.all_coords = np.argwhere(raw_label > 0)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_coord(self, index: int) -> Tuple[int, int]:
        if self.balanced_sampling:
            coords = self.class_coords[index % len(self.class_coords)]
        else:
            coords = self.all_coords
        row, col = coords[np.random.randint(0, len(coords))]
        return int(row), int(col)

    def _augment(self, hs_patch: np.ndarray, ms_patch: np.ndarray, label_patch: np.ndarray):
        if np.random.rand() < 0.5:
            hs_patch = np.flip(hs_patch, axis=0)
            ms_patch = np.flip(ms_patch, axis=0)
            label_patch = np.flip(label_patch, axis=0)
        if np.random.rand() < 0.5:
            hs_patch = np.flip(hs_patch, axis=1)
            ms_patch = np.flip(ms_patch, axis=1)
            label_patch = np.flip(label_patch, axis=1)
        k = np.random.randint(0, 4)
        if k:
            hs_patch = np.rot90(hs_patch, k, axes=(0, 1))
            ms_patch = np.rot90(ms_patch, k, axes=(0, 1))
            label_patch = np.rot90(label_patch, k, axes=(0, 1))
        return hs_patch, ms_patch, label_patch

    def __getitem__(self, index: int):
        row, col = self._sample_coord(index)
        row0, col0 = row, col
        row1, col1 = row0 + self.patch_size, col0 + self.patch_size
        hs_patch = self.hs[row0:row1, col0:col1, :]
        ms_patch = self.ms[row0:row1, col0:col1, :]
        label_patch = self.label[row0:row1, col0:col1]

        if self.augment:
            hs_patch, ms_patch, label_patch = self._augment(hs_patch, ms_patch, label_patch)

        hs_tensor = torch.from_numpy(np.ascontiguousarray(hs_patch.transpose(2, 0, 1))).float()
        ms_tensor = torch.from_numpy(np.ascontiguousarray(ms_patch.transpose(2, 0, 1))).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_patch)).long()
        return {"hs": hs_tensor, "ms": ms_tensor, "label": label_tensor}


def make_train_loader(config: Dict, arrays: Dict[str, np.ndarray]) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    dataset = HoustonPatchDataset(
        hs=arrays["hs"],
        ms=arrays["ms"],
        raw_label=arrays["train_label"],
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
        patch_size=data_cfg["patch_size"],
        samples_per_epoch=train_cfg["samples_per_epoch"],
        balanced_sampling=train_cfg.get("balanced_sampling", True),
        augment=train_cfg.get("augment", True),
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _window_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    ms: np.ndarray,
    device: torch.device,
    hs: Optional[np.ndarray] = None,
    tile_size: int = 256,
    stride: int = 192,
) -> np.ndarray:
    model.eval()
    height, width = ms.shape[:2]
    row_starts = _window_starts(height, tile_size, stride)
    col_starts = _window_starts(width, tile_size, stride)

    logits_sum: Optional[torch.Tensor] = None
    count = torch.zeros(1, height, width, dtype=torch.float32)

    for row in row_starts:
        for col in col_starts:
            row2 = min(row + tile_size, height)
            col2 = min(col + tile_size, width)
            ms_tile = torch.from_numpy(ms[row:row2, col:col2, :].transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            if hs is None:
                outputs = model(ms_tile)
            else:
                hs_tile = (
                    torch.from_numpy(hs[row:row2, col:col2, :].transpose(2, 0, 1))
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                outputs = model(hs_tile, ms_tile)
            logits = outputs["logits"].squeeze(0).detach().cpu()
            if logits_sum is None:
                logits_sum = torch.zeros(logits.shape[0], height, width, dtype=torch.float32)
            logits_sum[:, row:row2, col:col2] += logits
            count[:, row:row2, col:col2] += 1.0

    assert logits_sum is not None
    logits_sum = logits_sum / count.clamp_min(1.0)
    return logits_sum.argmax(dim=0).numpy().astype(np.int64)


def colorize_prediction(pred: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [166, 206, 227],
            [31, 120, 180],
            [178, 223, 138],
            [51, 160, 44],
            [251, 154, 153],
            [227, 26, 28],
            [253, 191, 111],
            [255, 127, 0],
            [202, 178, 214],
            [106, 61, 154],
            [255, 255, 153],
            [177, 89, 40],
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(pred, 0, len(palette) - 1)]
