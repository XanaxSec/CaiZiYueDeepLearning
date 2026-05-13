import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import colorize_prediction, load_houston_arrays, map_labels, predict_full_image
from src.metrics import compute_metrics, format_metrics
from src.models.networks import MSBaseline, StudentMSMoE, build_teacher_model, teacher_config_from_checkpoint
from src.train_utils import inference_window_config
from src.utils import ensure_dir, get_device, load_config, load_model_state, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Houston segmentation models.")
    parser.add_argument("--config", default="configs/houston_moe.yaml")
    parser.add_argument("--ckpt", default="checkpoints/student_best.pth")
    parser.add_argument("--model", choices=["student", "baseline", "teacher"], default="student")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_model(name: str, arrays, config, ckpt_path: str, device):
    num_classes = config["data"]["num_classes"]
    model_cfg = config["model"]
    if name == "student":
        return StudentMSMoE(
            ms_channels=arrays["ms"].shape[2],
            num_classes=num_classes,
            base_channels=model_cfg["base_channels"],
            num_experts=model_cfg["num_experts"],
            top_k=model_cfg["top_k"],
            dropout=model_cfg["dropout"],
        )
    if name == "baseline":
        return MSBaseline(
            ms_channels=arrays["ms"].shape[2],
            num_classes=num_classes,
            base_channels=model_cfg["base_channels"],
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    teacher_config = teacher_config_from_checkpoint(config, ckpt if isinstance(ckpt, dict) else {})
    return build_teacher_model(
        teacher_config,
        hs_channels=arrays["hs"].shape[2],
        ms_channels=arrays["ms"].shape[2],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    device = get_device(args.device)
    arrays = load_houston_arrays(config)
    model = build_model(args.model, arrays, config, args.ckpt, device).to(device)
    load_model_state(model, args.ckpt, device)

    hs = arrays["hs"] if args.model == "teacher" else None
    tile_size, stride = inference_window_config(config, args.model)
    pred = predict_full_image(
        model,
        arrays["ms"],
        device=device,
        hs=hs,
        tile_size=tile_size,
        stride=stride,
    )

    num_classes = config["data"]["num_classes"]
    ignore_index = config["data"]["ignore_index"]
    target = map_labels(arrays["test_label"], num_classes, ignore_index)
    metrics = compute_metrics(pred, target, num_classes, ignore_index)
    print(format_metrics(metrics))

    output_dir = config["paths"].get("output_dir", "outputs")
    ensure_dir(output_dir)
    metrics_path = os.path.join(output_dir, f"metrics_{args.model}.json")
    pred_npy_path = os.path.join(output_dir, f"pred_{args.model}.npy")
    pred_png_path = os.path.join(output_dir, f"pred_{args.model}.png")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(pred_npy_path, pred)
    plt.imsave(pred_png_path, colorize_prediction(pred))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved prediction to {pred_npy_path} and {pred_png_path}")


if __name__ == "__main__":
    main()
