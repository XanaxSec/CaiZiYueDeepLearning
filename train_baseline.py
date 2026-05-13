import argparse
import os

from tqdm import tqdm
import torch

from src.data import compute_class_weights, load_houston_arrays, make_train_loader
from src.losses import supervised_loss
from src.models.networks import MSBaseline
from src.train_utils import (
    build_optimizer,
    build_scheduler,
    evaluate_on_te,
    maybe_clip_gradients,
    move_batch,
    should_validate_epoch,
    validation_enabled,
    validation_metric_name,
)
from src.utils import count_parameters, get_device, load_config, save_checkpoint, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MS-only CNN baseline on Houston.")
    parser.add_argument("--config", default="configs/houston_moe.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    seed_everything(config.get("seed", 42))
    device = get_device(args.device)

    arrays = load_houston_arrays(config)
    loader = make_train_loader(config, arrays)
    num_classes = config["data"]["num_classes"]
    ignore_index = config["data"]["ignore_index"]
    model = MSBaseline(
        ms_channels=arrays["ms"].shape[2],
        num_classes=num_classes,
        base_channels=config["model"]["base_channels"],
    ).to(device)
    print(f"Training MSBaseline on {device}. Trainable parameters: {count_parameters(model):,}")

    class_weights = None
    if config["loss"].get("use_class_weights", True):
        class_weights = compute_class_weights(arrays["train_label"], num_classes, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    ckpt_dir = config["paths"].get("checkpoint_dir", "checkpoints")
    best_path = os.path.join(ckpt_dir, "baseline_best.pth")
    last_path = os.path.join(ckpt_dir, "baseline_last.pth")
    use_val = validation_enabled(config)
    metric_name = validation_metric_name(config)
    best_score = float("-inf")
    best_loss = float("inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"Baseline epoch {epoch}/{config['train']['epochs']}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            outputs = model(batch["ms"])
            sup_loss, items = supervised_loss(
                outputs, batch["label"], num_classes, ignore_index, config["loss"], class_weights
            )
            loss = sup_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            maybe_clip_gradients(model.parameters(), config)
            optimizer.step()
            running += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", ce=f"{items['ce']:.4f}")
        scheduler.step()
        avg_loss = running / max(len(loader), 1)
        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.6e}")
        save_checkpoint(last_path, model, optimizer, epoch, avg_loss, config, "MSBaseline")
        if use_val and should_validate_epoch(epoch, config["train"]["epochs"], config):
            print(f"Validation on TE.mat for baseline at epoch {epoch}:")
            metrics, score = evaluate_on_te(model, arrays, config, device, model_kind="baseline")
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    config,
                    "MSBaseline",
                    extra={"validation_metrics": metrics, "selection_metric": metric_name, "selection_score": score},
                )
                print(f"  saved best baseline checkpoint by {metric_name}={score:.6f}: {best_path}")
        elif not use_val and avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(best_path, model, optimizer, epoch, avg_loss, config, "MSBaseline")
            print(f"  saved best baseline checkpoint by train loss: {best_path}")


if __name__ == "__main__":
    main()
