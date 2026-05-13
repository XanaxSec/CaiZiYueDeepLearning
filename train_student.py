import argparse
import os

from tqdm import tqdm
import torch

from src.data import compute_class_weights, load_houston_arrays, make_train_loader
from src.losses import distillation_loss, supervised_loss
from src.models.networks import StudentMSMoE, build_teacher_model, get_input_mapping_type, teacher_config_from_checkpoint
from src.train_utils import (
    ValidationScheduler,
    build_optimizer,
    build_scheduler,
    evaluate_on_te,
    maybe_clip_gradients,
    move_batch,
    validation_enabled,
)
from src.utils import count_parameters, get_device, load_config, load_model_state, save_checkpoint, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MS-only MoE student with HS+MS teacher distillation.")
    parser.add_argument("--config", default="configs/houston_moe.yaml")
    parser.add_argument("--teacher", default="checkpoints/teacher_best.pth")
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
    model_cfg = config["model"]

    teacher_ckpt = torch.load(args.teacher, map_location=device)
    teacher_config = teacher_config_from_checkpoint(config, teacher_ckpt if isinstance(teacher_ckpt, dict) else {})
    teacher_mapping_type = get_input_mapping_type(teacher_config.get("model", {}).get("input_mapping"))
    student_mapping_type = get_input_mapping_type(model_cfg.get("input_mapping"))
    if teacher_mapping_type != student_mapping_type:
        raise ValueError(
            "Teacher and student input mappings differ "
            f"({teacher_mapping_type} vs {student_mapping_type}). "
            "Retrain the teacher with the current input_mapping before student distillation."
        )
    teacher = build_teacher_model(
        teacher_config,
        hs_channels=arrays["hs"].shape[2],
        ms_channels=arrays["ms"].shape[2],
    ).to(device)
    load_model_state(teacher, args.teacher, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = StudentMSMoE(
        ms_channels=arrays["ms"].shape[2],
        num_classes=num_classes,
        base_channels=model_cfg["base_channels"],
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        dropout=model_cfg["dropout"],
        input_mapping=model_cfg.get("input_mapping"),
    ).to(device)
    print(f"Training StudentMSMoE on {device}. Trainable parameters: {count_parameters(student):,}")
    print(f"Loaded frozen teacher from {args.teacher}")

    class_weights = None
    if config["loss"].get("use_class_weights", True):
        class_weights = compute_class_weights(arrays["train_label"], num_classes, device)
    optimizer = build_optimizer(student, config)
    scheduler = build_scheduler(optimizer, config)

    ckpt_dir = config["paths"].get("checkpoint_dir", "checkpoints")
    best_path = os.path.join(ckpt_dir, "student_best.pth")
    last_path = os.path.join(ckpt_dir, "student_last.pth")
    val_scheduler = ValidationScheduler(config, total_epochs=config["train"]["epochs"])
    use_val = validation_enabled(config)
    metric_name = val_scheduler.metric
    best_score = float("-inf")
    best_loss = float("inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
        student.train()
        running = 0.0
        progress = tqdm(loader, desc=f"Student epoch {epoch}/{config['train']['epochs']}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            with torch.no_grad():
                teacher_outputs = teacher(batch["hs"], batch["ms"])
            student_outputs = student(batch["ms"])
            sup_loss, sup_items = supervised_loss(
                student_outputs, batch["label"], num_classes, ignore_index, config["loss"], class_weights
            )
            kd_loss, kd_items = distillation_loss(
                student_outputs,
                teacher_outputs,
                batch["label"],
                num_classes,
                ignore_index,
                config["distillation"],
            )
            balance = config["loss"].get("moe_balance", 0.01) * student_outputs["balance_loss"]
            loss = sup_loss + kd_loss + balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            maybe_clip_gradients(student.parameters(), config)
            optimizer.step()
            running += float(loss.detach())
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                ce=f"{sup_items['ce']:.4f}",
                kd=f"{float(kd_loss.detach()):.4f}",
                router=f"{kd_items['kd_router']:.4f}",
            )
        scheduler.step()
        avg_loss = running / max(len(loader), 1)
        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.6e}")
        save_checkpoint(last_path, student, optimizer, epoch, avg_loss, config, "StudentMSMoE")
        if use_val and val_scheduler.should_validate(epoch):
            print(f"Validation on TE.mat for student at epoch {epoch}:")
            metrics, score = evaluate_on_te(student, arrays, config, device, model_kind="student")
            val_scheduler.update(epoch, metrics)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_path,
                    student,
                    optimizer,
                    epoch,
                    avg_loss,
                    config,
                    "StudentMSMoE",
                    extra={"validation_metrics": metrics, "selection_metric": metric_name, "selection_score": score},
                )
                print(f"  saved best student checkpoint by {metric_name}={score:.6f}: {best_path}")
        elif not use_val and avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(best_path, student, optimizer, epoch, avg_loss, config, "StudentMSMoE")
            print(f"  saved best student checkpoint by train loss: {best_path}")


if __name__ == "__main__":
    main()
