import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import build_model
from utils.checkpoint import _checkpoint, _restore_rng, load_model
from utils.dataset import FaceDataset, NoSingletonBatchSampler, check_split_leakage, read_manifest, seed_worker
from utils.seed import choose_device, seed_everything
from .evaluator import evaluate, make_eval_loader


def step_accumulated(optimizer, model, example_count):
    if example_count < 1:
        raise ValueError("An accumulation window must contain at least one example")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(example_count)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train(config, train_csv, val_csv, output, device="auto", resume=None):
    device = choose_device(device)
    training = config["training"]
    epochs = int(training.get("epochs", 50))
    micro_batch = int(training.get("micro_batch_size", 4))
    accumulate = int(training.get("accumulation_steps", 4))
    workers = int(training.get("workers", 0))
    patience = int(training.get("early_stopping_patience", 10))
    if epochs < 1 or accumulate < 1 or workers < 0 or patience < 0:
        raise ValueError("Invalid training epochs, accumulation, workers or patience")
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    train_records, val_records = read_manifest(train_csv), read_manifest(val_csv)
    check_split_leakage(train_records, val_records)
    sampler = NoSingletonBatchSampler(len(train_records), micro_batch, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(FaceDataset(train_records, config["data"], training=True), batch_sampler=sampler,
                        num_workers=workers, worker_init_fn=seed_worker, generator=generator,
                        pin_memory=device.type == "cuda")
    val_loader = make_eval_loader(val_records, config["data"], micro_batch, workers)
    model = build_model(config["model"]).to(device)
    if model.model_config.get("deploy"):
        raise ValueError("Training requires the unfused model (deploy=false)")
    config = {**config, "model": model.model_config}
    base_lr = float(training.get("learning_rate", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                 betas=tuple(training.get("betas", [0.9, 0.999])),
                                 eps=float(training.get("eps", 1e-8)),
                                 weight_decay=float(training.get("weight_decay", 0)))
    start, best_auc, bad_epochs = 0, -1.0, 0
    if resume:
        restored, saved_config, payload = load_model(resume, device)
        if saved_config != config:
            raise ValueError("Resume config must exactly match the saved configuration")
        if "optimizer" not in payload:
            raise ValueError("Cannot resume a deployment-only checkpoint")
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["epoch"]) + 1
        best_auc, bad_epochs = float(payload["best_auc"]), int(payload["bad_epochs"])
        _restore_rng(payload["rng"], generator)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError("Output already contains last.pt; choose a new directory or use --resume")
    if start >= epochs:
        raise ValueError("Checkpoint already reached the configured epoch count")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"device={device}; nominal effective batch={micro_batch*accumulate}; "
          "final partial accumulation is normalized by its actual image count", flush=True)
    for epoch in range(start, epochs):
        sampler.epoch = epoch
        model.train()
        lr = base_lr * (1 - epoch / epochs) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count, window_count = 0.0, 0, 0
        for batch_index, (images, labels, _) in enumerate(loader):
            logits = model(images.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; optimizer was not stepped")
            loss.backward()
            actual = len(labels)
            loss_sum += loss.item()
            count += actual
            window_count += actual
            if (batch_index + 1) % accumulate == 0 or batch_index + 1 == len(loader):
                # Divide SUM gradients once, including a shorter final accumulation window.
                step_accumulated(optimizer, model, window_count)
                window_count = 0
        metrics = evaluate(model, val_loader, val_records, device)
        improved = metrics["frame_auc"] > best_auc
        best_auc = max(best_auc, metrics["frame_auc"])
        bad_epochs = 0 if improved else bad_epochs + 1
        record = {"epoch": epoch + 1, "learning_rate": lr, "train_loss": loss_sum/count,
                  "train_images": count, **metrics}
        print(json.dumps(record, allow_nan=False), flush=True)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        payload = _checkpoint(model, config, optimizer, epoch, best_auc, bad_epochs, generator)
        torch.save(payload, output / "last.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        if patience and bad_epochs >= patience:
            print(f"Early stopping after {patience} epochs without validation AUC improvement", flush=True)
            break
    return {"best_validation_auc": best_auc, "last_checkpoint": str(output / "last.pt")}
