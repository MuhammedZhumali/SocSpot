"""Reproducible small-data fine-tuning; test is evaluated after model selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18

from .metrics import classification_metrics, select_thresholds
from .cache import prepare_cache
from .model import autocast, tensor_from_frames
from .video import LABELS, PREPROCESS, ROOT, duration, sample_frames, sha256, write_json

SPLITS = ROOT / "data/splits/initial_source_holdout"


def read_splits(directory=None):
    directory = Path(directory) if directory is not None else SPLITS
    documents = {part: json.loads((directory/f"{part}.json").read_text(encoding="utf-8"))
                 for part in ("train", "validation", "test")}
    owners, events, ids = {}, set(), set()
    for part, doc in documents.items():
        if not doc["clips"]:
            raise ValueError(f"Empty {part}")
        for c in doc["clips"]:
            group = c.get("evaluation_group", c["event_id"])
            if group in owners and owners[group] != part:
                raise ValueError("Evaluation group leakage")
            owners[group] = part
            if c["clip_id"] in ids or c["event_id"] in events:
                raise ValueError("Duplicate clip or known event across manifests")
            ids.add(c["clip_id"])
            events.add(c["event_id"])
            if c["label"] not in LABELS or not (ROOT/c["path"]).is_file():
                raise ValueError(f"Invalid training input: {c['clip_id']}")
    return documents


class Clips(Dataset):
    def __init__(self, rows, cache, augment=False):
        self.rows, self.cache, self.augment = rows, cache, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frames = np.load(self.cache[row["clip_id"]], allow_pickle=False)
        return tensor_from_frames(frames, self.augment), LABELS.index(row["label"])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    targets, probabilities, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device):
            logits = model(x)
        losses.extend(torch.nn.functional.cross_entropy(logits.float(), y, reduction="none").cpu().tolist())
        probabilities.extend(logits.float().softmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    metrics = classification_metrics(targets, np.asarray(probabilities).argmax(1))
    metrics["loss"] = float(np.mean(losses))
    return metrics, targets, probabilities


def train(args):
    if args.epochs <= args.warmup_epochs or args.batch_size < 1:
        raise ValueError("Need fine-tuning epochs after warmup and a positive batch size")
    run_dir = ROOT / "models" / args.run
    report_dir = ROOT / "data/reports" / args.run
    if (run_dir/"best.pt").exists() or (report_dir/"report.json").exists():
        raise ValueError("Run already exists; choose a new --run to preserve its test result")
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.hub.set_dir(str(ROOT/".tools/torch"))
    split_directory = ROOT / args.splits
    documents = read_splits(split_directory)
    hashes = {part: sha256(split_directory/f"{part}.json") for part in documents}
    write_json(report_dir/"config.json", {
        **vars(args), "preprocess": PREPROCESS, "labels": LABELS, "split_sha256": hashes,
        "selection": "best validation macro-F1, tie-break by validation loss",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "weights": "R3D_18_Weights.KINETICS400_V1", "torch": torch.__version__,
        "torchvision": torchvision.__version__, "labels_review_complete": False,
    })
    print(f"Device: {device}; preparing training/validation frame cache", flush=True)
    cache = prepare_cache(documents["train"]["clips"] + documents["validation"]["clips"])
    loaders = {part: DataLoader(Clips(documents[part]["clips"], cache, part == "train"),
                                batch_size=args.batch_size, shuffle=part == "train", num_workers=0)
               for part in ("train", "validation")}
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": .001},
        {"params": model.layer4.parameters(), "lr": .0001},
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = np.bincount([LABELS.index(r["label"]) for r in documents["train"]["clips"]], minlength=len(LABELS))
    weights = torch.tensor(counts.sum()/(len(LABELS)*counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_key, best_epoch, history, stale = (-1.0, -math.inf), 0, [], 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        phase = "head" if epoch <= args.warmup_epochs else "layer4_and_head"
        if epoch == args.warmup_epochs+1:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        model.train()
        # Small batches: preserve pretrained batch-normalization statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                module.eval()
        losses, begin = [], time.perf_counter()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        metrics, _, _ = evaluate(model, loaders["validation"], device)
        key = (metrics["macro_f1"], -metrics["loss"])
        improved = key > best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "labels": LABELS, "preprocess": PREPROCESS, "epoch": epoch,
                        "thresholds": {label:.7 for label in LABELS[:2]}, "split_sha256": hashes}, run_dir/"best.pt")
        else:
            stale += 1
        entry = {"epoch": epoch, "phase": phase, "training_loss": float(np.mean(losses)),
                 "validation": metrics, "selected": improved, "elapsed_sec": time.perf_counter()-begin}
        history.append(entry)
        write_json(report_dir/"history.json", history)
        print(f"Epoch {epoch}/{args.epochs} [{phase}] train_loss={entry['training_loss']:.4f} "
              f"val_f1={metrics['macro_f1']:.3f} val_accuracy={metrics['accuracy']:.3f} "
              f"seconds={entry['elapsed_sec']:.1f} best={best_epoch}", flush=True)
        if stale >= args.patience and epoch > args.warmup_epochs:
            break
    saved = torch.load(run_dir/"best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation, vy, vp = evaluate(model, loaders["validation"], device)
    saved["thresholds"] = select_thresholds(vy, vp)
    torch.save(saved, run_dir/"best.pt")
    # First use of test pixels and labels for evaluating a selected model.
    print("Model selected. Evaluating held-out test once.", flush=True)
    test_rows = documents["test"]["clips"]
    test_cache = prepare_cache(test_rows)
    test_loader = DataLoader(Clips(test_rows, test_cache), batch_size=args.batch_size, shuffle=False)
    test, ty, tp = evaluate(model, test_loader, device)
    majority = int(counts.argmax())
    predictions = [{"clip_id": row["clip_id"], "path": row["path"], "source_id": row["source_id"],
                    "true_label": LABELS[truth], "predicted_label": LABELS[int(np.argmax(prob))],
                    "scores": dict(zip(LABELS, prob)), "correct": truth == int(np.argmax(prob))}
                   for row, truth, prob in zip(test_rows, ty, tp, strict=True)]
    write_json(report_dir/"test_predictions.json", predictions)
    report = {"status": "experimental", "run": args.run, "best_epoch": best_epoch,
              "checkpoint": (run_dir/"best.pt").relative_to(ROOT).as_posix(),
              "model_sha256": sha256(run_dir/"best.pt"), "split_sha256": hashes,
              "validation": validation, "test": test, "thresholds": saved["thresholds"],
              "majority_baseline_test_accuracy": sum(t == majority for t in ty)/len(ty),
              "elapsed_training_and_evaluation_sec": time.perf_counter()-started,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0,
              "limitations": ["Provisional frame-based labels; full human review pending.",
                              "Source/style shortcuts and unidentified compilation repeats remain possible.",
                              "Clip classification metrics do not measure event detection in full videos.",
                              f"Validation-selected thresholds are based on {len(documents['validation']['clips'])} clips; scores are uncalibrated.",
                              "Previously evaluated holdouts do not provide a fresh final estimate after revisions.",
                              "HF training addition is restricted to non-commercial research."]}
    write_json(report_dir/"report.json", report)
    write_json(ROOT/"models/current.json", {"checkpoint": report["checkpoint"],
                                            "report": (report_dir/"report.json").relative_to(ROOT).as_posix()})
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="r3d18-v1")
    parser.add_argument("--splits", default="data/splits/review_v2_source_holdout",
                        help="Manifest directory; use initial_source_holdout to reproduce v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if not args.run.replace("-", "").replace("_", "").isalnum():
        parser.error("--run must be a simple directory name")
    train(args)


if __name__ == "__main__":
    main()
