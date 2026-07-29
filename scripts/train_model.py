#!/usr/bin/env python3
"""Train one preregistered architecture/seed using only PTB-XL folds 1-9."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_clinical.data import (
    InferenceRecordDataset,
    TrainingCropDataset,
    WaveformStore,
    load_normalization,
)
from ecg_clinical.engine import predict_records
from ecg_clinical.integrity import verify_preregistration_seal
from ecg_clinical.locking import exclusive_device_lock
from ecg_clinical.metrics import binary_negative_log_likelihood, discrimination_summary
from ecg_clinical.models import build_model, trainable_parameter_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xresnet1d101", "s4d"], required=True)
    parser.add_argument("--seed", type=int, choices=[17, 29, 43], required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--store", type=Path, default=Path("data/cache/waveforms/ptb_xl"))
    parser.add_argument(
        "--normalization", type=Path, default=Path("data/derived/training_normalization.json")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/preregistered_protocol.json")
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--share-device",
        action="store_true",
        help=(
            "Run without taking the exclusive device lock, sharing the accelerator with "
            "other jobs. Off by default, and appropriate only when the concurrent memory "
            "and throughput cost is acceptable. The run is otherwise identical and its "
            "results are unaffected."
        ),
    )
    return parser.parse_args()


def optimizer_and_scheduler(
    model_name: str,
    model: nn.Module,
    protocol: dict[str, Any],
    maximum_epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    model_protocol = protocol["models"][model_name]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_protocol.get("maximum_learning_rate", model_protocol.get("learning_rate"))),
        weight_decay=float(model_protocol["weight_decay"]),
    )
    if model_name == "xresnet1d101":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(model_protocol["maximum_learning_rate"]),
                epochs=maximum_epochs,
                steps_per_epoch=steps_per_epoch,
            )
        )
    else:
        scheduler = None
    return optimizer, scheduler


def main() -> int:
    args = parse_args()
    if args.max_epochs < 1 or args.max_epochs > 50:
        raise ValueError("--max-epochs must be between 1 and the preregistered maximum of 50")
    root = Path(__file__).resolve().parents[1]
    verify_preregistration_seal(root)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    protocol = json.loads(args.protocol.read_text())
    labels = json.loads(args.label_manifest.read_text())
    label_keys = [label["label_key"] for label in labels["labels"]]
    headline_indices = np.asarray(
        [label_keys.index(key) for key in labels["headline_label_keys"]], dtype=np.int64
    )
    mean, standard_deviation = load_normalization(args.normalization)
    store = WaveformStore(args.store)
    train_indices = store.indices_for_folds(tuple(range(1, 9)))
    validation_indices = store.indices_for_folds((9,))
    if args.train_limit is not None:
        train_indices = train_indices[: args.train_limit]
    if args.validation_limit is not None:
        validation_indices = validation_indices[: args.validation_limit]

    run_directory = args.run_root / args.model / f"seed_{args.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_directory / "train.log"),
        ],
    )
    device = choose_device(args.device)
    model = build_model(args.model).to(device)
    batch_size = int(protocol["models"][args.model]["batch_size"])
    if len(train_indices) < batch_size:
        batch_size = max(2, len(train_indices))
    steps_per_epoch = (len(train_indices) + batch_size - 1) // batch_size
    optimizer, scheduler = optimizer_and_scheduler(
        args.model, model, protocol, args.max_epochs, steps_per_epoch
    )
    loss_function = nn.BCEWithLogitsLoss()

    manifest = {
        "schema_version": 1,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "trainable_parameters": trainable_parameter_count(model),
        "maximum_epochs": args.max_epochs,
        "training_records": len(train_indices),
        "validation_records": len(validation_indices),
        "batch_size": batch_size,
        "protocol_sha256": sha256(args.protocol),
        "label_manifest_sha256": sha256(args.label_manifest),
        "normalization_sha256": sha256(args.normalization),
        "protected_evaluation_accessed": False,
        "development_limits": {
            "train_limit": args.train_limit,
            "validation_limit": args.validation_limit,
        },
    }
    atomic_json(manifest, run_directory / "run_manifest.json")

    training_dataset = TrainingCropDataset(
        store, train_indices, mean, standard_deviation, args.seed
    )
    validation_dataset = InferenceRecordDataset(store, validation_indices, mean, standard_deviation)
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_epoch = -1
    best_auroc = -float("inf")
    best_nll = float("inf")
    last_checkpoint = run_directory / "last.pt"
    if last_checkpoint.exists() and not args.no_resume:
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if checkpoint["maximum_epochs"] != args.max_epochs:
            raise ValueError("cannot resume with a different maximum epoch count")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_auroc = float(checkpoint["best_auroc"])
        best_nll = float(checkpoint["best_nll"])
        logging.info("resuming at epoch %d", start_epoch + 1)

    inference_record_batch = max(1, batch_size // 10)
    # A fully trained run that only needs its manifest finalized does no GPU work,
    # so it must not wait on the shared device lock.
    training_remaining = args.max_epochs - start_epoch
    take_lock = training_remaining > 0 and not args.share_device
    device_lock: AbstractContextManager[None] = (
        exclusive_device_lock(Path(protocol["training"]["training_lock"]))
        if take_lock
        else nullcontext()
    )
    if training_remaining <= 0:
        logging.info("no epochs remain; finalizing without acquiring the device lock")
    elif args.share_device:
        logging.info("share-device requested; training without the exclusive device lock")
    with device_lock:
        for epoch in range(start_epoch, args.max_epochs):
            epoch_started = time.monotonic()
            training_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(args.seed * 10_000 + epoch)
            training_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.num_workers,
                drop_last=False,
            )
            model.train()
            running_loss = 0.0
            seen_records = 0
            for batch_index, (inputs, targets) in enumerate(training_loader, start=1):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(inputs), targets)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                running_loss += float(loss.detach()) * len(inputs)
                seen_records += len(inputs)
                if batch_index % 50 == 0:
                    logging.info(
                        "epoch %d train batch %d/%d loss %.5f",
                        epoch + 1,
                        batch_index,
                        len(training_loader),
                        running_loss / seen_records,
                    )

            probabilities, targets, record_indices = predict_records(
                model,
                validation_dataset,
                device=device,
                record_batch_size=inference_record_batch,
                num_workers=args.num_workers,
                progress=lambda current, total, current_epoch=epoch: logging.info(
                    "epoch %d validation batch %d/%d", current_epoch + 1, current, total
                )
                if current % 25 == 0 or current == total
                else None,
            )
            summary = discrimination_summary(targets, probabilities, headline_indices)
            validation_nll = binary_negative_log_likelihood(targets, probabilities)
            train_loss = running_loss / seen_records
            elapsed_seconds = time.monotonic() - epoch_started
            epoch_result: dict[str, float | int] = {
                "epoch": epoch + 1,
                "training_loss": train_loss,
                "validation_macro_auroc": summary["macro_auroc"],
                "validation_macro_auprc": summary["macro_auprc"],
                "validation_nll": validation_nll,
                "elapsed_seconds": elapsed_seconds,
            }
            history.append(epoch_result)
            improved = summary["macro_auroc"] > best_auroc or (
                summary["macro_auroc"] == best_auroc and validation_nll < best_nll
            )
            if improved:
                best_epoch = epoch
                best_auroc = summary["macro_auroc"]
                best_nll = validation_nll
                atomic_torch_save(
                    {
                        "model": args.model,
                        "seed": args.seed,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "validation_macro_auroc": best_auroc,
                        "validation_nll": best_nll,
                        "run_manifest": manifest,
                    },
                    run_directory / "best.pt",
                )
                np.savez_compressed(
                    run_directory / "best_validation_predictions.npz",
                    probabilities=probabilities,
                    targets=targets,
                    record_indices=record_indices,
                )
            checkpoint_payload = {
                "maximum_epochs": args.max_epochs,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_epoch": best_epoch,
                "best_auroc": best_auroc,
                "best_nll": best_nll,
            }
            atomic_torch_save(checkpoint_payload, last_checkpoint)
            atomic_json({"epochs": history}, run_directory / "history.json")
            logging.info(
                "epoch %d complete train_loss=%.5f val_auc=%.5f val_nll=%.5f elapsed=%.1fs%s",
                epoch + 1,
                train_loss,
                summary["macro_auroc"],
                validation_nll,
                elapsed_seconds,
                " best" if improved else "",
            )

    manifest["completed_epochs"] = args.max_epochs
    manifest["best_epoch"] = best_epoch + 1
    manifest["best_validation_macro_auroc"] = best_auroc
    manifest["best_validation_nll"] = best_nll
    manifest["status"] = "complete"
    atomic_json(manifest, run_directory / "run_manifest.json")
    shutil.copy2(
        run_directory / "best_validation_predictions.npz", run_directory / "validation.npz"
    )
    logging.info("training complete: %s", run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
