#!/usr/bin/env python3
"""Run or resume the six preregistered training jobs sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--share-device",
        action="store_true",
        help="Forward --share-device to each run, skipping the exclusive device lock.",
    )
    return parser.parse_args()


def run_is_complete(run_directory: Path, maximum_epochs: int) -> bool:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (
        manifest.get("status") == "complete"
        and manifest.get("completed_epochs") == maximum_epochs
        and manifest.get("maximum_epochs") == maximum_epochs
        and all(value is None for value in manifest.get("development_limits", {}).values())
    )


def main() -> int:
    args = parse_args()
    training_script = Path(__file__).with_name("train_model.py")
    for architecture in ("xresnet1d101", "s4d"):
        for seed in (17, 29, 43):
            run_directory = args.run_root / architecture / f"seed_{seed}"
            if run_is_complete(run_directory, args.max_epochs):
                print(f"complete, skipping {architecture} seed {seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(training_script),
                "--model",
                architecture,
                "--seed",
                str(seed),
                "--max-epochs",
                str(args.max_epochs),
                "--run-root",
                str(args.run_root),
                "--threads",
                str(args.threads),
                "--device",
                args.device,
            ]
            if args.share_device:
                command.append("--share-device")
            print(f"starting {architecture} seed {seed}", flush=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
