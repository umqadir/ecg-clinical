"""Builders for tiny synthetic sealed repositories and fake waveform stores.

The integrity guards are only meaningful against real git objects, so these
helpers create throwaway repositories in ``tmp_path`` rather than mocking git.
Nothing here reads the project's own 1.6 GB waveform stores.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

STUDY_ID = "ecg-clinical-shift-v1"
PROTOCOL_ARTIFACT = "configs/preregistered_protocol.json"
CHOICES_PATH = "results/evaluation_choices.json"

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name: str) -> ModuleType:
    """Import one of the pipeline scripts as a module for direct unit testing."""

    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that module-level dataclasses can resolve
    # their own module while the script body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def init_repository(root: Path) -> None:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Integrity Test")
    git(root, "config", "commit.gpgsign", "false")


def write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


@dataclass
class SealedStudy:
    root: Path
    registration_commit: str
    choices_commit: str
    protocol_content: str
    choices: dict

    def write_preregistration_seal(self, **overrides: object) -> None:
        seal = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "registration_commit": self.registration_commit,
            "commit_store": ".git",
            "artifacts": {PROTOCOL_ARTIFACT: sha256_text(self.protocol_content)},
        }
        seal.update(overrides)
        write(self.root, "preregistration/SEAL.json", json.dumps(seal, indent=2))

    def write_evaluation_seal(self, **overrides: object) -> None:
        digest = sha256_text((self.root / CHOICES_PATH).read_text())
        seal = {
            "schema_version": 1,
            "choices_commit": self.choices_commit,
            "choices_path": CHOICES_PATH,
            "choices_sha256": digest,
            "commit_store": ".git",
        }
        seal.update(overrides)
        write(self.root, "results/EVALUATION_SEAL.json", json.dumps(seal, indent=2))


def default_choices(registration_commit: str) -> dict:
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "registration_commit": registration_commit,
        "protected_inference_completed": False,
        "architectures": {},
    }


def build_sealed_study(
    root: Path,
    *,
    choices_overrides: dict | None = None,
    orphan_choices: bool = False,
) -> SealedStudy:
    """Create a repository whose preregistration and evaluation seals both verify."""

    init_repository(root)
    protocol_content = json.dumps({"protocol": "frozen"}, indent=2)
    write(root, PROTOCOL_ARTIFACT, protocol_content)
    registration_commit = commit_all(root, "seal the preregistration")

    if orphan_choices:
        # An unrelated root commit: it contains the same worktree content but is
        # not a descendant of the registration commit.
        git(root, "checkout", "-q", "--orphan", "unrelated")

    choices = default_choices(registration_commit)
    choices.update(choices_overrides or {})
    write(root, CHOICES_PATH, json.dumps(choices, indent=2, sort_keys=True) + "\n")
    choices_commit = commit_all(root, "freeze the evaluation choices")

    study = SealedStudy(
        root=root,
        registration_commit=registration_commit,
        choices_commit=choices_commit,
        protocol_content=protocol_content,
        choices=choices,
    )
    study.write_preregistration_seal()
    study.write_evaluation_seal()
    return study


@pytest.fixture
def sealed_study(tmp_path: Path) -> SealedStudy:
    return build_sealed_study(tmp_path)


def write_fake_store(
    root: Path,
    *,
    records: int,
    folds: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Write a metadata/status/targets store with no signal matrix.

    ``signals.npy`` is written only when a test needs a digest for it; the
    validation-only code paths under test never open it.
    """

    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    if folds is None:
        folds = np.full(records, 1, dtype=np.int64)
    if targets is None:
        rng = np.random.default_rng(7)
        targets = rng.integers(0, 2, size=(records, 16)).astype(np.uint8)
    metadata = pd.DataFrame(
        {
            "record_id": [f"R{index:05d}" for index in range(records)],
            "patient_id": [str(index) for index in range(records)],
            "strat_fold": np.asarray(folds, dtype=np.int64),
        }
    )
    metadata.to_parquet(root / "metadata.parquet")
    np.save(root / "status.npy", np.ones(records, dtype=np.uint8))
    np.save(root / "targets.npy", targets)
    return {"targets": targets, "folds": np.asarray(folds, dtype=np.int64)}
