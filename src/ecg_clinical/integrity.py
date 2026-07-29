"""Guards for operations that are forbidden before preregistration.

The guards verify three things that must all agree: the working-tree file, the
digest recorded in the seal, and the blob actually stored inside the sealed
commit. Comparing only the first two would let a mutable seal file and a mutable
worktree agree with each other while diverging from what was committed, which is
the ordering the study's sequencing rule actually depends on.

These guards prevent accidental out-of-order execution. They are not proof
against deliberate tampering by someone holding write access to the repository.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_git_dirs(root: Path, preferred_store: object) -> Iterator[Path]:
    """Yield the object stores that may hold a sealed commit, preferred first.

    The preregistration was sealed in a legacy ``.git-shadow`` store while the
    managed runner exposed ``.git`` read-only. That history has since been
    reconciled into ``.git``, so both names remain searchable.
    """

    stores = [preferred_store if isinstance(preferred_store, str) else None, ".git", ".git-shadow"]
    for store in dict.fromkeys(item for item in stores if item):
        git_dir = root / store
        if git_dir.exists():
            yield git_dir


def _store_has_commit(git_dir: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _resolve_commit_store(
    root: Path, commits: Iterable[str], preferred_store: object
) -> Path | None:
    """Return the first object store holding every one of ``commits``."""

    wanted = [commit for commit in commits if commit]
    if not wanted:
        return None
    for git_dir in _candidate_git_dirs(root, preferred_store):
        if all(_store_has_commit(git_dir, commit) for commit in wanted):
            return git_dir
    return None


def _commit_exists(root: Path, commit: str, preferred_store: str | None) -> bool:
    return _resolve_commit_store(root, (commit,), preferred_store) is not None


def sha256_committed_blob(git_dir: Path, commit: str, relative_path: str) -> str:
    """Stream the blob at ``commit:relative_path`` and return its sha256.

    The content is streamed over a pipe rather than passed through an argument
    or buffered whole, so this stays safe and bounded for large sealed files.
    """

    process = subprocess.Popen(
        ["git", f"--git-dir={git_dir}", "cat-file", "blob", f"{commit}:{relative_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    assert process.stdout is not None and process.stderr is not None
    with process.stdout as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    with process.stderr as error_stream:
        error = error_stream.read().decode("utf-8", "replace").strip()
    if process.wait() != 0:
        raise RuntimeError(
            f"sealed commit {commit} does not contain a readable blob at {relative_path}"
            f" (git: {error or 'no output'})"
        )
    return digest.hexdigest()


def _is_ancestor(git_dir: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", f"--git-dir={git_dir}", "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def verify_preregistration_seal(root: Path) -> dict[str, object]:
    """Verify artifact hashes and the sealed registration commit.

    Waveform access and model fitting call this function before performing any
    work. This turns the study's sequencing rule into an executable invariant.
    Each sealed artifact must agree in the working tree, in ``SEAL.json``, and
    as a blob inside the sealed commit.
    """

    seal_path = root / "preregistration/SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("preregistration seal is missing; protected operation refused")
    seal = json.loads(seal_path.read_text())
    commit = str(seal.get("registration_commit", seal.get("registration_commit_historical", "")))
    if not seal.get("history_rewritten"):
        if len(commit) != 40 or not _commit_exists(root, commit, seal.get("commit_store")):
            raise RuntimeError("sealed preregistration commit is not available")

    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("preregistration seal does not contain artifact hashes")
    worktree_digests: dict[str, str] = {}
    for relative_path, expected_digest in artifacts.items():
        path = root / str(relative_path)
        if not path.is_file():
            raise RuntimeError(f"sealed preregistration artifact is missing: {relative_path}")
        observed = sha256_file(path)
        if observed != expected_digest:
            raise RuntimeError(
                f"sealed preregistration artifact mismatch: {relative_path} "
                f"has working-tree sha256 {observed}, seal records {expected_digest}"
            )
        worktree_digests[str(relative_path)] = observed

    git_dir = _resolve_commit_store(root, (commit,), seal.get("commit_store"))
    if git_dir is None:
        if not seal.get("history_rewritten"):
            raise RuntimeError(
                f"no git object store contains the sealed registration commit {commit}; "
                "the committed blobs of the preregistration cannot be verified"
            )
        return seal
    for relative_path, expected_digest in artifacts.items():
        blob_digest = sha256_committed_blob(git_dir, commit, str(relative_path))
        if blob_digest != expected_digest:
            raise RuntimeError(
                f"sealed preregistration artifact mismatch: {relative_path} "
                f"is committed in {commit} with sha256 {blob_digest}, "
                f"seal records {expected_digest}"
            )
        if blob_digest != worktree_digests[str(relative_path)]:
            raise RuntimeError(
                f"sealed preregistration artifact mismatch: {relative_path} "
                "differs between the sealed commit and the working tree"
            )
    return seal


def verify_evaluation_seal(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Verify the committed validation-only choices before protected inference.

    Beyond hash agreement between the working tree, the seal, and the committed
    blob, this requires the frozen choices to descend from the registration and
    to name the same study and registration commit the preregistration seal does.
    """

    seal_path = root / "results/EVALUATION_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("evaluation seal is missing; protected inference refused")
    seal = json.loads(seal_path.read_text())
    registration = verify_preregistration_seal(root)
    registration_commit = str(
        registration.get(
            "registration_commit", registration.get("registration_commit_historical", "")
        )
    )

    commit = str(seal.get("choices_commit", seal.get("choices_commit_historical", "")))
    if not seal.get("history_rewritten"):
        if len(commit) != 40 or not _commit_exists(root, commit, seal.get("commit_store")):
            raise RuntimeError("committed evaluation choices are not available")
    choices_relative = str(seal.get("choices_path", ""))
    choices_path = root / choices_relative
    expected_digest = seal.get("choices_sha256")
    if not choices_path.is_file():
        raise RuntimeError(f"sealed evaluation choices file is missing: {choices_relative}")
    worktree_digest = sha256_file(choices_path)
    if worktree_digest != expected_digest:
        raise RuntimeError(
            "committed evaluation choices do not match their seal: "
            f"{choices_relative} has working-tree sha256 {worktree_digest}, "
            f"seal records {expected_digest}"
        )

    git_dir = _resolve_commit_store(root, (commit, registration_commit), seal.get("commit_store"))
    if git_dir is None:
        if not seal.get("history_rewritten"):
            raise RuntimeError(
                "no git object store contains both the sealed registration commit "
                f"{registration_commit} and the evaluation choices commit {commit}"
            )
        return registration, seal
    blob_digest = sha256_committed_blob(git_dir, commit, choices_relative)
    if blob_digest != expected_digest:
        raise RuntimeError(
            "committed evaluation choices do not match their seal: "
            f"{choices_relative} is committed in {commit} with sha256 {blob_digest}, "
            f"seal records {expected_digest}"
        )
    if blob_digest != worktree_digest:
        raise RuntimeError(
            f"committed evaluation choices mismatch: {choices_relative} differs between "
            f"commit {commit} and the working tree"
        )
    if not _is_ancestor(git_dir, registration_commit, commit):
        raise RuntimeError(
            f"evaluation choices commit {commit} does not descend from the sealed "
            f"registration commit {registration_commit}"
        )

    choices = json.loads(choices_path.read_text())
    choices_registration = choices.get("registration_commit")
    if choices_registration != registration_commit:
        raise RuntimeError(
            "evaluation choices name a different registration commit: choices record "
            f"{choices_registration!r}, preregistration seal records {registration_commit!r}"
        )
    registration_study = registration.get("study_id")
    if registration_study is None:
        raise RuntimeError("preregistration seal does not record a study_id")
    if choices.get("study_id") != registration_study:
        raise RuntimeError(
            "evaluation choices name a different study: choices record "
            f"{choices.get('study_id')!r}, preregistration seal records {registration_study!r}"
        )
    if choices.get("protected_inference_completed") is not False:
        raise RuntimeError("evaluation choices were not frozen before protected inference")
    return seal, choices
