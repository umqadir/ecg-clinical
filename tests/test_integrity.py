import json
from pathlib import Path

import pytest
from conftest import (
    CHOICES_PATH,
    PROTOCOL_ARTIFACT,
    SealedStudy,
    build_sealed_study,
    commit_all,
    default_choices,
    sha256_text,
    write,
)

from ecg_clinical.integrity import (
    sha256_committed_blob,
    sha256_file,
    verify_evaluation_seal,
    verify_preregistration_seal,
)


def test_repository_preregistration_seal_is_valid() -> None:
    root = Path(__file__).parents[1]

    seal = verify_preregistration_seal(root)

    assert seal["registration_commit_historical"] == "f34661970a4bdf9986b9cf006f6bd8713f017fff"
    assert seal["history_rewritten"] is True
    assert seal["artifacts"]


def test_modified_sealed_artifact_is_rejected(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "protocol.json"
    artifact.write_text("original")
    seal_dir = tmp_path / "preregistration"
    seal_dir.mkdir()
    (seal_dir / "SEAL.json").write_text(
        json.dumps(
            {
                "registration_commit": "a" * 40,
                "commit_store": ".git-shadow",
                "artifacts": {"protocol.json": sha256_file(artifact)},
            }
        )
    )
    monkeypatch.setattr("ecg_clinical.integrity._commit_exists", lambda *args: True)
    artifact.write_text("modified")

    with pytest.raises(RuntimeError, match="artifact mismatch"):
        verify_preregistration_seal(tmp_path)


def test_missing_evaluation_seal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="evaluation seal is missing"):
        verify_evaluation_seal(tmp_path)


def test_synthetic_sealed_study_verifies(sealed_study: SealedStudy) -> None:
    seal, choices = verify_evaluation_seal(sealed_study.root)

    assert seal["choices_commit"] == sealed_study.choices_commit
    assert choices["registration_commit"] == sealed_study.registration_commit


def test_preregistration_blob_differing_from_worktree_is_rejected(tmp_path: Path) -> None:
    """The seal and the worktree agree with each other but not with the commit."""

    study = build_sealed_study(tmp_path)
    superseded_commit = study.registration_commit
    replacement = json.dumps({"protocol": "silently rewritten"}, indent=2)
    write(tmp_path, PROTOCOL_ARTIFACT, replacement)
    commit_all(tmp_path, "rewrite the protocol after sealing")

    # Seal and worktree both carry the replacement; the sealed commit does not.
    study.protocol_content = replacement
    study.write_preregistration_seal(registration_commit=superseded_commit)

    with pytest.raises(RuntimeError, match="is committed in .* seal records"):
        verify_preregistration_seal(tmp_path)


def test_evaluation_choices_blob_differing_from_worktree_is_rejected(
    sealed_study: SealedStudy,
) -> None:
    sealed_commit = sealed_study.choices_commit
    tampered = json.dumps(default_choices(sealed_study.registration_commit), indent=2) + "\n"
    write(sealed_study.root, CHOICES_PATH, tampered)
    commit_all(sealed_study.root, "reformat the frozen choices")
    # Re-seal against the reformatted worktree file while still naming the
    # original commit, so only the committed blob disagrees.
    sealed_study.write_evaluation_seal(choices_commit=sealed_commit)

    with pytest.raises(RuntimeError, match="is committed in .* seal records"):
        verify_evaluation_seal(sealed_study.root)


def test_non_descendant_choices_commit_is_rejected(tmp_path: Path) -> None:
    build_sealed_study(tmp_path, orphan_choices=True)

    with pytest.raises(RuntimeError, match="does not descend from the sealed registration commit"):
        verify_evaluation_seal(tmp_path)


def test_choices_naming_a_different_registration_commit_are_rejected(tmp_path: Path) -> None:
    build_sealed_study(tmp_path, choices_overrides={"registration_commit": "b" * 40})

    with pytest.raises(RuntimeError, match="different registration commit"):
        verify_evaluation_seal(tmp_path)


def test_choices_naming_a_different_study_are_rejected(tmp_path: Path) -> None:
    build_sealed_study(tmp_path, choices_overrides={"study_id": "some-other-study"})

    with pytest.raises(RuntimeError, match="different study"):
        verify_evaluation_seal(tmp_path)


def test_choices_written_after_inference_are_rejected(tmp_path: Path) -> None:
    build_sealed_study(tmp_path, choices_overrides={"protected_inference_completed": True})

    with pytest.raises(RuntimeError, match="not frozen before protected inference"):
        verify_evaluation_seal(tmp_path)


def test_worktree_choices_differing_from_the_seal_are_rejected(sealed_study: SealedStudy) -> None:
    path = sealed_study.root / CHOICES_PATH
    path.write_text(path.read_text() + "\n")

    with pytest.raises(RuntimeError, match="working-tree sha256"):
        verify_evaluation_seal(sealed_study.root)


def test_unknown_registration_commit_is_rejected(sealed_study: SealedStudy) -> None:
    sealed_study.write_preregistration_seal(registration_commit="c" * 40)

    with pytest.raises(RuntimeError, match="sealed preregistration commit is not available"):
        verify_preregistration_seal(sealed_study.root)


def test_committed_blob_digest_matches_the_file(sealed_study: SealedStudy) -> None:
    digest = sha256_committed_blob(
        sealed_study.root / ".git", sealed_study.registration_commit, PROTOCOL_ARTIFACT
    )

    assert digest == sha256_text(sealed_study.protocol_content)


def test_missing_committed_blob_is_reported(sealed_study: SealedStudy) -> None:
    with pytest.raises(RuntimeError, match="does not contain a readable blob"):
        sha256_committed_blob(
            sealed_study.root / ".git", sealed_study.registration_commit, "results/nothing.json"
        )
