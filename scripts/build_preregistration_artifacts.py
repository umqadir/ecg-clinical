#!/usr/bin/env python3
"""Generate the frozen harmonized-label audit from waveform-free metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ecg_clinical.harmonization import load_scored_groups, positive_mask

EXPECTED_HEADER_COUNTS = {"ptb-xl": 21_837, "chapman_shaoxing": 10_247, "ningbo": 34_905}
EVALUATION_COHORTS = ("ptb_validation", "ptb_test", "chapman_shaoxing", "ningbo")
SEMANTIC_EXCLUSIONS = {
    "251146004": (
        "cross-hospital source mappings conflate low QRS voltage and poor R-wave progression"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--ptb-database", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scored-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-positives", type=int, default=100)
    parser.add_argument("--min-eval-positives", type=int, default=25)
    parser.add_argument("--min-eval-negatives", type=int, default=25)
    parser.add_argument("--min-subgroup-positives", type=int, default=10)
    parser.add_argument("--min-subgroup-negatives", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sex(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"0", "female", "f"}:
        return "female"
    if text in {"1", "male", "m"}:
        return "male"
    return "unknown"


def normalize_age(value: object) -> float | None:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(age) or age < 0:
        return None
    return age


def age_group(value: object) -> str:
    age = normalize_age(value)
    if age is None:
        return "unknown"
    if age < 40:
        return "under_40"
    if age < 65:
        return "40_to_64"
    return "65_plus"


def cohort_summary(frame: pd.DataFrame) -> dict[str, object]:
    ages = frame["age"].map(normalize_age)
    sexes = frame["sex"].map(normalize_sex)
    return {
        "records": int(len(frame)),
        "age_available": int(ages.notna().sum()),
        "age_missing": int(ages.isna().sum()),
        "sex_counts": {str(key): int(value) for key, value in sexes.value_counts().items()},
        "sampling_frequency_hz": sorted(
            float(value) for value in frame.sampling_frequency_hz.unique()
        ),
        "num_leads": sorted(int(value) for value in frame.num_leads.unique()),
        "duration_seconds": sorted(
            float(value)
            for value in (frame.num_samples / frame.sampling_frequency_hz).round(6).unique()
        ),
    }


def main() -> int:
    args = parse_args()
    headers = pd.read_csv(args.headers, dtype={"record_id": str, "diagnoses": str})
    observed = headers.cohort.value_counts().to_dict()
    if observed != EXPECTED_HEADER_COUNTS:
        raise ValueError(f"unexpected header cohort counts: {observed}")
    if headers.record_id.duplicated().any():
        raise ValueError("record IDs are not globally unique")

    ptb_database = pd.read_csv(args.ptb_database)
    if len(ptb_database) != 21_799 or ptb_database.patient_id.nunique() != 18_869:
        raise ValueError("unexpected PTB-XL v1.0.3 database dimensions")

    ptb_headers = headers[headers.cohort == "ptb-xl"].copy()
    matches = ptb_headers.record_id.str.extract(r"^HR(\d+)$", expand=False)
    if matches.isna().any():
        raise ValueError("unexpected PTB-XL challenge record ID")
    ptb_headers["ecg_id"] = matches.astype(int)
    merged = ptb_database.merge(
        ptb_headers[["ecg_id", "diagnoses", "record_id"]],
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )
    if merged.diagnoses.isna().any():
        raise ValueError("current PTB-XL record lacks a Challenge 2021 SNOMED mapping")
    removed_ids = sorted(set(ptb_headers.ecg_id) - set(ptb_database.ecg_id))
    if len(removed_ids) != 38:
        raise ValueError(f"expected 38 removed v1.0.1 records, found {len(removed_ids)}")

    cohorts = {
        "ptb_train": merged[merged.strat_fold.between(1, 8)],
        "ptb_validation": merged[merged.strat_fold == 9],
        "ptb_test": merged[merged.strat_fold == 10],
        "chapman_shaoxing": headers[headers.cohort == "chapman_shaoxing"],
        "ningbo": headers[headers.cohort == "ningbo"],
    }
    groups = load_scored_groups(str(args.weights), str(args.scored_mapping))
    audit_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for group in groups:
        row: dict[str, object] = {
            "label_key": group.key,
            "snomed_codes": "|".join(group.codes),
            "diagnosis": group.name,
        }
        target_reasons: list[str] = []
        headline_reasons: list[str] = []
        for cohort_name, cohort in cohorts.items():
            positives = int(positive_mask(cohort.diagnoses, group.codes).sum())
            row[f"{cohort_name}_records"] = int(len(cohort))
            row[f"{cohort_name}_positives"] = positives
            row[f"{cohort_name}_negatives"] = int(len(cohort) - positives)
        if group.key in SEMANTIC_EXCLUSIONS:
            target_reasons.append(SEMANTIC_EXCLUSIONS[group.key])
        for cohort_name in cohorts:
            if int(row[f"{cohort_name}_positives"]) < 1:
                target_reasons.append(f"{cohort_name}_positives<1")
            if int(row[f"{cohort_name}_negatives"]) < 1:
                target_reasons.append(f"{cohort_name}_negatives<1")

        headline_reasons.extend(target_reasons)
        if int(row["ptb_train_positives"]) < args.min_train_positives:
            headline_reasons.append(f"ptb_train_positives<{args.min_train_positives}")
        for cohort_name in EVALUATION_COHORTS:
            if int(row[f"{cohort_name}_positives"]) < args.min_eval_positives:
                headline_reasons.append(f"{cohort_name}_positives<{args.min_eval_positives}")
            if int(row[f"{cohort_name}_negatives"]) < args.min_eval_negatives:
                headline_reasons.append(f"{cohort_name}_negatives<{args.min_eval_negatives}")
        target_eligible = not target_reasons
        headline_eligible = not headline_reasons
        row["target_eligible"] = target_eligible
        row["headline_eligible"] = headline_eligible
        row["target_exclusion_reasons"] = "|".join(target_reasons)
        row["headline_exclusion_reasons"] = "|".join(headline_reasons)
        audit_rows.append(row)
        if target_eligible:
            selected.append(
                {
                    "label_key": group.key,
                    "snomed_codes": list(group.codes),
                    "diagnosis": group.name,
                    "headline_eligible": headline_eligible,
                    "positive_counts": {name: int(row[f"{name}_positives"]) for name in cohorts},
                }
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "label_mapping_audit.csv", index=False)
    audit[audit.target_eligible].to_csv(output_dir / "harmonized_label_counts.csv", index=False)

    group_by_key = {group.key: group for group in groups}
    subgroup_rows: list[dict[str, object]] = []
    subgroup_common_labels: dict[str, dict[str, list[str]]] = {}
    for cohort_name in EVALUATION_COHORTS:
        frame = cohorts[cohort_name].copy()
        stratifications = {
            "age": (
                frame.age.map(age_group),
                ("under_40", "40_to_64", "65_plus"),
            ),
            "sex": (
                frame.sex.map(normalize_sex),
                ("female", "male"),
            ),
        }
        subgroup_common_labels[cohort_name] = {}
        for stratification, (assignments, levels) in stratifications.items():
            eligible_across_levels: list[str] = []
            for label in selected:
                group = group_by_key[str(label["label_key"])]
                label_has_support = True
                for level in levels:
                    subgroup = frame[assignments == level]
                    positives = int(positive_mask(subgroup.diagnoses, group.codes).sum())
                    negatives = int(len(subgroup) - positives)
                    subgroup_rows.append(
                        {
                            "cohort": cohort_name,
                            "stratification": stratification,
                            "subgroup": level,
                            "subgroup_records": int(len(subgroup)),
                            "label_key": group.key,
                            "diagnosis": group.name,
                            "positives": positives,
                            "negatives": negatives,
                        }
                    )
                    if (
                        positives < args.min_subgroup_positives
                        or negatives < args.min_subgroup_negatives
                    ):
                        label_has_support = False
                if label_has_support:
                    eligible_across_levels.append(group.key)
            subgroup_common_labels[cohort_name][stratification] = eligible_across_levels
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "subgroup_label_counts.csv", index=False)

    header_summaries = {
        cohort: cohort_summary(headers[headers.cohort == cohort])
        for cohort in EXPECTED_HEADER_COUNTS
    }
    ptb_split_summary = {
        name: {
            "records": int(len(frame)),
            "patients": int(frame.patient_id.nunique()),
        }
        for name, frame in cohorts.items()
        if name.startswith("ptb_")
    }
    manifest = {
        "schema_version": 1,
        "source_versions": {
            "ptb_xl": "1.0.3",
            "ptb_xl_plus": "1.0.1",
            "challenge_2021": "1.0.3",
        },
        "eligibility_rule": {
            "label_universe": "26 scored CinC 2021 diagnosis equivalence groups",
            "model_target_rule": (
                "at least one positive and negative in PTB train and every evaluation cohort, "
                "excluding prespecified semantic mismatches"
            ),
            "semantic_exclusions": SEMANTIC_EXCLUSIONS,
            "min_ptb_train_positives": args.min_train_positives,
            "min_positives_each_evaluation_cohort": args.min_eval_positives,
            "min_negatives_each_evaluation_cohort": args.min_eval_negatives,
            "evaluation_cohorts": list(EVALUATION_COHORTS),
            "min_positives_each_subgroup": args.min_subgroup_positives,
            "min_negatives_each_subgroup": args.min_subgroup_negatives,
        },
        "labels": selected,
        "headline_label_keys": [
            label["label_key"] for label in selected if label["headline_eligible"]
        ],
        "subgroup_common_label_keys": subgroup_common_labels,
        "cohorts": header_summaries,
        "ptb_splits_current_release": ptb_split_summary,
        "ptb_challenge_records_removed_in_v1_0_3": removed_ids,
        "source_sha256": {
            str(args.headers): sha256(args.headers),
            str(args.ptb_database): sha256(args.ptb_database),
            str(args.weights): sha256(args.weights),
            str(args.scored_mapping): sha256(args.scored_mapping),
        },
    }
    (output_dir / "harmonized_labels.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    headline_count = sum(bool(label["headline_eligible"]) for label in selected)
    print(
        f"selected {len(selected)} model targets and {headline_count} headline labels "
        f"from {len(groups)} scored diagnosis groups"
    )
    for label in selected:
        counts = ", ".join(f"{key}={value}" for key, value in label["positive_counts"].items())
        print(f"{label['label_key']}: {label['diagnosis']} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
