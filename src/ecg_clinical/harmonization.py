"""Canonical PhysioNet/CinC 2021 diagnosis equivalence groups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DiagnosisGroup:
    key: str
    codes: tuple[str, ...]
    name: str


def load_scored_groups(weights_path: str, mapping_path: str) -> list[DiagnosisGroup]:
    """Load the 26 scored groups, preserving official equivalent-code sets."""

    weights = pd.read_csv(weights_path, index_col=0)
    if weights.shape != (26, 26) or weights.index.tolist() != weights.columns.tolist():
        raise ValueError("unexpected CinC 2021 weights matrix")
    mapping = pd.read_csv(mapping_path, dtype={"SNOMEDCTCode": str})
    names = dict(zip(mapping["SNOMEDCTCode"], mapping["Dx"], strict=True))

    groups: list[DiagnosisGroup] = []
    for key in weights.index:
        codes = tuple(key.split("|"))
        code_names = [names[code] for code in codes]
        name = code_names[0] if len(set(code_names)) == 1 else " / ".join(code_names)
        groups.append(DiagnosisGroup(key=key, codes=codes, name=name))
    return groups


def positive_mask(diagnoses: pd.Series, codes: tuple[str, ...]) -> pd.Series:
    """Return a vectorized exact-code membership mask for pipe-delimited labels."""

    alternatives = "|".join(codes)
    return diagnoses.fillna("").str.contains(rf"(?:^|\|)(?:{alternatives})(?:\||$)", regex=True)
