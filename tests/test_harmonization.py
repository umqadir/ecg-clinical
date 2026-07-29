from pathlib import Path

import pandas as pd

from ecg_clinical.harmonization import load_scored_groups, positive_mask


def test_positive_mask_matches_exact_codes() -> None:
    diagnoses = pd.Series(["12|123", "123", "312", "12", ""])
    assert positive_mask(diagnoses, ("12",)).tolist() == [True, False, False, True, False]


def test_load_scored_equivalence_groups(tmp_path: Path) -> None:
    weights = pd.DataFrame([[1.0]], index=["1|2"], columns=["1|2"])
    weights_path = tmp_path / "weights.csv"
    weights.to_csv(weights_path)
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        {
            "SNOMEDCTCode": ["1", "2"],
            "Dx": ["first", "second"],
        }
    ).to_csv(mapping_path, index=False)

    try:
        load_scored_groups(str(weights_path), str(mapping_path))
    except ValueError as error:
        assert "weights matrix" in str(error)
    else:
        raise AssertionError("non-26x26 matrix should be rejected")
