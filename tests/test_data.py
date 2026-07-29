import json
from pathlib import Path

import numpy as np

from ecg_clinical.data import inference_window_starts, load_normalization


def test_inference_windows_span_the_record() -> None:
    starts = inference_window_starts()
    assert starts.tolist() == [0, 83, 167, 250, 333, 417, 500, 583, 667, 750]
    assert np.all(np.diff(starts) > 0)


def test_load_normalization(tmp_path: Path) -> None:
    path = tmp_path / "statistics.json"
    path.write_text(
        json.dumps(
            {
                "per_lead_mean_mv": list(range(12)),
                "per_lead_standard_deviation_mv": [2] * 12,
            }
        )
    )
    mean, standard_deviation = load_normalization(path)
    assert mean.shape == standard_deviation.shape == (12, 1)
    assert standard_deviation[0, 0] == 2
