import numpy as np

from ecg_clinical.waveforms import (
    decode_ptb_dat,
    parse_signal_specification,
    physical_signal,
)
from ecg_clinical.wfdb_header import parse_header

EXAMPLE = """A0001 2 500 5000
A0001.mat 16+24 1000/mV 16 0 1 2 0 I
A0001.mat 16+24 1000/mV 16 0 3 4 0 II
#Age: 74
#Sex: Male
#Dx: 426783006, 164889003
"""


def test_parse_header() -> None:
    parsed = parse_header(EXAMPLE)
    assert parsed.record_id == "A0001"
    assert parsed.num_leads == 2
    assert parsed.sampling_frequency_hz == 500
    assert parsed.num_samples == 5000
    assert parsed.lead_names == ("I", "II")
    assert parsed.age == "74"
    assert parsed.sex == "Male"
    assert parsed.diagnoses == ("164889003", "426783006")
    assert len(parsed.header_sha256) == 64


def test_parse_and_decode_physical_signal() -> None:
    lines = ["A0001 12 100 1000"]
    for index, lead in enumerate(
        ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    ):
        lines.append(f"A0001.dat 16 200(10)/mV 16 0 {index} 0 0 {lead}")
    specification = parse_signal_specification("\n".join(lines))
    assert specification.leads[3:6] == ("AVR", "AVL", "AVF")
    digital = np.arange(12_000, dtype="<i2").reshape(1000, 12)
    decoded = decode_ptb_dat(digital.tobytes(), specification)
    expected = physical_signal(digital.T, specification)
    np.testing.assert_allclose(decoded, expected)
    assert decoded.shape == (12, 1000)


def test_parse_header_preserves_source_matrix_name_when_record_id_differs() -> None:
    parsed = parse_header(EXAMPLE.replace("A0001.mat", "JA0001.mat"))

    assert parsed.record_id == "A0001"
    assert parsed.signal_file == "JA0001.mat"
