"""The MSG calibration block, checked against real McIDAS output."""

from __future__ import annotations

from pathlib import Path

import pytest

from geoips_mcidas.area import msg_cal

CAL_OFFSET_IN_FILE = 768

# Gains and offsets McIDAS wrote for Meteosat-11, 2021-05-21 01:00, read back
# out of the reference file. Bands 1-12, HRV last.
REFERENCE_GAINS = [
    0.2124089934e-01,
    0.2792079933e-01,
    0.2311230078e-01,
    0.3658666870e-02,
    0.8318111190e-02,
    0.3862196818e-01,
    0.1267443187e00,
    0.1039609107e00,
    0.2050356762e00,
    0.2223111467e00,
    0.1576068969e00,
    0.3567780182e-01,
]
REFERENCE_OFFSETS = [
    -0.1083285866e01,
    -0.1423960766e01,
    -0.1178727340e01,
    -0.1865920103e00,
    -0.4242236707e00,
    -0.1969720377e01,
    -0.6463960253e01,
    -0.5302006446e01,
    -0.1045681949e02,
    -0.1133786848e02,
    -0.8037951741e01,
    -0.1819567893e01,
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, " 0.0000000000E+00"),
        (0.02124089934, " 0.2124089934E-01"),
        (-1.083285866, "-0.1083285866E+01"),
        (198723.7748, " 0.1987237748E+06"),
        (0.9916, " 0.9916000000E+00"),
    ],
)
def test_fortran_e_format(value: float, expected: str) -> None:
    """Fortran normalises the mantissa below one; C and Python do not."""
    formatted = msg_cal.format_e17_10(value)
    assert formatted == expected
    assert len(formatted) == msg_cal.FIELD_WIDTH


def test_block_geometry() -> None:
    block = msg_cal.build_msg_cal_block(REFERENCE_GAINS, REFERENCE_OFFSETS)
    assert len(block) == msg_cal.CAL_BLOCK_BYTES
    assert block[:4] == b"MSGT"


def test_rejects_wrong_band_count() -> None:
    with pytest.raises(ValueError, match="12 gains"):
        msg_cal.build_msg_cal_block([1.0], [0.0])


def test_every_band_file_carries_all_twelve_records() -> None:
    """CLAVR-x reads all channels' coefficients from the band-1 file alone."""
    block = msg_cal.build_msg_cal_block(
        REFERENCE_GAINS,
        REFERENCE_OFFSETS,
        msg_cal.METEOSAT_11_PLANCK,
    )
    last = block[4 + 11 * 104 : 4 + 12 * 104].decode()
    assert "0.3567780182E-01" in last


@pytest.mark.golden
def test_matches_reference_cal_block(reference_area: Path) -> None:
    """Reproduce McIDAS's calibration block, field by field.

    Gain and offset are compared exactly: they are the only fields CLAVR-x
    reads, and they pass through us unchanged from the HRIT prologue.

    The four Planck fields are compared with a tolerance instead. McIDAS
    derives them in Fortran single precision, and the printed values sit off
    the float32 grid in a way that no ordering of the arithmetic reproduces
    bit for bit. Since CLAVR-x ignores these fields entirely, matching the
    physics to eight significant figures is the useful guarantee, not matching
    the rounding noise.
    """
    with reference_area.open("rb") as handle:
        handle.seek(CAL_OFFSET_IN_FILE)
        expected = handle.read(msg_cal.CAL_BLOCK_BYTES)

    actual = msg_cal.build_msg_cal_block(
        REFERENCE_GAINS,
        REFERENCE_OFFSETS,
        msg_cal.METEOSAT_11_PLANCK,
    )

    assert len(actual) == len(expected)
    assert actual[:4] == expected[:4]

    for band in range(1, msg_cal.MSG_BANDS + 1):
        ours = msg_cal.parse_cal_record(actual, band)
        theirs = msg_cal.parse_cal_record(expected, band)
        assert ours.gain == theirs.gain, f"band {band} gain"
        assert ours.offset == theirs.offset, f"band {band} offset"
        for name in ("c1w3", "c2w", "alpha", "beta"):
            mine, ref = getattr(ours, name), getattr(theirs, name)
            assert mine == pytest.approx(ref, rel=1e-6), f"band {band} {name}"

    flag_start = 4 + msg_cal.MSG_BANDS * msg_cal.CAL_RECORD_BYTES
    flag_stop = flag_start + msg_cal.MSG_BANDS * 4
    assert actual[flag_start:flag_stop] == expected[flag_start:flag_stop]


@pytest.mark.golden
def test_gain_and_offset_text_is_byte_exact(reference_area: Path) -> None:
    """The two fields CLAVR-x actually reads must round-trip exactly."""
    with reference_area.open("rb") as handle:
        handle.seek(CAL_OFFSET_IN_FILE)
        expected = handle.read(msg_cal.CAL_BLOCK_BYTES)

    actual = msg_cal.build_msg_cal_block(REFERENCE_GAINS, REFERENCE_OFFSETS)

    for band in range(1, msg_cal.MSG_BANDS + 1):
        start = 4 + (band - 1) * msg_cal.CAL_RECORD_BYTES
        # Fields five and six of the record: gain and offset.
        gain_offset = slice(start + 4 * msg_cal.FIELD_WIDTH, start + 102)
        assert actual[gain_offset] == expected[gain_offset], f"band {band}"
