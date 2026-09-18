"""The MSGT navigation block, checked against real McIDAS output."""

from __future__ import annotations

from pathlib import Path

import pytest

from geoips_mcidas.area import msgt_nav, words

# Meteosat-11 sits at 0 degrees, so the reference file's longitude word is 0
# and cannot reveal a sign error. IODC positions are used to test the sign.
IODC_LONGITUDE_EAST = 45.5


def test_block_is_512_bytes() -> None:
    block = msgt_nav.build_msgt_nav_block(0.0, band=1)
    assert len(block) == msgt_nav.NAV_BLOCK_BYTES == 512


def test_type_tag_is_ascii_in_natural_order() -> None:
    block = msgt_nav.build_msgt_nav_block(0.0, band=1)
    assert block[:4] == b"MSGT"


@pytest.mark.parametrize(
    ("degrees_east", "expected"),
    [
        (0.0, 0),
        (IODC_LONGITUDE_EAST, -455000),
        (-3.477996, 34780),
    ],
)
def test_longitude_is_stored_west_positive(degrees_east: float, expected: int) -> None:
    """A sign error here mis-navigates the image with no error raised."""
    assert msgt_nav.encode_sub_satellite_longitude(degrees_east) == expected


def test_scaling_factor_matches_mcidas() -> None:
    """13642337 / 2**16 pixels per degree, scaled by a million and truncated."""
    assert msgt_nav._scaling_factor() == 208165542


@pytest.mark.golden
def test_matches_reference_nav_block(reference_area: Path) -> None:
    """Every word we write must match what McIDAS-X wrote for the same scene."""
    with reference_area.open("rb") as handle:
        handle.seek(256)
        expected = handle.read(msgt_nav.NAV_BLOCK_BYTES)

    # The reference is a multi-band file, whose nav band word is 1.
    actual = msgt_nav.build_msgt_nav_block(0.0, band=1)

    assert actual[:4] == expected[:4]
    assert words.unpack_words(actual) == words.unpack_words(expected)
