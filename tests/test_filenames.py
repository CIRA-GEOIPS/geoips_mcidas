"""Filenames are an integration contract with CLAVR-x, not a convention."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from geoips_mcidas.area import filenames

OBSERVED = datetime(2021, 5, 21, 1, 0, tzinfo=UTC)
ALL_BANDS = range(1, 12)


def test_matches_the_existing_pipeline_convention() -> None:
    name = filenames.build_area_filename("meteosat-11", OBSERVED, 1)
    assert name == "met11_1_2021_141_0100.area"


def test_day_of_year_is_zero_padded() -> None:
    """A day of year below 100 must still be three digits."""
    early = datetime(2021, 1, 5, 12, 30, tzinfo=UTC)
    assert filenames.build_area_filename("meteosat-9", early, 1) == (
        "met9_1_2021_005_1230.area"
    )


@pytest.mark.parametrize("band", ALL_BANDS)
def test_clavrx_derives_every_sibling_we_write(band: int) -> None:
    """The whole scheme rests on this round trip.

    CLAVR-x receives only the band-1 name and splices the rest. If our name
    for band N differs from CLAVR-x's derivation of it, that band's file is
    simply not found -- and the run either dies or silently uses fewer
    channels.
    """
    band_1 = filenames.build_area_filename("meteosat-11", OBSERVED, 1)
    ours = filenames.build_area_filename("meteosat-11", OBSERVED, band)
    assert filenames.clavrx_sibling_filename(band_1, band) == ours


def test_bands_are_not_zero_padded() -> None:
    """CLAVR-x formats bands as I1.1 and I2.2, so neither pads."""
    band_1 = filenames.build_area_filename("meteosat-11", OBSERVED, 1)
    assert filenames.clavrx_sibling_filename(band_1, 9).count("_9_") == 1
    assert filenames.clavrx_sibling_filename(band_1, 10).count("_10_") == 1


@pytest.mark.parametrize(
    "platform",
    ["meteosat-8", "meteosat-9", "meteosat-10", "meteosat-11"],
)
def test_no_platform_prefix_introduces_an_earlier_marker(platform: str) -> None:
    """An underscore in the prefix would move the first '_1_' and break the
    splice for every band at once."""
    name = filenames.build_area_filename(platform, OBSERVED, 1)
    prefix = filenames.platform_token(platform)
    assert name.find(filenames.BAND_1_MARKER) == len(prefix)


def test_unknown_platform_is_rejected() -> None:
    with pytest.raises(KeyError, match="meteosat-99"):
        filenames.build_area_filename("meteosat-99", OBSERVED, 1)


def test_clavrx_splice_is_silent_without_a_marker() -> None:
    """Documents the failure mode we validate against: no error, bad name."""
    mangled = filenames.clavrx_sibling_filename("noband_2021_141.area", 5)
    assert mangled == "_5_band_2021_141.area"
