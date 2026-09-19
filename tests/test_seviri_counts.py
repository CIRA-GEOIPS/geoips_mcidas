"""Reading raw counts out of HRIT, checked against real McIDAS output.

These are the tests that matter most. Every other check verifies that we can
lay bytes out correctly; these verify that the bytes are the *right* ones.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geoips")
from geoips_mcidas.seviri import counts as seviri_counts

FULL_DISK = 3712
NARROWBAND_COUNT = 11
REFERENCE_DATA_OFFSET = 2368
REFERENCE_PREFIX_BYTES = 12


def reference_band_plane(area_path: Path, band: int) -> np.ndarray:
    """Extract one band from the multi-band reference file.

    Data is band-interleaved by pixel, behind a 12-byte per-line band list, so
    a band is a strided view rather than a contiguous block.
    """
    line_words = (REFERENCE_PREFIX_BYTES + FULL_DISK * NARROWBAND_COUNT * 2) // 2
    mapped = np.memmap(
        area_path,
        dtype="<u2",
        mode="r",
        offset=REFERENCE_DATA_OFFSET,
        shape=(FULL_DISK, line_words),
    )
    start = REFERENCE_PREFIX_BYTES // 2 + (band - 1)
    return np.ascontiguousarray(mapped[:, start::NARROWBAND_COUNT])


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("msg1", "meteosat-8"),
        ("msg2", "meteosat-9"),
        ("msg3", "meteosat-10"),
        ("MSG4", "meteosat-11"),
        ("msg2_iodc", "meteosat-9"),
    ],
)
def test_platform_names(tag: str, expected: str) -> None:
    assert seviri_counts.platform_name(tag) == expected


def test_unknown_platform_is_rejected() -> None:
    with pytest.raises(KeyError, match="msg9"):
        seviri_counts.platform_name("msg9")


def test_hrv_is_excluded() -> None:
    """CLAVR-x never reads HRV and its grid differs from the narrowbands."""
    assert 12 not in seviri_counts.NARROWBAND_BANDS
    assert len(seviri_counts.NARROWBAND_BANDS) == NARROWBAND_COUNT


def test_missing_prologue_is_reported(tmp_path: Path) -> None:
    with pytest.raises(seviri_counts.SeviriHritError, match="prologue"):
        seviri_counts.read_seviri_counts(tmp_path.glob("*"))


@pytest.mark.golden
def test_reads_every_narrowband(sample_counts: seviri_counts.SeviriCounts) -> None:
    result = sample_counts
    assert sorted(result.counts) == list(range(1, 12))
    assert all(
        image.shape == (FULL_DISK, FULL_DISK) for image in result.counts.values()
    )
    assert result.platform_name == "meteosat-11"
    assert result.sub_satellite_longitude_east == 0.0


@pytest.mark.golden
def test_calibration_matches_the_reference(
    sample_counts: seviri_counts.SeviriCounts,
) -> None:
    """The gains we read must be the gains McIDAS wrote into its cal block."""
    result = sample_counts
    assert len(result.gains) == 12
    assert result.gains[0] == pytest.approx(0.021240899339318275)
    assert result.offsets[0] == pytest.approx(-1.083285866305232)
    assert result.gains[8] == pytest.approx(0.2050356762076601)


@pytest.mark.golden
@pytest.mark.parametrize("band", [1, 5, 9, 11])
def test_counts_match_mcidas_pixel_for_pixel(
    sample_counts: seviri_counts.SeviriCounts,
    reference_area: Path,
    band: int,
) -> None:
    """The decisive test: our counts must equal McIDAS's, exactly.

    This is what pins the row order down. HRIT segment 1 is the southernmost
    strip, so stacking segments ascending gives a south-up image while AREA
    stores north-up. Without the flip roughly a quarter of pixels coincide --
    enough for an image to look plausible and be upside down.
    """
    np.testing.assert_array_equal(
        sample_counts.counts[band],
        reference_band_plane(reference_area, band),
    )


@pytest.mark.golden
def test_unflipped_counts_would_not_match(
    sample_counts: seviri_counts.SeviriCounts,
    reference_area: Path,
) -> None:
    """Guards the guard: confirm the flip is doing real work."""
    result = sample_counts
    expected = reference_band_plane(reference_area, 9)
    assert not np.array_equal(np.flipud(result.counts[9]), expected)


@pytest.mark.golden
def test_a_missing_segment_is_rejected(sample_hrit: Path) -> None:
    """The failure this module exists to survive must not pass silently.

    Partial reception over EUMETCast is routine. A dropped segment leaves a
    zero-filled stripe in a file of exactly the right size, which navigates
    correctly and is calibrated by CLAVR-x as though the zeros were
    measurements.
    """
    files = sorted(sample_hrit.glob("H-*"))
    dropped = next(f for f in files if "IR_108___-000004" in f.name)
    kept = [f for f in files if f != dropped]

    with pytest.raises(seviri_counts.SeviriHritError, match="do not tile the image"):
        seviri_counts.read_seviri_counts(kept)


@pytest.mark.golden
def test_missing_segment_error_names_the_band(sample_hrit: Path) -> None:
    files = sorted(sample_hrit.glob("H-*"))
    kept = [f for f in files if "IR_108___-000004" not in f.name]
    with pytest.raises(seviri_counts.SeviriHritError) as excinfo:
        seviri_counts.read_seviri_counts(kept)
    assert "9" in str(excinfo.value)


class _FakeHrit:
    """Minimal stand-in for HritFile, for building inconsistent scenes."""

    def __init__(
        self,
        name: str,
        moment: datetime,
        segment: int = 1,
        platform: str = "msg4",
    ) -> None:
        self.file_type = "image"
        self.band = "B01"
        self.segment = segment
        self.compressed = False
        self.annotation_metadata = {"start_datetime": moment, "platform": platform}
        self.metadata = {
            "block_1": {"num_lines": 464},
            "block_128": {"start_segment": 1, "end_segment": 8},
        }
        self._name = name


def test_two_observation_times_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An archive directory at a 15-minute cadence holds many cycles, and the
    filename order GeoIPS passes on is whatever the glob returned.

    The check compares the timestamp embedded in each file, not the filename,
    so a renamed copy is correctly treated as the same observation.
    """
    first = datetime(2021, 5, 21, 1, 0)
    second = datetime(2021, 5, 21, 1, 15)
    times = {"a": first, "b": second}

    segments = {"a": 1, "b": 2}
    monkeypatch.setattr(
        seviri_counts,
        "HritFile",
        lambda path: _FakeHrit(
            path,
            times[Path(path).name],
            segment=segments[Path(path).name],
        ),
    )
    with pytest.raises(
        seviri_counts.SeviriHritError,
        match="disagree on observation time",
    ):
        seviri_counts.read_seviri_counts([Path("a"), Path("b")])


def test_two_spacecraft_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """EUMETCast delivers the 0-degree and IODC services into one spool on the
    same 15-minute slots, so a mixed glob passes the observation-time check."""
    moment = datetime(2021, 5, 21, 1, 0)
    platforms = {"a": "msg4", "b": "msg2"}
    segments = {"a": 1, "b": 2}
    monkeypatch.setattr(
        seviri_counts,
        "HritFile",
        lambda path: _FakeHrit(
            path,
            moment,
            segment=segments[Path(path).name],
            platform=platforms[Path(path).name],
        ),
    )
    with pytest.raises(seviri_counts.SeviriHritError, match="disagree on platform"):
        seviri_counts.read_seviri_counts([Path("a"), Path("b")])


def test_a_duplicate_segment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    moment = datetime(2021, 5, 21, 1, 0)
    monkeypatch.setattr(
        seviri_counts,
        "HritFile",
        lambda path: _FakeHrit(path, moment, segment=1),
    )
    with pytest.raises(seviri_counts.SeviriHritError, match="duplicate segment"):
        seviri_counts.read_seviri_counts([Path("a"), Path("b")])


def test_a_single_observation_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the guard: the time check must not reject a valid scene."""
    moment = datetime(2021, 5, 21, 1, 0)
    segments = {"a": 1, "b": 2}
    monkeypatch.setattr(
        seviri_counts,
        "HritFile",
        lambda path: _FakeHrit(path, moment, segment=segments[Path(path).name]),
    )
    # No prologue in this set, so it fails on that instead of on consistency.
    with pytest.raises(seviri_counts.SeviriHritError, match="prologue"):
        seviri_counts.read_seviri_counts([Path("a"), Path("b")])
