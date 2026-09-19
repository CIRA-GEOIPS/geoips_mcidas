"""Assemble raw SEVIRI counts and calibration from HRIT segment files.

GeoIPS's own ``seviri_hrit`` reader converts counts to radiance and keeps only
the result, discarding both the counts and the prologue that holds the
per-band slope and intercept. An AREA file for CLAVR-x needs exactly what that
reader throws away, so this module reads the HRIT segments again for counts,
reusing the prologue parsing and segment reading in
``geoips.plugins.classes.readers.utils.hrit_reader``.

Compressed segments are not handled. ``HritFile`` can decompress, but doing so
needs a writable scratch directory and the EUMETSAT wavelet library, so this
module rejects a compressed file rather than half-supporting it.

**Row order.** Segment 1 of an HRIT image is the *southernmost* strip, so
stacking segments in ascending order yields a south-up image, while AREA files
store lines north-up. The assembled image is therefore flipped vertically.
This is not a guess: flipping reproduces a McIDAS-produced AREA file's pixel
data exactly -- 13,778,944 of 13,778,944 pixels on every band tested -- and
not flipping matches about 26%. A wrong orientation here would produce files
that load, display and process without complaint while being upside down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
from geoips.plugins.classes.readers.utils.hrit_reader import HritFile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from numpy.typing import NDArray

MSG_BANDS: Final = 12
"""Calibration covers all twelve channels even though only eleven are read."""

NARROWBAND_BANDS: Final = tuple(range(1, 12))
"""HRV is excluded: CLAVR-x never reads it and the grid differs."""

PLATFORM_NAMES: Final[dict[str, str]] = {
    "msg1": "meteosat-8",
    "msg2": "meteosat-9",
    "msg3": "meteosat-10",
    "msg4": "meteosat-11",
}

PROLOGUE_TYPE: Final = "prologue"
IMAGE_TYPE: Final = "image"

SegmentMap = dict[int, dict[int, HritFile]]
"""Band number to segment number to the HRIT file holding that strip."""


class SeviriCounts(NamedTuple):
    """Raw counts plus everything an AREA file needs to describe them."""

    counts: dict[int, NDArray[np.uint16]]
    gains: list[float]
    offsets: list[float]
    platform_name: str
    observation_time: datetime
    sub_satellite_longitude_east: float


class SeviriHritError(RuntimeError):
    """Raised when a set of HRIT files cannot yield a complete image."""


def platform_name(hrit_platform: str) -> str:
    """Map an HRIT platform tag such as ``msg4`` to a GeoIPS platform name.

    Raises
    ------
    KeyError
        If the tag is not a recognised MSG satellite.
    """
    key = hrit_platform.lower().replace("_iodc", "")
    if key not in PLATFORM_NAMES:
        raise KeyError(f"unrecognised MSG platform tag {hrit_platform!r}")
    return PLATFORM_NAMES[key]


def _assemble_band(
    segments: dict[int, HritFile], lines: int, elements: int
) -> NDArray[np.uint16]:
    """Stack a band's segments into one north-up full-disk image."""
    image = np.zeros((lines, elements), dtype=np.uint16)
    for number in sorted(segments):
        segment = segments[number]
        per_segment = int(segment.metadata["block_1"]["num_lines"])
        start = per_segment * (number - 1)
        image[start : start + per_segment, :] = segment._read_image_data()
    return np.flipud(image)


def _group_segments(
    paths: Iterable[Path],
) -> tuple[HritFile | None, dict[int, dict[int, HritFile]]]:
    """Split HRIT files into the prologue and per-band segment maps.

    Every file must belong to the same observation. An archive directory at a
    15-minute cadence holds many cycles, and the order GeoIPS hands filenames
    over is whatever the glob returned, so keeping the last of each would
    quietly assemble one image out of several scans.

    Raises
    ------
    SeviriHritError
        If the files span more than one observation time, or if any segment is
        compressed.
    """
    prologue: HritFile | None = None
    bands: SegmentMap = {}
    observation = None
    for path in paths:
        handle = HritFile(str(path))
        if handle.compressed:
            raise SeviriHritError(
                f"{path.name} is compressed; decompress the HRIT segments first",
            )
        moment = handle.annotation_metadata["start_datetime"]
        if observation is None:
            observation = moment
        elif moment != observation:
            raise SeviriHritError(
                f"HRIT files span more than one observation: {observation} and "
                f"{moment} ({path.name}). Pass the files for a single cycle.",
            )
        if handle.file_type == PROLOGUE_TYPE:
            prologue = handle
        elif handle.file_type == IMAGE_TYPE and handle.band is not None:
            number = int(handle.band.removeprefix("B"))
            if number in NARROWBAND_BANDS:
                bands.setdefault(number, {})[int(handle.segment)] = handle
    return prologue, bands


def _check_segments_complete(bands: SegmentMap, lines: int) -> None:
    """Require each band's segments to tile the full disk with no gaps.

    A missing segment is the normal partial-reception failure of an EUMETCast
    feed, and left alone it is invisible: the array is pre-allocated as zeros,
    so the gap becomes a black stripe in a file of exactly the right size,
    which navigates correctly, passes every structural check we have, and is
    calibrated by CLAVR-x as though the zeros were measurements.

    Raises
    ------
    SeviriHritError
        If any band's segments do not cover every line.
    """
    incomplete = {}
    for band, segments in sorted(bands.items()):
        covered = sum(
            int(handle.metadata["block_1"]["num_lines"]) for handle in segments.values()
        )
        if covered != lines:
            missing = sorted(set(range(1, max(segments) + 1)) - set(segments))
            incomplete[band] = missing or f"{covered} of {lines} lines"
    if incomplete:
        raise SeviriHritError(
            f"incomplete HRIT coverage; missing segments by band: {incomplete}. "
            "A gap would be written as zeros and calibrated as real data.",
        )


def read_seviri_counts(paths: Iterable[Path]) -> SeviriCounts:
    """Read raw counts and calibration from a set of HRIT files.

    Parameters
    ----------
    paths
        Every HRIT file for one observation time, including the prologue.

    Returns
    -------
    SeviriCounts
        Counts keyed by MSG band number, north-up.

    Raises
    ------
    SeviriHritError
        If the prologue is absent, if no image segments are present, if any
        band's segments do not tile the full disk, or if the files span more
        than one observation time.
    """
    prologue, bands = _group_segments(paths)
    if prologue is None:
        raise SeviriHritError("no HRIT prologue found; calibration is unavailable")
    if not bands:
        raise SeviriHritError("no HRIT image segments found")

    header = prologue.prologue
    grid = header["imageDescription"]["referenceGridVIS_IR"]
    lines = int(grid["numberOfLines"])
    elements = int(grid["numberOfColumns"])
    calibration = header["radiometricProcessing"]["level15ImageCalibration"]
    _check_segments_complete(bands, lines)

    any_image = next(iter(next(iter(bands.values())).values()))
    return SeviriCounts(
        counts={
            band: _assemble_band(segments, lines, elements)
            for band, segments in sorted(bands.items())
        },
        gains=[float(entry["slope"]) for entry in calibration[:MSG_BANDS]],
        offsets=[float(entry["offset"]) for entry in calibration[:MSG_BANDS]],
        platform_name=platform_name(prologue.annotation_metadata["platform"]),
        observation_time=prologue.annotation_metadata["start_datetime"],
        sub_satellite_longitude_east=float(any_image.geolocation_metadata["sublon"]),
    )
