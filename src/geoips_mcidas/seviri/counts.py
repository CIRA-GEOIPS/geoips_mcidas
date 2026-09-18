"""Assemble raw SEVIRI counts and calibration from HRIT segment files.

GeoIPS's own ``seviri_hrit`` reader converts counts to radiance and keeps only
the result, discarding both the counts and the prologue that holds the
per-band slope and intercept. An AREA file for CLAVR-x needs exactly what that
reader throws away, so this module reads the HRIT segments again for counts,
reusing the segment decompression and prologue parsing in
``geoips.plugins.classes.readers.utils.hrit_reader``.

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
    """Split HRIT files into the prologue and per-band segment maps."""
    prologue: HritFile | None = None
    bands: SegmentMap = {}
    for path in paths:
        handle = HritFile(str(path))
        if handle.file_type == PROLOGUE_TYPE:
            prologue = handle
        elif handle.file_type == IMAGE_TYPE and handle.band is not None:
            number = int(handle.band.removeprefix("B"))
            if number in NARROWBAND_BANDS:
                bands.setdefault(number, {})[int(handle.segment)] = handle
    return prologue, bands


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
        If the prologue or any image segment is absent.
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
