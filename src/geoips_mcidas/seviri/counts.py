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

from geoips_mcidas.errors import SeviriHritError

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


def _reject_inconsistent(handle: HritFile, seen: dict[str, object], name: str) -> None:
    """Require every file in a scene to agree on observation and spacecraft.

    EUMETCast delivers the 0-degree service and the IODC service into the same
    spool on the same 15-minute slots, so one glob over a mixed archive yields
    two spacecraft whose files all share a nominal time. Segments are keyed on
    band and segment number alone, so bands -- or strips within a band -- would
    come from the wrong satellite while the platform, sensor-source word,
    filename prefix and calibration block all came from whichever prologue was
    read last.

    Raises
    ------
    SeviriHritError
        On the first disagreement, naming the file.
    """
    observed = {
        "observation time": handle.annotation_metadata["start_datetime"],
        "platform": handle.annotation_metadata["platform"],
    }
    for field, value in observed.items():
        if field not in seen:
            seen[field] = value
        elif seen[field] != value:
            raise SeviriHritError(
                f"HRIT files disagree on {field}: {seen[field]} and {value} "
                f"({name}). Pass the files for a single cycle of one satellite.",
            )


def _group_segments(
    paths: Iterable[Path],
) -> tuple[HritFile | None, dict[int, dict[int, HritFile]]]:
    """Split HRIT files into the prologue and per-band segment maps.

    Raises
    ------
    SeviriHritError
        If any segment is compressed, if the files disagree on observation time
        or spacecraft, or if a second prologue is present.
    """
    prologue: HritFile | None = None
    bands: SegmentMap = {}
    seen: dict[str, object] = {}
    for path in paths:
        handle = HritFile(str(path))
        # Only image segments carry block_1; the prologue has no data to compress.
        image_header = handle.metadata.get("block_1", {})
        if int(image_header.get("compression", 0)) != 0:
            raise SeviriHritError(
                f"{path.name} is compressed; decompress the HRIT segments first",
            )
        _reject_inconsistent(handle, seen, path.name)
        if handle.file_type == PROLOGUE_TYPE:
            if prologue is not None:
                raise SeviriHritError(
                    f"more than one HRIT prologue supplied ({path.name}); "
                    "the calibration block would depend on file order",
                )
            prologue = handle
        elif handle.file_type == IMAGE_TYPE and handle.band is not None:
            number = int(handle.band.removeprefix("B"))
            if number in NARROWBAND_BANDS:
                segment = int(handle.segment)
                if segment in bands.get(number, {}):
                    raise SeviriHritError(
                        f"duplicate segment {segment} for band {number} ({path.name})",
                    )
                bands.setdefault(number, {})[segment] = handle
    return prologue, bands


def _check_segments_complete(bands: SegmentMap, lines: int) -> None:
    """Require each band's segments to tile the full disk exactly.

    A missing or short segment is the normal partial-reception failure of an
    EUMETCast feed, and left alone it is invisible: the array is pre-allocated
    as zeros, so the gap becomes a black stripe in a file of exactly the right
    size, which navigates correctly, passes every other check, and is
    calibrated by CLAVR-x as though the zeros were measurements.

    Summing the declared line counts is not enough. ``_assemble_band`` places
    each strip at ``num_lines * (segment - 1)`` using that segment's own
    declared height, so heights that merely add up to the right total still
    leave a gap and an overlap. The segment numbers must therefore be the
    contiguous run the prologue declares, and every segment must be the same
    height.

    Raises
    ------
    SeviriHritError
        If any band's segments do not tile the image exactly.
    """
    problems = {}
    for band, segments in sorted(bands.items()):
        block = next(iter(segments.values())).metadata["block_128"]
        expected = set(
            range(int(block["start_segment"]), int(block["end_segment"]) + 1)
        )
        if set(segments) != expected:
            problems[band] = f"missing segments {sorted(expected - set(segments))}"
            continue
        heights = {
            int(handle.metadata["block_1"]["num_lines"]) for handle in segments.values()
        }
        if len(heights) != 1:
            problems[band] = f"segments differ in height: {sorted(heights)}"
        elif heights.pop() * len(segments) != lines:
            problems[band] = f"segments cover {len(segments)} strips, not {lines} lines"
    if problems:
        raise SeviriHritError(
            f"HRIT segments do not tile the image: {problems}. "
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
    try:
        prologue, bands = _group_segments(paths)
    except SeviriHritError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise SeviriHritError(f"could not read the HRIT files: {exc}") from exc
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

    # Assemble band by band and drop each band's handles as we go. HritFile
    # caches every segment it reads as an int64 array -- four times the uint16
    # actually wanted -- so holding all 88 alive retains about 1.2 GB that has
    # already been copied into `counts`, plus one file descriptor each.
    counts: dict[int, NDArray[np.uint16]] = {}
    for band in sorted(bands):
        counts[band] = _assemble_band(bands.pop(band), lines, elements)

    return SeviriCounts(
        counts=counts,
        gains=[float(entry["slope"]) for entry in calibration[:MSG_BANDS]],
        offsets=[float(entry["offset"]) for entry in calibration[:MSG_BANDS]],
        platform_name=platform_name(prologue.annotation_metadata["platform"]),
        observation_time=prologue.annotation_metadata["start_datetime"],
        sub_satellite_longitude_east=float(
            header["imageDescription"]["projectionDescription"]["longitudeOfSSP"],
        ),
    )
