"""Assemble and write single-band SEVIRI AREA files.

Blocks must be laid out contiguously in the order directory, navigation,
calibration, data, because AREA readers derive each block's *length* from the
difference between successive offsets. A gap is not padding; it silently
becomes part of the preceding block.

Everything except :func:`write_area_file` is pure: the builders take metadata
and return bytes, which is what makes each block independently comparable
against a reference file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from geoips_mcidas.area import words
from geoips_mcidas.area.directory import (
    BYTES_PER_PIXEL,
    CAL_OFFSET,
    DATA_OFFSET,
    NAV_OFFSET,
    build_directory,
)
from geoips_mcidas.area.msg_cal import CAL_BLOCK_BYTES
from geoips_mcidas.area.msgt_nav import NAV_BLOCK_BYTES

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from geoips_mcidas.area.directory import SeviriAreaMetadata

COUNT_MAX: Final = 0xFFFF
"""Widest value a two-byte AREA element can hold. SEVIRI counts are 10-bit."""

COUNT_DTYPE: Final = np.dtype("<u2")
"""Counts are unsigned 16-bit, little-endian.

CLAVR-x does not byte-swap and does not detect byte order. It gates on
directory word 2 being 4; a big-endian file fails that test and is then
ignored entirely rather than reported.
"""


def encode_counts(counts: NDArray[np.integer]) -> bytes:
    """Encode an image of raw counts as the AREA data block.

    A single-band file has no line prefix, so lines are contiguous.

    Parameters
    ----------
    counts
        Two-dimensional array of raw counts, shaped ``(lines, elements)``.

    Returns
    -------
    bytes
        ``lines * elements * 2`` bytes, little-endian.

    Raises
    ------
    ValueError
        If the array is not two-dimensional.
    """
    if counts.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"expected a 2-D image, got shape {counts.shape}")
    # A dtype cast wraps silently, and a wrapped count is calibrated by
    # CLAVR-x as a real measurement. SEVIRI counts are 10-bit, so anything
    # outside the unsigned 16-bit range is a bug upstream, not data.
    low, high = int(counts.min()), int(counts.max())
    if low < 0 or high > COUNT_MAX:
        raise ValueError(
            f"counts outside 0..{COUNT_MAX} (saw {low}..{high}); "
            "writing them would wrap silently",
        )
    return np.ascontiguousarray(counts, dtype=COUNT_DTYPE).tobytes()


def encode_comment_cards(cards: tuple[str, ...]) -> bytes:
    """Encode the trailing AUDIT cards: fixed-width 80-character ASCII."""
    return b"".join(
        card.encode("ascii")[: words.COMMENT_CARD_BYTES].ljust(
            words.COMMENT_CARD_BYTES,
        )
        for card in cards
    )


def build_area_file(
    meta: SeviriAreaMetadata,
    nav_block: bytes,
    cal_block: bytes,
    counts: NDArray[np.integer],
) -> bytes:
    """Assemble a complete single-band AREA file in memory.

    Parameters
    ----------
    meta
        Directory metadata. Its ``n_lines``/``n_elements`` must match
        ``counts``.
    nav_block, cal_block
        Pre-built navigation and calibration blocks.
    counts
        Raw counts, shaped ``(meta.n_lines, meta.n_elements)``.

    Returns
    -------
    bytes
        The whole file, ``meta.total_bytes`` long.

    Raises
    ------
    ValueError
        If a block is the wrong size or the image shape disagrees with the
        metadata.
    """
    _check_block_sizes(nav_block, cal_block)
    if counts.shape != (meta.n_lines, meta.n_elements):
        raise ValueError(
            f"counts shape {counts.shape} does not match metadata "
            f"{(meta.n_lines, meta.n_elements)}",
        )

    contents = (
        build_directory(meta)
        + nav_block
        + cal_block
        + encode_counts(counts)
        + encode_comment_cards(meta.comment_cards)
    )
    if len(contents) != meta.total_bytes:
        raise ValueError(
            f"assembled {len(contents)} bytes, expected {meta.total_bytes}",
        )
    return contents


def _check_block_sizes(nav_block: bytes, cal_block: bytes) -> None:
    """Reject blocks whose length would shift every offset after them."""
    if len(nav_block) != NAV_BLOCK_BYTES:
        raise ValueError(
            f"navigation block is {len(nav_block)} bytes, expected {NAV_BLOCK_BYTES}",
        )
    if len(cal_block) != CAL_BLOCK_BYTES:
        raise ValueError(
            f"calibration block is {len(cal_block)} bytes, expected {CAL_BLOCK_BYTES}",
        )


def write_area_file(
    path: Path,
    meta: SeviriAreaMetadata,
    nav_block: bytes,
    cal_block: bytes,
    counts: NDArray[np.integer],
) -> Path:
    """Write one single-band AREA file to disk.

    The only side-effecting function in this module.

    Returns
    -------
    Path
        The path written, for convenience in building output lists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = build_area_file(meta, nav_block, cal_block, counts)
    # Rename into place rather than writing in place. A 27 MB write is not
    # instantaneous, and the directory block lands first: a poller that globs
    # the output directory would otherwise find a file that parses as a valid
    # AREA file while most of its pixels are still missing.
    tmp = path.with_name(f".{path.name}.partial")
    try:
        tmp.write_bytes(contents)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


__all__ = [
    "BYTES_PER_PIXEL",
    "CAL_OFFSET",
    "COUNT_DTYPE",
    "DATA_OFFSET",
    "NAV_OFFSET",
    "build_area_file",
    "encode_comment_cards",
    "encode_counts",
    "write_area_file",
]
