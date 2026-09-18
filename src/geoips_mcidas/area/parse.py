"""Read an AREA file back into its constituent blocks.

Block *lengths* are never stored -- they are the differences between the block
offsets in the directory. That is why the block order is fixed and why a gap
between blocks is not padding: it becomes part of whichever block precedes it.
Parsing therefore means reading the offsets first and slicing between them.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from geoips_mcidas.area import words

MIN_AREA_BYTES: Final = words.DIRECTORY_BYTES


class AreaBlocks(NamedTuple):
    """The blocks of an AREA file, sliced apart but not interpreted."""

    directory: bytes
    navigation: bytes
    calibration: bytes
    data: bytes
    comments: bytes


def is_area_file(raw: bytes) -> bool:
    """Return whether a byte string starts with a plausible AREA directory.

    The test is McIDAS's own: word 2 holds the format version, always 4.
    Because AREA files are written in host byte order with no marker, a
    big-endian file simply fails this test -- which is exactly how CLAVR-x
    ends up ignoring one rather than reporting it.
    """
    if len(raw) < MIN_AREA_BYTES:
        return False
    version = words.decode_word(raw[4:8])
    return version == words.VERSION_NUMBER


def directory_word(raw: bytes, word: int) -> int:
    """Read one 1-based directory word as a signed integer."""
    start = (word - 1) * words.WORD_BYTES
    return words.decode_word(raw[start : start + words.WORD_BYTES])


def parse_area_file(raw: bytes) -> AreaBlocks:
    """Slice an AREA file into directory, navigation, calibration and data.

    Parameters
    ----------
    raw
        The whole file.

    Returns
    -------
    AreaBlocks
        Empty ``bytes`` for any block the file does not carry.

    Raises
    ------
    ValueError
        If the file does not begin with a valid AREA directory.
    """
    if not is_area_file(raw):
        raise ValueError("not an AREA file: directory word 2 is not 4")

    nav_offset = directory_word(raw, words.NAV_OFFSET)
    cal_offset = directory_word(raw, words.CAL_OFFSET)
    data_offset = directory_word(raw, words.DATA_OFFSET)
    n_cards = directory_word(raw, words.COMMENT_CARDS)
    comment_bytes = n_cards * words.COMMENT_CARD_BYTES
    data_end = len(raw) - comment_bytes

    nav_end = cal_offset or data_offset
    return AreaBlocks(
        directory=raw[: words.DIRECTORY_BYTES],
        navigation=raw[nav_offset:nav_end] if nav_offset else b"",
        calibration=raw[cal_offset:data_offset] if cal_offset else b"",
        data=raw[data_offset:data_end],
        comments=raw[data_end:] if comment_bytes else b"",
    )
