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
    """Read one 1-based directory word as a signed integer.

    Raises
    ------
    ValueError
        If the word number is outside 1..64. Slicing would otherwise return a
        short or empty run of bytes and decode it to a plausible-looking
        number, where the writer's :func:`~geoips_mcidas.area.words.pack_words`
        rejects the same mistake.
    """
    if not 1 <= word <= words.DIRECTORY_WORDS:
        raise ValueError(f"word {word} outside 1..{words.DIRECTORY_WORDS}")
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
    _check_consistent(raw, nav_offset, cal_offset, data_offset, n_cards)

    comment_bytes = n_cards * words.COMMENT_CARD_BYTES
    data_end = len(raw) - comment_bytes
    nav_end = cal_offset or data_offset
    return AreaBlocks(
        directory=raw[: words.DIRECTORY_BYTES],
        navigation=raw[nav_offset:nav_end] if nav_offset else b"",
        calibration=raw[cal_offset:data_offset] if cal_offset else b"",
        data=raw[data_offset:data_end],
        comments=raw[data_end:],
    )


def _check_consistent(
    raw: bytes,
    nav_offset: int,
    cal_offset: int,
    data_offset: int,
    n_cards: int,
) -> None:
    """Reject a file whose bytes do not match the geometry it declares.

    Slicing past the end of a ``bytes`` is silent, so a truncated file yields
    an empty data block rather than an error -- and two truncated files with
    different pixels then compare as identical, which defeats the one tool
    that exists to catch bad output. The directory already carries the
    authoritative geometry, so check against it.

    Raises
    ------
    ValueError
        If an offset is out of range, the offsets are not in block order, or
        the file is not the length the directory implies.
    """
    if n_cards < 0:
        raise ValueError(f"negative comment card count: {n_cards}")
    for name, offset in (
        ("navigation", nav_offset),
        ("calibration", cal_offset),
        ("data", data_offset),
    ):
        if not 0 <= offset <= len(raw):
            raise ValueError(
                f"{name} offset {offset} outside the file (length {len(raw)})",
            )
    present = [o for o in (nav_offset, cal_offset, data_offset) if o]
    if present != sorted(present):
        raise ValueError(
            f"block offsets are not in order: nav {nav_offset}, "
            f"cal {cal_offset}, data {data_offset}",
        )

    line_bytes = directory_word(raw, words.PREFIX_BYTES) + (
        directory_word(raw, words.NUM_ELEMENTS)
        * directory_word(raw, words.NUM_BANDS)
        * directory_word(raw, words.BYTES_PER_PIXEL)
    )
    expected = (
        data_offset
        + directory_word(raw, words.NUM_LINES) * line_bytes
        + n_cards * words.COMMENT_CARD_BYTES
    )
    if len(raw) != expected:
        raise ValueError(
            f"file is {len(raw)} bytes; the directory describes {expected} "
            "(truncated, padded, or the directory is wrong)",
        )
