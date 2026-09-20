"""Word numbering and packing primitives for McIDAS AREA files.

McIDAS numbers directory words from 1; C and Python index from 0. That
off-by-one is the single likeliest source of bugs in an AREA writer, so every
word number in this package is defined here once, 1-based, and nothing else is
allowed to hard-code one.

Authority: the McIDAS Programmer's Manual
(https://www.ssec.wisc.edu/mcidas/doc/prog_man/2015/formats-1.html), verified
against the directory block of a McIDAS-produced file.

Byte order is **little-endian**: AREA files are written in host order, and
CLAVR-x neither swaps nor detects. It gates on word 2 == 4 and silently ignores
any file that fails, so a big-endian file is not a loud error, it is no error
at all.
"""

from __future__ import annotations

from typing import Final

WORD_BYTES: Final = 4
"""Every AREA block is an array of 32-bit words."""

DIRECTORY_WORDS: Final = 64
"""The directory block is 64 words = 256 bytes, always at file offset 0."""

DIRECTORY_BYTES: Final = DIRECTORY_WORDS * WORD_BYTES

BYTE_ORDER: Final = "little"

# --- Directory word numbers (1-based, as McIDAS numbers them) ---------------
STATUS: Final = 1
VERSION: Final = 2
SENSOR_SOURCE: Final = 3
YEAR_DAY: Final = 4
IMAGE_TIME: Final = 5
START_LINE: Final = 6
START_ELEMENT: Final = 7
Z_COORDINATE: Final = 8
"""Image depth. McIDAS writes 1; nothing reads it."""
NUM_LINES: Final = 9
NUM_ELEMENTS: Final = 10
BYTES_PER_PIXEL: Final = 11
LINE_RESOLUTION: Final = 12
ELEMENT_RESOLUTION: Final = 13
NUM_BANDS: Final = 14
PREFIX_BYTES: Final = 15
CREATION_DATE: Final = 17
CREATION_TIME: Final = 18
BAND_MAP: Final = 19
MEMO: Final = 25
"""32 ASCII characters spanning words 25-32."""
MEMO_WORDS: Final = 8
AREA_NUMBER: Final = 33
DATA_OFFSET: Final = 34
NAV_OFFSET: Final = 35
VALIDITY_CODE: Final = 36
ACTUAL_YEAR_DAY: Final = 46
ACTUAL_TIME: Final = 47
DOC_LENGTH: Final = 49
CAL_LENGTH: Final = 50
LEVEL_LENGTH: Final = 51
SOURCE_TYPE: Final = 52
CAL_TYPE: Final = 53
AVG_SAMPLE_FLAG: Final = 54
CAL_UNIT: Final = 58
CAL_SCALE: Final = 59
AUX_OFFSET: Final = 60
AUX_COUNT: Final = 61
CAL_OFFSET: Final = 63
COMMENT_CARDS: Final = 64

VERSION_NUMBER: Final = 4
"""Value of word 2 in every valid AREA file, and the endianness sentinel."""

COMMENT_CARD_BYTES: Final = 80
"""Comment (AUDIT) cards are fixed-width ASCII, appended after the data."""

MCIDAS_EPOCH_YEAR: Final = 1900
"""Dates are YYDDD with a 1900 epoch: 2021 encodes as 121, not 2021."""

WordValue = int | str
"""A word is either a signed 32-bit integer or exactly four ASCII characters."""


def encode_word(value: WordValue) -> bytes:
    """Encode one 32-bit AREA word.

    Parameters
    ----------
    value
        An integer, or a string of at most four characters. Strings are
        written as ASCII bytes in natural order and space-padded on the right;
        McIDAS stores type tags such as ``'RAW '`` and ``'MSGT'`` this way, and
        never byte-swaps them.

    Returns
    -------
    bytes
        Exactly four bytes.

    Raises
    ------
    ValueError
        If a string is longer than four characters, or an integer does not fit
        in a signed 32-bit word. A float is truncated by ``int()`` rather than
        rejected; callers pass word values, not measurements.
    """
    if isinstance(value, str):
        encoded = value.encode("ascii")
        if len(encoded) > WORD_BYTES:
            raise ValueError(f"ASCII word {value!r} exceeds {WORD_BYTES} bytes")
        return encoded.ljust(WORD_BYTES)
    try:
        return int(value).to_bytes(WORD_BYTES, BYTE_ORDER, signed=True)
    except OverflowError as exc:
        raise ValueError(
            f"{value} does not fit a signed {WORD_BYTES}-byte word"
        ) from exc


def decode_word(raw: bytes) -> int:
    """Decode one 32-bit AREA word as a signed integer.

    ASCII words decode to a meaningless integer by design: the caller knows
    which words are tags and reads those with :func:`decode_ascii_word`.
    """
    return int.from_bytes(raw, BYTE_ORDER, signed=True)


def decode_ascii_word(raw: bytes) -> str:
    """Decode one 32-bit AREA word as its four ASCII characters."""
    return raw.decode("ascii", errors="replace")


def pack_words(values: dict[int, WordValue], n_words: int) -> bytes:
    """Pack a sparse ``{word_number: value}`` mapping into a zero-filled block.

    Parameters
    ----------
    values
        1-based word numbers to values. Words not present are left zero, which
        is what McIDAS writes for every reserved or unused word.
    n_words
        Total length of the block in words.

    Returns
    -------
    bytes
        ``n_words * 4`` bytes.

    Raises
    ------
    ValueError
        If a word number falls outside ``1..n_words``.
    """
    block = bytearray(n_words * WORD_BYTES)
    for word, value in values.items():
        if not 1 <= word <= n_words:
            raise ValueError(f"word {word} outside 1..{n_words}")
        start = (word - 1) * WORD_BYTES
        block[start : start + WORD_BYTES] = encode_word(value)
    return bytes(block)


def unpack_words(block: bytes) -> list[int]:
    """Unpack a block into signed integers, one per 32-bit word."""
    return [
        decode_word(block[i : i + WORD_BYTES])
        for i in range(0, len(block) - len(block) % WORD_BYTES, WORD_BYTES)
    ]


def encode_yyddd(year: int, day_of_year: int) -> int:
    """Encode a date as McIDAS ``YYDDD`` with its 1900 epoch.

    2021 day 141 encodes as ``121141``. Writing a four-digit year instead
    yields the year 3921, which no reader rejects.
    """
    return (year - MCIDAS_EPOCH_YEAR) * 1000 + day_of_year


def decode_yyddd(yyddd: int) -> tuple[int, int]:
    """Decode McIDAS ``YYDDD`` into ``(year, day_of_year)``."""
    return MCIDAS_EPOCH_YEAR + yyddd // 1000, yyddd % 1000


def encode_hhmmss(hour: int, minute: int, second: int = 0) -> int:
    """Encode a time as the McIDAS packed-decimal ``HHMMSS`` integer.

    CLAVR-x integer-divides the seconds field away when parsing, so seconds
    beyond zero are discarded downstream regardless of what is written here.
    """
    return hour * 10000 + minute * 100 + second


def encode_memo(text: str) -> dict[int, WordValue]:
    """Split a memo string across the 32 ASCII characters of words 25-32."""
    padded = text.encode("ascii")[: MEMO_WORDS * WORD_BYTES].ljust(
        MEMO_WORDS * WORD_BYTES,
    )
    return {
        MEMO + i: padded[i * WORD_BYTES : (i + 1) * WORD_BYTES].decode("ascii")
        for i in range(MEMO_WORDS)
    }
