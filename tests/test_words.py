"""Word packing primitives.

The 1-based McIDAS / 0-based array off-by-one is this package's stated
likeliest source of bugs, so the boundaries get their own tests.
"""

from __future__ import annotations

import pytest

from geoips_mcidas.area import words


@pytest.mark.parametrize("word", [0, -1, words.DIRECTORY_WORDS + 1])
def test_pack_words_rejects_an_out_of_range_word(word: int) -> None:
    with pytest.raises(ValueError, match="outside 1"):
        words.pack_words({word: 1}, words.DIRECTORY_WORDS)


def test_pack_words_accepts_both_ends() -> None:
    block = words.pack_words({1: 7, words.DIRECTORY_WORDS: 9}, words.DIRECTORY_WORDS)
    unpacked = words.unpack_words(block)
    assert unpacked[0] == 7
    assert unpacked[-1] == 9


def test_encode_word_rejects_an_overlong_tag() -> None:
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        words.encode_word("MSGT!")


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1])
def test_encode_word_rejects_an_unrepresentable_integer(value: int) -> None:
    """The docstring promises ValueError; int.to_bytes raises OverflowError."""
    with pytest.raises(ValueError, match="does not fit"):
        words.encode_word(value)


@pytest.mark.parametrize("value", [0, 1, -1, 2**31 - 1, -(2**31)])
def test_integer_words_round_trip(value: int) -> None:
    assert words.decode_word(words.encode_word(value)) == value


@pytest.mark.parametrize("tag", ["MSGT", "RAW ", "MSG ", "    "])
def test_ascii_words_round_trip_in_natural_order(tag: str) -> None:
    assert words.decode_ascii_word(words.encode_word(tag)) == tag


def test_short_tags_are_space_padded() -> None:
    assert words.encode_word("K") == b"K   "


@pytest.mark.parametrize(
    ("year", "day"),
    [(1900, 1), (2021, 141), (2024, 366), (2099, 365)],
)
def test_yyddd_round_trips(year: int, day: int) -> None:
    assert words.decode_yyddd(words.encode_yyddd(year, day)) == (year, day)


def test_memo_spans_exactly_eight_words() -> None:
    encoded = words.encode_memo("x" * 100)
    assert len(encoded) == words.MEMO_WORDS
    assert set(encoded) == set(range(words.MEMO, words.MEMO + words.MEMO_WORDS))
