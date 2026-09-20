"""The AREA directory block, including the values CLAVR-x keys behaviour on."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from geoips_mcidas.area import directory, words

OBSERVED = datetime(2021, 5, 21, 1, 0, tzinfo=UTC)
CREATED = datetime(2026, 9, 18, 15, 45, 41, tzinfo=UTC)


def metadata(**overrides: object) -> directory.SeviriAreaMetadata:
    defaults: dict[str, object] = {
        "platform_name": "meteosat-11",
        "observation_time": OBSERVED,
        "band": 1,
        "creation_time": CREATED,
    }
    defaults.update(overrides)
    return directory.SeviriAreaMetadata(**defaults)  # type: ignore[arg-type]


def directory_words(meta: directory.SeviriAreaMetadata) -> dict[int, int]:
    """Return 1-based word numbers to values, as McIDAS numbers them."""
    unpacked = words.unpack_words(directory.build_directory(meta))
    return dict(enumerate(unpacked, start=1))


def test_block_is_256_bytes() -> None:
    assert len(directory.build_directory(metadata())) == words.DIRECTORY_BYTES


def test_version_word_is_the_endianness_sentinel() -> None:
    """Readers gate on this; a byte-swapped file fails it and is ignored."""
    assert directory_words(metadata())[words.VERSION] == 4


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("meteosat-8", 51),
        ("meteosat-9", 52),
        ("meteosat-10", 53),
        ("meteosat-11", 354),
    ],
)
def test_sensor_source_numbers(platform: str, expected: int) -> None:
    """CLAVR-x matches only these four and has no default branch."""
    assert directory.sensor_source_number(platform) == expected


def test_unknown_platform_is_rejected() -> None:
    with pytest.raises(KeyError, match="himawari-9"):
        directory.sensor_source_number("himawari-9")


def test_date_uses_the_1900_epoch() -> None:
    """A four-digit year is accepted by every reader and dates the image
    two millennia late, so this is worth asserting directly."""
    assert directory_words(metadata())[words.YEAR_DAY] == 121141


def test_date_encoding_survives_a_year_boundary() -> None:
    new_year = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)
    assert words.encode_yyddd(new_year.year, new_year.timetuple().tm_yday) == 122001
    assert words.decode_yyddd(122001) == (2022, 1)


def test_upper_left_coordinates_are_one() -> None:
    """CLAVR-x applies its EUMETSAT grid flips only when both are 1.

    The McIDAS server emits 3 here; the reference pipeline only ends up with 1
    because a separate imgcha step rewrites it. We replace both steps, so we
    have to write 1 ourselves.
    """
    got = directory_words(metadata())
    assert got[words.START_LINE] == 1
    assert got[words.START_ELEMENT] == 1


def test_single_band_layout() -> None:
    """Single-band files differ from the multi-band intermediate McIDAS makes
    first: one band, no line prefix, one bit in the band map."""
    got = directory_words(metadata(band=7))
    assert got[words.NUM_BANDS] == 1
    assert got[words.PREFIX_BYTES] == 0
    assert got[words.LEVEL_LENGTH] == 0
    assert got[words.BAND_MAP] == 1 << 6


def test_block_offsets_are_contiguous() -> None:
    """Readers derive block lengths from offset differences, so a gap is not
    padding -- it becomes part of the preceding block."""
    got = directory_words(metadata())
    assert got[words.NAV_OFFSET] == 256
    assert got[words.CAL_OFFSET] == 768
    assert got[words.DATA_OFFSET] == 2368


def test_ascii_words_are_stored_in_natural_order() -> None:
    block = directory.build_directory(metadata())
    source = (words.SOURCE_TYPE - 1) * 4
    cal = (words.CAL_TYPE - 1) * 4
    assert block[source : source + 4] == b"MSG "
    assert block[cal : cal + 4] == b"RAW "


def test_memo_spans_eight_words() -> None:
    block = directory.build_directory(metadata())
    start = (words.MEMO - 1) * 4
    assert block[start : start + 32] == b"MSG data from XRIT V2".ljust(32)


def test_total_size_accounts_for_comment_cards() -> None:
    meta = metadata(comment_cards=("written by geoips_mcidas",))
    assert meta.total_bytes == 2368 + 3712 * 3712 * 2 + 80


@pytest.mark.golden
def test_shared_words_match_the_reference(reference_directory: bytes) -> None:
    """Compare against real McIDAS output for the words that do not depend on
    band count.

    The comparison is an allowlist, not an exclusion list: it asserts only the
    words listed below. The reference is McIDAS's multi-band intermediate, so
    the band-dependent words (14, 15, 19, 51), the creation stamps (17, 18)
    and the comment count (64) all legitimately differ from our single-band
    output and are therefore not listed.
    """
    theirs = words.unpack_words(reference_directory)
    ours = words.unpack_words(directory.build_directory(metadata()))

    shared = [
        words.STATUS,
        words.VERSION,
        words.SENSOR_SOURCE,
        words.YEAR_DAY,
        words.IMAGE_TIME,
        words.START_LINE,
        words.START_ELEMENT,
        words.NUM_LINES,
        words.NUM_ELEMENTS,
        words.BYTES_PER_PIXEL,
        words.LINE_RESOLUTION,
        words.ELEMENT_RESOLUTION,
        words.DATA_OFFSET,
        words.NAV_OFFSET,
        words.VALIDITY_CODE,
        words.ACTUAL_YEAR_DAY,
        words.ACTUAL_TIME,
        words.DOC_LENGTH,
        words.CAL_LENGTH,
        words.SOURCE_TYPE,
        words.CAL_TYPE,
        words.AVG_SAMPLE_FLAG,
        words.CAL_UNIT,
        words.CAL_SCALE,
        words.AUX_OFFSET,
        words.AUX_COUNT,
        words.CAL_OFFSET,
    ]
    for word in shared:
        assert ours[word - 1] == theirs[word - 1], f"word {word}"


@pytest.mark.golden
def test_memo_matches_the_reference(reference_area: Path) -> None:
    with reference_area.open("rb") as handle:
        block = handle.read(words.DIRECTORY_BYTES)
    start = (words.MEMO - 1) * 4
    ours = directory.build_directory(metadata())
    assert ours[start : start + 32] == block[start : start + 32]
