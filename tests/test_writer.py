"""Assembling a whole AREA file, and the size identity that validates it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from geoips_mcidas.area import directory, msg_cal, msgt_nav, words, writer

OBSERVED = datetime(2021, 5, 21, 1, 0, tzinfo=UTC)
CREATED = datetime(2026, 9, 18, 15, 45, 41, tzinfo=UTC)
SMALL = 8
"""A tiny image keeps these tests fast; the format does not care about size."""


def small_metadata(**overrides: object) -> directory.SeviriAreaMetadata:
    defaults: dict[str, object] = {
        "platform_name": "meteosat-11",
        "observation_time": OBSERVED,
        "band": 1,
        "creation_time": CREATED,
        "n_lines": SMALL,
        "n_elements": SMALL,
    }
    defaults.update(overrides)
    return directory.SeviriAreaMetadata(**defaults)  # type: ignore[arg-type]


def blocks() -> tuple[bytes, bytes]:
    nav = msgt_nav.build_msgt_nav_block(0.0, band=1)
    cal = msg_cal.build_msg_cal_block([1.0] * 12, [0.0] * 12)
    return nav, cal


def ramp(lines: int = SMALL, elements: int = SMALL) -> np.ndarray:
    return np.arange(lines * elements, dtype=np.uint16).reshape(lines, elements)


def test_file_size_identity() -> None:
    """dataoff + lines * line_bytes + 80 * cards.

    This one arithmetic check exercises the block offsets, the absent line
    prefix, the pixel width and the comment cards at once -- it is how the
    layout model was validated against a real 303 MB McIDAS file.
    """
    meta = small_metadata(comment_cards=("card one", "card two"))
    nav, cal = blocks()
    contents = writer.build_area_file(meta, nav, cal, ramp())
    assert len(contents) == meta.total_bytes
    assert len(contents) == 2368 + SMALL * SMALL * 2 + 160


def test_blocks_are_contiguous_and_in_order() -> None:
    meta = small_metadata()
    nav, cal = blocks()
    contents = writer.build_area_file(meta, nav, cal, ramp())
    assert contents[words.DIRECTORY_BYTES : words.DIRECTORY_BYTES + 4] == b"MSGT"
    assert contents[directory.CAL_OFFSET : directory.CAL_OFFSET + 4] == b"MSGT"
    assert len(contents[: directory.NAV_OFFSET]) == words.DIRECTORY_BYTES


def test_counts_are_little_endian_unsigned() -> None:
    meta = small_metadata()
    nav, cal = blocks()
    counts = ramp()
    contents = writer.build_area_file(meta, nav, cal, counts)
    data = contents[directory.DATA_OFFSET : directory.DATA_OFFSET + counts.size * 2]
    recovered = np.frombuffer(data, dtype="<u2").reshape(counts.shape)
    np.testing.assert_array_equal(recovered, counts)


def test_directory_round_trips() -> None:
    meta = small_metadata(band=9)
    nav, cal = blocks()
    contents = writer.build_area_file(meta, nav, cal, ramp())
    unpacked = words.unpack_words(contents[: words.DIRECTORY_BYTES])
    assert unpacked[words.VERSION - 1] == 4
    assert unpacked[words.BAND_MAP - 1] == 1 << 8
    assert unpacked[words.NUM_LINES - 1] == SMALL


def test_rejects_a_mis_sized_nav_block() -> None:
    """A short block would shift every offset after it, silently."""
    _, cal = blocks()
    with pytest.raises(ValueError, match="navigation block"):
        writer.build_area_file(small_metadata(), b"MSGT", cal, ramp())


def test_rejects_a_mis_sized_cal_block() -> None:
    nav, _ = blocks()
    with pytest.raises(ValueError, match="calibration block"):
        writer.build_area_file(small_metadata(), nav, b"MSGT", ramp())


def test_rejects_shape_disagreement() -> None:
    nav, cal = blocks()
    with pytest.raises(ValueError, match="does not match metadata"):
        writer.build_area_file(small_metadata(), nav, cal, ramp(4, 4))


def test_rejects_non_2d_counts() -> None:
    with pytest.raises(ValueError, match="2-D image"):
        writer.encode_counts(np.zeros(4, dtype=np.uint16))


def test_comment_cards_are_fixed_width() -> None:
    encoded = writer.encode_comment_cards(("short", "x" * 200))
    assert len(encoded) == 160
    assert encoded[:80] == b"short".ljust(80)


def test_write_area_file_creates_parents(tmp_path: Path) -> None:
    meta = small_metadata()
    nav, cal = blocks()
    target = tmp_path / "nested" / "met11_1_2021_141_0100.area"
    written = writer.write_area_file(target, meta, nav, cal, ramp())
    assert written == target
    assert target.stat().st_size == meta.total_bytes
