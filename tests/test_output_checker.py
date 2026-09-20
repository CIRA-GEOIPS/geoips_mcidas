"""The AREA output checker, and the parser it relies on."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from geoips_mcidas.area import directory, msg_cal, msgt_nav, parse, words, writer

pytest.importorskip("geoips")
from geoips_mcidas.plugins.classes.output_checkers import mcidas_area
from geoips_mcidas.plugins.classes.output_formatters import mcidas_area_seviri
from geoips_mcidas.plugins.classes.readers import seviri_hrit_counts
from geoips_mcidas.seviri.counts import SeviriCounts
from tests.conftest import require_free_bytes

OBSERVED = datetime(2021, 5, 21, 1, 0, tzinfo=UTC)
CREATED = datetime(2026, 9, 18, 15, 45, 41, tzinfo=UTC)
SMALL = 4


def make_file(path: Path, *, created: datetime = CREATED, seed: int = 0) -> Path:
    meta = directory.SeviriAreaMetadata(
        platform_name="meteosat-11",
        observation_time=OBSERVED,
        band=1,
        creation_time=created,
        n_lines=SMALL,
        n_elements=SMALL,
        comment_cards=(f"produced at {created:%H%M%S}",),
    )
    counts = np.full((SMALL, SMALL), 100 + seed, dtype=np.uint16)
    return writer.write_area_file(
        path,
        meta,
        msgt_nav.build_msgt_nav_block(0.0, band=1),
        msg_cal.build_msg_cal_block([1.0] * 12, [0.0] * 12),
        counts,
    )


@pytest.fixture
def checker() -> mcidas_area.McidasAreaOutputCheckerPlugin:
    return mcidas_area.PLUGIN_CLASS()


def test_round_trips_through_the_parser(tmp_path: Path) -> None:
    raw = make_file(tmp_path / "a.area").read_bytes()
    blocks = parse.parse_area_file(raw)
    assert blocks.directory[4:8] == b"\x04\x00\x00\x00"
    assert blocks.navigation[:4] == b"MSGT"
    assert len(blocks.navigation) == msgt_nav.NAV_BLOCK_BYTES
    assert len(blocks.calibration) == msg_cal.CAL_BLOCK_BYTES
    assert len(blocks.data) == SMALL * SMALL * 2
    assert len(blocks.comments) == words.COMMENT_CARD_BYTES


def test_parser_rejects_non_area_bytes() -> None:
    with pytest.raises(ValueError, match="not an AREA file"):
        parse.parse_area_file(b"\x00" * 300)


def test_recognises_area_files_by_magic_not_extension(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """AREA files are conventionally named AREAnnnn with no suffix."""
    unsuffixed = make_file(tmp_path / "AREA0001")
    assert checker.correct_file_format(str(unsuffixed))


def test_declines_other_files(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("not an area file at all")
    assert not checker.correct_file_format(str(other))
    assert not checker.correct_file_format(str(tmp_path / "absent.area"))


def test_identical_files_match(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    a = make_file(tmp_path / "a.area")
    b = make_file(tmp_path / "b.area")
    assert checker.outputs_match(str(a), str(b))


def test_creation_time_is_ignored(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """Words 17 and 18 change on every run and say nothing about the data."""
    a = make_file(tmp_path / "a.area")
    b = make_file(tmp_path / "b.area", created=datetime(2030, 1, 1, tzinfo=UTC))
    assert checker.outputs_match(str(a), str(b))


def test_differing_pixels_do_not_match(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    a = make_file(tmp_path / "a.area")
    b = make_file(tmp_path / "b.area", seed=1)
    assert not checker.outputs_match(str(a), str(b))


def test_differing_navigation_does_not_match(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """A sub-satellite longitude error is invisible in the imagery but
    displaces every pixel, so the checker has to catch it."""
    meta = directory.SeviriAreaMetadata(
        platform_name="meteosat-11",
        observation_time=OBSERVED,
        band=1,
        creation_time=CREATED,
        n_lines=SMALL,
        n_elements=SMALL,
    )
    counts = np.zeros((SMALL, SMALL), dtype=np.uint16)
    cal = msg_cal.build_msg_cal_block([1.0] * 12, [0.0] * 12)
    a = writer.write_area_file(
        tmp_path / "a.area",
        meta,
        msgt_nav.build_msgt_nav_block(0.0, band=1),
        cal,
        counts,
    )
    b = writer.write_area_file(
        tmp_path / "b.area",
        meta,
        msgt_nav.build_msgt_nav_block(45.5, band=1),
        cal,
        counts,
    )
    assert not checker.outputs_match(str(a), str(b))


@pytest.mark.golden
def test_recognises_real_mcidas_output(
    reference_area: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    assert checker.correct_file_format(str(reference_area))


def test_a_truncated_file_is_rejected_not_silently_matched(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """Slicing past the end of bytes is silent, so a truncated file used to
    yield an empty data block -- and two such files with different pixels
    compared as identical, defeating the checker entirely."""
    a = make_file(tmp_path / "a.area", seed=0)
    b = make_file(tmp_path / "b.area", seed=1)
    for path in (a, b):
        path.write_bytes(path.read_bytes()[: words.DIRECTORY_BYTES])

    assert not checker.outputs_match(str(a), str(b))
    with pytest.raises(ValueError, match=r"outside the file|directory describes"):
        parse.parse_area_file(a.read_bytes())


@pytest.mark.parametrize("cards", [-1, 1000])
def test_an_impossible_comment_count_is_rejected(tmp_path: Path, cards: int) -> None:
    raw = bytearray(make_file(tmp_path / "a.area").read_bytes())
    start = (words.COMMENT_CARDS - 1) * words.WORD_BYTES
    raw[start : start + 4] = cards.to_bytes(4, "little", signed=True)
    with pytest.raises(ValueError, match=r"negative comment card|directory describes"):
        parse.parse_area_file(bytes(raw))


def test_directory_word_rejects_an_out_of_range_word(tmp_path: Path) -> None:
    raw = make_file(tmp_path / "a.area").read_bytes()
    with pytest.raises(ValueError, match="outside 1"):
        parse.directory_word(raw, 65)


def test_a_differing_directory_word_is_caught(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """Guards the guard: the volatile-word allowlist must not swallow
    everything else."""
    a = make_file(tmp_path / "a.area")
    raw = bytearray(a.read_bytes())
    start = (words.SENSOR_SOURCE - 1) * words.WORD_BYTES
    raw[start : start + 4] = (52).to_bytes(4, "little", signed=True)
    b = tmp_path / "b.area"
    b.write_bytes(bytes(raw))
    assert not checker.outputs_match(str(a), str(b))


@pytest.mark.golden
@pytest.mark.parametrize("band", [1, 5, 9, 11])
def test_accepts_a_real_mcidas_file_as_the_reference(
    reference_area: Path,
    sample_counts: SeviriCounts,
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
    band: int,
) -> None:
    """The comparison that matters most: our output against McIDAS's own.

    The reference is McIDAS's multi-band intermediate and ours is one of the
    single-band files CLAVR-x reads, so the band-layout words differ by
    construction and the pixels are compared plane for plane.
    """
    require_free_bytes(tmp_path, 60_000_000)
    scene = mcidas_area_seviri.scene_from_attrs(
        seviri_hrit_counts.counts_to_dataset(sample_counts).attrs,
    )
    written = mcidas_area_seviri.write_seviri_bands(
        {band: sample_counts.counts[band]},
        sample_counts.gains,
        sample_counts.offsets,
        scene,
        tmp_path,
    )
    assert checker.outputs_match(written[0], str(reference_area))


@pytest.mark.golden
def test_rejects_a_reference_for_a_different_band(
    reference_area: Path,
    sample_counts: SeviriCounts,
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """Guards the guard: matching against a multi-band reference must still
    compare the right plane, not merely find one that fits."""
    require_free_bytes(tmp_path, 60_000_000)
    scene = mcidas_area_seviri.scene_from_attrs(
        seviri_hrit_counts.counts_to_dataset(sample_counts).attrs,
    )
    written = mcidas_area_seviri.write_seviri_bands(
        {9: sample_counts.counts[5]},  # band 5 pixels labelled as band 9
        sample_counts.gains,
        sample_counts.offsets,
        scene,
        tmp_path,
    )
    assert not checker.outputs_match(written[0], str(reference_area))


def test_provenance_differences_do_not_block_a_match(
    tmp_path: Path,
    checker: mcidas_area.McidasAreaOutputCheckerPlugin,
) -> None:
    """A McIDAS file records the commands that made it; we record none."""
    a = make_file(tmp_path / "a.area")
    raw = bytearray(make_file(tmp_path / "b.area").read_bytes())
    for word, value in (
        (words.CREATION_DATE, 199001),
        (words.CREATION_TIME, 123456),
        (words.AREA_NUMBER, 7),
    ):
        start = (word - 1) * words.WORD_BYTES
        raw[start : start + 4] = value.to_bytes(4, "little", signed=True)
    b = tmp_path / "b.area"
    b.write_bytes(bytes(raw))
    assert checker.outputs_match(str(a), str(b))
