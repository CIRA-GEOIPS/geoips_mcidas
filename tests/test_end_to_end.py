"""HRIT in, AREA files out, checked against real McIDAS output.

This is the whole deliverable: the reader, the blocks, the filenames and the
writer, exercised on the same scene McIDAS-X converted.

A full-disk band is ~28 MB and a scene is eleven of them, so most of these
tests write one band at a time and delete it, keeping peak disk use to a
single file. Only the filename test writes a whole scene, and it skips when
the volume cannot hold one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geoips")
xr = pytest.importorskip("xarray")

from geoips_mcidas.area import msg_cal, parse, words
from geoips_mcidas.plugins.classes.output_checkers import mcidas_area
from geoips_mcidas.plugins.classes.output_formatters import mcidas_area_seviri
from geoips_mcidas.plugins.classes.readers import seviri_hrit_counts
from tests.conftest import require_free_bytes
from tests.test_seviri_counts import reference_band_plane

FULL_DISK = 3712
BAND_FILE_BYTES = 2368 + FULL_DISK * FULL_DISK * 2
SCENE_BYTES = BAND_FILE_BYTES * 11
EXPECTED_NAMES = [f"met11_{band}_2021_141_0100.area" for band in range(1, 12)]


def write_one_band(counts: object, band: int, target: Path) -> Path:
    """Write a single band file, bypassing the all-bands policy.

    The policy lives in the plugin's ``call``; the writer underneath is happy
    with a subset, which is what lets these tests stay within one file's
    worth of disk.
    """
    scene = mcidas_area_seviri.scene_from_attrs(
        seviri_hrit_counts.counts_to_dataset(counts).attrs,  # type: ignore[arg-type]
    )
    written = mcidas_area_seviri.write_seviri_bands(
        {band: counts.counts[band]},  # type: ignore[attr-defined]
        counts.gains,  # type: ignore[attr-defined]
        counts.offsets,  # type: ignore[attr-defined]
        scene,
        target,
    )
    return Path(written[0])


@pytest.mark.golden
@pytest.mark.parametrize("band", [1, 9, 11])
def test_pixels_match_mcidas(
    sample_counts: object,
    reference_area: Path,
    tmp_path: Path,
    band: int,
) -> None:
    """Our band file must hold exactly the pixels McIDAS wrote for it."""
    require_free_bytes(tmp_path, BAND_FILE_BYTES * 2)
    path = write_one_band(sample_counts, band, tmp_path)
    blocks = parse.parse_area_file(path.read_bytes())
    ours = np.frombuffer(blocks.data, dtype="<u2").reshape(FULL_DISK, FULL_DISK)
    np.testing.assert_array_equal(ours, reference_band_plane(reference_area, band))


@pytest.mark.golden
def test_file_layout_is_exact(sample_counts: object, tmp_path: Path) -> None:
    """2368 + 3712 * 3712 * 2: no line prefix on a single-band file."""
    require_free_bytes(tmp_path, BAND_FILE_BYTES * 2)
    path = write_one_band(sample_counts, 1, tmp_path)
    assert path.stat().st_size == BAND_FILE_BYTES

    header = words.unpack_words(path.read_bytes()[: words.DIRECTORY_BYTES])
    assert header[words.SENSOR_SOURCE - 1] == 354
    assert header[words.NUM_LINES - 1] == FULL_DISK
    assert header[words.NUM_ELEMENTS - 1] == FULL_DISK
    assert header[words.NUM_BANDS - 1] == 1
    assert header[words.PREFIX_BYTES - 1] == 0


@pytest.mark.golden
def test_calibration_block_matches_mcidas(
    sample_counts: object,
    reference_area: Path,
    tmp_path: Path,
) -> None:
    """Every band file carries all twelve records, because CLAVR-x reads them
    all out of the band-1 file."""
    require_free_bytes(tmp_path, BAND_FILE_BYTES * 2)
    path = write_one_band(sample_counts, 5, tmp_path)
    ours_block = parse.parse_area_file(path.read_bytes()).calibration

    with reference_area.open("rb") as handle:
        handle.seek(768)
        theirs_block = handle.read(msg_cal.CAL_BLOCK_BYTES)

    for band in range(1, 12):
        ours = msg_cal.parse_cal_record(ours_block, band)
        theirs = msg_cal.parse_cal_record(theirs_block, band)
        assert ours.gain == theirs.gain, f"band {band} gain"
        assert ours.offset == theirs.offset, f"band {band} offset"


@pytest.mark.golden
def test_navigation_records_the_band(sample_counts: object, tmp_path: Path) -> None:
    """Nav word 7 is the band; word 6 the sub-satellite longitude."""
    require_free_bytes(tmp_path, BAND_FILE_BYTES * 2)
    path = write_one_band(sample_counts, 7, tmp_path)
    nav = words.unpack_words(parse.parse_area_file(path.read_bytes()).navigation)
    assert nav[0:1] != [0]
    assert nav[6] == 7
    assert nav[5] == 0


@pytest.mark.golden
def test_checker_accepts_real_output(
    sample_counts: object,
    tmp_path: Path,
) -> None:
    require_free_bytes(tmp_path, BAND_FILE_BYTES * 3)
    first = write_one_band(sample_counts, 1, tmp_path)
    second = write_one_band(sample_counts, 2, tmp_path)
    checker = mcidas_area.PLUGIN_CLASS()
    assert checker.correct_file_format(str(first))
    assert checker.outputs_match(str(first), str(first))
    assert not checker.outputs_match(str(first), str(second))


@pytest.mark.golden
def test_full_scene_writes_every_band(sample_counts: object, tmp_path: Path) -> None:
    """The real path: one call, eleven files, names CLAVR-x can reconstruct."""
    require_free_bytes(tmp_path, SCENE_BYTES + BAND_FILE_BYTES)
    dataset = seviri_hrit_counts.counts_to_dataset(sample_counts)  # type: ignore[arg-type]
    tree = xr.DataTree(dataset=dataset, name="DATA")
    result = mcidas_area_seviri.PLUGIN_CLASS()(tree, tmp_path)
    written = [Path(p).name for p in result.attrs["output_products"]]
    assert written == EXPECTED_NAMES
