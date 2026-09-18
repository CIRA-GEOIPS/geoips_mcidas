"""The DataTree-native output formatter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
mcidas_area_seviri = pytest.importorskip(
    "geoips_mcidas.plugins.classes.output_formatters.mcidas_area_seviri",
)

from geoips_mcidas.area import directory, words  # noqa: E402

OBSERVED = datetime(2021, 5, 21, 1, 0, tzinfo=UTC)
SMALL = 4


def build_tree(bands: range | tuple[int, ...] = range(1, 12)) -> xr.DataTree:
    """A minimal tree shaped like what the SEVIRI counts reader produces."""
    data_vars = {
        mcidas_area_seviri.count_variable(band): (
            ("y", "x"),
            np.full((SMALL, SMALL), band * 10, dtype=np.uint16),
        )
        for band in bands
    }
    dataset = xr.Dataset(
        data_vars,
        attrs={
            mcidas_area_seviri.PLATFORM_ATTR: "meteosat-11",
            mcidas_area_seviri.START_TIME_ATTR: OBSERVED,
            mcidas_area_seviri.SUB_LON_ATTR: 0.0,
            mcidas_area_seviri.GAINS_ATTR: [1.0] * 12,
            mcidas_area_seviri.OFFSETS_ATTR: [0.0] * 12,
        },
    )
    return xr.DataTree(dataset=dataset, name="root")


def test_plugin_is_datatree_native() -> None:
    """No family, so the interface passes data through unconverted."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS
    assert plugin.data_tree is True
    assert not hasattr(plugin, "family")


def test_writes_one_file_per_band(tmp_path: Path) -> None:
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path)
    written = tree.attrs["output_products"]
    assert len(written) == 11
    assert sorted(p.name for p in tmp_path.glob("*.area")) == sorted(
        Path(p).name for p in written
    )


def test_records_output_products_on_the_tree(tmp_path: Path) -> None:
    """A family-less plugin hands results to the procflow this way; returning
    a plain list instead drops them silently."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path)
    assert isinstance(tree, xr.DataTree)
    assert Path(tree.attrs["output_products"][0]).name == "met11_1_2021_141_0100.area"


def test_all_bands_land_in_one_directory(tmp_path: Path) -> None:
    """CLAVR-x prepends a single input path to every derived sibling name."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path)
    parents = {Path(p).parent for p in tree.attrs["output_products"]}
    assert parents == {tmp_path}


def test_written_files_are_readable_area_files(tmp_path: Path) -> None:
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path)
    band_1 = Path(tree.attrs["output_products"][0])
    block = band_1.read_bytes()[: words.DIRECTORY_BYTES]
    unpacked = words.unpack_words(block)
    assert unpacked[words.VERSION - 1] == 4
    assert unpacked[words.SENSOR_SOURCE - 1] == 354
    assert unpacked[words.NUM_BANDS - 1] == 1
    assert unpacked[words.NUM_LINES - 1] == SMALL


def test_band_counts_round_trip(tmp_path: Path) -> None:
    """Each file must hold its own band's counts, not another's."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path)
    for path in map(Path, tree.attrs["output_products"]):
        band = int(path.name.split("_")[1])
        payload = path.read_bytes()[directory.DATA_OFFSET :]
        counts = np.frombuffer(payload, dtype="<u2")
        assert set(counts.tolist()) == {band * 10}


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        (tuple(range(2, 12)), "[1]"),
        (tuple(range(1, 11)), "[11]"),
        ((1, 2, 3), "[4, 5, 6, 7, 8, 9, 10, 11]"),
    ],
)
def test_missing_bands_fail_before_writing(
    tmp_path: Path,
    present: tuple[int, ...],
    missing: str,
) -> None:
    """CLAVR-x exits mid-run on a file it cannot open, naming a path we never
    wrote. Failing here names the bands instead."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    with pytest.raises(mcidas_area_seviri.MissingBandsError, match=r"\[1"):
        plugin(build_tree(present), tmp_path)
    assert list(tmp_path.glob("*.area")) == []
    with pytest.raises(mcidas_area_seviri.MissingBandsError) as excinfo:
        mcidas_area_seviri.check_band_coverage(set(present))
    assert missing in str(excinfo.value)


def test_complete_band_set_is_accepted() -> None:
    assert mcidas_area_seviri.check_band_coverage(set(range(1, 12))) is None


def test_missing_attributes_are_named(tmp_path: Path) -> None:
    tree = build_tree()
    del tree.attrs[mcidas_area_seviri.SUB_LON_ATTR]
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    with pytest.raises(KeyError, match="longitude_of_projection_origin"):
        plugin(tree, tmp_path)
