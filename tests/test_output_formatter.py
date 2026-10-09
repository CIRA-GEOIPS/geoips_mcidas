"""The DataTree-native output formatter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("xarray")
mcidas_area_seviri = pytest.importorskip(
    "geoips_mcidas.plugins.classes.output_formatters.mcidas_area_seviri",
)

from geoips_mcidas.area import directory, words
from geoips_mcidas.errors import MissingBandsError
from geoips_mcidas.seviri import area_files

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
    tree = plugin(build_tree(), tmp_path, allow_subset=True)
    written = tree.attrs["output_products"]
    assert len(written) == 11
    assert sorted(p.name for p in tmp_path.glob("*.area")) == sorted(
        Path(p).name for p in written
    )


def test_records_output_products_on_the_tree(tmp_path: Path) -> None:
    """A family-less plugin hands results to the procflow this way; returning
    a plain list instead drops them silently."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path, allow_subset=True)
    assert isinstance(tree, xr.DataTree)
    assert Path(tree.attrs["output_products"][0]).name == "met11_1_2021_141_0100.area"


def test_all_bands_land_in_one_directory(tmp_path: Path) -> None:
    """CLAVR-x prepends a single input path to every derived sibling name."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path, allow_subset=True)
    parents = {Path(p).parent for p in tree.attrs["output_products"]}
    assert parents == {tmp_path}


def test_written_files_are_readable_area_files(tmp_path: Path) -> None:
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    tree = plugin(build_tree(), tmp_path, allow_subset=True)
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
    tree = plugin(build_tree(), tmp_path, allow_subset=True)
    for path in map(Path, tree.attrs["output_products"]):
        band = int(path.name.split("_")[1])
        payload = path.read_bytes()[directory.DATA_BLOCK_OFFSET :]
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
    with pytest.raises(MissingBandsError, match=r"\[1"):
        plugin(build_tree(present), tmp_path, allow_subset=True)
    assert list(tmp_path.glob("*.area")) == []
    with pytest.raises(MissingBandsError) as excinfo:
        mcidas_area_seviri.check_band_coverage(set(present))
    assert missing in str(excinfo.value)


def test_complete_band_set_is_accepted() -> None:
    assert mcidas_area_seviri.check_band_coverage(set(range(1, 12))) is None


def test_missing_attributes_are_named(tmp_path: Path) -> None:
    tree = build_tree()
    del tree.attrs[mcidas_area_seviri.SUB_LON_ATTR]
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    with pytest.raises(KeyError, match="longitude_of_projection_origin"):
        plugin(tree, tmp_path, allow_subset=True)


def build_multi_input(tree: xr.DataTree) -> xr.DataTree:
    """Wrap a reader node the way the order-based procflow does.

    ``Workflow._collect_upstream_data`` builds a fresh ``multi_input`` tree per
    step and attaches each dependency as a child, so the root carries no
    variables at all.
    """
    wrapper = xr.DataTree(name="multi_input")
    wrapper["reader"] = xr.DataTree(dataset=tree.to_dataset(), name="reader")
    return wrapper


def test_finds_counts_under_the_obp_multi_input_wrapper(tmp_path: Path) -> None:
    """The shape a real order-based run delivers.

    Reading the root unconditionally reports every band missing from input
    that carries all eleven, which blames the data for a wiring mismatch.
    """
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    wrapper = build_multi_input(build_tree())
    assert list(wrapper.to_dataset().data_vars) == []

    result = plugin(wrapper, tmp_path, allow_subset=True, _obp_initiated=True)
    assert len(result.attrs["output_products"]) == 11


def test_resolver_prefers_the_root_when_it_holds_counts() -> None:
    """A directly-called tree must not be overtaken by a stray child."""
    tree = build_tree()
    tree["decoy"] = xr.DataTree(dataset=xr.Dataset(), name="decoy")
    resolved = mcidas_area_seviri.resolve_counts_dataset(tree)
    assert mcidas_area_seviri.count_variable(1) in resolved


def test_resolver_falls_back_to_the_root_when_nothing_has_counts() -> None:
    """Report through the band-coverage check, not from inside the resolver."""
    empty = xr.DataTree(name="multi_input")
    empty["reader"] = xr.DataTree(dataset=xr.Dataset(), name="reader")
    assert list(mcidas_area_seviri.resolve_counts_dataset(empty).data_vars) == []


def test_a_subset_is_refused_by_default(tmp_path: Path) -> None:
    """CLAVR-x hard-codes full-disk navigation, so a subset is mis-navigated
    silently rather than rejected."""
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    with pytest.raises(ValueError, match="full-disk"):
        plugin(build_tree(), tmp_path)
    assert list(tmp_path.glob("*.area")) == []


def test_band_one_is_written_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The band-1 file is what a downstream poller keys on, so it must not
    appear until the other ten exist."""
    order: list[int] = []
    real = area_files.writer.write_area_file

    def record(path: Path, meta: Any, *args: Any) -> Path:
        order.append(meta.band)
        written: Path = real(path, meta, *args)
        return written

    monkeypatch.setattr(area_files.writer, "write_area_file", record)
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    plugin(build_tree(), tmp_path, allow_subset=True)

    assert order[-1] == 1
    assert sorted(order) == list(range(1, 12))


@pytest.mark.parametrize(
    "bad",
    ["${GEOIPS_OUTDIRS}/seviri_area", "$HOME/out", "relative/path"],
)
def test_rejects_an_unusable_output_directory(bad: str) -> None:
    """pluginify leaves an unset ${VAR} as literal text, so the workflow cannot
    see the problem and mkdir would happily create the directory."""
    with pytest.raises(ValueError, match=r"unexpanded variable|must be absolute"):
        mcidas_area_seviri.PLUGIN_CLASS()(build_tree(), bad, allow_subset=True)


def test_band_one_does_not_survive_a_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Band 1's presence is the readiness signal, so a previous run's copy must
    not stand in for this run's.

    Filenames carry no run identity, so without an explicit removal a re-run
    that dies partway leaves eleven files with band 1 present and a silent mix
    of two runs -- in exactly the situation re-runs exist for.
    """
    plugin = mcidas_area_seviri.PLUGIN_CLASS()
    plugin(build_tree(), tmp_path, allow_subset=True)
    band_1 = tmp_path / "met11_1_2021_141_0100.area"
    assert band_1.exists()

    real = area_files.writer.write_area_file
    calls = {"n": 0}

    def fail_partway(*args: Any, **kwargs: Any) -> Path:
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("interrupted")
        written: Path = real(*args, **kwargs)
        return written

    monkeypatch.setattr(area_files.writer, "write_area_file", fail_partway)
    with pytest.raises(RuntimeError, match="interrupted"):
        plugin(build_tree(), tmp_path, allow_subset=True)

    assert not band_1.exists(), "a stale band 1 would advertise an incomplete set"
