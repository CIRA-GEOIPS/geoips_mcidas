"""The ``seviri2area`` command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geoips_mcidas import cli
from geoips_mcidas.area import parse
from tests.conftest import require_free_bytes

FULL_DISK = 3712
SCENE_BYTES = 11 * (2368 + FULL_DISK * FULL_DISK * 2)
EXPECTED_NAMES = [f"met11_{band}_2021_141_0100.area" for band in range(1, 12)]


def test_hrit_parser_loads_without_the_geoips_stack() -> None:
    """The release binary bundles numpy and nothing else, so the parser
    import must not run ``geoips/__init__``. A fresh interpreter is the only
    honest place to check what got imported."""
    pytest.importorskip("geoips")
    script = (
        "import sys\n"
        "from geoips_mcidas.cli import register_geoips_without_init\n"
        "register_geoips_without_init()\n"
        "from geoips_mcidas.seviri.counts import read_seviri_counts\n"
        "heavy = {'matplotlib', 'xarray', 'netCDF4', 'cartopy'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_directories_expand_to_their_hrit_files(tmp_path: Path) -> None:
    hrit = tmp_path / "H-000-MSG4__-PRO______-202105210100-__"
    hrit.touch()
    (tmp_path / "README").touch()
    (tmp_path / "H-subdir").mkdir()
    loose = tmp_path / "loose"
    loose.touch()
    assert cli.expand_inputs([tmp_path, loose]) == [hrit, loose]


def test_a_missing_input_is_an_error_not_a_skip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([str(tmp_path / "absent"), "-o", str(tmp_path)]) == 1
    assert "no such file or directory" in capsys.readouterr().err


def test_version_reports_the_package(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("seviri2area ")


@pytest.mark.golden
def test_converts_a_scene_to_what_mcidas_wrote(
    sample_hrit: Path,
    reference_area: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Directory in, eleven files out, band 1 listed first for CLAVR-x."""
    pytest.importorskip("geoips")
    require_free_bytes(tmp_path, SCENE_BYTES)
    assert cli.main([str(sample_hrit), "-o", str(tmp_path)]) == 0

    printed = capsys.readouterr().out.split()
    assert [Path(p).name for p in printed] == EXPECTED_NAMES
    blocks = parse.parse_area_file(Path(printed[0]).read_bytes())
    ours = np.frombuffer(blocks.data, dtype="<u2").reshape(FULL_DISK, FULL_DISK)
    theirs = parse.band_plane(reference_area.read_bytes(), 1)
    np.testing.assert_array_equal(ours, theirs)
