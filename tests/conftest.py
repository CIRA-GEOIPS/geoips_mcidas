"""Shared fixtures, including access to the McIDAS reference AREA file."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

if TYPE_CHECKING:
    from geoips_mcidas.seviri.counts import SeviriCounts

REFERENCE_ENV_VAR = "GEOIPS_MCIDAS_REFERENCE_AREA"
"""Overrides the search for the reference file, e.g. in CI."""

STRICT_ENV_VAR = "GEOIPS_MCIDAS_STRICT_GOLDEN"
"""Set in CI: a configured reference that is missing is an error, not a skip.

Without it the golden job is green when the data is unreachable, which is the
exact failure the job exists to catch.
"""


def _missing(what: str, env_var: str) -> NoReturn:
    """Skip, or fail when the caller said the data must be there."""
    message = f"{what} not found; set {env_var} to enable"
    if os.environ.get(STRICT_ENV_VAR):
        pytest.fail(f"{message} (strict mode via {STRICT_ENV_VAR})")
    pytest.skip(message)


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "recast_MSG"
    / "prog"
    / "mcidas"
    / "data"
    / "AREA0001"
)
"""A genuine McIDAS-X multi-band AREA file: Meteosat-11, 2021-05-21 01:00."""


def _reference_path() -> Path | None:
    override = os.environ.get(REFERENCE_ENV_VAR)
    candidate = Path(override) if override else DEFAULT_REFERENCE
    return candidate if candidate.is_file() else None


@pytest.fixture(scope="session")
def reference_area() -> Path:
    """Path to the reference AREA file, skipping the test when it is absent.

    The file is 303 MB and lives outside the repository, so golden tests
    deselect themselves rather than fail on a fresh clone or in CI.
    """
    path = _reference_path()
    if path is None:
        _missing("reference AREA file", REFERENCE_ENV_VAR)
    return path


@pytest.fixture(scope="session")
def reference_directory(reference_area: Path) -> bytes:
    """The 256-byte directory block of the reference file."""
    with reference_area.open("rb") as handle:
        return handle.read(256)


DEFAULT_HRIT_DIR = (
    Path(__file__).resolve().parents[2] / "recast_MSG" / "prog" / "sample_hrit"
)
"""The HRIT segments McIDAS converted into the reference AREA file."""

HRIT_ENV_VAR = "GEOIPS_MCIDAS_SAMPLE_HRIT"


@pytest.fixture(scope="session")
def sample_hrit() -> Path:
    """Directory of HRIT files matching the reference AREA file exactly.

    Same platform, same minute -- which is what makes a pixel-for-pixel
    comparison between our output and McIDAS's meaningful.
    """
    override = os.environ.get(HRIT_ENV_VAR)
    candidate = Path(override) if override else DEFAULT_HRIT_DIR
    if not candidate.is_dir():
        _missing("sample HRIT", HRIT_ENV_VAR)
    return candidate


@pytest.fixture(scope="session")
def sample_counts(sample_hrit: Path) -> SeviriCounts:
    """Counts read once per session.

    Reading and 10-bit-unpacking 90 HRIT segments takes about twenty seconds,
    so every golden test in the suite shares one result.
    """
    pytest.importorskip("geoips")
    from geoips_mcidas.seviri.counts import read_seviri_counts

    return read_seviri_counts(sample_hrit.glob("H-*"))


def require_free_bytes(path: Path, needed: int) -> None:
    """Skip a test when the filesystem cannot hold what it would write.

    A full-disk SEVIRI band is about 28 MB and a whole scene is eleven of
    them, so these tests can genuinely exhaust a tight volume. Skipping is
    better than failing with a disk error that looks like a code fault.
    """
    usage = shutil.disk_usage(path)
    if usage.free < needed:
        pytest.skip(
            f"needs {needed / 1e6:.0f} MB free, only "
            f"{usage.free / 1e6:.0f} MB available",
        )
