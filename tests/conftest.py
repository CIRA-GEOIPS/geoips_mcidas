"""Shared fixtures, including access to the McIDAS reference AREA file."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REFERENCE_ENV_VAR = "GEOIPS_MCIDAS_REFERENCE_AREA"
"""Overrides the search for the reference file, e.g. in CI."""

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
        pytest.skip(
            f"reference AREA file not found; set {REFERENCE_ENV_VAR} to enable",
        )
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
        pytest.skip(f"sample HRIT not found; set {HRIT_ENV_VAR} to enable")
    return candidate


@pytest.fixture(scope="session")
def sample_counts(sample_hrit: Path) -> object:
    """Counts read once per session.

    Decompressing and 10-bit-unpacking 90 HRIT segments takes tens of
    seconds; every golden test in the suite can share one result.
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
