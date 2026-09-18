"""Shared fixtures, including access to the McIDAS reference AREA file."""

from __future__ import annotations

import os
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
