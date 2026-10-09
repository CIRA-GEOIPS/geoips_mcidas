"""Write a SEVIRI scene as the per-band AREA files CLAVR-x reads.

This is the policy layer between raw counts and the AREA writer: which bands
must be present, what image size CLAVR-x can navigate, and the order files are
published in. It lives outside ``plugins/`` so that the GeoIPS output
formatter and the standalone command line share one implementation without
the command line importing GeoIPS's plugin machinery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from geoips_mcidas.area import msg_cal, msgt_nav, writer
from geoips_mcidas.area.directory import (
    FULL_DISK_ELEMENTS,
    FULL_DISK_LINES,
    SeviriAreaMetadata,
)
from geoips_mcidas.area.filenames import build_area_filename
from geoips_mcidas.errors import MissingBandsError
from geoips_mcidas.seviri.dataset import CLAVRX_BANDS

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from geoips_mcidas.seviri.scene import SeviriScene

LOG: Final = logging.getLogger(__name__)


def check_band_coverage(available: set[int]) -> None:
    """Require every band CLAVR-x reads to be present before writing any.

    CLAVR-x receives only the band-1 filename and derives the other ten by
    string splice. A band we skip becomes a file it cannot open, and its
    reader calls ``exit(EXIT_FAILURE)`` partway through a run -- reporting a
    filename that nothing ever wrote. Failing here instead names the missing
    bands while the cause is still visible.

    Parameters
    ----------
    available
        The MSG band numbers actually present in the input.

    Raises
    ------
    MissingBandsError
        If any of bands 1-11 is absent.
    """
    missing = sorted(set(CLAVRX_BANDS) - available)
    if missing:
        raise MissingBandsError(
            f"SEVIRI input is missing band(s) {missing}; CLAVR-x requires all "
            f"of {list(CLAVRX_BANDS)} and fails at runtime on any it cannot open",
        )


def check_full_disk(counts: dict[int, NDArray[np.integer]]) -> None:
    """Refuse anything that is not the full disk CLAVR-x assumes.

    CLAVR-x reads the image dimensions from the directory block but navigates
    with hard-coded full-disk constants (``COFF = LOFF = 1856``,
    ``CFAC = LFAC = -781648343``). A subset therefore runs to completion and is
    geolocated wrongly, with nothing reporting it.

    Raises
    ------
    ValueError
        If any band is not FULL_DISK x FULL_DISK.
    """
    wrong = {
        band: image.shape
        for band, image in sorted(counts.items())
        if image.shape != (FULL_DISK_LINES, FULL_DISK_ELEMENTS)
    }
    if wrong:
        raise ValueError(
            f"CLAVR-x navigates SEVIRI with hard-coded full-disk constants, so "
            f"only {FULL_DISK_LINES}x{FULL_DISK_ELEMENTS} may be written; got {wrong}",
        )


def write_seviri_bands(
    counts: dict[int, NDArray[np.integer]],
    gains: list[float],
    offsets: list[float],
    scene: SeviriScene,
    output_directory: Path,
) -> list[str]:
    """Write one AREA file per band and return the paths written.

    All files must share a directory: CLAVR-x prepends a single input path
    from its file list to every derived sibling name.
    """
    cal_block = msg_cal.build_msg_cal_block(
        gains,
        offsets,
        msg_cal.planck_table(scene.platform_name),
    )
    # Band 1 last. CLAVR-x is handed only the band-1 filename and derives the
    # other ten from it, so that file appearing is what tells a downstream
    # poller the set is ready. Writing it first advertises a scene whose other
    # ten files do not exist yet.
    #
    # Filenames carry no run identity, so on a re-run -- a backfill, or a
    # re-disseminated segment -- band 1 from the previous run would be present
    # throughout, and the readiness signal would be a lie for the whole of the
    # second run. Remove it first so its presence always means this run
    # finished.
    ordered = sorted(counts, key=lambda b: (b == 1, b))
    if 1 in counts:
        first = output_directory / build_area_filename(
            scene.platform_name,
            scene.observation_time,
            1,
        )
        first.unlink(missing_ok=True)

    written: dict[int, str] = {}
    for band in ordered:
        image = counts[band]
        meta = SeviriAreaMetadata(
            platform_name=scene.platform_name,
            observation_time=scene.observation_time,
            band=band,
            creation_time=scene.creation_time,
            n_lines=image.shape[0],
            n_elements=image.shape[1],
        )
        nav_block = msgt_nav.build_msgt_nav_block(
            scene.sub_satellite_longitude_east,
            band=band,
        )
        path = output_directory / build_area_filename(
            scene.platform_name,
            scene.observation_time,
            band,
        )
        writer.write_area_file(path, meta, nav_block, cal_block, image)
        LOG.info("Wrote %s", path)
        written[band] = str(path)
    return [written[band] for band in sorted(written)]
