"""Write MSG SEVIRI raw counts as McIDAS AREA files for CLAVR-x.

One call produces eleven files, one per narrowband channel, because CLAVR-x
reads a single AREA file per band and derives the sibling names itself. That
is why this plugin owns its filenames rather than accepting a list from a
filename formatter: the names are part of the format contract, not a
presentation choice.

The plugin is DataTree-native -- ``data_tree = True`` and no ``family`` -- which
is the migration GeoIPS recommends. Two consequences worth knowing: input and
output pass through the interface unconverted, and the plugin runs under the
order-based procflow only, since ``single_source`` dispatches on ``family``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from geoips.interfaces.class_based.output_formatters import (
    BaseOutputFormatterPlugin,
)

from geoips_mcidas.area import msg_cal, msgt_nav, writer
from geoips_mcidas.area.directory import (
    FULL_DISK_ELEMENTS,
    FULL_DISK_LINES,
    SeviriAreaMetadata,
)
from geoips_mcidas.area.filenames import build_area_filename
from geoips_mcidas.errors import MissingBandsError
from geoips_mcidas.seviri.scene import SeviriScene

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import xarray as xr
    from numpy.typing import NDArray

LOG: Final = logging.getLogger(__name__)

CLAVRX_BANDS: Final = tuple(range(1, 12))
"""The eleven narrowband channels CLAVR-x reads. HRV is never used."""

COUNT_VARIABLE_TEMPLATE: Final = "B{band:02d}Cnt"
"""Raw-count variable names, following the reader's ``B01Rad`` convention."""

GAINS_ATTR: Final = "seviri_calibration_gains"
OFFSETS_ATTR: Final = "seviri_calibration_offsets"
SUB_LON_ATTR: Final = "longitude_of_projection_origin"
PLATFORM_ATTR: Final = "platform_name"
START_TIME_ATTR: Final = "start_datetime"


def count_variable(band: int) -> str:
    """Return the raw-count variable name for an MSG band."""
    return COUNT_VARIABLE_TEMPLATE.format(band=band)


def resolve_counts_dataset(tree: xr.DataTree) -> xr.Dataset:
    """Find the dataset holding raw counts, wherever the caller put it.

    Called directly, the tree's own dataset carries the counts. Under the
    order-based procflow it does not: ``Workflow._collect_upstream_data``
    builds a fresh ``DataTree(name="multi_input")`` per step and attaches each
    dependency as a *child*, so the root is empty and the reader's output is
    one level down. A family-bearing plugin never sees this, because its
    conduit extracts the node into a typed kwarg; a DataTree-native plugin
    gets the wrapper.

    Reading the root unconditionally therefore reports every band missing from
    input that carries all of them, which blames the data for a wiring
    mismatch. Search the children when the root holds no counts.

    Parameters
    ----------
    tree
        Either a node whose dataset holds ``B01Cnt`` .. ``B11Cnt``, or a
        wrapper whose children include one.

    Returns
    -------
    xr.Dataset
        The dataset holding the counts, or the root's own dataset when no
        child has any -- so the caller's band-coverage check reports the
        problem rather than this function.
    """
    root = tree.to_dataset()
    if _holds_all_counts(root):
        return root
    for child in tree.children.values():
        candidate = child.to_dataset()
        if _holds_all_counts(candidate):
            return candidate
    return root


def _holds_all_counts(dataset: xr.Dataset) -> bool:
    """Whether a dataset carries every band, not merely one of them.

    Selecting on presence would let a partial sibling node shadow the complete
    one, and the resulting error would blame the data for a resolution
    mistake.
    """
    return all(count_variable(band) in dataset for band in CLAVRX_BANDS)


def collect_counts(dataset: xr.Dataset) -> dict[int, NDArray[np.integer]]:
    """Pull every available raw-count array out of a dataset."""
    return {
        band: np.asarray(dataset[count_variable(band)].to_numpy())
        for band in CLAVRX_BANDS
        if count_variable(band) in dataset
    }


REQUIRED_ATTRS: Final = (
    PLATFORM_ATTR,
    START_TIME_ATTR,
    SUB_LON_ATTR,
    GAINS_ATTR,
    OFFSETS_ATTR,
)


def require_attrs(attrs: Mapping[Hashable, Any]) -> None:
    """Check every attribute the writer needs, whatever path produced it.

    The gains and offsets are read outside ``scene_from_attrs``, so validating
    only inside it left them unchecked whenever a caller passed ``scene=``.

    Raises
    ------
    KeyError
        Naming every absent attribute at once.
    TypeError
        If the observation time is not a datetime. xarray attributes routinely
        hold numpy scalars and strings, and the failure would otherwise
        surface as an AttributeError from deep inside filename construction.
    """
    missing = [name for name in REQUIRED_ATTRS if name not in attrs]
    if missing:
        raise KeyError(f"input is missing required attributes: {missing}")
    moment = attrs[START_TIME_ATTR]
    if not isinstance(moment, datetime):
        raise TypeError(
            f"{START_TIME_ATTR} must be a datetime, got {type(moment).__name__}",
        )


def scene_from_attrs(attrs: Mapping[Hashable, Any]) -> SeviriScene:
    """Build scene metadata from a dataset's attributes."""
    return SeviriScene(
        platform_name=str(attrs[PLATFORM_ATTR]),
        observation_time=attrs[START_TIME_ATTR],
        sub_satellite_longitude_east=float(attrs[SUB_LON_ATTR]),
        creation_time=datetime.now(tz=UTC),
    )


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


def resolve_output_directory(output_directory: str | Path) -> Path:
    """Reject an output directory that is not a real absolute path.

    pluginify resolves a workflow's ``!ENV`` tag with ``os.path.expandvars``,
    which leaves an unset ``${VAR}`` as literal text rather than raising. GeoIPS
    announces a default for ``GEOIPS_OUTDIRS`` but does not put it in the
    environment, so unset is a fully supported state the YAML cannot see. Left
    alone, ``mkdir(parents=True)`` would create a directory literally named
    ``${GEOIPS_OUTDIRS}`` relative to the process working directory -- which
    under cron is not the operator's shell -- write 300 MB into it, and report
    eleven output products at a path CLAVR-x will never look in.

    Raises
    ------
    ValueError
        If the path still contains an unexpanded variable, or is relative.
    """
    resolved = Path(output_directory)
    if "$" in str(resolved):
        raise ValueError(
            f"output_directory contains an unexpanded variable: {resolved}. "
            "Set the environment variable the workflow references.",
        )
    if not resolved.is_absolute():
        raise ValueError(
            f"output_directory must be absolute, got {resolved}; "
            "a relative path resolves against the process working directory",
        )
    return resolved


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

    written: list[str] = []
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
        written.append(str(path))
    return sorted(written, key=_band_in_name)


def _band_in_name(path: str) -> int:
    """Sort written paths back into band order for reporting."""
    return int(Path(path).name.split("_")[1])


class McidasAreaSeviriOutputFormatterPlugin(BaseOutputFormatterPlugin):
    """Write SEVIRI raw counts as per-band McIDAS AREA files."""

    interface = "output_formatters"
    name = "mcidas_area_seviri"
    data_tree = True
    # Drops "family": a DataTree-native plugin declares none. The base
    # class types this as a plain list, so it cannot be a ClassVar here.
    required_attributes = ["interface", "name"]  # noqa: RUF012

    def call(
        self,
        data: xr.DataTree,
        output_directory: str | Path,
        scene: SeviriScene | None = None,
        allow_subset: bool = False,
    ) -> xr.DataTree:
        """Write per-band AREA files and record them on the tree.

        Parameters
        ----------
        data
            A DataTree whose root dataset carries raw-count variables named
            ``B01Cnt`` .. ``B11Cnt``, plus calibration gains and offsets and
            the sub-satellite longitude on its attributes.
        output_directory
            Directory to write every band file into.
        scene
            Overrides the metadata otherwise read from the tree attributes.
        allow_subset
            Write an image that is not the full disk. CLAVR-x navigates SEVIRI
            with hard-coded full-disk constants and would mis-navigate the
            result without reporting anything, so this is off by default. Turn
            it on only when the files are for a reader that takes navigation
            from the file, such as McIDAS-V.

        Returns
        -------
        xr.DataTree
            The input tree with ``output_products`` set on its root
            attributes, which is how a family-less plugin hands results to the
            order-based procflow.
        """
        dataset = resolve_counts_dataset(data)
        attrs = dict(data.attrs) | dict(dataset.attrs)
        target = resolve_output_directory(output_directory)
        require_attrs(attrs)
        counts = collect_counts(dataset)
        check_band_coverage(set(counts))
        if not allow_subset:
            check_full_disk(counts)

        resolved = scene or scene_from_attrs(attrs)
        written = write_seviri_bands(
            counts,
            list(attrs[GAINS_ATTR]),
            list(attrs[OFFSETS_ATTR]),
            resolved,
            target,
        )
        data.attrs["output_products"] = written
        return data


PLUGIN_CLASS = McidasAreaSeviriOutputFormatterPlugin
