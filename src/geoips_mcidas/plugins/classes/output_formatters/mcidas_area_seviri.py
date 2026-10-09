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

from geoips_mcidas.seviri.area_files import (
    check_band_coverage,
    check_full_disk,
    write_seviri_bands,
)
from geoips_mcidas.seviri.dataset import (
    CLAVRX_BANDS,
    GAINS_ATTR,
    OFFSETS_ATTR,
    PLATFORM_ATTR,
    REQUIRED_ATTRS,
    START_TIME_ATTR,
    SUB_LON_ATTR,
    count_variable,
)
from geoips_mcidas.seviri.scene import SeviriScene

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import xarray as xr
    from numpy.typing import NDArray

LOG: Final = logging.getLogger(__name__)


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
