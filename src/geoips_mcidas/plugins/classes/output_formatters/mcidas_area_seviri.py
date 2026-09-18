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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from geoips.interfaces.class_based.output_formatters import BaseOutputFormatterPlugin

from geoips_mcidas.area import msg_cal, msgt_nav, writer
from geoips_mcidas.area.directory import SeviriAreaMetadata
from geoips_mcidas.area.filenames import build_area_filename

if TYPE_CHECKING:
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


class MissingBandsError(ValueError):
    """Raised when the input does not carry every band CLAVR-x requires."""


@dataclass(frozen=True)
class SeviriScene:
    """Scene-level metadata shared by every band file of one observation."""

    platform_name: str
    observation_time: datetime
    sub_satellite_longitude_east: float
    creation_time: datetime


def count_variable(band: int) -> str:
    """Return the raw-count variable name for an MSG band."""
    return COUNT_VARIABLE_TEMPLATE.format(band=band)


def collect_counts(dataset: xr.Dataset) -> dict[int, NDArray[np.integer]]:
    """Pull every available raw-count array out of a dataset."""
    return {
        band: np.asarray(dataset[count_variable(band)].to_numpy())
        for band in CLAVRX_BANDS
        if count_variable(band) in dataset
    }


def scene_from_attrs(attrs: dict[str, object]) -> SeviriScene:
    """Build scene metadata from a dataset's attributes.

    Raises
    ------
    KeyError
        If a required attribute is absent.
    """
    missing = [
        name
        for name in (PLATFORM_ATTR, START_TIME_ATTR, SUB_LON_ATTR)
        if name not in attrs
    ]
    if missing:
        raise KeyError(f"input is missing required attributes: {missing}")
    return SeviriScene(
        platform_name=str(attrs[PLATFORM_ATTR]),
        observation_time=attrs[START_TIME_ATTR],  # type: ignore[arg-type]
        sub_satellite_longitude_east=float(attrs[SUB_LON_ATTR]),  # type: ignore[arg-type]
        creation_time=datetime.now(tz=UTC),
    )


def check_band_coverage(available: set[int]) -> None:
    """Decide whether a partial set of bands may be written.

    Parameters
    ----------
    available
        The MSG band numbers actually present in the input.

    Raises
    ------
    MissingBandsError
        If the set of bands is not acceptable to write.
    """
    # TODO(human): implement the missing-band policy.
    raise NotImplementedError


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
    written: list[str] = []
    for band in sorted(counts):
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
    return written


class McidasAreaSeviriOutputFormatterPlugin(BaseOutputFormatterPlugin):
    """Write SEVIRI raw counts as per-band McIDAS AREA files."""

    interface = "output_formatters"
    name = "mcidas_area_seviri"
    data_tree = True
    required_attributes = ("interface", "name")

    def call(
        self,
        data: xr.DataTree,
        output_directory: str | Path,
        scene: SeviriScene | None = None,
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

        Returns
        -------
        xr.DataTree
            The input tree with ``output_products`` set on its root
            attributes, which is how a family-less plugin hands results to the
            order-based procflow.
        """
        dataset = data.to_dataset() if data.ds is not None else data.ds
        attrs = dict(data.attrs) | dict(getattr(dataset, "attrs", {}))
        counts = collect_counts(dataset)
        check_band_coverage(set(counts))

        resolved = scene or scene_from_attrs(attrs)
        written = write_seviri_bands(
            counts,
            list(attrs[GAINS_ATTR]),  # type: ignore[call-overload]
            list(attrs[OFFSETS_ATTR]),  # type: ignore[call-overload]
            resolved,
            Path(output_directory),
        )
        data.attrs["output_products"] = written
        return data


PLUGIN_CLASS = McidasAreaSeviriOutputFormatterPlugin
