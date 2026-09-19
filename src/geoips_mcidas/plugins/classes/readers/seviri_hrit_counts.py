"""Read SEVIRI HRIT as raw counts, for writing McIDAS AREA files.

GeoIPS ships a ``seviri_hrit`` reader already, but it calibrates to radiance
and keeps only the result. An AREA file for CLAVR-x needs the counts
themselves plus the prologue's per-band slope and intercept, so this reader
exists alongside it rather than replacing it.

DataTree-native, matching the output formatter it feeds: ``data_tree = True``
and no ``family``, which means the tree passes through the interface
unconverted in both directions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import xarray as xr
from geoips.interfaces.class_based.readers import BaseReaderPlugin

from geoips_mcidas.plugins.classes.output_formatters.mcidas_area_seviri import (
    GAINS_ATTR,
    OFFSETS_ATTR,
    PLATFORM_ATTR,
    START_TIME_ATTR,
    SUB_LON_ATTR,
    count_variable,
)
from geoips_mcidas.seviri.counts import read_seviri_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geoips_mcidas.seviri.counts import SeviriCounts

LOG: Final = logging.getLogger(__name__)

SOURCE_NAME: Final = "seviri"
DATA_PROVIDER: Final = "eumetsat"
LINE_DIM: Final = "y"
ELEMENT_DIM: Final = "x"


def counts_to_dataset(result: SeviriCounts) -> xr.Dataset:
    """Turn assembled counts into the dataset the AREA formatter expects."""
    return xr.Dataset(
        {
            count_variable(band): ((LINE_DIM, ELEMENT_DIM), image)
            for band, image in sorted(result.counts.items())
        },
        attrs={
            PLATFORM_ATTR: result.platform_name,
            START_TIME_ATTR: result.observation_time,
            SUB_LON_ATTR: result.sub_satellite_longitude_east,
            GAINS_ATTR: result.gains,
            OFFSETS_ATTR: result.offsets,
            "source_name": SOURCE_NAME,
            "data_provider": DATA_PROVIDER,
            "end_datetime": result.observation_time,
        },
    )


class SeviriHritCountsReaderPlugin(BaseReaderPlugin):
    """Read MSG SEVIRI HRIT segments as raw counts."""

    interface = "readers"
    name = "seviri_hrit_counts"
    data_tree = True
    required_attributes = ("interface", "name")

    def call(
        self,
        data: xr.DataTree | None = None,
        filenames: Sequence[str] | None = None,
        metadata_only: bool = False,
    ) -> xr.DataTree:
        """Read HRIT files and return a DataTree of raw counts.

        Parameters
        ----------
        data
            The upstream tree, which the workflow engine injects and this
            reader does not use. Present because DataTree-native plugins
            receive it.
        filenames
            Every HRIT file for one observation time, prologue included.
        metadata_only
            Unused; counts and calibration come from the same files, so there
            is no cheaper metadata path worth maintaining.

        Returns
        -------
        xr.DataTree
            A single-node tree carrying ``B01Cnt`` .. ``B11Cnt``.

        Raises
        ------
        ValueError
            If no filenames were supplied.
        """
        del data, metadata_only
        if not filenames:
            raise ValueError("seviri_hrit_counts requires HRIT filenames")

        LOG.info("Reading %d SEVIRI HRIT files", len(filenames))
        result = read_seviri_counts(Path(name) for name in filenames)
        return xr.DataTree(dataset=counts_to_dataset(result), name="DATA")


PLUGIN_CLASS = SeviriHritCountsReaderPlugin
