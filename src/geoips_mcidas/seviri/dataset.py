"""The dataset contract shared by the SEVIRI reader and the AREA formatter.

Both plugins have to agree on the variable names and attributes that carry raw
counts and calibration between them. Defining that here rather than in either
plugin keeps the producer from importing the consumer, and gives the required
attribute list a single definition.

It also has to live outside ``plugins/``: the GeoIPS registry re-executes
plugin modules, so a name defined in one and imported by another would exist
twice with no guarantee the copies match.
"""

from __future__ import annotations

from typing import Final

MSG_BANDS: Final = 12
"""Channels an MSG calibration block describes: eleven narrowband plus HRV."""

CLAVRX_BANDS: Final = tuple(range(1, 12))
"""The eleven narrowband channels CLAVR-x reads. HRV is never used."""

COUNT_VARIABLE_TEMPLATE: Final = "B{band:02d}Cnt"
"""Raw-count variable names, following the reader's ``B01Rad`` convention."""

PLATFORM_ATTR: Final = "platform_name"
START_TIME_ATTR: Final = "start_datetime"
SUB_LON_ATTR: Final = "longitude_of_projection_origin"
GAINS_ATTR: Final = "seviri_calibration_gains"
OFFSETS_ATTR: Final = "seviri_calibration_offsets"

REQUIRED_ATTRS: Final = (
    PLATFORM_ATTR,
    START_TIME_ATTR,
    SUB_LON_ATTR,
    GAINS_ATTR,
    OFFSETS_ATTR,
)
"""Everything the formatter needs from the reader, in one place."""


def count_variable(band: int) -> str:
    """Return the raw-count variable name for an MSG band."""
    return COUNT_VARIABLE_TEMPLATE.format(band=band)
