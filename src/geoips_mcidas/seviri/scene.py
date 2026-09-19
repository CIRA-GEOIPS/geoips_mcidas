"""Scene-level metadata shared by every band file of one observation.

This lives outside ``plugins/`` deliberately. GeoIPS's plugin registry loads
plugin modules with ``spec.loader.exec_module`` without registering them in
``sys.modules``, and ``@dataclass`` resolves field types by looking the
defining module up there -- so a dataclass declared inside a plugin module
raises ``AttributeError: 'NoneType' object has no attribute '__dict__'`` at
registry-build time. Keeping domain types in ordinary modules avoids the
problem entirely, and is better separation anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class SeviriScene:
    """Metadata every band file of one observation shares."""

    platform_name: str
    observation_time: datetime
    sub_satellite_longitude_east: float
    creation_time: datetime
