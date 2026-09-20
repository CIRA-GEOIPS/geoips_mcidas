"""The 64-word AREA directory block for a single-band SEVIRI image.

CLAVR-x consumes **single-band** AREA files, one per channel. Those differ from
the multi-band image McIDAS produces first: one band, no line prefix, and a
band map with a single bit set. The block offsets stay put, because McIDAS
hard-codes them.

Several values here are load-bearing in ways that are not obvious:

``START_LINE`` and ``START_ELEMENT`` must both be 1. CLAVR-x keys its
EUMETSAT-grid line and element flips on exactly that, and the McIDAS server
emits 3 for a full disk -- the reference pipeline only gets 1 because a
separate ``imgcha`` step overwrites it afterwards. We write it directly.

``YEAR_DAY`` is ``YYDDD`` against a 1900 epoch. A four-digit year is not
rejected by anything; it simply yields an image dated two millennia late.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from geoips_mcidas.area import words
from geoips_mcidas.area.msg_cal import CAL_BLOCK_BYTES
from geoips_mcidas.area.msgt_nav import NAV_BLOCK_BYTES

if TYPE_CHECKING:
    from datetime import datetime

FULL_DISK_LINES: Final = 3712
FULL_DISK_ELEMENTS: Final = 3712
"""CLAVR-x hard-codes full-disk navigation, so any other size mis-navigates."""

PIXEL_BYTES: Final = 2
"""SEVIRI counts are 10-bit, stored in two bytes."""

RESOLUTION: Final = 3
"""Line and element resolution: the 3 km grid samples every third full-res
line and element of the 11136-point navigation grid."""

UPPER_LEFT_COORDINATE: Final = 1
SOURCE_TYPE: Final = "MSG "
CAL_TYPE: Final = "RAW "
"""Raw counts. CLAVR-x never checks this word, but it does assume raw data."""

CAL_UNIT: Final = "    "
CAL_SCALE: Final = 1
AVG_SAMPLE_FLAG: Final = 2
DEFAULT_MEMO: Final = "MSG data from XRIT V2"

# Block offsets, in bytes from the start of the file. McIDAS hard-codes these;
# deriving them from the block sizes keeps the one definition honest. Named
# *_BLOCK_OFFSET to distinguish them from the words.* names of the same idea,
# which are 1-based directory WORD numbers, not byte offsets.
NAV_BLOCK_OFFSET: Final = words.DIRECTORY_BYTES
CAL_BLOCK_OFFSET: Final = NAV_BLOCK_OFFSET + NAV_BLOCK_BYTES
DATA_BLOCK_OFFSET: Final = CAL_BLOCK_OFFSET + CAL_BLOCK_BYTES

SENSOR_SOURCE_NUMBERS: Final[dict[str, int]] = {
    "meteosat-8": 51,
    "meteosat-9": 52,
    "meteosat-10": 53,
    "meteosat-11": 354,
}
"""SSEC sensor source numbers, as CLAVR-x matches them.

CLAVR-x accepts only these four values for SEVIRI and has no default branch,
so an unrecognised number is not an error -- the sensor simply never gets
identified. Note 354 for Meteosat-11: older CLAVR-x releases used 54.
"""


def sensor_source_number(platform_name: str) -> int:
    """Map a GeoIPS platform name to its SSEC sensor source number.

    Raises
    ------
    KeyError
        If the platform is not one CLAVR-x recognises as SEVIRI.
    """
    key = platform_name.lower()
    if key not in SENSOR_SOURCE_NUMBERS:
        raise KeyError(
            f"{platform_name!r} is not a CLAVR-x SEVIRI platform; "
            f"expected one of {sorted(SENSOR_SOURCE_NUMBERS)}",
        )
    return SENSOR_SOURCE_NUMBERS[key]


@dataclass(frozen=True)
class SeviriAreaMetadata:
    """Everything needed to describe one single-band SEVIRI AREA file."""

    platform_name: str
    observation_time: datetime
    band: int
    creation_time: datetime
    n_lines: int = FULL_DISK_LINES
    n_elements: int = FULL_DISK_ELEMENTS
    memo: str = DEFAULT_MEMO
    comment_cards: tuple[str, ...] = field(default_factory=tuple)

    @property
    def line_bytes(self) -> int:
        """Bytes per image line. Single-band files carry no line prefix."""
        return self.n_elements * PIXEL_BYTES

    @property
    def data_bytes(self) -> int:
        """Total size of the image data block."""
        return self.n_lines * self.line_bytes

    @property
    def total_bytes(self) -> int:
        """Total file size, including the trailing comment cards."""
        return (
            DATA_BLOCK_OFFSET
            + self.data_bytes
            + len(self.comment_cards) * words.COMMENT_CARD_BYTES
        )


def _encode_datetime(moment: datetime) -> tuple[int, int]:
    """Return ``(yyddd, hhmmss)`` for a datetime."""
    return (
        words.encode_yyddd(moment.year, moment.timetuple().tm_yday),
        words.encode_hhmmss(moment.hour, moment.minute, moment.second),
    )


def build_directory(meta: SeviriAreaMetadata) -> bytes:
    """Build the 256-byte directory block for one single-band AREA file.

    Returns
    -------
    bytes
        Exactly ``words.DIRECTORY_BYTES``.
    """
    image_day, image_time = _encode_datetime(meta.observation_time)
    created_day, created_time = _encode_datetime(meta.creation_time)

    values: dict[int, words.WordValue] = {
        words.STATUS: 0,
        words.VERSION: words.VERSION_NUMBER,
        words.SENSOR_SOURCE: sensor_source_number(meta.platform_name),
        words.YEAR_DAY: image_day,
        words.IMAGE_TIME: image_time,
        words.START_LINE: UPPER_LEFT_COORDINATE,
        words.START_ELEMENT: UPPER_LEFT_COORDINATE,
        words.NUM_LINES: meta.n_lines,
        words.NUM_ELEMENTS: meta.n_elements,
        words.BYTES_PER_PIXEL: PIXEL_BYTES,
        words.LINE_RESOLUTION: RESOLUTION,
        words.ELEMENT_RESOLUTION: RESOLUTION,
        words.NUM_BANDS: 1,
        words.PREFIX_BYTES: 0,
        words.CREATION_DATE: created_day,
        words.CREATION_TIME: created_time,
        words.BAND_MAP: 1 << (meta.band - 1),
        words.DATA_OFFSET: DATA_BLOCK_OFFSET,
        words.NAV_OFFSET: NAV_BLOCK_OFFSET,
        words.VALIDITY_CODE: 0,
        words.ACTUAL_YEAR_DAY: image_day,
        words.ACTUAL_TIME: image_time,
        words.DOC_LENGTH: 0,
        words.CAL_LENGTH: 0,
        words.LEVEL_LENGTH: 0,
        words.SOURCE_TYPE: SOURCE_TYPE,
        words.CAL_TYPE: CAL_TYPE,
        words.AVG_SAMPLE_FLAG: AVG_SAMPLE_FLAG,
        words.CAL_UNIT: CAL_UNIT,
        words.CAL_SCALE: CAL_SCALE,
        words.AUX_OFFSET: 0,
        words.AUX_COUNT: 0,
        words.CAL_OFFSET: CAL_BLOCK_OFFSET,
        words.COMMENT_CARDS: len(meta.comment_cards),
    }
    values.update(words.encode_memo(meta.memo))
    return words.pack_words(values, words.DIRECTORY_WORDS)
