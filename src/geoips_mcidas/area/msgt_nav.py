"""The ``MSGT`` navigation block for MSG SEVIRI AREA files.

CLAVR-x reads this block but takes exactly one number out of it: word 6, the
sub-satellite longitude. It ignores the scaling factors entirely and uses its
own hard-coded EUMETSAT full-disk constants instead. Everything else here
exists so the files navigate correctly in McIDAS-V and other AREA readers.

That asymmetry is worth keeping in mind: a nav block wrong in almost every
field still yields correct CLAVR-x output, while a sign error in word 6 alone
mis-navigates the whole image without any error being raised.

``MSGT`` is not interchangeable with the plain ``MSG`` navigation type, which
puts the sub-satellite longitude in word 2. A reader expecting one and handed
the other misreads it silently.
"""

from __future__ import annotations

from typing import Final

from geoips_mcidas.area.words import WORD_BYTES, WordValue, pack_words

NAV_BLOCK_WORDS: Final = 128
NAV_BLOCK_BYTES: Final = NAV_BLOCK_WORDS * WORD_BYTES

NAV_TYPE: Final = "MSGT"

# --- Navigation word numbers (1-based within the nav block) ----------------
NAV_TYPE_WORD: Final = 1
LINE_OFFSET: Final = 2
COLUMN_OFFSET: Final = 3
LINE_FACTOR: Final = 4
COLUMN_FACTOR: Final = 5
SUB_SATELLITE_LONGITUDE: Final = 6
BAND: Final = 7
EARTH_MODEL: Final = 8

# --- Nominal MSG reference-grid constants ----------------------------------
# McIDAS hard-codes these rather than reading them from the HRIT prologue, so
# we emit the same nominal values instead of deriving per-file ones.
MSG_REFERENCE_PIXEL: Final = 1856
"""Sub-satellite point of the 3712x3712 grid."""

OFFSET_SCALE: Final = 10
"""Line/column offsets are stored multiplied by ten."""

MSG_SCALING_FACTOR: Final = 13642337
"""CFAC/LFAC as EUMETSAT defines it, in units of 2**-16 degrees per pixel."""

FACTOR_SCALE: Final = 1_000_000
"""Scaling factors are stored as pixels-per-degree multiplied by a million."""

LONGITUDE_SCALE: Final = 10_000
"""Sub-satellite longitude is stored in ten-thousandths of a degree."""

OBLATE_EARTH_MODEL: Final = 2
"""Suppresses a half-pixel correction in the McIDAS navigation module."""


def _scaling_factor() -> int:
    """Return the stored LFAC/CFAC value.

    ``13642337 / 2**16`` pixels per degree, scaled by a million and truncated
    -- which reproduces the 208165542 that McIDAS writes.
    """
    return int(MSG_SCALING_FACTOR / 2**16 * FACTOR_SCALE)


def encode_sub_satellite_longitude(degrees_east: float) -> int:
    """Encode a sub-satellite longitude for nav word 6.

    McIDAS stores this **west-positive**, so an eastward longitude is negated.
    CLAVR-x negates it back on read. Meteosat-11 at 0 degrees encodes as 0,
    and a satellite at 45.5E (the IODC position) as -455000.

    Parameters
    ----------
    degrees_east
        Sub-satellite longitude, positive east, as reported by the HRIT
        prologue.
    """
    return -round(degrees_east * LONGITUDE_SCALE)


def build_msgt_nav_block(
    sub_satellite_longitude_east: float,
    band: int,
    earth_model: int = OBLATE_EARTH_MODEL,
) -> bytes:
    """Build the 512-byte ``MSGT`` navigation block.

    Parameters
    ----------
    sub_satellite_longitude_east
        Sub-satellite longitude in degrees east.
    band
        MSG band number this file holds, 1-12.
    earth_model
        Earth-model enum from the HRIT prologue. Defaults to the oblate model,
        which is the normal MSG case.

    Returns
    -------
    bytes
        Exactly ``NAV_BLOCK_BYTES``.
    """
    offset = MSG_REFERENCE_PIXEL * OFFSET_SCALE
    factor = _scaling_factor()
    words: dict[int, WordValue] = {
        NAV_TYPE_WORD: NAV_TYPE,
        LINE_OFFSET: offset,
        COLUMN_OFFSET: offset,
        LINE_FACTOR: factor,
        COLUMN_FACTOR: factor,
        SUB_SATELLITE_LONGITUDE: encode_sub_satellite_longitude(
            sub_satellite_longitude_east,
        ),
        BAND: band,
        EARTH_MODEL: earth_model,
    }
    return pack_words(words, NAV_BLOCK_WORDS)
