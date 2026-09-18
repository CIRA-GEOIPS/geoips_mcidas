"""The MSG calibration block: ASCII text, not packed binary.

Unusually for AREA, this block is plain text. McIDAS writes a four-character
``MSGT`` tag followed by twelve fixed-width records, one per MSG channel
including HRV, then twelve four-character channel-processing flags. Because
every byte is printable ASCII, the block is never byte-swapped and is
identical on any host.

Each record holds six Fortran ``E17.10`` fields::

    c1w3, c2w, alpha, beta, gain, offset

CLAVR-x reads records 1-11 and uses only ``gain`` and ``offset``, applying
``radiance = gain * count + offset``. The four Planck fields exist for McIDAS-V
and other readers that convert radiance to brightness temperature; they are
zero for the solar channels (1-3) and HRV (12), which have no Planck constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

from geoips_mcidas.area.words import WORD_BYTES

if TYPE_CHECKING:
    from collections.abc import Sequence

CAL_BLOCK_BYTES: Final = 1600
CAL_TAG: Final = "MSGT"
CAL_RECORD_BYTES: Final = 104
"""Six 17-character fields plus two trailing spaces."""

FIELD_WIDTH: Final = 17
FIELD_DECIMALS: Final = 10
MSG_BANDS: Final = 12
"""Eleven narrowband channels plus HRV."""

DEFAULT_PCP_FLAG: Final = 2
"""Planned-channel-processing flag written for every channel by default."""

# Physical constants, as McIDAS uses them to build the radiance transform.
PLANCK_CONSTANT: Final = 6.626176e-34
SPEED_OF_LIGHT: Final = 2.99792458e8
BOLTZMANN_CONSTANT: Final = 1.380662e-23

_C1: Final = 2.0e5 * PLANCK_CONSTANT * SPEED_OF_LIGHT * SPEED_OF_LIGHT
_C2: Final = PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT
_WAVENUMBER_TO_PER_METRE: Final = 100.0


class PlanckCoefficients(NamedTuple):
    """EUMETSAT radiance-to-brightness-temperature constants for one channel."""

    wavenumber: float
    """Central wavenumber in cm^-1."""
    alpha: float
    beta: float


METEOSAT_11_PLANCK: Final[dict[int, PlanckCoefficients]] = {
    4: PlanckCoefficients(2555.280, 0.9916, 2.9438),
    5: PlanckCoefficients(1596.080, 0.9959, 2.0780),
    6: PlanckCoefficients(1361.748, 0.9990, 0.4929),
    7: PlanckCoefficients(1147.433, 0.9996, 0.1731),
    8: PlanckCoefficients(1034.851, 0.9998, 0.0597),
    9: PlanckCoefficients(931.122, 0.9983, 0.6256),
    10: PlanckCoefficients(839.113, 0.9988, 0.4002),
    11: PlanckCoefficients(748.585, 0.9981, 0.5635),
}
"""Thermal-channel constants for Meteosat-11.

Verified by reproducing the calibration block of a McIDAS-produced file to
within float32 precision. Tables for Meteosat-8, -9 and -10 are not included:
they would have to be transcribed from the EUMETSAT MSG Level 1.5 Image Data
Format Description, and guessing them is worse than writing zeros, which is
what :func:`build_msg_cal_block` does when no table is supplied. CLAVR-x
ignores these fields entirely, so their absence costs fidelity in other
readers, not correctness.
"""

PLANCK_TABLES: Final[dict[str, dict[int, PlanckCoefficients]]] = {
    "meteosat-11": METEOSAT_11_PLANCK,
}


def planck_table(platform_name: str) -> dict[int, PlanckCoefficients]:
    """Return the Planck table for a platform, or an empty one if unknown."""
    return PLANCK_TABLES.get(platform_name.lower(), {})


def format_e17_10(value: float) -> str:
    """Format a float the way Fortran's ``E17.10`` edit descriptor does.

    Fortran normalises the mantissa below one -- ``0.2124089934E-01`` -- where
    C and Python normalise it to a single leading digit. The difference is a
    factor of ten in the mantissa and one in the exponent.

    Returns
    -------
    str
        Exactly ``FIELD_WIDTH`` characters, right-justified.
    """
    if value == 0.0:
        return " 0." + "0" * FIELD_DECIMALS + "E+00"
    sign = "-" if value < 0 else " "
    mantissa, _, exponent = f"{abs(value):.{FIELD_DECIMALS - 1}E}".partition("E")
    digits = mantissa.replace(".", "")
    shifted = int(exponent) + 1
    return f"{sign}0.{digits}E{'+' if shifted >= 0 else '-'}{abs(shifted):02d}"


def _record(
    gain: float,
    offset: float,
    planck: PlanckCoefficients | None,
) -> bytes:
    """Build one 104-byte calibration record."""
    if planck is None:
        c1w3 = c2w = alpha = beta = 0.0
    else:
        w = _WAVENUMBER_TO_PER_METRE * planck.wavenumber
        c1w3 = _C1 * w * w * w
        c2w = _C2 * w
        alpha, beta = planck.alpha, planck.beta
    fields = (c1w3, c2w, alpha, beta, gain, offset)
    text = "".join(format_e17_10(field) for field in fields) + "  "
    return text.encode("ascii")


def build_msg_cal_block(
    gains: Sequence[float],
    offsets: Sequence[float],
    planck: dict[int, PlanckCoefficients] | None = None,
    pcp_flags: Sequence[int] | None = None,
) -> bytes:
    """Build the 1600-byte MSG calibration block.

    Every band file carries the **full** twelve-record block, not just its own
    record. CLAVR-x opens only the band-1 file and reads all eleven narrowband
    channels' coefficients out of it, so trimming the block per band would
    leave ten channels uncalibrated.

    Parameters
    ----------
    gains, offsets
        Per-band radiance slope and intercept from the HRIT prologue's
        Level 1.5 image calibration, indexed band 1..12.
    planck
        Thermal-channel constants keyed by band number. Bands absent from the
        table get zeros, which is correct for the solar channels and HRV.
    pcp_flags
        Planned-channel-processing flag per band; defaults to
        :data:`DEFAULT_PCP_FLAG` for all twelve.

    Returns
    -------
    bytes
        Exactly :data:`CAL_BLOCK_BYTES`.

    Raises
    ------
    ValueError
        If ``gains`` or ``offsets`` does not hold exactly twelve values.
    """
    if len(gains) != MSG_BANDS or len(offsets) != MSG_BANDS:
        raise ValueError(f"expected {MSG_BANDS} gains and offsets")
    table = planck or {}
    flags = pcp_flags if pcp_flags is not None else [DEFAULT_PCP_FLAG] * MSG_BANDS

    block = bytearray(CAL_BLOCK_BYTES)
    block[:WORD_BYTES] = CAL_TAG.encode("ascii")
    for index in range(MSG_BANDS):
        start = WORD_BYTES + index * CAL_RECORD_BYTES
        record = _record(gains[index], offsets[index], table.get(index + 1))
        block[start : start + CAL_RECORD_BYTES] = record

    flag_start = WORD_BYTES + MSG_BANDS * CAL_RECORD_BYTES
    for index, flag in enumerate(flags[:MSG_BANDS]):
        start = flag_start + index * WORD_BYTES
        block[start : start + WORD_BYTES] = (str(flag) * WORD_BYTES).encode("ascii")
    return bytes(block)


class CalRecord(NamedTuple):
    """One channel's calibration record, parsed back out of the block."""

    c1w3: float
    c2w: float
    alpha: float
    beta: float
    gain: float
    offset: float


def parse_cal_record(block: bytes, band: int) -> CalRecord:
    """Read one band's calibration record out of a calibration block.

    Parameters
    ----------
    block
        A full calibration block.
    band
        MSG band number, 1-12.

    Returns
    -------
    CalRecord
        The six fields, as floats.
    """
    start = WORD_BYTES + (band - 1) * CAL_RECORD_BYTES
    text = block[start : start + CAL_RECORD_BYTES].decode("ascii")
    fields = [
        float(text[i * FIELD_WIDTH : (i + 1) * FIELD_WIDTH])
        for i in range(len(CalRecord._fields))
    ]
    return CalRecord(*fields)
