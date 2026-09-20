"""AREA filenames, which CLAVR-x treats as part of the data format.

CLAVR-x is handed only the band-1 filename. It finds the other ten channels by
splicing the **first** occurrence of the literal ``"_1_"`` in the basename and
substituting each band number in turn::

    ipos = index(Channel_1_Filename, "_1_")
    Channel_X_Filename = Channel_1_Filename(1:ipos-1) // "_"//band//"_"
                         // Channel_1_Filename(ipos+3:ilen)

So the filename is an integration contract, not a convention. Two failure
modes, both silent:

* If the basename contains an earlier ``"_1_"`` than the band token, every
  sibling name is spliced in the wrong place and the files are not found.
* If it contains no ``"_1_"`` at all, Fortran's ``index`` returns 0 and the
  splice produces a corrupt name with no error raised.

:func:`build_area_filename` therefore checks, for band 1 only, that the band
token really is the first ``"_1_"`` in the name. Band 1 is the only name CLAVR-x
is given, so it is the only one whose splice point matters.
:func:`clavrx_sibling_filename` is a direct transcription of the CLAVR-x logic,
and the tests use it to confirm the two agree for all eleven bands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from datetime import datetime

BAND_1_MARKER: Final = "_1_"
AREA_SUFFIX: Final = ".area"

PLATFORM_FILE_TOKENS: Final[dict[str, str]] = {
    "meteosat-8": "met8",
    "meteosat-9": "met9",
    "meteosat-10": "met10",
    "meteosat-11": "met11",
}
"""Filename prefixes matching the existing CIRA pipeline.

None contains an underscore, which is what keeps the band token the first
``"_1_"`` in the basename.
"""


def platform_token(platform_name: str) -> str:
    """Map a GeoIPS platform name to its filename prefix.

    Raises
    ------
    KeyError
        If the platform has no known prefix.
    """
    key = platform_name.lower()
    if key not in PLATFORM_FILE_TOKENS:
        raise KeyError(
            f"{platform_name!r} has no AREA filename prefix; "
            f"expected one of {sorted(PLATFORM_FILE_TOKENS)}",
        )
    return PLATFORM_FILE_TOKENS[key]


def clavrx_sibling_filename(band_1_filename: str, band: int) -> str:
    """Derive another band's filename exactly as CLAVR-x does.

    A transcription of the Fortran splice, used both to validate our own
    filenames and to test that assumption directly.

    Parameters
    ----------
    band_1_filename
        The basename handed to CLAVR-x in its file list.
    band
        The MSG band whose filename to derive.

    Returns
    -------
    str
        The sibling basename. Note this reproduces CLAVR-x's behaviour
        faithfully, including its silence when ``"_1_"`` is absent.
    """
    position = band_1_filename.find(BAND_1_MARKER)
    if position < 0:
        # Fortran's index() returns 0, and the slice arithmetic below then
        # silently mangles the name rather than failing. Reproduced so tests
        # can assert on it.
        return f"_{band}_" + band_1_filename[2:]
    head = band_1_filename[:position]
    tail = band_1_filename[position + len(BAND_1_MARKER) :]
    return f"{head}_{band}_{tail}"


def build_area_filename(
    platform_name: str,
    observation_time: datetime,
    band: int,
) -> str:
    """Build the AREA basename for one band.

    ``met11_1_2021_141_0100.area`` -- prefix, band, year, zero-padded day of
    year, and UTC ``HHMM``. Seconds are absent because CLAVR-x discards them.

    Raises
    ------
    ValueError
        If the resulting name would not survive CLAVR-x's sibling splice.
    """
    prefix = platform_token(platform_name)
    stamp = (
        f"{observation_time.year}_{observation_time.timetuple().tm_yday:03d}_"
        f"{observation_time:%H%M}"
    )
    name = f"{prefix}_{band}_{stamp}{AREA_SUFFIX}"
    if band == 1:
        _validate_band_token(name, prefix)
    return name


def _validate_band_token(band_1_name: str, prefix: str) -> None:
    """Fail loudly on a name CLAVR-x would mis-splice quietly."""
    position = band_1_name.find(BAND_1_MARKER)
    if position != len(prefix):
        raise ValueError(
            f"{band_1_name!r} does not put the band token at the first "
            f"'{BAND_1_MARKER}'; CLAVR-x would derive the wrong sibling names",
        )
