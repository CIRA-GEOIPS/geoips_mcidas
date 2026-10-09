"""``seviri2area``: SEVIRI HRIT segments in, CLAVR-x AREA files out.

GeoIPS's order-based procflow cannot yet run the DataTree-native plugins, and a
CLAVR-x host rarely carries the GeoIPS scientific stack, so this drives the
same reader and writer the plugins use without going through GeoIPS at all.

**Why ``geoips/__init__`` is skipped.** The only part of GeoIPS this needs is
its HRIT parser, which depends on numpy and one stdlib-only helper. Importing
it the ordinary way runs ``geoips/__init__``, which pulls in netCDF4,
matplotlib, cartopy, xarray, dask and the plugin registry -- around 1,800
modules -- none of which the parser touches. Registering the package without
executing it keeps the command fast, and keeps the release binary down to
Python plus numpy.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Final

from geoips_mcidas.errors import SeviriHritError
from geoips_mcidas.seviri.area_files import (
    check_band_coverage,
    check_full_disk,
    write_seviri_bands,
)
from geoips_mcidas.seviri.scene import SeviriScene

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geoips_mcidas.seviri.counts import SeviriCounts

PROG: Final = "seviri2area"
HRIT_PREFIX: Final = "H-"
"""Every HRIT file name starts with this; anything else in a directory is not."""

LOG: Final = logging.getLogger(PROG)


def register_geoips_without_init() -> None:
    """Make ``geoips`` submodules importable without running its ``__init__``.

    Every intermediate package between ``geoips`` and the HRIT parser has an
    empty ``__init__``, so only the top level needs bypassing. Does nothing
    when GeoIPS is already imported, so it cannot shadow a real import.

    Raises
    ------
    ModuleNotFoundError
        If GeoIPS is not installed.
    """
    if "geoips" in sys.modules:
        return
    spec = importlib.util.find_spec("geoips")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("GeoIPS is not installed", name="geoips")
    sys.modules["geoips"] = importlib.util.module_from_spec(spec)


def read_counts(paths: Sequence[Path]) -> SeviriCounts:
    """Read HRIT segments through GeoIPS's parser, imported lightly."""
    register_geoips_without_init()
    # Deferred: the import must follow the registration above.
    from geoips_mcidas.seviri.counts import read_seviri_counts  # noqa: PLC0415

    return read_seviri_counts(paths)


def expand_inputs(paths: Sequence[Path]) -> list[Path]:
    """Replace each directory with the HRIT files directly inside it.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    entry
                    for entry in path.iterdir()
                    if entry.is_file() and entry.name.startswith(HRIT_PREFIX)
                ),
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return files


def convert(paths: Sequence[Path], output_directory: Path) -> list[str]:
    """Convert one SEVIRI cycle to the eleven AREA files CLAVR-x reads.

    Returns
    -------
    list[str]
        The files written, band 1 first.
    """
    counts = read_counts(expand_inputs(paths))
    check_band_coverage(set(counts.counts))
    check_full_disk(counts.counts)  # type: ignore[arg-type]
    scene = SeviriScene(
        platform_name=counts.platform_name,
        observation_time=counts.observation_time,
        sub_satellite_longitude_east=counts.sub_satellite_longitude_east,
        creation_time=datetime.now(tz=UTC),
    )
    return write_seviri_bands(
        counts.counts,  # type: ignore[arg-type]
        counts.gains,
        counts.offsets,
        scene,
        output_directory,
    )


def package_version() -> str:
    """Return the installed version, or ``unknown`` outside an install."""
    try:
        return version("geoips-mcidas")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Describe the command line."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Convert one cycle of uncompressed MSG SEVIRI HRIT segments "
            "(prologue plus every segment of bands 1-11) into the eleven "
            "McIDAS AREA files CLAVR-x reads. Prints the files written, band "
            "1 first; band 1 is the file to list for CLAVR-x."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="HRIT files, or directories whose H-* files are all one cycle",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(),
        help="directory to write the AREA files into (default: current)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log progress to stderr",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )
    # GeoIPS notes at INFO that the HRIT decompressor is absent. Compressed
    # segments are rejected with their own error, so the note only misleads.
    logging.getLogger("geoips.utils.context_managers").setLevel(logging.WARNING)
    try:
        written = convert(args.inputs, args.output_dir.resolve())
    except (SeviriHritError, ValueError, OSError, ModuleNotFoundError) as exc:
        # MissingBandsError is a ValueError; OSError covers unreadable input,
        # a missing path, a full disk and an unwritable output directory.
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
