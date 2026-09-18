"""Compare McIDAS AREA files for regression testing.

GeoIPS needs this to exist at all: ``identify_checker()`` asks every
registered checker whether it recognises a file and raises ``TypeError`` when
none does. The four built-in checkers all decline a binary AREA file, so
without this plugin ``--compare_path`` cannot be used on our output.

Two classes of difference are ignored deliberately, because they change on
every run and say nothing about correctness: the creation date and time in
directory words 17 and 18, and the trailing comment cards, which are an audit
trail of the commands that produced the file. Everything else -- every other
directory word, the navigation block, the calibration block and every pixel --
must match exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from geoips.interfaces.class_based.output_checkers import BaseOutputCheckerPlugin

from geoips_mcidas.area import parse, words

LOG: Final = logging.getLogger(__name__)

VOLATILE_WORDS: Final = frozenset({words.CREATION_DATE, words.CREATION_TIME})
"""Directory words that differ between runs of identical input."""


class McidasAreaOutputCheckerPlugin(BaseOutputCheckerPlugin):
    """Compare two McIDAS AREA files block by block."""

    interface = "output_checkers"
    family = "standard"
    name = "mcidas_area"

    VALID_EXTENSIONS = frozenset({".area"})

    def correct_file_format(self, fname: str) -> bool:
        """Return whether this checker recognises a file as an AREA file.

        Checks McIDAS's own magic -- directory word 2 holding the format
        version 4 -- rather than trusting the extension, since AREA files are
        also conventionally named ``AREAnnnn`` with no suffix at all.
        """
        path = Path(fname)
        try:
            with path.open("rb") as handle:
                header = handle.read(words.DIRECTORY_BYTES)
        except OSError:
            return False
        return parse.is_area_file(header)

    def _compare_directories(self, produced: bytes, expected: bytes) -> bool:
        """Compare directory words, skipping the ones that always differ."""
        ours = words.unpack_words(produced)
        theirs = words.unpack_words(expected)
        mismatched = [
            number
            for number, (a, b) in enumerate(zip(ours, theirs, strict=True), start=1)
            if a != b and number not in VOLATILE_WORDS
        ]
        if mismatched:
            LOG.info("AREA directory words differ: %s", mismatched)
            return False
        return True

    def outputs_match(self, output_product: str, compare_product: str) -> bool:
        """Compare two AREA files.

        Returns
        -------
        bool
            True when the directory (ignoring creation time), navigation,
            calibration and pixel data all match.
        """
        try:
            produced = parse.parse_area_file(Path(output_product).read_bytes())
            expected = parse.parse_area_file(Path(compare_product).read_bytes())
        except (OSError, ValueError):
            LOG.exception("Could not read AREA files for comparison")
            return False

        if not self._compare_directories(produced.directory, expected.directory):
            return False

        for label, ours, theirs in (
            ("navigation", produced.navigation, expected.navigation),
            ("calibration", produced.calibration, expected.calibration),
            ("data", produced.data, expected.data),
        ):
            if ours != theirs:
                LOG.info(
                    "AREA %s block differs: %d vs %d bytes",
                    label,
                    len(ours),
                    len(theirs),
                )
                return False
        return True

    def call(self, compare_path: str, output_products: list[str]) -> int:
        """Compare produced AREA files against the reference directory.

        Returns
        -------
        int
            0 when every comparison succeeded.
        """
        return int(self.compare_outputs(compare_path, output_products))


PLUGIN_CLASS = McidasAreaOutputCheckerPlugin
