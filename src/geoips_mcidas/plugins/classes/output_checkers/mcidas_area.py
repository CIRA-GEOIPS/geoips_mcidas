"""Compare McIDAS AREA files for regression testing.

GeoIPS needs this to exist at all: ``identify_checker()`` asks every
registered checker whether it recognises a file and raises ``TypeError`` when
none does. The four built-in checkers all decline a binary AREA file, so
without this plugin ``--compare_path`` cannot be used on our output.

A McIDAS-written file is the most valuable reference available, so the
comparison is built to accept one. Two kinds of difference are therefore
ignored, because they record how a file was made rather than what is in it:

* provenance -- the creation stamps in words 17 and 18, the AREA slot number
  in word 33, and the audit-trail card count in word 64 with the cards
  themselves. McIDAS records the command sequence that produced a file; we
  record none, and neither says anything about the pixels.
* band layout, and only when the reference is McIDAS's multi-band
  intermediate and ours is one of the single-band files CLAVR-x reads. Words
  14, 15, 19 and 51 and the line prefix all describe how many bands share the
  file, so comparing them across that boundary is meaningless. The pixels are
  compared band for band instead.

Everything that describes the data must still match exactly: geometry,
navigation, calibration, and every pixel of the band in question.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from geoips.interfaces.class_based.output_checkers import BaseOutputCheckerPlugin

from geoips_mcidas.area import msgt_nav, parse, words

LOG: Final = logging.getLogger(__name__)

PROVENANCE_WORDS: Final = frozenset(
    {
        words.CREATION_DATE,
        words.CREATION_TIME,
        words.AREA_NUMBER,
        words.COMMENT_CARDS,
    },
)
"""Words recording how a file was made, not what is in it."""

BAND_LAYOUT_WORDS: Final = frozenset(
    {
        words.NUM_BANDS,
        words.PREFIX_BYTES,
        words.BAND_MAP,
        words.LEVEL_LENGTH,
    },
)
"""Words describing how many bands share a file.

Only ignored when comparing a single-band file against a multi-band one, which
is the shape a McIDAS reference takes.
"""


class McidasAreaOutputCheckerPlugin(BaseOutputCheckerPlugin):
    """Compare two McIDAS AREA files block by block."""

    interface = "output_checkers"
    family = "standard"
    name = "mcidas_area"

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

    def _compare_directories(
        self,
        produced: bytes,
        expected: bytes,
        *,
        ignore_band_layout: bool,
    ) -> bool:
        """Compare directory words, skipping the ones that may legitimately differ."""
        skip = PROVENANCE_WORDS | (
            BAND_LAYOUT_WORDS if ignore_band_layout else frozenset()
        )
        ours = words.unpack_words(produced)
        theirs = words.unpack_words(expected)
        mismatched = [
            number
            for number, (a, b) in enumerate(zip(ours, theirs, strict=True), start=1)
            if a != b and number not in skip
        ]
        if mismatched:
            LOG.info(
                "AREA directory words differ: %s",
                {n: (ours[n - 1], theirs[n - 1]) for n in mismatched},
            )
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

        our_bands = parse.directory_word(produced.directory, words.NUM_BANDS)
        their_bands = parse.directory_word(expected.directory, words.NUM_BANDS)
        cross_layout = our_bands != their_bands

        if not self._compare_directories(
            produced.directory,
            expected.directory,
            ignore_band_layout=cross_layout,
        ):
            return False
        if not self._compare_navigation(produced, expected, cross_layout=cross_layout):
            return False
        if produced.calibration != expected.calibration:
            LOG.info(
                "AREA calibration blocks differ: %d vs %d bytes",
                len(produced.calibration),
                len(expected.calibration),
            )
            return False
        return self._compare_pixels(output_product, compare_product, cross_layout)

    def _compare_navigation(
        self,
        produced: parse.AreaBlocks,
        expected: parse.AreaBlocks,
        *,
        cross_layout: bool,
    ) -> bool:
        """Compare navigation, allowing the band word to differ across layouts."""
        ours = words.unpack_words(produced.navigation)
        theirs = words.unpack_words(expected.navigation)
        if len(ours) != len(theirs):
            LOG.info(
                "AREA navigation blocks differ in size: %d vs %d words",
                len(ours),
                len(theirs),
            )
            return False
        if cross_layout and ours and theirs:
            # Nav word 7 is the band number, which a multi-band file reports
            # as its first band rather than ours.
            ours[msgt_nav.BAND - 1] = theirs[msgt_nav.BAND - 1]
        if ours != theirs:
            differing = [
                i + 1
                for i, (a, b) in enumerate(zip(ours, theirs, strict=True))
                if a != b
            ]
            LOG.info("AREA navigation words differ: %s", differing)
            return False
        return True

    def _compare_pixels(
        self,
        output_product: str,
        compare_product: str,
        cross_layout: bool,
    ) -> bool:
        """Compare the pixels of every band our file carries.

        Going through :func:`~geoips_mcidas.area.parse.band_plane` rather than
        comparing raw data blocks is what lets a single-band file be checked
        against one plane of a multi-band McIDAS reference.
        """
        ours_raw = Path(output_product).read_bytes()
        theirs_raw = Path(compare_product).read_bytes()
        band_map = parse.directory_word(ours_raw, words.BAND_MAP)
        bands = [band for band in range(1, 33) if band_map & (1 << (band - 1))]
        for band in bands:
            try:
                mine = parse.band_plane(ours_raw, band)
                reference = parse.band_plane(theirs_raw, band)
            except ValueError:
                LOG.info("Band %d is not present in both files", band)
                return False
            if mine.shape != reference.shape or not (mine == reference).all():
                LOG.info("AREA band %d pixels differ", band)
                return False
        if not cross_layout:
            LOG.info("Compared %d band(s) like for like", len(bands))
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
