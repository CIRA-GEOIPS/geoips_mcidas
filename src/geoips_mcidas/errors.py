"""Exceptions raised by this package.

These live outside ``plugins/`` deliberately. The GeoIPS plugin registry loads
plugin modules with ``spec.loader.exec_module``, producing a second, distinct
copy of every class defined there. An exception class defined inside a plugin
module therefore has the same name and module as the one a caller imports and
still fails ``isinstance``, so ``except MissingBandsError`` never fires for the
exception the running plugin actually raised.
"""

from __future__ import annotations


class MissingBandsError(ValueError):
    """The input does not carry every band CLAVR-x requires."""


class SeviriHritError(RuntimeError):
    """A set of HRIT files cannot yield a complete, single-scene image."""
