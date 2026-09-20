# geoips_mcidas

A [GeoIPS](https://github.com/NRLMMD-GEOIPS/geoips) plugin package that writes
**McIDAS AREA files** from MSG SEVIRI data, for ingest by
[CLAVR-x](https://gitlab.ssec.wisc.edu/clavrx/clavrx-dev).

## Why

CLAVR-x has no HRIT reader. McIDAS AREA is its only native SEVIRI L1b input
path, so pipelines that run CLAVR-x over SEVIRI must first convert HRIT to AREA
— conventionally with McIDAS-X, which is paid, site-licensed and
non-redistributable.

This package replaces that step with a pure-Python writer, so the pipeline runs
anywhere GeoIPS runs.

## Scope

- Full-disk MSG SEVIRI only, 3712 x 3712 at 3 km. CLAVR-x hardcodes full-disk
  navigation constants, so a subset would be silently mis-navigated.
- 11 narrowband channels. HRV is not used by CLAVR-x.
- One AREA file per band, named `met11_<band>_<year>_<jday>_<hhmm>.area`.
  CLAVR-x is given only the band-1 filename and derives the rest by splicing on
  the first `_1_` in the basename.

## Provenance

The AREA format implementation is written from the public
[McIDAS Programmer's Manual](https://www.ssec.wisc.edu/mcidas/doc/prog_man/2015/formats-1.html)
and from measurements of real AREA files. No McIDAS-X or VisAD source is
copied: both are licence-encumbered and are used only as references of record.

## Development

```sh
nix develop      # toolchain
uv run pytest    # tests
```

## Verification status

The repository ships with a matched input/output pair as the basis for testing:
the HRIT segments for 2021-05-21 01:00 and the AREA file McIDAS-X produced from
exactly those segments. Tests marked `golden` compare against it and skip when
it is absent.

What is verified:

- **Counts match McIDAS pixel for pixel** — 13,778,944 of 13,778,944 on every
  band tested. This is what pins the row order down; HRIT segment 1 is the
  southernmost strip, so the assembled image is flipped to north-up. Without
  the flip roughly a quarter of pixels still coincide, which is enough for an
  image to look plausible while being upside down.
- **Navigation and calibration blocks match byte for byte**, Planck fields
  included.
- **Directory matches on every shared word.** Words 14, 15, 19 and 51 are
  excluded because the reference is McIDAS's multi-band intermediate, whereas
  CLAVR-x reads the single-band files — and those four are exactly what differ.
- **Filenames round-trip through CLAVR-x's sibling splice**, reimplemented from
  its Fortran and tested for all eleven bands.

What is *not* verified: a CLAVR-x run against these files. That needs
`met11_instr.dat` from the ancillary bundle and GFS ancillary data, neither of
which is present here.

The calibration block matches McIDAS byte for byte, Planck fields included. That
only works because the physical constants and the per-channel wavenumber, alpha
and beta are rounded through single precision before the double-precision
arithmetic — Fortran does that to an `E` literal assigned to a double precision
parameter, and skipping it makes every one of the 32 Planck fields differ in the
eighth significant figure.

## Upstream patches this package depends on

The reader and output formatter are DataTree-native — `data_tree = True` and no
`family` — which is the migration GeoIPS's own interface recommends. Registering
them needs two one-line changes to `pluginify` 1.0.0, and the `geoips` CLI needs
a third fix before `geoips run order_based` works at all.

All are written, verified, and on branches alongside this repository. See
[GEOIPS_ISSUES.md](GEOIPS_ISSUES.md) for the reproductions and patch status.
Without them, these plugins can still be imported and called directly, but not
resolved by name — `tests/test_plugin_registration.py` skips rather than fails
on an unpatched install.

With them applied, the full pipeline runs:

```sh
geoips run order_based seviri_hrit_mcidas_area /path/to/H-000-MSG4*
```

and produces eleven pixel-exact AREA files.
