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
