# geoips_mcidas

A [GeoIPS](https://github.com/NRLMMD-GEOIPS/geoips) plugin package that writes McIDAS AREA files from MSG SEVIRI data, for ingest by [CLAVR-x](https://gitlab.ssec.wisc.edu/clavrx/clavrx-dev).

## Why

CLAVR-x has no HRIT reader. McIDAS AREA is its only native SEVIRI L1b input
path, so pipelines that run CLAVR-x over SEVIRI must first convert HRIT to AREA.

Usually this is done with MciDAS-X, but this allows you to do that with GeoIPS instead
for use in a GeoIPS workflow. Yay!

## Provenance

The AREA format implementation is written from the public
[McIDAS Programmer's Manual](https://www.ssec.wisc.edu/mcidas/doc/prog_man/2015/formats-1.html)
and from measurements of real AREA files.

## Standalone binary

Until GeoIPS can run the workflow (see the warning below), `seviri2area` does
the conversion without it. Each
[release](https://github.com/CIRA-GEOIPS/geoips_mcidas/releases) attaches a
self-contained binary, with a `.sha256` beside it, for:

| Platform | x86_64                           | arm64                           |
| -------- | -------------------------------- | ------------------------------- |
| Linux    | `seviri2area-linux-x86_64`       | `seviri2area-linux-aarch64`     |
| macOS    | `seviri2area-macos-x86_64`       | `seviri2area-macos-arm64`       |
| Windows  | `seviri2area-windows-x86_64.exe` | `seviri2area-windows-arm64.exe` |

Linux needs glibc 2.28+ (RHEL/Rocky 8 and newer), macOS needs 11+. No Python
is needed on the host.

```sh
chmod +x seviri2area-linux-x86_64
./seviri2area-linux-x86_64 /path/to/hrit_cycle/ -o /path/to/area_out
```

The binaries are not signed. On macOS, clear the download quarantine first
with `xattr -d com.apple.quarantine seviri2area-macos-*`; on Windows,
SmartScreen may ask for confirmation on first run.

Pass the uncompressed HRIT files for one cycle (prologue plus every segment of
bands 1-11), or a directory that holds only that cycle. It writes the eleven
`metNN_B_YYYY_DDD_HHMM.area` files and prints their paths, band 1 first. Band 1
is the file to list for CLAVR-x. From a checkout, `uv run seviri2area` does the
same thing.

To cut a release, bump `version` in `pyproject.toml` and push a matching `vX.Y.Z`
tag. `scripts/build_binary.sh` builds the binary for the current platform.

## Development

```sh
nix develop      # toolchain
uv run pytest    # tests
```

## Verification status

The repository ships with a matched input/output pair as the basis for testing:

- the HRIT segments for 2021-05-21 01:00
- the AREA file McIDAS-X produced from those segments

Tests marked `golden` compare against it to make sure things don't break :(

## So far:

- Counts match McIDAS pixel for pixel
- Navigation and calibration blocks match byte for byte
- **A McIDAS-written file works directly as a `--compare_path` reference.** The
  output checker ignores provenance (creation stamps, AREA slot number and the
  audit-trail cards McIDAS records and we totally do not).

## WARNING - this is currently unsupported by GeoIPS

The reader and output formatter are DataTree-native which are currently .... broken in GeoIPS.

Once that supported, you can run this like this:

```sh
geoips run order_based seviri_hrit_mcidas_area /path/to/H-000-MSG4*
```

and produce pixel-exact AREA files :raised_hands:
