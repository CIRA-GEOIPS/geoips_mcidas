{
  description =
    "geoips_mcidas - McIDAS AREA file output formatter for GeoIPS, targeting CLAVR-x SEVIRI ingest.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;

        # Native libraries the GeoIPS scientific stack links against. Only
        # needed for the optional `geoips` extra -- the AREA writer core is
        # pure numpy and needs none of them.
        nativeLibs = with pkgs; [
          geos
          proj
          hdf5
          netcdf
          hdf4
          eccodes
          openblas
          zlib
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = nativeLibs ++ (with pkgs; [
            python
            uv
            pkg-config
            # gcc/gfortran are here for building McIDAS-X as a reference
            # generator for golden files, not for this package itself.
            gcc
            gfortran
            gnumake
          ]);

          shellHook = ''
            export PROJ_DIR=${pkgs.proj}
            export GEOS_DIR=${pkgs.geos}
            export HDF5_DIR=${pkgs.hdf5}
            export ECCODES_DIR=${pkgs.eccodes}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH
            export UV_PYTHON=${python}/bin/python
            echo "geoips_mcidas dev shell - 'uv run pytest' to test."
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
