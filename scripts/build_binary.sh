#!/usr/bin/env bash
# Build the standalone seviri2area binary into dist/, with a .sha256 beside it.
#
# The binary needs Python, numpy, this package and GeoIPS's HRIT parser --
# nothing else (see src/geoips_mcidas/cli.py). GeoIPS is therefore installed
# without its dependencies, and numpy and GeoIPS are pinned to uv.lock so the
# binary runs the versions the test suite ran.
#
# Runs on Linux, macOS, and Windows under Git Bash.
#
# Usage: scripts/build_binary.sh [output-name]
set -euo pipefail

cd "$(dirname "$0")/.."
name=${1:-seviri2area}
python=${PYTHON_VERSION:-3.12}
venv=.build-venv

rm -rf "$venv" build dist
uv venv --quiet --managed-python --python "$python" "$venv"
# Windows venvs keep executables in Scripts/ and binaries carry .exe.
bin=$venv/bin
exe=
if [ -d "$venv/Scripts" ]; then
	bin=$venv/Scripts
	exe=.exe
fi

# On macOS uv otherwise picks wheels for the build host's OS: numpy's are
# tagged macOS 14, so the binary would refuse to start on anything older.
# uv honours MACOSX_DEPLOYMENT_TARGET only alongside --python-platform.
target=
if [ "$(uname -s)" = Darwin ]; then
	export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
	target="--python-platform $(uname -m | sed 's/arm64/aarch64/')-apple-darwin"
fi

uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate \
	--extra geoips --format requirements-txt |
	grep -E '^(numpy|geoips)==' >"$venv/pins.txt"
# $target is deliberately unquoted: it is empty or two words.
# shellcheck disable=SC2086
{
	uv pip install --quiet --python "$venv" $target --no-deps -r "$venv/pins.txt"
	uv pip install --quiet --python "$venv" $target --no-deps .
	uv pip install --quiet --python "$venv" $target pyinstaller
}

# geoips/__init__ is never executed in the binary, but PyInstaller still
# follows its imports. Excluding them keeps GeoIPS's plugin machinery out.
"$bin/pyinstaller" --onefile --noconfirm --clean \
	--name "$name" \
	--copy-metadata geoips-mcidas \
	--exclude-module geoips.interfaces \
	--exclude-module geoips.commandline \
	--exclude-module geoips.xarray_utils \
	--exclude-module geoips.filenames \
	--exclude-module geoips.errors \
	--exclude-module pyPublicDecompWT \
	src/geoips_mcidas/cli.py

"dist/$name$exe" --version
# macOS ships shasum but not sha256sum; both write the same format.
(cd dist && if command -v sha256sum >/dev/null; then
	sha256sum "$name$exe"
else
	shasum -a 256 "$name$exe"
fi >"$name$exe.sha256")
