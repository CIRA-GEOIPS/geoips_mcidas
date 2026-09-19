"""Discovery of these plugins by GeoIPS.

The package registers a single ``geoips.plugin_packages`` entry point and
GeoIPS scans ``plugins/`` beneath it. That much works. Registry *metadata*
collection does not yet, for the reason captured below.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

pytest.importorskip("geoips")

from geoips_mcidas.plugins.classes.output_checkers import mcidas_area
from geoips_mcidas.plugins.classes.output_formatters import mcidas_area_seviri
from geoips_mcidas.plugins.classes.readers import seviri_hrit_counts

DATATREE_NATIVE = (
    seviri_hrit_counts.PLUGIN_CLASS,
    mcidas_area_seviri.PLUGIN_CLASS,
)


def test_entry_point_is_registered() -> None:
    names = {e.name for e in entry_points(group="geoips.plugin_packages")}
    assert "geoips_mcidas" in names


@pytest.mark.parametrize("plugin_class", DATATREE_NATIVE)
def test_datatree_native_plugins_declare_no_family(plugin_class: type) -> None:
    """``data_tree = True`` plus no ``family`` is the migration GeoIPS asks
    for; the interface then passes data through unconverted both ways."""
    assert plugin_class.data_tree is True
    assert not hasattr(plugin_class, "family")
    assert "family" not in plugin_class.required_attributes


def test_output_checker_keeps_its_family() -> None:
    """Output checkers have no DataTree-native mode; ``standard`` is correct."""
    assert mcidas_area.PLUGIN_CLASS.family == "standard"


@pytest.mark.xfail(
    reason=(
        "pluginify 1.0.0 reads plugin_class.family unconditionally in "
        "collect_class_plugin_metadata, so a DataTree-native plugin cannot be "
        "registered. Fix is one line: getattr(plugin_class, 'family', None). "
        "Until then these plugins must be invoked directly rather than "
        "resolved by name."
    ),
    raises=AttributeError,
    strict=True,
)
def test_registry_build_accepts_family_less_plugins() -> None:
    """Flips to passing once pluginify tolerates a missing ``family``."""
    from pluginify.create_plugin_registries import collect_class_plugin_metadata

    metadata = collect_class_plugin_metadata(mcidas_area_seviri.PLUGIN_CLASS)
    assert metadata["name"] == "mcidas_area_seviri"
    assert metadata["family"] is None
