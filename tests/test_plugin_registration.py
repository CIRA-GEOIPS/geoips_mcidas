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


def _pluginify_tolerates_missing_family() -> bool:
    """Whether the installed pluginify can register a family-less plugin.

    pluginify 1.0.0 reads ``plugin_class.family`` unconditionally in
    ``collect_class_plugin_metadata`` and again in ``plugin_is_valid``, so a
    DataTree-native plugin aborts the registry build for every installed
    package. See GEOIPS_ISSUES.md issues 1 and 2. These tests assert the
    behaviour we need and skip, rather than fail, on an unpatched install.
    """
    from pluginify.create_plugin_registries import collect_class_plugin_metadata

    try:
        collect_class_plugin_metadata(mcidas_area_seviri.PLUGIN_CLASS)
    except AttributeError:
        return False
    return True


requires_patched_pluginify = pytest.mark.skipif(
    not _pluginify_tolerates_missing_family(),
    reason="installed pluginify cannot register family-less plugins; "
    "see GEOIPS_ISSUES.md",
)


@requires_patched_pluginify
def test_registry_metadata_records_a_null_family() -> None:
    from pluginify.create_plugin_registries import collect_class_plugin_metadata

    metadata = collect_class_plugin_metadata(mcidas_area_seviri.PLUGIN_CLASS)
    assert metadata["name"] == "mcidas_area_seviri"
    assert metadata["family"] is None
    assert metadata["interface"] == "output_formatters"


@requires_patched_pluginify
@pytest.mark.parametrize(
    ("interface_name", "plugin_name"),
    [
        ("readers", "seviri_hrit_counts"),
        ("output_formatters", "mcidas_area_seviri"),
        ("output_checkers", "mcidas_area"),
    ],
)
def test_plugins_resolve_by_name(interface_name: str, plugin_name: str) -> None:
    """Resolution by name is what an order-based workflow actually needs."""
    from geoips import interfaces

    plugin = getattr(interfaces, interface_name).get_plugin(plugin_name)
    assert plugin.name == plugin_name
