"""The order-based workflow YAML.

GeoIPS's own workflow validator accepts a step with a nonsense ``kind``, so
"it validates" would say very little. These tests check the things that
actually break a run: that the plugin names in the YAML match the plugins this
package ships, and that the pipeline is reader-to-formatter with nothing in
between.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("geoips")

from geoips_mcidas.plugins.classes.output_formatters import mcidas_area_seviri
from geoips_mcidas.plugins.classes.readers import seviri_hrit_counts

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "geoips_mcidas"
    / "plugins"
    / "yaml"
    / "workflows"
    / "seviri_hrit_mcidas_area.yaml"
)


class EnvLoader(yaml.SafeLoader):
    """Resolve GeoIPS's ``!ENV`` tag to its literal text."""


def _resolve_env(loader: EnvLoader, node: yaml.ScalarNode) -> str:
    """Resolve GeoIPS's !ENV tag to its literal text."""
    return str(loader.construct_scalar(node))


EnvLoader.add_constructor("!ENV", _resolve_env)


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    # EnvLoader subclasses SafeLoader, so this cannot construct arbitrary
    # objects; it only teaches the safe loader about GeoIPS's !ENV tag.
    loaded: dict[str, Any] = yaml.load(
        WORKFLOW.read_text(),
        Loader=EnvLoader,  # noqa: S506
    )
    return loaded


def test_declares_this_package(workflow: dict) -> None:
    assert workflow["package"] == "geoips_mcidas"
    assert workflow["interface"] == "workflows"
    assert workflow["family"] == "order_based"


def test_step_names_match_shipped_plugins(workflow: dict) -> None:
    """A typo here fails only at run time, and only after reading the data."""
    steps = workflow["spec"]["steps"]
    assert steps["reader"]["name"] == seviri_hrit_counts.PLUGIN_CLASS.name
    assert steps["output_formatter"]["name"] == mcidas_area_seviri.PLUGIN_CLASS.name


def test_no_resampling_steps(workflow: dict) -> None:
    """CLAVR-x hardcodes full-disk navigation constants, so a sectored or
    interpolated product would be silently mis-navigated rather than simply
    different. The pipeline must hand it native counts."""
    kinds = {step["kind"] for step in workflow["spec"]["steps"].values()}
    assert kinds == {"reader", "output_formatter"}
    assert not kinds & {"sector", "interpolator", "algorithm", "colormapper"}


def test_formatter_receives_an_output_directory(workflow: dict) -> None:
    """The formatter owns its filenames but not its destination."""
    arguments = workflow["spec"]["steps"]["output_formatter"]["arguments"]
    assert "output_directory" in arguments


def test_formatter_depends_on_the_reader(workflow: dict) -> None:
    assert workflow["spec"]["steps"]["output_formatter"]["depends_on"] == ["reader"]
