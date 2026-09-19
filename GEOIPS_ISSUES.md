# Upstream issues found while building geoips_mcidas

Defects in GeoIPS core and `pluginify` encountered while writing an out-of-tree
plugin package (a DataTree-native reader and output formatter, plus a
family-bearing output checker). Each entry is written to be pasted into a
GitHub issue.

Environment for every reproduction below:

| Component | Version                      |
| --------- | ---------------------------- |
| geoips    | 1.20.0a0, commit `751e6533e` |
| pluginify | 1.0.0                        |
| xarray    | 2026.7.0                     |
| numpy     | 2.5.3                        |
| Python    | 3.13.5                       |
| Platform  | macOS 15 arm64               |

Ordered by severity.

______________________________________________________________________

## 1. Registry build aborts for every installed package when one plugin omits `family`

**Repository:** pluginify
**Severity:** blocker

### Summary

`collect_class_plugin_metadata` reads `plugin_class.family` without a default.
A DataTree-native plugin has no `family` by design, so the attribute read
raises `AttributeError`. The exception propagates out of
`create_plugin_registries`, which means no registry is written for *any*
installed package, not just the one holding the offending plugin.

### Steps to reproduce

Define a plugin that follows the DataTree-native pattern the output formatter
interface recommends:

```python
class ExampleOutputFormatterPlugin(BaseOutputFormatterPlugin):
    interface = "output_formatters"
    name = "example"
    data_tree = True
    required_attributes = ("interface", "name")  # drops "family"

    def call(self, data, output_directory):
        ...

PLUGIN_CLASS = ExampleOutputFormatterPlugin
```

Install the package and build registries:

```python
from importlib.metadata import entry_points
from pluginify.create_plugin_registries import create_plugin_registries

create_plugin_registries(list(entry_points(group="geoips.plugin_packages")), "json", "geoips")
```

### Actual behaviour

```
AttributeError: type object 'ExampleOutputFormatterPlugin' has no attribute 'family'
  File ".../pluginify/create_plugin_registries.py", line 892, in collect_class_plugin_metadata
    "family": plugin_class.family,
```

Nothing is written for geoips, for geoips_clavrx, or for the new package.
`geoips config create-registries` fails the same way.

### Expected behaviour

A plugin without `family` is registered with `family` recorded as null.

### Suggested fix

`create_plugin_registries.py:892`:

```python
"family": getattr(plugin_class, "family", None),
```

The surrounding code already carries a comment noting that attribute checks are
missing here:

```python
# NOTE: Probably should add some attribute checks here as
# 'collect_module_plugin_metadata' does
```

Patching this line in-process lets the full registry build succeed across all
three installed packages.

### Related

Issue 2 blocks plugin *resolution* even after this fix, so both are needed
before a family-less plugin can be used.

______________________________________________________________________

## 2. `plugin_is_valid` rejects a family-less plugin after the registry is built

**Repository:** pluginify
**Severity:** blocker

### Summary

`BaseInterface.plugin_is_valid` looks the plugin's family up in
`required_args` unconditionally. For a DataTree-native plugin the value is
absent (or `None` after issue 1 is fixed), and neither is a key in
`required_args`, so resolution raises.

### Location

`pluginify/interfaces/base.py:533`:

```python
if plugin.family not in self.required_args:
    raise PluginError(...)
```

with the same pattern at line 541 for `required_kwargs`. Reached from
`plugin_registry.py:750` via `interface_obj.plugin_is_valid(plugin)` inside
`get_class_plugin`.

### Why the fix for issue 1 is not sufficient

`getattr(plugin_class, "family", None)` makes the registry build succeed, but
`None` is not a key in `required_args`, so `interfaces.readers.get_plugin(...)`
still raises. The two sites need separate handling.

### Suggested fix

Return early for DataTree-native plugins, before the family lookup:

```python
if getattr(plugin, "data_tree", False):
    return True
```

### Impact

Until both are fixed, a DataTree-native plugin can only be imported and called
directly. It cannot be resolved by name, which means it cannot appear in an
order-based workflow.

______________________________________________________________________

## 3. `@dataclass` cannot be used in a plugin module

**Repository:** pluginify
**Severity:** high

### Summary

`add_class_plugin` loads plugin modules with `spec.loader.exec_module(module)`
and never inserts the module into `sys.modules`. `@dataclass` resolves field
types by looking its defining module up in `sys.modules`, gets `None`, and
raises. Any plugin module containing a dataclass therefore fails at
registry-build time.

This affects all class-based plugins, not only DataTree-native ones.

### Steps to reproduce

Add to any plugin module:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Example:
    value: str
```

Then build the registry.

### Actual behaviour

```
AttributeError: 'NoneType' object has no attribute '__dict__'
  File ".../dataclasses.py", line 757, in _is_type
    ns = sys.modules.get(cls.__module__).__dict__
  File ".../pluginify/create_plugin_registries.py", line 951, in add_class_plugin
    spec.loader.exec_module(module)
```

The traceback points into `dataclasses`, which makes the cause hard to find:
nothing in it mentions the loader.

### Suggested fix

Register the module before executing it, which is the documented pattern for
`importlib.util.spec_from_file_location`:

```python
module = util.module_from_spec(spec)
sys.modules[module_path] = module
try:
    spec.loader.exec_module(module)
except ImportError:
    del sys.modules[module_path]
    raise
```

### Workaround

Define dataclasses in an ordinary module outside `plugins/` and import them.

______________________________________________________________________

## 4. Every `geoips` CLI subcommand raises `KeyError: 'geoips_config_create'`

**Repository:** geoips
**Severity:** high

### Summary

The command-line interface is unusable on commit `751e6533e`. Every
subcommand fails during argument-parser construction, before any user input is
read.

### Steps to reproduce

```
$ geoips list readers
$ geoips config create-registries
$ geoips --help
```

### Actual behaviour

```
KeyError: "Error, the supplied command line instructions are improperly
formatted. You need an 'instructions' entry that contains a
'geoips_config_create' key."
  File ".../geoips/commandline/geoips_command.py", line 325, in __init__
```

Raised from `commandline_interface.py:116` during `__init__`, so no subcommand
is reachable.

### Probable cause

A command class expects a `geoips_config_create` entry in the command
instructions YAML that the shipped file does not provide. Either the
instructions file is missing the entry or the class was renamed without
updating it.

### Impact

`geoips list`, `geoips describe`, `geoips config create-registries`,
`geoips run` and `geoips test` are all unavailable, so registries can only be
built by calling `pluginify.create_plugin_registries` directly.

______________________________________________________________________

## 5. Order-based procflow passes DataTree-native plugins a wrapper node with no documented way to reach dependencies

**Repository:** geoips
**Severity:** high
**Type:** API gap

### Summary

`Workflow._collect_upstream_data` builds a fresh `DataTree(name="multi_input")`
per step and attaches each dependency as a child. A family-bearing plugin never
sees this: its conduit in `OBP_CONDUITS` extracts the relevant node into a
typed kwarg such as `xarray_obj`. A DataTree-native plugin has no conduit, so
it receives the wrapper itself, whose root dataset is empty.

Nothing in the interface documentation states this. The natural implementation
(`data.to_dataset()`) therefore returns an empty dataset, and the plugin
reports missing input for input that is present.

### Steps to reproduce

```python
import xarray as xr

upstream = xr.DataTree(name="multi_input")
upstream["reader"] = xr.DataTree(dataset=reader_output, name="reader")

plugin(upstream, output_directory, _obp_initiated=True)
```

with a DataTree-native output formatter whose `call` starts `data.to_dataset()`.

### Actual behaviour

```
multi_input root data_vars: []
children: ['reader']
```

The plugin sees no variables and fails, attributing the failure to its input
data rather than to the wrapper.

### Relevant source

- `geoips/interfaces/class_based/workflow.py:278-289` — wrapper construction
- `geoips/interfaces/class_based/workflow.py:612` — `plg(data=upstream, _obp_initiated=True, **arguments)`
- `geoips/utils/types/obp_conduits.py:147` — `"reader": {"kwarg": "xarray_obj", "extract": _extract_ds}`

### Suggested resolution

Either of:

1. Document that a DataTree-native plugin receives the `multi_input` wrapper
   and must resolve its own dependencies by child name, and provide a helper
   for it.
1. Apply the conduit extraction for DataTree-native plugins as well, so the
   payload arrives in the same shape a family-bearing plugin would get.

The first is a documentation change; the second changes behaviour. Either way,
the current combination is a trap, because the failure surfaces as a data
problem rather than a wiring problem.

______________________________________________________________________

## 6. `geoips test workflow` ignores `compare_path`

**Repository:** geoips
**Severity:** medium

### Summary

A workflow's `test:` block accepts `compare_path` and `output_checker_name`,
but `geoips test workflow` runs the workflow and does not compare its output.
A test whose outputs have regressed still passes.

### Location

`geoips/commandline/geoips_test.py:445`:

```python
# TODO: Add additional logic here for other parameters included in a workflow
```

### Impact

Regression testing for order-based workflows has to go through
`single_source` with `--compare_path`, which is unavailable to plugins without
a `family`, since `single_source.plot_data` dispatches on it. A DataTree-native
plugin currently has no supported output-comparison path.

______________________________________________________________________

## 7. Workflow validator accepts an unknown step `kind`

**Repository:** geoips
**Severity:** medium

### Summary

`workflows.validator.validate` returns successfully for a workflow whose step
declares a `kind` that does not correspond to any interface. The error only
appears at run time, after input data has been read.

### Steps to reproduce

```python
from geoips.interfaces import workflows

bad = {
    **valid_workflow,
    "spec": {"globals": {}, "steps": {"reader": {"kind": "not_a_kind"}}},
}
workflows.validator.validate(bad)   # returns a dict, raises nothing
```

### Expected behaviour

Validation rejects a `kind` outside the set of known interfaces, and a `name`
that is not a registered plugin for that interface.

### Impact

Workflow YAML cannot be checked in CI. A typo in a step name or kind is found
only by running the workflow.

______________________________________________________________________

## 8. `seviri_hrit` discards raw counts and the calibration prologue

**Repository:** geoips
**Severity:** medium
**Type:** enhancement

### Summary

`geoips/plugins/classes/readers/seviri_hrit.py` converts counts to radiance and
retains only the result. Both the counts and the prologue holding the per-band
slope and intercept are dropped, so a consumer needing either must re-read the
HRIT files.

### Details

- `countsToRad` (line 622) applies `offset + slope * counts` and clamps
  `counts <= 0` to `-999.0`, so the mapping is not invertible at the low end.

- The slope and offset come from
  `prologue["radiometricProcessing"]["level15ImageCalibration"]` at line 986
  and are not stored on the returned object.

- Line 794 carries the assignment that would retain the prologue, commented
  out:

  ```python
  # xarray_obj.attrs['prologue'] = pro
  ```

  The epilogue is stored (line 815); the prologue is not.

### Use case

Writing McIDAS AREA files for CLAVR-x requires raw counts plus the prologue
gain and offset verbatim, because CLAVR-x applies the calibration itself.
Re-reading and re-decompressing 90 HRIT segments to recover data the reader
already had in memory costs about 20 seconds per scene.

### Suggested change

A `retain_counts=True` option that adds count variables alongside the existing
`Rad`/`Ref`/`BT` ones, and uncommenting the prologue assignment or exposing the
calibration coefficients as an attribute.

______________________________________________________________________

## 9. `geoips` ships no `py.typed`, so downstream type checking is silently degraded

**Repository:** geoips
**Severity:** low

### Summary

There is no `py.typed` marker in the `geoips` package. Under PEP 561 a type
checker therefore treats the package as untyped, and every symbol imported from
it resolves to `Any`.

### Impact

A downstream package running `mypy --strict` gets a pass that is weaker than it
appears: calls into GeoIPS are unchecked, and `disallow_untyped_calls` does not
fire because the imports are `Any` rather than untyped. The failure mode is a
false sense of coverage.

### Suggested fix

Add an empty `geoips/py.typed` and include it in package data. Worth pairing
with a check that the annotations already present are accurate, since marking
the package typed makes them load-bearing.

______________________________________________________________________

## 10. Plugin development tutorial documents an entry-point group that is not read

**Repository:** geoips
**Severity:** low
**Type:** documentation

### Summary

`docs/source/tutorials/extending-with-plugins/output_formatter.rst:156`
instructs the reader to declare a per-interface entry point:

```toml
[project.entry-points."geoips.output_formatters"]
my_netcdf_output = "cool_plugins.plugins.classes.output_formatters.my_netcdf_output"
```

Nothing reads `geoips.output_formatters`. Discovery is by recursive scan of
`<package>/plugins/`, keyed off the single `geoips.plugin_packages` entry
point. `docs/source/architecture/plugin-registry.md:23-26` documents the
working form.

`docs/source/tutorials/setting-up-for-plugin-development.md:89-90` repeats the
per-interface claim.

### Impact

Following the tutorial produces a package whose plugins are not found, with no
error indicating why. The correct stanza is:

```toml
[project.entry-points."geoips.plugin_packages"]
my_package = "my_package"
```
