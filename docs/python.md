# Native Python package

The `oo-solver` distribution runs the C++ solver in the CPython process. It uses
the native CPU kernel selection and takes its model from the separately installable
`oo-reader` package. The Python and WebAssembly builds use the same analysis kernels.

## Analyses

Install the matching wheels with pip. For a local directory containing both wheels, run
`python -m pip install --find-links /path/to/wheels oo-solver`.
The interpreter, operating system and architecture must match the wheel tags.

```python
from oo_reader import read_model
from oo_solver import model_from_reader

deck = read_model("frequency-8.oos")
model = model_from_reader(deck, formulation="MixedHybridFormulation")
modes = model.run("modes", threads=1)
first_hz = modes["eigenvalues"][0]["rows"]["cycles"][0]
```

Build the snapshot once for repeated solves. The Reader exports immutable Arrow batches;
the Solver takes ownership of the Arrow buffers through their C release callbacks.
The nodal and element tables contain read-only NumPy views over the native result storage.
Retaining a view keeps its result allocation alive. Use `array.copy()` to keep a small
subset while releasing a much larger run. The solve releases the Python GIL.
The `threads` argument selects the native thread count.

The analysis names are `statics`, `modes`, `buckling` and `aeroelastic`. Result groups
include `nodal_tables`, `element_tables`, `grid_point_forces`, `eigenvalues` and
`diagnostics`. Trim runs also expose pressures, box forces, box moments and trim variables.
Further specialised outputs and settings are described by the WebAssembly API.

## External optimization

Declare variables, relations, responses, constraints and the objective in the model.
The optimizer owns the proposed points. Each evaluation retains its native solution
for subsequent derivative calls.

```python
from oo_solver import optimization_from_reader

problem = optimization_from_reader(read_model("sizing.oos"), analysis="statics")
x = problem.initial_values
point = problem.evaluate(x)
tip = point.responses[10]              # response ID from this model
jacobian = point.gradients()["responses"][10]
directional_change = point.jvp([1.0] * len(x))
weighted_gradient = point.vjp({10: [1.0]})  # for a single-item response
```

`problem.variables` lists the independent IDs, labels and bounds in design-vector order.
`point.objective` is the raw objective. `point.constraints` exposes bounds and normalized
values (feasible at or below zero). The gradient report counts field and full-analysis
perturbations. JVP concatenates response items in the order of `point.responses`.
VJP takes one weight per item of each selected response and returns a design-space
gradient. These products avoid constructing the dense Jacobian.

## Build and publish

Build with Python 3.10 or later, a C++20 compiler, CMake 3.25 or later and Ninja.
From a source checkout run `python -m pip wheel --no-deps . -w dist`.
Install the Reader wheel before testing the Solver. Run
`python python/tests/smoke.py` against the installed pair.

The **Python wheels** workflow builds CPython 3.10–3.14 wheels for Linux x86-64
and Windows x64, tests the installed distributions and creates the source archive.
Configure a PyPI trusted publisher for this repository, workflow
`python-wheels.yml` and environment `pypi`. Run the workflow with publishing enabled
when releasing a new version. Release the matching Reader first so pip can install
the Solver dependency during verification. The default workflow run only builds artifacts.
