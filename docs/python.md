# Native Python package

The `oo-solver` distribution runs the C++ solver in the CPython process. It uses
the native CPU kernel selection and takes its model from the separately installable
`oo-model` package. The Python and WebAssembly builds use the same analysis kernels.

## Analyses

Run `python -m pip install oo-solver==0.2.1` to get the Solver and its `oo-model` dependency. For a development build from adjacent checkouts, run `python -m pip install ../O_o_Model .` from the Solver directory.
Extract `frequency-8.oos` from the [benchmark inputs](https://github.com/jcdulas/O_o_Solver/raw/refs/heads/main/docs/benchmarks/structural-comparison-inputs.zip). Both examples give about **28.7583 Hz** for the first bending mode.
Initialize `reader = await loadOoModel()` and `solver = await loadOosSolver()` with the module imports in [Getting started](getting-started.md).
This benchmark uses the optional OOS adapter. In Node.js, copy the downloaded input into the model module with `reader.writeFile("/work/frequency-8.oos", await readFile("frequency-8.oos"))` after importing `readFile` from `node:fs/promises`. The [in-memory example](getting-started.md#run-linear-statics) supplies the deck entirely through API calls.

**JavaScript / WebAssembly**

```javascript
const deck = reader.readModel("/work/frequency-8.oos");
const model = solver.modelFromReader(reader, deck, { elementFormulation: "MixedHybridFormulation" });
try {
  const modes = solver.run(model, "modes", { threads: 1 });
  const firstHz = modes.eigenvalues[0].rows[0].cycles;
  console.log(firstHz);
} finally {
  model.delete(); deck.delete();
}
```

**Python / native C++**

```python
from oo_model import read_model
from oo_solver import model_from_reader

deck = read_model("frequency-8.oos")
model = model_from_reader(deck, formulation="MixedHybridFormulation")
modes = model.run("modes", threads=1)
first_hz = modes["eigenvalues"][0]["rows"]["cycles"][0]
print(first_hz)
```

Build the snapshot once for repeated solves. The Model exports immutable Arrow batches;
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

Use the in-memory rod from [Getting started](getting-started.md#run-linear-statics). Add the following design cards to the deck before releasing it. Response 10 is the axial displacement and variable 1 is its section area. Both examples return a displacement of 0.015 and a derivative of −0.0075 at area 2.
The optimizer owns the proposed points. Each evaluation retains its native solution
for subsequent derivative calls.

**JavaScript / WebAssembly**

```javascript
deck.addCards([
  { type: "DesignVar", fields: [1, "AREA", 2, 0.5, 5] },
  { type: "DesignPropRelation1", fields: [1, "RodProp", 1, "area", null, null, null, 1, 1] },
  { type: "DesignResp1", fields: [10, "TIP", "Displacement", null, null, 1, null, 2] },
]);
const problem = solver.optimizationFromReader(api, deck);
const point = problem.evaluate(problem.metadata().initialValues);
try {
  const tip = point.values().responses.find(r => r.id === 10).values[0];
  const gradient = point.gradients().responses.find(r => r.id === 10).values[0];
  console.log(tip, gradient);
  const directionalChange = point.jvp([1]);
  const weightedGradient = point.vjp({ responses: { 10: [1] } });
} finally {
  point.delete(); problem.delete();
}
```

**Python / native C++**

```python
from oo_solver import optimization_from_reader

deck.add_card("DesignVar", [1, "AREA", 2., 0.5, 5.])
deck.add_card("DesignPropRelation1", [1, "RodProp", 1, "area", None, None, None, 1, 1.])
deck.add_card("DesignResp1", [10, "TIP", "Displacement", None, None, 1, None, 2])
problem = optimization_from_reader(deck, analysis="statics")
x = problem.initial_values
point = problem.evaluate(x)
tip = point.responses[10]              # response ID from this model
jacobian = point.gradients()["responses"][10]
directional_change = point.jvp([1.0] * len(x))
weighted_gradient = point.vjp({10: [1.0]})  # for a single-item response
print(tip[0], jacobian[0, 0])
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
Install the Model wheel before testing the Solver. Run
`python python/tests/smoke.py` against the installed pair.

The **Python wheels** workflow builds CPython 3.10–3.14 wheels for Linux x86-64
and Windows x64, tests the installed distributions and creates the source archive.
Configure a PyPI trusted publisher for this repository, workflow
`python-wheels.yml` and environment `pypi`. Run the workflow with publishing enabled
when releasing a new version. Release the matching Model first so pip can install
the Solver dependency during verification. The default workflow run only builds artifacts.
