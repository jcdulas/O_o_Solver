# Optimization controlled by an external application

The examples assume a loaded model named `deck`. In JavaScript, `oor` and `solver` are the loaded WASM modules. In Python, `deck` comes from `oo_model.read_model`. See [Getting started](getting-started.md) for package setup.

An application owns the optimizer, trial points, acceptance rule and convergence test. Model owns the editable
model; Solver owns analysis snapshots and evaluates SOL 101, 103, 105 and 144. There are two entry points:

- `solver.optimizationFromReader` uses the native design model for declared sizing, shape and response types. Its
  analytic and adjoint derivatives are generally the fastest choice where supported.
- `solver.createStudy` edits a copy of a Model deck at each point. It can change card fields, case control and the
  solution, then select any numeric result or named Model query. It obtains derivatives by full analysis differences.
  Use this for model changes or output fields outside the native design model.

The two routes share Model and Solver packages. Neither route chooses a design point for the application.

## Inspect support before building an optimizer

**JavaScript / WebAssembly**

```js
const catalogue = solver.capabilities();
// catalogue.analyses lists 101, 103, 105 and 144, with derivative restrictions.
const model = solver.modelFromReader(oor, deck);
try {
  const available = solver.capabilities(model); // includes numeric property/material/element columns
  const prepared = solver.checkOptimization(model, declarations, settings);
  if (!prepared.canPrepare) throw new Error(prepared.error ?? JSON.stringify(prepared.analyses));
} finally { model.delete(); }
const readerSupport = oor.designCapabilities(deck); // editable cards and Model design diagnostics
```

**Python / native C++**

Python exposes the capability catalogue on the prepared problem. The JavaScript preflight report additionally checks declarations before problem construction.

```python
from oo_solver import optimization_from_reader

deck.validate_design()
problem = optimization_from_reader(deck)
print(problem.variables, problem.capabilities["analyses"])
```

`checkOptimization` checks construction and analysis support without solving. It cannot promise convergence or a
valid derivative at every point. A native optimization can request `analyses: { statics: [], modes: [], buckling: [],
aeroelastic: [] }`; an empty array selects all subcases for that solution. Direct `solver.run(model, "aeroelastic")`
returns typed structural tables and `aeroelastic` trim data. Its subcases include pressures, box forces and moments,
trim variables, stability derivatives, hinge moments and monitoring loads.

| Solution | Native design derivatives | Study responses and derivatives |
| --- | --- | --- |
| 101 static | Supported sizing and shape responses; selected second order products | Any stable numeric table/query result, full analysis differences |
| 103 modes | Eigenvalue, frequency and mode-selected field gradients; analytic Hessian-vector products where element second derivatives are closed | Eigenvalues, eigenvectors and modal stresses; mode references and full analysis differences |
| 105 buckling | Load factor gradients including static or trim preload, with analytic curvature for supported sizing variables | Load factors and result fields, full analysis differences |
| 144 static aeroelastic | Sizing and dynamic-pressure gradients, restrained spline/RBE3 shape tangents, and analytic Hessian products for supported structural sizing | Trim and structural outputs, including driven load changes, full analysis differences |

The [native design module](../src/design/README.md) details the exact response formulations and restrictions.
The [derivative coverage guide](derivatives.md) explains how to distinguish analytic and numerical paths in reports.
The [modal derivative guide](modal-derivatives.md) describes field selection, normalisation and the bordered systems
used by the C++ first- and second-order calculations.
The [coupled derivative guide](coupled-derivatives.md) describes analytic buckling and trim products and their limits.
When an explicit condensation plan places a driven unit inside a condensed part, native SOL 144 gradients run that
trim on the full structural path. This can increase memory use. Applied trim loads also require the full path.
`derivatives: "numerical"` enables full analysis gradients for a native design point when its native linearization is
unavailable. The returned report identifies `finiteDifference` and the number of additional evaluations. Check
derivatives against representative points, especially around contact changes, buckling mode crossings and trim limits.

## Define a study with arbitrary Model edits

**JavaScript / WebAssembly**

```js
const study = solver.createStudy(oor, deck, {
  version: 1,
  variables: [
    { id: 1, initialValue: 2, lower: 0.5, upper: 5 },
    { id: 2, initialValue: 10, lower: 5, upper: 20 },
  ],
  parameters: [
    { variableId: 1, target: { kind: "card", type: "RodProp", id: 1, field: "area" } },
    { variableId: 2, target: { kind: "card", type: "PointForceLoad", id: 2, slot: 4 } },
  ],
  analyses: [{ analysis: "statics", subcases: [1] }],
  responses: [{ id: 10, source: "nodal", kind: "displacements", subcase: 1,
    ids: [2], components: "t1" }],
  objective: { responseId: 10 },
  constraints: [{ responseId: 10, upper: 0.02 }],
});
try {
  const point = study.evaluate();
  try {
    console.log(point.values().objective, point.gradients().values);
    console.log(point.jvp([1, 0]).values, point.vjp(point.lagrangianWeights()).values);
    point.save("/work/accepted.ooh5", { state: { iteration: 1 } });
  } finally { point.delete(); }
} finally { study.delete(); }
```

**Python / native C++**

For an externally supplied area and load, evaluate the same tip response on a trial copy. Gradients through `optimization_from_reader` require corresponding design cards.

```python
from oo_solver import model_from_reader

trial = deck.copy()
trial.set_card_value("RodProp", 1, "area", 2.0)
trial.set_card_value("PointForceLoad", 2, "magnitude", 10.0)
run = model_from_reader(trial).run("statics", subcases=[1])
u = next(t for t in run["nodal_tables"] if t["kind"] == "displacements")
tip = u["values"][u["ids"].tolist().index(2), 0]
```

Card targets accept a persistent `key`, an `id`, a named field or a one-based raw slot, and a part descriptor. Case
targets set a command in a subcase or section; solution targets set the solution number. Parameters can combine
variables linearly or call a named synchronous hook. Variable kinds include `continuous`, `integer` and `choice`;
finite differences require continuous variables. Model's `deck.copy()` gives each trial an independent copy-on-write
model. The source deck stays untouched. Call `study.rebuild(newDeck, ...)` after accepting external edits.

Responses select nodal, element, grid point force, eigenvalue, pressure, box force/moment, hinge moment, trim variable
or stability derivative fields. They may also use a Model named query, including queries over the generated result
tables, or a named synchronous callback. Row selectors,
components, subcases, modes and reductions control the scalar/vector output; every response has a stable ID. See
[`StudyResponse`](../wasm/js/study.d.ts) for the complete contract. Hooks are registered separately from the
serializable definition:

**JavaScript / WebAssembly**

```js
const hooks = { responses: {
  stressCost: ({ runs }) => Math.max(...runs[0].elementTables
    .filter(table => table.kind === "stress").flatMap(table => Array.from(table.values))),
} };
// A response with { source: "callback", callback: "stressCost", id: 20 } calls this hook.
```

**Python / native C++**

```python
stress_cost = max(t["values"].max() for t in run["element_tables"]
                  if t["kind"] == "stress" and t["values"].size)
```

Request stress output in the model so the reduction has data.

For gradients, `point.gradients()` returns a row-major response/constraint/objective Jacobian with stable `keys`.
`jvp`, `vjp`, `hessianVector` and `hessian` compute full analysis products; the Hessian is a nested numerical
difference and can be costly. Steps respect variable bounds and use second-order one-sided differences at a bound.
Use `onBlock` for a large Jacobian and `maximumOutputBytes` to set a memory limit. A changed output layout fails
instead of silently matching unrelated rows. Tracked modal responses expose MAC and uncertainty; an uncertain or
repeated root refuses a smooth Jacobian by default. A forward directional `jvp` remains available. The application
should inspect the mode identity and treat nonsmooth points explicitly.

An auxiliary SOL 101 displacement field can generate a compact batched native shape basis once:

**JavaScript / WebAssembly**

```js
const basis = solver.shapeBasisFromReader(oor, auxiliaryDeck,
  { variableId: 7, cases: [{ subcase: 1, coefficient: 1 }], gridIds: [2, 3] });
const problem = solver.optimizationFromReader(oor, deck, {
  variables: [{ id: 7, initialValue: 0, lower: -1, upper: 1 }],
  shapeBatches: [basis], responses: designResponses,
});
```

**Python / native C++**

The helper that derives a shape basis from an auxiliary solve is exposed in JavaScript. In Python, supply the shape variables and relations in the deck before calling `optimization_from_reader(deck)`.

The basis converts displacement components into the basic frame. The auxiliary solve and Model geometry lookup run
once; later native evaluations consume the typed grid/direction arrays.

## Parallel evaluations and interruption

`study.evaluateMany(points)` yields between sequential points and accepts an `AbortSignal`. A synchronous WASM
solve cannot be interrupted on its own thread. Use `createOptimizationWorkerPool` when evaluations must run in
parallel or a hung solve must be stopped:

**JavaScript / WebAssembly**

```js
const pool = await createOptimizationWorkerPool({
  readerURL: new URL("../oo-model/index.js", import.meta.url),
  solverURL: new URL("../oos-solver/index.js", import.meta.url),
  model: modelBytes, mode: "study", definition, workers: 2,
});
try {
  const trial = await pool.evaluate(x, { gradients: true, timeoutMs: 30000, signal });
  console.log(trial.values.objective, trial.gradients.values);
} finally { pool.close(); }
```

**Python / native C++**

For independent model files, schedule each solve in its own process. The worker-pool timeout and cancellation protocol above belongs to JavaScript.

```python
from concurrent.futures import ProcessPoolExecutor
from oo_model import read_model
from oo_solver import model_from_reader

def first_hz(path):
    run = model_from_reader(read_model(path)).run("modes")
    return float(run["eigenvalues"][0]["rows"]["cycles"][0])

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        print(list(pool.map(first_hz, ["wing-a.oos", "wing-b.oos"])))
```

Each worker loads its own Model and Solver memories. `evaluateMany` queues points with backpressure; results are
owned typed arrays. Cancellation or timeout terminates the busy worker and starts a replacement. Named hooks in a
worker must be exported by a `hooksURL` module because functions cannot cross the worker boundary. A pool also
supports `mode: "native"` with native optimization declarations and settings. Keep model and worker count within the
application's memory budget.

## Persist and resume an accepted point

`point.save(path, { state })` writes a `.ooh5` with the accepted Model model, a versioned study definition and
optimizer state, structural result tables, eigenvalues, a typed `responseValues` table with the selected response
values and bounds, and named aerodynamic arrays. The model and result datasets are typed and can be loaded
selectively. The application owns the optimizer state and passes a serializable value.
The study document also preserves analysis diagnostics and modal tracking confidence, without duplicating mode shapes.

**JavaScript / WebAssembly**

```js
const archive = oor.openResults("/work/accepted.ooh5");
const accepted = archive.model();
try {
  const { study, state, diagnostics, trackedModes } = solver.restoreStudy(oor, accepted, hooks);
  try { console.log(state.iteration, diagnostics, trackedModes); }
  finally { study.delete(); }
} finally { accepted.delete(); archive.delete(); }
```

**Python / native C++**

Python reopens the accepted model from the same archive. The serialized JavaScript study state and mode-tracking helpers are specific to the WASM wrapper.

```python
from oo_model import read_model
from oo_solver import optimization_from_reader

accepted = read_model("accepted.ooh5")
problem = optimization_from_reader(accepted)  # uses design cards in the model
```

The study definition is stored as the Model document `oos.optimization.study`. Model preserves such documents
in the model part of `.ooh5` and reports their omission when exporting model text. Named result arrays and their
text labels are available through `results.named()` and `results.readNamed(index)`.

## Trim loads and trim-preloaded buckling

Use `StaticSubcaseSelection;1` in buckling subcase 2 to select the equilibrated state of trim subcase 1.
The solver uses its displacements to assemble geometric stiffness and differentiates the preload when computing a
buckling factor or a component of a simple buckling mode. The same design point can expose all of the following responses.

| Native response | Selection | Quantity |
| --- | --- | --- |
| `AeroPressureCoefficient` | Box ID in attribute A | Signed pressure jump coefficient, ΔCp |
| `AeroPressure` | Box ID | Pressure jump, q ΔCp |
| `AeroBoxForce` | Box ID | Signed normal force, q ΔCp times box area |
| `AeroBoxMoment` | Box ID | Box pitching moment from that force and the chordwise arm between its force point and centre |
| `HingeMoment` | Control surface ID | Sum of the selected box moments about the hinge axis |
| `BucklingFactor` | Mode number in attribute A | Load factor of the selected buckling subcase |
| `BucklingDisplacement` | Node targets, component digits in attribute A, mode number | Signed component of the recovered mode under the eigensolver's normalisation |

In model design cards, write `DesignResp1;6;PHI;Displacement;Buckling;;3;1;4` to select component 3 of mode 1
at node 4. This spelling preserves node references when Model exports its design tables. The native declaration
accepts `{ id: 6, type: "BucklingDisplacement", attributeA: 3, mode: 1, targets: [4] }`. Individual shape
derivatives require a simple root and a locally unique normalisation. The mode is not a physical displacement amplitude.

**JavaScript / WebAssembly**

```js
const problem = solver.optimizationFromReader(modelApi, deck, {}, {
  analyses: { aeroelastic: [1], buckling: [2] }
});
try {
  const point = problem.evaluate([0.1]);
  try {
    const rates = point.gradients().responses;
    console.log(rates.find(row => row.id === 4).values); // hinge moment
    console.log(rates.find(row => row.id === 5).values); // load factor
  } finally { point.delete(); }
} finally { problem.delete(); }
```

**Python / native C++**

```python
from oo_solver import prepare_optimization

problem = prepare_optimization(deck, analysis={"aeroelastic": [1], "buckling": [2]})
point = problem.evaluate([0.1])
rates = point.gradients()["responses"]
print(rates[4], rates[5])  # hinge moment and buckling factor
```

The [wing model](../wasm/test/fixtures/trim-preloaded-wing.oos) used by both examples has one thickness variable.
To design dynamic pressure as well, append the two records below. They add a variable in the model's pressure
units and bind it to trim set 5. The native relation uses `targetKind: "trim"`, `targetCard: "AeroTrim"`,
`targetId: 5` and `field: "dynamicPressure"`.

```text
DesignVar;2;Q;10.;5.;15.
DesignElemRelation1;2;AeroTrim;5;dynamicPressure;;;;2;1.
```

At thickness 0.1 and dynamic pressure 10, the fixture gives a positive factor of 1084.08554.
The derivatives of that factor with respect to thickness and dynamic pressure are approximately 14641.8270
and −116.270805. The corresponding hinge moment is −0.91111645, with derivatives 0.84832933 and −0.09490853.
These values are regression checks for this mesh.
[Independent reanalysis tests](../wasm/test/validation/trim-buckling-sensitivities.test.mjs)

## Gradients with respect to nodal coordinates

At a retained design point, request a scalar response and a list of structural node IDs. The solver returns one
row per ID in the supplied order. Its three entries differentiate that response with respect to basic-frame X, Y
and Z with the other nodes and design fields fixed. Omit the list to request every structural node. The optional
zero-based item index selects one component when a response has multiple items. No shape variables or CAD
directions need to be declared. A deck containing only response declarations can evaluate an empty design vector.
The mesh connectivity and aerodynamic boxes stay fixed.

**JavaScript / WebAssembly**

```js
const g = point.nodalCoordinateGradient(4, [6, 4]); // hinge moment on the reference wing
const [dx, dy, dz] = g.values.subarray(0, 3);       // derivatives at node 6
console.log(g.nodeIds, dx, dy, dz, g.report.analysisPerturbations);
```

**Python / native C++**

```python
g = point.nodal_coordinate_gradient(4, [6, 4])
dx, dy, dz = g["values"][0]  # node 6
print(g["node_ids"], dx, dy, dz, g["report"]["analysis_perturbations"])
```

**C++**

```cpp
const std::array<std::int32_t, 2> nodes{6, 4};
const auto g = point.nodalCoordinateGradient(4, nodes);
const double dMomentDx6 = g.values[0];
```

For a known physical displacement per unit change of a CAD parameter, contract these rows with those displacement
vectors and sum over the nodes. A remesh with different node correspondence requires a transfer from the caller.
Primary physical response items are supported. Fractional mass needs the original reference mass and a repeated
eigenvalue can require a directional derivative. Use the native design-variable API for response equations.

### RBE3 wing benchmark

The [six-node wing](../wasm/test/fixtures/trim-rbe3-wing.oos) has a shell, a compressive strut, four aerodynamic
boxes and a spline attachment at node 6. A four-leg RBE3 connects the attachment to the shell. Trim subcase 1
preloads buckling subcase 2. At thickness 0.1 and dynamic pressure 10 the two response derivatives are:

| Variable | Hinge moment derivative | First buckling factor derivative | Extra coupled analyses before → after | Gradient time before → after (ms) |
| --- | ---: | ---: | ---: | ---: |
| Thickness | 0.72621246 | 14453.7648 | 0 → 0 | 2.39 → 1.61 |
| Dynamic pressure | −0.09393312 | −117.64074 | 0 → 0 | 1.85 → 1.81 |
| Attachment 6, X | −0.05032213 | −35.537126 | 2 → 0 | 14.80 → 2.60 |
| Shell node 4, Y | −0.02791353 | 415.59990 | 0 after | 4.23 after |

The two responses together took 14.8 ms before and 2.6 ms after for attachment X on a local single-threaded
WASM64 run (median of five gradient calls after one warm-up, primal evaluation excluded). The new route used two
structural right-hand-side solves and zero new structural factorizations. For shell node 4, the element geometry
also changed and the report counted one local shape perturbation. Timings depend on the mesh and runtime.
The [regression](../wasm/test/validation/trim-rbe3-transfer.test.mjs) checks both responses against independent
positive and negative evaluations. It also checks an XYZ contraction on a different thickness using newly read meshes.

The first-order transfer differentiates surface and beam splines and both directions of each affected RBE3.
It reuses the spline, structural and pressure factors on restrained, full-path trim cases with fixed aerodynamic
geometry and purely aerodynamic loading, for local aerodynamic loads, trim variables and trim-preloaded factors.
An isolated RBE3 attachment can move while the mixed shell geometry stays fixed. Free-body support motion,
stability coefficient shape derivatives, moving mixed element geometry with formulation constraints and other
unsupported coupled configurations use the complete-analysis route. The report separates local geometry
perturbations and additional coupled analyses. At a moving interpolation kink a two-sided derivative can be undefined.

## Verification

The [study tests](../wasm/test/cases/studies.mjs) cover all four solutions, mixed load/property derivatives, modal
fields, trim outputs, coupled elastic constants and archive restoration. The [worker tests](../wasm/test/worker-pool.test.mjs)
exercise parallel points and termination of a blocked worker. Run the WASM test suite with a Memory64-capable Node.js.
