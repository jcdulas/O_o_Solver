# Optimization controlled by an external application

The examples assume a loaded model named `deck`. In JavaScript, `oor` and `solver` are the loaded WASM modules. In Python, `deck` comes from `oo_model.read_model`. In C++, `deck` is a `std::shared_ptr<oor::api::Deck>`. Include `<oos/api/Capabilities.hpp>` for the catalogue, and `<oor/formats/oos.hpp>` for the file-based parallel example. See [Getting started](getting-started.md) for package setup.

An application owns the optimizer, trial points, acceptance rule and convergence test. Model owns the editable
model; Solver owns analysis snapshots and evaluates SOL 101, 103, 105 and 144. There are two entry points:

- `solver.optimizationFromReader` uses the native design model for declared sizing, shape and response types. Its
  analytic and adjoint derivatives are generally the fastest choice where supported.
- `solver.createStudy` edits a copy of a Model deck at each point. It can change card fields, case control and the
  solution, then select any numeric result or named Model query. It obtains derivatives by full analysis differences.
  Use this for model changes or output fields outside the native design model.

The two routes share Model and Solver packages. Neither route chooses a design point for the application.

After a coordinate edit, build a new snapshot of the edited deck. Retained snapshots and evaluations keep their original geometry and gradients. The transfer automatically uses implicit identity rotations when all nodal displacements are in the basic frame. With local displacement frames it transfers the rotations for the edited geometry. The Model guide measures the first snapshot and successive rebuilds on a nodal mesh and a shell patch. [Snapshot benchmarks](https://jcdulas.github.io/O_o_Model/docs/model-clients.html#rebuilding-after-coordinate-edits)

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

**Python**

Python exposes the capability catalogue on the prepared problem. The JavaScript preflight report additionally checks declarations before problem construction.

```python
from oo_solver import optimization_from_reader

deck.validate_design()
problem = optimization_from_reader(deck)
print(problem.variables, problem.capabilities["analyses"])
```

**C++20**

```cpp
const auto snapshot = oos::api::modelFromDeck(deck);
const auto catalogue = oos::api::optimizationCapabilities(snapshot.get());
for (const auto& entry : catalogue.analyses)
    std::cout << entry.solution << " " << entry.analysis << '\n';
const auto support = oos::api::checkSupport(*snapshot, oos::api::Analysis::Statics);
for (const auto& refusal : support.refusals)
    std::cerr << refusal.message << '\n';
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

**Python**

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

**C++20**

```cpp
auto trial = deck->copy();
oor::api::setCardValue(*trial, "RodProp", 1, "area", "2.0");
oor::api::setCardValue(*trial, "PointForceLoad", 2, "magnitude", "10.0");
const auto model = oos::api::modelFromDeck(trial);
const auto run = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {.subcases = {1}});
for (const auto& table : run.tables.nodalTables)
    if (table.kind == "displacements")
        for (std::size_t row = 0; row < table.rowCount(); ++row)
            if (table.ids[row] == 2)
                std::cout << table.values[6 * row] << '\n';
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

**Python**

```python
stress_cost = max(t["values"].max() for t in run["element_tables"]
                  if t["kind"] == "stress" and t["values"].size)
```

**C++20**

```cpp
std::optional<double> stressCost;
for (const auto& table : run.tables.elementTables) {
    if (table.kind != "stress") continue;
    for (const double value : table.values.span())
        if (!stressCost || value > *stressCost) stressCost = value;
}
if (!stressCost) throw std::runtime_error("request element stress output");
std::cout << *stressCost;
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

**Python**

The native examples assume nodes 2 and 3 use basic displacement coordinates (CD = 0) and the auxiliary subcase requests displacement output. For other output frames, transform each translation to the basic frame before creating the shape relation. The target deck already contains the design responses.

```python
from oo_solver import model_from_reader, prepare_optimization

aux = model_from_reader(auxiliary_deck).run("statics", subcases=[1])
u = next(t for t in aux["nodal_tables"] if t["kind"] == "displacements")
trial = deck.copy()
trial.add_card("DesignVar", [7, "SHAPE", 0., -1., 1.])
for node in (2, 3):
    row = u["ids"].tolist().index(node)
    trial.add_card("DesignNodeMove", [7, node, 0, 1., *u["values"][row, :3].tolist()])
problem = prepare_optimization(trial)
```

**C++20**

```cpp
// integer() and real() construct typed fields as in Getting started.
auto trial = deck->copy();
const auto auxiliary = oos::api::modelFromDeck(auxiliaryDeck);
const auto aux = oos::api::runAnalysis(*auxiliary, oos::api::Analysis::Statics, {.subcases = {1}});
std::map<int, std::array<double, 3>> translations;
for (const auto& table : aux.tables.nodalTables)
    if (table.kind == "displacements")
        for (std::size_t i = 0; i < table.rowCount(); ++i)
            translations[table.ids[i]] = {table.values[6*i], table.values[6*i+1], table.values[6*i+2]};
const Field label{Field::Kind::Text, 0, 0., "SHAPE"};
std::vector<oor::api::CardInput> cards{
    {"DesignVar", {integer(7), label, real(0), real(-1), real(1)}}};
for (int node : {2, 3}) {
    const auto [x, y, z] = translations.at(node);
    cards.push_back({"DesignNodeMove", {integer(7), integer(node), integer(0),
                                        real(1), real(x), real(y), real(z)}});
}
oor::api::insertCards(*trial, cards);
const auto problem = oos::api::prepareOptimization(trial);
```

The JavaScript helper converts displacement components into the basic frame. The auxiliary solve and Model geometry lookup run
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

**Python**

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

**C++20**

```cpp
const auto firstHz = [](const char* path) {
    const auto model = oos::api::modelFromDeck(oor::formats::oos::read(path));
    const auto modes = oos::api::runAnalysis(*model, oos::api::Analysis::Modes, {});
    return modes.eigenvalues.at(0).rows.at(0).cycles;
};
auto a = std::async(std::launch::async, firstHz, "wing-a.oos");
auto b = std::async(std::launch::async, firstHz, "wing-b.oos");
std::cout << a.get() << " " << b.get();
```

Each worker loads its own Model and Solver memories. `evaluateMany` takes any number of points with backpressure: a
point is sent when a worker is free for it, so the `maximumQueuedPoints` limit of `evaluate` does not apply. Its first
failure rejects the call and no further point is started; the points already running end on their own unless the
`signal` aborts them. Results are owned typed arrays. Cancellation or timeout terminates the busy worker and starts a
replacement, as does a WebAssembly trap in a worker, whose module is not used again. Named hooks in a
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

**Python**

Python reopens the accepted model from the same archive. The serialized JavaScript study state and mode-tracking helpers are specific to the WASM wrapper.

```python
from oo_model import read_model
from oo_solver import optimization_from_reader

accepted = read_model("accepted.ooh5")
problem = optimization_from_reader(accepted)  # uses design cards in the model
```

**C++20**

```cpp
// Link the OOH5 adapter (oor::ooh5) and include <oor/formats/ooh5.hpp>.
const auto accepted = oor::formats::ooh5::read("accepted.ooh5");
const auto problem = oos::api::prepareOptimization(accepted); // reads the archived design cards
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

**Python**

```python
from oo_solver import prepare_optimization

problem = prepare_optimization(deck, analysis={"aeroelastic": [1], "buckling": [2]})
point = problem.evaluate([0.1])
rates = point.gradients()["responses"]
print(rates[4], rates[5])  # hinge moment and buckling factor
```

**C++20**

```cpp
oos::api::OptimizationSettings settings;
settings.analyses.statics.reset();
settings.analyses.aeroelastic = std::vector<std::int32_t>{1};
settings.analyses.buckling = std::vector<std::int32_t>{2};
const auto problem = oos::api::prepareOptimization(deck, settings);
const auto point = problem->evaluate({0.1});
const auto gradients = point->gradients({.responseIds = {4, 5}});
for (const auto& row : gradients.responses)
    std::cout << row.id << " " << row.values.at(0) << '\n';
```

The [wing model](../wasm/test/fixtures/trim-preloaded-wing.oos) used by all three examples has one thickness variable.
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

**Python**

```python
g = point.nodal_coordinate_gradient(4, [6, 4])
dx, dy, dz = g["values"][0]  # node 6
print(g["node_ids"], dx, dy, dz, g["report"]["analysis_perturbations"])
```

**C++**

```cpp
const std::array<std::int32_t, 2> nodes{6, 4};
const auto g = point->nodalCoordinateGradient(4, nodes);
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
