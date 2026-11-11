# Optimization controlled by an external application

An application owns the optimizer, trial points, acceptance rule and convergence test. Reader owns the editable
model; Solver owns analysis snapshots and evaluates SOL 101, 103, 105 and 144. There are two entry points:

- `solver.optimizationFromReader` uses the native design model for declared sizing, shape and response types. Its
  analytic and adjoint derivatives are generally the fastest choice where supported.
- `solver.createStudy` edits a copy of a Reader deck at each point. It can change card fields, case control and the
  solution, then select any numeric result or named Reader query. It obtains derivatives by full analysis differences.
  Use this for model changes or output fields outside the native design model.

The two routes share Reader and Solver packages. Neither route chooses a design point for the application.

## Inspect support before building an optimizer

```js
const catalogue = solver.capabilities();
// catalogue.analyses lists 101, 103, 105 and 144, with derivative restrictions.
const model = solver.modelFromReader(oor, deck);
try {
  const available = solver.capabilities(model); // includes numeric property/material/element columns
  const prepared = solver.checkOptimization(model, declarations, settings);
  if (!prepared.canPrepare) throw new Error(prepared.error ?? JSON.stringify(prepared.analyses));
} finally { model.delete(); }
const readerSupport = oor.designCapabilities(deck); // editable cards and Reader design diagnostics
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
| 105 buckling | Load factor gradients including the static preload; analytic Hessian products for closed mechanical-preload sizing formulations | Load factors and result fields, full analysis differences |
| 144 static aeroelastic | Native sizing gradients and analytic Hessian products for closed trim/state/coefficient formulations; shape columns from complete trim differences | Trim and structural outputs, including driven load changes, full analysis differences |

The [native design module](../src/design/README.md) details the exact response formulations and restrictions.
The [modal derivative guide](modal-derivatives.md) describes field selection, normalisation and the bordered systems
used by the C++ first- and second-order calculations.
The [coupled derivative guide](coupled-derivatives.md) describes analytic buckling and trim products and their limits.
When an explicit condensation plan places a driven unit inside a condensed part, native SOL 144 gradients run that
trim on the full structural path. This can increase memory use. Applied trim loads also require the full path.
`derivatives: "numerical"` enables full analysis gradients for a native design point when its native linearization is
unavailable. The returned report identifies `finiteDifference` and the number of additional evaluations. Check
derivatives against representative points, especially around contact changes, buckling mode crossings and trim limits.

## Define a study with arbitrary Reader edits

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

Card targets accept a persistent `key`, an `id`, a named field or a one-based raw slot, and a part descriptor. Case
targets set a command in a subcase or section; solution targets set the solution number. Parameters can combine
variables linearly or call a named synchronous hook. Variable kinds include `continuous`, `integer` and `choice`;
finite differences require continuous variables. Reader's `deck.copy()` gives each trial an independent copy-on-write
model. The source deck stays untouched. Call `study.rebuild(newDeck, ...)` after accepting external edits.

Responses select nodal, element, grid point force, eigenvalue, pressure, box force/moment, hinge moment, trim variable
or stability derivative fields. They may also use a Reader named query, including queries over the generated result
tables, or a named synchronous callback. Row selectors,
components, subcases, modes and reductions control the scalar/vector output; every response has a stable ID. See
[`StudyResponse`](../wasm/js/study.d.ts) for the complete contract. Hooks are registered separately from the
serializable definition:

```js
const hooks = { responses: {
  stressCost: ({ runs }) => Math.max(...runs[0].elementTables
    .filter(table => table.kind === "stress").flatMap(table => Array.from(table.values))),
} };
// A response with { source: "callback", callback: "stressCost", id: 20 } calls this hook.
```

For gradients, `point.gradients()` returns a row-major response/constraint/objective Jacobian with stable `keys`.
`jvp`, `vjp`, `hessianVector` and `hessian` compute full analysis products; the Hessian is a nested numerical
difference and can be costly. Steps respect variable bounds and use second-order one-sided differences at a bound.
Use `onBlock` for a large Jacobian and `maximumOutputBytes` to set a memory limit. A changed output layout fails
instead of silently matching unrelated rows. Tracked modal responses expose MAC and uncertainty; an uncertain or
repeated root refuses a smooth Jacobian by default. A forward directional `jvp` remains available. The application
should inspect the mode identity and treat nonsmooth points explicitly.

An auxiliary SOL 101 displacement field can generate a compact batched native shape basis once:

```js
const basis = solver.shapeBasisFromReader(oor, auxiliaryDeck,
  { variableId: 7, cases: [{ subcase: 1, coefficient: 1 }], gridIds: [2, 3] });
const problem = solver.optimizationFromReader(oor, deck, {
  variables: [{ id: 7, initialValue: 0, lower: -1, upper: 1 }],
  shapeBatches: [basis], responses: designResponses,
});
```

The basis converts displacement components into the basic frame. The auxiliary solve and Reader geometry lookup run
once; later native evaluations consume the typed grid/direction arrays.

## Parallel evaluations and interruption

`study.evaluateMany(points)` yields between sequential points and accepts an `AbortSignal`. A synchronous WASM
solve cannot be interrupted on its own thread. Use `createOptimizationWorkerPool` when evaluations must run in
parallel or a hung solve must be stopped:

```js
const pool = await createOptimizationWorkerPool({
  readerURL: new URL("../oor-reader/index.js", import.meta.url),
  solverURL: new URL("../oos-solver/index.js", import.meta.url),
  model: modelBytes, mode: "study", definition, workers: 2,
});
try {
  const trial = await pool.evaluate(x, { gradients: true, timeoutMs: 30000, signal });
  console.log(trial.values.objective, trial.gradients.values);
} finally { pool.close(); }
```

Each worker loads its own Reader and Solver memories. `evaluateMany` queues points with backpressure; results are
owned typed arrays. Cancellation or timeout terminates the busy worker and starts a replacement. Named hooks in a
worker must be exported by a `hooksURL` module because functions cannot cross the worker boundary. A pool also
supports `mode: "native"` with native optimization declarations and settings. Keep model and worker count within the
application's memory budget.

## Persist and resume an accepted point

`point.save(path, { state })` writes a `.ooh5` with the accepted Reader model, a versioned study definition and
optimizer state, structural result tables, eigenvalues, a typed `responseValues` table with the selected response
values and bounds, and named aerodynamic arrays. The model and result datasets are typed and can be loaded
selectively. The application owns the optimizer state and passes a serializable value.
The study document also preserves analysis diagnostics and modal tracking confidence, without duplicating mode shapes.

```js
const archive = oor.openResults("/work/accepted.ooh5");
const accepted = archive.model();
try {
  const { study, state, diagnostics, trackedModes } = solver.restoreStudy(oor, accepted, hooks);
  try { console.log(state.iteration, diagnostics, trackedModes); }
  finally { study.delete(); }
} finally { accepted.delete(); archive.delete(); }
```

The study definition is stored as the Reader document `oos.optimization.study`. Reader preserves such documents
in the model part of `.ooh5` and reports their omission when exporting model text. Named result arrays and their
text labels are available through `results.named()` and `results.readNamed(index)`.

## Verification

The [study tests](../wasm/test/cases/studies.mjs) cover all four solutions, mixed load/property derivatives, modal
fields, trim outputs, coupled elastic constants and archive restoration. The [worker tests](../wasm/test/worker-pool.test.mjs)
exercise parallel points and termination of a blocked worker. Run the WASM test suite with a Memory64-capable Node.js.
