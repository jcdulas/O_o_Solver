# Optimization driven by JavaScript

For model edits and result fields beyond native design declarations, see
[external optimization](external-optimization.md). It covers all four solution sequences, generic Reader card and
case-control parameters, numerical derivatives, parallel workers and accepted-point archives.

The application owns the optimization algorithm. WebAssembly evaluates design points, responses and derivatives;
JavaScript chooses the next point, convergence rules, multipliers and line search. No optimizer is started implicitly.

Reader and Solver have independent memories. Reader exports owned table packages, Solver keeps an immutable model
snapshot, and each evaluation keeps its own analysis factors and memory budget. Evaluating a trial point does not edit
the Reader model. Apply the accepted point explicitly when the iteration ends.

## Create and evaluate a problem

```javascript
const problem = solver.optimizationFromReader(oor, deck, {
  responses: [
    { id: 101, type: "Volume", label: "volume" },
    { id: 102, type: "Displacement", attributeA: 1, targets: [2], label: "tip" },
  ],
  constraints: [{ responseId: 102, upper: 0.02 }],
  objective: { responseId: 101, sense: "min" },
}, {
  memoryBytes: 512 * 1024 * 1024,
  maximumOutputBytes: 64 * 1024 * 1024,
});
const info = problem.metadata();
const x = info.initialValues;
const point = problem.evaluate(x);
try {
  const values = point.values();
  const derivatives = point.gradients();
  // Supply these to the JavaScript optimization algorithm.
  console.log(values.objective, values.constraints, derivatives.objective);
} finally {
  point.delete();
  problem.delete();
}
```

The model must contain its design variables and relations. Declarations supplied by JavaScript are appended to those
of the model; IDs must remain unique. Alternatively, `solver.createOptimization(model, declarations, settings)` takes
an existing solver model and JavaScript declarations, without importing any deck design tables. It accepts variables,
linear or equation relations, links, constants, shape directions, responses, constraints, an objective and callbacks.
Parsed equations can be supplied through the two equation table packages. The [API reference](api.rst) gives the full
declaration types.

Reader refuses design declarations that its export cannot represent, rather than silently dropping them.
Model and declaration changes require a new problem. A problem can evaluate many points, and several evaluations may
coexist. An evaluation retains its problem even if the JavaScript problem handle has been deleted.

## Variable, row and sign conventions

- `x` contains **independent variables only**. Entry `j` corresponds to
  `info.variables[info.independentIndices[j]]`, including its ID, bounds and move limit. Linked variables are derived.
- Response items carry `subcase`, `entity`, `component` and `layer`. Flattened derivative rows follow the order of
  `point.values().responses`, then each response's item order. Select by IDs and item labels, not by assumed positions.
- A response gradient is `itemCount × variableCount`, row-major. `gradients().constraints` follows the constraint
  values in exactly the same order. A masked-out gradient row contains NaN.
- A normalised constraint is feasible when `normalised <= 0`. Its `derivativeScale` is the derivative with respect
  to the raw response: negative for a lower bound, positive for an upper bound. This includes the near-zero bound scale.
- `objective` and its gradient contain the **raw response value**, including a maximisation objective.
  `lagrangianWeights()` changes the objective sign for maximisation, giving a minimisation convention.
- Variable bounds and field bounds are available in metadata. Linked variable values and sparse field derivatives
  are available through `problem.designPoint(x)`. The JavaScript algorithm must enforce these bounds; they are not
  silently converted into response constraints or clipped during a trial evaluation.

All returned numerical arrays own their data. They remain valid after an evaluation is deleted or the module's
memory grows. Sizes and sparse indices are JavaScript numbers; matrix indices start at zero.

## Derivative interfaces

| Interface on an evaluation | Result and intended use |
| --- | --- |
| `gradients({method, responseIds, itemMasks})` | Response Jacobians, objective gradient and normalised constraint gradients; `automatic`, `direct`, `adjoint` or `chained` method |
| `streamGradients(onBlock, block)` | Every Jacobian row, without retaining the full Jacobian; each block gives row and column offsets |
| `jvp(directions, {count, block, onBlock, responseIds, subcases})` | `J V`; directions and output are row-major with one column per direction |
| `vjp(weights, {block, onBlock})` | `W J`; dense or compressed sparse response weights, one row per weighted sum |
| `hessianVector(weights, directions, count)` | Hessian-vector products of a scalar weighted response sum; several directions share the linearisation |
| `hessian(weights)` | Dense reduced Hessian of the same sum, row-major, built by applying the second-order engine to the coordinate directions |
| `blocks(weights)` | Sparse stiffness, constraint transforms, pseudo-loads, item displacement weights and explicit design derivatives; optional second-order blocks |
| `analysisResults()` | Current point result arrays and diagnostics, including modal and buckling eigenvalues; same format as `solver.run` |
| `clearLinearisation()` | Releases the cached derivative linearisation; the analysis remains available |

Collected Jacobians, gradients and dense Hessians are bounded by `maximumOutputBytes` (256 MiB by default).
For large problems prefer products or callbacks. `onBlock` receives an owned copy of a product block and disables
collection of the full product. `streamGradients` also copies each block, so the consumer decides whether to keep it.
The analysis and retained derivative factors share `memoryBytes`, separately from copies retained by JavaScript.

```javascript
const weights = { rows: 1, responses: { 102: new Float64Array([1]) } };
const jtW = point.vjp(weights); // one weighted gradient
const jV = point.jvp(direction); // one direction, J*direction
const hv = point.hessianVector(weights, direction);
const denseHessian = point.hessian(weights);

// For a constrained optimizer: objectiveFactor*f + sum lambda[c]*g[c].
// g is normalised; f is sign-adjusted when the objective is maximised.
const lagrangian = point.lagrangianWeights(objectiveFactor, constraintMultipliers);
const lagrangianProduct = point.hessianVector(lagrangian, direction);
```

A dense weight array has `rows × response.itemCount` entries. The sparse alternative for one response is
`{rowStart, items, values}`: `rowStart` has `rows + 1` entries, starts at zero and ends at the entry count;
item indices within a row are strictly increasing. Missing responses have zero weight. Second-order calls require
exactly one row of weights. A response Hessian is obtained by weighting its desired item by one and all other items
by zero. Weighted Hessians also include mixed terms introduced by the supported equation response chain rule.

## JavaScript functions and their derivatives

Register a named object in `declarations.callables` and refer to it from an `equationResponses` declaration. Callbacks
are synchronous and batched on the calling thread. An asynchronous computation must finish before an evaluation
starts; a Promise is not a numerical batch result. A callback must not re-enter or delete its active evaluation.

`evaluate({arguments, items})` returns `{outputs, values}`. Each argument describes its name, component count and
`items × components` values. `evaluateWithPartials` additionally returns `partials[a]` with shape
`items × outputs × components(a)`, row-major. Missing derivatives are refused when requested.

For second order, `hessianVector` receives those arguments, `items`, `outputs`, output `weights` and argument
`directions`. It returns, per argument, the weighted Hessian-vector product with shape `items × components(a)`.
The solver chains these products through the structural analysis. The executable
[optimization tests](../wasm/test/cases/optimization.mjs) include `u -> u²` with its first and second derivatives.

For a relation of form `supplied`, `suppliedAt(x)` returns its `relationIndex`, value and gradient with respect to
the relation's arguments, in declaration order. It is also called at the initial design and perturbed states used
by second-order calculations. It must be deterministic and provide consistent values and derivatives at every call.

## Analyses, accuracy and current limits

The [threaded variant](threading.md) accepts `threads` in the problem settings and shares its workers across
retained evaluations. Numerical work uses the pool; JS callbacks stay on the calling thread.

The default is statics. `settings.analyses` can select `statics`, `modes`, `buckling` and `aeroelastic` subcases;
an absent kind is disabled, an empty array selects all of that kind. Responses, eigenvalue sensitivities and supported
model features follow the native [design module](../src/design/README.md). The WebAssembly facade does not add a
missing physical formulation. For eigenvalue responses with `tracksMode: true`, pass the previous point's
`values().trackedModes` as the second argument of `problem.evaluate(x, modeReferences)`. Returned mode shapes,
modal assurance values and uncertainty flags let JavaScript preserve and inspect the tracked mode between cycles.
`analysisResults()` includes tracking warnings. With `deckOutputs: true`, it also contains the outputs requested by
the deck, beyond those required by responses.

Selected bulk non-structural mass sets are supported in static optimization. Sizing derivatives keep their selected
mass share fixed; shape derivatives include the change of its length or area in the differentiated inertial load.
The selected set must use the per-length or per-area forms supported by [static runs](static-inputs.md).
An imposed axial deformation (`DeformationLoadSelection`) contributes to native static gradients through its
equivalent element load and the element recovery.

Gradients use closed forms where element families provide them and numerical element derivatives elsewhere.
The C++ element layer supplies exact sizing rates for scalar springs and masses, bushings, fasteners and the
gap spring rates within their initial branch, as well as concentrated mass matrices and rigid-body mass/inertia.
Supported mixed field derivatives also feed the static Hessian blocks when several fields reach one element.
`fieldPerturbations`, `hasHessianBlocks` and `perturbedLinearisations` identify which formulas a particular
calculation actually used; a native C++ derivative can still use numerical differences where no closed form exists.
Second-order products for static responses reuse element Hessian blocks where available and otherwise difference the
response linearisation along the direction. Static shapes, responses without supported second-order formulas,
and unsupported coupled responses difference native first-order gradients at nearby design points. Modal eigenvalues,
frequencies and selected fields use bordered tangent and adjoint derivatives when the element matrix and recovery
second derivatives are closed; their report sets `usesModalTangents: true` and `perturbedLinearisations: 0`.
Closed buckling and trim sizing products differentiate the coupled state directly, including the preload or support
motion, and set `usesBucklingTangents` or `usesTrimTangents`. See [coupled derivatives](coupled-derivatives.md) for
the formulas, supported loads and remaining numerical paths.
Products that difference native gradients respect variable bounds and reuse the current mode shapes
as tracking references. An ambiguous mode match stops the product. Reports expose the method through
`hessianBlocksReason`. These are numerical engineering derivatives, not a guarantee of exact arithmetic or
differentiability at a response's branch changes.

The native optimization API combines structural gradients with complete-analysis differences for shape columns
when trim or static inertia relief moves splines or rigid-body support modes.
`report.analysisPerturbations` counts the added analyses.
`jvp`, `vjp` and `streamGradients` use these completed columns too. Streaming keeps structural columns in blocks and
releases each completed shape column after emitting its block, so it can exceed the dense Jacobian output limit.
Second-order products require a supported first-order response. The [modal derivative guide](modal-derivatives.md)
describes mode-selected fields and their normalisation. Trim configurations without native gradients remain outside
that set. Sparse second-order
block export is narrower:
it requires the native closed-form blocks and refuses the condensed path, inertia relief and other cases documented
in `VectorJacobianProducts`. Unsupported operations throw a JavaScript error with the native reason; no zero Hessian
is substituted. Check derivatives at representative points before relying on a response in an optimization.

## Save the accepted design

For variables defined in the Reader model, pass their explicit IDs in solver order:

```javascript
const ids = Array.from(info.independentIndices, index => info.variables[index].id);
oor.applyDesignPoint(deck, ids, acceptedX);
const finalModelText = deck.toModelText();
```

Reader applies links, field changes and supported grid moves atomically. It rejects variable bound violations and,
by default, field bound violations. `setInitialValues` defaults to true, so the saved model starts at the accepted
point. A second application of that point does not move the shape again. Use `limits: "clip"` or `"ignore"` only when
the application explicitly needs that field-bound policy. Unsupported declarations refuse the operation.

Variables declared only in JavaScript do not create Reader cards. Their final fields must be written with Reader's
card-editing API, or their declarations must first be added to the model. Rebuild the solver problem after any such edit.

## Verification

Build both modules and run `node --test --test-concurrency=1 wasm/test/*.test.mjs` with a Memory64-capable runtime. The optimization tests
check an original axial bar against `u = P L / (E A)`, direct and adjoint gradients, mixed Hessian entries, dense and
sparse products, streamed Jacobians, a nonlinear JavaScript callback, modal gradients, memory limits and atomic Reader
updates. A JavaScript iteration also reaches the active displacement bound and saves the accepted design.
