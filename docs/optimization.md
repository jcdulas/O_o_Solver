# External optimizer API for JavaScript, Python and C++

The examples assume a loaded model named `deck`. In JavaScript, `oor` and `solver` are the loaded WASM modules. In Python, `deck` comes from `oo_model.read_model`. In C++, `deck` is a `std::shared_ptr<oor::api::Deck>`. See [Getting started](getting-started.md) for package setup.

For model edits and result fields beyond native design declarations, see
[external optimization](external-optimization.md). It covers all four solution sequences, generic Model card and
case-control parameters, numerical derivatives, parallel workers and accepted-point archives.

The application chooses the next point, convergence rules and line search. JavaScript calls the WebAssembly solver. Python and C++ call the native solver to evaluate the same model-defined design variables, responses and first derivatives.

The [derivative coverage guide](derivatives.md) summarizes first- and second-order paths for all four solutions,
their numerical fallbacks and the report fields that identify the calculation actually used.

In WebAssembly, Model and Solver have independent memories. Model exports owned table packages, Solver keeps an immutable model
snapshot, and each evaluation keeps its own analysis factors and memory budget. Evaluating a trial point does not edit
the model. Apply the accepted point explicitly when the iteration ends.

## Callbacks for an external optimizer

Prepare the problem once per model and retain the evaluation until the optimizer changes its candidate.
Use the same point for value and derivative callbacks. A regenerated mesh starts a new problem.
For a derivative-free optimizer, the compact values contain the raw objective, normalized constraints
(feasible at zero or below), a partial-response flag and a mode-tracking uncertainty flag.
A failed solve raises an exception. A partial point can have fewer constraint rows.
The initial variable magnitude (one for an initial zero) is available as a suggested scale.
The API takes physical coordinates. Response derivatives use those coordinates. Constraint derivatives include their normalization. The objective sense and selected response are available in JavaScript problem metadata and in Python `problem.objective_definition`.

**JavaScript / WebAssembly**

```js
const problem = solver.prepareOptimization(oor, deck);
try {
  const point = problem.evaluate();
  try {
    const { objective, constraints, partial, uncertainMode } = point.optimizerValues();
    const csr = point.constraintJacobian();
    try {
      const pattern = csr.structure();       // export once
      const derivatives = csr.values(point); // same order as pattern.columnIndices
    } finally { csr.delete(); }
  } finally { point.delete(); }
} finally { problem.delete(); }
```

**Python / NumPy**

```python
from oo_solver import prepare_optimization

problem = prepare_optimization(deck)
point = problem.evaluate(problem.initial_values)
values = point.optimizer_values()
csr = point.constraint_jacobian()
offsets, columns = csr.row_offsets, csr.column_indices
jacobian_data = csr.values(point)
next_point = problem.evaluate(next_x)
csr.values(next_point, out=jacobian_data)
```

**C++20**

```cpp
#include <oos/api/ConstraintJacobian.hpp>
#include <oos/api/ModelFromDeck.hpp>

auto problem = oos::api::prepareOptimization(deck);
auto point = problem->evaluate(problem->problem().initialValues());
auto values = point->optimizerValues();
oos::api::ConstraintJacobian csr(*point);
std::vector<double> data(csr.columnIndices().size());
csr.values(*point, data);
```

All three return the same zero-based CSR layout for the normalized constraints. Without dependency information
the layout conservatively includes every variable. If structural dependencies are known, supply one sorted list
of variable columns per constraint at construction. A derivative that is zero at a single point can become
nonzero later. A nonzero outside the supplied pattern, a different problem, or a change of constraint identity/order
raises an error. Discard the output buffer of a failed call. Use the layout's `constraints(point)` to check row
identity on a values-only trial. The CSR API streams native response derivatives in blocks. A model requiring full finite differences through external remeshing can use the values-only interface.

**JavaScript / WebAssembly**

```js
// Two independent variables, two simultaneous directions (columns of the identity).
const J = point.jvp(new Float64Array([1, 0, 0, 1]), { count: 2 });
const JT = point.vjp({ rows: 2, responses: { 10: [1, 2] } }); // response 10 has one item
```

**Python / NumPy**

```python
import numpy as np

directions = np.eye(2)
jv = point.jvp(directions)                # (response items, directions)
point.jvp(directions, out=jv)             # reuse a writable C-contiguous float64 array
jt = point.vjp({10: np.array([[1.], [2.]])})  # (weighted sums, variables)
```

**C++20**

```cpp
const std::array directions{1., 0., 0., 1.};
SingleBlockProduct jv, jt;
const auto forward = point->jvp(directions, 2, 2, jv);
const oos::design::ProductWeights weights{.rows = 2, .weights = {{10, {1., 2.}}}};
const auto adjoint = point->vjp(weights, 2, jt);
```

The C++ examples use [single_block_product.hpp](examples/single_block_product.hpp) to copy one row-major product block into a vector. Set the block size to the requested number of directions or weight rows.

A one-dimensional Python argument keeps the existing vector result. In a batched VJP each response's
weights have shape `(sums, items)`. JVP input and output buffers must be disjoint. The C++ equivalents
accept row-major spans and a product sink. Python and JavaScript both enter the native kernel once per batch.
For the capability catalogue use `solver.capabilities(model)`, `problem.capabilities` or
`oos::api::optimizationCapabilities(&snapshot)`. These describe candidate fields and analysis restrictions.
The gradient report records the perturbations used at the evaluated point.

[Run the three client benchmarks](benchmarks/optimizer-api/README.md).

## Create and evaluate a problem

**JavaScript / WebAssembly**

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

**Python**

Declare responses 101 and 102, the objective and the constraint in the input deck.

```python
from oo_solver import optimization_from_reader

problem = optimization_from_reader(deck)
point = problem.evaluate(problem.initial_values)
print(point.objective, point.constraints, point.gradients()["objective"])
```

**C++20**

```cpp
const auto problem = oos::api::prepareOptimization(deck);
const auto point = problem->evaluate(problem->problem().initialValues());
const auto values = point->optimizerValues();
const auto objectiveGradient = point->gradients().objective;
```

The model must contain its design variables and relations. Declarations supplied by JavaScript are appended to those
of the model; IDs must remain unique. Alternatively, `solver.createOptimization(model, declarations, settings)` takes
an existing solver model and JavaScript declarations, without importing any deck design tables. It accepts variables,
linear or equation relations, links, constants, shape directions, responses, constraints, an objective and callbacks.
Parsed equations can be supplied through the two equation table packages. The [API reference](api.rst) gives the full
declaration types.

Model refuses design declarations that its export cannot represent, rather than silently dropping them.
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

**JavaScript / WebAssembly**

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

**Python**

```python
jt_w = point.vjp({102: [1.0]})
j_v = point.jvp(direction)
hv = point.hessian_vector({102: [1.]}, direction)
```

**C++20**

```cpp
SingleBlockProduct jv, jt;
const oos::design::ProductWeights weights{.rows = 1, .weights = {{102, {1.0}}}};
const auto forward = point->jvp(direction, 1, 1, jv);
const auto adjoint = point->vjp(weights, 1, jt);
std::vector<double> hv;
const auto report = point->hessianVector(weights, direction, 1, hv);
const auto denseHessian = point->hessian(weights);
```

JavaScript also constructs response weights from objective and constraint multipliers. Python and C++ accept explicit item weights for Hessian-vector products.

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

Selected bulk non-structural mass sets are supported in static optimization and by the vibration responses on the
full path: the eigenvalue, frequency and mode-selected field derivatives and the analytic vibration Hessian-vector
products form their mass, and the rates of the element masses, with the set the MODES subcase selects. A buckling
subcase that selects a set is refused in an optimization. In statics, sizing derivatives keep their selected
mass share fixed; shape derivatives include the change of its length or area in the differentiated inertial load.
The selected set must use the per-length or per-area forms supported by [static runs](static-inputs.md).
An imposed axial deformation (`DeformationLoadSelection`) contributes to native static gradients through its
equivalent element load and the element recovery.

Gradients use closed forms where element families provide them and numerical element derivatives elsewhere.

For tetrahedra, hexahedra, wedges and pyramids, both formulations differentiate the static
stiffness with respect to isotropic Young's modulus and Poisson's ratio. This includes internal
displacements eliminated by the classic HEXA8 and the mixed solids. Both formulations also
differentiate the mass matrix with respect to material density. An omitted shear modulus is
supported when all affected elements are solids. If Poisson's ratio is instead inferred from a
fixed shear modulus, the material relation uses local perturbations to include that dependency.
Frequencies use the modal element matrices, including the retained interior fields of mixed
solids. Changes in Young's modulus and density can therefore be evaluated without perturbing a
material field. Mixed-solid mass dependence on Poisson's ratio, and the effect of this ratio
on the retained modal interior fields, still use local matrix perturbations. These first-order
rates are shared by the Jacobian and its products.

On a solid cantilever with one shared Young's modulus and a fixed nodal force, the following
local WASM times measure only the tip displacement gradient. Each value is the median of five
freshly evaluated points after two warm-ups. All four cases went from one field perturbation to
zero. [Reproduce the solid benchmark](benchmarks/optimizer-api/solid-material.mjs).

| Mesh | Formulation | Before | After |
| --- | --- | ---: | ---: |
| 8 HEXA8 | Classic | 4.43 ms | 0.99 ms |
| 8 HEXA8 | Mixed | 2,072 ms | 429 ms |
| 48 TETRA4 | Classic | 5.10 ms | 1.42 ms |
| 48 TETRA4 | Mixed | 1,617 ms | 300 ms |

For centred, uncoupled mixed shells, the stiffness derivative with respect to `ShellProp.thickness`
uses the membrane and shear terms (linear in thickness) and the eliminated bending term (cubic).
Blank corner thicknesses and proportional corner thicknesses are supported. An offset, membrane–bending
coupling or absolute corner thicknesses select the existing local derivative. Stress recovery, thermal
loads and mass can require their own derivatives. A displacement gradient under fixed nodal forces
or surface pressure can use the closed stiffness rate alone.

On a distorted cantilever with one shared thickness, the local WASM median time for the displacement
gradient fell from 477 to 111 ms for 32 QUAD4s and from 245 to 55 ms for 64 TRIA3s. The measurement
times only `point.gradients()` on a newly evaluated point, with two warm-ups and seven timed samples.
Both meshes reported zero field perturbations and zero additional factorisations.
[Reproduce the panel benchmark](benchmarks/optimizer-api/shell-thickness.mjs).

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
the formulas, supported loads and remaining numerical paths. [Bar and beam derivatives](beam-derivatives.md)
lists the analytic `Bar2`/`Beam2` section, material, release and recovery rates.
Their applied-load facets also provide thermal/initial-strain and line-load rates. Supported thermal field responses
and line-loaded displacement responses use those rates in static gradients and Hessian blocks. Linear force,
stress and strain items of loaded bars/beams now include their line-load recovery terms as well; shared `Bar2`
axial, shear and torque items select end A of its loaded station table.
Products that difference native gradients respect variable bounds and reuse the current mode shapes
as tracking references. An ambiguous mode match stops the product. Reports expose the method through
`hessianBlocksReason`. These are numerical engineering derivatives, not a guarantee of exact arithmetic or
differentiability at a response's branch changes.

In restrained trim, the first derivative of local aero loads and trim-preloaded buckling factors includes the
analytic spline and RBE3 transfer rates. The optimization API uses complete-analysis differences for other trim
configurations and for moving inertia-relief support modes. The [nodal API and benchmark](external-optimization.md#gradients-with-respect-to-nodal-coordinates)
give the coordinate conventions, examples and measured cost.

For static displacement sensitivities, moving the reference node or a weighted node of a
`RigidInterpolation` (RBE3) differentiates the constraint coefficients directly, including the
optional UM dependent DOFs. Gradients, JVPs and VJPs use the same derivative. Moving only the
reference node of an otherwise unchanged structure requires no finite differences when the loads
are geometry-independent. If the move also deforms a structural element, its own shape derivative
can still use local perturbations. The native derivative reports a diagnostic for a motion that
changes the spider's undetermined rigid motions or separates coincident points.

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

For variables defined in the model, pass their explicit IDs in solver order. For the C++ export below, include `<oor/api/Design.hpp>` and `<oor/formats/oos.hpp>` and link the optional OOS adapter (`oor::oos`).

**JavaScript / WebAssembly**

```javascript
const ids = Array.from(info.independentIndices, index => info.variables[index].id);
oor.applyDesignPoint(deck, ids, acceptedX);
const finalModelText = deck.toModelText();
```

**Python**

```python
ids = [v["id"] for v in problem.variables]
accepted = deck.with_design_point(ids, accepted_x)
accepted.write_model("accepted.oos")
```

**C++20**

```cpp
const auto& model = problem->problem().model();
std::vector<std::int32_t> ids;
for (const auto index : model.independentIndices())
    ids.push_back(model.variables()[index].id);
auto accepted = deck->copy();
const auto update = oor::api::applyDesignPoint(*accepted, ids, acceptedX);
const auto warnings = oor::formats::oos::write(*accepted, "accepted.oos");
```

Model applies links, field changes and supported grid moves atomically. It rejects variable bound violations and,
by default, field bound violations. `setInitialValues` defaults to true, so the saved model starts at the accepted
point. A second application of that point does not move the shape again. Use `limits: "clip"` or `"ignore"` only when
the application explicitly needs that field-bound policy. Unsupported declarations refuse the operation.

Variables declared only in JavaScript do not create Model cards. Their final fields must be written with Model's
card-editing API, or their declarations must first be added to the model. Rebuild the solver problem after any such edit.

## Verification

Build both modules and run `node --test --test-concurrency=1 wasm/test/*.test.mjs` with a Memory64-capable runtime. The optimization tests
check an original axial bar against `u = P L / (E A)`, direct and adjoint gradients, mixed Hessian entries, dense and
sparse products, streamed Jacobians, a nonlinear JavaScript callback, modal gradients, memory limits and atomic Model
updates. A JavaScript iteration also reaches the active displacement bound and saves the accepted design.


The [trim and buckling example](external-optimization.md#trim-loads-and-trim-preloaded-buckling) shows how to request
a signed box pressure, hinge moment, load factor and modal displacement on one design point in JavaScript, Python and C++.
