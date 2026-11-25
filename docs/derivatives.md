# Derivative coverage and reports

The native optimization API evaluates declared design variables and responses through C++. WebAssembly calls
the same implementation. A native derivative can combine analytic element formulas, adjoint solves and numerical
element differences; its report identifies the path used for the requested point and responses.

Start with [the optimization API](optimization.md) to declare a problem. Use
[external studies](external-optimization.md) for arbitrary Model edits or result queries: that route obtains
derivatives by complete analysis differences.

## Coverage by solution

Closed curvature requires linear sizing relations and the first, second and mixed element rates needed by every
weighted response. An available matrix derivative alone does not provide the corresponding recovered-field Hessian.

| Solution | Native first derivatives | Analytic second-order path | Main restrictions |
| --- | --- | --- | --- |
| SOL 101, linear statics | Supported sizing and shape responses, with applied-load and recovery terms | Element Hessian blocks for supported sizing responses | Unsupported recovery columns, shape curvature, nonlinear/derived relations and unavailable load rates retain numerical paths |
| SOL 103, normal modes | Eigenvalues, frequencies and mode-selected displacement/force/stress/strain fields | Bordered tangent and adjoint solves for simple modes | Repeated or unresolved individual modes, shape directions, unsupported element curvature and approximate condensed solutions |
| SOL 105, linear buckling | Load factor and selected simple-mode displacement, including static or trim preload sensitivity | Coupled preload and eigenvector tangents for load factors under mechanical, supported thermal/deformation or trim preloads | Repeated roots, unavailable element curvature, combined structural preloads and unsupported applied-load changes |
| SOL 144, static aeroelasticity | Trim variables, stability coefficients, structural fields, box pressure coefficients, pressure jumps, box forces/moments and hinge moments, with sizing or dynamic-pressure variables, plus restrained spline and RBE3 shape rates for local loads and trim variables | Differentiated trim state and coefficient equations for linear structural sizing | Free-body support geometry, spline curvature, dynamic-pressure curvature, unavailable element/recovery rates and unsupported applied-load changes |

The [modal guide](modal-derivatives.md) covers mode selection, normalization and repeated roots.
The [coupled guide](coupled-derivatives.md) gives the buckling and trim equations and their precise load restrictions.

## Bar and beam sizing

`Bar2` and `Beam2` provide analytic first, second and mixed rates for the supported raw section/material fields:

| Computation | Included effects |
| --- | --- |
| Elastic stiffness | Axial, bending, coupled bending through I12, torsion, shear flexibility and end releases |
| Mass | Density, section properties, NSM, lumped/coupled formulation and neutral-axis/NSM-centre offsets |
| Thermal and imposed-deformation loads | Expansion, reference-temperature precedence, imposed elongation and released-end condensation |
| Equivalent nodal line loads | Partial distributed loads and concentrated forces/moments on the span, with fixed load values and geometry |
| Linear recovered fields | Forces, stresses and strains, including thermal and line-load terms |
| Geometric stiffness partials | Section/material and release-map rates at fixed preload displacements and supplied temperatures |

The [bar and beam guide](beam-derivatives.md) lists exact field names, flattened station indices, result item
conventions and remaining numerical paths. The thermal buckling Hessian uses both the equivalent-load derivatives of the preload and the geometric-stiffness derivatives of the buckling pencil.

Raw sizing fields keep grid positions, orientation, load positions and load values fixed. Library section dimensions,
completed material defaults and nonlinear relations require additional chain rules and retain their documented
fallbacks. Stress extrema and margins do not acquire analytic curvature from the linear component rates alone.

Other element families provide their own subsets of closed rates. See the
[element derivative inventory](../src/elements/README.md#design-derivatives) for springs, masses, rods, shells,
connections and solids. Unsupported operations report their native reason; an unavailable derivative is not
replaced by a zero.

## Read a gradient report

| Member | Meaning |
| --- | --- |
| `fieldPerturbations` | Sizing fields whose element computations required perturbed snapshots |
| `shapePerturbations` | Shape directions evaluated by geometry perturbations |
| `analysisPerturbations` | Additional complete analyses used by the gradient calculation |
| `analyticDerivatives` | Element matrix/load computations supplied in closed form |

Read the counters together with the response and element coverage. Zero field/analysis perturbations describe
those parts of the calculation; they do not add support for a response outside the documented formulas.
Direct and adjoint methods change how the state sensitivities are solved, rather than the meaning of these counters.

## Read a Hessian-product report

| Member | Meaning |
| --- | --- |
| `hasHessianBlocks` | The static product used closed element Hessian blocks |
| `usesModalTangents` | The vibration product used analytic bordered tangents and adjoints |
| `usesBucklingTangents` | The buckling product included analytic preload and mode tangents |
| `usesTrimTangents` | The aeroelastic product used analytic coupled state/coefficient tangents |
| `perturbedLinearisations` | Native response linearizations evaluated at perturbed design points |
| `hessianBlocksReason` | Description of the selected closed path or why it was unavailable |

A closed modal, buckling or trim product can have `hasHessianBlocks: false`: that member identifies the separate
static element-block path. The corresponding tangent flag and `perturbedLinearisations: 0` identify the coupled
analytic path. Factorization and solve counts measure linear algebra work, not design perturbations.

**JavaScript / WebAssembly**

```js
// model contains a Bar2 with BarProp 7 and a tip node 42 in a static subcase.
const problem = solver.createOptimization(model, {
  variables: [{ id: 1, initialValue: 0.02, lower: 0.01, upper: 0.03 }],
  relations: [{ id: 1, targetKind: "property", targetCard: "BarProp", targetId: 7,
    field: "area", terms: [{ variableId: 1, coefficient: 1 }] }],
  responses: [{ id: 10, type: "Displacement", attributeA: 3, targets: [42] }],
});
try {
  const point = problem.evaluate();
  try {
    const gradient = point.gradients();
    const product = point.hessianVector({ responses: { 10: [1] } }, [1]);
    console.log(gradient.report.fieldPerturbations, gradient.report.analysisPerturbations);
    console.log(product.report.hasHessianBlocks, product.report.perturbedLinearisations);
    console.log(product.report.hessianBlocksReason);
  } finally { point.delete(); }
} finally { problem.delete(); }
```

**Python**

Define the area relation and tip response 10 in the deck. Python and C++ use these deck declarations for both derivative orders.

```python
from oo_solver import optimization_from_reader

problem = optimization_from_reader(deck, analysis="statics")
point = problem.evaluate(problem.initial_values)
displacement_gradient = point.gradients()["responses"][10]
curvature = point.hessian_vector({10: [1.]}, [1.])
print(curvature["report"])
```

**C++20**

```cpp
const auto problem = oos::api::prepareOptimization(deck);
const auto point = problem->evaluate(problem->problem().initialValues());
const auto gradient = point->gradients({.responseIds = {10}});
const auto& displacementGradient = gradient.responses.at(0).values;
const std::array direction{1.0};
const oos::design::ProductWeights weights{.rows = 1, .weights = {{10, {1.0}}}};
std::vector<double> curvature;
const auto report = point->hessianVector(weights, direction, 1, curvature);
std::cout << report.hasHessianBlocks << " " << report.perturbedLinearisations;
```

The [API reference](api.rst) includes the current TypeScript declarations, including mode selectors and report flags.
The [native element reference](element-derivative-api.rst) includes the C++ derivative facets directly from headers.

## Verification

The [test guide](testing.md#derivative-checks) lists the focused derivative suites. They compare complete analysis
differences, differences of gradients, direct/adjoint results and Jacobian/Hessian products on small models.
Each case uses the existing 900 ms watchdog and one-second ceiling. Timings are measured on the executing machine
and written to `build/test-results/`; they are not a fixed performance guarantee for larger models.
