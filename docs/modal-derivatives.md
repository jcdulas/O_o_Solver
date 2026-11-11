# Native modal derivatives

Vibration responses can select a mode explicitly. The calculation lives in C++ in `design/sensitivity/ModalItems`,
`design/aids/ModalHessian` and `linalg/eigen/BorderedEigenSystem`. Native and WebAssembly consumers use the same code.

```js
const declarations = {
  variables: [{ id: 1, initialValue: 0.2, lower: 0.1, upper: 0.4 }],
  relations: [{ id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 7,
    field: "thickness", terms: [{ variableId: 1, coefficient: 1 }] }],
  responses: [
    { id: 10, type: "Displacement", targets: [42], attributeA: 3, mode: 1, tracksMode: true },
    { id: 11, type: "Stress", propertyType: "Elements", targets: [100],
      attributeA: 9, mode: 1, tracksMode: true },
  ],
};
const problem = solver.optimizationFromReader(oor, deck, declarations, { analyses: { modes: [1] } });
const point = problem.evaluate();
const gradients = point.gradients();
const product = point.hessianVector({ responses: { 11: [1] } }, [1]);
// product.report.usesModalTangents identifies a closed second-order calculation.
point.delete();
problem.delete();
```

`mode` is numbered from 1. `Displacement`, `Stress`, `Strain`, `Force`, `CompositeStress` and `CompositeStrain`
support the selector. In a model `DesignResp1` card, a positive integer `attributeB` selects the mode of a
non-composite displacement, stress, strain or force response. Composite declarations use `mode` separately from
the ply selector. Returned field item labels include their selected `mode`; static item labels keep their existing layout.

`tracksMode` and the evaluation's `trackedModes` references work for these fields as for eigenvalue responses.
The result tables are indexed by mode as well as subcase, so a response to mode 2 does not read mode 1's rows.
Required nodal and element outputs are recovered automatically. A mode-selected field uses the full vibration path
even when a condensation plan was supplied, because its derivative needs the original pencil.

## First derivatives

For a simple vibration eigenpair, `K phi = lambda M phi`, the mass-normalised eigenvector satisfies
`phi^T M phi = 1`. Its linearised equation and normalisation form a symmetric bordered system:

```text
A = K - lambda M                 b = M phi
B = [ A    b ]
    [ b^T  0 ]
```

For response displacement weights `c`, solving `B [z; t] = [c; 0]` gives the derivative in a sizing field `p`:

```text
dr/dp = explicit recovery rate - z^T (K_p - lambda M_p) phi - t/2 * phi^T M_p phi
```

The pencil is reduced by the same constraints and rigid-element map as the eigensolution. The border is sparse,
balanced against the stiffness scale and eliminated last with indefinite pivoting. One factor serves all requested
field adjoints of that mode. The calculation does not require the unextracted modes or a complete modal basis.

Mass, Maximum and Point normalisations follow the extraction method. Maximum and Point include the derivative of
their output scale; a moving or tied Maximum pivot is a branch boundary. An individual eigenvector at a repeated
root has no unique derivative and is refused, including when its partner was not extracted. The pivot check uses
the original pencil's diagonal scales to detect numerically unresolved partners. Eigenvalue first derivatives
retain their separate repeated-root subspace formulation.

Element matrix and explicit recovery rates use closed forms where available and numerical element rates elsewhere.
`fieldPerturbations: 0` and `analysisPerturbations: 0` distinguish a fully closed sizing calculation. Modal field rows
feed response equations, `jvp`, `vjp` and `streamGradients` through the native response linearisation.

## Second derivatives

For linear sizing relations whose elements provide closed first, second and mixed matrix derivatives,
`ModalHessian` differentiates the bordered system. A tangent solve gives the mode direction, including its
mass-normalisation rate. A differentiated adjoint gives field-response curvature, including the output scale,
local recovery Hessians and mixed stiffness/recovery coefficients. Linear components and supported von Mises
columns use the element's displacement and field blocks. Frequency curvature includes both derivatives of
`sqrt(lambda)/(2*pi)`.

The report sets `usesModalTangents: true` and `perturbedLinearisations: 0`. `hasHessianBlocks` refers to the separate
static element-block path and remains false here. Requested roots share one factor per mode, including when an
eigenvalue, its frequency and its fields are weighted together. An eigenvalue Hessian at a repeated root is refused.

Unavailable second-order element/recovery formulas, derived or nonlinear sizing relations, shape directions,
approximate condensed vibration solutions and equation responses keep the general difference-of-native-gradients
path. The report then leaves `usesModalTangents` false. Trim geometry and buckling curvature use their own derivative
paths; the modal formulas do not replace them.

The modal tests compare closed scalar formulas, coupled modes, rigid constraints and rotated rigid-body mass
matrices against independent analysis differences. The normal test runner enforces a one-second ceiling per test.

[Bar and beam derivatives](beam-derivatives.md) cover `Bar2`/`Beam2` section and material rates, lumped/coupled
mass, polar inertia, mass-centre offsets and linear recovered-field curvature on this modal path.
