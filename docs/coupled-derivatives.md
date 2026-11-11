# Analytic buckling and trim curvature

The native C++ optimization API computes additional Hessian-vector products by differentiating the coupled solution
equations. The WebAssembly API calls the same implementation. No nearby design point is evaluated on these paths.

```js
const product = point.hessianVector({ responses: { 10: [1] } }, direction);
console.log(product.report.usesBucklingTangents, product.report.usesTrimTangents);
// An analytic coupled product reports perturbedLinearisations === 0.
```

Both paths require linear sizing relations and closed first, second and mixed element rates. They assemble actions
on element DOFs, reuse the accepted run's static/trim factors and stream field tangents. Storage does not grow as
the number of structural DOFs times the number of design variables. A dense requested Hessian still takes its
normal variable-by-variable output storage.

## Static aeroelastic trim

`design/aids/TrimHessian` differentiates the trimmed state and the restrained/unrestrained stability coefficient
states. At fixed aerodynamic and support geometry, the same coupled operator solves each tangent. For a field `p`
and a direction `v`, the structural and support-equilibrium pseudo-loads are:

```text
stiffness load = -(K_pv u + K_p u_v + K_v u_p)
inertial load  = -(M_pv (w-a) + M_p w_v + M_v w_p)
```

Here `w = D_r * supportAcceleration`; `a` is the fixed applied inertial acceleration. `TrimLinearisation` applies
the aerodynamic feedback, free trim variables and support equilibrium to these loads with its original factors.
For free-body stability coefficients the motion tangent also differentiates
`m_rr * acceleration = D_r^T * aerodynamicLoad`, including the mass projection's derivative.

Supported weighted responses are `TrimVariable`, `StabilityDerivative`, `Displacement`, `Stress`, `Strain` and
`Force`. Recovered outputs include their displacement Hessians, mixed displacement/field blocks and field curvature;
supported von Mises columns use their full chain rule. Recovery blocks are shared with the modal implementation in
`RecoveredFieldCurvature`. Dense and sparse response weights have the same meaning.

The report sets `usesTrimTangents: true`, `perturbedLinearisations: 0` and `factorisations: 0`: the factors already
belong to the accepted trim run. First-order trim mass rates also use the element's closed facet before selecting
numerical element rates. Maximum and minimum design bounds do not require one-sided steps on the analytic path.

## Linear buckling

`design/aids/BucklingHessian` includes both derivatives of the static preload and of the buckling shape. It uses
the equivalent symmetric eigenproblem

```text
Kd phi = mu K phi,    mu = -1/lambda,    phi^T K phi = 1
```

The elastic stiffness `K` is a positive metric for both positive and negative load factors. A sparse bordered
factor of `[Kd-mu K, K phi; (K phi)^T, 0]` gives the shape tangent without requiring unextracted modes.
The existing static factor gives the preload tangents:

```text
K u_p  = -K_p u
K u_pv = -(K_pv u + K_p u_v + K_v u_p)
```

At a mechanical preload, geometric stiffness is linear in the preload displacement. Its total mixed action includes
the fixed-preload field curvature, the two field/preload terms and `Kd(u_pv)`. The root curvature follows the
symmetric eigenpair formula and both derivatives of `lambda = -1/mu`.

The element layer now exposes `differentialStiffnessFieldMixedDerivative`. The shell quadrilateral and triangular
families supply thickness curvature, including their drilling variants. `Rod2` supplies first and mixed area/torsion
rates: its geometric stiffness is linear in area and independent of torsion. Scalar connection stiffness can
contribute to the elastic matrix and to load sharing without adding its own geometric stiffness.

`Bar2` and `Beam2` provide first and mixed sizing rates of geometric stiffness, including the derivatives of the
released-end completion map. Their section/material stiffness and mass curvature also feed the trim path.
A potentially section-dependent distributed-load facet only excludes a closed product when the selected subcase
actually contains distributed loads. Fixed concentrated-load cases can use the beam rates. See
[bar and beam derivatives](beam-derivatives.md) for the field list and remaining load/recovery restrictions.

For `ShellQuad4` and `ShellTria3`, thickness derivatives of stiffness, mass and recovered rows now use
`SecondOrderScalar`. Each scalar carries its value and first two derivatives through the ordinary kernel operations,
including quotients and square roots. This replaces the former difference of complex-step first derivatives.
The same calculation supplies geometric-stiffness curvature on warped quadrilaterals as well as flat shells.

The report sets `usesBucklingTangents: true` and `perturbedLinearisations: 0`. Each weighted simple root has one
bordered factor; repeated uses of the same root share it. Repeated or numerically unresolved roots are refused.

## Remaining numerical paths

Shape designs that move aerodynamic splines or support geometry continue to use complete trim differences.
Derived/nonlinear relations, equation responses, unavailable element/recovery curvature, condensed analyses and
unsupported load changes keep the existing difference-of-native-gradients path. Its tangent flags remain false.
Buckling with thermal/deformation preloads, inertia relief, combined subcases or mass-dependent applied loads also
keeps that general path. These restrictions are checked before using the closed product; no missing term is set
to zero as a substitute.

Tests compare closed load-sharing formulas of a rod, positive/negative buckling factors, quadrilateral/triangular
shells, restrained/free-body trim coefficients, recovered von Mises fields and inertial loads against independent
analysis/gradient differences. Shell cases include warped geometry and both drilling variants. Every test runs
under the existing one-second ceiling.
