# Analytic bar and beam sizing derivatives

`Bar2` and `Beam2` provide native C++ first, second and mixed sizing derivatives. Native and WebAssembly
optimization consumers use the same element facets. The formulas differentiate the existing flexibility
formulation, including shear flexibility, coupled bending, taper and end releases.

See [derivative coverage and reports](derivatives.md) for solution-level prerequisites and
the [native element API](element-derivative-api.rst) for declarations included directly from the C++ headers.

## Stations

A station is a cross-section at a position along the element axis. Its position is the ratio `x/L`, where x is the
distance from end A and L is the element length. A station does not require a mesh node at that position.

| Position `x/L` | Meaning |
| --- | --- |
| `0` | End A |
| `0.5` | Midspan |
| `1` | End B |

A `BeamProp` can define different areas, inertias and recovery points at several stations. Area, inertias, torsion,
NSM and axis/centre offsets are interpolated linearly between stations. Recovery points follow the station's
output/inheritance rules. Intermediate output stations follow their stress-output setting; both ends always have
a recovered row. The optimization item-code selector currently exposes the two ends.

The source station index used in `fieldItem` is its position in the property table's original station list. It is
not the ratio `x/L` or its rank after sorting. For source ratios `[0, 1, 0.4]`, `stationSection` item `8` is I2 of
source station 1, at end B. A uniform single-source property synthesizes an end-B section from the same values,
without creating another source station index.

## Supported fields

The names below are the columns accepted by JavaScript design relations. A vector or nested-list column uses
`fieldItem`; a scalar column leaves it unset. NSM denotes non-structural mass per unit length.

| Card | Field | Item |
| --- | --- | --- |
| `BarProp` | `area`, `bendingInertia1`, `bendingInertia2`, `productInertia12`, `torsionConstant`, `nonStructMass` | Scalar |
| `BarProp` | `shearFactor` | 0 or 1 |
| `BarProp` | `stressPointC`, `stressPointD`, `stressPointE`, `stressPointF` | 0 = y, 1 = z |
| `BeamProp` | `stationSection` | `6 * sourceStation + component`: A, I1, I2, I12, J, NSM |
| `BeamProp` | `stationStress` | `8 * sourceStation + component`: Cy, Cz, Dy, Dz, Ey, Ez, Fy, Fz |
| `BeamProp` | `shearFactor` | 0 or 1 |
| `BeamProp` | `neutralAxis`, `nsmCenter` | Ay, Az, By, Bz, numbered 0 to 3 |
| `IsoMat` | `youngsModulus`, `shearModulus`, `density` | Scalar, explicitly specified independent values |
| `IsoMat` | `thermalExpansion`, `referenceTemperature` | Scalar |

Source station indices refer to the property table's original order, before sorting by `stationRatio`.
The derivatives sort and interpolate the directions with the same rules as the element. A single-station property
also seeds its synthesized end B. `YesSameAsEndA` recovery stations inherit the derivative of end A's points;
their own unused point fields have zero derivative. Neutral-axis movement also changes the recovery points measured
from that axis. The mass-centre fields enter mass alone.

Explicit Poisson ratio has no mechanical effect in this beam formulation when E and G are both specified.
Structural damping and warping coefficients also have known zero mechanical rates in the supported formulation.
Derived isotropic defaults, such as a blank G computed from E and Poisson ratio, retain the existing numerical
chain rule. A material's summary-table copy and its card-table entry are recognized as the same independent field;
they no longer disable an otherwise available analytic derivative.

```js
const declarations = {
  variables: [{ id: 1, initialValue: 0.3, lower: 0.2, upper: 0.5 }],
  relations: [{ id: 1, targetKind: "property", targetCard: "BeamProp", targetId: 7,
    field: "stationSection", fieldItem: 1, // I1 at the first source station
    terms: [{ variableId: 1, coefficient: 1 }] }],
  responses: [{ id: 10, type: "Displacement", attributeA: 3, targets: [42] }],
};
const problem = solver.createOptimization(model, declarations);
const point = problem.evaluate();
const gradient = point.gradients();
const curvature = point.hessianVector({ responses: { 10: [1] } }, [1]);
point.delete();
problem.delete();
```

## Formulation and implementation

`BeamFieldDirections.cpp` maps a card field to section and material directions. `MixedScalar` carries a value,
two first partials and their mixed partial through ordinary arithmetic. Giving both partials the same seed produces
the same-field second derivative. No design step size is used.

`BeamSizing.cpp` integrates section-compliance rates at `BeamFlexibility`'s own quadrature points. For natural
flexibility F and natural stiffness k:

```text
k     = inverse(F)
k_p   = -k F_p k
k_pq  = k (F_q k F_p + F_p k F_q - F_pq) k
K_end = B^T k B
```

The bending compliance uses the full inverse of the 2 by 2 inertia tensor, including at I12 = 0. This retains
coupling derivatives that would disappear if the zero-valued off-diagonal entry were treated as an inactive field.

For released end DOFs r and retained DOFs a, the completion map is `C_ra = -inverse(K_rr) K_ra`. Its first and
mixed derivatives are carried through the condensation, rather than freezing the release map at the base section.
The geometric stiffness uses `C^T Kd(N) C`, so it includes both the axial-force rate and the release-map rates.
Fixed grid rotations and offsets are applied with the ordinary element transforms.

Mass is affine in the station section values at fixed density. Unit-density structural matrices and zero-density
NSM matrices provide exact coefficients. Product rules then include density/area and density/inertia cross terms.
Differentiated rigid-arm transforms include neutral-axis and NSM-centre curvature. Lumped and coupled mass keep
the element's existing choices: `Beam2` includes polar torsional inertia; `Bar2` does not.

`BeamSizingRecovery.cpp` differentiates end forces, station resultants and the linear stress/strain columns.
`BeamFieldDerivatives` packs those values into the common element facets. Recovery-slope rates reuse one resolved
kernel and evaluate exact unit columns; they do not perturb displacements or the model.

## Thermal and distributed applied loads

The shared `ElementLoadContext` carries the selected element's temperatures, pressures and line loads.
`loadFieldDerivative` and `loadFieldMixedDerivative` return grid-coordinate applied-load rates. Equal fields request
the same-field second derivative. An unsupported context returns false, preserving numerical element differences.
The default family implementation returns false; `Bar2` and `Beam2` supply closed thermal and line-load formulas.

`BeamSizingLoads.cpp` differentiates the same complementary-work integral as `BeamFlexibility`:

```text
q0      = integral(S^T c s0 dx)
f_line  = B^T k q0 + [F_total; M_total; 0; 0]
f_grid  = T_load^T C^T f_line
f_heat  = T_load^T K_condensed e_heat
e_heatB = length * (alpha * (T_mean - T_initial) + elongation / length)
```

The quadrature splits at the same station and load endpoints, including interior concentrated forces and moments.
The rates include compliance, natural-stiffness and release-map curvature. Load values, positions and directions
remain fixed. Material E/G and shear factors can change the equivalent nodal line loads and are marked accordingly.

The thermal strain follows element, nodal and default temperatures, with the material reference temperature used
only when no initial temperature is supplied for that element. An explicit zero initial temperature overrides the
reference. Expansion derivatives remain available at alpha = 0. Thermal and imposed-elongation effects also enter
first, second and mixed linear recovery and fixed-preload geometric-stiffness rates.

Static sensitivities select closed applied-load rates before constructing perturbed snapshots. Static Hessian
blocks include `lambda^T P_pq` for both diagonal and mixed design terms. Consequently, supported thermal recovered
fields and line-loaded displacement responses can use fully analytic sizing gradients and Hessian products.
The thermal and line contributions can share a context for second derivatives.

## Recovered fields under line loads

The ordinary recovery and its derivative facets accept optional line loads in `NodalDisplacements::lineLoads`.
The span borrows the subcase's element loads; each family selects the loads of the recovered batch row. It leaves
the ordinary recovery column layout intact. Native and WASM optimization therefore use the same item codes and
field derivatives in loaded and unloaded subcases.

The end forces and section equilibrium are differentiated together:

```text
f_end = K_condensed T u - C^T f_span - K_condensed e_heat
N(x)  = -f_Ax - F_left,x(x)
M1(x) = -m_Az + x f_Ay - M_left,z(x) + neutralY(x) N(x)
M2(x) =  m_Ay + x f_Az + M_left,y(x) + neutralZ(x) N(x)
```

`BeamSizingRecovery` follows the same inclusion convention at concentrated loads as `BeamResponse`: a load at the
interior section is excluded from its left resultant; the section at end B includes endpoint loads. The first,
second and mixed rates include equivalent-load and release-map derivatives, coupled bending, recovery-point
movement, thermal strain and neutral-axis offsets.

For fixed geometry, `r = S(p) u + r0(p, loads)`. The displacement slopes S and their field rates do not depend on
line-load values. The existing cached recovery slopes can therefore be reused across subcases; the loaded values
and explicit field rates retain each subcase's own loads. Static Hessian blocks combine these recovery rates with
the applied-load curvature described above, without perturbing the model.

Public `Bar2` results under line loads use the `Bar2LoadStations` table. Optimization maps end-specific bending and
stress/strain items to the corresponding end row. Shared axial force/stress, shear and torque items select end A.
The mapping also works when loaded and unloaded bars occupy different result tables in one batch. The public
result-table format is unchanged. Bar margin items have no equivalent overall-margin column in the station table.

## Solution paths and limits

The element rates feed SOL 101 static sensitivities, SOL 103 modal tangents, SOL 105 mechanical-preload buckling
tangents and SOL 144 trim tangents. A fully supported first-order evaluation reports `fieldPerturbations: 0` and
`analysisPerturbations: 0`. Closed curvature reports `perturbedLinearisations: 0`, with `hasHessianBlocks` for the
static path or the corresponding `usesModalTangents`, `usesBucklingTangents` or `usesTrimTangents` flag.
See [modal derivatives](modal-derivatives.md) and [coupled derivatives](coupled-derivatives.md) for solution restrictions.

Grid shape, orientation, offset and station-position design fields, library section dimensions and nonlinear
relations keep their existing numerical paths. Linear force, stress and strain responses at the supported bar/beam
ends now have analytic load-aware recovery. Intermediate `Beam2` stations are recovered and differentiated by the
element facets; the optimization item-code selector still accepts the two ends only. Thermal or distributed
preload curvature in coupled buckling/trim products retains the restrictions of those solution paths, even when
the element itself supplies the corresponding load or geometric-stiffness partials.
Stress extrema and margins remain `Other` recovery columns and do not acquire analytic curvature here.
Selected section properties at an inactive zero-stiffness boundary and zero shear factors are declined by these
formulas. The caller retains its existing validation and fallback; changing an active set has no smooth two-sided
derivative. Ordinary element support restrictions still apply, including offset-definition checks in buckling.

The WASM tests compare independent analysis differences and differences of gradients with these first and mixed
derivatives. Cases include zero I12, shear flexibility, end releases, unsorted tapered stations, inherited recovery
points, mass offsets, both mass formulations, modal fields, buckling load sharing, trim under gravity, thermal
sources, reference-temperature overrides, zero expansion, imposed elongation, partial/concentrated line loads and
all linear end-result items of loaded bars and beams. Mixed loaded/unloaded batches and combined thermal/line
loads exercise result lookup and per-element recovery context. Every test runs under the existing 900 ms watchdog
and one-second ceiling.
