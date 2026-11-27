# Laminated composite shells

Six shell elements of the solver take laminated sections: a stack of plies given by a `CompProp` or
`CompPropGlobal` card, each ply with its material, thickness and fibre angle. The solver builds the membrane, bending,
coupling and transverse shear stiffness of the stack, recovers the stresses, strains and failure indices of every
ply, and differentiates these results with respect to the ply thicknesses for design optimization.

Card and field names are those of the `.oos` model format. The O_o_Model card reference gives the record layouts of
the [laminate cards](https://github.com/jcdulas/O_o_Model/blob/main/docs/reference/cards/properties.md#compprop) and
of the [ply materials](https://github.com/jcdulas/O_o_Model/blob/main/docs/reference/cards/materials.md#orthomat2d).

## What is supported

| Topic | Support |
| --- | --- |
| Laminate cards | `CompProp`; `CompPropGlobal` (plies with global ply IDs) |
| Elements | `ShellQuad4`, `ShellTria3`, `ShellQuad8`, `ShellTria6`, `ShellQuad4Drilling`, `ShellTria3Drilling`, in `ClassicFormulation` (the default) and in `MixedHybridFormulation` |
| Ply materials | `OrthoMat2D` (orthotropic, with strengths), `AnisoMat2D`, `IsoMat` |
| Stacking | any sequence of plies, symmetric or not; the options `Symmetry`, `MembraneOnly`, `BendingOnly` and `SmearedLaminate` |
| Section | classical lamination theory (membrane, coupling and bending stiffness), transverse shear from the stacking sequence, thermal forces and moments of the plies |
| Analyses | statics, modes, buckling and static aeroelasticity, like any shell section |
| Ply results | stresses, strains and transverse shear stresses of each ply, failure indices and strength ratios of the plies and of their bonds |
| Failure criteria | `Hill`, `Hoffman`, `TsaiWu`, `StrainTheory` (maximum strain) |
| Design | ply thicknesses as design fields; ply stress, strain, failure index and strength ratio responses; first derivatives by the direct or the adjoint method |

## Example: a laminated cantilever strip

A strip of one `ShellQuad4`, 200 mm long and 50 mm wide, clamped at one end and loaded by 1 N at the other. Its
section is a [0/90/90/0] laminate of carbon/epoxy plies 0.25 mm thick. The units are N, mm and MPa.

```text
[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
StressOutput;All
StrainOutput;All
ElemForceOutput;All
[model]
# A cantilever strip of one four-node shell, 200 mm long and 50 mm wide (units: N, mm, MPa).
Node;1;;0.;0.;0.
Node;2;;200.;0.;0.
Node;3;;200.;50.;0.
Node;4;;0.;50.;0.
ShellQuad4;1;1;1;2;3;4
# A [0/90/90/0] laminate of 0.25 mm plies of material 10, Tsai-Wu failure indices, bond allowable 60 MPa.
CompProp;1;;;60.;TsaiWu;;;;10;0.25;0.;Yes;10;0.25;90.;Yes;10;0.25;90.;Yes;10;0.25;0.;Yes
# A carbon/epoxy ply: E1, E2, nu12, G12, G1Z, G2Z, density, three blank fields (expansion, reference temperature),
# then Xt, Xc, Yt, Yc, S, a blank damping and the Tsai-Wu interaction term F12.
OrthoMat2D;10;140000.;10000.;0.3;5000.;5000.;3500.;1.6e-9;;;;1500.;1200.;50.;200.;70.;;-3.7e-6
SinglePointConstrList;1;123456;1;4
PointForceLoad;2;2;;0.5;0.;0.;1.
PointForceLoad;2;3;;0.5;0.;0.;1.
```

The `CompProp` record gives, in order, the property ID, a blank `bottomOffset` (the stack is centred on the reference
surface), a blank `nonStructMass`, the `bondAllowable`, the `failureTheory`, blank `referenceTemperature`,
`structuralDamping` and `laminateOption`, then four plies of `plyMatId;plyThickness;plyAngle;plyStressOutput` from the
bottom. The element gives no material angle, so the material axis runs along its side from grid point 1 to grid
point 2, here the X axis: the ply angles are measured from it.

The tip deflects by 5.148 mm (5.150 mm in `MixedHybridFormulation`). Beam theory with the bending stiffness of the
stack, `P L³ / (3 b D11)`, gives 5.139 mm. The force table gives the moment `mx` = 2 N·mm/mm at the centre of the
element, `P (L − x) / b`. The ply tables give one row per ply at the centre of the element:

| Ply | Angle | `o11` (MPa) | `t1z` (MPa) | `failureIndex` | `bondFailureIndex` | `elementFailureIndex` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0° | 10.18 | 0.02545 | 0.000693 | 0.000424 | NaN |
| 2 | 90° | −0.2188 | 0.02606 | 0.003593 | 0.000434 | NaN |
| 3 | 90° | 0.2188 | 0.02545 | −0.003581 | 0.000424 | NaN |
| 4 | 0° | −10.18 | 0 | −0.000596 | NaN | 0.003593 |

The 90° ply in transverse tension (ply 2) has the largest index. The transverse shear stress `t1z` of a ply is the
one at its top face: largest at the mid-plane (top of ply 2), zero at the free top surface (ply 4).

Save the model as `laminate-strip.oos` and read its failure indices. Initialize `reader` and `solver` as in
[Getting started](getting-started.md). In Node.js, `readFile` comes from `node:fs/promises`; in C++, reading an `.oos`
file takes the optional OOS adapter (`oor::oos`).

**JavaScript / WebAssembly**

```javascript
reader.writeFile("/work/laminate-strip.oos", await readFile("laminate-strip.oos"));
const deck = reader.readModel("/work/laminate-strip.oos");
const model = solver.modelFromReader(reader, deck);
try {
  const result = solver.run(model, "statics");
  const table = result.elementTables.find(
    (t) => t.kind === "failureIndices" && t.elementType === "ShellQuad4Composite");
  const width = table.components.length;
  for (let row = 0; row < table.ids.length; ++row)
    console.log(`element ${table.ids[row]}, ply ${table.layers[row]}: ${table.values[row * width]}`);
} finally {
  model.delete(); deck.delete();
}
```

**Python**

```python
from oo_model import read_model
from oo_solver import model_from_reader

deck = read_model("laminate-strip.oos")
result = model_from_reader(deck).run("statics")
table = next(t for t in result["element_tables"]
             if t["kind"] == "failureIndices" and t["element_type"] == "ShellQuad4Composite")
for element, ply, row in zip(table["ids"], table["layers"], table["values"]):
    print(f"element {element}, ply {ply}: {row[0]}")
```

**C++20**

```cpp
const auto deck = oor::formats::oos::read("laminate-strip.oos");
const auto model = oos::api::modelFromDeck(deck);
const auto result = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {});
for (const auto& table : result.tables.elementTables)
    if (table.kind == "failureIndices" && table.elementType == "ShellQuad4Composite")
        for (std::size_t row = 0; row < table.rowCount(); ++row)
            std::cout << "element " << table.ids[row] << ", ply " << table.layers[row] << ": "
                      << table.values[row * table.componentCount()] << '\n';
```

## Laminate cards

`CompProp` and `CompPropGlobal` have the same fields; each ply of a `CompPropGlobal` also carries a global ply ID.

| Field | Use |
| --- | --- |
| `bottomOffset` | Height of the bottom of the stack above the reference surface. Blank: minus half the total thickness, the stack centred. A stack off the reference surface couples membrane and bending |
| `nonStructMass` | Mass per unit area added to that of the plies |
| `bondAllowable` | Allowable interlaminar shear stress of the bond between plies, for the bond indices. Blank or 0: no bond index |
| `failureTheory` | `Hill`, `Hoffman`, `TsaiWu` or `StrainTheory`. Blank: no failure index. The Hashin criteria are refused |
| `referenceTemperature` | Stress-free temperature of the laminate, used when the subcase gives no initial temperatures. The reference temperatures of the ply materials are not used |
| `structuralDamping` | Not used: the analyses of the solver are undamped |
| `laminateOption` | Blank: the plies as given. `Symmetry`: the plies given, then their mirror image, numbered on from the bottom (n plies give 2n). `MembraneOnly`, `BendingOnly`, `SmearedLaminate`: see [the section](#section-of-the-laminate). `SandwichCore` is refused |
| `plyMatId` | Material of the ply: `OrthoMat2D`, `AnisoMat2D` or `IsoMat` |
| `plyThickness` | Thickness of the ply; it must be positive |
| `plyAngle` | Angle of the ply's fibre direction from the material axis of the element, in degrees |
| `plyStressOutput` | `Yes`: the ply has rows in the ply tables. When no ply says `Yes`, every ply has rows |
| `globalPlyId` (`CompPropGlobal`) | Read but not used: the result rows number the plies by their position in the stack |

Fields of the element cards that matter for a laminate:

- `materialAngle` (degrees from the side G1→G2) or `matFrameId` (the x axis of that frame projected on the element)
  sets the material axis from which the ply angles are measured;
- `offset` moves the reference surface off the grid points, as for any shell;
- the corner thicknesses are not used: the plies give the thickness.

## Ply materials

| Card | What the solver reads |
| --- | --- |
| `OrthoMat2D` | `youngsModulus1`, `youngsModulus2`, `poissonRatio12`, `shearModulus12`; the transverse shear moduli `shearModulus1Z`, `shearModulus2Z` (blank: rigid in that direction); `density`; `thermalExpansion1`, `thermalExpansion2`; the strengths `tensionStrength1`, `compressionStrength1`, `tensionStrength2`, `compressionStrength2` (positive values; a blank compression strength takes the tension one), `shearStrength`; `interactionTerm12`, the coefficient F12 of the Tsai-Wu criterion (blank: 0); `strainAllowables` (1: the strengths are strains, for `StrainTheory`) |
| `AnisoMat2D` | the plane-stress matrix, `density`, the three expansion coefficients; no strengths, no transverse shear moduli (see [the section](#section-of-the-laminate)) |
| `IsoMat` | Young's modulus, Poisson's ratio, shear modulus (also the transverse shear modulus), `density`, expansion; no strengths |

The plane-stress matrix of an `OrthoMat2D` ply is that of an orthotropic lamina (Jones, *Mechanics of Composite
Materials*, 2nd ed., 1999, eq. 2.61), turned by the ply angle.

## Section of the laminate

The membrane, coupling and bending stiffnesses are those of the classical lamination theory (Jones, §4.2), summed over
the plies from the bottom, with `Q̄k` the plane-stress matrix of ply k turned by its angle and `zk−1`, `zk` the heights
of its faces above the reference surface:

```text
A = Σ Q̄k (zk − zk−1)      B = ½ Σ Q̄k (zk² − zk−1²)      D = ⅓ Σ Q̄k (zk³ − zk−1³)
```

`ShellQuad4`, `ShellTria3`, `ShellQuad8`, `ShellTria6` and `MixedHybridFormulation` take the coupling `B` of an
unsymmetric stack or of a stack off the reference surface.

**Transverse shear.** Along each material axis, the solver takes the shear stress distribution of the stack in
cylindrical bending (Whitney, *Structural Analysis of Laminated Anisotropic Plates*, 1987, §4.6): per unit shear force,
the stress at a height z is the first moment, about the neutral axis, of the ply moduli below z, divided by the reduced
bending stiffness `D − B²/A`. The shear stiffness H follows from the shear energy, `1/H = ∫ τ²/G dz`; for one material
it is `5/6 G t`. The modulus of a ply at the angle θ is `G1Z cos²θ + G2Z sin²θ` along x and `G1Z sin²θ + G2Z cos²θ`
along y. A blank `shearModulus1Z` or `shearModulus2Z` makes a ply rigid along that axis: it adds no flexibility, and a
sandwich whose faces leave them blank shears in its core. The laminate is rigid in transverse shear along an axis only
when every ply is; the element then uses its own stiff shear term.

This stiffness depends on the stacking sequence. Written as `H = κ Σ Gk tk`, with `Gk` the shear modulus of ply k
along the axis and `tk` its thickness, the factor κ is 5/6 for one material and takes another value for plies of
different moduli: the cross-ply [0/90]s of the example above has κ = 0.778 along x and 0.660 along y.

**Convention of the transverse shear.** The first-order shear deformation theory as it is often programmed takes one
factor for every stack, `H = 5/6 Σ Gk tk` (Reddy, *Mechanics of Laminated Composite Plates and Shells*, 2nd ed., 2004,
§3.4, with the shear correction coefficient 5/6). A model parameter chooses between the two:

| `Parameter;LaminateTransverseShear;…` | Transverse shear stiffness of a laminate |
| --- | --- |
| `Equilibrium` (the default) | from the shear stresses of the stack, as above |
| `FiveSixths` | `5/6 Σ Gk tk`, whatever the order of the plies; a ply rigid along an axis makes the laminate rigid along it |

The parameter holds for the whole model and for both formulations; any other value is refused (`invalid-parameter`).
The transverse shear stresses of the plies are those of the equilibrium under both. Use `FiveSixths` to compare with a
program or a closed form that takes that factor; `Equilibrium` is the closer of the two to a solid model of the stack.

The `report` of every run lists, for each laminate property the shells use, the stiffness along the two material
axes and the factor κ it amounts to, so that the convention can be read before two results are compared:

```python
for row in result["report"]["laminate_shear"]:
    print(row["card"], row["prop_id"], row["stiffness"], row["factor"])
# The laminate of the example above: CompProp 1 (3306.7, 2806.7) (0.778, 0.660)
```

In JavaScript the rows are `run.report.laminateShear`, each `{ card, propId, stiffness: [x, y], factor: [x, y] }`.
A factor of 0 means that it is not defined: the laminate, or one of its plies, is rigid along that axis. When a
factor is more than 1 % from 5/6, the run says so with the information `laminate-shear-factor`, which counts these
properties, names the furthest and gives the parameter above.

An `AnisoMat2D` ply gives no transverse shear modulus. In `ClassicFormulation` such a ply is rigid in transverse
shear, as an `OrthoMat2D` ply whose two moduli are blank: no modulus given, no flexibility. In
`MixedHybridFormulation` it takes the in-plane shear modulus G33 of its matrix along both axes, the closest estimate
the card allows (exact for an isotropic material written as an `AnisoMat2D`). Use `OrthoMat2D` plies to give the
moduli.

**Temperature.** A temperature change `ΔT + z T′` through the thickness (T′ the gradient of a `PlateTemperatureLoad`)
gives the thermal forces and moments of each ply from its expansion coefficients (Jones, §4.5). The ply stresses
subtract the thermal strain of each ply.

**Mass.** The mass per unit area is the sum of ply density times ply thickness, plus `nonStructMass`.

**Options.** `MembraneOnly` keeps the membrane stiffness alone, `BendingOnly` the bending stiffness alone.
`SmearedLaminate` ignores the stacking sequence: `D = A t²/12`, no coupling, and a rigid transverse shear.

## Ply results

The ply results are element tables of their own, of element type `<card>Composite` (`ShellQuad4Composite`,
`ShellTria3Composite`, `ShellQuad8Composite`, `ShellTria6Composite`, `ShellQuad4DrillingComposite`,
`ShellTria3DrillingComposite`). Each table has one row per ply with an output, at the centre of the element (grid
point 0); its layer is the ply number, 1 at the bottom.

| Table kind | Asked by | Columns |
| --- | --- | --- |
| `stress` | `StressOutput` | `o11`, `o22`, `t12`, `t1z`, `t2z`, `angle`, `major`, `minor`, `vonMises` |
| `strain` | `StrainOutput` | `e11`, `e22`, `e12`, `e1z`, `e2z`, `angle`, `major`, `minor`, `vonMises` |
| `failureIndices` | `StressOutput`, with a `failureTheory` | `failureIndex`, `bondFailureIndex`, `elementFailureIndex` |
| `strengthRatios` | `StressOutput`, with a `failureTheory` and `Parameter;StrengthRatioCompositeStress;Yes` | `strengthRatio`, `bondStrengthRatio`, `elementStrengthRatio` |

- The in-plane stresses and strains are those at the mid-thickness of the ply, in the ply axes (1 along the fibres).
  The strains include the thermal strain; the stresses are those of the mechanical strain `ε − α ΔT`. `major` and
  `minor` are the principal values of the in-plane components, `angle` the direction of the major one in degrees from
  axis 1, and `vonMises` their von Mises value.
- `t1z` and `t2z` are the transverse shear stresses along the material axes x and y of the element, not along the ply
  axes. They are those at the top face of each ply, from the distribution above, whatever the stack: the top ply's
  are zero, and a stack of equal plies has the parabola of a homogeneous plate, `1.5 Q/t` at its middle. Only a
  `SmearedLaminate`, whose stacking is ignored, gives every ply the mean stress `Q/t`. `e1z` and `e2z` are those
  stresses over the ply's modulus in the same direction (0 where the ply is rigid).
- The element's own `forces` table (`ElemForceOutput`) gives the force and moment resultants of the laminate.
- The model parameter `NoCompositeStress` chooses what a laminated element writes under `StressOutput` and
  `StrainOutput`: 1 (the default) the ply tables only; 0 the ply tables and the stresses and strains of the equivalent
  homogeneous shell; −1 the equivalent homogeneous shell only, with no ply table.

**Failure indices.** With σ1, σ2, τ12 the ply stresses and Xt, Xc, Yt, Yc, S the strengths of its material:

```text
Hill          σ1²/X² − σ1 σ2/X² + σ2²/Y² + τ12²/S²
Hoffman       (1/Xt − 1/Xc) σ1 + (1/Yt − 1/Yc) σ2 + σ1²/(Xt Xc) + σ2²/(Yt Yc) − σ1 σ2/(Xt Xc) + τ12²/S²
TsaiWu        (1/Xt − 1/Xc) σ1 + (1/Yt − 1/Yc) σ2 + σ1²/(Xt Xc) + σ2²/(Yt Yc) + τ12²/S² + 2 F12 σ1 σ2
StrainTheory  the largest of ε1/ε1t (−ε1/ε1c in compression), ε2/ε2t (−ε2/ε2c), |γ12|/γs
```

In `Hill`, X is Xt or Xc by the sign of σ1 (of σ1 σ2 in the product term) and Y is Yt or Yc by the sign of σ2.
`StrainTheory` reads the mechanical strains and takes as allowable strains the strengths over E1, E2 and G12, or the
strengths themselves when `strainAllowables` is 1. The `Hoffman` and `TsaiWu` indices have linear terms and can be
negative under compression.

The bond index of a ply is `max(|t1z|, |t2z|) / bondAllowable` at its top face, the bond with the ply above; the top
ply has none (NaN). The element index, the largest ply or bond index of the plies with an output, is on the last row
of the element; the other rows hold NaN.

**Strength ratios.** With `Parameter;StrengthRatioCompositeStress;Yes`, a `strengthRatios` table has the same rows.
The strength ratio of a ply is the factor λ that would bring it to failure, F(λσ) = 1: `1/√F` for `Hill`, `1/F` for
`StrainTheory`, and for `Hoffman` and `TsaiWu` the positive root of `a λ² + b λ = 1`, a and b the quadratic and linear
parts of the criterion at the ply stresses. The bond ratio is the inverse of the bond index. An unloaded ply or bond
has the ratio 1.0E20. The element ratio, the smallest ply or bond ratio, is on the last row of the element. A ratio
below 1 means failure, as an index above 1 does.

## Design and derivatives

The ply thicknesses of a `CompProp` or `CompPropGlobal` are design fields. A deck relation names ply k as
`plyThickness(k)`, one-based; a JavaScript relation names `field: "plyThickness"` with the zero-based
`fieldItem: k − 1`. Under `Symmetry`, the mirrored ply follows its ply. A blank `bottomOffset` keeps the stack centred
as a thickness changes; a given one stays where it is. The fields of a ply material (`OrthoMat2D` and the others) are
designed through material relations and reach every laminate whose plies use that material. The other fields of the
laminate cards, ply angles included, are refused when the design model is built.

| Response type | Items (`attributeA`) | Ply |
| --- | --- | --- |
| `CompositeStress` | 3 to 11: `o11`, `o22`, `t12`, `t1z`, `t2z`, `angle`, `major`, `minor`, `vonMises` | `attributeB`, the ply number (default 1); in JavaScript `plies` for several |
| `CompositeStrain` | 3 to 11: the same columns of the strain table | as above |
| `CompositeFailure` | 5: the ply's failure index; 7: its bond index | as above |
| `CompositeStrengthRatio` | 5: the ratio of the ply stresses; 7: the bond ratio | as above |

These responses take the elements of a `CompProp` or `CompPropGlobal` among the six shells. The strength ratio is the
one of the [strength ratio table](#ply-results); the response does not need the parameter of the table.
`CompositeStress` and `CompositeStrain` also take the mode selector of a modal analysis
([modal derivatives](modal-derivatives.md)).

How the derivatives along a ply thickness are formed:

| | `ClassicFormulation` | `MixedHybridFormulation` |
| --- | --- | --- |
| Stiffness, force resultants and ply rows | closed form, from the derivatives of A, B, D, H and of the ply heights | differences of the element computations |
| Mass | closed form on `ShellQuad4`, `ShellTria3` and their drilling variants; differences on `ShellQuad8`, `ShellTria6` | differences of the element computations |
| A subcase with a temperature load | differences of the element computations | differences of the element computations |

The ply stress and strain components, their von Mises values, the failure indices and the bond indices have exact
slopes in the displacements: an index through the gradient of its criterion at the ply stresses and strains, except
where the criterion has no derivative (two equal ratios of `StrainTheory`, two equal shear stresses of a bond
index), which is differenced. Strength ratios, angles and principal values are differenced over the element, and so
are the fields of the ply materials. Second derivatives (Hessian-vector products) along a ply thickness difference
the linearisation: their report gives `perturbedLinearisations` above 0 and names the ply thickness in
`hessianBlocksReason`. See [derivative coverage and reports](derivatives.md) for the counters.

Add these records to the example: one variable for the thickness of the two outer plies, the failure index of ply 2,
the bond index above ply 1 and the strength ratio of ply 2.

```text
# One variable: the thickness of the two outer 0-degree plies (plies 1 and 4 of the card).
DesignVar;1;OUTERPLY;0.25;0.1;0.5
DesignPropRelation1;1;CompProp;1;plyThickness(1);;;;1;1.
DesignPropRelation1;2;CompProp;1;plyThickness(4);;;;1;1.
# Item 5 of ply 2: its failure index; item 7 of ply 1: the bond index above it; item 5 of a strength ratio.
DesignResp1;20;PLYFI;CompositeFailure;CompProp;;5;2;1
DesignResp1;21;BONDFI;CompositeFailure;Elements;;7;1;1
DesignResp1;22;PLYSR;CompositeStrengthRatio;CompProp;;5;2;1
```

| Response | Value | Derivative |
| --- | ---: | ---: |
| 20, failure index of ply 2 | 0.003593 | −0.02457 |
| 21, bond index above ply 1 | 0.000424 | −0.000617 |
| 22, strength ratio of ply 2 | 207.0 | 1402 |

The automatic, direct and adjoint methods and complete analysis differences agree on these derivatives. Responses 20
and 21 alone report `fieldPerturbations: 0` and `analysisPerturbations: 0`; the strength ratio adds one element
difference per relation.

**JavaScript / WebAssembly**

```javascript
const problem = solver.optimizationFromReader(reader, deck);
const point = problem.evaluate(problem.metadata().initialValues);
try {
  const values = point.values().responses;
  const gradients = point.gradients();
  for (const id of [20, 21, 22])
    console.log(id, values.find((r) => r.id === id).values[0], gradients.responses.find((r) => r.id === id).values[0]);
  console.log(gradients.report.fieldPerturbations, gradients.report.analysisPerturbations);
} finally {
  point.delete(); problem.delete();
}
```

**Python**

```python
from oo_solver import optimization_from_reader

problem = optimization_from_reader(deck, analysis="statics")
point = problem.evaluate(problem.initial_values)
gradients = point.gradients()
for response in (20, 21, 22):
    print(response, point.responses[response][0], gradients["responses"][response][0, 0])
print(gradients["report"]["field_perturbations"], gradients["report"]["analysis_perturbations"])
```

**C++20**

```cpp
const auto problem = oos::api::prepareOptimization(deck);
const auto point = problem->evaluate(problem->problem().initialValues());
const auto gradient = point->gradients({.responseIds = {20, 21, 22}});
for (const auto& response : point->evaluation().responses)
    std::cout << response.id << " = " << response.values.at(0) << '\n';
for (const auto& response : gradient.responses)
    std::cout << "d" << response.id << "/dx = " << response.values.at(0) << '\n';
std::cout << gradient.report.fieldPerturbations << " " << gradient.report.analysisPerturbations << '\n';
```

## Limits

- The Hashin failure criteria and the `SandwichCore` option are refused (`unsupported-field`). A sandwich is modelled
  as plies: the face sheets and the core, each with its material.
- In `ClassicFormulation`, `ShellQuad4Drilling` and `ShellTria3Drilling` refuse a laminate that couples membrane and
  bending: an unsymmetric stack, or a stack off the reference surface. The four other shells take it, and so does
  `MixedHybridFormulation` on every shell. A coupling that is only round-off (a symmetric stack whose ply heights
  are not exact in binary) is not a coupling.
- The ply results are at the centre of the element and at the mid-thickness of each ply only; there are no ply rows
  at the corners, nor at the faces of the plies.
- The transverse shear stiffness is formed separately along the two material axes, without a term coupling them.
- The convention of the transverse shear stiffness (`LaminateTransverseShear`) is one for the model, not per property.
- Failure indices and strength ratios need an `OrthoMat2D` with positive `tensionStrength1`, `tensionStrength2` and
  `shearStrength` for every ply with an output. When one lacks them, the elements of the property write no failure
  index row, and the run warns of it once per property (`failure-index-not-written`). The element index covers the
  plies with an output only.
- An `AnisoMat2D` ply has no transverse shear moduli of its own: rigid in `ClassicFormulation`, its in-plane shear
  modulus in `MixedHybridFormulation`. Use `OrthoMat2D` plies where the transverse shear of the laminate matters.
- The cards `CompPropIntegration`, `CompPropFreqDamping` and `OrthoMat2DFailure` are not supported: the support check
  refuses a model that has them (`unsupported-card`).
- Only the ply thicknesses of the laminate cards are design fields; second derivatives along them are differenced.
- In `MixedHybridFormulation`, the membrane force at the centre of an unsymmetric laminate converges slowly on coarse
  meshes: on a simply supported antisymmetric [0/90] plate, 1.80 for 1.31 at 4 × 4 elements, 1.35 at 8 × 8 and 1.312
  at 16 × 16 ([specification](mixed-hybrid-formulation-spec.md), section 3.6). On the same 4 × 4 mesh the deflection
  is within 0.5 % of the exact solution, the moments and the ply stresses within 1 %.

## Verification

| Test | What it checks |
| --- | --- |
| [shell-statics](../wasm/test/validation/shell-statics.test.mjs), section 4 | `MixedHybridFormulation`: simply supported [0/90]s and [0/90] plates under pressure against the Navier solution of the classical lamination theory (deflection within 0.5 %, moments and ply stresses within 1 %) on 4 × 4 `ShellQuad4`, `ShellTria3` and a distorted mesh; exact uniform strain and curvature states of a [0/45/90] laminate (forces, moments, ply stresses); a [0/90]s plate with transverse shear against `ClassicFormulation` extrapolated from 32 × 32 and 64 × 64 elements |
| [shell-dynamics](../wasm/test/validation/shell-dynamics.test.mjs) | `MixedHybridFormulation`: the first four frequencies of a simply supported [0/90]s plate of 8 × 8 elements against the classical lamination theory |
| [composites](../wasm/test/cases/composites.mjs) | A symmetric stack of plies of 0.13 on the drilling shells; the parabola of the transverse shear in a stack of equal plies and its bond indices; the failure indices of the four criteria against their formulas and the strength ratios that bring each criterion to one; the warning of a failure theory without strengths; an `AnisoMat2D` ply rigid in transverse shear in `ClassicFormulation`, with its in-plane shear modulus in `MixedHybridFormulation` |
| [review-elements](../wasm/test/cases/review-elements.mjs) | `ClassicFormulation`: the tip deflection of a sandwich whose faces leave their transverse shear moduli blank, against its bending and shear stiffness; the transverse shear strain of each ply as its stress over the ply's modulus in that direction |
| [mixed-hybrid](../wasm/test/cases/mixed-hybrid.mjs) | An unsymmetric sandwich whose plies give no transverse shear moduli: the two formulations agree within 3 % |
| [assemblies](../wasm/test/validation/assemblies.test.mjs) | A band of [0/45/−45/90]s laminate with a material frame on a curved stiffened panel, in an assembly of shells, bars, beams and connections: its equilibrium, strain energy and mass, its results in other displacement systems, and the mixed formulation against finer classic models |
| [review-design](../wasm/test/cases/review-design.mjs) | A ply response whose ply number is not an integer is refused |
| [composites](../wasm/test/cases/composites.mjs), section 7 | The transverse shear stiffness of a thick cross-ply stack against the shear energy integrated apart from the solver, its factor in the report of the run, `LaminateTransverseShear = FiveSixths` and the tip deflection it changes, a wrong value of the parameter |
| [TransverseShearModulusChecks](../checks/shells/TransverseShearModulusChecks.cpp) | The transverse shear modulus of a ply by its angle, and a blank modulus as a rigid one; the sum of the plies' shear moduli and the rate of the stiffness along a ply thickness under `FiveSixths` |

The section and the ply results are in [Laminate.hpp](../include/oos/elements/shells/Laminate.hpp) and
[ShellRecovery.hpp](../include/oos/elements/shells/ShellRecovery.hpp), the strength ratio in
[StrengthRatio.hpp](../include/oos/design/responses/composite/StrengthRatio.hpp), the derivatives along a ply
thickness in [LaminateRate.hpp](../include/oos/elements/shells/LaminateRate.hpp).
