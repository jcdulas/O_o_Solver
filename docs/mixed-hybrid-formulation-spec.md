# Mixed hybrid element formulation: implementation specification

Status: accepted, 15 November 2026, implemented on the branch `mixed-hybrid-formulation`. The owner's decisions are
recorded in section 15.

This document specifies how to add a second formulation of the shell, line and solid cards to O_o_Solver, ported
from a reference implementation in Rust. It says what the formulation computes, what the solver must gain to carry
it (unknowns that do not belong to grid points), how each card, load, support and output maps onto it, how the port
is verified, and in which order to build it.

## 1. Scope

### 1.1 What is added

A formulation named `MixedHybridFormulation` (section 15, decision 1), for these cards:

| Group | Cards |
|---|---|
| shells | `ShellQuad4`, `ShellTria3`; `ShellQuad4Drilling`, `ShellTria3Drilling` (the same element: this formulation has no drilling rotation); `ShellQuad8`, `ShellTria6` (their corners make the element, each mid-side node given fixes the curve of its edge: section 4.1) |
| line elements | `Rod2`, `Rod2InlineSection`, `Tube2`, `Bar2`, `Beam2` |
| solids | `SolidTetra`, `SolidPyramid`, `SolidPenta`, `SolidHexa`, each with or without its mid-side nodes |

The three groups are one formulation: a line element on an edge of shells takes its displacement from them, and a
solid that shares an edge or a face with a shell or a line follows it there. The formulation is chosen for the whole
model by one switch (section 4.5): every card of the table then takes it together.

While the switch is on, the cards not listed keep their own formulation and connect to the new elements through the
grid points (section 5.4): the springs, bushings, gaps and masses. The fasteners, welds and glued contacts that pierce
an element of the new formulation use its point interpolation facet (section 5.10). `ShearPanel4` has no counterpart
and is refused (`unsupported-card`).

### 1.2 Analyses

| Analysis | Phase (section 14) |
|---|---|
| SOL 101, full path | 2 (shells), 4 (lines), 5 (solids) |
| SOL 103, full path | 3 |
| SOL 105, full path | 3 |
| SOL 144, and inertia relief in SOL 101 | 3 |
| condensed path, streamed path, components and superelements | 6 |
| design points, sensitivities and optimization | 6 |

### 1.3 Not in scope

- The nonlinear statics of the reference (incremental loading, large rotations). Only its second variation is used,
  as the differential stiffness of SOL 105.
- The other shell element of the reference crate (its covariant element), and that crate's own analysis, solver and
  storage.
- Any change to what `ClassicFormulation` computes: a model that selects no card in the new formulation runs exactly
  the code it runs today, bit for bit (section 13.5).

## 2. Reference implementation

The formulation exists and is validated in the Rust crate `covquad` (repository `ElementsFormulation`, private). The
port reproduces its mathematics; it is verified by the analytic checks of section 13, not by a comparison number by
number (section 15, decision 9).

The reference is the crate at commit `9f55f0b` (15 November 2026, branch `main`), the first that holds the solids of
every shape (document 35 of the crate). A later commit of the crate becomes the reference only when this line is
changed.

| Reference file | What it holds | Document of the crate |
|---|---|---|
| `src/tdnns/space.rs` | reference spaces of the shell (displacement, shear, edge rotation, moments, Regge membrane strains), dual bases, tables per shape and degree | 34, section 4 |
| `src/tdnns/element.rs` | geometry of a shell element (rational map and its derivatives), section law, linear element, loads, resultants | 34, sections 2 to 4, 6 |
| `src/tdnns/nonlinear.rs` | value-gradient-Hessian scalar, nonlinear strain measures, second variation (geometric stiffness) | 34, sections 2.5, 7 |
| `src/tdnns/line.rs` | line elements on shell edges and alone: kinematics, mixed hybrid core, section state | 34, section 8.1 |
| `src/volume.rs` | solid kernel of every shape: spaces, face and edge functions, element functions, stiffness, mass, interior fields, stresses, geometric stiffness | 35, and 28 |
| `src/tdnns/solid.rs` | solids in the analysis: edges and faces, skin, mid-side nodes, ties with shells and lines, supports, loads, load stiffness of following pressures | 35, 34 section 8.2 |
| `src/tdnns/analysis.rs` | numbering, supports, ties, loads, condensation of the interior, static, modal and buckling solutions, inertia relief | 34 |
| `src/model.rs` (`Topology`, `Reconstruction`), `src/surface.rs` | reading of the shell surface from the mesh: edge curves, rational patches, normals at the grid points | 25, 26, 34 section 9 |
| `src/section.rs`, `src/line.rs` (`LineSection`) | section laws of shells (A, B, D, transverse shear, thermal, plies) and line elements | 32, 27 |
| `src/analysis.rs` (`solid_skin`, `edge_held`) | skin of the solids, supports read along edges | 28, 34 section 5 |

The documents of the crate are the derivations and the measurements; this document does not repeat them. Read
document 34 (shells and lines) and document 35 (solids) before starting.

Rules of this repository that apply to the port (CLAUDE.md, LEGAL.md): neutral card names everywhere; the methods
cited by their publications (section 16), never by a program that implements them; English; plain C++20 readable by
an intermediate developer; the module layering.

## 3. The formulation in brief

k is the polynomial degree of the shell displacement, fixed at 4 (the lowest degree that passes the validation of the
reference; section 15, decision 7), p = k − 1. Each quadrature has 2k + 4 Gauss points per direction in a shell element
and on its edges, and d + 4 in a solid (d the degree of its stresses). The integrands of an element that is not affine
are rational: these rules converge on them exponentially.

### 3.1 Shells

The shell of Neunteufel and Schöberl in its TDNNS form [1-3, 6]. The displacement u is continuous (H1). The
transverse shear γ is a tangent field with tangential continuity. The bending moments m lie in the
Hellan-Herrmann-Johnson space, are normal-normal continuous and are eliminated in each element. The rotation about
each edge is the hybrid unknown. The membrane strains are interpolated in the Regge space, which removes membrane
locking.

| Unknown | Triangle | Quadrilateral | Shared |
|---|---|---|---|
| u, three components | P_k / W | Q_k / W | grid values, edge moments 3 (k − 1) per edge, interior |
| γ | P_p² | Q_(p,p+1) × Q_(p+1,p) | k per edge (shared between two elements, held at junctions), interior |
| edge rotation a | P_p per edge | P_p per edge | k per edge, one value for all the elements of the edge |
| m | P_p symmetric | quadrilateral HHJ space | eliminated in the element |
| membrane strain | P_p symmetric | Regge space of the quadrilateral | interpolated, no unknown |

- Geometry: the surface of an element is a rational patch X = N / W read from the mesh (section 4.4). W is 1 at the
  corners; on an edge, W is the denominator of that edge's conic, the same from both elements. The displacement is
  P / W with P in Q_k or P_k (rational isoparametric). Cylinders and spheres meshed along their parallels and
  meridians are then exact, and rigid motions are in the space.
- Constant states: on an element that is not affine, the three reference constants of the Regge and HHJ spaces are
  replaced by the constant states of a frame of the element. The frame is carried by minimal rotation (document 34,
  section 4.3); without this the patch tests fail on distorted elements.
- Element stiffness: K = K_0 + (G + C)ᵀ M_σ⁻¹ (G + C), with M_σ the compliance on the moment space, G the hybrid
  pairing of the moments with the curvature, and C the membrane-bending coupling, partially mixed (document 34,
  section 3.2). M_σ is equilibrated and factorised: K = Yᵀ Y with Y = L⁻¹ S (G + C).
- Joints: at an edge of one element γ · t is free unless supported. At an edge of two elements,
  γ_L · t_L + γ_R · t_R = 0. At an edge of three or more elements, γ · t = 0 for each. These rules do not depend on
  the orientation of the elements.
- Sections without transverse shear flexibility (the third material blank): γ = 0 exactly (the HHJ plate), no
  penalty.
- Offsets: the unknowns stay on the surface of the grid points; the section acts on the strains of the offset
  surface (document 34, section 2.3).
- Mass: ½ ∫ (I0 |u̇|² + 2 I1 u̇ · β̇ + I2 |β̇|²) dA, consistent.
- Geometric stiffness: the second variation of the nonlinear energy at u = 0 with the stresses of the preload. It
  includes the membrane forces, the moments in the elements and the moments at the kinks (folds).

### 3.2 Line elements

Document 34, section 8.1.

- On an edge of shells, a line element has no displacement of its own. Its axis is the trace of the first shell of
  the edge, and its rotation is θ = a_T t + t × du/ds (the edge rotation of the shells, and the slope of the trace).
- Alone on an edge (no shell has it), the line carries the unknowns of that edge. Its displacement is linear between
  its two grid points plus k − 1 functions s (1 − s) P_l(2s − 1) per component. Its twist ψ is a polynomial of degree
  p (k unknowns).
- The moments of the line are a vector polynomial of degree k, eliminated on the edge. At each end, the rotation Λ of
  the lines at the grid point is the hybrid unknown; these are the rotation components of the grid point
  (section 5.4).
- Shear flexibility enters through the complementary energy of V = t × dM/ds. The axial strain stays primal.
- Offsets of the axis (end offsets, neutral axis) enter the section law. On a curved edge the law is TᵀDT / μ, with
  μ = |t + do/ds|, and the mass is μ times that of the edge.
- Mass: the translation of the axis and the rotary inertia of the section. Geometric stiffness: from the axial force.

### 3.3 Solids

Document 35, and document 28 for the principle.

- The six stress components are unknowns of the element, of degree d = 4 in the space of the shape. They are put in
  equilibrium weakly against interior fields (bubbles) and eliminated (hybrid stress element in the sense of Pian [8]).
- The displacement of the boundary has degree 4. It is carried by three translations per grid point and by functions
  shared on edges and faces, written from the grid point IDs so that every element of an edge or face sees the same
  functions.

| Shape | Space of degree n | Bubble factor | Boundary fields, n = 4 |
|---|---|---|---|
| tetrahedron | P_n | 256 L0 L1 L2 L3 | 102 |
| pyramid | rational space of Bergot, Cohen and Duruflé [7] | 4 q² (1 − a²)(1 − b²)(1 + z) | 150 |
| pentahedron | P_n(triangle) ⊗ P_n(z) | 27 L0 L1 L2 (1 − z²) | 198 |
| hexahedron | Q_n | (1 − x²)(1 − y²)(1 − z²) | 294 |

- Every space has P_n on its triangular faces and Q_n on its quadrilateral faces: two elements of any shapes that
  share a face are conforming there.
- Functions of an element: the functions of its space whose traces are the face functions, taken level by level
  (each in the smallest space holding its traces, of least norm there; document 35, section 3.2). This makes a
  straight element exactly the map of its corner grid points.
- Modal analyses: each element keeps three interior fields as unknowns, the answer of its interior to a uniform
  acceleration (document 28, section 3.5). Static and buckling analyses do not have them.
- Geometric stiffness: ∫ σ : ∇u ∇v on the boundary fields with the bubbles eliminated.

### 3.4 Ties

Where the groups meet, some unknowns follow others (document 34, section 8.2; document 35, section 3.5). A tie is a
linear relation: dependent unknown = combination of independent unknowns.

1. An edge of the solids that is an edge of shells: its functions follow the trace of the shell, by the L2 projection
   of that trace less the linear part of its two grid points.
2. A face of the solids covered by a shell (a triangle, a quadrilateral, or half of a quadrilateral whose diagonal is
   an edge of the solids): its functions follow the displacement of the shell, by the same projection less what the
   grid points and the edges carry.
3. A line alone on an edge of the solids: its displacement functions follow those of the solid edge.
4. The rotation about an edge of shells or of lines alone that solids have, where no shell of the edge lies on a face
   of the solids (a fin standing on a block, a beam along its edge): the edge rotation unknowns follow half the curl
   of the solids, averaged over the solids of the edge.
5. At a grid point of line elements, the rotation of the lines in a direction where they have no stiffness (a rod; a
   bar bending in one plane) is written on the other components.
6. A line alone that ends at a grid point of shells where no line lies on a shell edge: its rotation Λ is the mean
   rotation of the shells at their corners there.

A line alone that ends at a single grid point of a solid is pinned there: its Λ is its own.

### 3.5 Unknowns at k = 4

| Where | Unknowns |
|---|---|
| grid point | 3 translations; 3 rotations where line elements are (otherwise section 5.4) |
| shell edge | 9 displacement moments, 4 shear, 4 rotation |
| shell quadrilateral interior | 51 |
| line-alone edge | 9 displacement, 4 twist |
| solid edge | 9 |
| solid face | 9 (triangle), 27 (quadrilateral) |
| solid interior fields (modal only) | 3 per element |

A large regular quadrilateral mesh has 37 unknowns per element once the shell interiors are eliminated, and 88 with
them.

### 3.6 Known limits of the reference

These are properties of the formulation; the port reproduces them and does not try to fix them.

- The validation campaign of the reference passes 1533 of 1552 counted rows (document 34, section 11). The failures
  are of four kinds:
  - free edges of twisting plates at coarse meshes (the Reissner-Mindlin layer);
  - kinks of 1.7e-3 rad between triangles read on a sphere, which this shell takes as folds;
  - imposed curvatures on irregular meshes whose inner edges are read smooth;
  - the membrane force at the centre of unsymmetric laminates at 4 × 4 elements.
- The geometry must be exact, or G1, where the shell is smooth: any kink between two elements is a fold for this
  shell (section 4.4).
- Two shell triangles on a quadrilateral face of a solid are tied to it by their grid points and edges only. A line
  alone on an edge with a mid-side node is straight.
- A rotation held at an isolated grid point with no line element there acts only through the edges that hold it
  (section 5.6).
- The offset of the axis of a line element is in its section law (the axial strain of the axis coupled to the
  curvatures), not a rigid arm: the torsion and the shear stay about the edge. A transverse force across the offset
  twists a `ClassicFormulation` bar through its rigid arm and not a bar of this formulation; in the plane of the
  offset the two agree exactly.

## 4. Input

### 4.1 Shells

| Field | Use |
|---|---|
| G1-G4 | corner grid points; the edges are the pairs of consecutive corners |
| PID → ShellProp, CompProp, CompPropGlobal | the section through `shells/ShellSection`: at a thickness t, A = t·G1, D = (12I/T³)·t³/12·G2, B = t²·coupling, transverse shear κG·t along the material axes, thermal resultants, density, NSM. This gives the reference's section point (`SectionPoint`: A, B, D, H, I0, I1, I2, thermal terms) |
| T1-T4 | thickness at the corners, interpolated at each quadrature point |
| THETA, MCID | the material direction, as `EdgeShearShellFamily` builds it (THETA from G1→G2, MCID projected), handed over as a direction vector in the tangent plane |
| ZOFFS | the offset of the section (section 3.1) |
| MID3 blank | no transverse shear flexibility: the section is a Kirchhoff section |
| MID2 blank (membrane only) | the Regge membrane strains alone: no moments, no shear, no edge rotation; the element adds no shear or rotation unknown to its edges (section 15, decision 8; not in the reference) |
| MID1 blank (bending only) | the moments and shear alone, the membrane stiffness zero; the in-plane unknowns that no other element stiffens are removed as singular (section 5.8; not in the reference) |
| Z1, Z2 | fibre distances of the stress output |
| G5-G8 (`ShellQuad8`), G4-G6 (`ShellTria6`) | each mid-side node given fixes its edge as the parabola through the two corners and it, as a mid-side node of a solid does (document 35, section 3.4); a partial set is accepted. The mid-side node carries no unknown: its translations follow the edge (an observer relation where something uses them, else filled after the solution, as the rotations of section 5.4) |
| T1-T4 of the drilling and higher-order cards | as T1-T4 of `ShellQuad4` (corner thicknesses) |

### 4.2 Line elements

| Card, field | Use |
|---|---|
| Rod2 (RodProp A, J), Rod2InlineSection, Tube2 | rod section: axial and torsion |
| Bar2 (BarProp A, I1, I2, I12, J) | bar section |
| BarProp K1, K2 | shear flexibility (the complementary energy of V) |
| Beam2 (BeamProp stations, taper, N1-N2 and M1-M2 offsets) | beam section, its law varying along the element |
| X1-X3, G0 | orientation of the section. It is needed alone; on a shell edge the normal of the first shell is taken when no orientation is given (`section_axes`) |
| WA, WB, OFFT | end offsets of the axis |
| PA, PB (pin flags) | on a line alone: the released components of an end take an unknown of their own (an entity of the end) instead of the grid point's; on a line on a shell edge, refused (`unsupported-field`): the line has no displacement of its own there (not in the reference) |
| BarProp C, D, E, F points | stress recovery points (section 10) |
| BeamShapeProp, warping (SA, SB) | refused, as in `ClassicFormulation` |

### 4.3 Solids

| Field | Use |
|---|---|
| corner grid points | shape from their count (4, 5, 6, 8), in the order of the input format |
| mid-side nodes | each given one fixes its edge as the parabola through it; a partial set is accepted (unlike `ClassicFormulation`) |
| SolidProp CORDM | material axes of the stresses and of an anisotropic law |
| SolidProp IN, ISOP, STRESS, FCTN | not used; an informational diagnostic when given |
| IsoMat, AnisoMat3D | the 6 × 6 law (`core/MaterialMatrices`), thermal expansion |

### 4.4 Declarations of the geometry

The surface of the shells is read from the mesh by the reading of the reference: smooth and raw edges, rational
patches, normals fitted at the grid points (documents 25 and 26 of the crate). The skin of the solids is read the same
way, across their sharp edges. The reading has no angle criterion, by the owner's choice: the model declares where
the surface is not smooth.

| Declaration | Meaning |
|---|---|
| raw edge (two grid points) | a straight edge along which the surface may fold; the mesh lines that cross it end there |
| raw element | its edges are raw: a facet |
| sharp edge of a solid skin (two grid points) | the skin is not smooth across it (edges of a block, rim of a bore); the edge follows its own line and may be curved |

The input format has no card for these declarations yet. Phase 2 takes them from the program, as the formulation
choice is taken:

- `SnapshotBuilder::setFormulationGeometry` and `api::ModelInput`;
- in JavaScript, `modelFromReader(oor, deck, { formulationGeometry: { rawEdges: [[g1, g2], ...], rawElements: [...],
  sharpEdges: [[g1, g2], ...] } })`, with grid point and element IDs;
- the declarations enter the snapshot's identity, as the formulation choice does.

The declarations come from the program only; the reader's data contract does not change (section 15, decision 3).

The declarations are mandatory (section 15, decision 2): a model with shells or solids in the new formulation and no
declarations is refused (`geometry-declarations-missing`). An empty declaration (`formulationGeometry: {}`) is
accepted and says that every edge is smooth, the reference's convention; a fold that is not declared is then
rounded. The geometry report (section 10.5) shows the largest kink and the largest rounding per edge. A declaration
that names an edge or element that is not in the model is refused (`geometry-declaration-unknown`).

### 4.5 Choosing the formulation

The formulation is chosen for the whole model by one switch:

- the model parameter `ElementFormulation`, such as `Parameter;ElementFormulation;MixedHybridFormulation`, in the
  model section or global case section (a per-subcase choice is refused, as for `<card>Formulation`);
- or the program: `SnapshotBuilder::setModelFormulation`, `api::ModelInput::setModelFormulation`, the
  `elementFormulation` option of `modelFromReader`, the program winning over the parameter.

The switch selects the new formulation for every card of section 1.1. A per-card choice of `MixedHybridFormulation`
(`ShellQuad4Formulation`, `elementFormulations`) is refused (`formulation-per-card`): the formulation is never chosen
for one card alone. While the switch is on, a per-card choice of another formulation for a card of section 1.1 is
refused (`formulation-mix`); the cards outside section 1.1 keep their per-card choices. `ClassicFormulation` as the
value of the switch is the default. Unknown names are refused as today (`unknown-formulation`).

## 5. Changes to the solver's architecture

### 5.1 Unknowns that are not grid points

Today every degree of freedom (DOF) belongs to a grid point (six) or a scalar point (one) (`ElementDof`, `DofMap`). The
new formulation has unknowns on edges, faces and element interiors, shared by every element around them. They become
a third kind of node, the entity:

- `input::PointKind::Entity`: "an unknown of an element formulation; it has no ID and no frame";
- `ElementDof::entity(row, index)`, where `row` is the entity's number and `index` the DOF within it (0 to 254: the
  node DOF counts are `std::uint8_t`; the largest entity has 51);
- `DofMap(nodes, points, layout)`: the entities are numbered after the scalar points, with their DOF counts from the
  layout (section 5.2). `gSetSize`, `nodeCount`, `nodeDofCounts`, `dofOf`, `labelOf` and `nodeDofOf` take them
  in. `findDof` and `pointIdOf` never return one: they have no ID;
- `NodalDisplacements::entity`: one value per entity DOF, read by the recovery and differential stiffness facets
  (`valueAt`);
- `DofLabel` of an entity DOF: kind `Entity`, its row and index, for the diagnostics. Diagnostics name the entity by
  what it is (edge between grid points 12 and 15, face of grid points ..., interior of element 7).

Every place that handles `PointKind::Scalar` today, about 30 files (search `PointKind::Scalar`, `scalarCount()`,
`nodeDofCounts(`), is reviewed. Those on the full path take entities. Those on the refused paths (condensed and
streamed statics, superelements, design sensitivities) refuse a DofMap that has entities, with a diagnostic, until
phase 6.

A model without entities is numbered exactly as today: the layout is empty, no offset changes, and the node graph,
ordering and factor are bit for bit those of today.

### 5.2 The formulation model

A new folder `src/elements/mixedHybrid/` (headers in `include/oos/elements/mixedHybrid/`) holds what the families
share. It is built once per snapshot through `SnapshotIndex::computedOnce<MixedHybridModel>`, which the shared
directors of the higher-order shells already use:

| Part | Content |
|---|---|
| topology | the elements of every card in the new formulation; edges as sorted pairs of grid point IDs, faces as sorted sets; the owners of each edge and face; the edge kinds (shell edge with one, two or more elements; line-alone edge; solid edge) |
| geometry | the reading of section 4.4: the curve of each edge, the patch of each shell element, the skin of the solids, the mid-side parabolas, in the priority of document 35 section 3.4 |
| ties | the relations of section 3.4, as dependent entity DOF = Σ coefficient × independent DOF |
| reference data | the reference spaces and dual bases per shape and degree (built once, immutable) |
| diagnostics | every topological problem, attached to the element it concerns, reported by the family of that element's card in `check()` |

The entity layout (the list of entities with their DOF counts) depends on the analysis:

| Layout | Shell interiors | Solid interior fields | Used by |
|---|---|---|---|
| static | kept (section 15, decision 6) | none | SOL 101, the preload and the eigenproblem of SOL 105, SOL 144, inertia relief |
| modal | kept | 3 per element | SOL 103 |

The solution passes the layout kind to `DofMap` and to the batch views. `ElementBatchView` gains the layout (a
pointer, null for every other formulation), so that the stiffness facet of a solid knows whether to emit the
interior fields. A view without a layout, given to a family of the new formulation, is an error the family reports.

Entities are numbered in a deterministic order (kind, then sorted grid point IDs, then element ID): the bits of a run
must not depend on the order of the rows. The canonical orientation of an edge or a face (the grid point it is written
from) is taken from the grid point IDs, so that the results do not depend on the numbering of the elements or on the
grid point order of an element (invariance tests, section 13.4).

### 5.3 Element families

One family per card of section 1.1, in the card's folder as the registry asks (`shells/shellQuad4/`
`ShellQuad4MixedHybridElement`, and so on), registered after the card's default in `ElementRegistry::standard()`.
The families are thin: they read their batch, look the elements up in the model of section 5.2, and call the kernels
of `mixedHybrid/`:

| Subfolder | Kernel |
|---|---|
| `mixedHybrid/core/` | polynomial bases, quadrature rules (Gauss, collapsed Gauss-Jacobi, triangle, tetrahedron), the value-gradient-Hessian scalar, cyclic Jacobi eigen solver, equilibrated Cholesky |
| `mixedHybrid/geometry/` | mesh reading, rational patches, edge curves |
| `mixedHybrid/shell/` | spaces, element, loads, resultants, geometric stiffness |
| `mixedHybrid/line/` | kinematics, mixed hybrid core, section state |
| `mixedHybrid/solid/` | spaces, face and edge functions, element functions, kernel |
| `mixedHybrid/model/` | topology, layout, ties, supports, rotations at grid points, rigid fields |

The facets offered: stiffness, mass, differential stiffness, loads (pressure, line, thermal), recovery. They do not
offer derivatives, and the design module refuses the formulation (phase 6). The element matrices are emitted over
their own DOFs (grid translations, grid rotations where the element uses them, entity DOFs), in the displacement
systems of the grid points as every family does: the kernel works in the basic system and the family rotates the
grid rows by CD.

### 5.4 Rotations at grid points

The shells of this formulation have no rotation at their grid points, and the solids have none. The six DOFs of a
grid point stay, with this meaning:

| Grid point | Rotation components |
|---|---|
| with line elements of the new formulation (on shell edges or alone), except the next row | independent: the rotation Λ of the lines; the directions without line stiffness tied (section 3.4, tie 5) |
| with shells and lines alone only, no line on a shell edge | dependent: the mean rotation of the shells at their corners (tie 6) |
| with shells only, or shells and solids | the mean rotation of the shells at their corners, by an observer relation where something else uses it, else excluded and filled after the solution (below) |
| with solids only | unused: no stiffness, removed by the automatic constraint of singular DOFs, as for the solids of `ClassicFormulation` |

The rotation of a shell at a corner is θ = ν × β + ν w, with w = ½ (∂1u · e2 − ∂2u · e1): it is exact for a rigid
motion. The mean is over the shell elements of the grid point, with equal weights (`fill_rotations` of the
reference).

**Observer relations.** A rotation component of a shell grid point is made dependent on the shell unknowns (an
observer relation, section 5.5) when something other than the formulation uses it:

- a load on it in any subcase (`PointMomentLoad`);
- an element of another card or formulation with stiffness or mass on it (a spring, a bushing, a point mass, a
  `ClassicFormulation` bar);
- a relation that names it as an independent DOF.

The couple of a point moment then acts on the mean corner rotation, as in the reference (document 34, section 6). A
spring sees the same rotation.

**Excluded rotations.** Every other rotation component of a shell grid point is excluded from the f set. It goes in a
new exclusive DOF set, `x` ("formulation-excluded"), which is not a support: it is not reported among the
constrained DOFs or their forces. After each solution, and for each eigenvector, the solution asks the formulation to
fill these components with the mean corner rotation (`MixedHybridModel::fillGridRotations`), so that the displacement
tables have them. Observer relations are not used everywhere because a corner rotation depends on all the unknowns of
the element: about 1000 terms per grid point. That is too many for the m set of a large model.

**Held or dependent rotations.** A rotation component held by a support, or made dependent by a user relation, has
no observer relation:

- a held one is read along the edges (section 5.6);
- a dependent one is the relation's own and carries no shell stiffness. The shells are not tied to that relation in
  rotation; a warning names the grid point and the relation (`rotation-not-transmitted`). The edge reading of rigid
  elements that would tie them is phase 6.

### 5.5 Formulation relations

The ties of section 3.4 and the observer relations of section 5.4 are linear relations dependent = Σ c × independent,
the same form as the rigid elements, the multipoint constraints, the fasteners and the glue. They enter
`constraints/RigidElimination` as one more source, `constraints/FormulationRelations`, built per constraint set:
which observer relations exist depends on the supports and relations of the set. Then:

- `RigidElimination` resolves chains as it does today (an observer relation may have a tied DOF among its terms);
- `MsetCouplings`: an element group adds couplings for a dependent DOF of a formulation relation only when one of its
  element matrices names that DOF. The rule for the user relations is unchanged, so that models without the new
  formulation keep their pattern;
- multipoint constraint forces (`computeMpcForces`) are not reported for formulation relations;
- a relation of the user that makes dependent a DOF the formulation already ties is refused (`dependent-twice`), as
  today.

### 5.6 Supports

A support is stated at grid points; the formulation reads it along the edges and faces (document 34, section 5;
document 35). For each constraint set the formulation derives supports of entity DOFs, added to the s set by
`SpcSet` as derived DOFs, with their values:

| Held at both grid points of an external shell edge | Entity DOFs held |
|---|---|
| translation along a direction | the displacement moments of the edge along it |
| rotation about the edge axis (t in the span of the held rotation axes) | the edge rotation |
| rotation about the conormal μ, with the normal translation held | the shear γ · t of the edge |

- Solids: an edge of the skin whose two grid points hold a translation direction holds its edge functions along it;
  a face whose grid points all hold it holds its face functions.
- Grid points with line elements hold Λ as stated.
- An entity DOF that follows another through a tie is held through it, never directly.
- Enforced values (`EnforcedMotionConstr`) are carried along the edge as by a rod: linear along the chord, cubic
  Hermite across it with the rotations held at both grid points. The edge rotation takes W² θ · X_τ with θ linear
  along the chord.

Directions. The reference states supports in global axes. Here the held components are in each grid point's CD
system. An edge holds a direction when the held directions at its two grid points contain it: the intersection of
the two spans of held CD axes, in the basic system. When the two CD systems give parallel axes (CD 0, or a common
rectangular CD) this is the reference's rule. When they do not (a cylindrical CD along a circle), phase 2 holds only
the common directions and warns (`support-not-carried-along-edge`). Phase 6 adds skew supports as relations
(u(s) · d(s) = value on the edge moments).

Reactions. The forces of held entity DOFs are added to the grid points of their edge or face by the transpose of the
carrying map above. That map is exact for rigid motions, so the resultant of the reported reactions equals the
applied loads, as every support force table must.

### 5.7 Rigid and affine fields

Several parts of the solver build a field over the g set from the positions of the grid points:

- the inertial loads (`loads/InertialLoads`: GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad as
  Mgg × a);
- the rigid-body modes D (`relief/RigidBodyModes`);
- the grid point weight generator;
- inertia relief;
- the effective masses of SOL 103.

On entity DOFs these fields are not zero: a uniform translation has edge coefficients on a curved edge (P = c W), and
a rotation has edge rotation values. The formulation therefore gives the field of every affine displacement
u(x) = b + A x on its entity DOFs (`MixedHybridModel::affineField(A, b, out)`), with the rotation unknowns taken from
the skew part of A. Every builder of such a field calls it after filling the grid rows. Tests: Mgg × (rigid
translation) gives the total mass on each axis, and D gives zero strain energy (section 13.4).

### 5.8 Singularities

- Entity DOFs do not enter the automatic constraint of singular DOFs: the formulation has no singular entity DOF
  except through a real mechanism, which the factorisation and `MechanismShapes` report. The one exception is the
  bending-only shell (MID1 blank, section 4.1): the formulation itself holds the in-plane entity DOFs that no
  element stiffens, as derived supports of zero value.
- The grid rotations of section 5.4 follow its table; the excluded set x is outside the free set before the
  singularity processor runs.

### 5.9 Refusals

Until phase 6 lifts them, a model with the new formulation is refused, with a diagnostic naming the reason, by:

- the condensed path, the streamed path, components, superelements, the condensation cache: `PathChoice` keeps the
  full path, and an explicit request for another path is refused (lifted in phase 6);
- the design and optimization interfaces (lifted in phase 6);
- (not refused in the implementation: a fastener, weld or glued contact on an element of the new formulation reads
  the translations of its corner grid points, section 12; the facet of section 5.10 would read every DOF of the
  element).

These stay refused after phase 6 (section 14, later work):

- grid point force balance output;
- a rigid element or a multipoint constraint whose dependent DOF is a rotation of a shell grid point
  (`rotation-not-transmitted`, section 5.4, is a warning only when the relation names it as independent);
- a following pressure in buckling.

### 5.10 Point interpolation facet

Phase 6 adds a facet that gives the displacement (and, for shells and lines, the rotation) at a point of an element
as a linear combination of its DOFs, grid and entity: `interpolateAt(view, row, parametric point, rows)`. The
piercing of fasteners and welds and the glue of glued contacts use it in place of the shape functions of
`ClassicFormulation`: the relation they write names entity DOFs, which `FormulationRelations` (section 5.5) accepts
as independent DOFs.

## 6. Kernels and numerics

- **Reference data.** Spaces, dual bases, quadrature rules and face and edge function tables are built once per shape
  and degree, immutable, shared between threads (built under a lock or at start).
- **Element functions of the solids.** The minimum-norm inverse of the trace matrix uses a cyclic Jacobi eigen solver
  with a relative threshold of 1e-12 of the largest eigenvalue. Do not substitute a general-purpose singular value or
  eigenvalue routine without testing it on the cases of document 35, section 4: the reference found two routines of a
  general library wrong on these matrices (reconstruction error 2.3e-5; eigenpairs 4e-2 off on a nearly diagonal
  matrix with a double eigenvalue). Each construction checks that the traces are reproduced (residual below 1e-8),
  and refuses the element otherwise.
- **Moment compliance M_σ** of the shells: equilibrated (diagonal scaling) before its Cholesky factorisation. On
  elongated elements its diagonal spans many decades.
- **Value-gradient-Hessian scalar** for the second variation: a small forward-mode type over the local variables of
  a point (15 on a shell point, 6 on an edge point). `shells/SecondOrderScalar` is a related type; a separate one is
  fine.
- **Determinism.** No result may depend on thread scheduling. Loops over quadrature points and assembly into an
  element matrix run in a fixed order.
- **Cost.** The reference measured about 6 ms per solid tetrahedron (stiffness, one thread). The shell element is to
  be measured in phase 2 against `ClassicFormulation` on the same model. Elements are built in parallel through
  `base::ThreadPool`, by ranges as every facet is.

## 7. Loads

| Load | Treatment |
|---|---|
| PointForceLoad, PointForceLoad1, PointForceLoad2 | on the grid translations |
| PointMomentLoad, PointMomentLoad1, PointMomentLoad2 | on Λ at grid points with lines; on the mean shell corner rotation elsewhere (observer relation, section 5.4); refused at a grid point with solids only (no rotation) |
| ShellPressureLoad, FacePressureLoad on shells | pressure facet: along the normal of the element surface, corner values interpolated on the element; with a direction (CID, N1-N3), a traction |
| FacePressureLoad on solids | pressure facet on the face named by `faceNodes` |
| PointsPressureLoad | as today (forces at the grid points) when the face is not a face of the formulation; on a shell or solid face of the formulation, through the pressure facet (phase 2 for shells, 5 for solids) |
| BeamDistributedLoad | line-load facet: on lines alone, forces and couples per length; on lines on shell edges, the edge loads of the reference (force on u, couple about the edge on the edge rotation, couple about μ on −β · t); a couple about the shell normal is refused, as in the reference |
| GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad | Mgg × a with a on entity DOFs from section 5.7; the recovery adds the particular stresses of the body force in solids (section 10.3) |
| NodeTemperatureLoad, DefaultTemperatureLoad, PlateTemperatureLoad, LineElemTemperatureLoad | thermal facet: free strains of the shells (mean and gradient), of the lines (axial and gradients), of the solids |
| AxialDeformationLoad | free axial strain of the lines (phase 4) |
| EnforcedMotionConstr | section 5.6 |

A following pressure (load stiffness in buckling) is not in phase 3. The differential stiffness facet receives the
preload displacements, not its loads; adding the loads (`ElementLoadContext`) to that facet is phase 6.

## 8. Mass and modal analysis

- The mass is the consistent mass of section 3, whatever `CoupledMass` asks: a lumped form of hierarchical and hybrid
  unknowns has no meaning (section 15, decision 4). An informational diagnostic says so when `CoupledMass` asks for
  the lumped mass.
- `WeightToMass` is applied by the assembler, as today. NonStructMass sets add mass per area or length through
  `MassOptions::addedNonStructuralMass`.
- SOL 103 uses the modal layout (section 5.2). Eigenvectors are filled as in section 5.4 before their tables are
  written.

## 9. Buckling

- The preload is solved in the static layout, the eigenproblem in the same layout: the shell interiors are kept in
  both, and the solids have no interior fields in either.
- `computeDifferentialStiffness` gives the second variation of the shells (membrane, moments, kinks) with the
  stresses recovered from the preload, the axial-force term of the lines, and the initial-stress matrix of the solids
  with their stresses (thermal part included).
- Offsets: under `OFFDEF ELMOFF` the classic families refuse an offset element in buckling; the new formulation
  carries offsets in its section law and is not refused.

## 10. Output

### 10.1 Grid point tables

- Displacements: translations as solved; rotations as solved (independent or observer), or filled (section 5.4);
  zero at grid points of solids only.
- Support forces: those of grid DOFs, plus the forces of held entity DOFs carried to the grid points (section 5.6).
- Multipoint constraint forces: user relations only.
- Entity DOFs never appear in any user table.

### 10.2 Shells

Rows of `shells/ShellRecovery` (forces, stresses at Z1 and Z2, strains and curvatures, plies and failure indices of
laminates), at the centre and at the corners, in the frame of `ClassicFormulation` (x along G1→G2 projected): the
same tables and columns, so that every client reads them unchanged. The resultants are those of the reference
(`resultants`): about the middle of an offset section.

### 10.3 Solids

Rows of `solids/SolidRecovery` (the 20 columns) at the centre and corners, in CORDM, from the stresses of the element
at those points. The particular stresses of a body force are added, so the recovery reads the subcase's
accelerations: `NodalDisplacements` gains them. The corner value of one element is 1 to 3 % from the converged value
on a regular mesh. The mean over the elements at a grid point is the accurate grid value (document 28), so a grid
point stress table is added (section 15, decision 5): for each grid point of solids of the new formulation, the mean
of the corner stresses of its solids, in the basic system, with the principal values of `solids/SolidRecovery`.

### 10.4 Line elements

The layouts of `Rod2`, `Bar2` and `Beam2` (end forces, stresses at C-F, extremes, margins), from the section forces of
the reference at the ends (`line_state`). The stresses follow from the forces as in `bars_beams/BeamResponse`.

### 10.5 Geometry report

A table (or diagnostics) per edge: the largest angle between the normals of its elements, and whether it was declared
raw or sharp (`geometry_report` of the reference). It lets the user see where the reading left kinks or rounded a
fold.

## 11. Support catalog and checks

- `addSupportTo`: the fields each family reads, with the rules of section 4 (refused fields: MID1 or MID2 blank, pin
  flags, BeamShapeProp, warping).
- `check()`: everything of section 4, the topology of section 5.2, the consistency of section 4.5, degenerate
  elements (a solid that folds: Jacobian of both signs; a shell with a zero-length edge), and the trace residual of the
  solids (section 6).

## 12. Deliberate deviations from the reference

| Deviation | Why |
|---|---|
| Observer relations and excluded rotations (section 5.4) | The reference has no other element at a shell grid point; it fills the rotations after the solution and puts point couples on the corner rotations, which is what section 5.4 does where nothing else uses them. |
| Supports in CD systems (section 5.6) | The reference has global axes only. |
| Consistent mass whatever `CoupledMass` asks | The reference has the consistent mass only. |
| Entity numbering | The order of unknowns changes only the round-off. |
| Element order on an edge ("the first shell of the edge") | Ascending element ID here, the order of the model in the reference. |
| Cards, properties and extensions not in the reference (section 15, decisions 8, 11, 12) | The owner's choice: every element switches together. |
| A BeamDistributedLoad on a line on a shell edge works on the kinematics of the line (θ = a_T t + t × du/ds, the offset o × f included); a couple about the shell normal is taken, on −μ · du/ds | The load is consistent with the energy of the line; the reference's edge traction puts the couple about μ on −β · t, which differs by the shear γ · t (zero for a Kirchhoff shell, small for a thin one). |
| The forces of a bar or beam on a shell edge are the section forces of the line kernel along it; the Bar2 row reads its shear forces, axial force and torque at its middle | The shell loads the line along the edge: no end forces describe the forces between the ends. A line alone keeps the end forces of its equilibrium (`r = K x − f`), exact at the ends. |
| PointsPressureLoad on a face of a shell or a solid of the formulation goes through the pressure facet (section 7), of the sign of the order of its grid points | The consistent load of the element instead of forces at the grid points. |
| Fasteners and welds pierce the shells of the formulation through the translations of the corners of the pierced element (the weighted least-squares interpolation of `fasteners/PiercingInterpolation`) | Exact for rigid motions, and the corner translations are unknowns of the formulation; the point interpolation facet of section 5.10 is not written yet. |
| Temperatures of a solid: the corner changes interpolated by the functions of the corners of the straight element at the point nearest to each point of the kernel; an expansion that differs between directions is refused under a temperature load | The kernel of the reference takes one expansion coefficient and a temperature field of the position. |
| A pressure along a direction on a solid face is taken when it is the same at every corner of the face | The kernel's traction field is uniform there. |

Implementation notes. The rows of the element tables are those of section 10; the grid point stress table of section
10.3 has the element type `SolidGridPoints` (one row per grid point, its ID as element and place, the columns of the
stress rows of the solids, basic system). The geometry report of section 10.5 is a set of information diagnostics:
`geometry-report` (the counts and the largest kink and rounding), `geometry-kink` (an edge whose normals differ by
more than 1°) and `geometry-rounding` (an edge along which the surface read turns by more than 15° from the plane of
the corners of a shell: a fold that is not declared raw). A moment at a grid point of solids alone is refused
(`load-unsupported`); an element or relation using such a rotation is warned (`rotation-held`).

## 13. Verification

### 13.1 Reference data

No exporter is written in the crate (section 15, decision 9): the port is not compared with the reference number by
number. It is verified by the analytic checks of sections 13.2 to 13.4, against the closed-form solutions and the
classical benchmarks the documents of the crate use, with the criteria of section 13.6. The ratios the documents of
the crate measure on those benchmarks are the expected values: the port reproduces them within the criteria.

### 13.2 Element checks

On single elements of every shape, flat, curved and distorted, in the native build (a test executable under a build
option, as the threading checks are, because they need no reader): symmetry of the stiffness, mass and differential
stiffness; the number of zero eigenvalues of the stiffness (the rigid motions only); constant states reproduced; the
reference data (dual bases, trace reproduction of the solids) checked at construction.

### 13.3 Model checks

Benchmarks through the whole path (reader, `modelFromReader`, run), against their closed-form or classical values:
plates (simply supported and clamped, thin and thick), the Scordelis-Lo roof, the pinched cylinder, the pinched
hemisphere, the twisted beam, a box beam with folds, laminates, offsets, line elements alone and on shells, solids of
every shape with and without mid-side nodes, shells and lines on solids, modes, buckling, inertia relief. Slow cases
go to a separate suite, not to the fast suite of `wasm/test` with its one-second limit per case.

### 13.4 Checks of this repository

In `wasm/test`, tiny models within one second per case:

- rigid motions of every element shape, curved and distorted: strain energy at most 1e-8 of the energy of a unit
  strain;
- constant states (patch tests) of shells on distorted meshes, of solids of every shape and of mixed meshes:
  displacements and stresses to 1e-6;
- invariance under renumbering of the grid points and of the elements, and under the corner an element starts from:
  1e-8;
- Mgg × rigid translation equals the mass; the rigid-body modes have zero energy;
- a cantilever of one line element alone: exact;
- the support forces balance the loads.

### 13.5 Non-regression

- Every existing test passes unchanged.
- A model that selects no new formulation is numbered, assembled and solved bit for bit as before (compare the
  results of the existing suites before and after each phase of section 5).
- `ELEMENT_FORMULATION_VERSION` does not change: no matrix of an existing formulation changes. The condensation cache
  is refused with the new formulation until phase 6.

### 13.6 Criteria

The criteria of the reference's campaign apply to the analytical checks and are fixed before the runs: 1e-8
invariants, 1e-6 exact states, 0.5 % displacements, frequencies and critical loads, 1 % stresses, at four elements per
half-wave. They are not loosened to make a check pass.

## 14. Phases

Each phase ends with the full test suite passing and its own checks of section 13.

| Phase | Content | Exit |
|---|---|---|
| 1 | infrastructure: entity kind, `DofMap` with a layout, `NodalDisplacements::entity`, the excluded set x, `FormulationRelations` in `RigidElimination` with the `MsetCouplings` rule, derived supports in `SpcSet`, the affine-field hook, the refusals, the global switch and its checks, the geometry declarations in `SnapshotBuilder`, `ModelInput` and the JavaScript option | existing suites bit for bit; a test formulation with one entity per element (a hand-made two-DOF bubble on `Rod2`) checks numbering, assembly, relations, supports and output |
| 2 | shells, SOL 101: reference data, mesh reading, the element, the drilling and higher-order cards with their mid-side nodes, supports, loads, recovery, geometry report | element and model checks of shells (sections 13.2 and 13.3); section 13.4 for shells |
| 3 | shells, SOL 103, 105, 144 and inertia relief: mass, modal layout, second variation, affine fields in every builder | the same, for modes, buckling and trim |
| 4 | line elements on shell edges and alone, Λ, tie 5 and 6, line loads | the same, for lines |
| 5 | solids of every shape, mid-side nodes, skin reading, ties 1 to 4, solid loads and recovery, the grid point stress table | the same, for solids and mixed models |
| 6 | the extensions the owner chose (section 15): point interpolation facet (fasteners, welds, glue), pin flags of lines alone, membrane-only and bending-only shells, condensed and streamed paths, components, superelements and the condensation cache, design derivatives | each with its own checks |
| later | not chosen yet: edge reading of rigid elements, skew supports, loads in the differential stiffness facet (following pressure), grid point force balance | |

## 15. Decisions of the owner (15 November 2026)

1. **Name of the formulation.** `MixedHybridFormulation`.
2. **Model without geometry declarations.** Refused: the declarations are mandatory for a model with shells or solids
   in the new formulation; an empty declaration says every edge is smooth (section 4.4).
3. **Where the declarations come from.** From the program only (`SnapshotBuilder`, `ModelInput`, the
   `formulationGeometry` option of `modelFromReader`). No card in the reader's data contract.
4. **Mass under `CoupledMass` lumped.** The consistent mass, with an informational diagnostic.
5. **Grid point stresses of the solids.** The element rows, plus a table of the means at the grid points (section
   10.3).
6. **Shell interiors in statics.** Kept as entity DOFs (one layout for statics and buckling; the global factorisation
   eliminates them first). No condensation in the element.
7. **Degree.** k = 4 fixed, no parameter.
8. **Membrane-only and bending-only shell properties** (MID2 or MID1 blank). Accepted, as new work (section 4.1,
   phase 6).
9. **Reference data.** No exporter in the crate; the port is verified by the analytic checks of section 13.
10. **Choice of the formulation.** One switch for the whole model, not a choice per card (section 4.5).
11. **Cards without a counterpart in the reference.** `ShellQuad4Drilling` and `ShellTria3Drilling` take the shell of
    the new formulation; `ShellQuad8` and `ShellTria6` too, their mid-side nodes fixing the curves of their edges;
    `ShearPanel4` is refused (section 1.1).
12. **Extensions.** Included in this work (phase 6): the point interpolation facet for fasteners, welds and glue; pin
    flags of lines alone; membrane-only and bending-only shells; the condensed and streamed paths, components,
    superelements, the condensation cache and the design derivatives.
13. **Scope of the branch.** Phases 1 to 6 in one go.

## 16. References

1. M. Neunteufel, J. Schöberl, The Hellan-Herrmann-Johnson method for nonlinear shells, Computers & Structures 225
   (2019) 106109.
2. M. Neunteufel, J. Schöberl, Avoiding membrane locking with Regge interpolation, Computer Methods in Applied
   Mechanics and Engineering 373 (2021) 113524.
3. M. Neunteufel, J. Schöberl, The Hellan-Herrmann-Johnson and TDNNS methods for linear and nonlinear shells,
   Computers & Structures 305 (2024) 107543.
4. M. I. Comodi, The Hellan-Herrmann-Johnson method: some new error estimates and postprocessing, Mathematics of
   Computation 52 (1989) 17-29.
5. S. H. Christiansen, On the linearization of Regge calculus, Numerische Mathematik 119 (2011) 613-640.
6. A. Pechstein, J. Schöberl, The TDNNS method for Reissner-Mindlin plates, Numerische Mathematik 137 (2017) 713-740.
7. M. Bergot, G. Cohen, M. Duruflé, Higher-order finite elements for hybrid meshes using new nodal pyramidal
   elements, Journal of Scientific Computing 42 (2010) 345-381.
8. T. H. H. Pian, Derivation of element stiffness matrices by assumed stress distributions, AIAA Journal 2 (1964)
   1333-1336.
