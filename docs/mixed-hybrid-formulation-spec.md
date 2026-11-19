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
- A rotation held at one grid point of shells with no line element there is no unknown of the shells: it is read
  along the edges whose two grid points hold it, and elsewhere tied to the mean rotation of the shells at the grid
  point (section 5.6; not in the reference, which reads it along the edges only).
- Measured by the validation suite of this repository (section 13.3, `wasm/test/validation/known-limits.mjs`), and
  reproduced by the reference on the same meshes:
  - distorted triangles on a curved surface keep kinks of 1 to 3 degrees however fine the mesh, which the shell takes
    as folds: the Scordelis-Lo roof is 30 % off and a pressurised cylinder up to 50 %, without converging. Such a
    mesh is warned (`geometry-kink`, section 10.5); its triangles are declared raw, or given mid-side nodes, or
    replaced by quadrilaterals, which read the same surfaces to 0.3 % when distorted;
  - a corner of the outline of the shells where a triangle meets another element is read as an outline that goes on:
    its two edges are rounded in the surface, and a triangle left without an angle there is refused
    (`degenerate-geometry`); the edges of the corner are declared raw (warned by `geometry-rounding`);
  - on a skin of solids, a mesh line of three grid points on a flat face is read as a curve when its middle grid
    point is off-centre: faces two cells wide lose the exactness of the linear stress states (1e-6 to 4e-6) and a
    strongly zigzagging line is refused as a fold; three cells or more per side, mid-side nodes or sharp edges
    avoid it;
  - a shell that continues solids, or stands on them (tie 4), passes its moment to them along a line: the solids are
    not told the thickness of the shell, and the joint softens as the solids are refined (a cantilever continued by
    solids deflects 1.26 to 3.8 times too much); a soft shell on stiff solids is exact;
  - an inertia, a spring or a moment at a grid point of shells acts on their mean rotation there: the answer to a
    concentrated moment is 1 to 2.5 % softer or stiffer than a fine `ClassicFormulation` model, whose own answer
    depends on its mesh;
  - the fibre stresses of a curved membrane carry the term of order t / R of the curvature measure with the sign
    opposite to the thick-wall solution (0.5 % at R / t = 100);
  - the rotary inertia of a line acts on the slope of its axis, not on the rotation of its section: the higher
    bending frequencies of a short shear-flexible beam are 0.3 to 1.1 % below Timoshenko's.
- A force at a grid point of shells or solids is a concentrated force on a continuum of degree 4: the grid point
  moves more than the section around it (the displacement under a point force has no limit as the element
  resolves it), and the stresses at that grid point mean nothing. The rest of the structure is not affected, and the
  resultants are exact. It shows where a load enters through a few grid points: the reference grid point of an
  interpolation element or of a rigid spider on the end of a cylinder moves 3 % more than the section on a fine
  mesh and 6 to 16 % more on coarse ones, a corner force on one solid several times more than the uniform traction
  of the same resultant. `ClassicFormulation` elements, whose grid point forces are the consistent loads of a
  traction, do not show it. Loads meant as tractions are given as pressures or line loads; a displacement wanted at
  a load introduction is read a few elements away.
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
| MID2 blank (membrane only) | the Regge membrane strains alone (Ā = A): no moments, no transverse shear, no rotation about the edges; its shear and edge rotation unknowns that no other element stiffens are held at zero (section 5.8). Offset, the membrane bends about the surface of the grid points (B' = z0 A, D' = z0² A: its strain is e + z0 κ) as a Kirchhoff shell. Its mass has no rotary inertia (I1 = I2 = 0), its thermal load is the force of the rise (section 15, decision 8; not in the reference) |
| MID1 blank (bending only) | the moments and shear alone, A = B = 0; the in-plane directions that no element stiffens are held (section 5.8); without MID3 its Kirchhoff limit is measured on 12 D / t², the membrane stiffness of a homogeneous plate; its thermal load is the moment of the gradient (not in the reference) |
| A membrane-only or bending-only laminate (its membrane or bending option) | as the two rows above |
| two shells on the same corner grids (two properties laid over one surface) | refused: the edges would hold two shells on one side, where the reading of the surface, the supports along the edges and the continuity of the shear count one shell per side; one property with both parts is the way |
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

Two rules of the reading of the mesh lines are not in the reference (`SurfaceReconstruction::candidatesAt`):

- A mesh transition on a boundary that goes straight on. Between quadrilaterals a line goes on across one
  quadrilateral on each side (a quadrilateral is a quarter of a turn), so three quadrilaterals between two edges
  that no element faces across (the outline, or a junction seen from one of its sides) are a re-entrant corner. The
  same three quadrilaterals are what a mesh transition leaves on a straight boundary. They are told apart by the
  corners of the three elements at the grid point: under 225 degrees in all, the boundary goes on and the two lines
  inside end there. Read as a corner, a transition of a rib on its junction with a skin bent the straight junction
  towards the mesh line inside the rib (kinks of 27 degrees between coplanar shells, a skin seven times too stiff).
- A side of a junction or of the outline that has quadrilaterals only at a grid point is read as between
  quadrilaterals, whatever the other sides have. The reference takes the rule of the grid points that have a
  triangle as soon as one element at the grid point is a triangle, on any side: a re-entrant corner of three
  quadrilaterals beside a triangle of another surface was rounded (the outline going round the corner).

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
  (`valueAt`, which checks that they were given: the displacements of a g-set vector are made by
  `NodalDisplacements::ofGSet` with the layout, zero displacements included);
- `DofLabel` of an entity DOF: kind `Entity`, its row and index, for the diagnostics. Diagnostics name the entity by
  what it is (edge between grid points 12 and 15, face of grid points ..., interior of element 7): `DofMap::describe`,
  which the messages of the factorisations take through `base::GridComponent::name` (a mechanism at an unknown of an
  edge is reported as "DOF 3 of the edge between grid points 12 and 15", on every path).

Every place that handles `PointKind::Scalar` today, about 30 files (search `PointKind::Scalar`, `scalarCount()`,
`nodeDofCounts(`), is reviewed. Every path takes them since phase 6: the full path, the condensed path (section 5.11),
the streamed path (section 5.12), external units (section 5.13), superelements (section 5.14) and the design
sensitivities (section 5.15).

A model without entities is numbered exactly as today: the layout is empty, no offset changes, and the node graph,
ordering and factor are bit for bit those of today.

### 5.2 The formulation model

A new folder `src/elements/mixedHybrid/` (headers in `include/oos/elements/mixedHybrid/`) holds what the families
share. It is built once per snapshot through `SnapshotIndex::computedOnce<MixedHybridModel>`, which the shared
directors of the higher-order shells already use; a condensed analysis takes the index of its condensation planner
(`StaticOptions::snapshotIndex`, `AnalysisModelOptions::index`), so that the model and the kernels of its elements are
built once for the plan and the analysis:

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

The facets offered: stiffness, mass, differential stiffness, loads (pressure, line, thermal), recovery, and
derivatives by differences of those facets (section 5.15). The element matrices are emitted over
their own DOFs (grid translations, grid rotations where the element uses them, entity DOFs), in the displacement
systems of the grid points as every family does: the kernel works in the basic system and the family rotates the
grid rows by CD.

### 5.4 Rotations at grid points

The shells of this formulation have no rotation at their grid points, and the solids have none. The six DOFs of a
grid point stay, with this meaning:

| Grid point | Rotation components |
|---|---|
| with line elements of the new formulation (on shell edges or alone), except the next row | independent: the rotation Λ of the lines; the directions without line stiffness tied (section 3.4, tie 5), and, at a grid point of shells whose lines all lie on shell edges, made the mean rotation of the shells in those directions (a rod on a shell edge stiffens no bending rotation: a couple, a spring or a rigid element there acts on the shells; `MixedHybridModel::findUnstiffenedLineRotations`; not in the reference) |
| with shells and lines alone only, no line on a shell edge | dependent: the mean rotation of the shells at their corners (tie 6) |
| with shells only, or shells and solids | the mean rotation of the shells at their corners, by an observer relation where something else uses it, else excluded and filled after the solution (below) |
| with solids only | unused: no stiffness, removed by the automatic constraint of singular DOFs, as for the solids of `ClassicFormulation` |

The rotation of a shell at a corner is θ = ν × β + ν w, with w = ½ (∂1u · e2 − ∂2u · e1): it is exact for a rigid
motion. The mean is over the shell elements of the grid point, with equal weights (`fill_rotations` of the
reference); over those that carry moments where one does (a membrane-only shell without offset turns only with its
surface: it counts only at a grid point where no shell bends; not in the reference).

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

**Held or dependent rotations.** A rotation component held by a support, or made dependent by a user relation (a
rigid element, a multipoint constraint), has no observer relation: it is read along the edges whose two grid points
have it, and tied to the mean rotation of the shells where no edge reads it (section 5.6). A warning
(`rotation-not-transmitted`) names a grid point whose dependent rotation no unknown of the shells is free to follow.

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

| Held at both grid points of a shell edge | Entity DOFs held |
|---|---|
| translation along a direction | the displacement moments of the edge along it |
| rotation about the edge axis (t in the span of the held rotation axes) | the edge rotation |
| rotation about the conormal μ, with the normal translation held | the shear γ · t of the edge |

- The edges read: the external edges (the reference's rule), and an edge inside the shells one of whose two grid
  points is inside them too, so that a line of supports across a plate holds the plate along it (the model files
  state supports at grid points only; the reference states such supports on the edges). An inner edge between two
  grid points of the outline is not read: the diagonal of a triangle at a corner, or the edge across a strip one
  element wide, joins two supported sides, and holding it would hold the shell between them.
- Solids: an edge of the skin whose two grid points hold a translation direction holds its edge functions along it;
  a face of the skin whose grid points all hold it holds its face functions.
- Supports at two grid points of one edge hold the edge between them: three supports that would hold a body
  isostatically at grid points far apart over-constrain it when they sit on the corners of one element.
- Grid points with line elements hold Λ as stated.
- An entity DOF that follows another through a tie is held through it, never directly.
- Enforced values (`EnforcedMotionConstr`) are carried along the edge as by a rod: linear along the chord, cubic
  Hermite across it with the rotations held at both grid points. The edge rotation takes W² θ · X_τ with θ linear
  along the chord.

Directions. The reference states supports in global axes. Here the held components are in each grid point's CD
system, and an edge or a face holds the directions its grid points share:

- the basic axes that lie in the span of the held CD axes of every grid point are held as supports of the entity DOFs
  (the table above);
- a held direction d that is no basic axis and is the same at every grid point of the edge or face (a rectangular
  CD turned from the basic system, the cutting plane of a sector in a cylindrical CD: a plane of symmetry in any
  orientation) is no support of one DOF. The trace along d of each displacement function l follows the chord,
  Σ_c d_c·q_(l,c) = m_l(a)·(u_a · d) + m_l(b)·(u_b · d) on a shell edge and Σ_c d_c·q_(m,c) = 0 on the functions of
  an edge or a face of solids (`FormulationModel::supportRows`), and these rows are eliminated as relations, each
  written on the component of the function along which d is largest (`FormulationConstraints::eliminateRows`). The
  rotations held about axes that are no basic axis are read by the spans of the held axes: the edge rotation where
  the tangent lies in both spans, the shear where the conormal does;
- a direction that differs between the grid points (a cylindrical CD along a circle: the radial direction turns) is
  held by the grid points only; the edge between them is free along it, and a warning counts such edges
  (`support-not-carried-along-edge`). Holding both directions of the plane they turn in holds the edge (their span
  is the same plane at both grid points). The relation u(s) · d(s) = value along such an edge is later work
  (section 14).

Reactions. The force of a held entity DOF is reported at the grid points of its edge or face as forces and moments
with the same resultant force and moment (`DerivedSupportDof::reactionTo`, the map of the grid point force balance,
section 10.1): the reported reactions balance the applied loads in force and in moment, along a curved edge too
(the map that carries the held values along the chord is not exact for a rotation of a curved edge: it gives the
values, not the reactions).

Relations of the model. A grid component that a relation of the model makes dependent (a rigid element, a multipoint
constraint) is read along the edges and faces as a held one is: an edge whose two grid points have a component held
or dependent follows them. Its entity DOFs that the supports alone do not hold become dependent on those grid
components, by the map that carries the values of a support along the edge (`FormulationConstraints`, the relations
"that follow"): a rigid element on the grid points of the rim of a hole, or along a line of grid points, moves the
edges between them rigidly, as it moves the straight sides of `ClassicFormulation` elements. Such an edge must
move with its grid points in every rigid motion, or the relation would act as a link to the ground. It therefore
follows only when its two grid points have their three translations held or dependent, and, where it is curved (its
points off its chord), their three rotations too: its trace is then that of the chord plus θ × δ, δ the offset of
the edge from its chord and θ the rotation between its grid points (two translations do not tell the rotation about
the chord, which moves the points of a curved edge). An edge that lacks them keeps to its grid points: its own
unknowns stay free. The relations of the formulation name the independent DOFs of the model's relations in place of their dependent ones
(`FormulationConstraints::addRelation`), so that no relation of the formulation chains through a relation of the
model, which the condensed path may eliminate elsewhere.

Rotations no edge reads. A rotation of a grid point of shells that a support holds, or a relation makes dependent,
with no edge to read it (the neighbour grid points do not have it), is tied to the mean rotation of the shells there
(the rotation of section 5.4): Σ_s d_s·(W_s·u − θ_s) = 0, W_s the row of the mean rotation and θ_s the grid component
(held, with its value, or dependent). Each such row is written on one unknown of the shells (the largest coefficient
among the entity DOFs that nothing else determines and no other relation reads; the rows eliminated one after the
other), a relation as any other. The directions d:

- a dependent rotation: each of its components (a rigid element joined to the shells at one grid point turns them,
  about their normal too);
- a held rotation: the axes of the span of the held components that lie in the plane of the shells (their common
  normal, `FormulationModel::commonShellNormal`). The rotation about the normal is the rotation of the membrane field
  in its plane, which the same support leaves free on a shell without drilling stiffness: the rotation about the
  normal held at every grid point of a plate, written to remove that singularity, stiffens nothing;
- a held rotation at a fold, or at a grid point that holds its rotations alone (no grid point joined to it by an
  edge holds a rotation): each held component. The six components held at one grid point hold a plate as a free
  body.

The translations of a mid-side grid point of shells (section 4.1) that a support holds or a relation makes dependent
are tied the same way to the point of its edge.

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
  except through a real mechanism, which the factorisation and `MechanismShapes` report. The exceptions are the
  membrane-only and bending-only shells (section 4.1) and the functions across a rod alone: the formulation itself
  holds the entity DOFs that no element stiffens, as derived supports of zero value in every constraint set (a
  support that holds one with a value along an edge takes precedence). Of the entities whose shells are all
  membrane-only or bending-only (`ShellPart::unstiffenedUnknowns`):
  - the shear of an edge none of whose shells has a transverse shear, and the interior shear of a membrane-only
    shell;
  - the rotation about an edge none of whose shells carries moments, unless a line on it turns it (its diagonal
    terms there above 1e-14 of its largest: not a rod without torsion constant);
  - the singular directions of each displacement function (an edge function l, an interior function) that no solid
    follows: the eigenvalues of the 3 × 3 block of the stiffness of the shells of the entity and of the lines on the
    edge, on the three components of the function, below 1e-8 of the largest (the default threshold of the grid
    points). The s singular directions V are held by the s components C that maximise |det V_C|, as the singularity
    processor of the grid points does. They lie along those components on a flat membrane or plate in a plane of
    the basic system; otherwise the displacement of the function keeps a part along them that carries no stress,
    which an information diagnostic counts (`unstiffened-directions-held`). On a curved surface the membrane
    stiffens the normal (and the bending the tangent directions it turns): nothing is held there.
  A membrane-only shell whose structure is a mechanism stays one (an open cylinder of membranes ovalises freely, a
  stiffener on a membrane edge twists about its axis): the factorisation reports it.
- The grid rotations of section 5.4 follow its table; the excluded set x is outside the free set before the
  singularity processor runs.
- The pivot ratio test of the factorisation (matrix diagonal over factor diagonal above `MaxDiagonalRatio`) reads, at
  an entity DOF, the diagonal of the elements' own stiffness when it is below that of the reduced matrix
  (`linalg::FactorOptions::referenceDiagonal`, `FsetSolverOptions::unreducedStiffness`): a stiff spring on the
  rotation of a grid point of shells puts k·c² on the diagonal of every unknown of its observer relation, though it
  stiffens one combination of them, and the test would read the others as mechanisms. A true mechanism at an entity
  DOF is still reported. The units and the interface of the condensed path and the streamed path keep the matrix
  diagonal.

### 5.9 Refusals

Phase 6 lifted the refusals of the first phases: the condensed path with its Craig-Bampton reduction, its load groups,
its recovery in sequence and its cache (section 5.11), the streamed path (section 5.12), external units (section 5.13),
superelements (section 5.14) and the design derivatives (section 5.15) take a model of the new formulation. A
fastener, a weld or a glued contact on an element of the new formulation is not refused: it reads the element through
section 5.10.

Each of them joins the formulation's elements to something that acts on grid points (a plugin's matrices, a part
reduced on its boundary) or differences them with respect to the positions, so each refuses, with a diagnostic, what
grid points alone cannot carry: an edge or a face whose unknowns would be joined at its grid points only, a rotation
of a shell grid point that something stiffens while no stiffness of the formulation acts on it there, a shape whose
derivative would need that of the formulation's relations (sections 5.13 to 5.15).

The grid point force balance takes the formulation (section 10.1), and a rigid element or a multipoint constraint
whose dependent DOF is a rotation of a shell grid point ties the shells to it (section 5.6).

A following pressure in buckling stays out (section 14, later work): no formulation of this solver has the load
stiffness of a pressure, and the differential stiffness facet receives the preload displacements, not its loads.

### 5.10 Point interpolation facet

Phase 6 adds the motion of a point of an element as a linear combination of its DOFs, grid and entity. It is a
function of the formulation model rather than a facet of the families, since the elements that read it (fasteners,
welds, glue) know the pierced element by its ID or its face, not by a batch row:

- `FormulationModel::shellPointMotion(elementId, point, samples)`: the displacement and the rotation of the shells
  where a fastener or a weld pierces them. Each sample is a point of a shell, placed by the shape functions of its
  card that the fastener piercing table gives, Σ N_j r_j / Σ N_j, r_j the reference corner of a corner grid point
  and the middle of its edge for a mid-side grid point (the shape functions of a card reproduce the linear
  functions of its parameters: on a flat element of straight sides this is the point itself). The rotation is that
  of the shell (θ = ν × β + ω ν, section 5.4), and the translation is carried rigidly from the point X of the
  surface to the piercing point P: u_P = u(X) + θ × (P − X), exact for every rigid motion.
  The motion is the weighted mean of these over the footprint of the fastener
  (`input/tables/PiercingFootprint.hpp`, `FastenerPiercingTable::meanFootprint`): the square of the area of its
  cross-section (side (√π / 2) D, FastenerProp or WeldProp D) about the point, in the tangent plane of the shells
  there, its sides along those of the pierced shell, sampled at its 3 × 3 Gauss points. Each of them is placed on
  the shell that holds its nearest point, the pierced one or a shell up to two rings of elements around it that
  continues its sheet (so the mesh may be finer than the fastener); a point on a side shared by several shells is
  read on each of them with an equal share of its weight (the rotation of the shell is not single-valued there),
  and a point the shells do not hold (a fastener on a free edge) is left out. The reason: a point load on a shell
  is singular (the displacement in the plane under a point force and the rotation under a point moment grow
  without bound as the elements resolve them), so the motion read at the point alone depends on the mesh and on
  where the point falls in it. Measured on two plates of ShellQuad4 joined by one fastener of 4 mm (shear
  compliance, the fastener on a grid point and inside an element, on 4, 8 and 16 cells of 30, 15 and 7.5 mm): 2.23
  and 1.42, 2.70 and 1.59, 3.37 and 1.79 (×10⁻³) at the point alone; 1.63 and 1.41, 1.68 and 1.56, 1.70 and 1.66
  over the footprint (1.73 and 1.72 on 32 cells), where the two places come together as the mesh is refined. On a
  grid point of a symmetric joint the footprint lies in the elements all around and keeps the symmetry (read in one
  of them, four corners that should lift alike lifted by 1.14, 1.27, 1.27 and 1.39). Without a diameter the one
  sample is the piercing point.
  The `ClassicFormulation` ties a Fastener, and a Weld of the forms PartPatch and ElemPatch, over the same square
  by its four corners (`elements/fasteners/PiercingInterpolation`, `interpolateFootprint`), with the same reading
  of the shell: the translation of the sheet, the rotation of the shell about the axes of its plane (the rotations
  of the grid points, by the shape functions) and the rotation of the sheet in its plane (from the translations of
  the corners). On the same plates its compliance is 1.39 and 1.23 (×10⁻³) on 8 cells of 15 mm, 1.44 and 1.49 on
  32 cells, 1.68 and 1.59 on 128 cells of 0.9 mm.
- `FormulationModel::facePointMotion(faceGridRows, shapeValues)`: the translations of a point of a face of a shell
  or of a solid (the primary face of a glued contact), placed the same way on the shell or on the reference shape of
  the solid (the functions of its kernel at that point; the interior fields of a modal analysis vanish on a face).

The fastener elements (`elements/fasteners/FastenerConnection`, through `resolveSideMotion`), the rows of the GA and
GB grid points (`assembly/FastenerConstraints`) and the glue (`assembly/GlueConstraints`) use them when the
formulation has the element, in place of the shape functions of `ClassicFormulation`: their matrices and relations
name entity DOFs, which the assembly and `RigidElimination` take as any DOF. A side with no element of the
formulation (a Weld face given by grid points, an element of another card) keeps the classic interpolation, over
the translations of grid points: such a tie loads the shell at points, and gives a joint more compliant than the
same joint read over a footprint (`wasm/test/validation/fasteners.test.mjs`, sections 8 and 9).

The elements of this formulation resolve the sheets between the fasteners: a stringer fastened to a skin at the
centre of each cell is not the stringer merged to the skin, as it is with ClassicFormulation elements as large as
the pitch (which have no shape left to part from each other once tied at their centres), but what the
ClassicFormulation gives on cells several times smaller (within 4 % of cells cut in 8 × 8 with rivets; the same
suite, section 7). The
element views carry the formulation model with the entity layout (`ElementBatchView::formulation`).

### 5.11 The condensed path

The static condensation of components (SOL 101; `reduction/`, `solutions/sol101/CondensedStatics`) takes the
formulation: each unit condensed onto its boundary, the interface problem solved, the units recovered, the results
those of the full path to round-off. Exposed by `api::AnalysisSettings::condensation` and, in JavaScript, by
`modelFromReader(oor, deck, { condensation: true })` (the component tables of the reader) and
`run(model, "statics", { condensation: "all" })`; the units are the components of the reader.

Entity nodes as points. The condensed path names its points by ID. A node of an entity (section 5.1: an entity is
split into nodes of at most six DOFs) takes the ID `DofMap::entityNodeId(node − firstEntityNode())`, a negative
number that no point of the input format has (`DofMap::nodeIdOf`, `DofMap::findNode`); its DOFs are labelled
`UnitPointKind::EntityNode`, components 1 to 6. The plan graph, the units, the boundaries, `K*` and the interface
problem take them as points.

The units take the entities of their elements. The components of the reader are made of elements and grid points; the
planner gives the plan the entity nodes of each element (`reduction::ElementEntityNodes`, from the groups of the
connectivity). An entity node is on the boundary of its unit when another unit or an element of no unit holds it (a
fastener that reads a pierced shell, section 5.10), and when its constraints vary between the subcases (below). The
interface elements reach the others through the matrices of their elements, as any point
(`PlannedUnits` reached points).

Relations. Each relation of the formulation is a relation of the plan of its own, keyed by its dependent node
(`reduction::formulationRelationKey`): eliminated inside its unit like a rigid element, or at the interface when a
point of it lies outside the unit (an observer relation at a grid point shared by units). A relation eliminated
inside a unit whose dependent DOF is on the boundary puts its independent points on the boundary, a point held in
every subcase included (`BoundaryPromotion`): the interface eliminates it again there.

What the subcases share. The units are condensed once for every subcase. `CommonConstraints` builds the formulation's
constraints of each constraint selection as the full path does (SPC set with derived supports, the model's
relations, `FormulationConstraints`): the derived supports and the excluded DOFs count with the supports (an excluded
DOF of every subcase is held in the units, without a reaction to report), the relations every selection has with the
same terms bit for bit are eliminated in the units, and the nodes of the relations, excluded DOFs and derived
supports some subcases only have are boundary points. Each constraint set then builds its own constraints at the
interface: its relations (those the units eliminate are left out, but at their boundary points), its excluded DOFs in
the x set (held at the interface, not reported), its derived supports.

After the recovery. Every unit of a model with the formulation is recovered: the excluded rotation of a grid point is
the mean of every shell there, which may lie in several units (`FormulationConstraints::fillExcluded` on the case's
g-set displacements, which the batches fill whole, in sequence too). The recovery unit by unit
(`GroupInputs::visitUnit`: each unit's recovery made once for every case and its results handed over alone, with the
memory mode `low_memory`, envelopes and no per-case table: `run(model, "statics", { condensation, envelopes,
keepsCaseTables: false })`) takes the formulation too. Each unit fills the excluded rotations of its interior points in
its own displacements (every shell there is the unit's) and keeps, per case, the values of its interior DOFs that its
elements at the interface's grid points have (`CondensedStatics::listKeptDofs`); the cases' vectors take them before
their excluded rotations are filled, and the stresses of the grid points of solids are made at the unit's interior
points with the unit, at the interface's points from the cases' vectors (`output::GridPointStressScope`). The reactions
of the derived supports
are carried to their grid points (`SpcSet::carryDerivedReactions`) and the formulation's relations are left out of
the multipoint constraint forces, as on the full path. Brought back to the gauge of the FreeBodySupport DOFs (inertia
relief solved on six farthest translations), the displacements move by the rigid motion on the entity DOFs too, with
the formulation's fields (`assembly::addRigidMotion` with `entityRigidModes`). The rigid-body modes of the units and of
the interface
(inertia relief, mass properties) take the formulation's rigid fields on the entity DOFs (section 5.7,
`UnitModel::entityRigidModes`). The singularity processor leaves the entity nodes out (section 5.8).

The eigenvalue analyses and the trim. Normal modes (the modal layout: the planner numbers the entities of that
analysis, `CondensationPlanner` taking the analysis kind) and buckling reduce the units by Craig-Bampton, with every
fixed-interface mode kept the roots of the full path; the trim condenses them statically with the structural points
of the splines retained (`trimRetainedPoints`). Their constraint sets build the formulation's constraints as the
static ones do (`CondensedStatics::formulationConstraintsOf`: the relations, the excluded DOFs filled in the
eigenvectors and in the trim's displacements, the derived supports with their reactions). Exposed by
`run(model, "modes" | "buckling" | "aeroelastic", { condensation })`; `condensation.modeCount` is the count of
fixed-interface modes per unit (0: those below the cut-off).

The cache. The key of a unit (`reduction::CacheKey`) names its entity nodes by their IDs and DOF counts, and mixes in
what the formulation reads around the unit's elements (`FormulationModel::describeSurroundings`: the elements of two
rings around its grid points, with their IDs, cards and the IDs and positions of their grid points, and the geometry
declarations of the model): the normals and the curves the unit's elements read come from their neighbours, which may
lie in other units, and the declarations come from the program, which O_o_Reader's stamps do not see. A change of the
declarations condenses every unit again. Exposed by `condensation: { cache }` with `solver.condensationCache()` (in
memory).

### 5.12 The streamed path

The full path streamed (SOL 101; `solutions/sol101/StreamedStatics`, exposed by `run(model, "statics", { streamed:
true })`) takes the formulation: the full path's equations solved without a factor of the whole model, the results
those of the full path to round-off.

- The nodes of the entities are nodes of the fronts as any other (`StreamedElements::nodeDofs`: a node's DOFs from the
  DOF map), and an element placed at a front brings every node of its entities. The pass that sums the n-set diagonal
  blocks (`StreamedFront::nodeDiagonals`) gives the entity DOFs a diagonal (the pivot ratios) and no block: they do not
  enter the singularity processor (section 5.8).
- The rigid-body motions on the entity DOFs are the formulation's fields of the six motions
  (`StreamedFront::entityRigidRowsOf`, section 5.7): the check that no element resists a rigid motion, inertia relief
  on the FreeBodySupport DOFs and the GravityLoad accelerations take them.
- The windows of the outputs carry the entities a region gives, each entity's values in a block of its own
  (`elements::DisplacementSlots::entityOffsets`, `output::RecoveryWindow`).
- The excluded rotation of a grid point is the mean over every shell there, and the stress of a grid point of solids
  the mean over every solid there, which may lie in other regions: for a model with entities the outputs keep the
  displacements of the whole g set of every vector of the load basis (`StreamedOutputs::keepsWholeVectors`, in the
  memory plan), fill the excluded DOFs there (`FormulationConstraints::fillExcluded`) and make the displacement rows
  and the grid point stress table from them once the downward pass is over.
- The grid point forces of a load basis' declared cuts are refused, as the grid point force balance is (section 5.9).

### 5.13 External units

An external unit (`solutions/sol101/ExternalUnitPath`) replaces the elements of a zone by a plugin's matrices on the
grid points of its contour. Exposed by `run(model, "statics", { externalUnits })`, each unit given by its zone and its
matrices (`api::ExternalUnitSettings`).

- The entities that only replaced elements act on are held (nothing acts on them any more). A load on them (a
  pressure on a replaced shell) is refused (`external-unit-formulation-loads`): the plugin's load source takes loads of
  grid points.
- An entity that a replaced element and a kept one share (an edge or a face across the contour) is refused
  (`external-unit-formulation-contour`): the plugin's matrices act on grid points, and the edge functions of the kept
  shell would be joined to nothing between them.
- The plugin's matrix may act on any component of its contour grid points: they count as used elsewhere (section 5.4),
  so that the rotation of a shell grid point it stiffens is an observer relation; the m-set couplings of the dense
  block are added to the node graph (`ConstrainedSystem`, for the superelements too).

### 5.14 Superelements

A part (`solutions/superelements/PartReduction`) is its own model, reduced on the grid points it shares with the
residual structure; exposed by `solver.runSuperelements(residual, parts, "statics" | "modes")`.

- A part builds the formulation's constraints of its subcases (`CommonConstraints` with its connectivity's stiffened
  components): its relations are eliminated in it, its excluded DOFs held in it and filled at its recovery, its entity
  nodes interior to it. An excluded DOF is no support: it is not held in the residual structure.
- Refused before any part is reduced: an edge or a face that a part and the residual structure (or two parts) both
  have (`superelement-boundary-formulation`), whose unknowns would be joined at its grid points only.
- Refused after the reduction: a component of a boundary point that a part's formulation excludes while an element of
  the residual structure stiffens it (`superelement-boundary-rotation`): the part has no stiffness on the rotation of
  its shell grid points. A part joined through translations, or through line elements of the part (their rotations
  are unknowns), is taken.
- The rotation of a boundary point that a part's formulation determines (the mean of the part's shells there), on
  which no element of the residual structure acts, is taken from the part's recovery into the residual's vectors
  before its tables are made (`copyDeterminedStatics`, `copyDeterminedModes` of `SuperelementSolution`): both tables
  give the structure's rotation there. Where shells of the residual structure meet the part's at a grid point, the
  residual's table gives the mean of its own shells and the part's table the mean of the part's.
- The residual structure builds its SPC sets with the formulation's derived supports and fills its excluded DOFs, as
  the full path does.

### 5.15 Design derivatives

The semi-analytic sensitivities (`design/sensitivity`) difference the facets of the elements on the snapshots of the
design point with one designed quantity moved; the families of the formulation offer `Facet::Derivatives` with every
computation changing (the default `FieldUse`).

- Every snapshot of a perturbation builds its own formulation model (its own index), and the element sets and the DOF
  map of the sensitivities take the entities (`elements::formulationEntitiesOf`, static layout); the analysis keeps
  the g-set displacements of its subcases (`StaticOptions::keptDisplacements`): no table holds the entities.
- The systems solve with the formulation's constraints: the relations, the excluded set, the derived supports; a
  solution's excluded DOFs are filled (`SensitivitySystem::fromEquations`), and a weight on an excluded DOF (a
  response that reads the rotation of a shell grid point) is moved onto the DOFs it is filled from
  (`FormulationConstraints::moveExcludedWeights`, the transpose of the fill), for the adjoint method, on the
  right-hand sides made sparse too (`AdjointColumns::reduceToEquations`).
- A shape reaches the elements whose geometry reads the grid points it moves (`FormulationModel::readGridRows`: the
  two rings of describeSurroundings), not only those that have them.
- A shape also moves the formulation's relations, u_m = Σ c_mj(v)·u_j (ties, observer relations), and the fill of its
  excluded DOFs, u_x = W(v)·u: u = T(v)·u_n. Their rows on each side of the perturbation come from that side's
  formulation model (`FormulationConstraints::relationsOn`, `fillExcludedOn`), and differences give, per subcase
  (`SensitivityComputation::addFormulationShapeTerms`, `FormulationShape`):
  - D = ∂u/∂v at fixed u_n: D_m = Σ (∂c_mj/∂v)·u_j + Σ c_mj·D_j on the dependent DOFs (the chains of the relations
    followed), D_x = W·D + (∂W/∂v)·u on the excluded ones;
  - the pseudo-load −K·D − (∂C/∂v)ᵀ·μ = −K·D − Σ_m λ_m·∂c_mj/∂v, λ the forces of the relations (K·u − P on their
    dependent DOFs, through the chains), so that Tᵀ·K·T·du_n = Tᵀ·(q − K·D − ∂Cᵀ·μ) and du = T·du_n + D;
  - the explicit part w·D of every item (its weights before the excluded ones are moved), the SPC force items given
    back their share of that pseudo-load (their explicit part reads it), a compliance −scale·(K·u − P)·D (its weights
    are scale·K·u, not scale·P).
  The roots take the same derivative of the reduced problem (`EigenItems::addFormulationShapeTerms`): with D_a the
  derivative of mode a at fixed φ_a,n, a cluster's projection along the shape takes D_aᵀ·R_b + D_bᵀ·R_a, R = (K − λ·M)·φ
  in vibration and (K + λ·K_d)·φ in buckling (the force of the relations on the mode); the constraints, Kgg, Mgg or
  K_d of each solve are made once.
- Where a shape bends a straight boundary of the shells, the reading of the surface changes branch
  (`SurfaceReconstruction`'s straight boundary): the matrices of the elements are not smooth there, and a shape
  gradient taken on the flat configuration is about 1e-4 off.
- The derivative of a buckling factor through its preload needs φᵀ K_d(e_k) φ for every DOF k of an element. The
  shells and solids of the formulation give it in one pass per element and mode
  (`ElementFamily::differentialStiffnessPreloadGradient`): K_d is linear in the stresses of the preload, which are
  linear in its unknowns, so the gradient is the transpose of the stress maps applied to the products of φ with the
  second variations of the measures (`geometricStiffnessGradient` of the shell kernel, `VolumeKernel::geometricGradient`).
  The lines form K_d of each of their DOFs, which are few (`EigenItems`).

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
preload displacements, not its loads; adding the loads (`ElementLoadContext`) to that facet is later work (section
14).

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
- Offsets: under `OffsetDefinition ElemOffsetsOnly` the classic families refuse an offset element in buckling; the
  new formulation carries offsets in its section law and is not refused.

## 10. Output

### 10.1 Grid point tables

- Displacements: translations as solved; rotations as solved (independent or observer), or filled (section 5.4);
  zero at grid points of solids only.
- Support forces: those of grid DOFs, plus the forces of held entity DOFs carried to the grid points (section 5.6).
- Multipoint constraint forces: user relations only.
- Entity DOFs never appear in any user table: the applied loads and the multipoint constraint forces on them (a
  pressure, the glue or the fastener of a face) are carried to the grid points of their entities as the reactions
  are, so that each table has the resultant of what it reports.
- Grid point force balance (`NodeForceOutput`). The elements also act on the unknowns of their edges, faces and
  interiors. Every force on an entity DOF (of an element, of a load, of a relation) is carried to the grid points of
  its entity (`assembly::EntityForceCarry`, `FormulationModel::entityGridRows`: the two grid points of an edge, those
  of a face, the corners of the element of an interior) as forces and moments with the same resultant force and
  moment: the work of the entity's forces in the six rigid motions (the values of section 5.7) is that of the carried
  forces. The set of least norm is taken, the moments counted against the forces with the size of the entity, so that
  moments are used only where forces cannot give the resultant (the torque about an edge, which its two grid points
  carry as moments: the bending moment through a section of a plate). The map depends on the entity alone, never on
  the element or the load, so what balances on an entity DOF balances once carried; a derived support takes its own
  map (section 5.6), as its reaction does. Then, as for `ClassicFormulation`: the rows of a grid point sum to zero,
  the rows of one element have no resultant, and the rows of the grid points of a free body give the forces on it.
  The forces of the formulation's own relations (ties, observer relations, tied rotations) are a row of their own,
  `elementFormulation`, with no resultant: they are no card of the model. The multipoint constraint force rows are
  not split by relation in a model with entities. A case that combines cases of several constraint sets (a SUBCOM)
  has no one set of relations: its `elementFormulation` row is what closes the balance of the grid point. The
  condensed path gives the table of the full path; the full path streamed gives no grid point forces for any
  formulation.

### 10.2 Shells

Rows of `shells/ShellRecovery` (forces, stresses at Z1 and Z2, strains and curvatures, plies and failure indices of
laminates), at the centre and at the corners, in the frame of `ClassicFormulation` (x along G1→G2 projected): the
same tables and columns, so that every client reads them unchanged. The resultants are those of the reference
(`resultants`): about the middle of an offset section.

### 10.3 Solids

Rows of `solids/SolidRecovery` (the 20 columns) at the centre and corners, in CORDM, from the stresses of the element
at those points. The particular stresses of a body force are added to those of the unknowns: the inertial loads of
this formulation are affine acceleration fields (GravityLoad uniform, RotationalForceLoad a(x) = A·(x − c), section
5.7), which the recovery receives for each subcase as twelve numbers (`elements::BodyAcceleration`, the gradient and
the constant in the basic system, times the mass factor; a SUBCOM combines those of its subcases), and the body
force of a solid is ρ·a(x). Every kernel of a solid keeps the particular stresses of the twelve unit affine body
forces (`VolumeKernel::affineBodyStresses`), so that a case with inertial loads costs no kernel of its own. The full path, the condensed path (its units recovered together or one by one) and the
streamed path give the same stresses; without them the stresses of a solid under its weight were first order in the
element size (0.2 % at the centres and up to 3 % at the corners of a 2 × 2 × 2 mesh). Not added: the inertia of an
inertia relief (the rigid accelerations that balance the loads), of a trim analysis, and of the vectors of a kept
load basis (later work, section 14). The pressure of a `FacePressureLoad` given at the corners of a face
is interpolated by position (the functions of the corners of the straight element at the nearest point, as the
temperatures), so that a pressure linear in the position is read as such on a face whose geometry the skin reads
curved. The strain energy density divides the energy by the volume the element itself gives
(`ElementFamily::computeVolumes`). The corner value of one element is 1 to 3 % from the converged value
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

- `addSupportTo`: the fields each family reads, with the rules of section 4 (refused fields: the plane-strain option
  MID2 = −1, pin flags of a line on a shell edge, BeamShapeProp, warping).
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
| Fasteners, welds and glued contacts read the shells and solids of the formulation at their point over every DOF of the element (section 5.10), the point placed by the shape functions of the card; a fastener or a weld reads the shells over the footprint of its cross-section | The reference has no such elements. |
| Temperatures of a solid: the corner changes interpolated by the functions of the corners of the straight element at the point nearest to each point of the kernel; an expansion that differs between directions is refused under a temperature load | The kernel of the reference takes one expansion coefficient and a temperature field of the position. |
| A pressure along a direction on a solid face is taken when it is the same at every corner of the face | The kernel's traction field is uniform there. |
| Membrane-only and bending-only shells (section 4.1): the strains of the part a section lacks are recovered as zero (the generalised inverse of [A B; Bᵀ D], `membraneBendingCompliance`), and the strains of an offset membrane-only section are one split (e, κ) of its strain e + z0 κ | The reference takes both parts; its compliance is the inverse of the whole section. |
| Two shells on the same corner grids are refused | The reading of the surface, the supports along the edges (an external edge has one shell) and the continuity of the shear (the second shell of an edge across from the first) of the reference count one shell per side of an edge. |
| The static condensation of components takes the formulation (section 5.11) | The reference solves the whole model at once. |
| Supports are read along the edges inside the shells that have a grid point inside them, as along the external ones (section 5.6) | The reference reads the grid supports along the external edges and states the others on the edges; a model file states supports at grid points only, and a line of supports inside a plate must hold it along the line. |
| Supports along a direction that is no basic axis, the same at the grid points of an edge or a face, are relations on its functions (section 5.6) | The reference has global axes only; a plane of symmetry in any orientation must hold the edges and faces in it. |
| The reactions of the held edges and faces, the loads and the forces of the relations on entity DOFs are reported at the grid points with their resultant force and moment (sections 5.6, 10.1) | The reference reports no table at grid points; the map that carries held values along a chord does not balance the moment on a curved edge. |
| At a grid point of shells whose lines all lie on shell edges, the rotation follows the shells in the directions the lines do not stiffen (section 5.4) | The reference holds those directions at zero (tie 5): a couple or a rigid element at the end of a rod on a shell edge would act on nothing. |
| A rotation held at one grid point of shells, or made dependent there by a relation of the model, is tied to the mean rotation of the shells (section 5.6); the edges and faces follow the relations of the model as they follow supports | The reference has no rigid element and reads a held rotation along the edges only: the six components held at one grid point would not hold a free body, and a rigid element joined at one grid point would hinge. |
| Grid point force balance with the forces of the entity DOFs carried to the grid points (section 10.1) | The reference has no such table. |
| A mesh transition on a straight outline or junction keeps it straight, and a side with quadrilaterals only is read as between quadrilaterals (section 4.4) | The reference reads three quadrilaterals at a boundary grid point as a re-entrant corner whatever their angles, and a quadrilateral side beside a triangle by the triangle rule. |

Implementation notes. The rows of the element tables are those of section 10; the grid point stress table of section
10.3 has the element type `SolidGridPoints` (one row per grid point, its ID as element and place, the columns of the
stress rows of the solids, basic system). The geometry report of section 10.5 is a set of diagnostics:
`geometry-report` (information: the counts and the largest kink and rounding); `geometry-kink` (an edge whose normals
differ by more than 1°: a warning along an edge of one or two shells that is not declared raw, which the shell takes
as a fold; an information along a raw edge or a junction); `geometry-rounding` (an edge along which the surface read
turns from the plane of the corners of a shell by more than 15°: an information, and a warning above 30°, a fold
that is not declared raw; and a warning for an edge of the outline that leaves its chord by more than 30° in the
surface: a corner of the outline read as a curve). A shell left without an angle at a corner by the reading is
refused (`degenerate-geometry`), with the edges to declare raw. A moment at a grid point of solids alone is refused
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
go to a separate suite, not to the fast suite of `wasm/test` with its one-second limit per case:
`wasm/test/validation/*.test.mjs` (`node --test`, about four minutes), six files of about 500 cases with the criteria
of section 13.6 (shells in statics; shells in modes, buckling and inertia relief; lines; solids; mixed meshes;
assemblies through every path). The cases that do not meet their criterion are listed with their reason in
`wasm/test/validation/known-limits.mjs` and reported as "todo": a limit of section 3.6, later work of section 14, or
a defect outside the formulation. No criterion is loosened there.

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
- `ELEMENT_FORMULATION_VERSION` does not change: no matrix of an existing formulation changes. The keys of the
  condensation cache of a model without the formulation are unchanged.

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
| 7 | finishing: supports read along the edges inside the shells and along directions that are no basic axis, rigid elements and multipoint constraints read along the edges, rotations no edge reads tied to the mean rotation of the shells, rotations the lines do not stiffen following the shells, grid point force balance and force tables carried to the grid points, entity names in the messages of the factorisations, the pivot ratio test at entity DOFs, the geometry report as warnings, the validation suite of section 13.3 (`wasm/test/validation`) and the defects it found, the particular stresses of body forces in the solids (section 10.3), the formulation's refusals listed by the support check before a run (`api::checkSupport`), curved edges that follow a relation of the model (section 5.6), mesh transitions on straight boundaries and re-entrant corners beside triangles (section 4.4) | each with its own checks |
| later | not chosen yet: supports along a direction that turns along an edge (a cylindrical CD along a circle, section 5.6), the particular stresses of the inertia of an inertia relief, of a trim analysis and of the vectors of a kept load basis in the solids (section 10.3), loads in the differential stiffness facet (following pressure), the pivot reference on the condensed and streamed paths | |

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
