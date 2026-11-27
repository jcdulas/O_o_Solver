# Comparing with another solver

Two solvers given the same structure rarely print the same numbers, and most of the difference is not an error of
either: the elements follow different theories, a support or a load means something slightly different, the meshes are
not converged to the same digit, and the timings measure different things. This page lists what to align before a
difference is read as an accuracy or a speed result, and what O_o_Solver reports to help.

## What a run tells

A run shows these messages as it ends (the console in JavaScript, the `warnings` module in Python) and keeps them in
the `diagnostics` of its result. Each names what was read and what to give instead.

| Code | Severity | What it tells |
| --- | --- | --- |
| `mixed-formulation-available` | information | The shells are computed in `ClassicFormulation`; `MixedHybridFormulation` reaches the converged values on coarse meshes |
| `edge-layer-not-resolved` | warning | Mixed formulation: soft supports on elements too deep for their edge layer (see [supports](#supports-soft-and-hard-simple-supports)) |
| `forces-read-along-edges` | information | Mixed formulation: forces at both grid points of outline edges were read as tractions |
| `point-forces-on-shells` | information | Mixed formulation: forces left at grid points of shells are concentrated forces (see [loads](#loads-along-an-edge)) |
| `laminate-shear-factor` | information | Laminates whose transverse shear stiffness is not 5/6 of the sum of their plies (see [laminates](#transverse-shear-of-a-laminate)) |

## Checklist

| Item | Align | O_o_Solver |
| --- | --- | --- |
| Shell theory | Thin plate (no transverse shear) or shear-deformable | `shearMatId` of `ShellProp` blank: no transverse shear; given: shear-deformable |
| Transverse shear of a laminate | From the shear stresses of the stack, or 5/6 of the plies' shear moduli times their thicknesses | `Parameter;LaminateTransverseShear;Equilibrium` (default) or `FiveSixths`; `report` gives the stiffness and factor of each laminate |
| Element formulation | Displacement elements, or a formulation converged on coarse meshes | `ClassicFormulation` (default) or `MixedHybridFormulation` |
| Simple support | Soft (deflection held) or hard (deflection and the rotation about the edge normal held) | What the constraint cards hold; the thin-plate formulas are those of the hard support |
| Forces along an edge | Nodal forces of a line load, or point forces | Mixed formulation: forces at both grid points of an outline edge are read as a traction |
| Load on a surface | A pressure, or the nodal forces that stand for it | Mixed formulation: give the pressure (`ShellPressureLoad`); forces left at grid points are concentrated forces (`point-forces-on-shells`) |
| Mass | Lumped or consistent | `Parameter;CoupledMass;1` for the consistent mass of the classic elements (default `-1`: lumped); the mixed formulation always uses its consistent mass |
| Mesh | A convergence sequence, not one mesh | Report the element count and the formulation with each value |
| Eigenvalue units | Hz, rad/s, eigenvalue | The eigenvalue rows give `eigenvalue` (ω² or the load factor), `radians` and `cycles` |
| Time | The same work, threads and machine | `report` of every run: threads, kernels, sizes, stage times, memory |

## Theory: what each element converges to

The classic shells (`ShellQuad4`, `ShellTria3`, `ShellQuad8`, `ShellTria6`) are displacement elements with transverse
shear, written from the publications cited in their formulation documents
([four-node](shell-quad4-formulation.md), [three-node](shell-tria3-formulation.md),
[eight- and six-node](shell-quad8-tria6-formulation.md)). Like most displacement shells they approach the converged
value as the mesh is refined.

`MixedHybridFormulation` ([specification](mixed-hybrid-formulation-spec.md)) resolves the shear-deformable shell much
more closely on a given mesh: it reaches four to five digits of the shear-deformable solution where displacement elements
need many times more elements. It therefore converges to the shear-deformable answer, which is not the thin-plate
formula of a handbook: on a hard-supported square plate whose span is a hundred thicknesses the two differ by about
0.05 % (deflection, frequency and critical load), and by about one percent with soft supports (next section).

To compare with a thin-plate formula, give the section no transverse shear: leave `shearMatId` (field 6 of `ShellProp`)
blank. Both formulations then give the thin-plate value, the mixed one from coarse meshes. To compare with a
shear-deformable reference (a solid model, a thick-plate solution), keep the shear material.

## Transverse shear of a laminate

Two programs given the same stack can differ in its transverse shear stiffness before any element is involved. Written
as `H = κ Σ Gk tk` along a material axis (`Gk` the transverse shear modulus of ply k, `tk` its thickness):

- many programs and closed forms take κ = 5/6 for every stack, the factor of a homogeneous section;
- O_o_Solver by default takes the stiffness from the shear stresses that the equilibrium gives in the stack
  ([laminated shells](composites.md#section-of-the-laminate)). κ is then 5/6 for one material only: 0.78 along the
  fibres of the outer plies of a carbon/epoxy [0/90]s and 0.66 across them, a transverse shear stiffness 7 % and
  21 % below that of the factor 5/6.

On a thick laminate the difference is visible in the deflections, the frequencies and the critical loads, and it does
not shrink with the mesh: it is a difference of model, not of discretisation. Before comparing:

1. read the information `laminate-shear-factor` of a run, which counts the laminates whose factor is more than
   1 % from 5/6 and names the furthest, and `report.laminateShear` (Python: `report["laminate_shear"]`): one row per
   laminate property, with its stiffness and its factor κ along the two axes;
2. where a factor is not 5/6 and the reference uses 5/6, run the model with
   `Parameter;LaminateTransverseShear;FiveSixths`: both then solve the same plate, and what is left is the elements
   and the mesh;
3. to judge which convention is closer to the structure, compare with a solid model of the stack: the default follows
   it more closely.

A mesh convergence sequence separates the two kinds of difference: a discretisation error falls as the mesh is
refined, a difference of convention stays.

## Supports: soft and hard simple supports

A simply supported edge of a shear-deformable plate can be held in two ways:

- **soft**: the deflection alone is held; the rotation about the normal to the edge, in the plane of the plate, is free;
- **hard**: the deflection and that rotation are held.

Example, a plate in the x-y plane with an edge along x (normal to the edge: y). The bending rotation of that edge is
about x and stays free in both cases. The soft support holds component 3 (the deflection) along the edge; the hard
support holds components 3 and 5 (the deflection and the rotation about y, which would twist the edge). A thin plate
cannot twist an edge whose deflection is held, so the distinction exists only with transverse shear.

The thin-plate formulas are those of the hard support. A soft support on a shear-deformable plate has an edge layer
about a third of the thickness wide, in which the twisting moment drops to zero; its converged values are about t/a
softer than the formulas. On a square plate 1 m wide and 10 mm thick, measured with the mixed formulation: deflection
+0.91 %, first frequency −0.48 %, critical load −0.94 % against the thin-plate formulas.

A mesh that does not resolve that layer lands between the two values, so a coarse mesh can look closer to the formula
than a fine one. The mixed formulation signals it with the warning `edge-layer-not-resolved` when a soft support lies
along elements deeper than four thicknesses. To compare:

- with a thin-plate formula: hold the rotation too (hard support), or give the section no transverse shear;
- with the converged soft-support value: add a row of elements about one thickness wide along the supported edges;
  2 × 2 elements plus that row give the converged values to 0.01 %.

## Loads along an edge

A line load is often given as forces at the grid points of the edge (half a share at the ends). The classic elements
take them as they are. On the mixed formulation's continuum a force at a point is a concentrated force, which moves its
grid point more than the section; the formulation therefore reads forces given at both grid points of an outline edge
as the linear traction along the edge whose consistent nodal forces they are (resultant force and moment unchanged),
and reports it with the information `forces-read-along-edges`. A force alone at one grid point stays a point force:
compare displacements away from it, or spread it over an edge.

The same holds for a load on a surface given as a force at each grid point (a pressure times the area around the
point). The classic elements take these forces as the load vector of the pressure. The mixed formulation takes each
as a concentrated force: the plate dimples at every loaded grid point, and the displacement there grows as the mesh is
refined instead of converging. The run says so once per load set with the information `point-forces-on-shells`, which
counts the forces left at grid points of the shells. Give the load itself: `ShellPressureLoad` for a load per unit
area, forces at both grid points of each loaded outline edge for a load along it.

Buckling loads given as forces along an edge are thus read as the edge compression they describe: the critical load of a
simply supported plate (hard support) is exact with shear on 2 × 2 elements.

## Eigenvalues

- Mass: the classic elements use a lumped mass unless the model sets `Parameter;CoupledMass;1`; consistent and lumped
  masses give frequencies that bracket the converged one differently. The mixed formulation always uses its consistent
  mass (and reports it with an information message when a lumped mass is requested).
- Rows of an eigenvalue table: `eigenvalue` is ω² (rad²/s²) for normal modes and the load factor for buckling;
  `cycles` is the frequency in Hz.
- A free structure has six zero frequencies (rounded zeros of either sign) before its first elastic mode.

## Mesh density

Compare convergence sequences: the same structure on two or three meshes refined by halving, with each value given with
its element count and formulation. A single coarse mesh says more about the elements than about the solver, and a
single fine mesh hides which solver gets there first. The [structural benchmark comparison](structural-benchmark-comparison.md)
shows both formulations on such sequences.

## Run time

- **Threads.** Natively (Python, C++) an analysis runs on every core of the machine unless `threads` asks otherwise; the
  results do not depend on the count. The default JavaScript package runs on its calling thread; the shared-memory
  package (`O_o_Solver-wasm-threads.zip`, [threading guide](threading.md)) runs on several.
- **What to time.** Reading the model file, building the snapshot and running the analysis are three steps; time each
  apart. The `report` of a run gives its wall time, the time of each stage (setup, assembly, factorisation, solve or
  eigenvalues, recovery), the threads, the dense kernels (`avx2` on a recent x86-64 processor), the degrees of freedom
  and the largest memory its budget held.
- **What to compare.** Compare times at the same accuracy rather than on the same mesh: the mixed formulation costs much
  more per element than the classic one, and reaches a given accuracy with far fewer. For size, the classic formulation
  is the one to compare. [Large models](large-models.md) gives the measured scaling.
