# Comparing with another solver

Two solvers given the same structure rarely print the same numbers, and most of the difference is not an error of
either: the elements follow different theories, a support or a load means something slightly different, the meshes are
not converged to the same digit, and the timings measure different things. This page lists what to align before a
difference is read as an accuracy or a speed result, and what O_o_Solver reports to help.

## Checklist

| Item | Align | O_o_Solver |
| --- | --- | --- |
| Shell theory | Thin plate (no transverse shear) or shear-deformable | `shearMatId` of `ShellProp` blank: no transverse shear; given: shear-deformable |
| Element formulation | Displacement elements, or a formulation converged on coarse meshes | `ClassicFormulation` (default) or `MixedHybridFormulation` |
| Simple support | Soft (deflection held) or hard (deflection and the rotation about the edge normal held) | What the constraint cards hold; the thin-plate formulas are those of the hard support |
| Forces along an edge | Nodal forces of a line load, or point forces | Mixed formulation: forces at both grid points of an outline edge are read as a traction |
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
