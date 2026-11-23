# Fast input combination tests

Run the built Reader and Solver packages with a Memory64-capable Node.js runtime:

```bash
node wasm/test/run.mjs
OOS_SOLVER_DIST=build/wasm-threads/dist node wasm/test/run.mjs
```

On PowerShell, set `$env:OOS_SOLVER_DIST` before the second command. `OOR_READER_DIST` selects a different Reader
package; its default is `../O_o_Reader/build/wasm/dist`. The ordinary `node --test --test-concurrency=1 wasm/test/*.test.mjs` entry point
also runs the complete suite.

## One second per case

Reader, Solver and the optional threading probe initialize once in a reusable worker, before tests start. Each
input combination has a **900 ms watchdog in its parent process**, plus the test runner's **1,000 ms timeout**.
The watchdog terminates the worker even when synchronous WASM blocks its event loop. Focused tests have a 1,000 ms
timeout, and the full-suite command runs test files sequentially to avoid resource contention. Separate regression tests
deliberately block workers in JavaScript and in an infinite WASM loop, and check that both are interrupted.

The budget includes parsing the tiny model, exporting its tables, support checks, calculation, assertions,
cleanup and the message round trip. Initial module loading is setup, reported as part of the suite's total elapsed
time rather than charged to the first case. A timeout fails the suite; later cases fail immediately if their worker
has been terminated. A slow case is never silently skipped or accepted as a numerical refusal.

The command writes `build/test-results/wasm.json` or `build/test-results/wasm-threads.json`: every test name, status
and duration, the ten slowest tests, maximum duration, failures and any duration reaching one second. It exits with
a nonzero status for a numerical failure or an exceeded limit. `OOS_TEST_RESULTS` changes the report directory.
The separate worker-pool tests create their pools during module setup and give each test a 1,000 ms timeout.

These limits are those of a developer's machine. A slower or shared machine sets `OOS_TEST_TIME_FACTOR` (a number of
at least 1): the watchdog, the ceiling of each test and the limit of the report are multiplied by it, so that a case
that hangs is still interrupted.

To investigate one input without running the entire matrix:

```bash
node wasm/test/run.mjs --test-name-pattern="line matrix element=Tube2 load=gravity"
```

## Derivative checks

The [derivative coverage guide](derivatives.md) describes which paths these suites exercise. Each case compares
small, independently perturbed analyses or gradients with the native derivative; product checks also compare
direct/adjoint results, symmetry and Jacobian/Hessian actions where applicable.

| Suite | Main checks |
| --- | --- |
| [Bar and beam derivatives](../wasm/test/cases/beam-derivatives.mjs) | Section/material first and mixed rates, release maps, lumped/coupled mass, thermal sources, span loads, loaded end results and mixed loaded/unloaded batches |
| [Modal derivatives](../wasm/test/cases/modal-derivatives.mjs) | Mode-selected fields, normalization, bordered derivatives and root/field Hessian products |
| [Buckling derivatives](../wasm/test/cases/buckling-derivatives.mjs) | Mechanical preload tangents, load sharing, simple-root curvature and shell geometric stiffness |
| [Trim derivatives](../wasm/test/cases/trim-derivatives.mjs) | Structural/trim fields, stability coefficients, support accelerations and mass-dependent loads |

For example, run the bar/beam cases with the same watchdog as the full suite:

```bash
node wasm/test/run.mjs --test-name-pattern="analytic (bar sizing|beam)"
```

The timing report records the tested revision's local execution. Use representative models to assess larger
optimization workloads; the small derivative fixtures verify formulas and interactions.

## Validation of the mixed hybrid formulation

The [structural benchmark comparison](structural-benchmark-comparison.md) reports both formulations on the same meshes,
the closest results observed under refinement, and commercial solver values with their sources. The input files and replay are included.

The [thin-plate solid benchmark](thin-plate-solid-bending.md) reports measured deflection and surface stress errors
for tetrahedra and hexahedra, with model files for replaying the degree and mesh refinements.

The benchmarks of `MixedHybridFormulation` against closed forms and classical values
([specification](mixed-hybrid-formulation-spec.md), sections 13.3 and 13.6) are a separate suite without a time
limit, in [wasm/test/validation](../wasm/test/validation):

```bash
node --test wasm/test/validation/*.test.mjs
```

| File | Main checks |
| --- | --- |
| `shell-statics` | Plates, the Scordelis-Lo roof, the pinched cylinder and hemisphere, the twisted beam, folded sections, laminates, offsets, pressure vessels, thermal loads, patch tests and invariance |
| `shell-dynamics` | Frequencies of plates, cylinders and rings, buckling of plates and cylinders, inertia relief, stiffened panels, masses, springs and moments at grid points of shells |
| `lines` | Frames, trusses, shear deformation, tapered beams, offsets, pin flags, distributed and thermal loads, frequencies and buckling of beams |
| `solids` | Patch tests, beams, thick cylinders and spheres, every load, anisotropy, near incompressibility, frequencies and buckling, for the four shapes with and without mid-side nodes |
| `mixed-meshes` | Shells on faces of solids, fins, lines along edges of solids and of shells; a plate continued by solids in bending and in tension, on meshes of the solids refined along and through the thickness (the ties of the shell over its thickness) |
| `assemblies` | Three assemblies through every path (full, condensed, streamed), equilibrium of the tables, invariance, determinism, robustness of the input |
| `classic-shells` | The `ShellQuad4`, `ShellTria3`, `ShellQuad8` and `ShellTria6` of `ClassicFormulation` ([shell-quad4-formulation.md](shell-quad4-formulation.md), [shell-tria3-formulation.md](shell-tria3-formulation.md), [shell-quad8-tria6-formulation.md](shell-quad8-tria6-formulation.md)): each case with quadrilaterals and with two triangles per quadrilateral (three ways of cutting; two for the `ShellTria6`): a cantilever strip in bending, in torsion and bent in its plane, plates under a pressure, the Scordelis-Lo roof, the pinched hemisphere, frequencies of a plate; the twisted beam, the free thermal expansion of a quarter cylinder; the normals shared or given at the grid points, the drilling factor, the notice of an inertia on a shell normal, the warning of a load path through the drilling stabilisation (a moment about the normal on one grid point, a bar attached at one grid point), the modes of a ring with the coupled mass; the corner rows of the `ShellQuad4` (the `CornerResults` describer: its centre and four grid points, the centre alone without it, and the centre alone for the `ShellTria3`; a strip pulled, bent in its plane, bent by end moments and by a tip force; a block of load cases against each case alone; the hoop force of a cylinder under pressure on flat and on warped elements); the bounds are the measured closeness of the triangles to the quadrilateral, and the measured values of the quadratic elements |
| `fasteners` | The `Fastener` and `Weld` elements in both formulations, on synthetic joints: one fastener between two plates (its six stiffnesses, its force table, a rigid motion), where it pierces the shells (grid point, inside, side, near a corner; `ShellQuad4`, `ShellTria3`, `ShellQuad8`, `ShellTria6`, flat and warped), every way of locating it and its axes, the refused forms, a lap joint against the model of two bars joined by springs, two skins riveted together between the separate and the monolithic plate (statics, modes, buckling), two concentric cylindrical skins, a stringer fastened to a skin, the forms of the `Weld` card against its short beam, one joint written with a `Fastener`, a `Weld` and a `Bushing` tied by hand over the footprint of the fastener, the streamed path and a temperature rise, a flange bolted to a skin along one of its edges (fasteners against springs between the grid points); the cases the `MixedHybridFormulation` does not meet are marked `todo` with their measured values |
| `glue` | The glued contact of shells in both formulations, on synthetic joints: two plates glued with a gap between their mid-surfaces follow a rigid motion; a strip in two parts glued end to end by its rotations is the strip in one piece and a web glued on a skin is the web meshed with it, in both formulations (the edges between glued grid points tied along their length in the MixedHybridFormulation); what the glue says of the rotations it cannot tie (the face of a solid); small joints against their closed forms (a cantilever and a simply supported beam of two glued layers, a lap splice in bending and in tension, a web glued on a rigid skin, a plate glued on a solid under a uniform strain, two blocks of solids glued end to end) |
| `classic-stiffened-panel` | The `Bar2`, `Beam2` and `Rod2` of `ClassicFormulation` on shells that are not plane: a twisted stiffened panel (skin of `ShellQuad4` and `ShellTria3`, stringers, frames and edge caps as line elements with offsets along the normal of the skin) against the same panel with webs of shells, with a thin and a thick skin; the axis of a section given by its type and dimensions (the neutral axis of a `Bar2` on a `BarShapeProp`, the shear centre of a `Beam2` on a `BeamShapeProp`) |

The cases that do not meet their criterion are listed with their reason in `known-limits.mjs` and reported as "todo":
no criterion is loosened. The model files of the suite carry the declarations of their geometry as a preprocessor
writes them (`SurfaceEdge`, `SurfaceFacet`, `SurfaceLine` records, or `Parameter;GeometryDeclarations;Given`: the
helpers `surfaceRecords` and `prepared` of `support.mjs`); the program only chooses the formulation.
`node wasm/test/helpers/untimed.mjs [suite ...] [--match text]` runs the cases of the fast
suite without its watchdog, for a machine busy with something else.

## Continuous integration

Two workflows run on a push to `main` and on a pull request that change the sources.

`WebAssembly` builds the package, then runs on it:

- the fast suite (`node wasm/test/run.mjs`, then the numerical derivative and worker pool tests), with
  `OOS_TEST_TIME_FACTOR` at 10 for the shared runners; its timing report is kept as the `O_o_Solver-test-timings`
  artifact;
- the validation suite, its files dealt to four runners by their rank in `wasm/test/validation`.

Both load the O_o_Reader package of the `wasm-latest` release of the O_o_Reader repository
(`.github/scripts/fetch-reader-package.sh`), so a change that needs a change of the reader passes once the reader's
own change is on its main branch. The releases of the package are published only when both suites pass.

`Native` builds the libraries with GCC 12, the oldest compiler the sources are written for, with the warnings as
errors, and runs the native checks (`checks/`).

## Coverage domains

Numerical values and model sizes are unbounded, so no finite suite can enumerate every possible input. The suite
exhaustively enumerates the declared finite Cartesian domains below, and preserves the existing focused tests of
optimization, modes, formulations and worker lifetime. Inputs are generated deterministically from original small
models; no random sample replaces a declared combination.

The [external study cases](../wasm/test/cases/studies.mjs) additionally exercise SOL 101/103/105/144, full-analysis
and native gradients, auxiliary shape directions, coupled material constants, named Reader result queries and
accepted-point archives. [Numerical derivative tests](../wasm/test/numerical-derivatives.test.mjs) cover bounds,
discrete variables, output layout changes and memory limits. [Worker pool tests](../wasm/test/worker-pool.test.mjs)
cover independent package memories, parallel points and termination/replacement of a worker blocked in synchronous
JavaScript. Run these standalone tests with `node --test --test-concurrency=1 wasm/test/*.test.mjs` in addition to `node wasm/test/run.mjs`.

| Matrix | Domains | Cases |
| --- | --- | ---: |
| Line elements | 5 elements × 4 loads × 5 NSM states × 2 mass formulations × 2 unit scales × 2 constraint forms × 2 selection scopes | 1,600 |
| Added property mass and load boundaries | 5 elements × 5 NSM states × 2 property masses × 2 inertial loads × 3 signs | 300 |
| Point constraints and scaling | 5 elements × 3 loads × 2 signs × 2 moduli × 2 lengths | 120 |
| Surface elements | 7 elements × 2 loads × 5 NSM states × 2 mass formulations × 2 unit scales | 280 |
| Solid patch tests | 4 topologies × 2 interpolation orders × 2 signs × 2 moduli × 2 mass settings | 64 |
| Connectors | 9 families × 2 load signs | 18 |
| Concentrated masses | 6 families × 2 masses × 2 unit scales | 24 |
| Load combinations | direct/single/nested × 2 loads × 5 NSM states × 2 scopes | 60 |
| Multipoint constraints | equation/union × 2 loads × 5 NSM states × 2 scopes | 40 |
| Combined subcases | 2 scopes × 3 first mass choices × 3 second mass choices × all/selected output | 36 |
| Unsupported selections | 5 selections × 2 scopes × 2 loads × 2 mass choices | 40 |
| Invalid NSM | 5 elements × 2 loads × 5 invalid inputs | 50 |
| Thread counts and NSM | 5 elements × 5 NSM states × available counts among 1/2/4/8 | 25–100 |

The five NSM states are absent, present but unselected, direct selection, target list and union. The four primary
loads are point force, gravity, uniform temperature and imposed element deformation. Additional cases cover
node-directed force and acceleration. Constraint sets use lists or unions; a separate matrix checks individual
point constraints. Selections appear locally or are inherited from global Case Control.

All **31 element card types** currently returned by `elementFormulations()` have an executable fixture. A registry
guard fails when a new family lacks a fixture, and a separate guard checks the primary Cartesian count. Linear and
quadratic solids and higher-order shells are included. Invalid inputs check their specific refusal diagnostics,
including unsupported nonlinear/dynamic requests and undistributed lumped NSM.

The independent oracles are axial extension `F L / (E A)`, thermal expansion, prescribed extension, uniform gravity,
spring extension `F/K`, scalar frequency squared `K/M`, and constant-stress patch solutions `u = stress*x/E`.
Shell NSM under gravity is also compared with an equivalent material density. Existing tests retain direct/adjoint
gradient and Hessian checks. Small models check correctness and interactions; these are not large-model performance
benchmarks or exhaustive tests of every material, contact option, control command or mesh geometry.

The [output stations of the bars](../wasm/test/cases/bar-stations.mjs) (`Bar2Stations`) are checked on a cantilever of
two `Bar2`: the rows at the ends and at the stations of a bar, listed or equally spaced, against the moment of the
beam; under a distributed load; in a block of load cases; with an output set; a station beyond the end of its bar
(refused) and a card that names no bar (a warning).

Implementations live in `wasm/test/cases/`; the worker, watchdog and shared runtime are in `wasm/test/helpers/`.
