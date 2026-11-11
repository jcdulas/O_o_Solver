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

Implementations live in `wasm/test/cases/`; the worker, watchdog and shared runtime are in `wasm/test/helpers/`.
