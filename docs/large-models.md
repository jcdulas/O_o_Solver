# Large models

O_o_Solver solves its equations with linear algebra written in the project: no external solver library sits between a
model and its factor. This page says what it does with a large model, how it scales on one machine, which settings
matter, and where its limits are.

## What the solver does

- **Sparse direct solver.** The equations are factorised by a supernodal multifrontal LDLᵀ: without pivoting for
  statics (with a test of the pivot ratios that names the equations of a singular model), with Bunch-Kaufman pivoting
  inside each front for the shifted problems of eigenvalues, whose inertia it counts. Independent subtrees of the
  elimination tree run in parallel, and the threads share the large fronts near the top.
- **Fill-reducing ordering.** The grid points of a large mesh are ordered by geometric nested dissection (coordinate
  bisection, a separator of least weight across each cut, the leaves by constrained minimum degree), halves in
  parallel; a small mesh, or one without coordinates, by the approximate minimum degree (Amestoy, Davis and Duff,
  1996). The factor's size and operation count are known before the numerical work starts.
- **Assembly.** Element matrices are formed in parallel and summed by ownership of block columns: every term is added
  by one thread in element order, with the bits of a serial loop.
- **Dense kernels.** The fronts are factorised by packed, register-blocked kernels: AVX2 and FMA on the x86-64
  processors that have them (chosen at run time), portable reference kernels elsewhere, the 128-bit vector instructions
  in WebAssembly.
- **Load cases.** All load cases of a constraint set share one factorisation; their right-hand sides are solved
  together, and a deck with many cases is reduced to a compact basis of load vectors first.
- **Eigenvalues.** Normal modes and buckling factors come from a block shift-invert Lanczos solver with full
  reorthogonalisation; Sturm sequence counts at the shifts confirm that no eigenvalue of the requested range is missed.
  Its shifted matrices are scaled by the diagonal of the stiffness before they are factorised, so that the pivots do
  not depend on the units of the model. Two other solvers take less memory
  ([below](#modes-and-buckling-of-a-model-whose-factor-does-not-fit)).
- **Memory.** Every large array is reserved in a memory budget (by default the memory the machine has left): a run that
  would exceed it stops with a message instead of swapping, and the factorisation reserves its storage before it
  eliminates the first column. When the memory left is short for the parallel order of the factorisation, the solver
  takes a postorder of the elimination tree, whose working storage is several times smaller, and runs longer with the
  same results.

The algorithms and their sources are described in the [linear algebra guide](../src/linalg/README.md).

## Threads

An analysis run from Python or C++ uses every core of the machine unless `threads` asks for another count, and gives
the same results on any count. The default WebAssembly package runs on its calling thread; the shared-memory package
(`O_o_Solver-wasm-threads.zip` in each [release](https://github.com/jcdulas/O_o_Solver/releases)) runs on several
([threading guide](threading.md)).

## Measured scaling

The [scaling benchmark](benchmarks/scaling/README.md) generates the models it measures (a flat plate, a closed
cylinder and a wing box of `ShellQuad4`, and a block of `SolidHexa`, from 10⁴ to 2 × 10⁶ degrees of freedom) and runs
statics, the first ten normal modes and the first buckling factor on one thread and on every core, with the Python
packages installed from PyPI. Anyone can run it again on their own machine.

Measured with oo-solver 0.2.5 on a desktop computer: Intel Core i5-12400F (6 cores, 12 threads), 32 GB of memory of
which 16 to 18 GB were free, Windows 11. Times are the wall time of the analysis on 12 threads, with the time on one
thread in parentheses; the memory is the peak of the process. All [results](benchmarks/scaling/results.md), size by
size, are in the benchmark's folder with their [raw data](benchmarks/scaling/results.json).

About one million degrees of freedom:

| Model | DOF | Statics | Normal modes (10) | Buckling (first factor) |
| --- | ---: | --- | --- | --- |
| Plate | 1 000 008 | 5.2 s (14.0 s), 4.3 GB | 32 s (58 s), 7.4 GB | 24 s (60 s), 6.9 GB |
| Cylinder | 999 844 | 5.4 s (14.9 s), 4.3 GB | 37 s (61 s), 6.8 GB | 50 s (92 s), 6.3 GB |
| Wing box | 1 028 400 | 5.0 s (15.4 s), 5.4 GB | 31 s (65 s), 8.3 GB | 34 s (99 s), 7.8 GB |
| Block of hexahedra | 317 952 | 5.3 s (20.6 s), 5.6 GB | 16 s (51 s), 8.2 GB | 23 s (76 s), 7.7 GB |

About two million degrees of freedom:

| Model | DOF | Statics | Normal modes (10) | Buckling (first factor) |
| --- | ---: | --- | --- | --- |
| Plate | 2 006 238 | 11.6 s (32.9 s), 8.9 GB | over the free memory | over the free memory |
| Cylinder | 2 003 968 | 11.8 s (35.3 s), 8.9 GB | 65 s (122 s), 13.8 GB | 108 s (211 s), 12.9 GB |
| Wing box | 1 931 400 | 10.6 s (34.2 s), 10.1 GB | over the free memory | 73 s (164 s), 14.7 GB |

- **Time and memory grow nearly in proportion to the size** for the shells: a static analysis takes about 5 s and
  4.5 to 5 GB per million degrees of freedom on this machine, the first ten modes or the first buckling factor 25 to
  50 s.
- **Threads.** On 12 threads a static analysis of one to two million degrees of freedom runs 2.7 to 3.2 times faster
  than on one, an eigenvalue analysis of a million 1.7 to 2.9 times; the block, whose fronts are larger, gains 3.1 to 3.9 times. The
  results are the same on any number of threads.
- **The models are right as well as fast.** Every run is checked against a closed-form value: the plate and the
  cylinder converge to their thin-shell references within a few hundredths of a percent (the cylinder's buckling
  factor approaches its reference from above, +0.26 % at two million degrees of freedom), the wing box and the block
  settle within one percent of their beam models.
- **Memory is the limit.** A three-dimensional mesh fills its factor far more than a shell mesh: the block of a million
  degrees of freedom needs about 30 GB (a factor of 14 GB and 13 GB of fronts), more than was free. The solver stopped
  those runs, and the eigenvalue runs marked above, with a message naming what they needed; the streamed path solves a
  static analysis whose factor does not fit.

These tables are those of version 0.2.5. The sources of this revision go further on the runs that were over the
free memory: the plate's first ten modes at two million degrees of freedom take 73 s and 11.2 GB, its first buckling
factor 39 s and 12.3 GB, and the block of a million degrees of freedom asks 1.9 GB of fronts in place of 13 GB (it then
needs 18 GB in all). The tables will be measured again with the next version.

## Modes and buckling of a model whose factor does not fit

The Lanczos solver holds the factor of the shifted stiffness in memory. Two other solvers do without it, chosen by the
`eigenSolver` setting (`eigen_solver` in Python):

- **`substructuring`**: multilevel substructuring on the elimination tree of the model (Bennighof and Lehoucq, *An
  automated multilevel substructuring method for eigenspace computation in linear elastodynamics*, SIAM J. Sci.
  Comput. 25, 2004). The subtrees of the tree are reduced to their lowest modes, then the reduced problem is solved.
  It factorises no matrix of the whole model. Its roots are approximate: the lowest ones best, all within a few
  thousandths of the Lanczos roots at its default cut-off (`cutoffFactor` raises it).
- **`outOfCore`**: the Lanczos solver with its factor in scratch files. The same roots as in memory, the memory of the
  factor traded for disk space and for the time of its reads. The scratch files go to `scratchFolder` (`scratch_dir`
  in Python), by default the folder named by the `OOS_SCRATCH` environment variable, else a folder of the system's
  temporary one.

Both take the matrices from the elements, front by front: no matrix of the whole model is assembled. The outputs that
need one (the constraint forces, the modal effective mass, the grid point weight) are not produced, which the run says
in an information diagnostic. Both are refused with a condensation.

The first ten modes of the plate above, on 12 threads of the same machine (sources of this revision; the difference is
the largest relative difference of the ten eigenvalues from the Lanczos solver in memory; a frequency differs by half
of it):

| Solver | 1 000 008 DOF | 2 006 238 DOF | Difference |
| --- | --- | --- | --- |
| `lanczos` | 24 s, 6.5 GB | 73 s, 11.2 GB | reference |
| `substructuring` | 45 s, 2.6 GB | 96 s, 5.5 GB | 1.5 × 10⁻³, 2.6 × 10⁻³ |
| `outOfCore` | 83 s, 2.7 GB | 308 s, 5.2 GB | 4 × 10⁻⁹ |

About 2.5 times less memory for both at one million degrees of freedom, half at two million. In a buckling analysis
the static preload keeps its own factor in memory, which bounds the gain: 4.1 GB for 6.0 GB on the plate of a million
degrees of freedom, where the first factor takes 18 s (`lanczos`), 48 s (`substructuring`, 3 × 10⁻⁶ from it) and 116 s
(`outOfCore`).

## Settings that matter for size

| Setting | Python | JavaScript | C++ (`oos::api::AnalysisSettings`) |
| --- | --- | --- | --- |
| Threads | `run(..., threads=None)` | `run(model, analysis, { threads })` | `threads` (`AUTOMATIC_THREADS`) |
| Memory bound | `memory_bytes` | `memoryBytes` | `memoryBytes` |
| Statics without a factor of the whole model | `streamed=True` | `streamed: true` | `isStreamed` |
| Eigenvalue solver of modes and buckling | `eigen_solver`, `scratch_dir`, `cutoff_factor` | `eigenSolver` | `eigenSolver` |
| Condensed components (Craig-Bampton for modes and buckling) | — | `condensation` | `condensation` |
| Only some subcases | `subcases` | `subcases` | `subcases` |

The streamed path solves a static analysis whose factor would not fit in memory: the equations are solved region by
region and the outputs are made as they come. The condensed path reduces components to their boundaries and can keep
them in a condensation cache, so that the components a run finds there unchanged are not condensed again
([API reference](api.rst)).

## Choosing a formulation for size

`ClassicFormulation`, the default, is the one to use for size: its shells carry six unknowns per grid point. The
`MixedHybridFormulation` resolves each element far more closely (88 unknowns for a quadrilateral at its default degree
4) and costs much more per element; it reaches a given accuracy with many fewer elements, which is where it pays.
`Parameter;MixedHybridDegree;Auto` lowers the degree of the regular elements. Compare the two at the same accuracy,
not on the same mesh ([comparing with another solver](comparing-solvers.md)).

## Reading the report of a run

Every result carries a `report`: the threads, the dense kernel family (`avx2`, `reference`, `wasm128`), the degrees of
freedom of the model and of the largest system factorised, the number of factorisations, the wall time of the run and
of each stage (statics: setup, assembly, loadBasis, factorisation, solve, recovery; modes: setup, assembly,
eigenvalues, recovery), the eigenvalue solver that ran, and the largest memory the run's budget held. The budget
counts the matrices, factors and tables the solver reserves; the process holds a little more.

## Limits

- One machine, shared memory: the solver does not distribute a model over several machines; the memory of the machine
  bounds the size of the factor (the streamed path lifts that bound for statics).
- WebAssembly: the module's memory is limited to 16 GB, and browsers often allow less.
- A static analysis and the preload of a buckling analysis hold their factor in memory (statics has the streamed
  path); the design sensitivities of modes and buckling use the Lanczos solver in memory.
