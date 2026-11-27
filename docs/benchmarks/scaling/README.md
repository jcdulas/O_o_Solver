# Scaling benchmark

How the time and the memory of an analysis grow with the size of a model and with the number of threads. Four
parametric models are refined from about 10 000 to about 2 000 000 degrees of freedom; each is run in linear statics,
in normal modes and in linear buckling, on one thread and on every logical core. Each run also returns one value of
its results that is compared with a reference from plate, shell or beam theory, so that the tables show that the
models are right as well as how long they take.

The scripts use the published Python packages `oo-solver` and `oo-model` and nothing of the repository, so that anyone
can run them on their own machine.

## Files

| File | Purpose |
| --- | --- |
| [models.py](models.py) | The four models, written in the `.oos` text format, and their reference values. |
| [run.py](run.py) | The runner: one process per run, results in a JSON file. |
| [report.py](report.py) | Markdown tables from the JSON file. |
| [results.md](results.md), [results.json](results.json) | The measurements published in [Large models](../../large-models.md), and their raw data. |

## Running it

From a copy of the repository (or of this folder; adapt the paths):

```bash
python -m venv scaling-venv
source scaling-venv/bin/activate            # Windows: scaling-venv\Scripts\activate
python -m pip install oo-solver psutil      # oo-model comes with oo-solver
python docs/benchmarks/scaling/run.py --output scaling-results.json
python docs/benchmarks/scaling/report.py scaling-results.json --output scaling-results.md
```

The full benchmark (every model, every size up to 2 000 000 DOF, three analyses, two thread counts) takes hours and
needs a machine with a large memory: the block's memory grows fastest. A run that would exceed the memory limit is
stopped before the machine runs out, and the larger sizes of that model are skipped. Run it on a machine with nothing
else running, since other work changes the times. A first try on the small sizes takes a few minutes:

```bash
python docs/benchmarks/scaling/run.py --dofs 1e4 3e4 --output small.json
```

Options of `run.py`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--models` | all | `plate`, `cylinder`, `wingbox`, `block`. |
| `--dofs` | `1e4 3e4 1e5 3e5 1e6 2e6` | Target DOF counts; each model takes the size whose DOF count is nearest. |
| `--sizes` | | Size parameters `n` instead of DOF counts (`python models.py --ladder` lists them). |
| `--analyses` | all | `statics`, `modes`, `buckling`. |
| `--threads` | `1` and the logical cores | Thread counts passed to `run(..., threads=N)`. |
| `--formulation` | the solver's default | Element formulation passed to `model_from_reader`, e.g. `MixedHybridFormulation`. |
| `--repeat` | 3 | Samples of a configuration whose runs all take less than `--repeat-below` seconds (10); the median is kept. |
| `--time-limit` | 1800 s | A run longer than this (read + snapshot + analysis) is the last size of its model and analysis. |
| `--kill-after` | 3 × time limit | A run is stopped after this long; the larger sizes are skipped. |
| `--memory-limit` | 0.9 | A run whose memory exceeds this fraction of the RAM is stopped; the larger sizes are skipped. |
| `--output` | `scaling-results.json` | The JSON file, rewritten after every run. |
| `--resume` | | Keep the runs already in the output file and measure only the missing ones. |
| `--work-dir`, `--keep-models` | a temporary directory | Where the generated decks are written, and whether they are kept. |

`python models.py plate 40 statics plate.oos` writes one deck to look at; `python models.py --check DIRECTORY` writes
every deck near 10 000 DOF, reads each with `oo_model` and lists its diagnostics (there should be none).

## What is measured

Each run is a fresh Python process, so that its peak memory belongs to that run alone. The process:

1. reads the deck: `deck = oo_model.read_model(path)` (*read*);
2. builds the solver's model: `model = oo_solver.model_from_reader(deck)` (*snapshot*), then releases the deck;
3. runs the analysis: `result = model.run(analysis, threads=N)` (*run*), with displacements requested at every grid
   point (no stresses);
4. takes the check value from the result.

| Column | Meaning |
| --- | --- |
| DOF | Free degrees of freedom: 6 per shell grid point, 3 per solid grid point, less the held components. |
| Read + snapshot | Seconds to read the deck and build the solver's model, on one thread. |
| 1 thread, N threads | Wall time, in seconds, of `run` with `threads=1` and `threads=N`. |
| Speed-up | Time on 1 thread divided by the time on N threads. |
| Peak memory | Peak resident memory of the process (working set on Windows, `ru_maxrss` on Linux and macOS), on 1 thread and on N threads. It includes Python and the imported packages, about 40 MiB. |
| Check | Relative difference between the check value and its reference (below). |

The JSON file keeps, for every run, the three times (the median when repeated, with every sample), the peak memory
after each step, the check value and its reference, the frequencies or buckling factors found, the diagnostics of the
deck and of the solver, and the `report` entry of the result when the solver gives one. Its header records the
processor, the cores, the memory, the operating system, the Python and package versions and the date.

## The models

Every model is a steel or aluminium structure in SI units, meshed with elements of equal sides. The decks set
`Parameter;GeometryDeclarations;Given` and declare no edge: the fold angle of the model decides every edge, which is
what the mixed formulation needs to read the surfaces when `--formulation MixedHybridFormulation` is given.

| Model | Mesh | Supports | Statics | Buckling |
| --- | --- | --- | --- | --- |
| `plate` | Square plate 1 × 1 m, 10 mm thick, n × n ShellQuad4 | Simply supported: each side holds the deflection and the rotation that would tilt the side out of its line; the in-plane rigid motions are held at one side | Uniform pressure | Uniform compression on one side, reacted by the opposite side |
| `cylinder` | Closed cylinder, radius 1 m, length 2 m, 10 mm thick; n ShellQuad4 around and as many per metre along | End diaphragms (each end ring holds its displacements in its plane); the ring at mid-length, a plane of symmetry, holds the axial displacement | Internal pressure | Uniform axial compression, equal and opposite forces on the two end rings |
| `wingbox` | Box beam 10 m long, 1 m between the spars, 0.25 m deep; skins 4 mm, spars 6 mm, a 3 mm rib at the end of each of 20 bays; n ShellQuad4 across the depth of a spar | Root section clamped | Vertical tip force shared by the two spar webs | The same tip force; the compressed skin buckles between ribs and spars |
| `block` | Block 8 × 1 × 1 m of SolidHexa (8 nodes), n elements across, 8 n along | One end face clamped | Transverse force on the other end face | Axial compression on the other end face |

The modes analysis asks for the first 10 modes of the same model, the buckling analysis for the first buckling factor
above zero (a static subcase for the preload, then the buckling subcase). The plate and the cylinder in compression
are the classic buckling problems; the wing box under its tip load shows local buckling of a stiffened structure.
The block's graph is three-dimensional at every size, so its factorisation fills far more than that of the shells for
the same DOF count.

### References

| Model | Statics | Modes (first frequency) | Buckling (first factor) |
| --- | --- | --- | --- |
| `plate` | Centre deflection, Navier's series for the thin plate [1] | π/a² √(D/ρt) [2] | 4π²D/(a²N) [3] |
| `cylinder` | Mean radial displacement at mid-length, pR²/(Et) | Lowest mode m, n of Sanders' shell theory between diaphragms [4, 5] | Lowest mode m, n of the same theory, the axial force working on the axial gradient of the three displacements as in the solver's differential stiffness |
| `wingbox` | Tip deflection of a Timoshenko beam: bending of skins and spars, shear of the spar webs | Timoshenko cantilever, the ribs' mass spread along the span | none |
| `block` | Tip deflection of a Timoshenko beam, shear coefficient of Cowper [6] | Timoshenko cantilever | Clamped-free column with shear (Engesser) [3] |

The plate and cylinder references are limits of thin plate and shell theory: the check falls to a few hundredths of a
per cent as the mesh is refined (what is left is mostly the transverse shear of a 10 mm shell), except the cylinder's
buckling factor, which approaches its reference from above (+3 % at 10 000 DOF, +0.5 % at 100 000 DOF). The wing box
and the block are compared with beam models, which leave out the restraint of the clamped section and the stiffness
the ribs and the faces add: there the check settles between a few tenths of a per cent and one per cent instead of
tending to zero.

1. S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., McGraw-Hill, 1959, section 30.
2. A. W. Leissa, *Vibration of Plates*, NASA SP-160, 1969.
3. S. P. Timoshenko, J. M. Gere, *Theory of Elastic Stability*, 2nd ed., McGraw-Hill, 1961.
4. J. L. Sanders Jr., *An improved first-approximation theory for thin shells*, NASA TR R-24, 1959.
5. A. W. Leissa, *Vibration of Shells*, NASA SP-288, 1973.
6. G. R. Cowper, The shear coefficient in Timoshenko's beam theory, *J. Appl. Mech.* 33 (1966) 335–340.
