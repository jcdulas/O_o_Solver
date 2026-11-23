# O_o_Solver

Solve finite element models for statics, vibration, buckling and aeroelastic trim, and evaluate design sensitivities with the same C++20 library from **JavaScript (WebAssembly)** or **Python (native C++)**.

**[Sphinx documentation](https://jcdulas.github.io/O_o_Solver/)** ·
**[Benchmark comparison — Classic, Mixed Auto and commercial solvers](docs/structural-benchmark-comparison.md)** ·
[Derivative coverage](https://jcdulas.github.io/O_o_Solver/docs/derivatives.html) ·
[API reference](https://jcdulas.github.io/O_o_Solver/docs/api.html)

## Install

| Interface | Install |
| --- | --- |
| JavaScript / WebAssembly | Extract the [Reader ZIP](https://github.com/jcdulas/O_o_Reader/releases/download/v0.2.0/O_o_Reader-wasm.zip) and [Solver ZIP](https://github.com/jcdulas/O_o_Solver/releases/download/v0.2.0/O_o_Solver-wasm.zip) into `reader/` and `solver/`, respectively, next to your script. Use Node.js 24+ or a browser with Memory64 support. |
| Python / native C++ | Run `python -m pip install oo-solver`. [PyPI](https://pypi.org/project/oo-solver/) provides wheels for CPython 3.10–3.14 on Windows x64 and Linux x86-64. pip installs `oo-reader` as a dependency. |

## Run linear statics

Download [axial.oos](https://raw.githubusercontent.com/jcdulas/O_o_Solver/main/docs/examples/axial.oos) next to your script. The rod has length 3, area 2 and Young's modulus 1000. An axial force of 10 at node 2 gives a displacement of **0.015** from `FL/(EA)`, in the length units of the model. Save the JavaScript example as `example.mjs` and run `node example.mjs`, or save the Python example as `example.py` and run `python example.py`.

**JavaScript / WebAssembly**

```javascript
import { readFile } from "node:fs/promises";
import { loadOorReader } from "./reader/index.js";
import { loadOosSolver } from "./solver/index.js";

const reader = await loadOorReader();
const solver = await loadOosSolver();
reader.writeFile("/work/axial.oos", await readFile("axial.oos", "utf8"));
const deck = reader.readModel("/work/axial.oos");
const model = solver.modelFromReader(reader, deck);
try {
  const run = solver.run(model, "statics");
  const u = run.nodalTables.find(t => t.kind === "displacements");
  const row = u.ids.indexOf(2);
  console.log(u.values[6 * row]); // axial displacement at node 2
} finally {
  model.delete();
  deck.delete();
}
```

**Python / native C++**

```python
from oo_reader import read_model
from oo_solver import model_from_reader

deck = read_model("axial.oos")
model = model_from_reader(deck)
run = model.run("statics")
u = next(t for t in run["nodal_tables"] if t["kind"] == "displacements")
row = u["ids"].tolist().index(2)
print(u["values"][row, 0])  # axial displacement at node 2
```

JavaScript returns the six displacement components in a flat typed array. Python returns a read-only NumPy array with one row per node and one column per component. Both examples locate the row by node ID.

Both interfaces accept `statics` (SOL 101), `modes` (SOL 103), `buckling` (SOL 105) and `aeroelastic` (SOL 144). They use the same classic and mixed element kernels.

JavaScript also exposes condensation, streamed solutions, external units and superelements. The [API reference](docs/api.rst) describes these advanced analysis settings and the native Python entry points.

## Documentation

[Optimization with JavaScript and Python](docs/optimization.md): native design interfaces and derivatives.
[External optimization](docs/external-optimization.md): SOL 101, 103, 105 and 144 studies, full-analysis derivatives,
workers and accepted-point archives.
[Bar and beam derivatives](docs/beam-derivatives.md): analytic section, material, mass and recovery rates.

**[Read Sphinx documentation](https://jcdulas.github.io/O_o_Solver/)** · [Architecture with diagrams](docs/architecture.md) · [Documentation index](index.md)

The Documentation workflow publishes the HTML site and provides a downloadable archive.
See [read, download and build instructions](docs/documentation.md).

## Shell elements

ShellQuad4 is the four-node shell of [docs/shell-quad4-formulation.md](docs/shell-quad4-formulation.md) (directors
at the nodes, enhanced membrane strains and curvatures, transverse shear tied at the mid-sides with its residual
bending flexibility, a stabilised rotation about the normal), and ShellTria3 the same element brought to three nodes
([docs/shell-tria3-formulation.md](docs/shell-tria3-formulation.md)), so that two triangles on the grid points of a
quadrilateral behave closely to it; that document gives the measured closeness and its limits. ShellQuad8 and
ShellTria6 are the shells with a director at each node on the quadratic shape functions
([docs/shell-quad8-tria6-formulation.md](docs/shell-quad8-tria6-formulation.md)), with the same conventions: the
normals shared at the grid points, the stabilised rotation about the normal, the free thermal dilatation; the
eight-node element with uniform reduced integration, the six-node element with tied transverse shears and membrane
strains. All are written from methods of the open literature, cited in the documents; the choices of this project
are listed in [src/elements/README.md](src/elements/README.md).

## Element formulations

Each element card can have several formulations. `ClassicFormulation` is the default for every current
card. A choice applies to all elements of one card type. The model sets it with
`Parameter;ShellQuad4Formulation;ClassicFormulation`, or with a global
`SubcaseParameter;ShellQuad4Formulation;ClassicFormulation` before the first subcase.

A program can override that choice with
`solver.modelFromReader(oor, model, { elementFormulations: { ShellQuad4: "ClassicFormulation" } })`.
In Python, set a per-card choice in the input model before calling `model_from_reader(deck)`.
Card names must match `solver.elementFormulations()` exactly. Unknown cards and unavailable formulations
are reported before a calculation. Details: [src/elements/README.md](src/elements/README.md).

`MixedHybridFormulation` is chosen for the whole model, never for one card: every shell, line and solid card takes it
together, with `Parameter;ElementFormulation;MixedHybridFormulation` or
`solver.modelFromReader(oor, model, { elementFormulation: "MixedHybridFormulation", formulationGeometry: {...} })`.
In Python use `model_from_reader(deck, formulation="MixedHybridFormulation")` and put geometry declarations in the deck.
Its shells and solids read their geometry from the mesh. The model's `ShellNormalAngle` decides where the surface
folds, and the declarations correct it edge by edge (`formulationGeometry`, or the `SurfaceEdge`, `SurfaceFacet` and
`SurfaceLine` cards of the deck: smooth, sharp and raw edges, raw elements, mesh lines); a declaration is mandatory,
and an empty one says that the angle decides every edge. `Model.surfaceEdges()` gives what was read at every edge. It
takes every path: full, condensed, streamed, external units, superelements and the design derivatives.
See [docs/mixed-hybrid-formulation-spec.md](docs/mixed-hybrid-formulation-spec.md).

## Layout

| Folder | What |
| --- | --- |
| `include/oos/<module>/`, `src/<module>/` | the library, in modules that each use only the ones before them: `base` → `linalg` → `input` → `elements` → `assembly` → `reduction` → `aero` → `output` → `solutions` → `design` → `api` |
| `python/` | Native Python extension, public package and installed-wheel tests |
| `wasm/` | the WebAssembly module: the embind interface (`src/Bindings.cpp`), the JavaScript wrapper (`js/`), its Node.js test (`test/`) |
| `cmake/` | compiler warnings and the dense kernel families |
| `docs/` | the table package format |

Each module has a `README.md` in `src/<module>/` that says what it holds and where to start reading.

## Build

For the JavaScript distribution, use the WebAssembly build below. For a Python wheel, see the [native build and installation guide](docs/python.md#build-and-publish).

For the shared-memory build, use the `wasm-threads` preset and [threading guide](docs/threading.md).

Native (Windows: MSYS2 UCRT64 first on the PATH, `C:\msys64\ucrt64\bin`; Linux: GCC 12 or later):

```bash
cmake --preset release
cmake --build build/release -j
```

The `Native` workflow makes this build with GCC 12 and runs the native checks of the element kernels
(`-DOOS_BUILD_NATIVE_CHECKS=ON`, `build/release/checks/oos_native_checks`).

WebAssembly (64-bit WebAssembly, Memory64: Node.js 24 or a browser that supports it; the Emscripten SDK on the PATH):

```bash
emcmake cmake -S . -B build/wasm -G Ninja -DCMAKE_BUILD_TYPE=Release -DOOS_BUILD_WASM=ON -DOOS_WARNINGS_AS_ERRORS=OFF
cmake --build build/wasm -j
node --test --test-concurrency=1 wasm/test/*.test.mjs      # needs the O_o_Reader package built in ../O_o_Reader/build/wasm/dist
```

The [fast input combination tests](docs/testing.md) enforce a one-second ceiling per case. Use
`node wasm/test/run.mjs` for a compact summary and a JSON timing report.

The package is in `build/wasm/dist` (`oos_solver_core.js`, `.wasm`, `index.js`, `index.d.ts`, `package.json`).
The `WebAssembly` workflow builds it on every push to `main` that changes the sources, runs the test suites on it
(see [continuous integration](docs/testing.md#continuous-integration)) and, when they pass, publishes it as
`O_o_Solver-wasm.zip` in two [releases](https://github.com/jcdulas/O_o_Solver/releases): `wasm-latest`, a pre-release
that follows the branch, and one release kept for each version of `wasm/package.json` (`v0.2.0`, ...), made by the
first build of that version and not changed afterwards; the newest of these is the latest release of the repository.
Raise the version in `wasm/package.json` to keep a new one. Each run also keeps the package 90 days as its
`O_o_Solver-wasm` artifact.

The default WebAssembly build runs on its calling thread. The `wasm-threads` variant enables shared-memory
workers through the same C++ thread pool. Both variants factorise with the dense kernels written on the 128-bit vector
instructions of WebAssembly, which give the bits of the portable reference kernels.

## Licence

MIT, see [LICENSE](LICENSE). The legal choices behind the project are in [LEGAL.md](LEGAL.md).
