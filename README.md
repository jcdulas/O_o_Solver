# O_o_Solver

**[Sphinx documentation](https://jcdulas.github.io/O_o_Solver/)** ·
[Derivative coverage](https://jcdulas.github.io/O_o_Solver/docs/derivatives.html) ·
[API reference](https://jcdulas.github.io/O_o_Solver/docs/api.html)

A linear finite element solver written in C++20 and compiled to WebAssembly for the web. It solves models supplied
as typed table packages by O_o_Reader: linear statics (SOL 101), normal modes (SOL 103), linear buckling (SOL 105),
static aeroelastic trim (SOL 144) and design evaluations driven by JavaScript. A static analysis can condense the
components of the model onto their boundaries (the condensed path), with the results of the full path.

O_o_Reader loads native `.oos` models or `.ooh5` model archives, and the solver takes the reader's tables:

```js
import { loadOorReader } from "oor-reader";
import { loadOosSolver } from "oos-solver";

const oor = await loadOorReader();
const solver = await loadOosSolver();

oor.writeFile("/work/plate.oos", modelText);
const deck = oor.readModel("/work/plate.oos");
const model = solver.modelFromReader(oor, deck);   // the reader's tables, handed over as table packages
solver.elementFormulations();                      // [{ card: "ShellQuad4", formulations }], the default first

model.checkSupport("statics");                     // { supported, refusals, ignored, isComplete }
const run = solver.run(model, "statics");          // or "modes", "buckling"
run.nodalTables[0];                                // { kind: "displacements", subcase, ids, values (rows x 6), ... }
run.elementTables;                                 // stress, strain, forces, by element type ("ShellQuad4")
run.eigenvalues;                                   // modes and buckling: one table per subcase
run.diagnostics;                                   // warnings, with the card and field they are about

// The condensed path: the reader's component tables, then each component condensed onto its boundary.
const parts = solver.modelFromReader(oor, deck, { condensation: true });
solver.run(parts, "statics", { condensation: "all" }).condensation; // { condensedUnits, fullUnits, interfaceDofs }
solver.run(parts, "modes", { condensation: { units: "all", modeCount: 40 } }); // Craig-Bampton; "buckling" too
const cache = solver.condensationCache();          // units condensed once for the runs given it
solver.run(parts, "statics", { condensation: { units: "all", cache, memoryMode: "low_memory" } });

solver.run(model, "statics", { streamed: true });  // the full path without a factor of the whole model
solver.run(model, "statics", { externalUnits: [{ zone, labels, stiffness }] }); // zones replaced by matrices
solver.runSuperelements(residual, [{ id: 1, model: part, boundary: [[101, 101]] }], "statics");
```

The interface and its diagnostics use the same neutral names as the model: card types such as `ShellQuad4`,
table kinds such as `singlePointConstrForces`, and field names such as `bendingMomentA1`.

The whole interface: [wasm/js/index.d.ts](wasm/js/index.d.ts). How a table crosses from the reader's module to the
solver's: [docs/table-package.md](docs/table-package.md).

## Documentation

[JavaScript optimization](docs/optimization.md): native design interfaces and derivatives.
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
Card names must match `solver.elementFormulations()` exactly. Unknown cards and unavailable formulations
are reported before a calculation. Details: [src/elements/README.md](src/elements/README.md).

`MixedHybridFormulation` is chosen for the whole model, never for one card: every shell, line and solid card takes it
together, with `Parameter;ElementFormulation;MixedHybridFormulation` or
`solver.modelFromReader(oor, model, { elementFormulation: "MixedHybridFormulation", formulationGeometry: {...} })`.
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
| `wasm/` | the WebAssembly module: the embind interface (`src/Bindings.cpp`), the JavaScript wrapper (`js/`), its Node.js test (`test/`) |
| `cmake/` | compiler warnings and the dense kernel families |
| `docs/` | the table package format |

Each module has a `README.md` in `src/<module>/` that says what it holds and where to start reading.

## Build

For the shared-memory build, use the `wasm-threads` preset and [threading guide](docs/threading.md).

Native (Windows: MSYS2 UCRT64 first on the PATH, `C:\msys64\ucrt64\bin`; Linux: GCC 12 or later):

```bash
cmake --preset release
cmake --build build/release -j
```

WebAssembly (64-bit WebAssembly, Memory64: Node.js 24 or a browser that supports it; the Emscripten SDK on the PATH):

```bash
emcmake cmake -S . -B build/wasm -G Ninja -DCMAKE_BUILD_TYPE=Release -DOOS_BUILD_WASM=ON -DOOS_WARNINGS_AS_ERRORS=OFF
cmake --build build/wasm -j
node --test --test-concurrency=1 wasm/test/*.test.mjs      # needs the O_o_Reader package built in ../O_o_Reader/build/wasm/dist
```

The [fast input combination tests](docs/testing.md) enforce a one-second ceiling per case. Use
`node wasm/test/run.mjs` for a compact summary and a JSON timing report.

The package is in `build/wasm/dist` (`oos_solver_core.js`, `.wasm`, `index.js`, `index.d.ts`, `package.json`).

The default WebAssembly build runs on its calling thread. The `wasm-threads` variant enables shared-memory
workers through the same C++ thread pool. Both variants use the portable reference dense kernels.

## Licence

MIT, see [LICENSE](LICENSE). The legal choices behind the project are in [LEGAL.md](LEGAL.md).
