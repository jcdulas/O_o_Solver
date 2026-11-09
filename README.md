# O_o_Solver

A linear finite element solver written in C++20 and compiled to WebAssembly for the web. It solves models written in
the O_o model format (`.oos`, O_o_Reader `docs/model-format.md`; O_o_Converter converts decks of the Nastran bulk data
format to it): linear statics, normal modes, linear buckling and design evaluations with derivatives for a
JavaScript-driven optimization. The C++ library also provides static aeroelastic trim and condensation of components.

It does not read model files itself. The O_o_Reader module reads the model, and the solver takes the reader's tables:

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
```

The interface and its diagnostics use the same neutral names as the model: card types such as `ShellQuad4`,
table kinds such as `singlePointConstrForces`, and field names such as `bendingMomentA1`.

The whole interface: [wasm/js/index.d.ts](wasm/js/index.d.ts). How a table crosses from the reader's module to the
solver's: [docs/table-package.md](docs/table-package.md).

## Documentation

[JavaScript optimization](docs/optimization.md): design interfaces, derivatives and accepted model updates.

**[Read online](https://jcdulas.github.io/O_o_Solver/)** · [Architecture with diagrams](docs/architecture.md) · [Documentation index](index.md)

The Documentation workflow publishes the HTML site and provides a downloadable archive.
See [read, download and build instructions](docs/documentation.md).

## Shell elements

ShellQuad4 and ShellTria3 follow the formulation of NASTRAN-95, the program NASA released as open source in 2015
(degenerated shell with assumed edge transverse shear and shear correction factors for the QUAD4; constant strain
membrane, linear bending and fitted edge shear for the TRIA3; no drilling stiffness). The implementation is written from
a mathematical description of that formulation, not translated from its Fortran; the differences are listed in
[src/elements/README.md](src/elements/README.md).

## Element formulations

Each element card can have several formulations. `ClassicFormulation` is the default for every current
card. A choice applies to all elements of one card type. The model sets it with
`Parameter;ShellQuad4Formulation;ClassicFormulation`, or with a global
`SubcaseParameter;ShellQuad4Formulation;ClassicFormulation` before the first subcase.

A program can override that choice with
`solver.modelFromReader(oor, model, { elementFormulations: { ShellQuad4: "ClassicFormulation" } })`.
Card names must match `solver.elementFormulations()` exactly. Unknown cards and unavailable formulations
are reported before a calculation. Details: [src/elements/README.md](src/elements/README.md).

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
node --test wasm/test/*.test.mjs      # needs the O_o_Reader package built in ../O_o_Reader/build/wasm/dist
```

The package is in `build/wasm/dist` (`oos_solver_core.js`, `.wasm`, `index.js`, `index.d.ts`, `package.json`).

The default WebAssembly build runs on its calling thread. The `wasm-threads` variant enables shared-memory
workers through the same C++ thread pool. Both variants use the portable reference dense kernels.

## Licence

MIT, see [LICENSE](LICENSE). The legal choices behind the project are in [LEGAL.md](LEGAL.md).
