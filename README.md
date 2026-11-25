# O_o_Solver

Run static, modal, buckling and aeroelastic analyses from JavaScript (WebAssembly), Python or native C++20. An application builds a deck with O_o_Model, passes it to the Solver and reads result arrays directly.

**[Sphinx documentation](https://jcdulas.github.io/O_o_Solver/)** ·
**[Benchmark comparison — Classic, Mixed Auto and commercial solvers](docs/structural-benchmark-comparison.md)** ·
**[External aeroelastic coupling guide](docs/external-aeroelastic-coupling.md)** ·
[Derivative coverage](https://jcdulas.github.io/O_o_Solver/docs/derivatives.html) ·
[API reference](https://jcdulas.github.io/O_o_Solver/docs/api.html)

## Install

| Interface | Install |
| --- | --- |
| JavaScript / WebAssembly | Extract the [Model ZIP](https://github.com/jcdulas/O_o_Model/releases/download/v0.2.3/O_o_Model-wasm.zip) and [Solver ZIP](https://github.com/jcdulas/O_o_Solver/releases/download/v0.2.3/O_o_Solver-wasm.zip) into `model/` and `solver/`, respectively, next to your script. Use Node.js 24+ or a browser with Memory64 support. |
| Python | Run `python -m pip install oo-solver==0.2.3` to get the Solver and its `oo-model` dependency. |
| C++20 / vcpkg | Add `oo-solver` from the [O_o vcpkg registry](https://github.com/jcdulas/O_o_vcpkg) and link `oos::api`. [Native CMake example](https://github.com/jcdulas/O_o_vcpkg/tree/main/examples/rod) |

Applications can link an external optimizer to the [JavaScript, Python or C++ evaluation API](docs/optimization.md). The [client benchmark](docs/benchmarks/optimizer-api/README.md) exercises all three interfaces on identical inputs.

## Couple an external aerodynamic solver

Use Solver 0.2.3 to couple a structural model to an external aerodynamic solver. Prepare the transfer once per model geometry.

| Operation | JavaScript / WebAssembly | Python | C++20 |
| --- | --- | --- | --- |
| Prepare the coupling | `const t = model.aeroTransfer()` | `t = model.aero_transfer()` | `oos::api::AeroCoupling t(snapshot)` |
| Apply box forces and solve SOL 101 | `const s = t.solveStatic(1, f)` | `s = t.solve_static(1, f)` | `auto s = t.solveStatic(1, f)` |
| Project SOL 103 modes onto boxes | `const m = t.solveModes(2)` | `m = t.solve_modes(2)` | `auto m = t.solveModes(2)` |
| Project new forces onto the retained modes | `m.modalBasis.generalizedLoads(f)` | `m["modal_basis"].generalized_loads(f)` | `m.modalBasis->generalizedLoads(f)` |
| Reuse the SOL 101 preload in SOL 105 | `t.solveBuckling(2, s.preload)` | `t.solve_buckling(2, s["preload"])` | `t.solveBuckling(2, *s.preload)` |

Here `f` contains one signed normal force per box in the order reported by the transfer. The external solver reads the box displacements and slopes after each static solve and controls convergence. SOL 144 provides the built-in trim calculation. The [four-box RBE3 wing example](docs/external-aeroelastic-coupling.md) includes complete calls in all three languages, handle lifetimes and numerical results.

## Run linear statics

Build a two-node rod in memory. Its length is 3, its area is 2, and its Young’s modulus is 1000. A force of 10 at the free end gives an axial displacement of `FL/(EA) = 0.015`. All three examples find the displacement by node ID.

**JavaScript / WebAssembly**

```javascript
import { loadOoModel } from "./model/index.js";
import { loadOosSolver } from "./solver/index.js";

const api = await loadOoModel();
const solver = await loadOosSolver();
const deck = api.createModel();
let snapshot;
try {
  deck.setSolution("LinearStatics");
  deck.caseControl.global.set("SinglePointConstrSelection", 1);
  deck.caseControl.global.set("DisplacementOutput", "All");
  deck.caseControl.addSubcase(1, { commands: [{ name: "StaticLoadSelection", value: 2 }] });
  deck.addCard({ type: "Node", fields: [1, null, 0, 0, 0] });
  deck.addCard({ type: "Node", fields: [2, null, 3, 0, 0] });
  deck.addCard({ type: "Rod2", fields: [1, 1, 1, 2] });
  deck.addCard({ type: "RodProp", fields: [1, 1, 2] });
  deck.addCard({ type: "IsoMat", fields: [1, 1000, null, 0.3, 1] });
  deck.addCard({ type: "SinglePointConstrList", fields: [1, 123456, 1] });
  deck.addCard({ type: "SinglePointConstrList", fields: [1, 23456, 2] });
  deck.addCard({ type: "PointForceLoad", fields: [2, 2, 0, 10, 1, 0, 0] });
  deck.applyDefaults();
  snapshot = solver.modelFromReader(api, deck);
  const result = solver.run(snapshot, "statics");
  const u = result.nodalTables.find(table => table.kind === "displacements");
  console.log(u.values[6 * u.ids.indexOf(2)]); // 0.015
} finally {
  snapshot?.delete();
  deck.delete();
}
```

**Python**

```python
from oo_model import create_model
from oo_solver import model_from_reader

deck = create_model()
deck.set_solution("LinearStatics")
deck.set_case_control([
    {"name": "SinglePointConstrSelection", "fields": ["1"]},
    {"name": "DisplacementOutput", "fields": ["All"]},
    {"name": "SubcaseStart", "fields": ["1"]},
    {"name": "StaticLoadSelection", "fields": ["2"]},
])
deck.add_card("Node", [1, None, 0., 0., 0.])
deck.add_card("Node", [2, None, 3., 0., 0.])
deck.add_card("Rod2", [1, 1, 1, 2])
deck.add_card("RodProp", [1, 1, 2.])
deck.add_card("IsoMat", [1, 1000., None, 0.3, 1.])
deck.add_card("SinglePointConstrList", [1, 123456, 1])
deck.add_card("SinglePointConstrList", [1, 23456, 2])
deck.add_card("PointForceLoad", [2, 2, 0, 10., 1., 0., 0.])
deck.apply_defaults()
snapshot = model_from_reader(deck)
result = snapshot.run("statics")
u = next(table for table in result["nodal_tables"] if table["kind"] == "displacements")
print(u["values"][u["ids"].tolist().index(2), 0])  # 0.015
```

**C++20**

```cpp
#include <oor/api/CardEdits.hpp>
#include <oor/api/ControlEdits.hpp>
#include <oos/api/Analyses.hpp>
#include <oos/api/ModelFromDeck.hpp>

#include <cmath>
#include <iostream>

using Field = oor::model::FieldValue;
Field integer(int value) { return {Field::Kind::Integer, value, 0., {}}; }
Field real(double value) { return {Field::Kind::Real, 0, value, {}}; }

int main()
{
    auto deck = oor::api::Deck::createModel();
    oor::api::setSolution(*deck, "LinearStatics");
    const oor::api::ControlRecord controls[] = {
        {"SinglePointConstrSelection", {"1"}}, {"DisplacementOutput", {"All"}},
        {"SubcaseStart", {"1"}}, {"StaticLoadSelection", {"2"}}
    };
    oor::api::setCaseControl(*deck, controls);
    const oor::api::CardInput cards[] = {
        {"Node", {integer(1), {}, real(0), real(0), real(0)}},
        {"Node", {integer(2), {}, real(3), real(0), real(0)}},
        {"Rod2", {integer(1), integer(1), integer(1), integer(2)}},
        {"RodProp", {integer(1), integer(1), real(2)}},
        {"IsoMat", {integer(1), real(1000), {}, real(0.3), real(1)}},
        {"SinglePointConstrList", {integer(1), integer(123456), integer(1)}},
        {"SinglePointConstrList", {integer(1), integer(23456), integer(2)}},
        {"PointForceLoad", {integer(2), integer(2), integer(0), real(10), real(1), real(0), real(0)}}
    };
    oor::api::insertCards(*deck, cards);
    deck->applyDefaults();
    const auto snapshot = oos::api::modelFromDeck(deck);
    const auto result = oos::api::runAnalysis(*snapshot, oos::api::Analysis::Statics, {});
    for (const auto& table : result.tables.nodalTables)
        if (table.kind == "displacements")
            for (std::size_t row = 0; row < table.rowCount(); ++row)
                if (table.ids[row] == 2)
                {
                    const double displacement = table.values[6 * row];
                    std::cout << "Axial displacement at node 2: " << displacement << '\n';
                    return std::abs(displacement - 0.015) < 1e-12 ? 0 : 1;
                }
    std::cerr << "Missing displacement at node 2\n";
    return 1;
}
```

JavaScript transfers typed tables between the Model and Solver WASM memories. Python and C++ transfer Arrow buffers within one process. The result arrays are available to the caller immediately after the solve.

All three interfaces support `statics` (SOL 101), `modes` (SOL 103), `buckling` (SOL 105) and `aeroelastic` (SOL 144). They use the same classic and mixed element kernels.

JavaScript also exposes condensation, streamed solutions, external units and superelements. The [API reference](docs/api.rst) describes these advanced analysis settings and the native Python entry points.

## Documentation

[Optimization with JavaScript, Python and C++](docs/optimization.md): native design interfaces and derivatives.
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
In Python and C++, set the per-card choice in the deck before calling `model_from_reader(deck)` or `oos::api::modelFromDeck(deck)`, respectively.
Card names must match `solver.elementFormulations()` exactly. Unknown cards and unavailable formulations
are reported before a calculation. Details: [src/elements/README.md](src/elements/README.md).

`MixedHybridFormulation` is chosen for the whole model, never for one card: every shell, line and solid card takes it
together, with `Parameter;ElementFormulation;MixedHybridFormulation` or
`solver.modelFromReader(oor, model, { elementFormulation: "MixedHybridFormulation", formulationGeometry: {...} })`.
In Python use `model_from_reader(deck, formulation="MixedHybridFormulation")` and in C++ use `oos::api::modelFromDeck(deck, {.elementFormulation = "MixedHybridFormulation"})`. Both read geometry declarations from the deck.
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
node --test --test-concurrency=1 wasm/test/*.test.mjs      # needs the O_o_Model package built in ../O_o_Model/build/wasm/dist
```

The [fast input combination tests](docs/testing.md) enforce a one-second ceiling per case. Use
`node wasm/test/run.mjs` for a compact summary and a JSON timing report.

The package is in `build/wasm/dist` (`oos_solver_core.js`, `.wasm`, `index.js`, `index.d.ts`, `package.json`).
The `WebAssembly` workflow builds it on every push to `main` that changes the sources, runs the test suites on it
(see [continuous integration](docs/testing.md#continuous-integration)) and, when they pass, publishes it as
`O_o_Solver-wasm.zip` in two [releases](https://github.com/jcdulas/O_o_Solver/releases): `wasm-latest`, a pre-release
that follows the branch, and one release kept for each version of `wasm/package.json` (`v0.2.1`, ...), made by the
first build of that version and not changed afterwards; the newest of these is the latest release of the repository.
Raise the version in `wasm/package.json` to keep a new one. Each run also keeps the package 90 days as its
`O_o_Solver-wasm` artifact.

The default WebAssembly build runs on its calling thread. The `wasm-threads` variant enables shared-memory
workers through the same C++ thread pool. Both variants factorise with the dense kernels written on the 128-bit vector
instructions of WebAssembly, which give the bits of the portable reference kernels.

## Licence

MIT, see [LICENSE](LICENSE). The legal choices behind the project are in [LEGAL.md](LEGAL.md).
