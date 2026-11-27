# Getting started

Run static, modal, buckling and aeroelastic analyses through the O_o_Solver API in JavaScript (WebAssembly), Python or C++20. An application builds a deck with O_o_Model, passes it to the Solver and reads result arrays directly.

(build-the-packages)=
## Install the packages

| Interface | Install |
| --- | --- |
| JavaScript / WebAssembly | Extract the [Model ZIP](https://github.com/jcdulas/O_o_Model/releases/download/v0.2.4/O_o_Model-wasm.zip) and [Solver ZIP](https://github.com/jcdulas/O_o_Solver/releases/download/v0.2.6/O_o_Solver-wasm.zip) into `model/` and `solver/`, respectively, next to your script. Use Node.js 24+ or a browser with Memory64 support. |
| Python | Run `python -m pip install oo-solver==0.2.6` to get the Solver and its `oo-model` dependency. |
| C++20 / vcpkg | Add `oo-solver` from the [O_o vcpkg registry](https://github.com/jcdulas/O_o_vcpkg) and link `oos::api`. [Native CMake example](https://github.com/jcdulas/O_o_vcpkg/tree/main/examples/rod) |

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

## Solve a model built in memory

The [JavaScript / WebAssembly example](examples/memory-plate.mjs) builds four nodes in a typed batch and passes the
deck to `solver.modelFromReader(reader, deck)`. Import its `solvePlate` function and call it with your initialized
Model and Solver. The [Python example](examples/memory_plate.py) uses NumPy columns and
`model_from_reader(deck)`. Run it with a Model build exposing `create_model`.

The [C++20 example](examples/memory_plate.cpp) builds the same plate with typed card fields. Build it as the executable source in the linked native CMake example.

All three return axial displacement by node ID for a unit square, thickness 0.01, Young's modulus 100000 and zero
Poisson's ratio. A total tensile force of 1 is split between nodes 2 and 3. The uniform extension is
`F L / (E t w) = 0.001`. The three examples work with a Model built without file adapters. With the Python packages installed and the C++ example built, check all three interfaces from the Solver checkout:

~~~sh
python checks/verify_memory_plate.py path/to/model/index.js path/to/solver/index.js --cpp path/to/plate-executable
~~~

## Interpret support and results

In JavaScript, call `model.checkSupport` for the requested analysis before solving. Inspect both `refusals` and `ignored`; a loaded model
does not imply that every card or option participates in the calculation.

In Python, `model.run("statics")["diagnostics"]` gives the analysis diagnostics. Result arrays are row-major. A nodal displacement row contains six components; element tables describe their own
component order. Match rows by their `ids` rather than assuming the input order.
In C++, inspect the refusals and ignored entries from `oos::api::checkSupport(*model, oos::api::Analysis::Statics)` and the diagnostics in the analysis output.
See the [API declarations](api.rst) for shapes, ownership and result fields.

## Element formulations

In JavaScript, `solver.elementFormulations()` lists the available formulations. Set geometry declarations in the deck and select the mixed formulation when building the snapshot.

| JavaScript / WebAssembly | Python | C++20 |
| --- | --- | --- |
| `solver.modelFromReader(api, deck, { elementFormulation: "MixedHybridFormulation" })` | `model_from_reader(deck, formulation="MixedHybridFormulation")` | `oos::api::modelFromDeck(deck, {.elementFormulation = "MixedHybridFormulation"})` |

Set a per-card selection in the input deck in any language. JavaScript also accepts per-card overrides in `elementFormulations`. The [element guide](../src/elements/README.md) describes the geometric conventions and limits.

The `elementFormulation` option (or the `ElementFormulation` model parameter) chooses the formulation of the whole
model. `MixedHybridFormulation` puts every shell, line and solid card in the mixed hybrid formulation together; it
needs geometry declarations, from the `formulationGeometry` option or from the `SurfaceEdge`, `SurfaceFacet` and
`SurfaceLine` cards of the deck: the smooth, sharp and raw edges of the mesh, its raw elements and its mesh lines. Its
elements default to degree 4. The model parameter `MixedHybridDegree` also accepts a uniform 2 or 3.
With `Auto`, shells start at degree 3. The solver enriches the outline of thin curved patches, strongly distorted
cells, cells whose sagitta exceeds their thickness, and shell edges shared with solids. One shell formulation handles the degrees on both sides of
each edge. An empty declaration (`formulationGeometry: {}`, or `Parameter;GeometryDeclarations;Given` in the deck) leaves every edge
to the model's `ShellNormalAngle`. `model.surfaceEdges()` gives what was read at every edge (see the
[specification](mixed-hybrid-formulation-spec.md), sections 4.4 and 10.5).

Set the model parameter once to let the solver choose the shell degrees.

```text
Parameter;MixedHybridDegree;Auto
```

In a full static analysis, Auto checks the loaded solution and solves again only if additional cells need degree 4.
For multiple load cases it uses one final layout. The diagnostic gives the number of solves and the result lists
the selected elements in `selectedQuarticShells` (JavaScript), `selected_quartic_shells` (Python) or `selectedQuarticShells` (C++). Modes, buckling, trim and static paths that require a fixed layout
use the degrees selected from geometry.

**JavaScript / WebAssembly**

```javascript
const result = solver.run(model, "statics");
```

**Python**

```python
result = model.run("statics")
```

**C++20**

```cpp
const auto result = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {});
```

### Loads and supports along the edges of mixed shells

A mixed hybrid shell resolves what a load or a support at grid points really states. Two readings follow, both
described in the [specification](mixed-hybrid-formulation-spec.md) (sections 3.6, 5.6 and 7).

- **Forces along an edge.** Forces at both grid points of an edge of the outline of the shells are read as the
  traction they stand for, with the same resultant force and moment; the information `forces-read-along-edges` gives
  the count. A force at a grid point alone, or inside the shells, stays a concentrated force: the grid point then
  moves more than the section around it, so read the displacement a few elements away.
- **Soft supports.** A simple support that holds the deflection alone leaves the rotation about the normal of the
  edge free. A shell with transverse shear then has an edge layer a third of the thickness long, and the plate is
  softer than the thin-plate formulas say, by about the thickness over its size: these formulas are those of the
  hard support, which holds that rotation too. The warning `edge-layer-not-resolved` counts the supported edges
  whose elements are deeper than four thicknesses. A row of elements about one thickness wide along these edges
  resolves the layer on any mesh.

To compare a result with a thin-plate formula, hold the rotation about the normal of each supported edge, or leave
the transverse shear material of the shell property blank. To compare two solvers on a soft support, compare each
with the limit of the model on refined meshes.

The [module guide](modules.md) covers assembly, linear algebra, reduction, output and design calculations.
