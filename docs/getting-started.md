# Getting started

Run static, modal, buckling and aeroelastic analyses through the O_o_Solver API in JavaScript (WebAssembly) or Python (native C++). An application builds a deck with O_o_Model, passes it to the Solver and reads result arrays directly.

(build-the-packages)=
## Install the packages

| Interface | Install |
| --- | --- |
| JavaScript / WebAssembly | Extract the [Model ZIP](https://github.com/jcdulas/O_o_Model/releases/download/v0.2.1/O_o_Model-wasm.zip) and [Solver ZIP](https://github.com/jcdulas/O_o_Solver/releases/download/v0.2.1/O_o_Solver-wasm.zip) into `model/` and `solver/`, respectively, next to your script. Use Node.js 24+ or a browser with Memory64 support. |
| Python / native C++ | Run `python -m pip install oo-solver==0.2.1` to get the Solver and its `oo-model` dependency. |

## Run linear statics

Build a two-node rod in memory. Its length is 3, its area is 2, and its Young’s modulus is 1000. A force of 10 at the free end gives an axial displacement of `FL/(EA) = 0.015`. Both examples find the displacement by node ID.

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

**Python / native C++**

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

JavaScript transfers typed tables between the Model and Solver WASM memories. Python transfers Arrow buffers within one process. The result arrays are available to the caller immediately after the solve.

## Solve a model built in memory

The [JavaScript / WebAssembly example](examples/memory-plate.mjs) builds four nodes in a typed batch and passes the
deck to `solver.modelFromReader(reader, deck)`. Import its `solvePlate` function and call it with your initialized
Model and Solver. The [Python / native C++ example](examples/memory_plate.py) uses NumPy columns and
`model_from_reader(deck)`. Run it with a Model build exposing `create_model`.

Both return axial displacement by node ID for a unit square, thickness 0.01, Young's modulus 100000 and zero
Poisson's ratio. A total tensile force of 1 is split between nodes 2 and 3. The uniform extension is
`F L / (E t w) = 0.001`. The two examples work with a Model built without file adapters. With both native packages installed,
run the same check against local JavaScript distributions from the Solver checkout:

~~~sh
python checks/verify_memory_plate.py path/to/model/index.js path/to/solver/index.js
~~~

## Interpret support and results

In JavaScript, call `model.checkSupport` for the requested analysis before solving. Inspect both `refusals` and `ignored`; a loaded model
does not imply that every card or option participates in the calculation.

In Python, `model.run("statics")["diagnostics"]` gives the analysis diagnostics. Result arrays are row-major. A nodal displacement row contains six components; element tables describe their own
component order. Match rows by their `ids` rather than assuming the input order.
See the [API declarations](api.rst) for shapes, ownership and result fields.

## Element formulations

In Python, use `model_from_reader(deck, formulation="MixedHybridFormulation")` for a whole-model choice and set per-card parameters and surface declarations in the input deck.
In JavaScript, use `solver.elementFormulations()` to discover supported card types and their formulation names.
An `elementFormulations` choice passed to `modelFromReader` applies to all elements of that card type and takes
precedence over the model parameter. The [element guide](../src/elements/README.md) describes the formulations,
geometric conventions and current limits.

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
the selected elements in `selectedQuarticShells` (JavaScript) or `selected_quartic_shells` (Python). Modes, buckling, trim and static paths that require a fixed layout
use the degrees selected from geometry.

**JavaScript / WebAssembly**

```javascript
const result = solver.run(model, "statics");
```

**Python / native C++**

```python
result = model.run("statics")
```

The [module guide](modules.md) covers assembly, linear algebra, reduction, output and design calculations.
