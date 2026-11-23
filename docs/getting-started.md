# Getting started

Solve finite element models for statics, vibration, buckling and aeroelastic trim, and evaluate design sensitivities with the same C++20 library from **JavaScript (WebAssembly)** or **Python (native C++)**.

(build-the-packages)=
## Install the packages

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
