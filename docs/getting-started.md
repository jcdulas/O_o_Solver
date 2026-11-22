# Getting started

O_o_Solver receives typed model tables from O_o_Reader. Its JavaScript entry points expose linear statics,
normal modes, linear buckling and [JavaScript-driven optimization](optimization.md). The C++ library has additional entry points described in the
[solution guide](../src/solutions/README.md) and [design guide](../src/design/README.md).

## Build the packages

Build [O_o_Reader](https://github.com/jcdulas/O_o_Reader/blob/main/README.md) first, then follow the solver's
[build instructions](../README.md#build). Keep the two checkouts next to each other for the example below.
The default packages are written to `build/wasm/dist/` in each repository. The solver also provides a
[shared-memory variant](threading.md) in `build/wasm-threads/dist/`.

Use Node.js 24 or later, or a browser supporting Memory64. The reader and solver exchange tables through the
[table package contract](table-package.md); use compatible builds of the two projects.

## Run linear statics

See [static input checks and added mass](static-inputs.md) for support diagnostics and per-subcase mass selections.

This two-node bar is fixed at its first node and loaded axially at its second node. Save the script as
`solve-model.mjs` in the solver repository root:

~~~javascript
import { loadOorReader } from "../O_o_Reader/build/wasm/dist/index.js";
import { loadOosSolver } from "./build/wasm/dist/index.js";

const oor = await loadOorReader();
const solver = await loadOosSolver();
const text = [
  "[executive]", "SolutionSequence;101",
  "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
  "StaticLoadSelection;2", "DisplacementOutput;All",
  "[model]", "Node;1;;0.;0.;0.", "Node;2;;2.;0.;0.",
  "Bar2;1;1;1;2;0.;;1.;0.",
  "BarProp;1;1;0.01;1.e-6;2.e-6;1.e-6",
  "IsoMat;1;7.e10;;0.3",
  "SinglePointConstrList;1;123456;1",
  "PointForceLoad;2;2;;100.;1.;0.;0.",
].join("\n") + "\n";

oor.writeFile("/work/bar.oos", text);
const deck = oor.readModel("/work/bar.oos");
const model = solver.modelFromReader(oor, deck);
try {
  const support = model.checkSupport("statics");
  if (!support.isComplete) {
    throw new Error(support.refusals.map((item) => item.text).join("\n"));
  }
  const run = solver.run(model, "statics");
  const displacements = run.nodalTables.find((table) => table.kind === "displacements");
  if (!displacements) throw new Error("Missing displacement results.");
  const row = Array.from(displacements.ids).indexOf(2);
  if (row < 0) throw new Error("Missing results for node 2.");
  const axialDisplacement = displacements.values[row * 6];
  console.log("Axial displacement:", axialDisplacement);
  const expected = 100 * 2 / (7e10 * 0.01);
  if (!Number.isFinite(axialDisplacement) || Math.abs(axialDisplacement / expected - 1) > 1e-10) {
    throw new Error("Unexpected axial displacement.");
  }
} finally {
  model.delete();
  deck.delete();
}
~~~

Run `node solve-model.mjs`. The expected axial displacement is approximately `2.857142857e-7` in the consistent
length units of the model. Numerical input uses a consistent unit system; the example does not perform unit conversion.

## Interpret support and results

Call `checkSupport` for the requested analysis before solving. Inspect both `refusals` and `ignored`; a loaded model
does not imply that every card or option participates in the calculation.

Result arrays are row-major. A nodal displacement row contains six components; element tables describe their own
component order. Match rows by their `ids` rather than assuming the input order.
See the [API declarations](api.rst) for shapes, ownership and result fields.

## Element formulations

Use `solver.elementFormulations()` to discover supported card types and their formulation names.
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
the selected elements in `selectedQuarticShells`. Modes, buckling, trim and static paths that require a fixed layout
use the degrees selected from geometry.

```js
const result = solver.run(model, "statics");
```

The [module guide](modules.md) covers assembly, linear algebra, reduction, output and design calculations.
