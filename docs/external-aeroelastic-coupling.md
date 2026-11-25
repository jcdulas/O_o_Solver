# Coupling an external aerodynamic solver

O_o_Solver can provide the structural half of an aeroelastic calculation in JavaScript/WebAssembly, Python or C++20. This interface is available in Solver 0.2.2. O_o_Model supplies the structural mesh, aerodynamic boxes, spline attachments and subcases.

| Calculation | What the external driver supplies | What O_o_Solver returns |
| --- | --- | --- |
| SOL 101, static equilibrium | A static subcase and one signed normal force per box | Structural results, box-centre displacement and slope, and the complete static preload |
| SOL 103, free vibration | A modal subcase with an eigenvalue method | Frequencies, mode shapes on the boxes, and a reusable basis for projecting aerodynamic forces onto each mode |
| SOL 105, linear buckling | A buckling subcase and its equilibrated SOL 101 preload | Buckling factors and buckling shapes on the boxes |
| SOL 144, built-in static trim | A trim subcase, trim variables and flight condition | The trimmed aerodynamic loads and structural response. The trim solver applies the same spline transfers internally. |

## Exchange forces and motion

Read `boxIds` (Python `box_ids`) once and assemble every load vector in that order. If the aerodynamic solver produces a pressure jump, multiply by the box area and choose the sign so a positive force acts along the box normal. The load acts at the box force point. The motion returned for each box consists of `plunge`, its centre displacement along the normal in model length units, and `slope`, its dimensionless chordwise slope.

In a static coupling loop, the external driver updates the forces from the current box motion and calls SOL 101 again. Each call solves a complete static case and adds the supplied forces to the loads selected in that subcase. The driver decides when the equilibrium has converged. Pass that iteration's preload directly to SOL 105. It retains the full structural displacement field, including internal degrees of freedom. The buckling subcase must select the corresponding static subcase on the same snapshot.

For modal coupling, solve SOL 103 once at a given geometry. Project each subsequent force vector through the returned modal basis. Entry *k* of the generalized load is the work of the nodal loads against mode *k*, with the normalization used in the eigenvalue table. Rebuild the snapshot and coupling object after changing the model geometry or spline attachments.

An external structural solver can also call the two transfer operators independently. The API reference describes node order, six-component fields in the basic frame, and how forces and moments accumulate at shared spline attachments. [Spline transfer API](api.rst)

## Reproduce the RBE3 wing calculation

Use the [four-box wing model](../wasm/test/fixtures/trim-rbe3-wing.oos) with the default classic formulation. The boxes have IDs 100 through 103. Node 6 is a dependent RBE3 attachment. The examples apply forces of `[1, -2, 3, 5]` to subcase 1. Subcase 2 selects the eigenvalue method and references subcase 1 for buckling.

All three examples read the same file through the O_o_Model OOS adapter. Place `trim-rbe3-wing.oos` in the working directory. For the Node.js example, place the JavaScript packages in `model/` and `solver/` beside the script. For Python, install the native package with `python -m pip install oo-solver==0.2.2`. For C++, install `oo-solver` from the vcpkg registry. To read this fixture, add a checkout of the [O_o_OOS adapter](https://github.com/jcdulas/O_o_OOS) to the application build after `find_package(oo-solver CONFIG REQUIRED)` and link the application to `oos::api` and `oor::oos`.

**JavaScript / WebAssembly (Node.js)**

```javascript
import { readFile } from "node:fs/promises";
import { loadOoModel } from "./model/index.js";
import { loadOosSolver } from "./solver/index.js";

const reader = await loadOoModel();
const solver = await loadOosSolver();
reader.writeFile("/work/wing.oos", await readFile("trim-rbe3-wing.oos"));
const deck = reader.readModel("/work/wing.oos");
let model, transfer, state, modes;
try {
  model = solver.modelFromReader(reader, deck);
  transfer = model.aeroTransfer();
  const forces = Float64Array.of(1, -2, 3, 5);
  state = transfer.solveStatic(1, forces);
  modes = transfer.solveModes(2);
  const modalLoads = modes.modalBasis.generalizedLoads(forces);
  const buckling = transfer.solveBuckling(2, state.preload);
  console.log(state.boxMotions[0].plunge[0], modes.eigenvalues[0].rows[0].cycles,
              buckling.eigenvalues[0].rows[0].eigenvalue, modalLoads);
} finally {
  modes?.modalBasis.delete();
  state?.preload.delete();
  transfer?.delete();
  model?.delete();
  deck.delete();
}
```

**Python**

```python
import numpy as np
from oo_model import read_model
from oo_solver import model_from_reader

model = model_from_reader(read_model("trim-rbe3-wing.oos"))
transfer = model.aero_transfer()
forces = np.array([1., -2., 3., 5.])
state = transfer.solve_static(1, forces)
modes = transfer.solve_modes(2)
modal_loads = modes["modal_basis"].generalized_loads(forces)
buckling = transfer.solve_buckling(2, state["preload"])
print(state["box_motions"][0]["plunge"][0], modes["eigenvalues"][0]["rows"]["cycles"][0],
      buckling["eigenvalues"][0]["rows"]["eigenvalue"][0], modal_loads)
```

**C++20**

```cpp
#include <oor/formats/oos.hpp>
#include <oos/api/AeroCoupling.hpp>
#include <oos/api/ModelFromDeck.hpp>
#include <array>
#include <iostream>

int main() {
    auto deck = oor::formats::oos::read("trim-rbe3-wing.oos");
    oos::api::AeroCoupling transfer(oos::api::modelFromDeck(deck));
    const std::array<double, 4> forces{1., -2., 3., 5.};
    auto state = transfer.solveStatic(1, forces);
    auto modes = transfer.solveModes(2);
    auto modalLoads = modes.modalBasis->generalizedLoads(forces);
    auto buckling = transfer.solveBuckling(2, *state.preload);
    std::cout << state.boxMotions.at(0).plunge.at(0) << ' '
              << modes.analysis.eigenvalues.at(0).rows.at(0).cycles << ' '
              << buckling.analysis.eigenvalues.at(0).rows.at(0).eigenvalue;
    for (double q : modalLoads) std::cout << ' ' << q;
    std::cout << '\n';
}
```

| Quantity | Expected value, rounded |
| --- | ---: |
| Normal displacement at the centre of box 100 | 0.0024797 |
| First natural frequency, cycles per unit time | 10.2596 |
| First buckling factor for this preload | 455.816 |

The [coupling regression tests](../wasm/test/validation/aero-transfer.test.mjs) also compare these paths with an independently loaded structural deck, rotate the displacement frame of a dependent attachment, check inverse load scaling in mixed Auto buckling, and recover the restrained SOL 144 displacements by applying its equilibrated box forces in SOL 101.
