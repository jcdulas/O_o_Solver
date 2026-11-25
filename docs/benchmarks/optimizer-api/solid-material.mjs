import { performance } from "node:perf_hooks";
import { oor, solver } from "../../../wasm/test/helpers/runtime.mjs";

// Same cantilever, mesh vertices, load, design variable and response in all four runs.
function cantilever(card) {
  const nx = 4, ny = 2;
  const node = (i, j, k) => 1 + i + (nx + 1) * (j + (ny + 1) * k);
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
    "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  for (let k = 0; k <= 1; k++)
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i <= nx; i++) {
        const id = node(i, j, k);
        lines.push(`Node;${id};;${i / nx};${j * 0.4 / ny};${k * 0.2}`);
        if (i === 0) lines.push(`SinglePointConstrList;1;123456;${id}`);
      }
  let elements = 0;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const corners = [node(i, j, 0), node(i + 1, j, 0), node(i + 1, j + 1, 0), node(i, j + 1, 0),
        node(i, j, 1), node(i + 1, j, 1), node(i + 1, j + 1, 1), node(i, j + 1, 1)];
      if (card === "SolidHexa") lines.push(`${card};${++elements};1;${corners.join(";")}`);
      else for (const [a, b] of [[1, 2], [2, 3], [3, 7], [7, 4], [4, 5], [5, 1]])
        lines.push(`${card};${++elements};1;${[corners[0], corners[a], corners[b], corners[6]].join(";")}`);
    }
  const tip = node(nx, ny, 1);
  lines.push("SolidProp;1;1", "IsoMat;1;1.e6;;0.25;2.", "Parameter;GeometryDeclarations;Given",
    `PointForceLoad;2;${tip};;1.;1.;0.;0.`, "DesignVar;1;E;1.e6;5.e5;2.e6",
    "DesignMatRelation1;1;IsoMat;1;youngsModulus;;;;1;1.", `DesignResp1;10;U;Displacement;;;1;;${tip}`);
  return { text: lines.join("\n") + "\n", elements };
}

for (const card of ["SolidHexa", "SolidTetra"]) {
  const { text, elements } = cantilever(card);
  oor.writeFile("/work/solid-material.oos", text);
  const deck = oor.readModel("/work/solid-material.oos", { strict: true });
  try {
    for (const elementFormulation of ["ClassicFormulation", "MixedHybridFormulation"]) {
      const problem = solver.optimizationFromReader(oor, deck, {}, {}, { elementFormulation, formulationGeometry: {} });
      try {
        const times = [];
        for (let i = 0; i < 7; i++) {
          const point = problem.evaluate([1e6]);
          try {
            const start = performance.now();
            const result = point.gradients();
            const elapsed = performance.now() - start;
            if (i >= 2) times.push(elapsed);
            if (i === 6) console.log(JSON.stringify({ card, elements, elementFormulation,
              medianMs: times.sort((a, b) => a - b)[2], gradient: result.responses[0].values[0],
              fieldPerturbations: result.report.fieldPerturbations }));
          } finally { point.delete(); }
        }
      } finally { problem.delete(); }
    }
  } finally { deck.delete(); }
}
