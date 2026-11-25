import { oor, solver } from "../../../wasm/test/helpers/runtime.mjs";
import { performance } from "node:perf_hooks";

// A 1 by 0.5 cantilever with distorted interior nodes and a transverse tip force.
// Both meshes cover the same panel and every element uses the same design property.
function panel(card) {
  const nx = 8, ny = 4;
  const node = (i, j) => j * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
    "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  for (let j = 0; j <= ny; j++)
    for (let i = 0; i <= nx; i++) {
      const interior = i > 0 && i < nx && j > 0 && j < ny;
      const dx = interior ? 0.032 * Math.sin(i + 2 * j) : 0;
      lines.push(`Node;${node(i, j)};;${i / nx + dx};${j / (2 * ny) - 0.4 * dx};0.`);
    }
  let elements = 0;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const [a, b, c, d] = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
      if (card === "ShellQuad4") lines.push(`${card};${++elements};1;${a};${b};${c};${d}`);
      else lines.push(`${card};${++elements};1;${a};${b};${c}`, `${card};${++elements};1;${a};${c};${d}`);
    }
  for (let j = 0; j <= ny; j++) lines.push(`SinglePointConstrList;1;123456;${node(0, j)}`);
  const tip = node(nx, ny / 2);
  lines.push("ShellProp;1;1;0.025;1;;1", "IsoMat;1;7.e10;;0.3;2700.", `PointForceLoad;2;${tip};;100.;0.;0.;1.`,
    "DesignVar;1;T;0.025;0.01;0.04", "DesignPropRelation1;1;ShellProp;1;thickness;;;;1;1.",
    `DesignResp1;10;TIP;Displacement;;;3;;${tip}`);
  return { text: lines.join("\n") + "\n", elements };
}

for (const card of ["ShellQuad4", "ShellTria3"]) {
  const { text, elements } = panel(card);
  const file = "/work/shell-thickness.oos";
  oor.writeFile(file, text);
  const deck = oor.readModel(file, { strict: true });
  const problem = solver.optimizationFromReader(oor, deck, {}, {},
    { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} });
  try {
    const times = [];
    for (let i = 0; i < 9; i++) {
      const point = problem.evaluate([0.025]);
      try {
        const start = performance.now();
        const result = point.gradients();
        const elapsed = performance.now() - start;
        if (i >= 2) times.push(elapsed);
        if (i === 8) console.log(JSON.stringify({ card, elements, medianMs: times.sort((a, b) => a - b)[3],
          gradient: result.responses[0].values[0], report: result.report }));
      } finally { point.delete(); }
    }
  } finally { problem.delete(); deck.delete(); }
}
