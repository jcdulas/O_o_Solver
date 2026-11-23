// From the repository root, extract thin-plate-inputs.zip into build/thin-plate first.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CLASSIC, MIXED, nodal, runModel } from "../../wasm/test/validation/support.mjs";

const directory = process.argv[2] ?? "build/thin-plate";
const { reference, cases } = JSON.parse(readFileSync(new URL("thin-plate-results.json", import.meta.url)));
for (const sample of cases) {
  const { mesh, formulation, degree, centreNodes } = sample;
  let model = readFileSync(join(directory, mesh + ".oos"), "utf8");
  if (degree !== null) model += `Parameter;MixedHybridDegree;${degree}\n`;
  const run = runModel(model, formulation === "mixed" ? MIXED : CLASSIC);
  const deflection = -centreNodes.reduce((sum, id) => sum + nodal(run, "displacements", id)[2], 0) / 2;
  const wError = 100 * (deflection / reference.centreDeflectionMetres - 1);
  const card = mesh.startsWith("tet") ? "SolidTetra" : "SolidHexa";
  const table = run.elementTables.find(t => t.elementType === card && t.kind === "stress" && t.subcase === 1);
  const columns = ["oxx", "oyy"].map(component => table.components.indexOf(component));
  assert.ok(columns.every(col => col >= 0));
  const errors = centreNodes.map((id, face) => {
    const expected = face === 0 ? reference.bottomStressPa : reference.topStressPa;
    const values = [];
    table.nodeIds.forEach((node, row) => {
      if (node === id) for (const col of columns) {
        const stress = table.values[row * table.components.length + col];
        values.push(100 * Math.abs(stress / expected - 1));
      }
    });
    assert.ok(values.length > 0, `Missing corner stress at node ${id}`);
    return Math.max(...values);
  });
  const stressError = Math.max(...errors);
  const reactions = run.nodalTables.find(t => t.kind === "singlePointConstrForces" && t.subcase === 1);
  const vertical = reactions.ids.reduce((sum, _, i) => sum + reactions.values[6 * i + 2], 0);
  const label = [mesh, formulation, degree].filter(Boolean).join(" ");
  console.log(`${label}: deflection ${wError.toFixed(5)}%, surface stress ${stressError.toFixed(5)}%`);
  assert.ok(Math.abs(vertical - 250) < 1e-4, `${label}: vertical reaction ${vertical} N`);
  assert.ok(Math.abs(wError - sample.deflectionErrorPct) < 1e-4, label + " deflection");
  assert.ok(Math.abs(stressError - sample.surfaceStressErrorPct) < 1e-4, label + " surface stress");
}
console.log(`Verified ${cases.length} plate calculations.`);
