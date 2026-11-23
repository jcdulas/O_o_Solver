// Extract structural-comparison-inputs.zip into build/structural-comparison-inputs before running.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CLASSIC, MIXED, runModel } from "../../wasm/test/validation/support.mjs";

function measurement(run, probe) {
  if (probe.field === "frequency" || probe.field === "buckling") {
    const values = run.eigenvalues[0].rows.map(row => row.eigenvalue).filter(value => value > 0);
    assert.ok(values.length, "Missing positive eigenvalue");
    const first = Math.min(...values);
    return probe.field === "frequency" ? Math.sqrt(first) / (2 * Math.PI) : first;
  }
  if (probe.field === "displacement") {
    const table = run.nodalTables.find(t => t.kind === "displacements" && t.subcase === 1);
    const values = probe.nodes.map(id => {
      const row = table.ids.indexOf(id);
      assert.ok(row >= 0, `Missing displacement at node ${id}`);
      return table.values[6 * row + probe.component];
    });
    return probe.scale * values.reduce((a, b) => a + b, 0) / values.length;
  }
  assert.equal(probe.field, "cornerStress");
  const table = run.elementTables.find(t => t.elementType === probe.card && t.kind === "stress" && t.subcase === 1);
  const component = table.components.indexOf(probe.column);
  assert.ok(component >= 0);
  const values = [];
  table.nodeIds.forEach((id, row) => {
    if (id === probe.node) values.push(table.values[row * table.components.length + component]);
  });
  assert.ok(values.length > 0, `Missing corner stress at node ${probe.node}`);
  return values.reduce((a, b) => a + b, 0) / values.length;
}

const directory = process.argv[2] ?? "build/structural-comparison-inputs";
const cases = JSON.parse(readFileSync(new URL("structural-comparison-results.json", import.meta.url)));
for (const c of cases) {
  const input = readFileSync(join(directory, c.file), "utf8");
  for (const [label, options] of [["classic", CLASSIC], ["mixed", MIXED]]) {
    const text = label === "mixed" ? input.trimEnd() + "\nParameter;MixedHybridDegree;Auto\n" : input;
    const run = runModel(text, options, c.analysis);
    for (const probe of c.probes) {
      const value = measurement(run, probe);
      const name = `${c.id}/${c.mesh}/${label}/${probe.name}`;
      console.log(`${name}  ${value.toPrecision(10)}`);
      assert.ok(Math.abs(value - probe[label]) <= 1e-6 * Math.abs(probe.target), name);
    }
  }
}
