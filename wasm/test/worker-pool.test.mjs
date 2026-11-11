import assert from "node:assert/strict";
import { after, test } from "node:test";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { oor, readerDist, dist } from "./helpers/runtime.mjs";
import { axial } from "./helpers/optimization-models.mjs";
const { createOptimizationWorkerPool } = await import(pathToFileURL(path.join(dist, "index.js")));

const definition = { version: 1,
  variables: [{ id: 1, initialValue: 2, lower: 0.5, upper: 5 }, { id: 2, initialValue: 10, lower: 5, upper: 20 }],
  parameters: [{ variableId: 1, target: { kind: "card", type: "RodProp", id: 1, field: "area" } },
    { variableId: 2, target: { kind: "card", type: "PointForceLoad", id: 2, slot: 4 } }],
  analyses: [{ analysis: "statics" }], responses: [{ id: 1, source: "nodal", kind: "displacements", ids: [2], components: "t1" }],
  objective: { responseId: 1 } };
function config(extra = {}) {
  const deck = axial();
  try {
    deck.writeArchive("/work/worker-study.ooh5");
    return { model: oor.readFile("/work/worker-study.ooh5"), readerURL: pathToFileURL(path.join(readerDist, "index.js")),
      definition, workers: 1, ...extra };
  } finally { deck.delete(); }
}
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} differs from ${expected}`);

const sharedPool = await createOptimizationWorkerPool(config({ workers: 2 }));
const timeoutPool = await createOptimizationWorkerPool(config({ definition: { ...definition, update: "blockOnSmallArea" },
  hooksURL: new URL("./fixtures/study-hooks.mjs", import.meta.url) }));
const nativePool = await createOptimizationWorkerPool(config({ mode: "native", definition: {
  variables: [{ id: 1, initialValue: 2, lower: 0.5, upper: 5 }],
  relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area", terms: [{ variableId: 1, coefficient: 1 }] }],
  responses: [{ id: 1, type: "Displacement", attributeA: 1, targets: [2] }], objective: { responseId: 1, subcase: 1 } } }));
after(() => { sharedPool.close(); timeoutPool.close(); nativePool.close(); });

test("independent package workers transfer results and gradients", { timeout: 1000 }, async () => {
  const results = await sharedPool.evaluateMany([[2, 10], [3, 12]], { includeResults: true, gradients: true });
  [0.015, 0.012].forEach((value, i) => near(results[i].values.objective, value));
  assert.ok(results[0].analyses[0].nodalTables[0].values instanceof Float64Array);
  near(results[0].gradients.values[0], -0.0075);
});

test("queued work and abort leave the study pool usable", { timeout: 1000 }, async () => {
  near((await sharedPool.evaluate([1, 10])).values.objective, 0.03);
  const aborted = new AbortController(); aborted.abort();
  await assert.rejects(sharedPool.evaluate([2, 10], { signal: aborted.signal }), { name: "AbortError" });
  near((await sharedPool.evaluate([2, 10])).values.objective, 0.015);
});

test("a timeout interrupts synchronous work and its replacement worker accepts the next point", { timeout: 1000 }, async () => {
  await assert.rejects(timeoutPool.evaluate([0.75, 10], { timeoutMs: 30 }), /timed out/);
  near((await timeoutPool.evaluate([2, 10])).values.objective, 0.015);
});

test("native optimization problems use the same worker protocol", { timeout: 1000 }, async () => {
  const point = await nativePool.evaluate([2], { gradients: true });
  near(point.values.objective, 0.015); near(point.gradients.objective[0], -0.0075);
});
