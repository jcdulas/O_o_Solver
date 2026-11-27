import assert from "node:assert/strict";
import { after, test } from "node:test";
import { pathToFileURL } from "node:url";
import { EventEmitter } from "node:events";
import path from "node:path";
import { oor, readerDist, dist } from "./helpers/runtime.mjs";
import { axial } from "./helpers/optimization-models.mjs";
import { CASE_CEILING_MS } from "./helpers/budget.mjs";
const { createOptimizationWorkerPool, OptimizationWorkerPool } = await import(pathToFileURL(path.join(dist, "index.js")));

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
// The threaded package's runtime puts the Worker of node:worker_threads on globalThis when it loads: a pool started
// after it must still use the events of Node.js. The same Worker is put there for the single-threaded package.
const { Worker: NodeWorker } = await import("node:worker_threads");
const globalWorker = Object.getOwnPropertyDescriptor(globalThis, "Worker");
globalThis.Worker = NodeWorker;
const runtimeWorkerPool = await createOptimizationWorkerPool(config()).finally(() => {
  if (globalWorker) Object.defineProperty(globalThis, "Worker", globalWorker); else delete globalThis.Worker;
});
after(() => { sharedPool.close(); timeoutPool.close(); nativePool.close(); runtimeWorkerPool.close(); });

test("independent package workers transfer results and gradients", { timeout: CASE_CEILING_MS }, async () => {
  const results = await sharedPool.evaluateMany([[2, 10], [3, 12]], { includeResults: true, gradients: true });
  [0.015, 0.012].forEach((value, i) => near(results[i].values.objective, value));
  assert.ok(results[0].analyses[0].nodalTables[0].values instanceof Float64Array);
  near(results[0].gradients.values[0], -0.0075);
});

test("queued work and abort leave the study pool usable", { timeout: CASE_CEILING_MS }, async () => {
  near((await sharedPool.evaluate([1, 10])).values.objective, 0.03);
  const aborted = new AbortController(); aborted.abort();
  await assert.rejects(sharedPool.evaluate([2, 10], { signal: aborted.signal }), { name: "AbortError" });
  near((await sharedPool.evaluate([2, 10])).values.objective, 0.015);
});

test("a timeout interrupts synchronous work and its replacement worker accepts the next point", { timeout: CASE_CEILING_MS }, async () => {
  await assert.rejects(timeoutPool.evaluate([0.75, 10], { timeoutMs: 30 }), /timed out/);
  near((await timeoutPool.evaluate([2, 10])).values.objective, 0.015);
});

test("native optimization problems use the same worker protocol", { timeout: CASE_CEILING_MS }, async () => {
  const point = await nativePool.evaluate([2], { gradients: true });
  near(point.values.objective, 0.015); near(point.gradients.objective[0], -0.0075);
});

test("a pool started after the threaded runtime put a Worker on globalThis evaluates", { timeout: CASE_CEILING_MS },
  async () => { near((await runtimeWorkerPool.evaluate([2, 10])).values.objective, 0.015); });

// ---- the scheduling of the pool on stand-in workers: they answer at once, so that only the pool is measured.
// A point whose first value is FAILING is refused as an ordinary error, TRAPPING as a fault of the module.
const FAILING = -1, TRAPPING = -2;
class StubWorker extends EventEmitter {
  static created = 0;
  static started = [];
  constructor() { super(); ++StubWorker.created; }
  postMessage(message) {
    setImmediate(() => {
      if (message.type === "initialize") return this.emit("message", { type: "ready", metadata: {} });
      StubWorker.started.push(message.x[0]);
      if (message.x[0] === FAILING || message.x[0] === TRAPPING)
        return this.emit("message", { type: "error", id: message.id, message: `point ${message.x[0]} failed`,
          fatal: message.x[0] === TRAPPING });
      this.emit("message", { type: "result", id: message.id, result: { values: { x: message.x } } });
    });
  }
  terminate() { return Promise.resolve(); }
}
async function stubPool(settings) {
  StubWorker.created = 0; StubWorker.started = [];
  const pool = new OptimizationWorkerPool(StubWorker, true, { model: new Uint8Array(1) },
    { workers: 2, initializationTimeoutMs: 1000, maximumQueuedPoints: 4, ...settings });
  await pool.initialize();
  return pool;
}

test("evaluateMany takes more points than the queue holds, in order", { timeout: CASE_CEILING_MS }, async () => {
  const pool = await stubPool();
  try {
    const points = Array.from({ length: 64 }, (_, index) => [index]);
    const results = await pool.evaluateMany(points);
    assert.deepEqual(results.map(result => result.values.x[0]), points.map(point => point[0]));
    const seen = [];
    assert.deepEqual(await pool.evaluateMany(points, { onPoint: (result, index) => { seen.push(index); } }), []);
    assert.deepEqual(seen.sort((a, b) => a - b), points.map(point => point[0]));
    // evaluate keeps its limit: the points beyond the workers and the queue are refused at once
    const direct = points.slice(0, 12).map(point => pool.evaluate(point));
    const outcomes = await Promise.allSettled(direct);
    assert.equal(outcomes.filter(outcome => outcome.status === "fulfilled").length, 6);
    assert.ok(outcomes.filter(outcome => outcome.status === "rejected")
      .every(outcome => /maximumQueuedPoints exceeded/.test(outcome.reason.message)));
  } finally { pool.close(); }
});

test("evaluateMany stops feeding points at the first failure", { timeout: CASE_CEILING_MS }, async () => {
  const pool = await stubPool();
  try {
    const points = Array.from({ length: 64 }, (_, index) => [index === 5 ? FAILING : index]);
    await assert.rejects(pool.evaluateMany(points), /point -1 failed/);
    await new Promise(resolve => setTimeout(resolve, 20));
    // the points before the failure, the failure, and at most the point the other worker had taken
    assert.ok(StubWorker.started.length <= 8, `${StubWorker.started.length} points were started`);
    assert.equal(StubWorker.created, 2, "an ordinary error keeps its worker");
    assert.equal((await pool.evaluate([7])).values.x[0], 7);
  } finally { pool.close(); }
});

test("a worker whose module trapped is replaced before the next point", { timeout: CASE_CEILING_MS }, async () => {
  const pool = await stubPool();
  try {
    await assert.rejects(pool.evaluate([TRAPPING]), /point -2 failed/);
    assert.equal((await pool.evaluate([3])).values.x[0], 3);
    assert.equal((await pool.evaluate([4])).values.x[0], 4);
    assert.equal(StubWorker.created, 3, "the worker that trapped was replaced once");
    assert.equal(pool.slots.length, 2);
  } finally { pool.close(); }
});
