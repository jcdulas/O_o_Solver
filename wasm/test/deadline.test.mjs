import assert from "node:assert/strict";
import { after, test } from "node:test";
import { CaseWorker } from "./helpers/controller.mjs";

const workers = ["javascript", "wasm"].map(probe => ({ probe, blocked: new CaseWorker({ probe }) }));
await Promise.all(workers.map(({ blocked }) => blocked.ready));
after(() => Promise.all(workers.map(({ blocked }) => blocked.close())));
for (const { probe, blocked } of workers)
  test(`the deadline interrupts a worker blocked in ${probe}`, { timeout: 1000 }, async () => {
    const started = performance.now();
    await assert.rejects(blocked.run(0, 30), /exceeded 30 ms/);
    assert.ok(performance.now() - started < 1000);
    await assert.rejects(blocked.run(0), /terminated/);
  });
