import assert from "node:assert/strict";
import { after, test } from "node:test";
import { CaseWorker, CASE_BUDGET_MS } from "./helpers/controller.mjs";

const worker = new CaseWorker();
const cases = await worker.ready; // Initialization is shared and is not part of a test's execution budget.
after(() => worker.close());
for (const [index, entry] of cases.entries())
  test(entry.name, { skip: entry.skip, timeout: 1000, concurrency: false }, async () => {
    const started = performance.now();
    const result = await worker.run(index);
    assert.ok(result.elapsed < CASE_BUDGET_MS, `worker calculation exceeded ${CASE_BUDGET_MS} ms`);
    assert.ok(performance.now() - started < 1000, "test exceeded one second including message delivery");
  });
