// A compact, reproducible artifact containing every individual duration and failure.
export default async function* report(source) {
  const started = performance.now();
  const tests = [];
  for await (const event of source) {
    if (event.type !== "test:pass" && event.type !== "test:fail") continue;
    const { name, skip, details } = event.data;
    tests.push({ name, durationMs: details.duration_ms,
      status: skip ? "skipped" : event.type === "test:pass" ? "passed" : "failed",
      ...(details.error ? { error: details.error.stack ?? String(details.error) } : {}) });
  }
  const active = tests.filter(t => t.status !== "skipped");
  yield JSON.stringify({ elapsedMs: performance.now() - started, ceilingMs: 1000, workerBudgetMs: 900,
    passed: tests.filter(t => t.status === "passed").length,
    failed: tests.filter(t => t.status === "failed").length,
    skipped: tests.filter(t => t.status === "skipped").length,
    maximumMs: Math.max(0, ...active.map(t => t.durationMs)),
    overLimit: active.filter(t => t.durationMs >= 1000),
    slowest: active.toSorted((a, b) => b.durationMs - a.durationMs).slice(0, 10), tests }, null, 2) + "\n";
}
