import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { CASE_BUDGET_MS, CASE_CEILING_MS } from "./helpers/budget.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const variant = process.env.OOS_SOLVER_DIST
  ? path.basename(path.dirname(path.resolve(process.env.OOS_SOLVER_DIST))) : "wasm";
const directory = path.resolve(process.env.OOS_TEST_RESULTS ?? path.join(root, "build/test-results"));
mkdirSync(directory, { recursive: true });
const report = path.join(directory, `${variant}.json`);
console.log(`Running ${variant} cases with a ${CASE_BUDGET_MS} ms watchdog and a ${CASE_CEILING_MS} ms test ceiling...`);
const result = spawnSync(process.execPath, ["--test", `--test-reporter=${pathToFileURL(path.join(here, "helpers/reporter.mjs"))}`,
  `--test-reporter-destination=${report}`, ...process.argv.slice(2), path.join(here, "deadline.test.mjs"),
  path.join(here, "fast.test.mjs")], { stdio: "inherit", cwd: root });
if (result.error) throw result.error;
const reportText = readFileSync(report, "utf8");
if (!reportText.trim()) {
  console.error("Test runner ended without a timing report.");
  process.exit(result.status || 1);
}
const summary = JSON.parse(reportText);
console.log(`${summary.passed} passed, ${summary.failed} failed, ${summary.skipped} skipped; ` +
  `slowest ${(summary.maximumMs).toFixed(1)} ms; total ${(summary.elapsedMs / 1000).toFixed(1)} s.`);
for (const failure of summary.tests.filter(t => t.status === "failed").slice(0, 10))
  console.error(`${failure.name}\n${failure.error}`);
console.log(`Timing report: ${report}`);
process.exitCode = result.status || summary.failed || summary.overLimit.length ? 1 : 0;
