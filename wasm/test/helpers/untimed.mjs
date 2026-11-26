// Runs the cases of the fast suite in this process without their time limit, and prints the failures and the slowest
// cases: for a machine busy with something else, where the 900 ms watchdog of run.mjs says nothing about the results.
//
//   node wasm/test/helpers/untimed.mjs [suite ...] [--match text]
//
// With no suite named, every suite runs. The timings are measured, not judged.

import { cases } from "./register.mjs";

const ALL_SUITES = ["smoke", "optimization", "modal-derivatives", "trim-derivatives", "buckling-derivatives",
  "beam-derivatives", "studies", "static-inputs", "threading", "static-matrix", "connections", "controls",
  "mixed-hybrid", "mixed-hybrid-geometry", "mixed-hybrid-energy", "condensation", "streamed", "reader-fixes", "bar-stations",
  "review-assembly", "review-bindings", "review-design", "review-elements", "review-reduction", "review-solutions"];
const SLOWEST_SHOWN = 8;

const words = process.argv.slice(2);
const matchAt = words.indexOf("--match");
const match = matchAt >= 0 ? words[matchAt + 1] : null;
const suites = (matchAt >= 0 ? words.slice(0, matchAt) : words);
for (const suite of suites.length ? suites : ALL_SUITES) await import(`../cases/${suite}.mjs`);

const failures = [];
const timings = [];
let passed = 0;
let skipped = 0;
for (const entry of cases) {
  if (entry.skip || (match && !entry.name.includes(match))) { ++skipped; continue; }
  const started = performance.now();
  try {
    await entry.body();
    ++passed;
  } catch (error) {
    failures.push({ name: entry.name, error: error?.stack ?? String(error) });
  }
  timings.push({ name: entry.name, elapsed: performance.now() - started });
}
for (const failure of failures) console.error(`FAILED ${failure.name}\n${failure.error}\n`);
timings.sort((a, b) => b.elapsed - a.elapsed);
for (const timing of timings.slice(0, SLOWEST_SHOWN)) console.log(`${timing.elapsed.toFixed(0)} ms  ${timing.name}`);
console.log(`${passed} passed, ${failures.length} failed, ${skipped} skipped.`);
process.exitCode = failures.length ? 1 : 0;
