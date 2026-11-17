// Shared helpers of the validation suite of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md,
// section 13.3): benchmarks through the whole path (reader, modelFromReader, run) against closed-form or classical
// values, with the criteria of section 13.6. These cases have no time limit: they are not part of the fast suite.
//
// Run: node --test wasm/test/validation/*.test.mjs   (Node.js 24; needs ../O_o_Reader/build/wasm/dist)

import assert from "node:assert/strict";
import { test as nodeTest } from "node:test";
import { oor, solver } from "../helpers/runtime.mjs";
import { KNOWN_LIMITS } from "./known-limits.mjs";

export { oor, solver };

// node:test's test, with the cases of known-limits.mjs marked "todo" with their reason: they still run and report
// what they measure, without failing the suite.
export function test(name, optionsOrBody, maybeBody) {
  const [options, body] = typeof optionsOrBody === "function" ? [{}, optionsOrBody] : [optionsOrBody, maybeBody];
  const limit = KNOWN_LIMITS.find((entry) => name.includes(entry.match));
  return nodeTest(name, limit ? { ...options, todo: limit.reason } : options, body);
}

// The criteria of section 13.6, fixed before the runs.
export const CRITERIA = { invariant: 1e-8, exact: 1e-6, displacement: 5e-3, frequency: 5e-3, criticalLoad: 5e-3,
  stress: 1e-2 };

export const MIXED = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };
export const CLASSIC = {};

// A real as a model file writes it: with a decimal point or an exponent.
export function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

let deckCount = 0;

// Reads `text` as a model file, builds the model with `options` and runs `action(model)`.
export function withModel(text, options, action) {
  const file = `/work/validation-${++deckCount}.oos`;
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try { return action(model); } finally { model.delete(); }
  } finally { deck.delete(); }
}

// Runs `analysis` ("statics", "modes", "buckling", "aeroelastic") on the model of `text` and returns the result; an
// error diagnostic fails the case.
export function runModel(text, options, analysis = "statics", settings = undefined) {
  return withModel(text, options, (model) => {
    const run = solver.run(model, analysis, settings);
    const errors = run.diagnostics.filter((d) => d.severity === "error");
    assert.equal(errors.length, 0, JSON.stringify(errors));
    return run;
  });
}

// The six values of grid point `node` in the nodal table `kind` ("displacements", ...) of `subcase`.
export function nodal(run, kind, node, subcase = 1) {
  const table = run.nodalTables.find((t) => t.kind === kind && t.subcase === subcase);
  assert.ok(table, `a ${kind} table of subcase ${subcase}`);
  const row = table.ids.indexOf(node);
  return row < 0 ? [0, 0, 0, 0, 0, 0] : Array.from(table.values.slice(row * 6, row * 6 + 6));
}

// |actual / expected − 1| ≤ tolerance.
export function assertRatio(actual, expected, tolerance, what) {
  assert.ok(Math.abs(actual / expected - 1) <= tolerance,
    `${what}: ${actual}, expected ${expected} (ratio ${(actual / expected).toFixed(6)}, tolerance ${tolerance})`);
}
