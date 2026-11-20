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

// The mixed hybrid formulation is chosen by the program; the declarations of its geometry are written in the model
// file, as a preprocessor writes them (`surfaceRecords`).
export const MIXED = { elementFormulation: "MixedHybridFormulation" };
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

// The records by which a preprocessor writes the declarations of the geometry of a model in its model file (spec
// section 4.4): a `SurfaceEdge` for each edge it states smooth, sharp or raw ([g1, g2], or [g1, g2, element] for one
// side of a junction), a `SurfaceFacet` for each raw element, a `SurfaceLine` for each mesh line it continues
// ([g1, g, g2]) or ends ([g1, g]). Other formulations do not read these records, so that one model file serves the
// comparisons. The records are numbered from 1: one call for a model.
export function surfaceRecords({ smoothEdges = [], sharpEdges = [], rawEdges = [], rawElements = [],
  lineContinuations = [], lineEnds = [] } = {}) {
  const records = [];
  let [edges, lines] = [0, 0];
  const edge = (state) => ([g1, g2, element]) =>
    records.push(`SurfaceEdge;${++edges};${g1};${g2};${state}${element === undefined ? "" : `;${element}`}`);
  smoothEdges.forEach(edge("Smooth"));
  sharpEdges.forEach(edge("Sharp"));
  rawEdges.forEach(edge("Raw"));
  for (const element of rawElements) records.push(`SurfaceFacet;${element}`);
  for (const [g1, g, g2] of lineContinuations) records.push(`SurfaceLine;${++lines};${g1};${g};${g2}`);
  for (const [g1, g] of lineEnds) records.push(`SurfaceLine;${++lines};${g1};${g}`);
  return records;
}

// The record of a model file that was prepared and has nothing to declare: the fold angle of the model
// (ShellNormalAngle) then decides every edge, and the rules of the solver every mesh line.
export const GEOMETRY_PREPARED = "Parameter;GeometryDeclarations;Given";

const isSurfaceRecord = (record) => /^Surface(Edge|Facet|Line);/.test(record);

// The model records `records` as a prepared model file holds them: with GEOMETRY_PREPARED when none of them declares
// the geometry.
export function prepared(records) {
  return records.some(isSurfaceRecord) ? records : [...records, GEOMETRY_PREPARED];
}

// The model file `text` (prepared: see `prepared`) with the declarations `geometry` in place of what it declares.
// The model section is the last of a model file: the records are written at its end.
export function withSurface(text, geometry) {
  const kept = text.trimEnd().split("\n").filter((line) => !isSurfaceRecord(line) && line !== GEOMETRY_PREPARED);
  return [...kept, ...prepared(surfaceRecords(geometry)), ""].join("\n");
}

// The model file `text` without any declaration of its geometry: a model that no preprocessor prepared.
export function unprepared(text) {
  return text.split("\n").filter((line) => !isSurfaceRecord(line) && line !== GEOMETRY_PREPARED).join("\n");
}

// The edges that two elements of `elements` (each with its corner grid points, in order) share: the inner edges of a
// surface of shells, which a preprocessor declares smooth on a surface without fold.
export function innerEdges(elements) {
  const count = new Map();
  for (const corners of elements)
    corners.forEach((a, k) => {
      const b = corners[(k + 1) % corners.length];
      const key = a < b ? `${a};${b}` : `${b};${a}`;
      count.set(key, (count.get(key) ?? 0) + 1);
    });
  return [...count].filter(([, shells]) => shells === 2).map(([key]) => key.split(";").map(Number));
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
