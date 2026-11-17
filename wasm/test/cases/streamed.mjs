import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { real } from "../helpers/static-models.mjs";
import { overlappingPlates, platePair } from "./condensation.mjs";

// The full path streamed (run(model, "statics", { streamed: true })): the full path's equations solved without a
// factor of the whole model, the outputs made region by region. Its displacements and element tables are those of
// the full path to round-off, in ClassicFormulation and in MixedHybridFormulation (whose entities of edges, faces and
// interiors its fronts and windows carry, its excluded rotations filled from the whole vectors).

// The relative difference below which two results are one: the round-off of the solves on these small models, larger
// for the mixed formulation (its unknowns of several scales: moments of the edge displacements, rotations, shears).
const SAME_RESULT = 1e-9;
const MIXED_SAME_RESULT = 1e-6;
const MIXED_OPTIONS = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };

// Runs `action(model)` on the model of `text` read with `options`.
function withModel(text, options, action) {
  oor.writeFile("/work/streamed.oos", text);
  const deck = oor.readModel("/work/streamed.oos", { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck, options);
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

// The largest difference between the tables of a kind of two runs, against their largest value (the principal angles
// left out: at a point of equal principal stresses round-off turns them by 180 degrees).
function relativeDifference(full, streamed, kind, tables = "nodalTables") {
  let largest = 0;
  let difference = 0;
  const fullTables = full[tables].filter((table) => table.kind === kind);
  assert.ok(fullTables.length > 0, `the full path gives ${kind}`);
  for (const table of fullTables) {
    const other = streamed[tables].find((candidate) => candidate.kind === kind && candidate.subcase === table.subcase &&
      (candidate.elementType ?? "") === (table.elementType ?? ""));
    assert.ok(other, `the streamed path gives ${kind} ${table.elementType ?? ""} of subcase ${table.subcase}`);
    assert.deepEqual([...other.ids], [...table.ids], `the rows of ${kind} ${table.elementType ?? ""}`);
    const width = table.components.length;
    for (let k = 0; k < table.values.length; ++k) {
      if (/angle/i.test(table.components[k % width])) continue;
      largest = Math.max(largest, Math.abs(table.values[k]));
      difference = Math.max(difference, Math.abs(table.values[k] - other.values[k]));
    }
  }
  return difference / Math.max(largest, Number.MIN_VALUE);
}

// The full and the streamed runs of a model, compared on the displacements and the element stresses (`elementTypes`:
// the stress tables that must be there).
function assertStreamedIsFull(text, options, tolerance, elementTypes = []) {
  withModel(text, options, (model) => {
    const full = solver.run(model, "statics");
    assert.ok(!full.diagnostics.some((d) => d.severity === "error"), JSON.stringify(full.diagnostics));
    const streamed = solver.run(model, "statics", { streamed: true });
    assert.ok(!streamed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(streamed.diagnostics));
    for (const type of elementTypes)
      assert.ok(streamed.elementTables.some((table) => table.kind === "stress" && table.elementType === type),
        `the streamed path gives the ${type} stresses`);
    const displacements = relativeDifference(full, streamed, "displacements");
    assert.ok(displacements < tolerance, `displacements ${displacements}`);
    const stresses = relativeDifference(full, streamed, "stress", "elementTables");
    assert.ok(stresses < tolerance, `stresses ${stresses}`);
  });
}

test("the streamed path of two plates joined by springs is the full path's", () => {
  assertStreamedIsFull(platePair("springs"), {}, SAME_RESULT, ["ShellQuad4"]);
});

// Springs on the translations: the rotations of the shells' grid points are excluded (filled from every shell there
// after the solve); on the rotations too: observer relations; a fastener: it reads the shells' entities.
for (const connection of ["springs", "rotationSprings", "fastener"])
  test(`the streamed path of two mixed hybrid plates connection=${connection} is the full path's`, () => {
    assertStreamedIsFull(overlappingPlates(connection), MIXED_OPTIONS, MIXED_SAME_RESULT, ["ShellQuad4"]);
  });

// A block of 2 × 1 × 1 SolidHexa (its twelve edges sharp), clamped at x = 0, pulled and bent at x = 2 and under
// gravity: the solids' entities in the windows, the grid point stresses (SolidGridPoints) from the whole vectors.
function solidBlock() {
  const nx = 2;
  const node = (i, j, k) => (k * 2 + j) * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "StressOutput;All", "[model]"];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i <= nx; ++i) lines.push(`Node;${node(i, j, k)};;${real(i)};${real(j)};${real(k)}`);
  for (let i = 0; i < nx; ++i)
    lines.push(`SolidHexa;${i + 1};1;${[node(i, 0, 0), node(i + 1, 0, 0), node(i + 1, 1, 0), node(i, 1, 0),
      node(i, 0, 1), node(i + 1, 0, 1), node(i + 1, 1, 1), node(i, 1, 1)].join(";")}`);
  lines.push("SolidProp;1;1", "IsoMat;1;2.e11;;0.3;7800.", "GravityLoad;2;;9.81;0.;-1.;0.");
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j) {
      lines.push(`SinglePointConstrList;1;123;${node(0, j, k)}`);
      lines.push(`PointForceLoad;2;${node(nx, j, k)};;1.e6;1.;0.;${real(0.5 * (k - 0.5))}`);
    }
  const sharpEdges = [];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i < nx; ++i) sharpEdges.push([node(i, j, k), node(i + 1, j, k)]);
  for (const i of [0, nx])
    for (let t = 0; t <= 1; ++t) sharpEdges.push([node(i, 0, t), node(i, 1, t)], [node(i, t, 0), node(i, t, 1)]);
  return { text: lines.join("\n") + "\n",
    options: { elementFormulation: "MixedHybridFormulation", formulationGeometry: { sharpEdges } } };
}

test("the streamed path of a mixed hybrid block of SolidHexa under end forces and gravity is the full path's", () => {
  const { text, options } = solidBlock();
  assertStreamedIsFull(text, options, MIXED_SAME_RESULT, ["SolidHexa", "SolidGridPoints"]);
});

// A free strip of 4 × 1 mixed hybrid ShellQuad4 under a pressure on half of it and a corner force, balanced by
// inertia relief on six FreeBodySupport DOFs: the support motions take the rigid-body motions of the shells'
// entities.
function freeStrip() {
  const nx = 4;
  const node = (i, j) => j * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "StressOutput;All", "[model]"];
  for (let j = 0; j <= 1; ++j)
    for (let i = 0; i <= nx; ++i) lines.push(`Node;${node(i, j)};;${real(0.5 * i)};${real(0.4 * j)};0.`);
  for (let i = 0; i < nx; ++i)
    lines.push(`ShellQuad4;${i + 1};1;${node(i, 0)};${node(i + 1, 0)};${node(i + 1, 1)};${node(i, 1)}`);
  lines.push("ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.", "ShellPressureLoad;2;100.;1;Through;2",
    `PointForceLoad;2;${node(nx, 1)};;10.;0.;0.;1.`, "Parameter;InertiaRelief;-1",
    `FreeBodySupport;${node(0, 0)};123;${node(nx, 0)};23;${node(0, 1)};3`);
  return lines.join("\n") + "\n";
}

test("the streamed path of a free mixed hybrid strip under inertia relief is the full path's", () => {
  assertStreamedIsFull(freeStrip(), MIXED_OPTIONS, MIXED_SAME_RESULT, ["ShellQuad4"]);
});
