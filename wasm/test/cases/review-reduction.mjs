// Cases of the review of the condensation and of the outputs: the key of a condensed unit reads the normals its
// shells share at their grid points, and a model with an external unit gives the element tables of the model without
// it (the corner rows, the vectorised recovery of several cases).

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { real } from "../helpers/static-models.mjs";

// The relative difference below which two results are one (round-off of two paths on these small models).
const SAME_RESULT = 1e-6;
// The divisions of each plate: its centre grid point is interior to its unit.
const DIVISIONS = 2;

// Two plates of n × n ShellQuad4 side by side, A on x in [0, 1] held along x = 0, B on x in [1, 2] loaded along
// x = 2, their own grid points on x = 1 joined by springs on components 1 to 5 between coincident points (two
// components of the reader; stiffnesses that no rigid motion strains, as the K* of an external unit must be).
// `subcases` static subcases, the odd ones under load set 2, the even ones under load set 3. Gives the deck, the
// external unit whose matrix is that of the springs, and the grid point at the centre of plate A.
function springJoinedPlates({ subcases = 1, output = ["StressOutput;All"], extra = [] } = {}) {
  const n = DIVISIONS;
  const id = (plate, i, j) => plate * 100 + i * (n + 1) + j + 1;
  const bulk = [];
  for (const plate of [0, 1])
    for (let i = 0; i <= n; ++i)
      for (let j = 0; j <= n; ++j) bulk.push(`Node;${id(plate, i, j)};;${real(plate + i / n)};${real(j / n)};0.`);
  let element = 1;
  for (const plate of [0, 1])
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        bulk.push(`ShellQuad4;${element++};1;${id(plate, i, j)};${id(plate, i + 1, j)};${id(plate, i + 1, j + 1)};` +
          `${id(plate, i, j + 1)}`);
  bulk.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.");
  const springs = []; // [ID, grid of A, grid of B, component]
  for (let j = 0; j <= n; ++j)
    for (let c = 1; c <= 5; ++c) {
      springs.push([500 + springs.length, id(0, n, j), id(1, 0, j), c]);
      const [spring, a, b] = springs[springs.length - 1];
      bulk.push(`SpringDofInline;${spring};1000.;${a};${c};${b};${c}`);
    }
  bulk.push(`SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, j) => id(0, 0, j)).join(";")}`);
  for (let j = 0; j <= n; ++j)
    bulk.push(`PointForceLoad;2;${id(1, n, j)};;1.;0.;0.3;1.`, `PointForceLoad;3;${id(1, n, j)};;2.;1.;0.;-0.5`);
  const cases = [];
  for (let s = 1; s <= subcases; ++s)
    cases.push(`SubcaseStart;${s}`, "SinglePointConstrSelection;1", `StaticLoadSelection;${s % 2 ? 2 : 3}`);
  const text = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", ...output, ...cases, "[model]", ...bulk, ...extra,
    "Parameter;AutoSingularityConstr;Yes", ""].join("\n");
  // K* of the springs: k on the diagonal of both DOFs of a spring, −k between them.
  const labels = [];
  const rowOf = new Map();
  for (const [, a, b, c] of springs)
    for (const grid of [a, b])
      if (!rowOf.has(`${grid}:${c}`)) {
        rowOf.set(`${grid}:${c}`, labels.length);
        labels.push([grid, c]);
      }
  const size = labels.length;
  const stiffness = new Array(size * size).fill(0);
  for (const [, a, b, c] of springs) {
    const [p, q] = [rowOf.get(`${a}:${c}`), rowOf.get(`${b}:${c}`)];
    stiffness[p * size + p] += 1000;
    stiffness[q * size + q] += 1000;
    stiffness[p * size + q] -= 1000;
    stiffness[q * size + p] -= 1000;
  }
  const grids = [...new Set(labels.map(([grid]) => grid))];
  const unit = { zone: { id: 1, name: "springs", elements: springs.map(([spring]) => spring),
    boundary: grids.map((grid) => ({ grid, components: "12345" })) }, labels, stiffness };
  return { text, unit, centreOfA: id(0, n / 2, n / 2) };
}

// Runs `action(model)` on the model of a deck, read with its component tables when `condensation` is asked.
function withDeck(text, options, action) {
  oor.writeFile("/work/review-reduction.oos", text);
  const deck = oor.readModel("/work/review-reduction.oos", { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck, options);
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

function assertNoError(run) {
  assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
}

// The tables of a kind of two runs hold the same rows (elements or points, places) and the same values to
// SAME_RESULT of their largest. The principal angles are left out: at a point of equal principal stresses round-off
// turns them by 180 degrees. With `elementType`, the tables of that element type only.
function assertSameTables(expected, actual, kind, tables, elementType = null) {
  const expectedTables = expected[tables].filter((table) => table.kind === kind &&
    (elementType === null || table.elementType === elementType));
  assert.ok(expectedTables.length > 0, `the reference run gives ${kind}`);
  for (const table of expectedTables) {
    const other = actual[tables].find((candidate) => candidate.kind === kind && candidate.subcase === table.subcase &&
      (candidate.elementType ?? "") === (table.elementType ?? ""));
    const what = `${kind} ${table.elementType ?? ""} of subcase ${table.subcase}`;
    assert.ok(other, `the run gives ${what}`);
    assert.deepEqual([...other.ids], [...table.ids], `the rows of ${what}`);
    if (table.nodeIds) assert.deepEqual([...other.nodeIds], [...table.nodeIds], `the places of ${what}`);
    const width = table.components.length;
    // A value that is not a number (a quantity an element does not define) is compared as such.
    const isCompared = (k) => !/angle/i.test(table.components[k % width]) && Number.isFinite(table.values[k]);
    const largest = Math.max(...Array.from(table.values, (value, k) => (isCompared(k) ? Math.abs(value) : 0)), 1e-30);
    table.values.forEach((value, k) => {
      const where = `${what}, row ${Math.floor(k / width)} ${table.components[k % width]}`;
      if (isCompared(k))
        assert.ok(Math.abs(other.values[k] - value) < SAME_RESULT * largest,
          `${where}: ${other.values[k]} and ${value}`);
      else if (Number.isNaN(value))
        assert.ok(Number.isNaN(other.values[k]), `${where}: ${other.values[k]} where the reference has no value`);
    });
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The condensation cache: the normal a ShellQuad4 takes at a grid point (the mean over the elements around it, or a
// ShellNormal card) changes its stiffness, and neither is seen by the reader's revision stamp of the unit: the key of
// the unit reads it, so that the unit is condensed again, and stored.

test("the condensation cache condenses a unit again when a ShellNormal card changes the normals of its shells", () => {
  const plain = springJoinedPlates();
  const turned = springJoinedPlates({ extra: [`ShellNormal;${plain.centreOfA};;0.05;0.;1.`] });
  const cache = solver.condensationCache();
  try {
    const settings = { condensation: { units: "all", cache } };
    withDeck(plain.text, { condensation: true }, (model) => {
      const first = solver.run(model, "statics", settings);
      assertNoError(first);
      assert.deepEqual(first.condensation.condensedUnits, [1, 2]);
      assert.equal(cache.statistics().misses, 2, "the first run condenses both units");
      solver.run(model, "statics", settings);
      assert.equal(cache.statistics().memoryHits, 2, JSON.stringify(cache.statistics()));
    });
    withDeck(turned.text, { condensation: true }, (model) => {
      const full = solver.run(model, "statics");
      const before = cache.statistics();
      const condensed = solver.run(model, "statics", settings);
      const after = cache.statistics();
      assertNoError(condensed);
      assert.ok(!condensed.diagnostics.some((d) => d.code === "cache-key-stale"),
        JSON.stringify(condensed.diagnostics));
      // Plate A is condensed again (its key differs); both units are looked up once.
      assert.ok(after.misses - before.misses >= 1, JSON.stringify(after));
      assert.equal(after.misses - before.misses + after.memoryHits - before.memoryHits, 2, JSON.stringify(after));
      for (const kind of ["displacements", "singlePointConstrForces"])
        assertSameTables(full, condensed, kind, "nodalTables");
      assertSameTables(full, condensed, "stress", "elementTables");
      // The new condensation was stored: the next run takes both units from the cache.
      solver.run(model, "statics", settings);
      const last = cache.statistics();
      assert.equal(last.memoryHits - after.memoryHits, 2, JSON.stringify(last));
      assert.equal(last.misses, after.misses, JSON.stringify(last));
    });
  } finally {
    cache.delete();
  }
});

// ---------------------------------------------------------------------------------------------------------------------
// External units: the elements of the model that the unit does not replace are recovered by their own families in
// full. The unit replaces the springs alone here, so every table of the shells is that of the model with its springs.

test("an external unit keeps the corner rows of the ShellQuad4 of the model", () => {
  const { text, unit } = springJoinedPlates({
    output: ["StressOutput;All;CornerResults", "ElemForceOutput;All;CornerResults"] });
  withDeck(text, {}, (model) => {
    const full = solver.run(model, "statics");
    const external = solver.run(model, "statics", { externalUnits: [unit] });
    assertNoError(external);
    for (const kind of ["displacements", "singlePointConstrForces"])
      assertSameTables(full, external, kind, "nodalTables");
    for (const kind of ["stress", "forces"]) {
      const table = external.elementTables.find((candidate) => candidate.kind === kind &&
        candidate.elementType === "ShellQuad4");
      assert.ok(table, `the run with the external unit gives the ${kind} of the ShellQuad4`);
      assert.ok(table.nodeIds.some((node) => node !== 0), `the ${kind} table holds the corner rows`);
      assertSameTables(full, external, kind, "elementTables", "ShellQuad4");
    }
  });
});

// The element tables of a block of cases are made from the recovery slopes of each element when the cases are enough
// to pay for them (the vectorised recovery): with an external unit as without one.
const VECTORISED_SUBCASES = 8;

test("an external unit gives the element tables of a block of cases of the model", () => {
  const { text, unit } = springJoinedPlates({ subcases: VECTORISED_SUBCASES,
    output: ["StressOutput;All;CornerResults", "StrainOutput;All", "ElemForceOutput;All"] });
  withDeck(text, {}, (model) => {
    const full = solver.run(model, "statics");
    const external = solver.run(model, "statics", { externalUnits: [unit] });
    assertNoError(external);
    assertSameTables(full, external, "displacements", "nodalTables");
    for (const kind of ["stress", "strain", "forces"]) {
      const count = external.elementTables.filter((table) => table.kind === kind &&
        table.elementType === "ShellQuad4").length;
      assert.equal(count, VECTORISED_SUBCASES, `a ${kind} table of the ShellQuad4 per subcase`);
      assertSameTables(full, external, kind, "elementTables", "ShellQuad4");
    }
  });
});
