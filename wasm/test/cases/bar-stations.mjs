// The output stations of the Bar2 (the Bar2Stations cards): a bar with stations is written in the station layout
// (the table of the bars under distributed loads), at its end A, at its stations and at its end B, in every subcase;
// the other bars keep their ordinary table. A cantilever of two bars along X, clamped at its first grid point.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

const BAR = { length: 2, area: 1e-3, inertia: 2e-7, young: 7e10 };
// The recovery point C of the section, at this distance from the neutral axis in plane 1.
const POINT_C = 0.02;

// The cantilever: bars 1 (grid points 1-2) and 2 (2-3), with the Case Control `control` and the cards `cards`.
function cantilever({ control = [], cards = [] } = {}) {
  return ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All", "ElemForceOutput;All",
    "StressOutput;All", "SinglePointConstrSelection;1", ...control, "[model]",
    "Node;1;;0.;0.;0.", `Node;2;;${real(BAR.length)};0.;0.`, `Node;3;;${real(2 * BAR.length)};0.;0.`,
    // The orientation vector along Y: plane 1 is XY, plane 2 is XZ.
    "Bar2;1;7;1;2;0.;;1.;0.", "Bar2;2;7;2;3;0.;;1.;0.",
    `BarProp;7;1;${real(BAR.area)};${real(BAR.inertia)};${real(BAR.inertia)};1.e-7;;${real(POINT_C)};0.`,
    `IsoMat;1;${real(BAR.young)};;0.3`, "SinglePointConstrList;1;123456;1", ...cards, ""].join("\n");
}

const ONE_SUBCASE = ["SubcaseStart;1", "StaticLoadSelection;2"];
// A unit force along Y at the free end, and a pull of 3 along X.
const TIP_LOADS = ["PointForceLoad;2;3;;1.;3.;1.;0."];

let deckCount = 0;

function run(text, options = {}) {
  const file = `/work/bar-stations-${++deckCount}.oos`;
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  try {
    assert.deepEqual(deck.diagnostics().filter((d) => d.severity === "error"), [], "the model file is read");
    const model = solver.modelFromReader(oor, deck, options);
    try { return solver.run(model, "statics"); } finally { model.delete(); }
  } finally { deck.delete(); }
}

// The rows of an element table of a subcase, each with its element, its position along the bar and its components.
function rowsOf(result, elementType, kind, subcase = 1) {
  const table = result.elementTables.find((t) => t.elementType === elementType && t.kind === kind &&
    t.subcase === subcase);
  if (!table) return [];
  const width = table.components.length;
  return Array.from(table.ids, (id, r) => {
    const row = { id, position: table.positions?.length ? table.positions[r] : 0 };
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    return row;
  });
}

const close = (actual, expected, what, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${what}: ${actual}, expected ${expected}`);

// The moment of the cantilever under the unit tip force, at the distance x from its clamped end.
const tipMoment = (x) => 2 * BAR.length - x;

for (const [form, card] of [["listed as fractions of its length", "Bar2Stations;1;FractionalLength;0.25;;0.5;0.75"],
  ["equally spaced along its length", "Bar2Stations;1;ActualLength;;3;0.5;0.5"],
  ["listed in disorder, with its ends", "Bar2Stations;1;FractionalLength;0.75;;1.;0.25;0.5"]])
  test(`a Bar2 with stations ${form} is written at its ends and at its stations`, () => {
    const plain = run(cantilever({ control: ONE_SUBCASE, cards: TIP_LOADS }));
    const result = run(cantilever({ control: ONE_SUBCASE, cards: [...TIP_LOADS, card] }));
    assert.ok(!result.diagnostics.some((d) => d.severity === "error" || d.code === "ignored-card"),
      JSON.stringify(result.diagnostics));
    // Bar 2 keeps its ordinary table; bar 1 leaves it for the station layout.
    assert.deepEqual(rowsOf(result, "Bar2", "forces").map((row) => row.id), [2]);
    const stations = rowsOf(result, "Bar2LoadStations", "forces");
    assert.deepEqual(stations.map((row) => row.id), [1, 1, 1, 1, 1]);
    assert.deepEqual(stations.map((row) => row.position), [0, 0.25, 0.5, 0.75, 1]);
    const ends = rowsOf(plain, "Bar2", "forces").find((row) => row.id === 1);
    for (const row of stations) {
      // The force along Y bends the bar in its plane 1: |M1| = L_total − x, a constant shear and the pull.
      close(Math.abs(row.bendingMoment1), tipMoment(row.position * BAR.length), `moment at ${row.position}`);
      close(row.bendingMoment2, 0, `moment in plane 2 at ${row.position}`);
      close(row.shear1, ends.shear1, `shear at ${row.position}`);
      close(row.axial, 3, `axial force at ${row.position}`);
    }
    // The ends are those of the bar without stations, signs included.
    close(stations[0].bendingMoment1, ends.bendingMomentA1, "moment at end A");
    close(stations[4].bendingMoment1, ends.bendingMomentB1, "moment at end B");
    // The stresses of the stations: the bending stress of point C follows the moment, on the axial stress.
    const stresses = rowsOf(result, "Bar2LoadStations", "stress");
    assert.deepEqual(stresses.map((row) => row.position), [0, 0.25, 0.5, 0.75, 1]);
    for (const row of stresses) {
      close(row.axial, 3 / BAR.area, `axial stress at ${row.position}`);
      close(Math.abs(row.sxc), tipMoment(row.position * BAR.length) * POINT_C / BAR.inertia,
        `bending stress of point C at ${row.position}`);
    }
    // The displacements do not depend on the stations.
    const tip = (r) => Array.from(r.nodalTables.find((t) => t.kind === "displacements").values.slice(12, 18));
    assert.deepEqual(tip(result), tip(plain));
  });

test("the stations of a Bar2 under a distributed load carry the load between end A and the station", () => {
  // A uniform load q along Y on bar 1 alone: with the free end unloaded, the shear at x is q (L − x) and the moment
  // q (L − x)² / 2 on bar 1, nothing on bar 2.
  const q = 4;
  const result = run(cantilever({ control: ONE_SUBCASE, cards: [
    `BeamDistributedLoad;2;1;ForceY;FractionalLength;0.;${real(q)};1.;${real(q)}`,
    "Bar2Stations;1;FractionalLength;0.5", "Bar2Stations;2;FractionalLength;0.5"] }));
  assert.ok(!result.diagnostics.some((d) => d.severity === "error"), JSON.stringify(result.diagnostics));
  assert.deepEqual(rowsOf(result, "Bar2", "forces"), [], "both bars are in the station layout");
  const stations = rowsOf(result, "Bar2LoadStations", "forces");
  assert.deepEqual(stations.map((row) => [row.id, row.position]), [[1, 0], [1, 0.5], [1, 1], [2, 0], [2, 0.5], [2, 1]]);
  for (const row of stations) {
    const left = row.id === 1 ? BAR.length * (1 - row.position) : 0;
    close(Math.abs(row.shear1), q * left, `shear of bar ${row.id} at ${row.position}`);
    close(Math.abs(row.bendingMoment1), 0.5 * q * left * left, `moment of bar ${row.id} at ${row.position}`);
  }
});

test("the stations of a Bar2 are written in every subcase of a block of load cases", () => {
  // Nine subcases (the recovery by blocks) of the tip loads scaled: the rows of each are the first one's times its
  // factor, bar 2 in its ordinary table.
  const cases = 9;
  const control = Array.from({ length: cases }, (_, k) => [`SubcaseStart;${k + 1}`, `StaticLoadSelection;${11 + k}`])
    .flat();
  const loads = Array.from({ length: cases }, (_, k) => `PointForceLoad;${11 + k};3;;${real(k + 1)};3.;1.;0.`);
  const result = run(cantilever({ control, cards: [...loads, "Bar2Stations;1;FractionalLength;0.25;;0.5"] }));
  assert.ok(!result.diagnostics.some((d) => d.severity === "error"), JSON.stringify(result.diagnostics));
  const first = rowsOf(result, "Bar2LoadStations", "forces", 1);
  assert.deepEqual(first.map((row) => row.position), [0, 0.25, 0.5, 1]);
  for (let c = 2; c <= cases; ++c) {
    assert.deepEqual(rowsOf(result, "Bar2", "forces", c).map((row) => row.id), [2],
      `the ordinary rows of subcase ${c}`);
    rowsOf(result, "Bar2LoadStations", "forces", c).forEach((row, r) => {
      assert.equal(row.position, first[r].position);
      for (const name of ["bendingMoment1", "shear1", "axial"])
        close(row[name], c * first[r][name], `${name} at ${row.position} of subcase ${c}`, 1e-8);
    });
  }
});

test("a station beyond the end of its Bar2 is refused, a card that names no Bar2 is pointed out", () => {
  const beyond = cantilever({ control: ONE_SUBCASE, cards: [...TIP_LOADS, "Bar2Stations;1;ActualLength;2.5"] });
  assert.throws(() => run(beyond), /Bar2Stations 1.*between the ends of its Bar2/s);
  const unknown = run(cantilever({ control: ONE_SUBCASE,
    cards: [...TIP_LOADS, "Bar2Stations;1;FractionalLength;0.5", "Bar2Stations;99;FractionalLength;0.5"] }));
  const warnings = unknown.diagnostics.filter((d) => d.code === "ignored-card");
  assert.equal(warnings.length, 1, JSON.stringify(unknown.diagnostics));
  assert.equal(warnings[0].card, "Bar2Stations");
  assert.equal(warnings[0].id, 99);
  assert.equal(warnings[0].severity, "warning");
  assert.deepEqual(rowsOf(unknown, "Bar2LoadStations", "forces").map((row) => row.position), [0, 0.5, 1]);
});

test("the stations of a Bar2 follow the selection of the output request", () => {
  // The forces of bar 2 alone: its stations, and no row of bar 1 in either table.
  const text = cantilever({ control: ["SetDefinition;5;2", ...ONE_SUBCASE], cards: [...TIP_LOADS,
    "Bar2Stations;1;FractionalLength;0.5", "Bar2Stations;2;FractionalLength;0.5"] })
    .replace("ElemForceOutput;All", "ElemForceOutput;5");
  const result = run(text);
  assert.ok(!result.diagnostics.some((d) => d.severity === "error"), JSON.stringify(result.diagnostics));
  assert.deepEqual(rowsOf(result, "Bar2LoadStations", "forces").map((row) => [row.id, row.position]),
    [[2, 0], [2, 0.5], [2, 1]]);
  assert.deepEqual(rowsOf(result, "Bar2", "forces"), []);
});
