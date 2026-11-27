// Cases of the eigenvalue solvers a run may choose for normal modes and buckling (settings.eigenSolver):
// - the Lanczos solver with its factor in scratch files finds the roots of the Lanczos solver in memory;
// - multilevel substructuring finds them within its accuracy;
// - the report of the run names the solver that ran;
// - a solver is refused where no eigenvalue problem is solved, and a name or a key the interface does not have.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver } from "../helpers/runtime.mjs";
import { real, withModel } from "../helpers/static-models.mjs";

// A square plate of CELLS × CELLS ShellQuad4 (units: N, mm, t): steel, 2 mm thick, its side X = 0 clamped and its
// three other sides held along Z.
const CELLS = 6;
const SIDE = 120;
const node = (i, j) => 1 + i + (CELLS + 1) * j;

function plateRecords() {
  const records = [];
  for (let j = 0; j <= CELLS; ++j)
    for (let i = 0; i <= CELLS; ++i)
      records.push(`Node;${node(i, j)};;${real((SIDE * i) / CELLS)};${real((SIDE * j) / CELLS)};0.`);
  for (let j = 0; j < CELLS; ++j)
    for (let i = 0; i < CELLS; ++i)
      records.push(`ShellQuad4;${1 + i + CELLS * j};1;${node(i, j)};${node(i + 1, j)};${node(i + 1, j + 1)};` +
        `${node(i, j + 1)}`);
  const clamped = [];
  const held = [];
  for (let k = 0; k <= CELLS; ++k) {
    clamped.push(node(0, k));
    held.push(node(CELLS, k));
    if (k > 0 && k < CELLS) held.push(node(k, 0), node(k, CELLS));
  }
  records.push("ShellProp;1;1;2.;1;;1", "IsoMat;1;210000.;;0.3;7.85e-9",
    `SinglePointConstrList;1;123456;${clamped.join(";")}`, `SinglePointConstrList;1;3;${held.join(";")}`,
    "LanczosEigenMethod;3;;;4");
  return records;
}

// The modes deck (four modes), and the buckling deck: a compression along X on the side X = SIDE, then its first four
// load factors.
function modesDeck() {
  return ["[executive]", "SolutionSequence;103", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "EigenMethodSelection;3", "[model]", ...plateRecords(), ""].join("\n");
}

function bucklingDeck() {
  const loads = [];
  for (let k = 0; k <= CELLS; ++k)
    loads.push(`PointForceLoad;2;${node(CELLS, k)};;${k === 0 || k === CELLS ? "5." : "10."};-1.;0.;0.`);
  return ["[executive]", "SolutionSequence;105", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "SubcaseStart;2", "SinglePointConstrSelection;1", "EigenMethodSelection;3",
    "StaticSubcaseSelection;1", "[model]", ...plateRecords(), ...loads, ""].join("\n");
}

// The eigenvalues of the last eigenvalue table of a run, and its report.
function rootsOf(text, analysis, eigenSolver) {
  return withModel(text, (model) => {
    const run = solver.run(model, analysis, eigenSolver === undefined ? {} : { eigenSolver });
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const rows = run.eigenvalues[run.eigenvalues.length - 1].rows;
    return { values: rows.map((row) => row.eigenvalue), report: run.report };
  });
}

function assertRoots(actual, expected, tolerance, what) {
  assert.equal(actual.length, expected.length, `${what}: the number of roots`);
  actual.forEach((value, k) => assert.ok(Math.abs(value / expected[k] - 1) <= tolerance,
    `${what}, root ${k + 1}: ${value}, expected ${expected[k]} (tolerance ${tolerance})`));
}

for (const [analysis, deck] of [["modes", modesDeck], ["buckling", bucklingDeck]]) {
  test(`${analysis}: the Lanczos solver with its factor in scratch files finds the roots of the one in memory`, () => {
    const reference = rootsOf(deck(), analysis);
    assert.equal(reference.report.eigenSolver, "lanczos");
    assert.equal(reference.values.length, 4);
    assert.ok(reference.values.every((value) => value > 0));
    const named = rootsOf(deck(), analysis, "outOfCore");
    assert.equal(named.report.eigenSolver, "outOfCore");
    assertRoots(named.values, reference.values, 1e-8, "outOfCore");
    const inFolder = rootsOf(deck(), analysis, { solver: "outOfCore", scratchFolder: "/work" });
    assertRoots(inFolder.values, reference.values, 1e-8, "outOfCore in a folder");
  });

  test(`${analysis}: multilevel substructuring finds the roots within its accuracy`, () => {
    const reference = rootsOf(deck(), analysis);
    const approximate = rootsOf(deck(), analysis, "substructuring");
    assert.equal(approximate.report.eigenSolver, "substructuring");
    assertRoots(approximate.values, reference.values, 5e-3, "substructuring");
    // A higher cut-off keeps more modes of each subtree: at least as accurate.
    const finer = rootsOf(deck(), analysis, { solver: "substructuring", cutoffFactor: 8 });
    assertRoots(finer.values, reference.values, 5e-3, "substructuring with a higher cut-off");
  });
}

test("an eigenvalue solver is refused where no eigenvalue problem is solved", () => {
  withModel(bucklingDeck(), (model) => {
    assert.throws(() => solver.run(model, "statics", { eigenSolver: "substructuring" }),
      /chosen for normal modes and buckling only/);
    // The default solver named is no choice: statics runs.
    assert.ok(solver.run(model, "statics", { eigenSolver: "lanczos" }).report.eigenSolver === "");
  });
});

test("an eigenvalue solver or a key the interface does not have is refused by name", () => {
  withModel(modesDeck(), (model) => {
    assert.throws(() => solver.run(model, "modes", { eigenSolver: "amls" }),
      /eigenSolver must be "lanczos", "substructuring" or "outOfCore", found "amls"/);
    assert.throws(() => solver.run(model, "modes", { eigenSolver: { solver: "outOfCore", folder: "/work" } }),
      (error) => error instanceof TypeError && /unknown key 'folder'/.test(error.message) &&
        error.message.includes("scratchFolder"));
    assert.throws(() => solver.run(model, "modes", { eigenSolver: { solver: "substructuring", cutoffFactor: -1 } }),
      /cut-off factor/);
  });
});
