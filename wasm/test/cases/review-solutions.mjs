// Cases of the review of the solution sequences (src/solutions): each one states a rule a subcase, a path or a method
// range follows, on a model small enough to be checked by hand (rods of stiffness k = E A / L = 2000, a column under
// its own weight).
import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { displacement, near, real } from "../helpers/static-models.mjs";
import { platePair } from "./condensation.mjs";

// Runs `action(model)` on the model of `text`, read with `options` (those of modelFromReader).
function withModel(text, action, options = {}, name = "review-solutions") {
  oor.writeFile(`/work/${name}.oos`, text);
  const deck = oor.readModel(`/work/${name}.oos`, { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck, options);
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

const deckOf = (solution, control, bulk) =>
  ["[executive]", `SolutionSequence;${solution}`, "[case]", ...control, "[model]", ...bulk, ""].join("\n");

const hasNoError = (run) =>
  assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));

// --- 1. The streamed path and the enforced displacements ----------------------------------------------------------

// A chain of two rods 1-2-3 along x, node 1 held, node 3 held by the SPC set too. Load set 2 is a force of 10 on node
// 2: u2 = 10 / (2 k) = 0.0025. Load set 3 moves node 3 by 0.06 (an enforced displacement) and pulls node 2 by 4:
// u2 = (k · 0.06 + 4) / (2 k) = 0.031.
function enforcedChain(loadSets) {
  const control = ["DisplacementOutput;All"];
  loadSets.forEach((set, k) =>
    control.push(`SubcaseStart;${k + 1}`, "SinglePointConstrSelection;1", `StaticLoadSelection;${set}`));
  return deckOf(101, control, ["Node;1;;0.;0.;0.", "Node;2;;1.;0.;0.", "Node;3;;2.;0.;0.", "Rod2;1;1;1;2",
    "Rod2;2;1;2;3", "RodProp;1;1;2.", "IsoMat;1;1000.;;0.3;1.", "SinglePointConstrList;1;123456;1;3",
    "SinglePointConstrList;1;23456;2", "PointForceLoad;2;2;;10.;1.;0.;0.", "PointForceLoad;3;2;;4.;1.;0.;0.",
    "EnforcedMotionConstr;3;3;1;0.06"]);
}

test("the full path gives each subcase of a constraint set its own enforced displacement", () => {
  withModel(enforcedChain([2, 3]), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run, 2, 0, 1), 0.0025, "node 2 under the force");
    near(displacement(run, 3, 0, 1), 0, "node 3 held in the subcase without the enforced displacement");
    near(displacement(run, 3, 0, 2), 0.06, "node 3 at its enforced displacement");
    near(displacement(run, 2, 0, 2), 0.031, "node 2 between the support and the moved end");
  });
});

// The streamed path takes no enforced displacement: it must say so whichever subcase of the constraint set has it
// (it read the enforced values of the first subcase for every one, and solved a later one with its DOF at zero).
for (const [order, loadSets] of [["last", [2, 3]], ["first", [3, 2]]])
  test(`the streamed path refuses an enforced displacement of the ${order} subcase of a constraint set`, () => {
    withModel(enforcedChain(loadSets), (model) => {
      assert.throws(() => solver.run(model, "statics", { streamed: true }), /streamed-load-unsupported/);
    });
  });

// --- 2. The constraint sets of the residual structure of a superelement run ---------------------------------------

// The residual structure is the chain 1-5-2 (node 1 held), the part the rod 2-3, joined at node 2. Subcase 1: 10 on
// node 5 and the part's 5 on node 3, so u5 = 15 / k, u2 = u5 + 5 / k, u3 = u2 + 5 / k. Subcase 2 selects the same SPC
// set, but its load set moves node 2 by 0.06 (an enforced displacement on a DOF the SPC set does not hold) beside 2 on
// node 5 and the part's 4 on node 3: u5 = (k · 0.06 + 2) / (2 k), u3 = 0.06 + 4 / k. The two subcases hold different
// DOFs: two constraint sets, though they select one SPC set.
function superelementChain() {
  const control = ["DisplacementOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "SubcaseStart;2", "SinglePointConstrSelection;1", "StaticLoadSelection;3"];
  const common = ["RodProp;1;1;2.", "IsoMat;1;1000.;;0.3;1."];
  return {
    residual: deckOf(101, control, ["Node;1;;0.;0.;0.", "Node;5;;1.;0.;0.", "Node;2;;2.;0.;0.", "Rod2;1;1;1;5",
      "Rod2;2;1;5;2", ...common, "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;5;2",
      "PointForceLoad;2;5;;10.;1.;0.;0.", "PointForceLoad;3;5;;2.;1.;0.;0.", "EnforcedMotionConstr;3;2;1;0.06"]),
    part: deckOf(101, control, ["Node;2;;2.;0.;0.", "Node;3;;3.;0.;0.", "Rod2;11;1;2;3", ...common,
      "SinglePointConstrList;1;23456;2;3", "PointForceLoad;2;3;;5.;1.;0.;0.", "PointForceLoad;3;3;;4.;1.;0.;0."]),
  };
}

test("the residual structure of a superelement run solves subcases that hold different DOFs apart", () => {
  const decks = superelementChain();
  const read = (text, name) => {
    oor.writeFile(`/work/${name}.oos`, text);
    return oor.readModel(`/work/${name}.oos`);
  };
  const residualDeck = read(decks.residual, "review-superelement-residual");
  const partDeck = read(decks.part, "review-superelement-part");
  let residual;
  let part;
  try {
    residual = solver.modelFromReader(oor, residualDeck);
    part = solver.modelFromReader(oor, partDeck);
    const run = solver.runSuperelements(residual, [{ id: 1, model: part, boundary: [[2, 2]] }], "statics");
    hasNoError(run);
    const of = (id) => run.superelements.find((entry) => entry.id === id);
    const k = 2000;
    near(displacement(of(0), 5, 0, 1), 15 / k, "subcase 1, node 5");
    near(displacement(of(0), 2, 0, 1), 20 / k, "subcase 1, node 2");
    near(displacement(of(1), 3, 0, 1), 25 / k, "subcase 1, node 3 of the part");
    near(displacement(of(0), 2, 0, 2), 0.06, "subcase 2, node 2 at its enforced displacement");
    near(displacement(of(0), 5, 0, 2), (k * 0.06 + 2) / (2 * k), "subcase 2, node 5 follows the moved node");
    near(displacement(of(1), 3, 0, 2), 0.06 + 4 / k, "subcase 2, node 3 of the part");
  } finally {
    part?.delete();
    residual?.delete();
    partDeck.delete();
    residualDeck.delete();
  }
});

// --- 3 and 4. The preload and the range of a buckling subcase ------------------------------------------------------

// A column clamped at its foot under its own weight along its axis (the model of static-inputs.mjs): load set 2 is
// the weight, load set 3 twice the weight (half the load factor), load set 4 the weight reversed (a tension: the load
// factors change sign).
const COLUMN = ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "Bar2;1;1;1;2;0.;;1.;0.", "BarProp;1;1;2.;0.1;0.1;0.05",
  "IsoMat;1;1000.;;0.3;1.", "SinglePointConstrList;1;123456;1", "GravityLoad;2;;10.;-1.;0.;0.",
  "GravityLoad;3;;20.;-1.;0.;0.", "GravityLoad;4;;10.;1.;0.;0.", "LanczosEigenMethod;5;;;1"];
const STATIC = (subcase, load) => [`SubcaseStart;${subcase}`, "SinglePointConstrSelection;1",
  `StaticLoadSelection;${load}`];
const BUCKLING = (subcase, ...more) => [`SubcaseStart;${subcase}`, "SinglePointConstrSelection;1",
  "EigenMethodSelection;5", ...more];

// The load factors of buckling subcase `subcase` of the column with the Case Control `control`.
function loadFactors(control, subcase, bulk = COLUMN) {
  return withModel(deckOf(105, control, bulk), (model) => {
    const run = solver.run(model, "buckling");
    hasNoError(run);
    return run.eigenvalues.find((table) => table.subcase === subcase).rows.map((row) => row.eigenvalue);
  });
}

const sameFactor = (actual, expected, what) =>
  assert.ok(Math.abs(actual / expected - 1) < 1e-9, `${what}: ${actual} differs from ${expected}`);

// The load factor under the weight, its preload named (StaticSubcaseSelection).
const weightFactor = () => loadFactors([...STATIC(1, 2), ...BUCKLING(2, "StaticSubcaseSelection;1")], 2)[0];

test("a buckling subcase without StaticSubcaseSelection takes the first static subcase of the deck", () => {
  const named = weightFactor();
  assert.ok(named > 0, `load factor ${named}`);
  // Two static subcases before it: the first one (the weight), not the one just before (twice the weight).
  sameFactor(loadFactors([...STATIC(1, 2), ...STATIC(2, 3), ...BUCKLING(3)], 3)[0], named, "after two static subcases");
  // The static subcases after it: still the first of them.
  sameFactor(loadFactors([...BUCKLING(1), ...STATIC(2, 2), ...STATIC(3, 3)], 1)[0], named, "before the static subcases");
  // A named preload is kept.
  sameFactor(loadFactors([...STATIC(1, 2), ...STATIC(2, 3), ...BUCKLING(3, "StaticSubcaseSelection;2")], 3)[0],
    named / 2, "the named static subcase");
});

test("a load of the global section does not make a buckling subcase its own preload", () => {
  // The global section selects twice the weight, which the buckling subcase inherits; static subcase 1 selects the
  // weight: the preload is subcase 1.
  const control = ["SinglePointConstrSelection;1", "StaticLoadSelection;3", "SubcaseStart;1", "StaticLoadSelection;2",
    "SubcaseStart;2", "EigenMethodSelection;5"];
  sameFactor(loadFactors(control, 2)[0], weightFactor(), "the first static subcase");
});

test("a deck of one subcase with the load and the method is its own preload", () => {
  const control = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "EigenMethodSelection;5"];
  sameFactor(loadFactors(control, 1)[0], weightFactor(), "the subcase itself");
});

test("a buckling method range wholly below zero gives the negative load factors", () => {
  const critical = weightFactor();
  // Under the reversed weight the column buckles at the load factor -critical: asked by a range around it.
  const [lower, upper] = [-1.5 * critical, -0.5 * critical];
  const bulk = [...COLUMN, `LanczosEigenMethod;6;${real(lower)};${real(upper)}`];
  const control = [...STATIC(1, 4), "SubcaseStart;2", "SinglePointConstrSelection;1", "EigenMethodSelection;6",
    "StaticSubcaseSelection;1"];
  const factors = loadFactors(control, 2, bulk);
  assert.ok(factors.length >= 1, "a load factor in the range");
  for (const factor of factors) assert.ok(factor > lower && factor < upper, `load factor ${factor} in the range`);
  assert.ok(Math.abs(factors[0] / -critical - 1) < 1e-7, `load factor ${factors[0]}, expected ${-critical}`);
});

// --- 5. A trim subcase and the enforced displacements of its load set ---------------------------------------------

// A wing of two plates of 2 × 2 ShellQuad4 joined by springs, held at its root and trimmed on one lifting panel (the
// wing of condensation.mjs). Subcase 2 selects a load set that pulls the tip and moves a root grid point.
function trimmedWing() {
  const n = 2;
  const id = (plate, i, j) => plate * 100 + j * (n + 1) + i + 1;
  const bulk = [];
  const ids = [];
  for (const plate of [0, 1])
    for (let j = 0; j <= n; ++j)
      for (let i = 0; i <= n; ++i) {
        ids.push(id(plate, i, j));
        bulk.push(`Node;${id(plate, i, j)};;${real(i / n)};${real(plate * 1.2 + j / n)};0.`);
      }
  let element = 0;
  for (const plate of [0, 1])
    for (let j = 0; j < n; ++j)
      for (let i = 0; i < n; ++i)
        bulk.push(`ShellQuad4;${++element};1;${id(plate, i, j)};${id(plate, i + 1, j)};${id(plate, i + 1, j + 1)};` +
          `${id(plate, i, j + 1)}`);
  let spring = 500;
  for (let i = 0; i <= n; ++i)
    for (let c = 1; c <= 5; ++c)
      bulk.push(`SpringDofInline;${spring++};1.e4;${id(0, i, n)};${c};${id(1, i, 0)};${c}`);
  bulk.push(`SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, i) => id(0, i, 0)).join(";")}`,
    "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1.e6;;0.3;1.", "AeroStaticParams;0;0;1.;2.2;2.2",
    "AeroLiftingPanelProp1;10", "AeroLiftingPanel1;100;10;0;2;2;0;0;1;0.;0.;0.;1.;0.;2.2;0.;1.",
    `IdListSet1;50;${ids.join(";")}`, "AeroSurfaceSpline1;60;100;100;103;50", "AeroRigidTrimVariable;70;AngleOfAttack",
    "AeroTrim;5;0.1;10.;AngleOfAttack;0.05", `PointForceLoad;9;${id(1, n, n)};;1.;0.;0.;1.`,
    `EnforcedMotionConstr;9;${id(0, 0, 0)};3;0.01`, "Parameter;AutoSingularityConstr;Yes");
  return deckOf(144, ["DisplacementOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "TrimSelection;5",
    "SubcaseStart;2", "SinglePointConstrSelection;1", "TrimSelection;5", "StaticLoadSelection;9"], bulk);
}

test("a trim subcase whose load set brings an enforced displacement is refused, naming the subcase", () => {
  withModel(trimmedWing(), (model) => {
    assert.throws(() => solver.run(model, "aeroelastic"), (error) => {
      const message = String(error?.message ?? error);
      assert.match(message, /trim-enforced/);
      assert.match(message, /\[trim-enforced\][^\n]*subcase 2:/);
      assert.doesNotMatch(message, /\[trim-enforced\][^\n]*subcase 1:/);
      return true;
    });
  });
});

// --- 7. The outputs only the full path makes -----------------------------------------------------------------------

// The grid point weight generator (Parameter WeightReferenceNode) and the UserDofSet table (Parameter DofSetPrint)
// come from the mass and the DOF sets of the full path: the condensed and streamed paths say that they are not made.
const isNoteOf = (parameter) => (d) =>
  d.code === "output-not-produced" && d.severity === "information" && d.message.includes(parameter);

// (The condensed path needs the component tables of the reader: the model option `condensation`.)
for (const [path, settings, modelOptions] of [["streamed", { streamed: true }, {}],
  ["condensed", { condensation: "all" }, { condensation: true }]])
  test(`statics on the ${path} path says that the grid point weight and the DOF set table are not produced`, () => {
    const text = platePair("springs") + "Parameter;WeightReferenceNode;0\nParameter;DofSetPrint;0\n";
    withModel(text, (model) => {
      const full = solver.run(model, "statics");
      hasNoError(full);
      for (const parameter of ["WeightReferenceNode", "DofSetPrint"])
        assert.equal(full.diagnostics.filter(isNoteOf(parameter)).length, 0, `${parameter} on the full path`);
      const run = solver.run(model, "statics", settings);
      hasNoError(run);
      assert.equal(run.diagnostics.filter(isNoteOf("WeightReferenceNode")).length, 1, "the weight, once per run");
      assert.ok(run.diagnostics.filter(isNoteOf("DofSetPrint")).length >= 1, "the DOF set table");
    }, modelOptions);
  });
