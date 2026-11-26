// Cases of the review of the input and assembly modules (src/input, src/assembly): input the solver must refuse
// rather than take without a word, each on a model small enough to be checked by hand. A rod of stiffness
// k = E A / L = 1000 * 2 / 3 along x, node 1 held, node 2 free along x: under its own weight (density 1, g = 10) the
// tip moves by rho g L^2 / (2 E) = 0.045.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { displacement, near } from "../helpers/static-models.mjs";

const ROD = ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "Rod2;1;1;1;2", "RodProp;1;1;2.",
  "IsoMat;1;1000.;;0.3;1.;1.e-5;0.", "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;2",
  "GravityLoad;2;;10.;1.;0.;0."];
const CONTROL = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All"];
const TIP_UNDER_WEIGHT = 0.045;

const deckOf = (control, bulk) =>
  ["[executive]", "SolutionSequence;101", "[case]", ...control, "[model]", ...bulk, ""].join("\n");

// Runs `action(model)` on the model of `text`.
function withModel(text, action) {
  oor.writeFile("/work/review-assembly.oos", text);
  const deck = oor.readModel("/work/review-assembly.oos", { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck);
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

const hasNoError = (run) =>
  assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));

// The statics of the deck stop on a diagnostic of `code` whose message holds every text of `texts`.
function refuses(control, bulk, code, ...texts) {
  withModel(deckOf(control, bulk), (model) => assert.throws(() => solver.run(model, "statics"), (error) => {
    assert.match(error.message, new RegExp(code));
    for (const text of texts) assert.ok(error.message.includes(text), `"${text}" in: ${error.message}`);
    return true;
  }));
}

// The control lines with one selection replaced.
const selecting = (command, set) => CONTROL.map((line) => (line.startsWith(command) ? `${command};${set}` : line));

// --- 1. A set a subcase selects must be defined by a card ---------------------------------------------------------

test("the rod of the review cases is solved without a diagnostic of an undefined set", () => {
  withModel(deckOf(CONTROL, ROD), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run), TIP_UNDER_WEIGHT, "tip under the weight");
  });
});

test("a subcase that selects a constraint set no card defines is refused", () => {
  refuses(selecting("SinglePointConstrSelection", 9), ROD, "spc-unknown-set", "constraint set 9");
});

test("a subcase that selects a multipoint constraint set no card defines is refused", () => {
  refuses([...CONTROL, "MultiPointConstrSelection;99"], ROD, "mpc-unknown-set", "constraint set 99");
});

test("a subcase that selects a load set no card defines is refused", () => {
  refuses(selecting("StaticLoadSelection", 99), ROD, "load-unknown-set", "load set 99");
});

test("a load combination that combines a load set no card defines is refused", () => {
  // Set 2 is the weight; set 77 has no card: left out of the sum, the combination would be the weight alone.
  refuses(selecting("StaticLoadSelection", 30), [...ROD, "LoadCombination;30;1.;1.;2;1.;77"], "load-unknown-set",
    "load set 77");
});

test("a load combination of defined load sets is solved", () => {
  const bulk = [...ROD, "PointForceLoad;3;2;;2.;1.;0.;0.", "LoadCombination;30;1.;1.;2;1.;3"];
  withModel(deckOf(selecting("StaticLoadSelection", 30), bulk), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    // The weight, and the force of 2 on the tip: F / k = 2 * 3 / (1000 * 2) = 0.003.
    near(displacement(run), TIP_UNDER_WEIGHT + 0.003, "tip under the weight and the force");
  });
});

test("a load set that only enforced displacements define is a defined load set", () => {
  // Node 2 held along x by the constraint set, and moved by 0.06 by the load set 3 (no other card of set 3).
  const bulk = [...ROD.filter((line) => line !== "SinglePointConstrList;1;23456;2"),
    "SinglePointConstrList;1;123456;2", "EnforcedMotionConstr;3;2;1;0.06"];
  withModel(deckOf(selecting("StaticLoadSelection", 3), bulk), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run), 0.06, "tip at its enforced displacement");
  });
});

test("a subcase that selects a temperature set no card defines is refused", () => {
  refuses([...CONTROL, "TemperatureSelection;7;TemperatureLoadPart"], ROD, "load-unknown-set", "set 7");
});

test("a load set and a temperature set of one number that only temperature cards define are solved", () => {
  // The thermal elongation of the free rod, alpha * dT * L = 1.e-5 * 20 * 3.
  const control = [...selecting("StaticLoadSelection", 3), "TemperatureSelection;3;TemperatureLoadPart"];
  withModel(deckOf(control, [...ROD, "DefaultTemperatureLoad;3;20."]), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run), 6e-4, "tip under the temperature");
  });
});

// --- 2. A line load written on a pressure card ---------------------------------------------------------------------

// One ShellQuad4 on the rectangle 3 x 1, held along x = 0, under the FacePressureLoad `load` of set 2.
const plateUnder = (load) => ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "Node;3;;3.;1.;0.", "Node;4;;0.;1.;0.",
  "ShellQuad4;1;1;1;2;3;4", "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "SinglePointConstrList;1;123456;1;4",
  load];

test("a FacePressureLoad written as a line load is refused, not applied as a pressure", () => {
  // Field 13 (surfaceOrLine) is Line: a load per unit length along the edges of the element.
  refuses(CONTROL, plateUnder("FacePressureLoad;2;1;10.;;;;;;;;;;Line"), "load-unsupported", "line load");
});

test("a FacePressureLoad written as a surface load is applied", () => {
  for (const load of ["FacePressureLoad;2;1;10.", "FacePressureLoad;2;1;10.;;;;;;;;;;Surface"])
    withModel(deckOf(CONTROL, plateUnder(load)), (model) => {
      const run = solver.run(model, "statics");
      hasNoError(run);
      assert.ok(Math.abs(displacement(run, 2, 2)) > 0, "the pressure bends the plate");
    });
});

// --- 3. The thermal expansion of a rigid element -------------------------------------------------------------------

// The rod with a third point carried by a RigidBodyStar of expansion coefficient `alpha` from its tip.
const withRigidStar = (alpha) => [...ROD, "Node;3;;4.;0.;0.", `RigidBodyStar;7;2;123456;3${alpha}`,
  "DefaultTemperatureLoad;3;20."];
const THERMAL_CONTROL = ["SubcaseStart;1", "SinglePointConstrSelection;1", "TemperatureSelection;3;TemperatureLoadPart",
  "DisplacementOutput;All"];

test("a rigid element with a thermal expansion coefficient is refused under a thermal load", () => {
  refuses(THERMAL_CONTROL, withRigidStar(";1.e-5"), "rigid-expansion-unsupported", "RigidBodyStar 7");
});

test("a rigid element without expansion coefficient follows its point under a thermal load", () => {
  withModel(deckOf(THERMAL_CONTROL, withRigidStar("")), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run, 2), 6e-4, "tip under the temperature");
    near(displacement(run, 3), 6e-4, "the point the rigid element carries");
  });
});

test("the expansion coefficient of a rigid element is not refused without a thermal load", () => {
  withModel(deckOf(CONTROL, withRigidStar(";1.e-5")), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run, 3), TIP_UNDER_WEIGHT, "the point the rigid element carries, under the weight");
  });
});

// --- 4. A non-structural mass on a property of another card ---------------------------------------------------------

test("a non-structural mass that names a property card the property is not is refused", () => {
  // Property 1 is a RodProp: a card that names BarProp 1 names no property of the model, and must not load the rod.
  refuses([...CONTROL, "NonStructMassSelection;20"], [...ROD, "NonStructMass;20;BarProp;1;2."], "nsm-unknown-target",
    "BarProp");
});

test("a non-structural mass that names the property card of the property is added", () => {
  // (rho A + nsm) g L^2 / (2 E A) with nsm = 2 per unit length: twice the tip displacement under the weight.
  withModel(deckOf([...CONTROL, "NonStructMassSelection;20"], [...ROD, "NonStructMass;20;RodProp;1;2."]), (model) => {
    const run = solver.run(model, "statics");
    hasNoError(run);
    near(displacement(run), 2 * TIP_UNDER_WEIGHT, "tip under the weight and the added mass");
  });
});

// --- 5. The system of a beam spline --------------------------------------------------------------------------------

test("a beam spline whose system is cylindrical is refused", () => {
  // The axis of a beam spline is the y axis of a rectangular system: the wing of the trim fixture, its surface spline
  // made a beam spline along the y axis of system 8, here a cylindrical one.
  const wing = readFileSync(new URL("../fixtures/trim-rbe3-wing.oos", import.meta.url), "utf8");
  const surface = "AeroSurfaceSpline1;60;100;100;103;50";
  assert.ok(wing.includes(surface), "the fixture has the surface spline 60");
  const text = wing.replace(surface, "CylindricalFrame2;8;;0.;0.;0.;0.;0.;1.;1.;0.3;0.\n" +
    "AeroBeamSpline2;60;100;100;103;50;0.01;1.2;8;-1.;-1.;Both");
  withModel(text, (model) => assert.throws(() => model.aeroTransfer().delete(), (error) => {
    assert.match(error.message, /AeroBeamSpline2 60/);
    assert.match(error.message, /rectangular/);
    return true;
  }));
});
