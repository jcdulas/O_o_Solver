// Independent elastic-rod checks: u_tip = (rho*A + nsm)*g*L^2/(2*E*A).
import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

const BULK = `Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Rod2;1;1;1;2
RodProp;1;1;2.
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
SinglePointConstrList;1;23456;2
GravityLoad;2;;10.;1.;0.;0.
`;
const CASE = `SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
`;
function deckOf(control = CASE, bulk = BULK) {
  oor.writeFile("/work/static-inputs.oos", `[executive]\nSolutionSequence;101\n[case]\n${control}[model]\n${bulk}`);
  return oor.readModel("/work/static-inputs.oos", { strict: true });
}
function withModel(control, bulk, action) {
  const deck = deckOf(control, bulk);
  let model;
  try { model = solver.modelFromReader(oor, deck); return action(model, deck); }
  finally { model?.delete(); deck.delete(); }
}
function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} differs from ${expected}`);
}
function tip(run, subcase = 1) {
  const table = run.nodalTables.find(t => t.kind === "displacements" && t.subcase === subcase);
  assert.ok(table, `displacements of subcase ${subcase}`);
  return table.values[Array.from(table.ids).indexOf(2) * 6];
}

test("support includes source material and solution cards omitted from analysis queries", () => {
  const bulk = BULK + "CreepMat;1;;;CreepLaw;;;;;112;1.e-9;2.;1.e-4;1.;1.e-9;0.01\n" +
    "NonlinearStaticParams;10;2;0.\n";
  withModel(CASE + "NonlinearStaticParamsSelection;10\n", bulk, model => {
    const report = model.checkSupport("statics");
    assert.equal(report.isComplete, false);
    for (const card of ["CreepMat", "NonlinearStaticParams"])
      assert.ok(report.refusals.some(d => d.code === "unsupported-card" && d.card === card), card);
    assert.ok(report.refusals.some(d => d.code === "unsupported-selection" && d.field === "nonlinearStaticParamsSelection"));
    assert.throws(() => solver.run(model, "statics"), /CreepMat|NonlinearStaticParams/);
  });
});

test("an inherited unsupported selection is reported even without a matching bulk card", () => {
  withModel("NonlinearStaticParamsSelection;10\n" + CASE, BULK, model => {
    const report = model.checkSupport("statics");
    assert.equal(report.isComplete, false);
    assert.ok(report.refusals.some(d => d.field === "nonlinearStaticParamsSelection" && d.id === 1));
    assert.throws(() => solver.run(model, "statics"), /nonlinearStaticParamsSelection/);
  });
});

test("source refusals survive optimization overlays", () => {
  withModel(CASE, BULK + "CreepMat;1;;;CreepLaw;;;;;112;1.e-9;2.;1.e-4;1.;1.e-9;0.01\n", model => {
    const problem = solver.createOptimization(model, {
      variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
      relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
        terms: [{ variableId: 1, coefficient: 1 }] }],
      responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2] }],
      objective: { responseId: 10, subcase: 1 },
    });
    try { assert.throws(() => problem.evaluate(new Float64Array([2.5])), /CreepMat/); }
    finally { problem.delete(); }
  });
});

test("a nonlinear analysis kind cannot silently select a linear calculation", () => {
  withModel(CASE + "AnalysisKind;NonlinearStatics\n", BULK, model => {
    assert.ok(model.checkSupport("statics").refusals.some(d => d.code === "unsupported-analysis"));
    assert.throws(() => solver.run(model, "statics"), /NonlinearStatics/);
  });
});

test("an unselected mass set adds nothing; selection changes inertial loads", () => {
  const bulk = BULK + "NonStructMass;20;RodProp;1;2.\n";
  for (const selected of [false, true])
    withModel(CASE + (selected ? "NonStructMassSelection;20\n" : ""), bulk, model => {
      assert.equal(model.checkSupport("statics").isComplete, true);
      const run = solver.run(model, "statics");
      near(tip(run), selected ? 0.09 : 0.045);
      assert.ok(!run.diagnostics.some(d => d.code === "nsm-not-selected"));
    });
});

test("global mass selection is inherited and a subcase override has its own mass", () => {
  const control = `SinglePointConstrSelection;1
StaticLoadSelection;2
DisplacementOutput;All
NonStructMassSelection;20
SubcaseStart;1
SubcaseStart;2
NonStructMassSelection;21
SubcaseStart;3
`;
  withModel(control, BULK + "NonStructMass;20;RodProp;1;2.\nNonStructMass;21;RodProp;1;4.\n", model => {
    const run = solver.run(model, "statics");
    near(tip(run, 1), 0.09);
    near(tip(run, 2), 0.135);
    near(tip(run, 3), 0.09);
  });
});

test("mass unions and element-target lists add their contributions once", () => {
  const bulk = BULK + "NonStructMass;20;RodProp;1;2.\nNonStructMass1;21;Element;1.;1\n" +
    "NonStructMassUnion;30;20;21\n";
  withModel(CASE + "NonStructMassSelection;30\n", bulk, model => near(tip(solver.run(model, "statics")), 0.1125));
});

test("selected mass is additional to property mass and observes WeightToMass", () => {
  const bulk = BULK.replace("RodProp;1;1;2.", "RodProp;1;1;2.;;;1.") +
    "NonStructMass;20;RodProp;1;2.\nParameter;WeightToMass;0.5\n";
  withModel(CASE + "NonStructMassSelection;20\n", bulk, model => near(tip(solver.run(model, "statics")), 0.05625));
});

test("shell mass selection uses mass per area", () => {
  const shell = `Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Node;3;;3.;1.;0.
Node;4;;0.;1.;0.
ShellQuad4;1;1;1;2;3;4
ShellProp;1;1;0.1;1;;1
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1;4
SinglePointConstrList;1;23456;2;3
GravityLoad;2;;10.;1.;0.;0.
NonStructMass;20;ShellProp;1;0.1
`;
  const base = withModel(CASE, shell, model => tip(solver.run(model, "statics")));
  assert.ok(base > 0);
  withModel(CASE + "NonStructMassSelection;20\n", shell, model => near(tip(solver.run(model, "statics")), 2 * base));
});

test("unknown, incompatible and undistributed mass selections fail before solving", () => {
  const cases = [
    ["", /nsm-unknown-set/],
    ["NonStructMass;20;RodProp;999;2.\n", /nsm-unknown-target/],
    ["NonStructMass;20;ShellProp;1;2.\n", /nsm-kind-mismatch/],
    ["NonStructMassLumped;20;RodProp;1;6.\n", /nsm-lumped-unsupported/],
  ];
  for (const [extra, message] of cases)
    withModel(CASE + "NonStructMassSelection;20\n", BULK + extra, model => {
      assert.throws(() => solver.run(model, "statics"), message);
    });
});

test("an invalid selected set is checked even without an inertial load", () => {
  withModel(CASE + "NonStructMassSelection;20\n", BULK.replace("GravityLoad;2;;10.;1.;0.;0.",
    "PointForceLoad;2;2;;10.;1.;0.;0."), model => {
    assert.throws(() => solver.run(model, "statics"), /nsm-unknown-set/);
  });
});

// A deck of another solution sequence than the statics of deckOf.
function withDeck(solution, control, bulk, action) {
  oor.writeFile("/work/static-inputs-eigen.oos", `[executive]\nSolutionSequence;${solution}\n[case]\n${control}[model]\n${bulk}`);
  const deck = oor.readModel("/work/static-inputs-eigen.oos", { strict: true });
  let model;
  try { model = solver.modelFromReader(oor, deck); return action(model, deck); }
  finally { model?.delete(); deck.delete(); }
}
const eigenvalueOf = (run, subcase) => run.eigenvalues.find(t => t.subcase === subcase).rows[0].eigenvalue;
const relativelyNear = (actual, expected, what) =>
  assert.ok(Math.abs(actual / expected - 1) < 1e-10, `${what}: ${actual} differs from ${expected}`);

test("selected mass supports static optimization, and the trim analysis refuses it", () => {
  withModel(CASE + "NonStructMassSelection;20\n", BULK + "NonStructMass;20;RodProp;1;2.\n", model => {
    for (const analysis of ["statics", "modes", "buckling"])
      assert.ok(!model.checkSupport(analysis).refusals.some(d => d.field === "nonStructMass"), analysis);
    assert.ok(model.checkSupport("aeroelastic").refusals.some(d => d.field === "nonStructMass"));
    const problem = solver.createOptimization(model, {
      variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
      relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
        terms: [{ variableId: 1, coefficient: 1 }] }],
      responses: [{ id: 10, type: "Volume" }], objective: { responseId: 10 },
    });
    try { assert.equal(problem.metadata().variables.length, 1); }
    finally { problem.delete(); }
  });
});

// The axial mode of the rod: omega^2 = k / m with m the mass at its free end, in proportion to rho*A + nsm. The
// structural mass per length is 2 (rho = 1, A = 2): a selected set of 2 per length doubles the mass, one of 4
// triples it.
const MODES_CASE = "SinglePointConstrSelection;1\nEigenMethodSelection;5\n";
const MODES_BULK = BULK + "LanczosEigenMethod;5;;;1\nNonStructMass;20;RodProp;1;2.\nNonStructMass;21;RodProp;1;4.\n";

test("a selected mass set lowers the frequencies of the real eigenvalue analysis by its mass", () => {
  const control = MODES_CASE + "SubcaseStart;1\nSubcaseStart;2\nNonStructMassSelection;20\n" +
    "SubcaseStart;3\nNonStructMassSelection;21\nSubcaseStart;4\nNonStructMassSelection;20\n";
  withDeck(103, control, MODES_BULK, model => {
    assert.equal(model.checkSupport("modes").isComplete, true);
    const run = solver.run(model, "modes");
    const bare = eigenvalueOf(run, 1);
    assert.ok(bare > 0);
    relativelyNear(eigenvalueOf(run, 2), bare / 2, "the set of 2 per length");
    relativelyNear(eigenvalueOf(run, 3), bare / 3, "the set of 4 per length");
    relativelyNear(eigenvalueOf(run, 4), bare / 2, "the same set in another subcase");
  });
});

test("the mass of a selected set is the mass of the property field in the eigenvalue analysis", () => {
  const selected = withDeck(103, MODES_CASE + "NonStructMassSelection;20\nSubcaseStart;1\n", MODES_BULK,
    model => eigenvalueOf(solver.run(model, "modes"), 1));
  const inProperty = withDeck(103, MODES_CASE + "SubcaseStart;1\n",
    MODES_BULK.replace("RodProp;1;1;2.", "RodProp;1;1;2.;;;2."), model => eigenvalueOf(solver.run(model, "modes"), 1));
  relativelyNear(selected, inProperty, "selected set against the property mass");
});

test("a selected mass set per area lowers the frequencies of a shell by its mass", () => {
  const shell = `Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Node;3;;3.;1.;0.
Node;4;;0.;1.;0.
ShellQuad4;1;1;1;2;3;4
ShellProp;1;1;0.1;1;;1
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1;4
SinglePointConstrList;1;23456;2;3
LanczosEigenMethod;5;;;1
NonStructMass;20;ShellProp;1;0.1
`;
  const control = MODES_CASE + "SubcaseStart;1\nSubcaseStart;2\nNonStructMassSelection;20\n";
  withDeck(103, control, shell, model => {
    const run = solver.run(model, "modes");
    relativelyNear(eigenvalueOf(run, 2), eigenvalueOf(run, 1) / 2, "the structural mass per area doubled");
  });
});

test("a selected mass set weighs in the preload of a buckling analysis", () => {
  // A column clamped at its foot under its own weight along its axis: the selected set doubles the weight, so the
  // load factor of the first buckling mode is half that of the column alone.
  const column = `Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Bar2;1;1;1;2;0.;;1.;0.
BarProp;1;1;2.;0.1;0.1;0.05
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
GravityLoad;2;;10.;-1.;0.;0.
LanczosEigenMethod;5;;;1
NonStructMass;20;BarProp;1;2.
`;
  const control = (selection) => `SubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n${selection}` +
    `SubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;5\nStaticSubcaseSelection;1\n${selection}`;
  const factorOf = (selection) => withDeck(105, control(selection), column, model => {
    assert.equal(model.checkSupport("buckling").isComplete, true);
    return eigenvalueOf(solver.run(model, "buckling"), 2);
  });
  const alone = factorOf("");
  assert.ok(alone > 0, `load factor ${alone}`);
  relativelyNear(factorOf("NonStructMassSelection;20\n"), alone / 2, "the weight doubled");
});

test("an eigenvalue analysis checks its selected mass set, and refuses it where it would be left out", () => {
  // an unknown set stops the analysis
  withDeck(103, MODES_CASE + "NonStructMassSelection;99\nSubcaseStart;1\n", MODES_BULK, model => {
    assert.throws(() => solver.run(model, "modes"), /nsm-unknown-set/);
  });
  // the modes of a design problem are differentiated without the set: refused
  withDeck(103, MODES_CASE + "NonStructMassSelection;20\nSubcaseStart;1\n", MODES_BULK, model => {
    const problem = solver.createOptimization(model, {
      variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
      relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
        terms: [{ variableId: 1, coefficient: 1 }] }],
      responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }], objective: { responseId: 10, subcase: 1 },
    }, { analyses: { modes: [1] } });
    try { assert.throws(() => problem.evaluate(new Float64Array([2.5])), /NonStructMassSelection/); }
    finally { problem.delete(); }
  });
});

test("the model's ModelParams OffsetDefinition says where the mass of an offset element is", () => {
  // A Bar2 along x offset by 0.5 along z, clamped at one end, under a gravity along x. With the mass on the offset
  // (LoadRecoveryOffsets) the weight acts on the axis: the bar stretches, u = ρ g L² / (2 E). With the mass at the
  // grid points (ElemOffsetsOnly, the default) the weight acts 0.5 from the axis and bends the bar too.
  const bulk = (value) => (value ? `ModelParams;OffsetDefinition;${value}\n` : "") +
    "Node;1;;0.;0.;0.\nNode;2;;3.;0.;0.\nBar2;1;1;1;2;0.;;1.;0.;;;;0.;0.;0.5;0.;0.;0.5\n" +
    "BarProp;1;1;2.;0.1;0.1;0.05\nIsoMat;1;1000.;;0.3;1.\nSinglePointConstrList;1;123456;1\n" +
    "GravityLoad;2;;10.;1.;0.;0.\n";
  const tipOf = (value) => {
    let values = null;
    withModel(CASE, bulk(value), model => {
      const table = solver.run(model, "statics").nodalTables.find(t => t.kind === "displacements");
      values = Array.from(table.values.slice(6, 12));
    });
    return values;
  };
  const onOffsets = tipOf("LoadRecoveryOffsets");
  near(onOffsets[0], 0.045);
  assert.ok(Math.abs(onOffsets[2]) < 1e-12, `deflection ${onOffsets[2]} with the mass on the offset`);
  const atGridPoints = tipOf("ElemOffsetsOnly");
  assert.ok(Math.abs(atGridPoints[2]) > 0.1, `deflection ${atGridPoints[2]} with the mass at the grid points`);
  assert.deepEqual(tipOf(null), atGridPoints);
  // A value O_o_Solver does not take is refused, not ignored.
  withModel(CASE, bulk("NoMassOffsets"), model => {
    assert.ok(model.checkSupport("statics").refusals.some(d => d.code === "unsupported-parameter"));
    assert.throws(() => solver.run(model, "statics"), /OffsetDefinition/);
  });
});
