import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { axial, wing, column } from "../helpers/optimization-models.mjs";

const near = (actual, expected, tolerance = 2e-5) => assert.ok(Number.isFinite(actual) &&
  Math.abs(actual - expected) <= tolerance * Math.max(1e-8, Math.abs(expected)), `${actual} differs from ${expected}`);
const variable = (id, initialValue, lower, upper) => ({ id, initialValue, lower, upper });
const card = (variableId, type, id, field) => ({ variableId, target: { kind: "card", type, id, field } });
const tip = { id: 10, source: "nodal", kind: "displacements", subcase: 1, ids: [2], components: "t1" };
const barStudy = deck => solver.createStudy(oor, deck, { version: 1,
  variables: [variable(1, 2, 0.5, 5), variable(2, 10, 5, 20)],
  parameters: [card(1, "RodProp", 1, "area"), { variableId: 2, target: { kind: "card", type: "PointForceLoad", id: 2, slot: 4 } }],
  analyses: [{ analysis: "statics" }], responses: [tip], objective: { responseId: 10 },
  constraints: [{ responseId: 10, upper: 0.02 }] });

test("study capabilities expose the four solutions and preparation refuses invalid targets", () => {
  const analyses = solver.capabilities().analyses;
  assert.deepEqual(analyses.map(entry => entry.solution), [101, 103, 105, 144]);
  assert.ok(analyses.filter(entry => entry.solution !== 101).every(entry => entry.secondOrder));
  assert.equal(analyses.find(entry => entry.solution === 144).shapeFirstOrder, true);
  const deck = axial(), model = solver.modelFromReader(oor, deck);
  try {
    assert.ok(solver.capabilities(model).fields.some(field => field.targetCard === "RodProp" && field.field === "area"));
    assert.ok(oor.designCapabilities(deck).nativeCards.includes("RodProp"));
    const check = solver.checkOptimization(model, { variables: [variable(1, 2, 1, 3)],
      relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "absent",
        terms: [{ variableId: 1, coefficient: 1 }] }] });
    assert.equal(check.canPrepare, false); assert.match(check.error, /absent/);
  } finally { model.delete(); deck.delete(); }
});

test("study auxiliary static displacement becomes a batched shape direction", () => {
  const deck = axial();
  let problem, reference, moved;
  try {
    const basis = solver.shapeBasisFromReader(oor, deck, { variableId: 7, cases: [{ subcase: 1 }], gridIds: [2] });
    assert.deepEqual(Array.from(basis.gridIds), [2]);
    near(basis.directions[0], 0.015);
    near(basis.directions[1], 0);
    near(basis.directions[2], 0);
    problem = solver.optimizationFromReader(oor, deck, {
      variables: [variable(7, 0, 0, 1)], shapeBatches: [basis],
      responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2] }],
      objective: { responseId: 10, subcase: 1 },
    });
    reference = problem.evaluate([0]); moved = problem.evaluate([1]);
    near(reference.values().objective, 0.015);
    near(moved.values().objective, 0.015075);
    near(Number(deck.cardById("Node", 2).fields[2]), 3);
  } finally { moved?.delete(); reference?.delete(); problem?.delete(); deck.delete(); }
});

test("study drives load and property fields and returns bounded derivatives and products", () => {
  const deck = axial();
  const study = barStudy(deck);
  const point = study.evaluate();
  try {
    near(point.values().objective, 0.015);
    const gradient = point.gradients();
    near(gradient.values[0], -0.0075); near(gradient.values[1], 0.0015);
    near(point.jvp([1, 2]).values[0], -0.0045);
    const weights = point.lagrangianWeights();
    const product = point.vjp(weights); near(product.values[0], -0.0075); near(product.values[1], 0.0015);
    const directions = point.jvp([1, 0, 0, 1], { count: 2 });
    near(directions.values[0], -0.0075); near(directions.values[1], 0.0015);
    const transpose = point.vjp([...weights, ...weights], { rows: 2 });
    near(transpose.values[0], -0.0075); near(transpose.values[3], 0.0015);
    assert.equal(deck.cardById("RodProp", 1).values.area, 2);
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study discrete case selections and solution targets update independent trials", () => {
  const deck = axial();
  deck.addCard({ type: "PointForceLoad", fields: [3, 2, null, 20, 1, 0, 0] });
  deck.setSolution(103);
  const study = solver.createStudy(oor, deck, { version: 1,
    variables: [{ ...variable(1, 2, 2, 3), kind: "choice", choices: [2, 3] }],
    parameters: [{ target: { kind: "solution" }, constant: 101 },
      { variableId: 1, target: { kind: "case", command: "StaticLoadSelection", subcase: 1 } }],
    analyses: [{ analysis: "statics" }], responses: [tip], objective: { responseId: 10 } });
  const first = study.evaluate([2]), second = study.evaluate([3]);
  try {
    near(first.values().objective, 0.015); near(second.values().objective, 0.03);
    assert.throws(() => second.gradients(), /discrete/);
  } finally { first.delete(); second.delete(); study.delete(); deck.delete(); }
});

test("study Hessian-vector products include mixed load and property terms", () => {
  const deck = axial(), study = barStudy(deck), point = study.evaluate();
  try {
    const hessian = point.hessianVector(point.lagrangianWeights(), [1, 0]);
    near(hessian.values[0], 0.0075, 1e-3); near(hessian.values[1], -0.00075, 1e-3);
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study named parameter and response callbacks differentiate application metrics", () => {
  const deck = axial();
  const definition = { version: 1, variables: [variable(1, 1, 0.5, 2), variable(2, 10, 5, 20)],
    parameters: [{ callback: "twiceArea", target: { kind: "card", type: "RodProp", id: 1, field: "area" } },
      { variableId: 2, target: { kind: "card", type: "PointForceLoad", id: 2, slot: 4 } }],
    analyses: [{ analysis: "statics" }], responses: [{ id: 1, source: "callback", callback: "squareTip" }],
    objective: { responseId: 1 } };
  const hooks = { parameters: { twiceArea: ({ variables }) => 2 * variables[1] },
    responses: { squareTip: ({ runs }) => {
      const table = runs[0].nodalTables.find(table => table.kind === "displacements");
      const row = table.ids.indexOf(2);
      return table.values[row * 6] ** 2;
    } } };
  const study = solver.createStudy(oor, deck, definition, {}, hooks), point = study.evaluate();
  try {
    near(point.values().objective, 0.015 ** 2);
    const gradient = point.gradients();
    near(gradient.values[0], -2 * 0.015 ** 2, 5e-3);
    near(gradient.values[1], 2 * 0.015 ** 2 / 10, 5e-3);
    point.save("/work/callback-study.ooh5");
    const archive = oor.openResults("/work/callback-study.ooh5"), model = archive.model();
    try {
      assert.throws(() => solver.restoreStudy(oor, model), /missing .* callback/);
      const restored = solver.restoreStudy(oor, model, hooks), again = restored.study.evaluate();
      try { near(again.values().objective, point.values().objective); }
      finally { again.delete(); restored.study.delete(); }
    } finally { model.delete(); archive.delete(); }
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study archives restore the accepted model point and optimizer state", () => {
  const deck = axial(), study = barStudy(deck);
  try {
    const accepted = study.evaluate([3, 12]);
    try {
      near(accepted.values().objective, 0.012);
      accepted.save("/work/accepted-study.ooh5", { state: { iteration: 8 } });
      const results = oor.openResults("/work/accepted-study.ooh5"), model = results.model();
      try {
        assert.equal(results.tables()[0].analysis, "static");
        const saved = results.readNamed(results.named().findIndex(result => result.kind === "responseValues"));
        near(saved.arrays.value.values[0], 0.012);
        assert.equal(saved.arrays.designCycle.values[0], 8);
        assert.equal(saved.arrays.isObjective.values[0], 1);
        assert.deepEqual(saved.texts.responseType, ["nodal"]);
        const restored = solver.restoreStudy(oor, model);
        assert.equal(restored.state.iteration, 8);
        const again = restored.study.evaluate();
        try { near(again.values().objective, 0.012); } finally { again.delete(); restored.study.delete(); }
      } finally { model.delete(); results.delete(); }
    } finally { accepted.delete(); }
  } finally { study.delete(); deck.delete(); }
});

test("study archive keeps each constraint set for a shared response item", () => {
  const deck = axial();
  const study = solver.createStudy(oor, deck, { version: 1, variables: [variable(1, 2, 1, 3)],
    parameters: [card(1, "RodProp", 1, "area")], analyses: [{ analysis: "statics" }],
    responses: [tip], objective: { responseId: 10 },
    constraints: [{ responseId: 10, lower: 0.01, upper: 0.02 }, { responseId: 10, upper: 0.03 }] });
  const point = study.evaluate();
  try {
    point.save("/work/multi-constraint-study.ooh5");
    const archive = oor.openResults("/work/multi-constraint-study.ooh5");
    try {
      const saved = archive.readNamed(archive.named().findIndex(result => result.kind === "responseValues"));
      assert.deepEqual(Array.from(saved.arrays.value.values), [0.015, 0.015]);
      assert.deepEqual(Array.from(saved.arrays.constraintSet.values), [1, 2]);
      assert.deepEqual(Array.from(saved.arrays.upper.values), [0.02, 0.03]);
      near(saved.arrays.lower.values[0], 0.01);
      assert.ok(Number.isNaN(saved.arrays.lower.values[1]));
    } finally { archive.delete(); }
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study modal vector and stress responses preserve mode identity during differentiation", () => {
  const deck = axial();
  deck.caseControl.section(deck.caseControl.sections().find(section => section.kind === "subcase").key).set("EigenMethodSelection", 3);
  const study = solver.createStudy(oor, deck, { version: 1, variables: [variable(1, 1000, 500, 2000)],
    parameters: [card(1, "IsoMat", 1, "youngsModulus")], analyses: [{ analysis: "modes", subcases: [1] }],
    responses: [
      { id: 1, source: "eigenvalue", subcase: 1, mode: 1, tracksMode: true, components: "cycles" },
      { id: 2, source: "nodal", kind: "eigenvectors", subcase: 1, mode: 1, tracksMode: true, ids: [2], components: "t1" },
      { id: 3, source: "element", kind: "stress", subcase: 1, mode: 1, tracksMode: true, components: "axial" },
    ] });
  const point = study.evaluate();
  try {
    const values = point.values(), gradient = point.gradients();
    near(gradient.values[0], values.responses[0].values[0] / 2000);
    near(gradient.values[1], 0, 1e-3);
    near(gradient.values[2], values.responses[2].values[0] / 1000);
    assert.ok(values.trackedModes.every(mode => mode.mac > 0.99 && !mode.isUncertain));
    point.save("/work/modal-study.ooh5");
    const archive = oor.openResults("/work/modal-study.ooh5"), accepted = archive.model();
    try {
      assert.ok(archive.tables().some(table => table.analysis === "modes"));
      const restored = solver.restoreStudy(oor, accepted);
      try {
        assert.equal(restored.diagnostics.length, 1);
        assert.ok(restored.trackedModes.every(mode => mode.mac > 0.99 && !mode.isUncertain));
        assert.ok(restored.trackedModes.every(mode => mode.shape === undefined));
      } finally { restored.study.delete(); }
    } finally { accepted.delete(); archive.delete(); }
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study buckling sensitivities include the preload and match full-analysis differences", () => {
  const deck = column(); const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: [variable(1, 0.01, 0.005, 0.015)],
    relations: [{ id: 1, targetKind: "property", targetCard: "BarProp", targetId: 1,
      field: "bendingInertia1", terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 1, type: "BucklingFactor", attributeA: 1, tracksMode: true, subcases: [2] }],
    objective: { responseId: 1, subcase: 2 },
  }, { analyses: { buckling: [2] } });
  const point = problem.evaluate();
  try {
    assert.ok(point.values().objective > 0);
    near(point.gradients().objective[0], point.gradients({ method: "finiteDifference" }).objective[0], 5e-3);
    const step = 1e-5;
    const plus = problem.evaluate([0.01 + step], point.values().trackedModes);
    const minus = problem.evaluate([0.01 - step], point.values().trackedModes);
    try {
      const expected = (plus.gradients().responses[0].values[0] - minus.gradients().responses[0].values[0]) / (2 * step);
      assert.ok(Math.abs(point.hessianVector({ rows: 1, responses: { 1: [1] } }, [1]).values[0] - expected) < 1e-6);
    } finally { plus.delete(); minus.delete(); }
    assert.equal(point.analysisResults().eigenvalues[0].analysis, "buckling");
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("study SOL 105 archive preserves buckling eigenvalues and the accepted model", () => {
  const deck = column();
  const study = solver.createStudy(oor, deck, { version: 1,
    variables: [variable(1, 0.012, 0.005, 0.02)],
    parameters: [card(1, "BarProp", 1, "bendingInertia1")],
    analyses: [{ analysis: "buckling", subcases: [2] }],
    responses: [{ id: 1, source: "eigenvalue", subcase: 2, mode: 1, components: "eigenvalue" }],
    objective: { responseId: 1 } });
  const point = study.evaluate();
  try {
    point.save("/work/buckling-study.ooh5", { state: { cycle: 3 } });
    const archive = oor.openResults("/work/buckling-study.ooh5"), accepted = archive.model();
    try {
      assert.ok(archive.tables().some(table => table.analysis === "buckling"));
      assert.ok(archive.tables().some(table => table.kind === "eigenvalues"));
      const restored = solver.restoreStudy(oor, accepted);
      const again = restored.study.evaluate();
      try { assert.equal(restored.state.cycle, 3); near(again.values().objective, point.values().objective); }
      finally { again.delete(); restored.study.delete(); }
    } finally { accepted.delete(); archive.delete(); }
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("study SOL 144 imports trim links and exports complete typed aerodynamic archives", () => {
  const deck = wing({ link: true }); const model = solver.modelFromReader(oor, deck);
  try {
    assert.equal(solver.checkSupport(model, "aeroelastic").isComplete, true);
    const run = solver.run(model, "aeroelastic"), trim = run.aeroelastic.subcases[0];
    near(trim.variables.find(variable => variable.label === "SideSlip").value, 0.01);
    assert.equal(trim.pressures.length, 4);
    const study = solver.createStudy(oor, deck, { version: 1, variables: [variable(1, 10, 5, 20)],
      parameters: [card(1, "AeroTrim", 5, "dynamicPressure")], analyses: [{ analysis: "aeroelastic" }],
      responses: [{ id: 1, source: "pressure", subcase: 1, reduce: "sum" },
        { id: 2, source: "stabilityDerivative", subcase: 1, variable: "AngleOfAttack", components: "cz" }] });
    const point = study.evaluate();
    try {
      assert.ok(point.gradients().values.every(Number.isFinite));
      point.save("/work/trim-study.ooh5");
      const archive = oor.openResults("/work/trim-study.ooh5");
      try {
        assert.ok(archive.named().some(result => result.kind === "stabilityDerivatives"));
        const index = archive.named().findIndex(result => result.kind === "aeroPressures");
        assert.deepEqual(archive.readNamed(index).arrays.pressureCoefficient.values, trim.pressures);
      } finally { archive.delete(); }
    } finally { point.delete(); study.delete(); }
  } finally { model.delete(); deck.delete(); }
});

test("study SOL 144 sizing and shape gradients have explicit native and numerical paths", () => {
  const deck = wing(), model = solver.modelFromReader(oor, deck);
  const declarations = { variables: [variable(1, 0.1, 0.05, 0.2)],
    relations: [{ id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 1, type: "Displacement", attributeA: 3, targets: [3], subcases: [1] }] };
  const problem = solver.createOptimization(model, declarations, { analyses: { aeroelastic: [1] } });
  const point = problem.evaluate();
  try {
    near(point.gradients().responses[0].values[0], point.gradients({ method: "finiteDifference" }).responses[0].values[0], 5e-3);
    const step = 1e-4;
    const plus = problem.evaluate([0.1 + step]);
    const minus = problem.evaluate([0.1 - step]);
    try {
      const expected = (plus.gradients().responses[0].values[0] - minus.gradients().responses[0].values[0]) / (2 * step);
      near(point.hessianVector({ rows: 1, responses: { 1: [1] } }, [1]).values[0], expected, 5e-3);
    } finally { plus.delete(); minus.delete(); }
  }
  finally { point.delete(); problem.delete(); }
  const shaped = solver.createOptimization(model, { ...declarations, variables: [...declarations.variables, variable(2, 0, -0.05, 0.05)],
    shapes: [3, 4].map(gridId => ({ variableId: 2, gridId, direction: [0, 0, 1] })) }, { analyses: { aeroelastic: [1] } });
  const moved = shaped.evaluate();
  try {
    const native = moved.gradients();
    const numerical = moved.gradients({ method: "finiteDifference" });
    native.responses[0].values.forEach((value, i) => near(value, numerical.responses[0].values[i], 5e-3));
    assert.equal(native.report.analysisPerturbations, 2);
    const shapeCurvature = moved.hessianVector({ rows: 1, responses: { 1: [1] } }, [0, 1]);
    assert.ok(shapeCurvature.values.every(Number.isFinite));
    assert.equal(shapeCurvature.report.perturbedLinearisations, 2);
  } finally { moved.delete(); shaped.delete(); model.delete(); deck.delete(); }
});

function trimShapeProblem(model, maximumOutputBytes) {
  const settings = { analyses: { aeroelastic: [1] } };
  if (maximumOutputBytes !== undefined) settings.maximumOutputBytes = maximumOutputBytes;
  return solver.createOptimization(model, {
    variables: [variable(1, 0.1, 0.05, 0.2), variable(2, 0, -0.05, 0.05)],
    relations: [{ id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    shapes: [3, 4].map(gridId => ({ variableId: 2, gridId, direction: [0, 0, 1] })),
    responses: [{ id: 1, type: "Displacement", attributeA: 3, targets: [3], subcases: [1] }],
  }, settings);
}

test("trim shape columns agree across gradients, products and streaming", () => {
  const deck = wing(), model = solver.modelFromReader(oor, deck);
  const problem = trimShapeProblem(model);
  const point = problem.evaluate();
  try {
    const gradient = point.gradients().responses[0].values;
    near(point.jvp([0, 1]).values[0], gradient[1], 5e-3);
    const transpose = point.vjp({ responses: { 1: [1] } }).values;
    transpose.forEach((value, column) => near(value, gradient[column], 5e-3));
    const streamed = new Float64Array(gradient.length);
    point.streamGradients(block => {
      for (let row = 0; row < block.rowCount; ++row)
        streamed.set(block.values.subarray(row * block.columnCount, (row + 1) * block.columnCount),
          (block.rowBegin + row) * gradient.length + block.columnBegin);
    }, 1);
    streamed.forEach((value, column) => near(value, gradient[column], 5e-3));
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("trim shape gradients stream beyond the dense output limit", () => {
  const deck = wing(), model = solver.modelFromReader(oor, deck);
  const problem = trimShapeProblem(model, 8);
  const point = problem.evaluate();
  try {
    assert.throws(() => point.gradients(), /maximumOutputBytes/);
    const values = new Float64Array(2);
    point.streamGradients(block => values.set(block.values, block.columnBegin), 1);
    assert.ok(values.every(Number.isFinite));
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("trim shape derivatives use a feasible one-sided step at a bound", () => {
  const deck = wing(), model = solver.modelFromReader(oor, deck);
  const problem = trimShapeProblem(model);
  const point = problem.evaluate([0.1, 0.05]);
  try {
    const gradient = point.gradients();
    assert.equal(gradient.report.analysisPerturbations, 1);
    const step = 1e-5;
    const inside = problem.evaluate([0.1, 0.05 - step]);
    try {
      const expected = (point.values().responses[0].values[0] - inside.values().responses[0].values[0]) / step;
      near(gradient.responses[0].values[1], expected, 3e-3);
    } finally { inside.delete(); }
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("native trim sizing differentiates a fixed applied point load", () => {
  const deck = wing({ load: true }), model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: [variable(1, 0.1, 0.05, 0.2)],
    relations: [{ id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 1, type: "Displacement", attributeA: 3, targets: [3], subcases: [1] }],
  }, { analyses: { aeroelastic: [1] } });
  const point = problem.evaluate();
  try {
    near(point.gradients().responses[0].values[0], point.gradients({ method: "finiteDifference" }).responses[0].values[0], 5e-3);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("native trim sizing differentiates applied inertial loads", () => {
  const deck = wing({ load: true });
  deck.addCard({ type: "GravityLoad", fields: [20, null, 10, 0, 0, 1] });
  const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: [variable(1, 1, 0.5, 2)],
    relations: [{ id: 1, targetKind: "material", targetCard: "IsoMat", targetId: 1, field: "density",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 1, type: "Displacement", attributeA: 3, targets: [3], subcases: [1] }],
  }, { analyses: { aeroelastic: [1] } });
  const point = problem.evaluate();
  try {
    const gradient = point.gradients().responses[0].values[0];
    assert.ok(Math.abs(gradient) > 1e-7);
    near(gradient, point.gradients({ method: "finiteDifference" }).responses[0].values[0], 5e-3);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("study native material derivatives recompute an implicit shear modulus", () => {
  const source = axial();
  const text = source.toModelText().replace("RodProp;1;1;2.", "RodProp;1;1;2.;0.1")
    .replace("PointForceLoad", "PointMomentLoad").replace("SinglePointConstrList;1;23456;2", "SinglePointConstrList;1;12356;2");
  source.delete(); oor.writeFile("/work/study-torsion.oos", text);
  const deck = oor.readModel("/work/study-torsion.oos", { strict: true }), model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, { variables: [variable(1, 1000, 500, 2000), variable(2, 0.3, 0.1, 0.45)],
    relations: ["youngsModulus", "poissonRatio"].map((field, i) => ({ id: i + 1, targetKind: "material", targetCard: "IsoMat", targetId: 1,
      field, terms: [{ variableId: i + 1, coefficient: 1 }] })),
    responses: [{ id: 1, type: "Displacement", attributeA: 4, targets: [2], subcases: [1] }] });
  const point = problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 0.78);
    const gradient = point.gradients().responses[0].values;
    near(gradient[0], -0.00078, 1e-3); near(gradient[1], 0.6, 1e-3);
    const numerical = point.gradients({ method: "finiteDifference" }).responses[0].values;
    gradient.forEach((value, i) => near(value, numerical[i], 1e-3));
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("study query responses expose model mass and its design derivative without a callback", () => {
  const deck = axial();
  const study = solver.createStudy(oor, deck, { version: 1, variables: [variable(1, 2, 1, 3)],
    parameters: [card(1, "RodProp", 1, "area")], analyses: [{ analysis: "statics" }],
    responses: [{ id: 1, source: "query", query: "massByProperty", column: "mass", idColumn: "propId", reduce: "sum" }],
    objective: { responseId: 1 } });
  const point = study.evaluate();
  try { near(point.values().objective, 6); near(point.gradients().values[0], 3); }
  finally { point.delete(); study.delete(); deck.delete(); }
});

test("study result queries consume generated tables with matching model geometry", () => {
  const deck = axial();
  const study = solver.createStudy(oor, deck, { version: 1, variables: [variable(1, 10, 5, 20)],
    parameters: [{ variableId: 1, target: { kind: "card", type: "PointForceLoad", id: 2, slot: 4 } }],
    analyses: [{ analysis: "statics" }],
    responses: [{ id: 1, source: "query", query: "maxDisplacement", column: "value", idColumn: "nodeId",
      reduce: "max" }], objective: { responseId: 1 } });
  const point = study.evaluate();
  try { near(point.values().objective, 0.015); near(point.gradients().values[0], 0.0015); }
  finally { point.delete(); study.delete(); deck.delete(); }
});
