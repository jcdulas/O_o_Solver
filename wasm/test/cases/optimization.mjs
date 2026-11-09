// Original axial bar: u = P*L/(E*A). Every derivative below follows from this scalar expression.
import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

const MODEL = `[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
[model]
Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Rod2;1;1;1;2
RodProp;1;1;2.
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
SinglePointConstrList;1;23456;2
PointForceLoad;2;2;;10.;1.;0.;0.
DesignVar;1;AREA;2.;0.5;5.
DesignVar;2;MODULUS;1000.;500.;2000.
DesignPropRelation1;1;RodProp;1;area;;;;1;1.
DesignMatRelation1;2;IsoMat;1;youngsModulus;;;;2;1.
`;
const declarations = {
  responses: [
    { id: 10, label: "tip", type: "Displacement", attributeA: 1, targets: [2] },
    { id: 11, label: "volume", type: "Volume" },
  ],
  constraints: [{ responseId: 10, lower: 0.005, upper: 0.02 }],
  objective: { responseId: 11 },
};
function near(actual, expected, relative = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= relative * Math.max(Math.abs(expected), 1e-10),
    `${actual} differs from ${expected}`);
}
function vectorNear(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => near(value, expected[i], tolerance));
}
function makeDeck(name = "optimization") {
  oor.writeFile(`/work/${name}.oos`, MODEL);
  return oor.readModel(`/work/${name}.oos`, { strict: true });
}

test("JavaScript drives first and second derivatives through independent WebAssembly memories", () => {
  const deck = makeDeck();
  const problem = solver.optimizationFromReader(oor, deck, declarations);
  const point = problem.evaluate();
  try {
    vectorNear(problem.metadata().initialValues, [2, 1000]);
    const values = point.values();
    near(values.responses.find(r => r.id === 10).values[0], 0.015);
    near(values.objective, 6);
    assert.equal(values.constraints.length, 2);
    const arrays = point.analysisResults();
    const displacements = arrays.nodalTables.find(t => t.kind === "displacements");
    near(displacements.values[Array.from(displacements.ids).indexOf(2) * 6], 0.015);
    const direct = point.gradients({ method: "direct" });
    const adjoint = point.gradients({ method: "adjoint" });
    const expectedGradient = [-0.0075, -0.000015];
    vectorNear(direct.responses.find(r => r.id === 10).values, expectedGradient);
    vectorNear(adjoint.responses.find(r => r.id === 10).values, expectedGradient);
    vectorNear(direct.objective, [3, 0]);
    vectorNear(direct.constraints[0], [1.5, 0.003]);
    vectorNear(direct.constraints[1], [-0.375, -0.00075]);
    const jvp = point.jvp([1, 0, 0, 1], { count: 2, block: 1 });
    vectorNear(jvp.values, [...expectedGradient, 3, 0]);
    const weighted = { rows: 1, responses: { 10: [1] } };
    vectorNear(point.vjp(weighted).values, expectedGradient);
    const sparse = { rows: 1, responses: { 10: { rowStart: [0, 1], items: [0], values: [1] } } };
    vectorNear(point.vjp(sparse).values, expectedGradient);
    const expectedHessian = [0.0075, 0.0000075, 0.0000075, 0.00000003];
    vectorNear(point.hessian(weighted), expectedHessian, 3e-5);
    vectorNear(point.hessianVector(weighted, [1, 0, 0, 1], 2).values, expectedHessian, 3e-5);
    vectorNear(point.hessianVector(weighted, [2, 100]).values, [0.01575, 0.000018], 3e-5);
    const weights = point.lagrangianWeights(2, [0.1, 0.4]);
    vectorNear(point.vjp(weights).values, [6, 0]); // -0.1/0.005 + 0.4/0.02 = 0
    const stream = new Float64Array(4);
    point.streamGradients(block => {
      for (let r = 0; r < block.rowCount; ++r)
        stream.set(block.values.subarray(r * block.columnCount, (r + 1) * block.columnCount),
          (block.rowBegin + r) * 2 + block.columnBegin);
    }, 1);
    vectorNear(stream, [...expectedGradient, 3, 0]);
    const blocks = point.blocks();
    assert.equal(blocks.explicitParts.columns, 2);
    assert.ok(blocks.dofPoints.length >= 2);
    assert.throws(() => point.jvp([1]), /size|direction|variables/i);
    assert.throws(() => point.vjp({ rows: 1, responses: { 999: [1] } }), /response/i);
    assert.throws(() => problem.evaluate([NaN, 1000]), /finite/);
    assert.throws(() => point.hessianVector({ responses: { 10: [NaN] } }, [1, 0]), /finite/);
    assert.throws(() => point.streamGradients(() => point.values()), /re-entered/);
    const kept = point.values().responses[0].values;
    problem.delete(); // point retains the problem, factor and callback owners
    vectorNear(point.vjp(weighted).values, expectedGradient);
    point.delete();
    near(kept[0], 0.015);
    assert.throws(() => point.values(), /deleted/);
  } finally { point.delete(); problem.delete(); deck.delete(); }
});

test("Reader applies a final design atomically and accepts explicit solver variable order", () => {
  const deck = makeDeck("apply-design");
  try {
    assert.equal(oor.designVariables(deck).variables.length, 2);
    const before = deck.toModelText();
    assert.throws(() => oor.applyDesignPoint(deck, [1, 2], [3, NaN]), /finite/);
    assert.throws(() => oor.applyDesignPoint(deck, [1], [9]), /bounds/);
    assert.equal(deck.toModelText(), before);
    const update = oor.applyDesignPoint(deck, [2, 1], [1200, 3]);
    assert.equal(update.changedFields, 2);
    vectorNear(oor.designVariables(deck).variables.map(v => v.initialValue), [3, 1200]);
    assert.equal(oor.applyDesignPoint(deck, [1, 2], [3, 1200]).changedFields, 0);
  } finally { deck.delete(); }
});

test("a JavaScript response supplies its gradient and Hessian-vector chain rule", () => {
  const deck = makeDeck("callback-design");
  const square = {
    evaluate({ arguments: [u] }) { return { values: Array.from(u.values, x => x * x) }; },
    evaluateWithPartials({ arguments: [u] }) {
      return { values: Array.from(u.values, x => x * x), partials: [Array.from(u.values, x => 2 * x)] };
    },
    hessianVector({ weights, directions: [du] }) {
      return [Array.from(du.values, (x, i) => 2 * weights[i] * x)];
    },
  };
  const problem = solver.optimizationFromReader(oor, deck, {
    ...declarations, callables: { square },
    equationResponses: [{ id: 12, label: "squared-tip", callable: "square",
      arguments: [{ kind: "DesignResp1", id: 10, name: "u" }] }],
  });
  const point = problem.evaluate();
  try {
    const u = 0.015;
    const gradient = point.gradients().responses.find(r => r.id === 12).values;
    vectorNear(gradient, [-u * u, -2 * u * u / 1000]);
    const hessian = point.hessian({ responses: { 12: [1] } });
    vectorNear(hessian, [1.5 * u * u, 2 * u * u / 1000,
      2 * u * u / 1000, 6 * u * u / 1e6], 5e-4);
  } finally { point.delete(); problem.delete(); deck.delete(); }
});

test("a JavaScript iteration reaches the active displacement constraint", () => {
  const deck = makeDeck("iteration-design");
  const problem = solver.optimizationFromReader(oor, deck, declarations);
  let x = new Float64Array([2, 1000]);
  try {
    for (let iteration = 0; iteration < 6; ++iteration) {
      const point = problem.evaluate(x);
      try {
        const u = point.values().responses.find(r => r.id === 10).values[0];
        const du = point.gradients({ responseIds: [10] }).responses[0].values[0];
        x = new Float64Array([x[0] - (u - 0.02) / du, x[1]]);
      } finally { point.delete(); }
    }
    near(x[0], 1.5, 1e-8);
    oor.applyDesignPoint(deck, [1, 2], x);
    near(oor.designVariables(deck).variables[0].initialValue, 1.5, 1e-8);
  } finally { problem.delete(); deck.delete(); }
});

test("JavaScript declares a design independently of deck design cards and bounds derivative output", () => {
  const deck = makeDeck("declared-design");
  const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    ...declarations,
    objective: { responseId: 11, sense: "max" },
    variables: [
      { id: 1, initialValue: 2, lower: 0.5, upper: 5 },
      { id: 2, initialValue: 1000, lower: 500, upper: 2000 },
    ],
    relations: [
      { id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
        terms: [{ variableId: 1, coefficient: 1 }] },
      { id: 2, targetKind: "material", targetCard: "IsoMat", targetId: 1, field: "youngsModulus",
        terms: [{ variableId: 2, coefficient: 1 }] },
    ],
  }, { maximumOutputBytes: 16 });
  const point = problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 0.015);
    assert.throws(() => point.gradients(), /maximumOutputBytes/);
    assert.throws(() => point.hessian({ responses: { 10: [1] } }), /maximumOutputBytes/);
    let blocks = 0;
    point.streamGradients(() => ++blocks, 1);
    assert.ok(blocks > 0);
    vectorNear(point.vjp({ responses: { 10: [1] } }).values, [-0.0075, -0.000015]);
    vectorNear(point.vjp(point.lagrangianWeights(2)).values, [-6, 0]);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("a modal design exposes frequency derivatives and refuses unsupported second order", () => {
  const deck = (() => {
    const text = MODEL.replace("StaticLoadSelection;2", "EigenMethodSelection;3") + "LanczosEigenMethod;3;;;1\n";
    oor.writeFile("/work/modal-design.oos", text);
    return oor.readModel("/work/modal-design.oos", { strict: true });
  })();
  const problem = solver.optimizationFromReader(oor, deck, {
    responses: [{ id: 30, type: "Frequency", attributeA: 1, tracksMode: true }],
    objective: { responseId: 30, subcase: 1 },
  }, { analyses: { modes: [1] } });
  const point = problem.evaluate();
  try {
    const frequency = point.values().objective;
    const gradient = point.gradients().objective;
    near(gradient[0], 0, 1e-4); // axial stiffness and mass both proportional to A
    near(gradient[1], frequency / 2000, 1e-5);
    const next = problem.evaluate([2, 1100], point.values().trackedModes);
    try {
      near(next.values().trackedModes[0].mac, 1);
      assert.equal(next.analysisResults().eigenvalues.length, 1);
    } finally { next.delete(); }
    assert.throws(() => point.hessianVector({ responses: { 30: [1] } }, [1, 0]), /static|eigen|Frequency|MODES|modal/i);
  } finally { point.delete(); problem.delete(); deck.delete(); }
});
