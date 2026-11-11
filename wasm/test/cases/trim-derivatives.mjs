import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { wing } from "../helpers/optimization-models.mjs";

function near(actual, expected, tolerance = 3e-5) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${actual} differs from ${expected}`);
}

function build({ free = false, inertia = false, response = "fields" } = {}) {
  const deck = wing({ load: true });
  if (free) deck.addCard({ type: "PointMassInertia", fields: [2, 3, 0, 2, 0, 0.2, 0, 0.1, 0, 0.1, 0, 0, 0.1] });
  else deck.addCard({ type: "SpringDofInline", fields: [2, 1000, 1, 3, 3, 3, null, 2] });
  if (inertia) deck.addCard({ type: "GravityLoad", fields: [20, null, 10, 0, 0, 1] });
  let source = deck;
  if (free) {
    deck.caseControl.global.remove("SinglePointConstrSelection");
    const labels = ["RigidAccelX", "RigidAccelY", "RigidAccelZ", "RigidAngularAccelX", "RigidAngularAccelY",
      "RigidAngularAccelZ"];
    labels.forEach((label, i) => deck.addCard({ type: "AeroRigidTrimVariable", fields: [71 + i, label] }));
    const text = deck.toModelText().replace(/SinglePointConstrList;[^\n]+/g, "") +
      "\nFreeBodySupport;1;123;2;23;4;3\n";
    oor.writeFile("/work/trim-free-derivatives.oos", text);
    source = oor.readModel("/work/trim-free-derivatives.oos", { strict: true });
  }
  const model = solver.modelFromReader(oor, source);
  const x = free ? [0.1, 2, 0.2] : [0.1, 1000, 2];
  const responses = response === "coefficients" ? [
    { id: 10, type: "StabilityDerivative", attributeA: 70, attributeB: free ? 0 : 1, targets: [3] },
  ] : response === "variables" ? [
    { id: 10, type: "TrimVariable", attributeA: 73 },
  ] : response === "vonMises" ? [
    { id: 10, type: "Stress", propertyType: "Elements", targets: [1], attributeA: 9 },
  ] : [
    { id: 10, type: "Displacement", attributeA: 3, targets: [3] },
    { id: 11, type: "Stress", propertyType: "Elements", targets: [2], attributeA: 2 },
    { id: 12, type: "Force", propertyType: "Elements", targets: [2], attributeA: 2 },
  ];
  try {
    const problem = solver.createOptimization(model, {
      variables: x.map((value, i) => ({ id: i + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
      relations: [
        { id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness",
          terms: [{ variableId: 1, coefficient: 1 }] },
        { id: 2, targetKind: "element", targetCard: free ? "PointMassInertia" : "SpringDofInline", targetId: 2,
          field: free ? "mass" : "stiffness",
          terms: [{ variableId: 2, coefficient: 1 }] },
        { id: 3, targetKind: "element", targetCard: free ? "PointMassInertia" : "SpringDofInline", targetId: 2,
          field: free ? "offset" : "stressCoefficient", fieldItem: free ? 1 : -1,
          terms: [{ variableId: 3, coefficient: 1 }] },
      ], responses,
    }, { analyses: { aeroelastic: [1] } });
    return { problem, x, dispose() { problem.delete(); model.delete(); if (source !== deck) source.delete(); deck.delete(); } };
  } catch (error) { model.delete(); if (source !== deck) source.delete(); deck.delete(); throw error; }
}

function check(built, weights, { sparse = false } = {}) {
  const point = built.problem.evaluate();
  const n = built.x.length;
  try {
    const selected = sparse ? Object.fromEntries(Object.entries(weights).map(([id, values]) =>
      [id, { rowStart: [0, values.length], items: values.map((_, i) => i), values }])) : weights;
    const identity = Array.from({ length: n * n }, (_, i) => +(Math.floor(i / n) === i % n));
    const result = point.hessianVector({ responses: selected }, identity, n);
    assert.equal(result.report.usesTrimTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
    assert.equal(result.report.factorisations, 0); // the factors belong to the accepted trim run
    assert.ok(result.values.some(value => Math.abs(value) > 1e-8), "the fixture must have nonzero curvature");
    for (let column = 0; column < n; ++column) {
      const step = 1e-4 * built.x[column], plus = [...built.x], minus = [...built.x];
      plus[column] += step;
      minus[column] -= step;
      const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
      try {
        const a = positive.gradients().responses, b = negative.gradients().responses;
        for (let row = 0; row < n; ++row) {
          let expected = 0;
          for (let response = 0; response < a.length; ++response)
            for (let item = 0; item < a[response].itemCount; ++item)
              expected += weights[a[response].id][item] *
                (a[response].values[item * n + row] - b[response].values[item * n + row]) / (2 * step);
          near(result.values[row * n + column], expected);
          near(result.values[row * n + column], result.values[column * n + row]);
        }
      } finally { positive.delete(); negative.delete(); }
    }
    const direction = [0.02, -50, 0.3];
    const product = point.hessianVector({ responses: weights }, direction);
    for (let row = 0; row < n; ++row)
      near(product.values[row], direction.reduce((value, rate, col) => value + result.values[row * n + col] * rate, 0));
    return point.gradients();
  } finally { point.delete(); built.dispose(); }
}

for (const inertia of [false, true]) test(`analytic trim field Hessian includes applied inertia=${inertia}`, () => {
  const gradients = check(build({ inertia }), { 10: [1], 11: [0.02], 12: [-0.03] });
  assert.equal(gradients.report.fieldPerturbations, 0);
  assert.equal(gradients.report.analysisPerturbations, 0);
});

test("analytic trim von Mises Hessian chains local recovery curvature", () => {
  check(build({ response: "vonMises" }), { 10: [1] });
});

test("analytic trim restrained stability Hessian accepts sparse weights", () => {
  check(build({ response: "coefficients" }), { 10: [1] }, { sparse: true });
});

test("analytic trim free-body stability Hessian differentiates mass relief", () => {
  check(build({ free: true, response: "coefficients" }), { 10: [1] });
});

test("analytic trim free-variable Hessian differentiates support accelerations", () => {
  check(build({ free: true, response: "variables", inertia: true }), { 10: [1] });
});
