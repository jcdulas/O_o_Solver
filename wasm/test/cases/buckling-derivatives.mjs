import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

function near(actual, expected, tolerance = 3e-5) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${actual} differs from ${expected}`);
}

function build(text, x, relations, count = 1) {
  oor.writeFile("/work/buckling-derivatives.oos", text);
  const deck = oor.readModel("/work/buckling-derivatives.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck);
  try {
    const problem = solver.createOptimization(model, {
      variables: x.map((value, i) => ({ id: i + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
      relations: relations.map((target, i) => ({ id: i + 1, ...target,
        terms: [{ variableId: i + 1, coefficient: 1 }] })),
      responses: Array.from({ length: count }, (_, i) => ({ id: 10 + i, type: "BucklingFactor", attributeA: i + 1,
        tracksMode: true, subcases: [2] })),
    }, { analyses: { buckling: [2] } });
    return { problem, dispose() { problem.delete(); model.delete(); deck.delete(); } };
  } catch (error) { model.delete(); deck.delete(); throw error; }
}

for (const tension of [false, true]) test(`analytic buckling Hessian includes rod load sharing tension=${tension}`, () => {
  const text = `[executive]\nSolutionSequence;105\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\nSubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\nStaticSubcaseSelection;1\n[model]\nNode;1;;0.;0.;0.\nNode;2;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;456;2\nRod2;1;1;1;2\nRodProp;1;1;2.\nIsoMat;1;1000.;400.;0.25;1.\nSpringDofInline;2;500.;1;1;2;1\nSpringDofInline;3;10.;1;2;2;2\nSpringDofInline;4;20.;1;3;2;3\nPointForceLoad;2;2;;1.;${tension ? 1 : -1}.;0.;0.\nLanczosEigenMethod;3;;;2\n`;
  const built = build(text, [2, 500], [
    { targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area" },
    { targetKind: "element", targetCard: "SpringDofInline", targetId: 2, field: "stiffness" },
  ], 2);
  const point = built.problem.evaluate();
  try {
    // N_rod = P * (EA/L)/(EA/L+k); lambda_y = 20 * (1+k/(500*A)), lambda_z = 2 lambda_y.
    const sign = tension ? -1 : 1;
    near(point.values().responses[0].values[0], sign * 30);
    const gradients = point.gradients();
    near(gradients.responses[0].values[0], sign * -5);
    near(gradients.responses[0].values[1], sign * 0.02);
    assert.equal(gradients.report.fieldPerturbations, 0);
    const result = point.hessianVector({ responses: { 10: [1], 11: [0.2] } }, [1, 0, 0, 1], 2);
    assert.equal(result.report.usesBucklingTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
    assert.equal(result.report.factorisations, 2);
    [5, -0.01, -0.01, 0].forEach((value, i) => near(result.values[i], sign * 1.4 * value));
    const sparse = point.hessianVector({ responses: { 10: { rowStart: [0, 1], items: [0], values: [1] } } }, [0.2, 10]);
    near(sparse.values[0], sign * 0.9);
    near(sparse.values[1], sign * -0.002);
    const bounded = built.problem.evaluate([1, 250]);
    try {
      const product = bounded.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2);
      assert.equal(product.report.usesBucklingTangents, true);
      assert.equal(product.report.perturbedLinearisations, 0);
      [20, -0.04, -0.04, 0].forEach((value, i) => near(product.values[i], sign * value));
    } finally { bounded.delete(); }
  } finally { point.delete(); built.dispose(); }
});

const shells = [
  { triangles: false, extracted: 1 }, { triangles: false, extracted: 2 },
  { triangles: true, extracted: 1 }, { triangles: true, extracted: 2 },
  { triangles: false, extracted: 2, warped: true },
  { triangles: false, extracted: 2, drilling: true }, { triangles: true, extracted: 2, drilling: true },
];
for (const { triangles, extracted, warped = false, drilling = false } of shells)
  test(`analytic buckling shell Hessian triangles=${triangles} roots=${extracted} warped=${warped} drilling=${drilling}`, () => {
    const family = `Shell${triangles ? "Tria3" : "Quad4"}${drilling ? "Drilling" : ""}`;
    const elements = triangles ? `${family};1;1;1;2;3\n${family};2;1;1;3;4` : `${family};1;1;1;2;3;4`;
    const text = `[executive]\nSolutionSequence;105\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\nSubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\nStaticSubcaseSelection;1\n[model]\nNode;1;;0.;0.;0.\nNode;2;;2.;0.;0.\nNode;3;;2.;1.;0.\nNode;4;;0.;1.;0.\nSinglePointConstrList;1;123456;1;4\nSinglePointConstrList;1;246;2;3\n${elements}\nShellProp;1;1;0.2;1;;1\nIsoMat;1;1000.;400.;0.25;1.\nSpringDofInline;3;100.;1;1;2;1\nSpringDofInline;4;10.;1;3;3;3\nPointForceLoad;2;2;;0.5;-1.;0.;0.\nPointForceLoad;2;3;;0.5;-1.;0.;0.\nLanczosEigenMethod;3;;;${extracted}\n`;
    const x = [0.2, 100, 10];
    const modelText = warped ? text.replace("Node;3;;2.;1.;0.", "Node;3;;2.;1.;0.15") : text;
    const built = build(modelText, x, [
      { targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness" },
      { targetKind: "element", targetCard: "SpringDofInline", targetId: 3, field: "stiffness" },
      { targetKind: "element", targetCard: "SpringDofInline", targetId: 4, field: "stiffness" },
    ], extracted);
    const point = built.problem.evaluate();
    try {
      const weights = Object.fromEntries(Array.from({ length: extracted }, (_, i) => [10 + i, [1 / (i + 1)]]));
      const hessian = point.hessianVector({ responses: weights }, [1, 0, 0, 0, 1, 0, 0, 0, 1], 3);
      assert.equal(hessian.report.usesBucklingTangents, true);
      assert.equal(hessian.report.perturbedLinearisations, 0);
      assert.equal(hessian.report.factorisations, extracted);
      for (let column = 0; column < x.length; ++column) {
        const step = 1e-4 * x[column], plus = [...x], minus = [...x];
        plus[column] += step;
        minus[column] -= step;
        const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
        try {
          const a = positive.gradients().responses, b = negative.gradients().responses;
          for (let row = 0; row < x.length; ++row) {
            let expected = 0;
            for (let response = 0; response < extracted; ++response)
              expected += weights[10 + response][0] * (a[response].values[row] - b[response].values[row]) / (2 * step);
            near(hessian.values[row * 3 + column], expected);
            near(hessian.values[row * 3 + column], hessian.values[column * 3 + row]);
          }
        } finally { positive.delete(); negative.delete(); }
      }
    } finally { point.delete(); built.dispose(); }
  });
