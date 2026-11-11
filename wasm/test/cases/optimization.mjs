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

for (const family of ["SpringDofProperty", "SpringDofInline", "SpringScalarProperty", "SpringScalarInline"])
  test(`${family} uses exact spring field derivatives`, () => {
    const scalar = family.includes("Scalar");
    const property = family.includes("Property");
    const spring = {
      SpringDofProperty: "SpringDofProperty;1;1;1;1;2;1\nSpringProp;1;1000.;;2.",
      SpringDofInline: "SpringDofInline;1;1000.;1;1;2;1;;2.",
      SpringScalarProperty: "SpringScalarProperty;1;1;101;0\nSpringProp;1;1000.;;2.",
      SpringScalarInline: "SpringScalarInline;1;1000.;101;0",
    }[family];
    const model = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\n${scalar ? "" : "SinglePointConstrSelection;1\n"}StaticLoadSelection;2\n[model]\n${scalar
      ? "ScalarPoint;101\nScalarPointLoad;2;101;10."
      : "Node;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nPointForceLoad;2;2;;10.;1.;0.;0."}\n${spring}\n`;
    oor.writeFile(`/work/${family}-derivatives.oos`, model);
    const deck = oor.readModel(`/work/${family}-derivatives.oos`, { strict: true });
    const nativeModel = solver.modelFromReader(oor, deck);
    const relations = [{ id: 1, targetKind: property ? "property" : "element",
      targetCard: property ? "SpringProp" : family, targetId: 1, field: "stiffness",
      terms: [{ variableId: 1, coefficient: 1 }] }];
    const responses = [{ id: 10, type: "Force", propertyType: "Elements", attributeA: 2, targets: [1] }];
    const variables = [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 }];
    if (!scalar) responses.push({ id: 11, type: "Displacement", attributeA: 1, targets: [2] });
    if (family !== "SpringScalarInline") {
      variables.push({ id: 2, initialValue: 2, lower: 1, upper: 3 });
      relations.push({ id: 2, targetKind: property ? "property" : "element",
        targetCard: property ? "SpringProp" : family, targetId: 1, field: "stressCoefficient",
        terms: [{ variableId: 2, coefficient: 1 }] });
      responses.push({ id: 12, type: "Stress", propertyType: "Elements", attributeA: 2, targets: [1] });
    }
    const problem = solver.createOptimization(nativeModel, { variables, relations, responses });
    const point = problem.evaluate();
    try {
      const gradients = point.gradients();
      assert.ok(gradients.report.analyticDerivatives >= 1);
      assert.equal(gradients.report.fieldPerturbations, 0);
      near(point.values().responses.find(r => r.id === 10).values[0], scalar ? 10 : -10);
      near(gradients.responses.find(r => r.id === 10).values[0], 0, 1e-4);
      if (!scalar) {
        near(gradients.responses.find(r => r.id === 11).values[0], -1e-5, 1e-5);
        const direction = variables.length === 2 ? [1, 0] : [1];
        const curvature = point.hessianVector({ responses: { 11: [1] } }, direction);
        near(curvature.values[0], 2e-8, 2e-3);
      }
      if (variables.length === 2) {
        const stress = gradients.responses.find(r => r.id === 12).values;
        near(stress[0], 0, 1e-4);
        near(stress[1], scalar ? 10 : -10, 1e-5);
      }
    } finally { point.delete(); problem.delete(); nativeModel.delete(); deck.delete(); }
  });

for (const family of ["SpringScalarProperty", "SpringScalarInline"])
  test(`${family} supplies an exact modal eigenvalue gradient`, () => {
    const property = family === "SpringScalarProperty";
    const spring = property ? "SpringScalarProperty;1;1;101;0\nSpringProp;1;1000."
      : "SpringScalarInline;1;1000.;101;0";
    const model = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nEigenMethodSelection;3\n[model]\nScalarPoint;101\n${spring}\nMassScalarInline;2;2.;101;0\nLanczosEigenMethod;3;;;1\n`;
    oor.writeFile(`/work/${family}-modal-derivative.oos`, model);
    const deck = oor.readModel(`/work/${family}-modal-derivative.oos`, { strict: true });
    const nativeModel = solver.modelFromReader(oor, deck);
    const problem = solver.createOptimization(nativeModel, {
      variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 }],
      relations: [{ id: 1, targetKind: property ? "property" : "element",
        targetCard: property ? "SpringProp" : family, targetId: 1, field: "stiffness",
        terms: [{ variableId: 1, coefficient: 1 }] }],
      responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }],
    }, { analyses: { modes: [1] } });
    const point = problem.evaluate();
    try {
      near(point.values().responses[0].values[0], 500, 1e-5);
      const gradients = point.gradients();
      near(gradients.responses[0].values[0], 0.5, 1e-5);
      assert.ok(gradients.report.analyticDerivatives >= 1);
    } finally { point.delete(); problem.delete(); nativeModel.delete(); deck.delete(); }
  });

test("axial bushing property rates cover stiffness, stress and strain", () => {
  const model = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nPointForceLoad;2;2;;10.;1.;0.;0.\nBushing1D;1;1;1;2;0\nBushProp1D;1;1000.;;2.;3.;4.\n`;
  oor.writeFile("/work/axial-bushing-derivatives.oos", model);
  const deck = oor.readModel("/work/axial-bushing-derivatives.oos", { strict: true });
  const nativeModel = solver.modelFromReader(oor, deck);
  const fields = ["stiffness", "stressFactor", "strainFactor"];
  const initial = [1000, 3, 4];
  const problem = solver.createOptimization(nativeModel, {
    variables: initial.map((value, index) => ({ id: index + 1, initialValue: value,
      lower: value * 0.5, upper: value * 1.5 })),
    relations: fields.map((field, index) => ({ id: index + 1, targetKind: "property",
      targetCard: "BushProp1D", targetId: 1, field,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [
      { id: 10, type: "Displacement", attributeA: 1, targets: [2] },
      { id: 11, type: "Stress", propertyType: "Elements", attributeA: 5, targets: [1] },
      { id: 12, type: "Strain", propertyType: "Elements", attributeA: 6, targets: [1] },
    ],
  });
  const point = problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.ok(gradients.report.analyticDerivatives >= 1);
    assert.equal(gradients.report.fieldPerturbations, 0);
    vectorNear(gradients.responses.find(r => r.id === 10).values, [-1e-5, 0, 0], 1e-5);
    vectorNear(gradients.responses.find(r => r.id === 11).values, [0, 10, 0], 1e-5);
    vectorNear(gradients.responses.find(r => r.id === 12).values, [-4e-5, 0, 0.01], 1e-5);
  } finally { point.delete(); problem.delete(); nativeModel.delete(); deck.delete(); }
});

test("axial bushing mass and stiffness have exact modal rates", () => {
  const model = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nBushing1D;1;1;1;2;0\nBushProp1D;1;1000.;;2.;3.;4.\nLanczosEigenMethod;3;;;1\n`;
  oor.writeFile("/work/axial-bushing-modal.oos", model);
  const deck = oor.readModel("/work/axial-bushing-modal.oos", { strict: true });
  const nativeModel = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(nativeModel, {
    variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 },
      { id: 2, initialValue: 2, lower: 1, upper: 3 }],
    relations: ["stiffness", "mass"].map((field, index) => ({ id: index + 1,
      targetKind: "property", targetCard: "BushProp1D", targetId: 1, field,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }],
  }, { analyses: { modes: [1] } });
  const point = problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 1000, 1e-5);
    const gradients = point.gradients();
    vectorNear(gradients.responses[0].values, [1, -500], 1e-5);
    assert.ok(gradients.report.analyticDerivatives >= 2);
    assert.equal(gradients.report.fieldPerturbations, 0);
  } finally { point.delete(); problem.delete(); nativeModel.delete(); deck.delete(); }
});

for (const family of ["MassScalarInline", "MassScalarProperty", "MassDofInline", "MassDofProperty"])
  test(`${family} supplies an exact modal mass rate`, () => {
    const scalar = family.includes("Scalar");
    const property = family.includes("Property");
    const points = scalar ? "ScalarPoint;101" : "Node;1;;0.;0.;0.\nNode;2;;1.;0.;0.";
    const constraints = scalar ? "" : "SinglePointConstrSelection;1\n";
    const support = scalar ? "" : "SinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\n";
    const spring = scalar ? "SpringScalarInline;1;1000.;101;0"
      : "SpringDofInline;1;1000.;1;1;2;1";
    const connection = scalar ? "101;0" : "2;1;0;0";
    const mass = property ? `${family};2;2;${connection}\nMassProp;2;2.`
      : `${family};2;2.;${connection}`;
    const model = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\n${constraints}EigenMethodSelection;3\n[model]\n${points}\n${support}${spring}\n${mass}\nLanczosEigenMethod;3;;;1\n`;
    oor.writeFile(`/work/${family}-modal-mass-rate.oos`, model);
    const deck = oor.readModel(`/work/${family}-modal-mass-rate.oos`, { strict: true });
    const nativeModel = solver.modelFromReader(oor, deck);
    const problem = solver.createOptimization(nativeModel, {
      variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
      relations: [{ id: 1, targetKind: property ? "property" : "element",
        targetCard: property ? "MassProp" : family, targetId: 2, field: "mass",
        terms: [{ variableId: 1, coefficient: 1 }] }],
      responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }],
    }, { analyses: { modes: [1] } });
    const point = problem.evaluate();
    try {
      near(point.values().responses[0].values[0], 500, 1e-5);
      const gradients = point.gradients();
      near(gradients.responses[0].values[0], -250, 1e-5);
      assert.ok(gradients.report.analyticDerivatives >= 1);
      assert.equal(gradients.report.fieldPerturbations, 0);
    } finally { point.delete(); problem.delete(); nativeModel.delete(); deck.delete(); }
  });

const RIGID_MASS_FIELDS = [
  { field: "mass", value: 2 },
  ...[0.2, -0.3, 0.4].map((value, fieldItem) => ({ field: "offset", fieldItem, value })),
  ...[1, 0.07, 1.4, -0.03, 0.05, 1.8].map((value, fieldItem) => ({ field: "inertia", fieldItem, value })),
];
const MATRIX_MASS_FIELDS = Array.from({ length: 21 }, (_, fieldItem) => {
  let row = 0;
  while ((row + 1) * (row + 2) / 2 <= fieldItem) ++row;
  const column = fieldItem - row * (row + 1) / 2;
  return { field: "massMatrix", fieldItem, value: row === column ? 2 + row : 0.03 * (1 + column) };
});

function pointMassOptimization(card, fields) {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nRectFrame2;5;;0.;0.;0.;0.;0.;1.;0.;1.;0.\nRectFrame2;6;;0.;0.;0.;1.;0.;0.;0.;1.;0.\nNode;1;;0.;0.;0.\nNode;2;;0.;0.;0.;6\nSinglePointConstrList;1;123456;1\nBushing;1;1;1;2;;;;;0\nBushProp;1;Stiffness;1000.;1300.;1700.;800.;1100.;1500.\n${card};2;2;5;2.\nLanczosEigenMethod;3;;;6\n`;
  oor.writeFile(`/work/${card}-analytic-rates.oos`, text);
  const deck = oor.readModel(`/work/${card}-analytic-rates.oos`, { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: fields.map(({ value }, index) => ({ id: index + 1, initialValue: value,
      lower: value - 1, upper: value + 1 })),
    relations: fields.map(({ field, fieldItem }, index) => ({ id: index + 1, targetKind: "element",
      targetCard: card, targetId: 2, field, fieldItem,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [1, 2, 3].map(mode => ({ id: mode, type: "Eigenvalue", attributeA: mode })),
  }, { analyses: { modes: [1] } });
  return { problem, dispose() { problem.delete(); model.delete(); deck.delete(); } };
}

for (const [card, fields] of [["PointMassInertia", RIGID_MASS_FIELDS], ["PointMassMatrix", MATRIX_MASS_FIELDS]])
  for (const [column, field] of fields.entries())
    test(`${card} exact mass rate ${field.field}[${field.fieldItem ?? "scalar"}]`, () => {
      const fixture = pointMassOptimization(card, fields);
      const x = fields.map(entry => entry.value);
      const point = fixture.problem.evaluate();
      try {
        const gradients = point.gradients();
        assert.ok(gradients.report.analyticDerivatives >= 1);
        assert.equal(gradients.report.fieldPerturbations, 0);
        const step = 1e-4;
        const plus = [...x], minus = [...x];
        plus[column] += step;
        minus[column] -= step;
        const positive = fixture.problem.evaluate(plus), negative = fixture.problem.evaluate(minus);
        try {
          const a = positive.values().responses, b = negative.values().responses;
          for (let mode = 0; mode < a.length; ++mode)
            near(gradients.responses[mode].values[column], (a[mode].values[0] - b[mode].values[0]) / (2 * step), 2e-4);
        } finally { positive.delete(); negative.delete(); }
      } finally { point.delete(); fixture.dispose(); }
    });

test("six-component bushing gives exact stiffness and recovery rates", () => {
  const text = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n[model]\nNode;1;;0.;0.;0.\nNode;2;;0.;0.;0.\nSinglePointConstrList;1;123456;1\nBushing;1;1;1;2;;;;;0\nBushProp;1;Stiffness;1000.;1300.;1700.;800.;1100.;1500.\nPointForceLoad;2;2;;10.;1.;2.;3.\nPointMomentLoad;2;2;;5.;1.;2.;3.\n`;
  oor.writeFile("/work/bushing-analytic-rates.oos", text);
  const deck = oor.readModel("/work/bushing-analytic-rates.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const x = [1000, 1300, 1700, 800, 1100, 1500, 2, 3, 4, 5];
  const problem = solver.createOptimization(model, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: x.map((value, index) => ({ id: index + 1, targetKind: "property", targetCard: "BushProp",
      targetId: 1, field: index < 6 ? "stiffness" : "recovery", fieldItem: index < 6 ? index : index - 6,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: ["Force", "Stress", "Strain"].flatMap((type, group) => Array.from({ length: 6 }, (_, component) => ({
      id: 10 + 6 * group + component, type, propertyType: "Elements", attributeA: 2 + component, targets: [1],
    }))),
  });
  const point = problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.ok(gradients.report.analyticDerivatives >= 6);
    for (let column = 0; column < x.length; ++column) {
      const step = 1e-4 * x[column];
      const plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = problem.evaluate(plus), negative = problem.evaluate(minus);
      try {
        const a = positive.values().responses, b = negative.values().responses;
        for (let response = 0; response < a.length; ++response) {
          const expected = (a[response].values[0] - b[response].values[0]) / (2 * step);
          const actual = gradients.responses[response].values[column];
          assert.ok(Math.abs(actual - expected) <= 1e-8 * Math.max(1, Math.abs(expected)), `${actual} differs from ${expected}`);
        }
      } finally { positive.delete(); negative.delete(); }
    }
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

for (const grounded of [false, true]) test(`six-component bushing exact modal mass rate grounded=${grounded}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;0.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nBushing;1;1;${grounded ? "2;0" : "1;2"};;;;;0\nBushProp;1;Stiffness;1000.;1300.;1700.;800.;1100.;1500.\nLanczosEigenMethod;3;;;1\n`;
  oor.writeFile(`/work/bushing-mass-${grounded}.oos`, text);
  const deck = oor.readModel(`/work/bushing-mass-${grounded}.oos`, { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
    relations: [{ id: 1, targetKind: "property", targetCard: "BushProp", targetId: 1, field: "mass",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }],
  }, { analyses: { modes: [1] } });
  const point = problem.evaluate();
  try {
    const gradients = point.gradients();
    near(point.values().responses[0].values[0], grounded ? 500 : 1000, 1e-6);
    near(gradients.responses[0].values[0], grounded ? -250 : -500, 1e-6);
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.ok(gradients.report.analyticDerivatives >= 1);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

for (const family of ["SpringDofInline", "SpringDofProperty", "SpringScalarProperty", "Bushing1D", "Bushing"])
  test(`${family} uses analytic mixed stiffness and stress Hessian blocks`, () => {
    const scalar = family === "SpringScalarProperty";
    const property = family !== "SpringDofInline";
    const card = family.startsWith("Spring") ? (property ? "SpringProp" : family)
      : family === "Bushing1D" ? "BushProp1D" : "BushProp";
    const coefficient = family.startsWith("Spring") ? "stressCoefficient"
      : family === "Bushing1D" ? "stressFactor" : "recovery";
    const records = {
      SpringDofInline: "SpringDofInline;1;1000.;1;1;2;1;;2.",
      SpringDofProperty: "SpringDofProperty;1;1;1;1;2;1\nSpringProp;1;1000.;;2.",
      SpringScalarProperty: "SpringScalarProperty;1;1;101;0\nSpringProp;1;1000.;;2.",
      Bushing1D: "Bushing1D;1;1;1;2;0\nBushProp1D;1;1000.;;;2.;1.",
      Bushing: "Bushing;1;1;1;2;;;;;0\nBushProp;1;Stiffness;1000.;1000.;1000.;1000.;1000.;1000.",
    }[family];
    const setup = scalar ? "ScalarPoint;101\nScalarPointLoad;2;101;10.\nSpringScalarInline;3;500.;101;0"
      : "Node;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nPointForceLoad;2;2;;10.;1.;0.;0.\nSpringDofInline;3;500.;1;1;2;1";
    const text = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\n${scalar ? "" : "SinglePointConstrSelection;1\n"}StaticLoadSelection;2\n[model]\n${setup}\n${records}\n`;
    oor.writeFile(`/work/${family}-mixed-hessian.oos`, text);
    const deck = oor.readModel(`/work/${family}-mixed-hessian.oos`, { strict: true });
    const model = solver.modelFromReader(oor, deck);
    const problem = solver.createOptimization(model, {
      variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 },
        { id: 2, initialValue: 2, lower: 1, upper: 3 }],
      relations: ["stiffness", coefficient].map((field, index) => ({ id: index + 1,
        targetKind: property ? "property" : "element", targetCard: card, targetId: 1, field,
        fieldItem: family === "Bushing" ? 0 : -1, terms: [{ variableId: index + 1, coefficient: 1 }] })),
      responses: [{ id: 10, type: "Stress", propertyType: "Elements",
        attributeA: family === "Bushing1D" ? 5 : 2, targets: [1] }],
    });
    const point = problem.evaluate();
    try {
      const result = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2);
      assert.equal(result.report.hasHessianBlocks, true, result.report.hessianBlocksReason);
      assert.equal(result.report.perturbedLinearisations, 0);
      const sign = family === "SpringDofInline" || family === "SpringDofProperty" ? -1 : 1;
      vectorNear(result.values, [sign * -2 * 2 * 10 * 500 / 1500 ** 3,
        sign * 10 * 500 / 1500 ** 2, sign * 10 * 500 / 1500 ** 2, 0], 1e-5);
      const blocks = point.blocks({ responses: { 10: [1] } });
      for (let entry = 0; entry < blocks.hessianDesign.length; ++entry) {
        const isMixed = blocks.designSupport.row[entry] !== blocks.designSupport.column[entry];
        near(blocks.hessianDesign[entry], isMixed ? sign * 10 / 1500 : 0, 1e-5);
      }
    } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
  });

test("rigid mass and offset use analytic mixed inertial Hessian blocks", () => {
  const text = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n[model]\nNode;1;;0.;0.;0.\nNode;2;;0.;0.;0.\nSinglePointConstrList;1;123456;1\nBushing;1;1;1;2;;;;;0\nBushProp;1;Stiffness;1000.;1300.;1700.;800.;1100.;1500.\nPointMassInertia;2;2;;2.\nGravityLoad;2;;10.;1.;0.;0.\n`;
  oor.writeFile("/work/rigid-mass-mixed-hessian.oos", text);
  const deck = oor.readModel("/work/rigid-mass-mixed-hessian.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const problem = solver.createOptimization(model, {
    variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 },
      { id: 2, initialValue: 0.3, lower: 0.1, upper: 0.5 }],
    relations: ["mass", "offset"].map((field, index) => ({ id: index + 1, targetKind: "element",
      targetCard: "PointMassInertia", targetId: 2, field, fieldItem: index === 0 ? -1 : 1,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [{ id: 10, type: "Displacement", attributeA: 6, targets: [2] }],
  });
  const point = problem.evaluate();
  try {
    const result = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2);
    assert.equal(result.report.hasHessianBlocks, true, result.report.hessianBlocksReason);
    assert.equal(result.report.perturbedLinearisations, 0);
    vectorNear(result.values, [0, -10 / 1500, -10 / 1500, 0], 1e-5);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

for (const opening of [-1, 1]) test(`gap spring rates are analytic in the initial branch opening=${opening}`, () => {
  const closed = opening < 0;
  const text = `[executive]\nSolutionSequence;101\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;${closed ? "456" : "23456"};2\nPointForceLoad;2;2;;10.;1.;0.;0.\n${closed ? "PointForceLoad;2;2;;20.;0.;1.;0.\nPointForceLoad;2;2;;30.;0.;0.;1.\n" : ""}Gap2;1;1;1;2;;0.;1.;0.\nGapProp;1;${opening}.;0.;1000.;50.;300.\n`;
  oor.writeFile(`/work/gap-analytic-${opening}.oos`, text);
  const deck = oor.readModel(`/work/gap-analytic-${opening}.oos`, { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const x = [1000, 50, 300];
  const problem = solver.createOptimization(model, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: ["closedStiffness", "openStiffness", "transverseStiffness"].map((field, index) => ({ id: index + 1,
      targetKind: "property", targetCard: "GapProp", targetId: 1, field,
      terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2] },
      ...Array.from({ length: 8 }, (_, component) => ({ id: 20 + component, type: "Force",
        propertyType: "Elements", attributeA: 2 + component, targets: [1] }))],
  });
  const point = problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.equal(gradients.report.fieldPerturbations, 0);
    vectorNear(gradients.responses[0].values, closed ? [-1e-5, 0, 0] : [0, -0.004, 0], 1e-6);
    vectorNear(gradients.responses.find(r => r.id === 20).values, [0, 0, 0], 1e-4);
    if (closed) vectorNear(gradients.responses.find(r => r.id === 24).values, [0, 0, -20 / 300 ** 2], 1e-6);
    const result = point.hessianVector({ responses: { 10: [1] } }, [1, 1, 1]);
    assert.equal(result.report.hasHessianBlocks, true, result.report.hessianBlocksReason);
    assert.equal(result.report.perturbedLinearisations, 0);
    vectorNear(result.values, closed ? [2e-8, 0, 0] : [0, 20 / 50 ** 3, 0], 1e-6);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

test("fastener condensation preserves analytic stiffness and mass rates", () => {
  const points = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0.1],[1,0,0.1],[1,1,0.1],[0,1,0.1],
    [0.5,0.5,0.05],[0.5,0.5,0],[0.5,0.5,0.1]];
  const real = value => Number.isInteger(value) ? `${value}.` : String(value);
  const bulk = points.map(([x,y,z], index) => `Node;${index+1};;${real(x)};${real(y)};${real(z)}`);
  bulk.push("ShellQuad4;1;1;1;2;3;4", "ShellQuad4;2;2;5;6;7;8", "ShellProp;1;1;0.1;1;;1",
    "ShellProp;2;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "Fastener;101;3;Elements;1;2;9;10;11",
    "FastenerProp;3;0.1;0;1;1000.;1100.;1200.;400.;500.;600.", "SinglePointConstrList;1;123456;1;2;3;4",
    "SinglePointConstrList;1;12456;5;6;7;8;9", "PointForceLoad;2;11;;10.;0.;0.;1.", "GravityLoad;2;;10.;0.;0.;1.");
  const text = ["[case]", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "[model]", ...bulk, ""].join("\n");
  oor.writeFile("/work/fastener-analytic-rates.oos", text);
  const deck = oor.readModel("/work/fastener-analytic-rates.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck);
  const x = [1000, 1100, 1200, 400, 500, 600, 2];
  const problem = solver.createOptimization(model, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: x.map((value, index) => ({ id: index + 1, targetKind: "property", targetCard: "FastenerProp", targetId: 3,
      field: index < 3 ? "translationalStiffness" : index < 6 ? "rotationalStiffness" : "mass",
      fieldItem: index < 6 ? index % 3 : -1, terms: [{ variableId: index + 1, coefficient: 1 }] })),
    responses: [{ id: 10, type: "Displacement", attributeA: 3, targets: [11] },
      { id: 11, type: "Force", propertyType: "Elements", attributeA: 2, targets: [101] },
      { id: 12, type: "Stress", propertyType: "Elements", attributeA: 2, targets: [101] }],
  });
  const point = problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.ok(gradients.report.analyticDerivatives >= 6);
    for (let column = 0; column < x.length; ++column) {
      const step = 1e-4 * x[column], plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = problem.evaluate(plus), negative = problem.evaluate(minus);
      try {
        const a = positive.values().responses, b = negative.values().responses;
        for (let response = 0; response < a.length; ++response) {
          const expected = (a[response].values[0] - b[response].values[0]) / (2 * step);
          const actual = gradients.responses[response].values[column];
          assert.ok(Math.abs(actual - expected) <= 1e-8 * Math.max(1, Math.abs(expected)));
        }
      } finally { positive.delete(); negative.delete(); }
    }
    const result = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 0, 0, 0, 0]);
    assert.equal(result.report.hasHessianBlocks, true, result.report.hessianBlocksReason);
    assert.equal(result.report.perturbedLinearisations, 0);
  } finally { point.delete(); problem.delete(); model.delete(); deck.delete(); }
});

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

test("inertia-relief shape gradients include moved support geometry", () => {
  const model = `[executive]
SolutionSequence;101
[case]
SubcaseStart;1
StaticLoadSelection;2
[model]
Node;1;;0.;0.;0.
Node;2;;1.;0.;0.
Node;3;;0.;1.;0.
ShellTria3;1;1;1;2;3
ShellProp;1;1;0.1;1;;1
IsoMat;1;1000.;;0.3;1.
FreeBodySupport;1;123;2;23;3;3
Parameter;InertiaRelief;-1
PointForceLoad;2;3;;1.;1.;0.;0.
DesignVar;1;MOVE;0.;-0.2;0.2
DesignNodeMove;1;3;0;1.;1.;0.;0.
`;
  oor.writeFile("/work/optimization-relief-shape.oos", model);
  const deck = oor.readModel("/work/optimization-relief-shape.oos", { strict: true });
  const problem = solver.optimizationFromReader(oor, deck, {
    responses: [{ id: 10, label: "tip", type: "Displacement", attributeA: 1, targets: [3] }],
  });
  const point = problem.evaluate();
  try {
    const native = point.gradients();
    assert.equal(native.report.analysisPerturbations, 2);
    const gradient = native.responses[0].values[0];
    const positive = problem.evaluate([0.0001]);
    const negative = problem.evaluate([-0.0001]);
    try {
      const numerical = (positive.values().responses[0].values[0] - negative.values().responses[0].values[0]) / 0.0002;
      near(gradient, numerical, 2e-3);
      near(point.jvp([1]).values[0], gradient, 1e-6);
      near(point.vjp({ rows: 1, responses: { 10: [1] } }).values[0], gradient, 1e-6);
      const streamed = [];
      point.streamGradients(block => streamed.push(...block.values), 1);
      near(streamed[0], gradient, 1e-6);
    } finally { positive.delete(); negative.delete(); }
  } finally { point.delete(); problem.delete(); deck.delete(); }
});

test("an imposed axial deformation has a complete native load derivative", () => {
  const model = MODEL.replace("StaticLoadSelection;2", "StaticLoadSelection;2\nDeformationLoadSelection;7")
    .replace("DesignVar;1;AREA", "AxialDeformationLoad;7;1;0.03\nDesignVar;1;AREA")
    .replace("DesignVar;2;MODULUS;1000.;500.;2000.",
      "DesignVar;2;MODULUS;1000.;500.;2000.\nDesignVar;3;LENGTH;0.;-0.5;0.5\nDesignNodeMove;3;2;0;1.;1.;0.;0.");
  oor.writeFile("/work/optimization-deformation.oos", model);
  const deck = oor.readModel("/work/optimization-deformation.oos", { strict: true });
  const problem = solver.optimizationFromReader(oor, deck, declarations);
  const point = problem.evaluate();
  try {
    const displacement = point.values().responses.find(response => response.id === 10).values[0];
    near(displacement, 0.045);
    for (const method of ["direct", "adjoint"]) {
      const gradient = point.gradients({ method }).responses.find(response => response.id === 10).values;
      vectorNear(gradient, [-0.0075, -0.000015, 0.005], 2e-5);
    }
  } finally { point.delete(); problem.delete(); deck.delete(); }
});

test("selected non-structural mass contributes to sizing and shape gradients", () => {
  const model = MODEL.replace("StaticLoadSelection;2", "StaticLoadSelection;2\nNonStructMassSelection;20")
    .replace("PointForceLoad;2;2;;10.;1.;0.;0.", "GravityLoad;2;;10.;1.;0.;0.")
    .replace("DesignVar;1;AREA", "NonStructMass;20;RodProp;1;2.\nDesignVar;1;AREA")
    .replace("DesignVar;2;MODULUS;1000.;500.;2000.",
      "DesignVar;2;MODULUS;1000.;500.;2000.\nDesignVar;3;LENGTH;0.;-0.5;0.5\nDesignNodeMove;3;2;0;1.;1.;0.;0.");
  oor.writeFile("/work/optimization-selected-mass.oos", model);
  const deck = oor.readModel("/work/optimization-selected-mass.oos", { strict: true });
  const problem = solver.optimizationFromReader(oor, deck, declarations);
  const point = problem.evaluate();
  try {
    const displacement = point.values().responses.find(response => response.id === 10).values[0];
    near(displacement, 0.09);
    const gradient = point.gradients().responses.find(response => response.id === 10).values;
    vectorNear(gradient, [-0.0225, -0.00009, 0.06], 3e-5);
    const curvature = point.hessianVector({ responses: { 10: [1] } }, [0, 0, 1]).values;
    vectorNear(curvature, [-0.015, -0.00006, 0.02], 5e-4);
  } finally { point.delete(); problem.delete(); deck.delete(); }
});

test("a global response Hessian uses native first-order gradients", () => {
  const deck = makeDeck("global-response-hessian");
  const problem = solver.optimizationFromReader(oor, deck, {
    responses: [{ id: 12, label: "mass fraction", type: "FractionalMass" }],
    objective: { responseId: 12 },
  });
  const point = problem.evaluate();
  try {
    near(point.values().objective, 1);
    const product = point.hessianVector({ responses: { 12: [1] } }, [1, 0]).values;
    assert.ok(product.every(value => Math.abs(value) < 1e-7), `expected zero curvature, found ${product}`);
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

test("a modal design exposes frequency gradients and native-gradient curvature", () => {
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
    const curvature = point.hessianVector({ rows: 1, responses: { 30: [1] } }, [0, 1]);
    near(curvature.values[1], -frequency / (4 * 1000 ** 2), 5e-3);
    near(curvature.values[0], 0, 1e-3);
    assert.equal(curvature.report.perturbedLinearisations, 2);
    assert.throws(() => point.hessianVector({ responses: { 30: { rowStart: [0, 2], items: [0], values: [1] } } },
      [0, 1]), /sparse weights/);
  } finally { point.delete(); problem.delete(); deck.delete(); }
});
