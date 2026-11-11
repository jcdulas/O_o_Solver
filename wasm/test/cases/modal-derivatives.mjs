import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

function near(actual, expected, tolerance = 1e-7) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${actual} differs from ${expected}`);
}

function fixture(text, declarations, readDesignCards = false) {
  oor.writeFile("/work/modal-field-derivatives.oos", text);
  const deck = oor.readModel("/work/modal-field-derivatives.oos", { strict: true });
  const model = readDesignCards ? null : solver.modelFromReader(oor, deck);
  try {
    const settings = { analyses: { modes: [1] } };
    const problem = readDesignCards ? solver.optimizationFromReader(oor, deck, declarations, settings)
      : solver.createOptimization(model, declarations, settings);
    return { problem, dispose() { problem.delete(); model?.delete(); deck.delete(); } };
  } catch (error) { model?.delete(); deck.delete(); throw error; }
}

for (const norm of ["Mass", "Maximum", "Point"]) test(`modal field analytic mass normalisation ${norm}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nSpringDofInline;1;1000.;1;1;2;1;;2.\nMassDofInline;2;2.;2;1;0;0\nLanczosEigenMethod;3;;;1;;;;${norm}\n`;
  const x = [1000, 2, 2];
  const modelText = norm === "Point" ? text.replace(`LanczosEigenMethod;3;;;1;;;;Point`,
    "RealEigenMethod;3;Lanczos;;;;1;Point;2;1") : text;
  const built = fixture(modelText, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: [
      { id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1, field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] },
      { id: 2, targetKind: "element", targetCard: "MassDofInline", targetId: 2, field: "mass", terms: [{ variableId: 2, coefficient: 1 }] },
      { id: 3, targetKind: "element", targetCard: "SpringDofInline", targetId: 1, field: "stressCoefficient", terms: [{ variableId: 3, coefficient: 1 }] },
    ],
    responses: [
      { id: 10, type: "Displacement", attributeA: 1, targets: [2], mode: 1, tracksMode: true },
      { id: 11, type: "Stress", propertyType: "Elements", attributeA: 2, targets: [1], mode: 1 },
      { id: 12, type: "Force", propertyType: "Elements", attributeA: 2, targets: [1], mode: 1 },
    ],
  });
  const point = built.problem.evaluate();
  try {
    const values = point.values(), gradients = point.gradients();
    const shape = norm === "Mass" ? 1 / Math.sqrt(2) : 1;
    const massRate = norm === "Mass" ? -0.5 / 2 ** 1.5 : 0;
    near(values.responses[0].values[0], shape);
    assert.equal(values.responses[0].items[0].mode, 1);
    [0, massRate, 0].forEach((expected, i) => near(gradients.responses[0].values[i], expected));
    [-2 * shape, -2000 * massRate, -1000 * shape].forEach((expected, i) => near(gradients.responses[1].values[i], expected));
    [-shape, -1000 * massRate, 0].forEach((expected, i) => near(gradients.responses[2].values[i], expected));
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.equal(gradients.report.analysisPerturbations, 0);
    assert.equal(gradients.report.factorisations, 1);
    assert.equal(gradients.report.solves, 3);
    const jvp = point.jvp([0, 1, 0]);
    near(jvp.values[0], massRate);
    near(jvp.values[1], -2000 * massRate);
    const vjp = point.vjp({ responses: { 10: [1] } });
    [0, massRate, 0].forEach((expected, i) => near(vjp.values[i], expected));
    const stream = new Float64Array(9);
    point.streamGradients(block => {
      for (let row = 0; row < block.rowCount; ++row)
        stream.set(block.values.subarray(row * block.columnCount, (row + 1) * block.columnCount),
          (block.rowBegin + row) * 3 + block.columnBegin);
    }, 1);
    gradients.responses.flatMap(response => Array.from(response.values)).forEach((expected, i) => near(stream[i], expected));
    const secondMass = norm === "Mass" ? 0.75 / 2 ** 2.5 : 0;
    const expectedHessians = [
      [0, 0, 0, 0, secondMass, 0, 0, 0, 0],
      [0, -2 * massRate, -shape, -2 * massRate, -2000 * secondMass, -1000 * massRate,
        -shape, -1000 * massRate, 0],
      [0, -massRate, 0, -massRate, -1000 * secondMass, 0, 0, 0, 0],
    ];
    for (let response = 0; response < expectedHessians.length; ++response) {
      const result = point.hessianVector({ responses: { [10 + response]: [1] } },
        [1, 0, 0, 0, 1, 0, 0, 0, 1], 3);
      assert.equal(result.report.usesModalTangents, true);
      assert.equal(result.report.perturbedLinearisations, 0);
      expectedHessians[response].forEach((expected, i) => near(result.values[i], expected));
    }
  } finally { point.delete(); built.dispose(); }
});

for (const norm of ["Mass", "Maximum", "Point"])
  for (const extracted of [1, 2]) test(`modal field bordered derivatives ${norm} extracted=${extracted}`, () => {
    const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nNode;3;;2.;0.;0.\nNode;4;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2;3\nRigidBodyStar;10;3;123456;4\nSpringDofInline;1;1000.;1;1;2;1;;2.\nSpringDofInline;2;500.;2;1;3;1\nMassDofInline;3;2.;2;1;0;0\nMassDofInline;4;3.;4;1;0;0\nLanczosEigenMethod;3;;;${extracted};;;;${norm}\n`;
    const x = [1000, 500, 2];
    const modelText = norm === "Point" ? text.replace(`LanczosEigenMethod;3;;;${extracted};;;;Point`,
      `RealEigenMethod;3;Lanczos;;;;${extracted};Point;2;1`) : text;
    const built = fixture(modelText, {
      variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
      relations: [
        { id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1, field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] },
        { id: 2, targetKind: "element", targetCard: "SpringDofInline", targetId: 2, field: "stiffness", terms: [{ variableId: 2, coefficient: 1 }] },
        { id: 3, targetKind: "element", targetCard: "MassDofInline", targetId: 3, field: "mass", terms: [{ variableId: 3, coefficient: 1 }] },
      ],
      responses: Array.from({ length: extracted }, (_, index) => index + 1).flatMap(mode => [
        { id: mode * 10, type: "Displacement", attributeA: 1, targets: [2, 4], mode, tracksMode: true },
        { id: mode * 10 + 1, type: "Stress", propertyType: "Elements", attributeA: 2, targets: [1], mode },
        { id: mode * 10 + 2, type: "Force", propertyType: "Elements", attributeA: 2, targets: [2], mode },
      ]),
    });
    const point = built.problem.evaluate();
    try {
      const gradients = point.gradients();
      assert.equal(gradients.report.fieldPerturbations, 0);
      assert.equal(gradients.report.analysisPerturbations, 0);
      for (let column = 0; column < x.length; ++column) {
        const step = 1e-4 * x[column], plus = [...x], minus = [...x];
        plus[column] += step;
        minus[column] -= step;
        const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
        try {
          const a = positive.values().responses, b = negative.values().responses;
          for (let response = 0; response < a.length; ++response)
            for (let item = 0; item < a[response].values.length; ++item)
              near(gradients.responses[response].values[item * x.length + column],
                (a[response].values[item] - b[response].values[item]) / (2 * step), 2e-6);
        } finally { positive.delete(); negative.delete(); }
      }
      const weights = { 10: [1, -0.3], 11: [0.02], 12: [0.01] };
      if (extracted === 2) Object.assign(weights, { 20: [-0.2, 0.1], 21: [0.005], 22: [-0.02] });
      const hessian = point.hessianVector({ responses: weights }, [1, 0, 0, 0, 1, 0, 0, 0, 1], 3);
      assert.equal(hessian.report.usesModalTangents, true);
      assert.equal(hessian.report.perturbedLinearisations, 0);
      for (let column = 0; column < x.length; ++column) {
        const step = 1e-4 * x[column], plus = [...x], minus = [...x];
        plus[column] += step;
        minus[column] -= step;
        const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
        try {
          const a = positive.gradients().responses, b = negative.gradients().responses;
          for (let row = 0; row < x.length; ++row) {
            let expected = 0;
            for (let response = 0; response < a.length; ++response)
              for (let item = 0; item < a[response].itemCount; ++item)
                expected += weights[a[response].id][item] *
                  (a[response].values[item * 3 + row] - b[response].values[item * 3 + row]) / (2 * step);
            near(hessian.values[row * 3 + column], expected, 1e-5);
          }
        } finally { positive.delete(); negative.delete(); }
      }
    } finally { point.delete(); built.dispose(); }
  });

for (const coupled of [false, true]) test(`modal field continuum recovery coupled=${coupled}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;3.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nRod2;1;1;1;2\nRodProp;1;1;2.\nIsoMat;1;1000.;400.;0.25;1.\nParameter;CoupledMass;${coupled ? 1 : -1}\nLanczosEigenMethod;3;;;1\n`;
  const built = fixture(text, {
    variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
    relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [
      { id: 10, type: "Displacement", attributeA: 1, targets: [2], mode: 1 },
      ...["Stress", "Strain", "Force"].map((type, index) => ({ id: 11 + index, type,
        propertyType: "Elements", attributeA: 2, targets: [1], mode: 1 })),
    ],
  });
  const point = built.problem.evaluate();
  try {
    // Rod2 uses the mean of lumped and consistent axial mass when coupled: M_free = 5 rho A L / 12.
    const shape = Math.sqrt((coupled ? 12 / 5 : 2) / 6);
    const gradients = point.gradients();
    const expected = [-shape / 4, -1000 * shape / 12, -shape / 12, 1000 * shape / 6];
    gradients.responses.forEach((response, index) => near(response.values[0], expected[index]));
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.equal(gradients.report.analysisPerturbations, 0);
  } finally { point.delete(); built.dispose(); }
});

for (const norm of ["Mass", "Maximum"]) test(`analytic modal field Hessian chains shell von Mises ${norm}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;2.;0.;0.\nNode;3;;2.;1.;0.\nNode;4;;0.;1.;0.\nSinglePointConstrList;1;123456;1;4\nSinglePointConstrList;1;23456;2\nSinglePointConstrList;1;13456;3\nShellQuad4;1;1;1;2;3;4\nShellProp;1;1;0.2;1;;1\nIsoMat;1;1000.;400.;0.25;1.\nSpringDofInline;2;100.;1;1;2;1\nLanczosEigenMethod;3;;;2;;;;${norm}\n`;
  const x = [0.2, 100];
  const built = fixture(text, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: [{ id: 1, targetKind: "property", targetCard: "ShellProp", targetId: 1, field: "thickness",
      terms: [{ variableId: 1, coefficient: 1 }] },
    { id: 2, targetKind: "element", targetCard: "SpringDofInline", targetId: 2, field: "stiffness",
      terms: [{ variableId: 2, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Stress", propertyType: "Elements", attributeA: 9, targets: [1], mode: 1 }],
  });
  const point = built.problem.evaluate();
  try {
    const result = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2);
    assert.equal(result.report.usesModalTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
    for (let column = 0; column < x.length; ++column) {
      const step = 1e-4 * x[column], plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
      try {
        const a = positive.gradients().responses[0].values, b = negative.gradients().responses[0].values;
        for (let row = 0; row < x.length; ++row)
          near(result.values[row * 2 + column], (a[row] - b[row]) / (2 * step), 3e-5);
      } finally { positive.delete(); negative.delete(); }
    }
    near(result.values[1], result.values[2]);
  } finally { point.delete(); built.dispose(); }
});

for (const extracted of [1, 2]) test(`modal field rejects an individual derivative at a repeated root extracted=${extracted}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nNode;3;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2;3\nSpringDofInline;1;1000.;1;1;2;1\nSpringDofInline;2;1000.;1;1;3;1\nMassDofInline;3;2.;2;1;0;0\nMassDofInline;4;2.;3;1;0;0\nLanczosEigenMethod;3;;;2\n`;
  const modelText = text.replace("LanczosEigenMethod;3;;;2", `LanczosEigenMethod;3;;;${extracted}`);
  const built = fixture(modelText, {
    variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 }],
    relations: [{ id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1,
      field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2], mode: 1 }],
  });
  const point = built.problem.evaluate();
  try {
    assert.throws(() => point.gradients(), /repeated|singular/i);
    assert.throws(() => point.hessianVector({ responses: { 10: [1] } }, [1]), /repeated|singular/i);
  }
  finally { point.delete(); built.dispose(); }
});

test("modal field keeps an isolated root across widely separated stiffness scales", () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nNode;3;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2;3\nSpringDofInline;1;1000.;1;1;2;1\nSpringDofInline;2;1.e12;1;1;3;1\nMassDofInline;3;2.;2;1;0;0\nMassDofInline;4;2.;3;1;0;0\nLanczosEigenMethod;3;;;1\n`;
  const built = fixture(text, {
    variables: [{ id: 1, initialValue: 2, lower: 1, upper: 3 }],
    relations: [{ id: 1, targetKind: "element", targetCard: "MassDofInline", targetId: 3,
      field: "mass", terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2], mode: 1 }],
  });
  const point = built.problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 1 / Math.sqrt(2));
    near(point.gradients().responses[0].values[0], -0.5 / 2 ** 1.5);
    const result = point.hessianVector({ responses: { 10: [1] } }, [1]);
    assert.equal(result.report.usesModalTangents, true);
    near(result.values[0], 0.75 / 2 ** 2.5);
  } finally { point.delete(); built.dispose(); }
});

test("modal field rejects tied Maximum normalisation on a simple root", () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nNode;3;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2;3\nSpringDofInline;1;1000.;1;1;2;1\nSpringDofInline;2;1000.;1;1;3;1\nSpringDofInline;3;500.;2;1;3;1\nMassDofInline;4;2.;2;1;0;0\nMassDofInline;5;2.;3;1;0;0\nLanczosEigenMethod;3;;;2;;;;Maximum\n`;
  const built = fixture(text, {
    variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 }],
    relations: [{ id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1,
      field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2], mode: 1 }],
  });
  const point = built.problem.evaluate();
  try {
    assert.throws(() => point.gradients(), /tied MAX normalisation/);
    assert.throws(() => point.hessianVector({ responses: { 10: [1] } }, [1]), /tied MAX normalisation/);
  } finally { point.delete(); built.dispose(); }
});

test("modal field mode selection persists in model response cards", () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nSpringDofInline;1;1000.;1;1;2;1\nMassDofInline;2;2.;2;1;0;0\nLanczosEigenMethod;3;;;1\nDesignResp1;10;PHI;Displacement;;;1;1;2\n`;
  const built = fixture(text, {
    variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 }],
    relations: [{ id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1,
      field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] }],
  }, true);
  const point = built.problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 1 / Math.sqrt(2));
    near(point.gradients().responses[0].values[0], 0);
  } finally { point.delete(); built.dispose(); }
});

for (const norm of ["Mass", "Maximum"]) test(`analytic modal Hessian eigenvalue and frequency ${norm}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2\nSpringDofInline;1;1000.;1;1;2;1\nMassDofInline;2;2.;2;1;0;0\nLanczosEigenMethod;3;;;1;;;;${norm}\n`;
  const built = fixture(text, {
    variables: [{ id: 1, initialValue: 1000, lower: 500, upper: 1500 },
      { id: 2, initialValue: 2, lower: 1, upper: 3 }],
    relations: [
      { id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1,
        field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] },
      { id: 2, targetKind: "element", targetCard: "MassDofInline", targetId: 2,
        field: "mass", terms: [{ variableId: 2, coefficient: 1 }] },
    ],
    responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }, { id: 11, type: "Frequency", attributeA: 1 }],
  });
  const point = built.problem.evaluate();
  try {
    const eigen = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2);
    [0, -0.25, -0.25, 250].forEach((expected, i) => near(eigen.values[i], expected));
    assert.equal(eigen.report.usesModalTangents, true);
    assert.equal(eigen.report.perturbedLinearisations, 0);
    assert.equal(eigen.report.factorisations, 1);
    assert.equal(eigen.report.solves, 2);
    const frequency = point.values().responses[1].values[0];
    const expected = [-frequency / (4 * 1000 ** 2), -frequency / (4 * 1000 * 2),
      -frequency / (4 * 1000 * 2), 3 * frequency / (4 * 2 ** 2)];
    const result = point.hessianVector({ responses: { 11: { rowStart: [0, 1], items: [0], values: [1] } } }, [1, 0, 0, 1], 2);
    expected.forEach((value, i) => near(result.values[i], value));
    assert.equal(result.report.usesModalTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
    const combined = point.hessianVector({ responses: { 10: [2], 11: [3] } }, [1, 0, 0, 1], 2);
    [0, -0.25, -0.25, 250].forEach((value, i) => near(combined.values[i], 2 * value + 3 * expected[i]));
    assert.equal(combined.report.factorisations, 1); // the two responses read the same root
  } finally { point.delete(); built.dispose(); }
});

for (const extracted of [1, 2]) test(`analytic modal Hessian coupled roots extracted=${extracted}`, () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nNode;1;;0.;0.;0.\nNode;2;;1.;0.;0.\nNode;3;;2.;0.;0.\nNode;4;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nSinglePointConstrList;1;23456;2;3\nRigidBodyStar;10;3;123456;4\nSpringDofInline;1;1000.;1;1;2;1\nSpringDofInline;2;500.;2;1;3;1\nMassDofInline;3;2.;2;1;0;0\nMassDofInline;4;3.;4;1;0;0\nLanczosEigenMethod;3;;;${extracted}\n`;
  const x = [1000, 500, 2];
  const built = fixture(text, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
    relations: [
      { id: 1, targetKind: "element", targetCard: "SpringDofInline", targetId: 1, field: "stiffness", terms: [{ variableId: 1, coefficient: 1 }] },
      { id: 2, targetKind: "element", targetCard: "SpringDofInline", targetId: 2, field: "stiffness", terms: [{ variableId: 2, coefficient: 1 }] },
      { id: 3, targetKind: "element", targetCard: "MassDofInline", targetId: 3, field: "mass", terms: [{ variableId: 3, coefficient: 1 }] },
    ],
    responses: Array.from({ length: extracted }, (_, index) => ({ id: 10 + index,
      type: "Eigenvalue", attributeA: index + 1, tracksMode: true })),
  });
  const point = built.problem.evaluate();
  try {
    const weights = Object.fromEntries(Array.from({ length: extracted }, (_, index) => [10 + index, [index === 0 ? 1 : -0.4]]));
    const result = point.hessianVector({ responses: weights }, [1, 0, 0, 0, 1, 0, 0, 0, 1], 3);
    assert.equal(result.report.usesModalTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
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
          near(result.values[row * 3 + column], expected, 2e-6);
        }
      } finally { positive.delete(); negative.delete(); }
    }
    for (let row = 0; row < 3; ++row)
      for (let column = 0; column < 3; ++column)
        near(result.values[row * 3 + column], result.values[column * 3 + row]);
  } finally { point.delete(); built.dispose(); }
});

test("analytic modal Hessian includes rigid mass offset curvature in rotated frames", () => {
  const text = `[executive]\nSolutionSequence;103\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]\nRectFrame2;5;;0.;0.;0.;0.;0.;1.;0.;1.;0.\nRectFrame2;6;;0.;0.;0.;1.;0.;0.;0.;1.;0.\nNode;1;;1.;2.;3.\nNode;2;;1.;2.;3.;6\nSinglePointConstrList;1;123456;1\nBushing;1;1;1;2;;;;;0\nBushProp;1;Stiffness;1000.;1300.;1700.;800.;1100.;1500.\nPointMassInertia;2;2;5;2.\nLanczosEigenMethod;3;;;2\n`;
  const x = [2, 0.2, -0.3, 0.4];
  const modelText = text.replace("PointMassInertia;2;2;5;2.",
    "PointMassInertia;2;2;5;2.;;;;1.;0.07;1.4;-0.03;0.05;1.8");
  const built = fixture(modelText, {
    variables: x.map((value, index) => ({ id: index + 1, initialValue: value, lower: value - 1, upper: value + 1 })),
    relations: [
      ...x.map((value, index) => ({ id: index + 1, targetKind: "element", targetCard: "PointMassInertia", targetId: 2,
        field: index === 0 ? "mass" : "offset", fieldItem: index === 0 ? -1 : index - 1,
        terms: [{ variableId: index + 1, coefficient: 1 }] })),
    ],
    responses: [{ id: 10, type: "Eigenvalue", attributeA: 1 }, { id: 11, type: "Frequency", attributeA: 2 }],
  });
  const point = built.problem.evaluate();
  try {
    const directions = Array.from({ length: 16 }, (_, index) => index % 5 === 0 ? 1 : 0);
    const result = point.hessianVector({ responses: { 10: [1], 11: [0.3] } }, directions, 4);
    assert.equal(result.report.usesModalTangents, true);
    assert.equal(result.report.perturbedLinearisations, 0);
    for (let column = 0; column < x.length; ++column) {
      const step = 1e-4, plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
      try {
        const a = positive.gradients().responses, b = negative.gradients().responses;
        for (let row = 0; row < x.length; ++row) {
          const expected = (a[0].values[row] - b[0].values[row] + 0.3 * (a[1].values[row] - b[1].values[row])) / (2 * step);
          near(result.values[row * 4 + column], expected, 3e-5);
        }
      } finally { positive.delete(); negative.delete(); }
    }
    for (let row = 0; row < 4; ++row)
      for (let column = 0; column < 4; ++column)
        near(result.values[row * 4 + column], result.values[column * 4 + row], 1e-6);
  } finally { point.delete(); built.dispose(); }
});
