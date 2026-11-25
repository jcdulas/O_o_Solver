import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { wing, column } from "../helpers/optimization-models.mjs";
import { oor, solver, test } from "./support.mjs";

test("Trim box, hinge, buckling factor and mode gradients follow the re-equilibrated wing", () => {
  const text = readFileSync(new URL("../fixtures/trim-preloaded-wing.oos", import.meta.url), "utf8");
  oor.writeFile("/work/trim-preloaded-wing.oos", text);
  const deck = oor.readModel("/work/trim-preloaded-wing.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck, {}, { analyses: { aeroelastic: [1], buckling: [2] } });
    try {
      const h = 1e-5;
      const p = problem.evaluate(), plus = problem.evaluate([0.1 + h]), minus = problem.evaluate([0.1 - h]);
      try {
        const gradient = p.gradients(), a = plus.values().responses, b = minus.values().responses;
        assert.equal(gradient.report.analysisPerturbations, 0);
        assert.equal(gradient.responses.length, 7);
        const jvp = p.jvp([1]).values;
        for (const [i, row] of gradient.responses.entries()) {
          const value = rows => rows.find(r => r.id === row.id).values[0];
          const check = (value(a) - value(b)) / (2 * h);
          assert.ok(Math.abs(check) > 1e-8 && Math.abs(row.values[0] - check) < 2e-4 * Math.abs(check),
            "response " + row.id + ": tangent " + row.values[0] + ", reanalysis " + check);
          assert.ok(Math.abs(jvp[i] - check) < 2e-4 * Math.abs(check));
          const adjoint = p.vjp({ responses: { [row.id]: [1] } }).values[0];
          assert.ok(Math.abs(adjoint - check) < 2e-4 * Math.abs(check));
        }
        const weights = { 1: [1], 2: [0.5], 3: [-2], 4: [1], 7: [0.125] };
        const hv = p.hessianVector({ responses: weights }, [1]);
        const ga = plus.gradients().responses, gb = minus.gradients().responses;
        const check = Object.entries(weights).reduce((sum, [id, [weight]]) =>
          sum + weight * (ga.find(r => r.id === +id).values[0] - gb.find(r => r.id === +id).values[0]) / (2 * h), 0);
        assert.equal(hv.report.usesTrimTangents, true);
        const buckling = p.hessianVector({ responses: { 5: [1] } }, [1]);
        const bucklingCheck = (ga.find(row => row.id === 5).values[0] - gb.find(row => row.id === 5).values[0]) / (2*h);
        assert.equal(buckling.report.usesBucklingTangents, true);
        assert.equal(buckling.report.usesTrimTangents, true);
        assert.equal(buckling.report.perturbedLinearisations, 0);
        assert.ok(Math.abs(buckling.values[0] - bucklingCheck) < 2e-5 * Math.abs(bucklingCheck));
        assert.ok(Math.abs(hv.values[0] - check) < 1e-5 * Math.abs(check));
      } finally { minus.delete(); plus.delete(); p.delete(); }
    } finally { problem.delete(); }
  } finally { deck.delete(); }
});

test("Dynamic pressure differentiates the trim balance, dimensional loads and buckling preload", () => {
  const source = readFileSync(new URL("../fixtures/trim-preloaded-wing.oos", import.meta.url), "utf8") +
    "DesignVar;2;Q;10.;5.;15.\nDesignElemRelation1;2;AeroTrim;5;dynamicPressure;;;;2;1.\n" +
    "DesignResp1;8;CZ;StabilityDerivative;;;70;1;3\n";
  oor.writeFile("/work/trim-q.oos", source);
  const deck = oor.readModel("/work/trim-q.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck, {}, { analyses: { aeroelastic: [1], buckling: [2] } });
    try {
      const point = problem.evaluate(), plus = problem.evaluate([0.1, 10.001]), minus = problem.evaluate([0.1, 9.999]);
      try {
        const jac = point.gradients();
        assert.equal(jac.report.analysisPerturbations, 0);
        const values = p => p.values().responses.map(row => row.values[0]);
        const a = values(plus), b = values(minus), jvp = point.jvp([0, 1]).values;
        for (const [i, row] of jac.responses.entries()) {
          const expected = (a[i] - b[i]) / 0.002;
          assert.ok(Math.abs(expected) > 1e-10 && Math.abs(row.values[1] - expected) < 2e-4 * Math.abs(expected),
            "response " + row.id + ": dq " + row.values[1] + ", reanalysis " + expected);
          assert.ok(Math.abs(jvp[i] - expected) < 2e-4 * Math.abs(expected));
        }
      } finally { minus.delete(); plus.delete(); point.delete(); }
    } finally { problem.delete(); }
  } finally { deck.delete(); }
});

test("Dynamic pressure drives free trim acceleration and unrestrained stability derivatives", () => {
  const wingDeck = wing({ load: true });
  let text;
  try {
    wingDeck.caseControl.global.remove("SinglePointConstrSelection");
    wingDeck.addCard({ type: "PointMassInertia", fields: [2, 3, 0, 2, 0, 0.2, 0, 0.1, 0, 0.1, 0, 0, 0.1] });
    for (const [i, label] of ["RigidAccelX", "RigidAccelY", "RigidAccelZ", "RigidAngularAccelX",
      "RigidAngularAccelY", "RigidAngularAccelZ"].entries())
      wingDeck.addCard({ type: "AeroRigidTrimVariable", fields: [71 + i, label] });
    text = wingDeck.toModelText().replace(/SinglePointConstrList;[^\n]+/g, "") + "\nFreeBodySupport;1;123;2;23;4;3\n";
  } finally { wingDeck.delete(); }
  text += "DesignVar;1;Q;10.;5.;15.\nDesignElemRelation1;1;AeroTrim;5;dynamicPressure;;;;1;1.\n" +
    "DesignResp1;1;AZ;TrimVariable;;;73\nDesignResp1;2;CZ;StabilityDerivative;;;70;0;3\n" +
    "DesignResp1;3;P;AeroPressure;;;101\n";
  compareScalarDesign(text, { aeroelastic: [1] }, 10, 0.001);
});

test("A mode component differentiates the static preload and stiffness of a stepped column", () => {
  const original = column();
  let text;
  try {
    text = original.toModelText().replace("Bar2;2;1;", "Bar2;2;2;") +
      "\nBarProp;2;2;0.12;0.015;0.03;0.012\nIsoMat;2;1200.;;0.3;1.\n" +
      "DesignVar;1;E;1200.;600.;1800.\nDesignMatRelation1;1;IsoMat;2;youngsModulus;;;;1;1.\n" +
      "DesignResp1;1;PHI;Displacement;Buckling;;2;1;2\n";
  } finally { original.delete(); }
  compareScalarDesign(text, { buckling: [2] }, 1200, 0.05);
});

function compareScalarDesign(text, analyses, x, h) {
  oor.writeFile("/work/trim-mode-validation.oos", text);
  const deck = oor.readModel("/work/trim-mode-validation.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck, {}, { analyses });
    try {
      const point = problem.evaluate([x]), plus = problem.evaluate([x + h]), minus = problem.evaluate([x - h]);
      try {
        const result = point.gradients();
        assert.equal(result.report.analysisPerturbations, 0);
        const a = plus.values().responses, b = minus.values().responses;
        for (const [i, row] of result.responses.entries()) {
          const expected = (a[i].values[0] - b[i].values[0]) / (2 * h);
          assert.ok(Math.abs(expected) > 1e-11 && Math.abs(row.values[0] - expected) < 2e-4 * Math.abs(expected),
            "response " + row.id + ": tangent " + row.values[0] + ", reanalysis " + expected);
        }
      } finally { minus.delete(); plus.delete(); point.delete(); }
    } finally { problem.delete(); }
  } finally { deck.delete(); }
}
