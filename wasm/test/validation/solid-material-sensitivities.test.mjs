// First-order material gradients checked against independent reanalyses of the same design.
import assert from "node:assert/strict";
import { oor, solver, test } from "./support.mjs";
import { solidModel } from "../helpers/surface-models.mjs";

function solidMaterialTest(element, elementFormulation, field,
  { derivedPoisson = false, zeroDensity = false, anisotropic = false, order = "linear" } = {}) {
  const tip = { SolidTetra: 2, SolidHexa: 2, SolidPenta: 4, SolidPyramid: 5 }[element];
  const axis = element === "SolidPyramid" ? 3 : 1;
  const x = { youngsModulus: 1000, poissonRatio: 0.23, density: zeroDensity ? 0 : 2 }[field];
  let { text } = solidModel({ element, order, sign: 1, modulus: 1000, coupled: 1 });
  text = text.replace("IsoMat;1;1000.;;0.;1.", derivedPoisson ? "IsoMat;1;1000.;400.;;2."
    : `IsoMat;1;1000.;;0.23;${zeroDensity ? 0 : 2}.`);
  if (field === "density") text = text.replace(/^PointForceLoad;.*$/gm, "") +
    `GravityLoad;2;;10.;${axis === 1 ? "1.;0.;0." : "0.;0.;1."}\n`;
  if (anisotropic) text = text.replace(/^IsoMat;1;.*$/m,
    "AnisoMat3D;1;1300.;230.;170.;0.;0.;0.;1100.;190.;0.;0.;0.;900.;0.;0.;0.;370.;0.;0.;400.;0.;310.;2.");
  const material = anisotropic ? "AnisoMat3D" : "IsoMat";
  text += `DesignVar;1;X;${x};${x * 0.5};${zeroDensity ? 1 : x * 1.5}\nDesignMatRelation1;1;${material};1;${field};;;;1;1.\n` +
    `DesignResp1;10;U;Displacement;;;${axis};;${tip}\n`;
  if (field === "density")
    text += "DesignResp1;11;MASS;Weight\nDesignResp1;12;JYY;Weight;;;5;5\nDesignResp1;13;VOL;Volume\n";
  if (element === "SolidHexa" && field === "poissonRatio" && order === "linear")
    text = text.replace("Node;7;;1.;1.;1.", "Node;7;;1.12;0.97;1.08");
  oor.writeFile("/work/solid-material.oos", text);
  const deck = oor.readModel("/work/solid-material.oos", { strict: true });
  const problem = solver.optimizationFromReader(oor, deck, {}, {}, { elementFormulation, formulationGeometry: {} });
  try {
    const point = problem.evaluate([x]);
    try {
      const result = point.gradients();
      const actual = result.responses[0].values[0];
      const h = 1e-4 * Math.max(1, x);
      const plus = problem.evaluate([x + h]), minus = problem.evaluate([zeroDensity ? x : x - h]);
      try {
        const displacement = p => p.values().responses[0].values[0];
        const expected = (displacement(plus) - displacement(minus)) / (zeroDensity ? h : 2 * h);
        assert.ok(Math.abs(expected) > 1e-12 && Math.abs(actual - expected) < 2e-6 * Math.abs(expected),
          `gradient ${actual}, independent solves ${expected}`);
        for (const response of result.responses.slice(1)) {
          const value = p => p.values().responses.find(r => r.id === response.id).values[0];
          const numerical = (value(plus) - value(minus)) / (zeroDensity ? h : 2 * h);
          assert.ok(Math.abs(response.values[0] - numerical) <= 1e-7 * Math.max(1e-8, Math.abs(numerical)));
        }
      } finally { minus.delete(); plus.delete(); }
      assert.equal(result.report.fieldPerturbations, derivedPoisson ? 1 : 0);
      assert.equal(result.report.analysisPerturbations, 0);
      if (element === "SolidHexa" && field === "density") {
        // The unit cube has mass rho and Jyy = rho * integral(x*x + z*z) = 2*rho/3 about the origin.
        const rate = id => result.responses.find(response => response.id === id).values[0];
        assert.ok(Math.abs(rate(11) - 1) < 1e-10);
        assert.ok(Math.abs(rate(12) - 2 / 3) < 1e-10);
        assert.equal(rate(13), 0);
      }
      if (element === "SolidTetra" && !zeroDensity) {
        assert.ok(Math.abs(point.jvp([1]).values[0] - actual) < 1e-10 * Math.abs(actual));
        assert.ok(Math.abs(point.vjp({ responses: { 10: [1] } }).values[0] - actual) < 1e-10 * Math.abs(actual));
      }
    } finally { point.delete(); }
  } finally { problem.delete(); deck.delete(); }
}

for (const formulation of ["ClassicFormulation", "MixedHybridFormulation"]) {
  for (const element of ["SolidTetra", "SolidHexa", "SolidPenta", "SolidPyramid"])
    for (const field of ["youngsModulus", "poissonRatio", "density"])
      test(`${formulation} ${element} analytic solid ${field}`, () => solidMaterialTest(element, formulation, field));
  test(`${formulation} solid derived Poisson ratio follows E at fixed G`, () =>
    solidMaterialTest("SolidTetra", formulation, "youngsModulus", { derivedPoisson: true }));
  test(`${formulation} solid unit-density mass at zero density`, () =>
    solidMaterialTest("SolidTetra", formulation, "density", { zeroDensity: true }));
  test(`${formulation} anisotropic solid density`, () =>
    solidMaterialTest("SolidHexa", formulation, "density", { anisotropic: true }));
  test(`${formulation} quadratic tetrahedron Poisson ratio`, () =>
    solidMaterialTest("SolidTetra", formulation, "poissonRatio", { order: "quadratic" }));
}

// A modal HEXA8 carries interior displacement fields in the mixed formulation. Its frequency
// and a static displacement can use the same material variable in a single optimisation.
for (const elementFormulation of ["ClassicFormulation", "MixedHybridFormulation"])
  for (const field of ["youngsModulus", "density", "poissonRatio"])
    test(`${elementFormulation} HEXA8 frequency derivative for ${field}`, () => {
      const x = { youngsModulus: 1000, density: 2, poissonRatio: 0.23 }[field];
      let { text } = solidModel({ element: "SolidHexa", order: "linear", sign: 1, modulus: 1000, coupled: 1 });
      text = text.replace("IsoMat;1;1000.;;0.;1.", "IsoMat;1;1000.;;0.23;2.")
        .replace("[model]", "SubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\n[model]");
      text += `LanczosEigenMethod;3;;;2\nDesignVar;1;X;${x};${x * 0.5};${x * 1.5}\n` +
        `DesignMatRelation1;1;IsoMat;1;${field};;;;1;1.\nDesignResp1;10;F;Frequency;;;1\n` +
        "DesignResp1;11;U;Displacement;;;1;;2\n";
      oor.writeFile("/work/solid-frequency.oos", text);
      const deck = oor.readModel("/work/solid-frequency.oos", { strict: true });
      const problem = solver.optimizationFromReader(oor, deck, {}, { analyses: { statics: [1], modes: [2] } },
        { elementFormulation, formulationGeometry: {} });
      try {
        const point = problem.evaluate([x]);
        try {
          const gradient = point.gradients();
          const check = point.gradients({ method: "finiteDifference" });
          for (const row of gradient.responses) {
            const reference = check.responses.find(r => r.id === row.id);
            for (let i = 0; i < row.values.length; i++)
              assert.ok(Math.abs(row.values[i] - reference.values[i]) <= 2e-5 * Math.max(1e-8, Math.abs(reference.values[i])),
                `response ${row.id}: ${row.values[i]}, independent analyses ${reference.values[i]}`);
          }
          if (field !== "poissonRatio" || elementFormulation === "ClassicFormulation")
            assert.equal(gradient.report.fieldPerturbations, 0);
          assert.equal(gradient.report.analysisPerturbations, 0);
        } finally { point.delete(); }
      } finally { problem.delete(); deck.delete(); }
    });

for (const elementFormulation of ["ClassicFormulation", "MixedHybridFormulation"])
  for (const element of ["SolidTetra", "SolidHexa", "SolidPenta", "SolidPyramid"])
    test(elementFormulation + " " + element + " geometric stiffness material tangents", () => {
      let { text } = solidModel({ element, order: "linear", sign: -1, modulus: 1000, coupled: 1 });
      text = text.replace("IsoMat;1;1000.;;0.;1.", "IsoMat;1;1000.;;0.23;1.")
        .replace("[model]", "SubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\nStaticSubcaseSelection;1\n[model]");
      // Break mesh symmetries so that the selected factor is simple.
      if (element === "SolidHexa") text = text.replace("Node;7;;1.;1.;1.", "Node;7;;1.12;0.97;1.08");
      text += "LanczosEigenMethod;3;0.001;;1\nDesignVar;1;E;1000.;500.;1500.\nDesignVar;2;NU;0.23;0.1;0.4\n" +
        "DesignMatRelation1;1;IsoMat;1;youngsModulus;;;;1;1.\nDesignMatRelation1;2;IsoMat;1;poissonRatio;;;;2;1.\n" +
        "DesignResp1;10;FACTOR;BucklingFactor;;;1\n";
      oor.writeFile("/work/solid-buckling.oos", text);
      const deck = oor.readModel("/work/solid-buckling.oos", { strict: true });
      const problem = solver.optimizationFromReader(oor, deck, {}, { analyses: { buckling: [2] } },
        { elementFormulation, formulationGeometry: {} });
      try {
        const point = problem.evaluate();
        try {
          const result = point.gradients();
          assert.equal(result.report.fieldPerturbations, 0);
          assert.equal(result.report.analysisPerturbations, 0);
          for (const [i, h] of [0.01, 1e-5].entries()) {
            const a = [1000, 0.23], b = [...a];
            a[i] += h; b[i] -= h;
            const plus = problem.evaluate(a), minus = problem.evaluate(b);
            try {
              const expected = (plus.values().responses[0].values[0] - minus.values().responses[0].values[0]) / (2 * h);
              assert.ok(Math.abs(result.responses[0].values[i] - expected) < 1e-5 * Math.max(1, Math.abs(expected)),
                "column " + i + ": " + result.responses[0].values[i] + ", reference " + expected);
            } finally { minus.delete(); plus.delete(); }
          }
        } finally { point.delete(); }
      } finally { problem.delete(); deck.delete(); }
    });
