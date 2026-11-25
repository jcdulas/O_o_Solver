import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { oor, solver, test } from "./support.mjs";

const wing = readFileSync(new URL("../fixtures/trim-rbe3-wing.oos", import.meta.url), "utf8");
const thickness = "DesignVar;1;T;0.1;0.05;0.2\nDesignPropRelation1;1;ShellProp;1;thickness;;;;1;1.";
const cases = [
  ["thickness", thickness, 0.1, 1e-5],
  ["dynamic pressure", "DesignVar;1;Q;10.;5.;15.\nDesignElemRelation1;1;AeroTrim;5;dynamicPressure;;;;1;1.", 10, 1e-3],
  ["leg Y", "DesignVar;1;Y;0.;-0.1;0.1\nDesignNodeMove;1;4;0;1.;0.;1.;0.", 0, 2e-5],
  ["attachment X", "DesignVar;1;X;0.;-0.1;0.1\nDesignNodeMove;1;6;0;1.;1.;0.;0.", 0, 2e-5],
];

function withProblem(text, run, modelOptions = {}) {
  oor.writeFile("/work/rbe3-wing.oos", text + "\n");
  const deck = oor.readModel("/work/rbe3-wing.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck, {}, { analyses: { aeroelastic: [1], buckling: [2] } }, modelOptions);
    try { return run(problem); } finally { problem.delete(); }
  } finally { deck.delete(); }
}

function at(problem, x, run) {
  const point = problem.evaluate([x]);
  try { return run(point); } finally { point.delete(); }
}

const values = point => point.values().responses.map(row => row.values[0]);

function centralDifference(evaluate, x, step) {
  const low = evaluate(x - step), high = evaluate(x + step);
  return high.map((value, i) => (value - low[i]) / (2 * step));
}

function near(actual, expected, tolerance = 3e-5) {
  assert.ok(Math.abs(actual - expected) < tolerance * Math.max(1e-5, Math.abs(expected)),
    actual + " differs from " + expected);
}

for (const [name, cards, x, step] of cases) {
  test("RBE3 trim and buckling: " + name, () => withProblem(wing + cards, problem => {
    const expected = centralDifference(value => at(problem, value, values), x, step);
    at(problem, x, point => {
      const times = [];
      let jacobian;
      for (let i = 0; i < 6; ++i) {
        const start = performance.now();
        jacobian = point.gradients();
        if (i > 0) times.push(performance.now() - start);
      }
      const rates = jacobian.responses.map(row => row.values[0]);
      rates.forEach((value, i) => near(value, expected[i]));
      assert.equal(jacobian.report.analysisPerturbations, 0);
      assert.equal(jacobian.report.factorisations, 0);
      if (name === "attachment X") {
        assert.equal(jacobian.report.shapePerturbations, 0);
        for (const [i, id] of [4, 5].entries()) {
          const xyz = point.nodalCoordinateGradient(id, [6]);
          assert.deepEqual(Array.from(xyz.nodeIds), [6]);
          assert.equal(xyz.values.length, 3);
          assert.equal(xyz.report.analysisPerturbations, 0);
          near(xyz.values[0], expected[i]);
        }
        const jvp = point.jvp([1]).values;
        jvp.forEach((value, i) => near(value, expected[i]));
        const adjoint = point.vjp({ responses: { 4: [0.4], 5: [0.01] } }).values;
        near(adjoint[0], 0.4 * expected[0] + 0.01 * expected[1]);
        assert.throws(() => point.nodalCoordinateGradient(4, [6, 6]), /duplicate/);
      }
      console.log(JSON.stringify({ name, rates, medianMs: times.sort((a, b) => a - b)[2], report: jacobian.report }));
    });
  }));
}

test("the XYZ gradient contracts with a physical move at the current thickness", () => {
  // Move the physical coordinates in new decks so the reference also rebuilds the RBE3.
  const nodes = [6, 4], moves = [[0.3, -0.71, 0.42], [-0.1, 0.2, 0.33]];
  const evaluate = (step, read) => {
    const moved = wing.replace(/^Node;(\d+);;([^\n]+)/gm, (line, id, fields) => {
      const row = nodes.indexOf(Number(id));
      if (row === -1) return line;
      const coords = fields.split(";").map((value, axis) => Number(value) + step * moves[row][axis]);
      return "Node;" + id + ";;" + coords.join(";");
    });
    return withProblem(moved + thickness, problem => at(problem, 0.12, read));
  };
  const expected = centralDifference(step => evaluate(step, values), 0, 1e-5);
  evaluate(0, point => {
    for (const [i, id] of [4, 5].entries()) {
      const xyz = point.nodalCoordinateGradient(id, nodes);
      assert.deepEqual(Array.from(xyz.nodeIds), nodes);
      assert.equal(xyz.report.analysisPerturbations, 0);
      const contraction = xyz.values.reduce((sum, rate, j) => sum + rate * moves[Math.floor(j / 3)][j % 3], 0);
      near(contraction, expected[i]);
    }
  });
});

for (const [kind, flexibility] of [["bending", "-1.;-1."], ["bending and torsion", "0.01;0.02"]]) {
  test("beam spline transports loads and motion with " + kind, () => {
    const beam = wing.replace("AeroSurfaceSpline1;60;100;100;103;50",
      "RectFrame2;8;;0.;0.;0.;0.;0.;1.;1.;0.3;0.\nAeroBeamSpline2;60;100;100;103;50;0.01;1.2;8;" + flexibility + ";Both");
    withProblem(beam + "DesignVar;1;Y;0.;-0.1;0.1\nDesignNodeMove;1;6;0;1.;0.;1.;0.", problem => {
      const expected = centralDifference(value => at(problem, value, values), 0, 1e-5);
      at(problem, 0, point => {
        const gradient = point.gradients();
        assert.equal(gradient.report.analysisPerturbations, 0);
        gradient.responses.forEach((row, i) => {
          assert.ok(Math.abs(expected[i]) > 1e-7);
          near(row.values[0], expected[i], 2e-4);
        });
      });
    });
  });
}


test("moving an isolated RBE3 spline attachment on a mixed shell reuses the trim", () => {
  const cards = "DesignVar;1;X;0.;-0.1;0.1\nDesignNodeMove;1;6;0;1.;1.;0.;0.";
  withProblem(wing + cards, problem => {
    const expected = centralDifference(value => at(problem, value, values), 0, 1e-5);
    at(problem, 0, point => {
      const gradient = point.gradients();
      assert.equal(gradient.report.analysisPerturbations, 0);
      for (const [i, id] of [4, 5].entries()) {
        near(gradient.responses[i].values[0], expected[i]);
        const xyz = point.nodalCoordinateGradient(id, [6]);
        assert.equal(xyz.report.analysisPerturbations, 0);
        near(xyz.values[0], expected[i]);
      }
    });
  }, { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} });
});
