import assert from "node:assert/strict";
import { test } from "node:test";
import { differenceStencil, directionalDifference, numericalJacobian, checkPoint } from "../js/numerical-derivatives.js";

const variables = [1, 2].map(id => ({ id, lower: 0, upper: 10, initialValue: id }));
const evaluate = ([a, b]) => ({ keys: ["product", "squares"], values: Float64Array.of(a * b, a * a + 3 * b * b) });
const near = (a, b) => assert.ok(Math.abs(a - b) < 2e-6 * Math.max(1, Math.abs(b)), `${a} differs from ${b}`);

test("full-analysis differences give the Jacobian and directional derivative of coupled responses", { timeout: 1000 }, () => {
  const x = Float64Array.of(2, 3);
  const matrix = numericalJacobian(evaluate, x, variables);
  [3, 2, 4, 18].forEach((value, i) => near(matrix.values[i], value));
  assert.equal(matrix.report.evaluations, 4);
  const direction = directionalDifference(evaluate, x, variables, [2, -1]);
  [4, -10].forEach((value, i) => near(direction.values[i], value));
  assert.equal(direction.evaluations, 2);
});

test("second-order one-sided stencils stay within bounds and preserve derivatives", { timeout: 1000 }, () => {
  for (const x of [Float64Array.of(0, 10), Float64Array.of(10, 0)]) {
    const calls = [];
    const bounded = point => { calls.push(point); checkPoint(point, variables); return evaluate(point); };
    const matrix = numericalJacobian(bounded, x, variables);
    [x[1], x[0], 2 * x[0], 6 * x[1]].forEach((value, i) => near(matrix.values[i], value));
    assert.ok(calls.every(point => point.every(value => value >= 0 && value <= 10)));
  }
});

test("a forward product gives a one-sided derivative at a non-smooth point", { timeout: 1000 }, () => {
  const variable = [{ id: 1, lower: -1, upper: 1 }];
  const absolute = ([x]) => ({ keys: ["abs"], values: Float64Array.of(Math.abs(x)) });
  near(directionalDifference(absolute, [0], variable, [-1], { side: "forward" }).values[0], 1);
});

test("fixed and discrete variables have explicit derivative contracts", { timeout: 1000 }, () => {
  const fixed = [{ id: 1, lower: 2, upper: 2 }, variables[1]];
  const matrix = numericalJacobian(evaluate, [2, 3], fixed);
  assert.equal(matrix.values[0], 0); assert.equal(matrix.values[2], 0);
  assert.throws(() => differenceStencil([2, 3], [{ ...variables[0], kind: "integer" }, variables[1]], [1, 0]), /discrete/);
  assert.throws(() => checkPoint([1.5, 3], [{ ...variables[0], kind: "integer" }, variables[1]]), /integer/);
  assert.equal(directionalDifference(evaluate, [2, 3], variables, [0, 0]).evaluations, 0);
});

test("layout changes and oversized outputs fail before a misleading Jacobian is returned", { timeout: 1000 }, () => {
  const base = evaluate([2, 3]);
  assert.throws(() => numericalJacobian(x => ({ ...evaluate(x), keys: ["changed", "squares"] }), [2, 3], variables, {}, base), /layout/);
  assert.throws(() => numericalJacobian(evaluate, [2, 3], variables, { maximumOutputBytes: 8 }, base), /maximumOutputBytes/);
  const blocks = [];
  const result = numericalJacobian(evaluate, [2, 3], variables, { maximumOutputBytes: 8, onBlock: block => blocks.push(block) }, base);
  assert.equal(result.values, undefined); assert.equal(blocks.length, 2);
  assert.throws(() => differenceStencil([2, 3], variables, [1, 0], { relativeStep: -1 }), /steps/);
});
