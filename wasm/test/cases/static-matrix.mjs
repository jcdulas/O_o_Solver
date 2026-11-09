import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver, info } from "../helpers/runtime.mjs";
import { product, label, lineModel, withModel, displacement, near, assertFinite,
  LINE_ELEMENTS, MASS_SETS } from "../helpers/static-models.mjs";
import { SURFACE_ELEMENTS, SOLID_ELEMENTS, surfaceModel, solidModel } from "../helpers/surface-models.mjs";

// Exhaustive Cartesian product of the declared finite domains. No random sampling or discarded combinations.
export const LINE_AXES = {
  element: LINE_ELEMENTS,
  load: ["force", "gravity", "thermal", "deformation"],
  nsm: MASS_SETS,
  coupled: [0, 1],
  weight: [0.5, 1],
  constraint: ["list", "union"],
  scope: ["local", "global"],
};
let count = 0;
for (const values of product(LINE_AXES)) {
  ++count;
  test(`line matrix ${label(values)}`, () => {
    const { text, expected } = lineModel(values);
    withModel(text, model => {
      const support = model.checkSupport("statics");
      assert.equal(support.isComplete, true, JSON.stringify(support.refusals));
      const run = solver.run(model, "statics");
      assertFinite(run);
      near(displacement(run), expected, "independent axial solution");
    });
  });
}
test("line matrix contains every declared Cartesian combination exactly once", () => {
  assert.equal(count, Object.values(LINE_AXES).reduce((n, values) => n * values.length, 1));
  assert.equal(count, 1600);
});

for (const values of product({ element: LINE_ELEMENTS, nsm: MASS_SETS, propertyMass: [0, 1],
  load: ["gravity", "acceleration"], sign: [-1, 0, 1] }))
  test(`mass and load boundaries ${label(values)}`, () => {
    const { text, expected } = lineModel(values);
    withModel(text, model => {
      const run = solver.run(model, "statics");
      assertFinite(run);
      near(displacement(run), expected);
    });
  });

for (const values of product({ element: LINE_ELEMENTS, load: ["force1", "thermal", "deformation"],
  constraint: ["point"], sign: [-1, 1], modulus: [500, 2000], length: [1, 6] }))
  test(`point constraints and scaling ${label(values)}`, () => {
    const { text, expected } = lineModel(values);
    withModel(text, model => near(displacement(solver.run(model, "statics")), expected));
  });

for (const threads of [1, 2, 4, 8].filter(n => n <= info.maximumThreads))
  for (const values of product({ element: LINE_ELEMENTS, nsm: MASS_SETS }))
    test(`selected masses at ${threads} threads ${label(values)}`, () => {
      const { text, expected } = lineModel({ ...values, load: "gravity" });
      withModel(text, model => near(displacement(solver.run(model, "statics", { threads })), expected));
    });

for (const values of product({ element: SURFACE_ELEMENTS, load: ["force", "gravity"], nsm: MASS_SETS,
  coupled: [0, 1], weight: [0.5, 1] }))
  test(`surface matrix ${label(values)}`, () => {
    const built = surfaceModel(values);
    const run = withModel(built.text, model => solver.run(model, "statics"));
    assertFinite(run);
    if (values.load === "force") {
      built.points.forEach(([x], i) => near(displacement(run, i + 1), 0.01 * x, "constant membrane stress"));
    } else {
      const reference = surfaceModel({ ...values, nsm: "none", density: built.densityEquivalent });
      const expected = withModel(reference.text, model => solver.run(model, "statics"));
      built.points.forEach((_, i) => near(displacement(run, i + 1), displacement(expected, i + 1), "equivalent area mass"));
    }
  });

for (const values of product({ element: SOLID_ELEMENTS, order: ["linear", "quadratic"],
  sign: [-1, 1], modulus: [500, 2000], coupled: [0, 1] }))
  test(`solid patch ${label(values)}`, () => {
    const built = solidModel(values);
    withModel(built.text, model => {
      const run = solver.run(model, "statics");
      assertFinite(run);
      built.points.forEach((p, i) => near(displacement(run, i + 1, built.axis),
        values.sign * 10 / values.modulus * p[built.axis], "constant solid stress"));
    });
  });
