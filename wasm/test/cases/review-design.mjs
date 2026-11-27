// Cases of the review of the design optimisation (src/design and its interface in src/api): each one guards a defect
// the review found, on a model of a few elements. The gradients are compared with the central differences of complete
// evaluations, the Hessian-vector products with the differences of the gradients.
import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

// The tolerances of the design cases of mixed-hybrid.mjs: a sizing gradient against the differences of the analysis,
// and a shape gradient (a small difference of large displacements at the step 1e-4).
const GRADIENT_TOLERANCE = 1e-5;
const SHAPE_GRADIENT_TOLERANCE = 1e-4;

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

// |actual − expected| within `tolerance` of the larger of |expected| and `scale` (1, as in buckling-derivatives.mjs;
// 0 for a relative comparison).
function near(actual, expected, tolerance = 3e-5, scale = 1) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(scale, Math.abs(expected)),
    `${actual} differs from ${expected}`);
}

// Every entry of `actual` against `expected`, each multiplied by the magnitude of its variable (`scales`), within
// `tolerance` of the largest of them: the entries of variables of very different magnitudes are then comparable.
function scaledNear(actual, expected, scales, tolerance, what) {
  assert.equal(actual.length, expected.length, what);
  const largest = Math.max(...expected.map((value, k) => Math.abs(value) * scales[k]));
  assert.ok(largest > 0, `${what}: the expected values are all zero`);
  expected.forEach((value, k) => assert.ok(Number.isFinite(actual[k]) &&
    Math.abs(actual[k] - value) * scales[k] <= tolerance * largest,
    `${what}, entry ${k}: ${actual[k]} differs from ${value}`));
}

// The text of a model file: its solution sequence, its Case Control records and the records of its model.
function deckText(solution, caseRecords, modelRecords) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseRecords, "[model]", ...modelRecords, ""]
    .join("\n");
}

const STATIC_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2"];
// Subcase 2 is the buckling of the preload of subcase 1.
const BUCKLING_CASE = [...STATIC_CASE, "SubcaseStart;2", "SinglePointConstrSelection;1", "EigenMethodSelection;3",
  "StaticSubcaseSelection;1"];
const MODES_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"];

function readDeck(text, name, strict = true) {
  const file = `/work/review-design-${name}.oos`;
  oor.writeFile(file, text);
  return strict ? oor.readModel(file, { strict: true }) : oor.readModel(file);
}

// The design problem of `declarations` (variables, relations, responses) on the model of `text`, whose design cards
// are not read. `dispose` releases the problem, the model and the deck.
function declared(text, name, declarations, settings = {}, strict = true) {
  const deck = readDeck(text, name, strict);
  let model = null;
  try {
    model = solver.modelFromReader(oor, deck);
    const problem = solver.createOptimization(model, declarations, settings);
    return { problem, dispose() { problem.delete(); model.delete(); deck.delete(); } };
  } catch (error) { model?.delete(); deck.delete(); throw error; }
}

// Runs `action(problem)` on the design problem of the design cards of `text` and of `declarations`, the model read
// with `modelOptions` (those of modelFromReader).
function withDeckDesign(text, name, declarations, settings, action, modelOptions = {}) {
  const deck = readDeck(text, name);
  try {
    const problem = solver.optimizationFromReader(oor, deck, declarations, settings, modelOptions);
    try { return action(problem); } finally { problem.delete(); }
  } finally { deck.delete(); }
}

// The relation that writes variable `variableId` in one field of a card.
const relation = (variableId, targetKind, targetCard, targetId, field) => ({ id: variableId, targetKind, targetCard,
  targetId, field, terms: [{ variableId, coefficient: 1 }] });

// The native gradient of every response (by each of `methods`; [undefined]: the default) against the central
// differences of the analysis (by default a step of 1e-4 of the value of each variable), relative to the largest of
// each response's gradient. The helper of the design cases of mixed-hybrid.mjs: responses of one item.
function assertGradients(problem, x, tolerance, steps = x.map((value) => 1e-4 * (Math.abs(value) || 1)),
  methods = [undefined]) {
  const valuesAt = (point) => point.values().responses.map((response) => response.values[0]);
  const point = problem.evaluate(x);
  try {
    const natives = methods.map((method) => point.gradients(method ? { method } : undefined));
    for (let v = 0; v < x.length; ++v) {
      const h = steps[v];
      const plus = problem.evaluate(x.map((value, k) => (k === v ? value + h : value)));
      const minus = problem.evaluate(x.map((value, k) => (k === v ? value - h : value)));
      try {
        const [up, down] = [valuesAt(plus), valuesAt(minus)];
        natives.forEach((native, m) => native.responses.forEach((response, r) => {
          const scale = Math.max(...response.values.map(Math.abs), 1e-12);
          const numerical = (up[r] - down[r]) / (2 * h);
          assert.ok(Math.abs(response.values[v] - numerical) <= tolerance * scale,
            `${methods[m] ?? "default"} method, response ${r}, variable ${v + 1}: native ${response.values[v]}, ` +
            `differences ${numerical}`);
        }));
      } finally { plus.delete(); minus.delete(); }
    }
  } finally { point.delete(); }
}

// Σ weights[id][item] · ∂r(id, item)/∂x at the design point x: the gradient whose differences a Hessian-vector
// product of these weights is compared with.
function weightedGradient(problem, x, weights) {
  const point = problem.evaluate(x);
  try {
    const sum = new Array(x.length).fill(0);
    for (const response of point.gradients().responses) {
      const row = weights[response.id];
      if (!row) continue;
      for (let item = 0; item < response.itemCount; ++item)
        for (let j = 0; j < x.length; ++j) sum[j] += row[item] * response.values[item * x.length + j];
    }
    return sum;
  } finally { point.delete(); }
}

// Column j of the Hessian of Σ weights · r at x: the central difference of its gradient, at a step of `relative` of
// |x_j| (each variable its own step).
function hessianColumn(problem, x, weights, j, relative = 1e-4) {
  const h = relative * Math.abs(x[j]);
  const at = (sign) => weightedGradient(problem, x.map((value, k) => (k === j ? value + sign * h : value)), weights);
  const [up, down] = [at(1), at(-1)];
  return up.map((value, k) => (value - down[k]) / (2 * h));
}

// A ShellQuad4 held along its edge x = 0 (grid points 1 and 4), its grid points 2 and 3 left with their translations
// along x and z and their rotation about y (the plate of buckling-derivatives.mjs), under a force along z on each;
// `property` its property and material records, `records` the other records of the model.
function plateDeck({ length = 2, width = 1, force = 1,
  property = ["ShellProp;1;1;0.2;1;;1", "IsoMat;1;1000.;;0.3;1."], records = [] } = {}) {
  return deckText(101, STATIC_CASE, ["Node;1;;0.;0.;0.", `Node;2;;${real(length)};0.;0.`,
    `Node;3;;${real(length)};${real(width)};0.`, `Node;4;;0.;${real(width)};0.`, "ShellQuad4;1;1;1;2;3;4", ...property,
    "SinglePointConstrList;1;123456;1;4", "SinglePointConstrList;1;246;2;3",
    `PointForceLoad;2;2;;${real(force)};0.;0.;1.`, `PointForceLoad;2;3;;${real(force)};0.;0.;1.`, ...records]);
}

// The rod of optimization.mjs (u = P·L/(E·A), L = 3, A = 2, E = 1000, ρ = 1) without its load, its two sizing
// variables (the area and the modulus) and the shape variable that moves its free end along its axis.
const ROD = ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "Rod2;1;1;1;2", "RodProp;1;1;2.", "IsoMat;1;1000.;;0.3;1.",
  "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;2"];
const ROD_SIZING = ["DesignVar;1;AREA;2.;0.5;5.", "DesignVar;2;MODULUS;1000.;500.;2000.",
  "DesignPropRelation1;1;RodProp;1;area;;;;1;1.", "DesignMatRelation1;2;IsoMat;1;youngsModulus;;;;2;1."];
const ROD_LENGTH = ["DesignVar;3;LENGTH;0.;-0.5;0.5", "DesignNodeMove;3;2;0;1.;1.;0.;0."];

// --- 1. An equation response that reads a buckling factor ------------------------------------------------------------

// The gradient of an equation h(λ) of a buckling factor kept the explicit part of dλ/dx only: the term through the
// displacements of the preload was lost, whichever method solved it. Expected: dh/dx = h'(λ)·dλ/dx, the gradient of
// the factor declared beside it, and the differences of the analysis.
//
// The model is the rod of buckling-derivatives.mjs, which shares its axial load with a spring: N_rod = P·(E·A/L) /
// (E·A/L + k) and λ = 20·(1 + k/(500·A)), so dλ/dA = −5 and dλ/dk = 0.02 at A = 2, k = 500. At fixed displacements
// the factor varies as 1/A and not at all with k (−15 and 0): neither variable is given by the explicit part.
const SHARED_LOAD_ROD = deckText(105, BUCKLING_CASE, ["Node;1;;0.;0.;0.", "Node;2;;2.;0.;0.",
  "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;456;2", "Rod2;1;1;1;2", "RodProp;1;1;2.",
  "IsoMat;1;1000.;400.;0.25;1.", "SpringDofInline;2;500.;1;1;2;1", "SpringDofInline;3;10.;1;2;2;2",
  "SpringDofInline;4;20.;1;3;2;3", "PointForceLoad;2;2;;1.;-1.;0.;0.", "LanczosEigenMethod;3;;;2"]);
const SHARED_LOAD_X = [2, 500];

// h(λ) = λ², with its derivative.
const squared = {
  evaluate({ arguments: [factor] }) { return { values: Array.from(factor.values, (value) => value * value) }; },
  evaluateWithPartials({ arguments: [factor] }) {
    return { values: Array.from(factor.values, (value) => value * value),
      partials: [Array.from(factor.values, (value) => 2 * value)] };
  },
};

// Response 10 is the first buckling factor, response 12 its square.
function factorEquation() {
  return declared(SHARED_LOAD_ROD, "factor-equation", {
    variables: SHARED_LOAD_X.map((value, i) => ({ id: i + 1, initialValue: value, lower: value / 2,
      upper: value * 2 })),
    relations: [relation(1, "property", "RodProp", 1, "area"),
      relation(2, "element", "SpringDofInline", 2, "stiffness")],
    responses: [{ id: 10, type: "BucklingFactor", attributeA: 1, tracksMode: true, subcases: [2] }],
    callables: { squared },
    equationResponses: [{ id: 12, label: "factor-squared", callable: "squared", subcases: [2],
      arguments: [{ kind: "DesignResp1", id: 10, name: "factor" }] }],
  }, { analyses: { buckling: [2] } });
}

for (const method of ["automatic", "direct", "adjoint", "chained"])
  test(`the gradient of an equation of a buckling factor includes its preload by the ${method} method`, () => {
    const built = factorEquation();
    const x = SHARED_LOAD_X;
    const point = built.problem.evaluate();
    try {
      const values = point.values().responses;
      const factor = values.find((response) => response.id === 10).values[0];
      near(factor, 30);
      near(values.find((response) => response.id === 12).values[0], factor * factor);
      const gradients = point.gradients({ method }).responses;
      const plain = gradients.find((response) => response.id === 10).values;
      const chained = gradients.find((response) => response.id === 12).values;
      near(plain[0], -5);
      near(plain[1], 0.02);
      for (let column = 0; column < x.length; ++column) {
        near(chained[column], 2 * factor * plain[column], 1e-4);
        const step = 1e-4 * x[column], plus = [...x], minus = [...x];
        plus[column] += step;
        minus[column] -= step;
        const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
        try {
          const square = (moved) => moved.values().responses.find((response) => response.id === 12).values[0];
          near(chained[column], (square(positive) - square(negative)) / (2 * step), 1e-4);
        } finally { positive.delete(); negative.delete(); }
      }
    } finally { point.delete(); built.dispose(); }
  });

// The same defect in the products: a vjp weighted on the equation, and the row of the equation in a jvp, are its
// gradient (the case above checks that gradient).
test("the products of an equation of a buckling factor are the rows of its gradient", () => {
  const built = factorEquation();
  const point = built.problem.evaluate();
  try {
    const row = point.values().responses.findIndex((response) => response.id === 12);
    const gradient = point.gradients().responses.find((response) => response.id === 12).values;
    const reverse = point.vjp({ rows: 1, responses: { 12: [1] } }).values;
    const forward = point.jvp([1, 0, 0, 1], { count: 2 }).values; // one row per response, one column per direction
    for (let column = 0; column < 2; ++column) {
      near(reverse[column], gradient[column], 1e-6);
      near(forward[row * 2 + column], gradient[column], 1e-6);
    }
  } finally { point.delete(); built.dispose(); }
});

// --- 2. The roots of a subcase that selects a non-structural mass set ------------------------------------------------

// The derivatives of the roots of a real eigenvalue subcase formed the mass of the elements without the
// non-structural mass set the subcase selects, which its analysis has. Expected: the gradients are those of the
// analysis, with the set, in a sizing variable and in a shape variable (which changes the length or the area the set
// is spread on).
//
// The rod with a lumped mass: m = (ρ·A + μ)·L/2 at its free end, μ = 2 the mass of the set per length (as much as
// the rod's own), so λ = 2·E·A / ((ρ·A + μ)·L²) = 1000/9.
const MASS_SET_CASE = [...MODES_CASE, "NonStructMassSelection;20"];
const MASS_SET_ROD = ["Parameter;CoupledMass;-1", "LanczosEigenMethod;3;;;1", "NonStructMass;20;RodProp;1;2."];
const ROOT_RESPONSES = [{ id: 30, type: "Frequency", attributeA: 1 }, { id: 31, type: "Eigenvalue", attributeA: 1 }];

test("the root gradients of a rod with a selected mass set are the analysis' in its area, modulus and length", () => {
  const text = deckText(103, MASS_SET_CASE, [...ROD, ...ROD_SIZING, ...ROD_LENGTH, ...MASS_SET_ROD]);
  withDeckDesign(text, "mass-set-rod", { responses: ROOT_RESPONSES }, { analyses: { modes: [1] } }, (problem) => {
    const x = [2, 1000, 0];
    assertGradients(problem, x, SHAPE_GRADIENT_TOLERANCE);
    const point = problem.evaluate(x);
    try {
      const root = point.values().responses.find((response) => response.id === 31).values[0];
      near(root, 1000 / 9, 1e-6);
      // dλ/dA = 2·E·μ / (L²·(ρ·A + μ)²), dλ/dE = λ/E, dλ/dL = −2·λ/L (−1.5·λ/L without the set in the mass rate).
      const gradient = point.gradients().responses.find((response) => response.id === 31).values;
      [2 * 1000 * 2 / (9 * 16), root / 1000, -2 * root / 3].forEach((expected, k) =>
        near(gradient[k], expected, 1e-4, 0));
    } finally { point.delete(); }
  });
});

// The same on a shell, whose set is a mass per area (as much as its own: ρ·t = 0.1): the membrane modes of the
// ShellQuad4 of static-inputs.mjs, in its thickness and in the move of its free edge along x.
test("the root gradients of a shell with a selected mass set are the analysis' in its thickness and length", () => {
  const text = deckText(103, MASS_SET_CASE, ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "Node;3;;3.;1.;0.",
    "Node;4;;0.;1.;0.", "ShellQuad4;1;1;1;2;3;4", "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.",
    "SinglePointConstrList;1;123456;1;4", "SinglePointConstrList;1;23456;2;3", "LanczosEigenMethod;3;;;1",
    "NonStructMass;20;ShellProp;1;0.1", "DesignVar;1;THICK;0.1;0.05;0.2",
    "DesignPropRelation1;1;ShellProp;1;thickness;;;;1;1.", "DesignVar;2;LENGTH;0.;-0.5;0.5",
    "DesignNodeMove;2;2;0;1.;1.;0.;0.", "DesignNodeMove;2;3;0;1.;1.;0.;0."]);
  withDeckDesign(text, "mass-set-shell", { responses: ROOT_RESPONSES }, { analyses: { modes: [1] } },
    (problem) => assertGradients(problem, [0.1, 0], SHAPE_GRADIENT_TOLERANCE));
});

// The pencil K − λ·M of the derivatives of a mode shape and of the second derivatives of a root had the same mass
// without the set: it was not the pencil of the modes. Expected: the gradient of a mode-selected displacement is
// given and is the analysis', and the Hessian-vector product of the frequency is the difference of its gradients.
test("a mode shape and the curvature of a frequency take the mass set their subcase selects", () => {
  const text = deckText(103, MASS_SET_CASE, [...ROD, ...ROD_SIZING, ...MASS_SET_ROD]);
  withDeckDesign(text, "mass-set-shape", { responses: [
    { id: 12, type: "Displacement", attributeA: 1, targets: [2], mode: 1 },
    { id: 30, type: "Frequency", attributeA: 1 }] }, { analyses: { modes: [1] } }, (problem) => {
    const x = [2, 1000];
    assertGradients(problem, x, GRADIENT_TOLERANCE);
    const point = problem.evaluate(x);
    try {
      // The mass-normalised shape of the free end is 1/√m, m = (ρ·A + μ)·L/2 = 6: its rate in A is −φ·ρ / (2·(ρ·A +
      // μ)) = −φ/8, and none in E.
      const shape = point.values().responses.find((response) => response.id === 12).values[0];
      near(Math.abs(shape), 1 / Math.sqrt(6), 1e-6);
      const rates = point.gradients().responses.find((response) => response.id === 12).values;
      near(rates[0], -shape / 8, 1e-5, 0);
      const weights = { 30: [1] };
      const product = point.hessianVector({ responses: weights }, [1, 0, 0, 1], 2);
      for (let column = 0; column < x.length; ++column)
        scaledNear(x.map((_, row) => product.values[row * x.length + column]),
          hessianColumn(problem, x, weights, column), x, 1e-4, `column ${column} of the Hessian of the frequency`);
    } finally { point.delete(); }
  });
});

// --- 3. The differenced Hessian-vector products ----------------------------------------------------------------------

// The products that difference the linearisation stepped every variable by 1e-4 of the largest |x_j| of the whole
// vector, and did not look at the bounds: beside a stiffness of 1e5, a thickness of 2 moved by 10. Expected: each
// moved variable steps by its own magnitude, within its bounds, on one side at a bound.
//
// A plate 10 × 10 and 2 thick (E = 7e6) bent by forces on its free edge, with a spring of 1e5 under grid point 2: the
// bending stiffness of the plate (some 1e5 too) is cubic in the thickness, so a step of several thicknesses gives a
// difference that is not the derivative. The weight on the volume has no closed second derivative: it sends the
// product to the differences (the volume itself is linear in the thickness and adds no curvature).
const STENCIL_X = [2, 1e5];
const STENCIL_WEIGHTS = { 10: [1], 11: [1e-3] };

function stencilProblem(lowestThickness) {
  const text = plateDeck({ length: 10, width: 10, force: 1000,
    property: ["ShellProp;1;1;2.;1;;1", "IsoMat;1;7.e6;;0.3;1."], records: ["SpringDofInline;2;1.e5;1;3;2;3"] });
  return declared(text, "stencil", {
    variables: [{ id: 1, initialValue: STENCIL_X[0], lower: lowestThickness, upper: 4 },
      { id: 2, initialValue: STENCIL_X[1], lower: 5e4, upper: 2e5 }],
    relations: [relation(1, "property", "ShellProp", 1, "thickness"),
      relation(2, "element", "SpringDofInline", 2, "stiffness")],
    responses: [{ id: 10, type: "Displacement", attributeA: 3, targets: [3] }, { id: 11, type: "Volume" }],
  });
}

function assertDifferenced(product) {
  assert.equal(product.report.hasHessianBlocks, false, "the product is expected to take the differences");
  assert.ok(product.report.perturbedLinearisations > 0, product.report.hessianBlocksReason);
}

test("a differenced Hessian-vector product steps each variable by its own magnitude", () => {
  const built = stencilProblem(1);
  const point = built.problem.evaluate();
  try {
    const columns = STENCIL_X.map((_, j) => hessianColumn(built.problem, STENCIL_X, STENCIL_WEIGHTS, j));
    for (const direction of [[1, 0], [1, 1]]) {
      const product = point.hessianVector({ responses: STENCIL_WEIGHTS }, direction);
      assertDifferenced(product);
      const expected = STENCIL_X.map((_, i) => direction[0] * columns[0][i] + direction[1] * columns[1][i]);
      scaledNear(product.values, expected, STENCIL_X, 1e-4, `direction ${direction}`);
    }
  } finally { point.delete(); built.dispose(); }
});

test("a differenced Hessian-vector product at a lower bound steps inside the bounds", () => {
  const built = stencilProblem(STENCIL_X[0]); // the thickness is on its lower bound
  const point = built.problem.evaluate();
  try {
    const product = point.hessianVector({ responses: STENCIL_WEIGHTS }, [1, 0]);
    assertDifferenced(product);
    // The difference of the gradients on the side of the bound that is allowed, of the second order:
    // (−3·g(t) + 4·g(t + h) − g(t + 2h)) / (2h).
    const h = 1e-4 * STENCIL_X[0];
    const [g0, g1, g2] = [0, 1, 2].map((k) =>
      weightedGradient(built.problem, [STENCIL_X[0] + k * h, STENCIL_X[1]], STENCIL_WEIGHTS));
    const expected = g0.map((value, i) => (-3 * value + 4 * g1[i] - g2[i]) / (2 * h));
    scaledNear(product.values, expected, STENCIL_X, 1e-3, "the thickness on its lower bound");
  } finally { point.delete(); built.dispose(); }
});

// --- 4. A shape variable of the mixed hybrid formulation under a distributed load ------------------------------------

// A shape variable of a model in MixedHybridFormulation under a pressure or a gravity threw ("expected a vector of N
// values"): the loads of the moved snapshots were not laid out as the formulation's unknowns. Expected: the gradient
// of the tip deflection is the analysis'.
const MIXED_HYBRID = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };

// The strip of the shape case of mixed-hybrid.mjs: three ShellQuad4 along x (1 long, 1 wide, 0.1 thick, ρ = 1), held
// at x = 0, under the records `loads` of load set 2, with one shape variable that moves grid points 5 and 6 out of
// the plane. The gradient is taken on the warped strip (0.05), where the reading of the surface is smooth.
function warpedStrip(loads) {
  const n = 3;
  const records = [];
  for (let i = 0; i <= n; ++i)
    for (let j = 0; j <= 1; ++j) records.push(`Node;${i * 2 + j + 1};;${real(i / n)};${real(j)};0.`);
  for (let i = 0; i < n; ++i) records.push(`ShellQuad4;${i + 1};1;${i * 2 + 1};${i * 2 + 3};${i * 2 + 4};${i * 2 + 2}`);
  records.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "SinglePointConstrList;1;123456;1;2", ...loads,
    "DesignVar;3;MOVE;0.;-0.1;0.1", "DesignNodeMove;3;5;0;1.;0.;0.;1.", "DesignNodeMove;3;6;0;1.;0.;0.;1.");
  return { tip: 2 * n + 1,
    text: deckText(101, [...STATIC_CASE, "DisplacementOutput;All", "StressOutput;All"], records) };
}

// The third load: forces at the three grid points 4, 6 and 8 of one side, which the formulation reads as a traction
// along the two edges between them (spec section 7). The variable moves grid point 6: the traction follows the
// edges, and the loads change with the shape as no force at a grid point does.
for (const [name, loads] of [["a pressure", ["ShellPressureLoad;2;1.;1;Through;3"]],
  ["its weight", ["GravityLoad;2;;10.;0.;0.;-1."]],
  ["forces along a side", ["PointForceLoad;2;4;;1.;0.;0.;1.", "PointForceLoad;2;6;;1.;0.;0.;1.",
    "PointForceLoad;2;8;;1.;0.;0.;0.5"]]])
  test(`the shape gradient of a strip in MixedHybridFormulation under ${name} is the analysis'`, () => {
    const strip = warpedStrip(loads);
    withDeckDesign(strip.text, "mixed-shape-load",
      { responses: [{ id: 10, type: "Displacement", attributeA: 3, targets: [strip.tip] }] }, {},
      (problem) => assertGradients(problem, [0.05], SHAPE_GRADIENT_TOLERANCE, [1e-4]), MIXED_HYBRID);
  });

// --- 5. The self-adjoint responses under inertia relief --------------------------------------------------------------

// The gradient of a compliance or of a mean displacement took the displacements as its adjoint under inertia relief
// too, where the loads move with the mass: it was wrong without a word. Expected: it is refused, naming the inertia
// relief. The model is the free triangle of optimization.mjs, with its thickness as the variable (a shape variable
// under relief is differenced on complete analyses, which refuse nothing).
const RELIEVED_TRIANGLE = deckText(101, ["SubcaseStart;1", "StaticLoadSelection;2"], ["Node;1;;0.;0.;0.",
  "Node;2;;1.;0.;0.", "Node;3;;0.;1.;0.", "ShellTria3;1;1;1;2;3", "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.",
  "FreeBodySupport;1;123;2;23;3;3", "Parameter;InertiaRelief;-1", "PointForceLoad;2;3;;1.;1.;0.;0.",
  "DesignVar;1;THICK;0.1;0.05;0.2", "DesignPropRelation1;1;ShellProp;1;thickness;;;;1;1."]);

for (const type of ["Compliance", "MeanDisplacement"])
  test(`the gradient of a ${type} response under inertia relief is refused`, () => {
    // The gradient of a displacement of the same model is given: the refusal is the response's, not the model's.
    withDeckDesign(RELIEVED_TRIANGLE, "relief",
      { responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [3] }] }, {}, (problem) => {
        const point = problem.evaluate();
        try {
          const rate = point.gradients().responses[0].values[0];
          assert.ok(Number.isFinite(rate) && rate !== 0, `the rate of the displacement is ${rate}`);
        } finally { point.delete(); }
      });
    assert.throws(() => withDeckDesign(RELIEVED_TRIANGLE, "relief", { responses: [{ id: 20, type }] }, {},
      (problem) => {
        const point = problem.evaluate();
        try { point.gradients(); } finally { point.delete(); }
      }), /inertia relief/i);
  });

// --- 6. A mass fraction of the elements of a property ----------------------------------------------------------------

// A mass fraction (FractionalMass) with properties as its targets selected its elements by their property ID alone,
// whatever the property card: the elements of a BarProp of the same ID as the ShellProp it names were counted too.
// Expected: the elements that reference that card and that ID. O_o_Model gives the property cards one ID space, so
// that a ShellProp and a BarProp of one ID cannot be read here: this case guards what the selection must give (the
// mass of the shell over its initial mass, whatever the bar does), not the collision itself.
test("a mass fraction of a property follows the elements of that property alone", () => {
  const text = plateDeck({ records: ["Bar2;2;2;1;2;0.;;1.;0.", "BarProp;2;1;0.1;0.01;0.02;0.01"] });
  const built = declared(text, "mass-fraction", {
    variables: [{ id: 1, initialValue: 0.2, lower: 0.1, upper: 0.4 },
      { id: 2, initialValue: 0.1, lower: 0.05, upper: 0.2 }],
    relations: [relation(1, "property", "ShellProp", 1, "thickness"), relation(2, "property", "BarProp", 2, "area")],
    responses: [{ id: 12, type: "FractionalMass", propertyType: "ShellProp", targets: [1] }],
  });
  const fractionAt = (x) => {
    const point = built.problem.evaluate(x);
    try { return point.values().responses[0].values[0]; } finally { point.delete(); }
  };
  try {
    near(fractionAt([0.2, 0.1]), 1, 1e-12);
    near(fractionAt([0.2, 0.2]), 1, 1e-12); // twice the area of the bar: not a mass of the selection
    near(fractionAt([0.3, 0.1]), 1.5, 1e-9); // the mass of the shell is proportional to its thickness
    const point = built.problem.evaluate();
    try {
      const gradient = point.gradients().responses[0].values;
      near(gradient[0], 1 / 0.2, 1e-5);
      near(gradient[1], 0, 1e-9);
    } finally { point.delete(); }
    assertGradients(built.problem, [0.2, 0.1], GRADIENT_TOLERANCE);
  } finally { built.dispose(); }
});

// --- 7. An objective that is one item of a response of several -------------------------------------------------------

// An objective response that is constrained in another subcase has an item in each: its gradient was left out (it was
// given for a response of one item only). Expected: the row of the item of the subcase of the objective.
test("an objective that is one of the items of its response has the gradient of that item", () => {
  // The rod under 10 in subcase 1 (u = 0.015, the objective) and under 20 in subcase 2 (u = 0.03, constrained).
  const text = deckText(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "SubcaseStart;2", "SinglePointConstrSelection;1", "StaticLoadSelection;3"],
  [...ROD, ...ROD_SIZING, "PointForceLoad;2;2;;10.;1.;0.;0.", "PointForceLoad;3;2;;20.;1.;0.;0."]);
  withDeckDesign(text, "objective-item", {
    responses: [{ id: 10, label: "tip", type: "Displacement", attributeA: 1, targets: [2] }],
    objective: { responseId: 10, subcase: 1 },
    constraints: [{ responseId: 10, upper: 0.05, subcases: [2] }],
  }, {}, (problem) => {
    const point = problem.evaluate();
    try {
      const values = point.values();
      const tip = values.responses.find((response) => response.id === 10);
      assert.deepEqual(tip.items.map((item) => item.subcase).sort(), [1, 2]);
      near(values.objective, 0.015, 1e-6, 0);
      assert.equal(values.constraints.length, 1);
      assert.equal(values.constraints[0].label.subcase, 2);
      const gradients = point.gradients();
      assert.ok(gradients.objective, "the objective has a gradient");
      assert.equal(gradients.objective.length, 2);
      const rows = gradients.responses.find((response) => response.id === 10).values;
      const item = tip.items.findIndex((entry) => entry.subcase === 1);
      const reverse = point.vjp(point.lagrangianWeights(1)).values;
      [-0.0075, -0.000015].forEach((expected, k) => {
        near(gradients.objective[k], expected, 1e-6, 0);
        near(gradients.objective[k], rows[item * 2 + k], 1e-12, 0);
        near(rows[(1 - item) * 2 + k], 2 * expected, 1e-6, 0); // the item of subcase 2 is another row
        near(reverse[k], expected, 1e-6, 0);
      });
    } finally { point.delete(); }
  });
});

// --- 8. Declarations that are refused --------------------------------------------------------------------------------

// An element response on a property that no element references had no item, and a constraint on it constrained
// nothing, without a word. Expected: the declaration is refused, naming the property.
test("an element response on a property that no element references is refused", () => {
  const text = plateDeck({ records: ["ShellProp;2;1;0.3;1;;1"] });
  const stressOf = (property) => ({
    variables: [{ id: 1, initialValue: 0.2, lower: 0.1, upper: 0.4 }],
    relations: [relation(1, "property", "ShellProp", 1, "thickness")],
    responses: [{ id: 11, type: "Stress", propertyType: "ShellProp", attributeA: 9, targets: [property] }],
  });
  // The property of the element is accepted: the refusal below is that of the property without an element.
  declared(text, "unused-property", stressOf(1)).dispose();
  assert.throws(() => {
    const refused = declared(text, "unused-property", stressOf(2));
    try { refused.problem.evaluate().delete(); } finally { refused.dispose(); }
  }, /no element/i);
});

// The lamina of a ply response (attributeB, a real) was truncated: 2.7 read ply 2. Expected: a lamina that is not an
// integer is refused.
test("a ply response whose lamina is not an integer is refused", () => {
  // A laminate of two plies, with the records of the laminated plates of the validation cases (read as they are).
  const text = plateDeck({ property: ["CompProp;1;;;;;;;;1;0.1;0.;Yes;1;0.1;90.;Yes",
    "OrthoMat2D;1;1.3e11;9.e9;0.3;5.e9;5.e9;3.e9;1600."] });
  const plyStress = (lamina) => ({ responses: [{ id: 11, type: "CompositeStress", propertyType: "Elements",
    attributeA: 3, attributeB: lamina, targets: [1] }] });
  // Ply 2 is accepted: the refusal below is that of the lamina.
  declared(text, "lamina", plyStress(2), {}, false).dispose();
  assert.throws(() => {
    const refused = declared(text, "lamina", plyStress(2.7), {}, false);
    try { refused.problem.evaluate().delete(); } finally { refused.dispose(); }
  }, /integer/i);
});

// --- 9. A Hessian-vector product weighted on a share of the strain energy --------------------------------------------

// The share of an element in the strain energy of the model (item 3 of a StrainEnergy response) has second
// derivatives that read the stiffness of every element, which neither the closed blocks nor the differenced
// linearisation hold: the product was wrong without a word. Expected: the products of the linearisation refuse the
// weight, naming the percentage; the interface may then difference the gradients instead, and the product it gives
// is then the curvature of the share. Never a wrong product.
//
// Two rods in a row carry the same force: the share of rod 1 is p = 100·A2 / (A1 + A2), 60 at A1 = 2, A2 = 3.
test("a Hessian-vector product weighted on a share of the strain energy is refused or is its curvature", () => {
  const text = deckText(101, STATIC_CASE, ["Node;1;;0.;0.;0.", "Node;2;;1.;0.;0.", "Node;3;;2.;0.;0.",
    "Rod2;1;1;1;2", "Rod2;2;2;2;3", "RodProp;1;1;2.", "RodProp;2;1;3.", "IsoMat;1;1000.;;0.3;1.",
    "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;2;3", "PointForceLoad;2;3;;10.;1.;0.;0."]);
  const built = declared(text, "energy-share", {
    variables: [{ id: 1, initialValue: 2, lower: 1, upper: 4 }, { id: 2, initialValue: 3, lower: 1.5, upper: 6 }],
    relations: [relation(1, "property", "RodProp", 1, "area"), relation(2, "property", "RodProp", 2, "area")],
    responses: [{ id: 10, type: "StrainEnergy", propertyType: "Elements", attributeA: 3, targets: [1] }],
  });
  const point = built.problem.evaluate();
  try {
    near(point.values().responses[0].values[0], 60, 1e-6);
    // dp/dA1 = −100·A2 / (A1 + A2)², dp/dA2 = 100·A1 / (A1 + A2)².
    const gradient = point.gradients().responses[0].values;
    near(gradient[0], -12, 1e-4);
    near(gradient[1], 8, 1e-4);
    let product = null;
    try { product = point.hessianVector({ responses: { 10: [1] } }, [1, 0, 0, 1], 2); }
    catch (error) { assert.match(error.message, /percentage/i); }
    if (product !== null) {
      assert.equal(product.report.hasHessianBlocks, false, "no closed block holds the energy of the whole model");
      // d²p/dA1² = 200·A2 / (A1 + A2)³, d²p/dA1dA2 = 100·(A2 − A1) / (A1 + A2)³, d²p/dA2² = −200·A1 / (A1 + A2)³.
      [4.8, 0.8, 0.8, -3.2].forEach((expected, k) => near(product.values[k], expected, 1e-4));
    }
  } finally { point.delete(); built.dispose(); }
});
