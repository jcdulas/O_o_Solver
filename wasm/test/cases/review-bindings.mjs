// The JavaScript interface at its edges: what an archive of a study holds of the grid point forces, the option
// objects that name a key the interface does not have, and the wrappers of a native design problem after delete().
import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { axial, column } from "../helpers/optimization-models.mjs";

const near = (actual, expected, tolerance = 2e-5) => assert.ok(Number.isFinite(actual) &&
  Math.abs(actual - expected) <= tolerance * Math.max(1e-8, Math.abs(expected)), `${actual} differs from ${expected}`);
const tip = { id: 10, source: "nodal", kind: "displacements", subcase: 1, ids: [2], components: "t1" };
const definition = (extra = {}) => ({ version: 1,
  variables: [{ id: 1, initialValue: 2, lower: 0.5, upper: 5 }],
  parameters: [{ variableId: 1, target: { kind: "card", type: "RodProp", id: 1, field: "area" } }],
  analyses: [{ analysis: "statics" }], responses: [tip], objective: { responseId: 10 }, ...extra });
// The message of a refused key names the key and the keys its level has.
const refuses = (work, key, accepted) => assert.throws(work, error => {
  assert.ok(error instanceof TypeError, `expected a TypeError, found ${error}`);
  assert.match(error.message, new RegExp(`unknown key '${key}'`));
  assert.ok(error.message.includes(accepted), `the message does not list ${accepted}: ${error.message}`);
  return true;
});

test("a study point with grid point forces is saved under the names of the result catalogue", () => {
  const deck = axial();
  deck.caseControl.global.set("NodeForceOutput", "All");
  const study = solver.createStudy(oor, deck, definition({ responses: [tip,
    { id: 11, source: "gridPointForce", subcase: 1, ids: [2], rowNames: ["appliedLoad"], components: "t1" }] }));
  const point = study.evaluate();
  try {
    const run = point.analysisResults()[0];
    assert.equal(run.gridPointForces.length, 1, "the request gives one table of grid point forces");
    near(point.values().responses.find(response => response.id === 11).values[0], 10);
    assert.equal(point.save("/work/review-node-forces.ooh5").finished, true);
    const archive = oor.openResults("/work/review-node-forces.ooh5");
    try {
      const tables = archive.tables();
      const forces = tables.find(table => table.kind === "nodePointForces");
      assert.ok(forces, `no nodePointForces table among ${tables.map(table => table.kind)}`);
      assert.deepEqual(forces.components, ["f1", "f2", "f3", "m1", "m2", "m3"]);
      const read = archive.read(forces.index);
      assert.equal(read.ids.length, run.gridPointForces[0].nodeIds.length);
      assert.deepEqual(Array.from(read.values), Array.from(run.gridPointForces[0].values));
      assert.deepEqual(read.texts.elementName,
        Array.from(run.gridPointForces[0].nameIndices, index => run.gridPointForces[0].names[index]));
    } finally { archive.delete(); }
  } finally { point.delete(); study.delete(); deck.delete(); }
});

test("modelFromReader refuses an option it does not have", () => {
  const deck = axial(), { card, formulations } = solver.elementFormulations()[0];
  try {
    // `formulation` is the keyword of the Python interface: here it would leave ClassicFormulation in force.
    refuses(() => solver.modelFromReader(oor, deck, { formulation: "MixedHybridFormulation" }), "formulation",
      "elementFormulation");
    refuses(() => solver.modelFromReader(oor, deck, { formulationGeometry: { smoothEdge: [[1, 2]] } }), "smoothEdge",
      "smoothEdges");
    assert.throws(() => solver.modelFromReader(oor, deck, { condensation: "all" }), /true or false/);
    for (const options of [undefined, {}, { elementFormulation: "ClassicFormulation" },
      { elementFormulations: { [card]: formulations[0] }, formulationGeometry: {}, condensation: false },
      { formulationGeometry: null }])
      solver.modelFromReader(oor, deck, options).delete();
  } finally { deck.delete(); }
});

test("run refuses a setting it does not have and keeps every setting it has", () => {
  const deck = axial(), model = solver.modelFromReader(oor, deck);
  try {
    refuses(() => solver.run(model, "statics", { subcase: 1 }), "subcase", "subcases");
    refuses(() => solver.run(model, "statics", { condensation: { unit: [1] } }), "unit", "units");
    refuses(() => solver.runSuperelements(model, [], "statics", { streamed: true }), "streamed", "subcases");
    const run = solver.run(model, "statics", { subcases: [1], memoryBytes: undefined, threads: 1,
      condensation: undefined, streamed: false, externalUnits: [], envelopes: [], keepsCaseTables: true });
    assert.ok(Array.isArray(run.nodalTables) && Array.isArray(run.envelopes));
    assert.equal(run.diagnostics.filter(diagnostic => diagnostic.severity === "error").length, 0);
  } finally { model.delete(); deck.delete(); }
});

// The report of a run: its threads and kernels, the size of its problem, its stages in the order it runs them, and a
// wall time that covers them.
function checkReport(report, { dofs, stages, factorisations }) {
  assert.ok(Number.isInteger(report.threads) && report.threads >= 1, `threads ${report.threads}`);
  assert.ok(["reference", "avx2", "avx512", "wasm128"].includes(report.kernel), `kernel ${report.kernel}`);
  assert.equal(report.dofs, dofs);
  assert.ok(report.solvedDofs >= 1 && report.solvedDofs <= report.dofs, `solvedDofs ${report.solvedDofs}`);
  assert.ok(report.factorisations >= factorisations, `factorisations ${report.factorisations}`);
  assert.deepEqual(Object.keys(report.stages), stages);
  const staged = Object.values(report.stages).reduce((sum, seconds) => sum + seconds, 0);
  assert.ok(Object.values(report.stages).every(seconds => seconds >= 0), "a stage has a negative time");
  assert.ok(staged <= report.seconds + 1e-6, `the stages (${staged} s) exceed the run (${report.seconds} s)`);
  assert.ok(report.peakBudgetBytes > 0, "the budget held nothing");
  assert.ok(Array.isArray(report.laminateShear), "the laminates of the report are a list");
}

test("a run reports its threads, kernels, sizes, stage times and budget memory", () => {
  const rod = axial(), rodModel = solver.modelFromReader(oor, rod);
  const bars = column(), barModel = solver.modelFromReader(oor, bars);
  try {
    checkReport(solver.run(rodModel, "statics").report, { dofs: 12, factorisations: 1,
      stages: ["setup", "assembly", "loadBasis", "factorisation", "solve", "recovery"] });
    assert.equal(solver.run(rodModel, "statics", { threads: 1 }).report.threads, 1);
    checkReport(solver.run(barModel, "buckling").report, { dofs: 18, factorisations: 2,
      stages: ["preload", "assembly", "eigenvalues", "recovery"] });
  } finally { rodModel.delete(); rod.delete(); barModel.delete(); bars.delete(); }
});

test("a study definition refuses a key of any level it does not have", () => {
  const deck = axial();
  const create = extra => solver.createStudy(oor, deck, definition(extra)).delete();
  try {
    refuses(() => create({ constraint: [] }), "constraint", "constraints");
    refuses(() => create({ variables: [{ id: 1, initialValue: 2, lower: 0.5, upper: 5, bounds: [0, 1] }] }), "bounds",
      "lower");
    // A misspelt `components` would select the six components of the row.
    refuses(() => create({ responses: [{ ...tip, components: undefined, component: "t1" }] }), "component",
      "components");
    refuses(() => create({ parameters: [{ variable: 1, target: { kind: "card", type: "RodProp", id: 1, field: "area" } }] }),
      "variable", "variableId");
    refuses(() => create({ parameters: [{ variableId: 1,
      target: { kind: "card", type: "RodProp", id: 1, fieldName: "area" } }] }), "fieldName", "field");
    refuses(() => create({ parameters: [{ variableId: 1,
      target: { kind: "case", command: "StaticLoadSelection", subcases: 1 } }] }), "subcases", "subcase");
    refuses(() => create({ analyses: [{ analysis: "statics", subcase: 1 }] }), "subcase", "subcases");
    refuses(() => create({ objective: { responseId: 10, direction: "min" } }), "direction", "sense");
    refuses(() => create({ constraints: [{ responseId: 10, max: 1 }] }), "max", "upper");
    create({ constraints: [{ responseId: 10, items: [0], lower: 0, upper: 1, scale: 1 }],
      objective: { responseId: 10, item: 0, sense: "min" } });
  } finally { deck.delete(); }
});

test("a deleted design problem says so, and its numerical gradients count their analyses", () => {
  const deck = axial();
  let point;
  const problem = solver.optimizationFromReader(oor, deck, {
    variables: [{ id: 1, initialValue: 2, lower: 0.5, upper: 5 }],
    relations: [{ id: 1, targetKind: "property", targetCard: "RodProp", targetId: 1, field: "area",
      terms: [{ variableId: 1, coefficient: 1 }] }],
    responses: [{ id: 10, type: "Displacement", attributeA: 1, targets: [2] }],
    constraints: [{ responseId: 10, upper: 0.02 }, { responseId: 10, lower: 0.0005 }],
    objective: { responseId: 10, subcase: 1 } });
  try {
    point = problem.evaluate([2]);
    // The rate of each normalised constraint with its response: 1/|bound| above the floor of the scale, the floor
    // (0.001) below it, negative for a lower bound.
    const scales = point.values().constraints.map(constraint => [constraint.side, constraint.derivativeScale]);
    assert.equal(scales.length, 2);
    near(scales.find(([side]) => side === "upper")[1], 1 / 0.02);
    near(scales.find(([side]) => side === "lower")[1], -1 / 0.001);
    const report = point.gradients({ method: "finiteDifference" }).report;
    assert.equal(report.method, "finiteDifference");
    assert.equal(report.analysisPerturbations, report.evaluations);
    assert.ok(report.analysisPerturbations > 0);
  } finally { point?.delete(); problem.delete(); deck.delete(); }
  assert.throws(() => problem.metadata(), /has been deleted/);
  assert.throws(() => problem.designPoint([2]), /has been deleted/);
  assert.throws(() => problem.evaluate([2]), /has been deleted/);
});
