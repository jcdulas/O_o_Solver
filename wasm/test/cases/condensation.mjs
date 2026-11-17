import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { product, label, real } from "../helpers/static-models.mjs";

// The condensed path of a static analysis: the components of the reader condensed onto their boundaries, the
// interface problem solved, the units recovered. Its results are those of the full path to round-off.

// The relative difference below which two results are one (round-off of the condensation on these small models).
const SAME_RESULT = 1e-9;

// Runs `action(model)` on a model read with its component tables.
function withComponents(text, action) {
  oor.writeFile("/work/condensation.oos", text);
  const deck = oor.readModel("/work/condensation.oos", { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck, { condensation: true });
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

// The largest difference between the tables of a kind of two runs, against their largest value. The principal
// angles are left out: at a point of equal principal stresses (or no shear) round-off turns them by 180 degrees.
function relativeDifference(full, condensed, kind, tables = "nodalTables") {
  let largest = 0;
  let difference = 0;
  const fullTables = full[tables].filter((table) => table.kind === kind);
  assert.ok(fullTables.length > 0, `the full path gives ${kind}`);
  for (const table of fullTables) {
    const other = condensed[tables].find((candidate) => candidate.kind === kind && candidate.subcase === table.subcase &&
      (candidate.elementType ?? "") === (table.elementType ?? ""));
    assert.ok(other, `the condensed path gives ${kind} of subcase ${table.subcase}`);
    assert.deepEqual([...other.ids], [...table.ids], `the rows of ${kind}`);
    const width = table.components.length;
    for (let k = 0; k < table.values.length; ++k) {
      if (/angle/i.test(table.components[k % width])) continue;
      largest = Math.max(largest, Math.abs(table.values[k]));
      difference = Math.max(difference, Math.abs(table.values[k] - other.values[k]));
    }
  }
  return difference / Math.max(largest, Number.MIN_VALUE);
}

function assertSameResults(full, condensed) {
  assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
  for (const kind of ["displacements", "singlePointConstrForces"])
    assert.ok(relativeDifference(full, condensed, kind) < SAME_RESULT, kind);
  assert.ok(relativeDifference(full, condensed, "stress", "elementTables") < SAME_RESULT, "stress");
}

// Two square plates of n × n ShellQuad4, A on x in [0, 1] held along x = 0, B on x in [1.5, 2.5] loaded along
// x = 2.5, joined across the gap by `connection`. Two subcases: a transverse load and an in-plane load, the second
// with B's far corner held too (a point constrained in some subcases only). Rigid bars across the gap would make the
// drilling rotations of A's edge hinges unless the shells stiffen them: ShellQuad4Drilling with rigidBodyStar.
const PLATE_DIVISIONS = 3;
const CONNECTIONS = ["springs", "rigidBodyStar", "rigidInterpolation", "mpc"];

export function platePair(connection, n = PLATE_DIVISIONS) {
  const id = (plate, i, j) => plate * 1000 + i * (n + 1) + j + 1;
  const card = connection === "rigidBodyStar" ? "ShellQuad4Drilling" : "ShellQuad4";
  const bulk = [];
  for (const plate of [0, 1])
    for (let i = 0; i <= n; ++i)
      for (let j = 0; j <= n; ++j) bulk.push(`Node;${id(plate, i, j)};;${real(plate * 1.5 + i / n)};${real(j / n)};0.`);
  let element = 1;
  for (const plate of [0, 1])
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        bulk.push(`${card};${element++};1;${id(plate, i, j)};${id(plate, i + 1, j)};` +
          `${id(plate, i + 1, j + 1)};${id(plate, i, j + 1)}`);
  bulk.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.");
  const control = [];
  const pairs = Array.from({ length: n + 1 }, (_, j) => [id(0, n, j), id(1, 0, j)]);
  switch (connection) {
  case "springs": {
    let spring = 500;
    for (const [a, b] of pairs)
      for (let c = 1; c <= 5; ++c) bulk.push(`SpringDofInline;${spring++};1000.;${a};${c};${b};${c}`);
    break;
  }
  case "rigidBodyStar":
    pairs.forEach(([a, b], k) => bulk.push(`RigidBodyStar;${600 + k};${a};123456;${b}`));
    break;
  case "rigidInterpolation": {
    // A connection-only point between the plates: B's edge follows it rigidly, it follows the last two columns of A
    // on average.
    const legs = [...pairs.map(([a]) => a), ...pairs.map(([a]) => a - (n + 1))];
    bulk.push("Node;900;;1.25;0.5;0.", `RigidBodyStar;700;900;12345;${pairs.map(([, b]) => b).join(";")}`,
      `RigidInterpolation;701;900;123456;1;1.;123;${legs.length};${legs.join(";")}`);
    break;
  }
  case "mpc":
    control.push("MultiPointConstrSelection;10");
    for (const [a, b] of pairs)
      for (let c = 1; c <= 5; ++c) bulk.push(`MultiPointConstr;10;${b};${c};1.;${a};${c};-1.`);
    break;
  default:
    throw new Error(`unknown connection ${connection}`);
  }
  for (let j = 0; j <= n; ++j) bulk.push(`SinglePointConstrList;1;123456;${id(0, 0, j)}`);
  bulk.push(`SinglePointConstrList;3;123456;${pairs.map(([a]) => a - 3 * (n + 1)).join(";")}`,
    `SinglePointConstrList;3;3;${id(1, n, n)}`);
  for (let j = 0; j <= n; ++j) {
    bulk.push(`PointForceLoad;2;${id(1, n, j)};;1.;0.;0.;1.`);
    bulk.push(`PointForceLoad;4;${id(1, n, j)};;10.;1.;0.5;0.`);
  }
  const cases = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "SubcaseStart;2", "SinglePointConstrSelection;3", "StaticLoadSelection;4"];
  return ["[executive]", "SolutionSequence;101", "[case]", ...control, "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "StressOutput;All", ...cases, "[model]", ...bulk,
    "Parameter;AutoSingularityConstr;Yes", ""].join("\n");
}

for (const values of product({ connection: CONNECTIONS, units: ["all", "recommended", "listed"] }))
  test(`condensed statics of two plates ${label(values)}`, () => {
    withComponents(platePair(values.connection), (model) => {
      const full = solver.run(model, "statics");
      assert.equal(full.condensation, null, "a run without condensation reports none");
      const units = values.units === "listed" ? { units: [1], fullUnits: [] } : values.units;
      const condensed = solver.run(model, "statics", { condensation: units });
      assert.ok(condensed.condensation, "the summary of the condensation");
      if (values.units === "listed") assert.deepEqual(condensed.condensation.condensedUnits, [1]);
      else if (values.units === "all") assert.deepEqual(condensed.condensation.condensedUnits, [1, 2]);
      assert.ok(condensed.condensation.condensedUnits.length > 0 || values.units === "recommended");
      assertSameResults(full, condensed);
    });
  });

test("condensed statics keeps the listed full units full", () => {
  withComponents(platePair("springs"), (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: { units: "all", fullUnits: [2] } });
    assert.deepEqual(condensed.condensation.condensedUnits, [1]);
    assert.deepEqual(condensed.condensation.fullUnits, [2]);
    assertSameResults(full, condensed);
  });
});

test("condensed statics retains the asked points on the boundary", () => {
  withComponents(platePair("springs"), (model) => {
    const full = solver.run(model, "statics");
    const plain = solver.run(model, "statics", { condensation: "all" });
    const retained = solver.run(model, "statics", { condensation: { units: "all", retainedNodes: [6, 1006] } });
    assert.ok(retained.condensation.interfaceDofs > plain.condensation.interfaceDofs, "retained points join the interface");
    assertSameResults(full, retained);
  });
});

for (const sign of [-1, 1]) test(`condensed statics of a Fastener between two patches sign=${sign}`, () => {
  const points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1],
    [0.5, 0.5, 0.05], [0.5, 0.5, 0], [0.5, 0.5, 0.1]];
  const bulk = points.map(([x, y, z], i) => `Node;${i + 1};;${real(x)};${real(y)};${real(z)}`);
  bulk.push("ShellQuad4;1;1;1;2;3;4", "ShellQuad4;2;2;5;6;7;8", "ShellProp;1;1;0.1;1;;1",
    "ShellProp;2;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "Fastener;101;3;Elements;1;2;9;10;11",
    "FastenerProp;3;0.1;0;1;1000.;1000.;1000.;0.;0.;0.", "SinglePointConstrList;1;123456;1;2;3;4",
    "SinglePointConstrList;1;12456;5;6;7;8;9", `PointForceLoad;2;11;;${real(sign * 10)};0.;0.;1.`);
  const text = ["[case]", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "StressOutput;All", "[model]", ...bulk, ""].join("\n");
  withComponents(text, (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: "all" });
    // The lower patch is held at every grid point: nothing to condense, kept full.
    assert.deepEqual(condensed.condensation.condensedUnits, [2]);
    assert.deepEqual(condensed.condensation.fullUnits, [1]);
    assert.match(condensed.condensation.note, /unit 1 is kept full/);
    assertSameResults(full, condensed);
  });
});

test("condensed statics of one component takes the full path", () => {
  const text = ["[case]", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "[model]", "Node;1;;0.;0.;0.", "Node;2;;1.;0.;0.", "Node;3;;1.;1.;0.",
    "Node;4;;0.;1.;0.", "ShellQuad4;1;1;1;2;3;4", "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.",
    "SinglePointConstrList;1;123456;1;4", "PointForceLoad;2;2;;1.;0.;0.;1.", "Parameter;AutoSingularityConstr;Yes",
    ""].join("\n");
  withComponents(text, (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: "all" });
    assert.deepEqual(condensed.condensation.condensedUnits, []);
    assert.match(condensed.condensation.note, /one unit/);
    assert.ok(relativeDifference(full, condensed, "displacements") === 0, "the full path's results");
  });
});

test("condensed statics needs the component tables", () => {
  oor.writeFile("/work/condensation.oos", platePair("springs"));
  const deck = oor.readModel("/work/condensation.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck);
  try {
    assert.throws(() => solver.run(model, "statics", { condensation: "all" }), /component tables .*condensation: true/);
  } finally {
    model.delete();
    deck.delete();
  }
});

// Normal modes and buckling condense their units by Craig-Bampton (eigenPlatePair, CRAIG_BAMPTON_MODES below); the
// streamed path is a path of statics only.
for (const analysis of ["modes", "buckling"]) {
  test(`Craig-Bampton ${analysis} of two plates are those of the full path`, () => {
    withComponents(eigenPlatePair(analysis, PLATE_DIVISIONS), (model) => {
      const full = solver.run(model, analysis);
      const condensed = solver.run(model, analysis,
        { condensation: { units: "all", modeCount: CRAIG_BAMPTON_MODES } });
      assert.deepEqual(condensed.condensation.condensedUnits, [1, 2]);
      const difference = eigenvalueDifference(full, condensed);
      assert.ok(difference < CRAIG_BAMPTON_DIFFERENCE, `roots ${difference}`);
    });
  });
  test(`the streamed path is refused for ${analysis}`, () => {
    withComponents(eigenPlatePair(analysis, PLATE_DIVISIONS), (model) => {
      assert.throws(() => solver.run(model, analysis, { streamed: true }), /streamed path is offered for statics/);
    });
  });
}

test("condensation names an unknown unit", () => {
  withComponents(platePair("springs"), (model) => {
    assert.throws(() => solver.run(model, "statics", { condensation: { units: [7] } }), /unit 7/);
    assert.throws(() => solver.run(model, "statics", { condensation: "most" }), /expected "all"/);
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// MixedHybridFormulation (docs/mixed-hybrid-formulation-spec.md section 5.11): the units take the edges, faces and
// interiors of their elements, the formulation's relations and supports every subcase shares are eliminated inside
// them, the excluded rotations are filled after the recovery.

// The difference below which two results of the mixed formulation are one: its unknowns of several scales (moments of
// the edge displacements, rotations, shears) leave more round-off in the condensed stiffness than the classic shells.
const MIXED_SAME_RESULT = 1e-6;
// The divisions of each mixed plate (its elements carry many unknowns: two by two keeps a case within its budget).
const MIXED_DIVISIONS = 2;

// Two plates of n × n ShellQuad4 overlapping over a column of elements, B 0.1 above A, A held along x = 0, B loaded
// along its far edge and both by pressures, joined over the overlap by `connection` (two columns of points: no hinge).
// The second subcase holds B's far edge transversely (its edge unknowns held in that subcase only: on the boundary).
export function overlappingPlates(connection) {
  const n = MIXED_DIVISIONS;
  const id = (plate, i, j) => plate * 1000 + i * (n + 1) + j + 1;
  const origin = [0, 2 / 3];
  const bulk = [];
  for (const plate of [0, 1])
    for (let i = 0; i <= n; ++i)
      for (let j = 0; j <= n; ++j)
        bulk.push(`Node;${id(plate, i, j)};;${real(origin[plate] + i / n)};${real(j / n)};${plate ? "0.1" : "0."}`);
  let element = 1;
  for (const plate of [0, 1])
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        bulk.push(`ShellQuad4;${element++};1;${id(plate, i, j)};${id(plate, i + 1, j)};` +
          `${id(plate, i + 1, j + 1)};${id(plate, i, j + 1)}`);
  bulk.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.");
  const pairs = [];
  for (const i of [n - 1, n]) for (let j = 0; j <= n; ++j) pairs.push([id(0, i, j), id(1, i - (n - 1), j)]);
  const control = [];
  let spring = 500;
  switch (connection) {
  case "springs":
  case "rotationSprings":
    // Springs on the rotations make observer relations at their grid points (spec section 5.4).
    for (const [a, b] of pairs)
      for (let c = 1; c <= (connection === "springs" ? 3 : 5); ++c)
        bulk.push(`SpringDofInline;${spring++};1000.;${a};${c};${b};${c}`);
    break;
  case "mpc":
    control.push("MultiPointConstrSelection;10");
    for (const [a, b] of pairs)
      for (let c = 1; c <= 3; ++c) bulk.push(`MultiPointConstr;10;${b};${c};1.;${a};${c};-1.`);
    break;
  case "rigidBodyStar":
    // The corner pairs only: the edges of B between two dependent grid points follow them (spec section 5.6), each
    // by relations as long as the observer relation of the rotation of A's grid point they reach.
    pairs.filter((_, k) => k % (n + 1) !== 1)
      .forEach(([a, b], k) => bulk.push(`RigidBodyStar;${600 + k};${a};123;${b}`));
    break;
  case "fastener":
  case "fastenerGivenPoints":
    // A fastener reads the unknowns of the shells it pierces (spec section 5.10): they go on the boundaries.
    bulk.push("FastenerProp;3;0.1;0;1;1000.;1000.;1000.;100.;100.;100.");
    for (let j = 0; j < n; ++j) {
      const x = real(origin[1] + 0.5 / n);
      const y = real((j + 0.5) / n);
      bulk.push(`Node;${2000 + j};;${x};${y};0.05`);
      const [elementA, elementB] = [(n - 1) * n + 1 + j, n * n + 1 + j];
      if (connection === "fastener") {
        bulk.push(`Fastener;${700 + j};3;Elements;${elementA};${elementB};${2000 + j}`);
        continue;
      }
      bulk.push(`Node;${2100 + j};;${x};${y};0.`, `Node;${2200 + j};;${x};${y};0.1`,
        `Fastener;${700 + j};3;Elements;${elementA};${elementB};${2000 + j};${2100 + j};${2200 + j}`);
    }
    break;
  default:
    throw new Error(`unknown connection ${connection}`);
  }
  const heldEdge = Array.from({ length: n + 1 }, (_, j) => id(0, 0, j)).join(";");
  bulk.push(`SinglePointConstrList;1;123456;${heldEdge}`, `SinglePointConstrList;3;123456;${heldEdge}`,
    `SinglePointConstrList;3;3;${id(1, n, n - 1)};${id(1, n, n)}`);
  for (let j = 0; j <= n; ++j)
    bulk.push(`PointForceLoad;2;${id(1, n, j)};;1.;0.;0.;1.`, `PointForceLoad;4;${id(1, n, j)};;10.;1.;0.5;0.`);
  bulk.push(`ShellPressureLoad;2;0.5;${n * n + 1}`, `ShellPressureLoad;4;-0.3;${n * n + n}`);
  const cases = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "SubcaseStart;2", "SinglePointConstrSelection;3", "StaticLoadSelection;4"];
  return ["[executive]", "SolutionSequence;101", "[case]", ...control, "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "StressOutput;All", ...cases, "[model]", ...bulk,
    "Parameter;AutoSingularityConstr;Yes", ""].join("\n");
}

function withMixedComponents(text, action) {
  oor.writeFile("/work/condensation.oos", text);
  const deck = oor.readModel("/work/condensation.oos", { strict: true });
  let model;
  try {
    model = solver.modelFromReader(oor, deck, { condensation: true, elementFormulation: "MixedHybridFormulation",
      formulationGeometry: {} });
    return action(model);
  } finally {
    model?.delete();
    deck.delete();
  }
}

const MIXED_CONNECTIONS = ["springs", "rotationSprings", "mpc", "rigidBodyStar", "fastener", "fastenerGivenPoints"];
for (const connection of MIXED_CONNECTIONS)
  test(`condensed statics of two mixed hybrid plates connection=${connection}`, () => {
    withMixedComponents(overlappingPlates(connection), (model) => {
      const full = solver.run(model, "statics");
      const condensed = solver.run(model, "statics", { condensation: "all" });
      assert.deepEqual(condensed.condensation.condensedUnits, [1, 2]);
      assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
      for (const kind of ["displacements", "singlePointConstrForces"])
        assert.ok(relativeDifference(full, condensed, kind) < MIXED_SAME_RESULT, kind);
      assert.ok(relativeDifference(full, condensed, "stress", "elementTables") < MIXED_SAME_RESULT, "stress");
    });
  });

test("condensed statics of mixed hybrid plates keeps a listed unit full", () => {
  withMixedComponents(overlappingPlates("springs"), (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: { units: [2] } });
    assert.deepEqual(condensed.condensation.condensedUnits, [2]);
    assert.deepEqual(condensed.condensation.fullUnits, [1]);
    for (const kind of ["displacements", "singlePointConstrForces"])
      assert.ok(relativeDifference(full, condensed, kind) < MIXED_SAME_RESULT, kind);
  });
});

// The memory modes, the cache and the eigenvalue analyses of the condensed path on mixed hybrid plates.
test("condensed statics of mixed hybrid plates recovered in sequence", () => {
  withMixedComponents(overlappingPlates("springs"), (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: { units: "all", memoryMode: "low_memory" } });
    assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
    for (const kind of ["displacements", "singlePointConstrForces"])
      assert.ok(relativeDifference(full, condensed, kind) < MIXED_SAME_RESULT, kind);
    assert.ok(relativeDifference(full, condensed, "stress", "elementTables") < MIXED_SAME_RESULT, "stress");
  });
});

// The recovery unit by unit (memoryMode "low_memory", envelopes and no per-case table): each unit's recovery made once
// for every case, the interface's points recovered from the cases' vectors. The excluded rotation of a grid point of the
// interface and the stress of a grid point of solids there read the interior unknowns of the units' elements around it,
// which the units keep for the cases (CondensedStatics::listKeptDofs): the envelopes are the full path's.
function assertSameEnvelopes(full, condensed) {
  assert.ok(full.envelopes.length > 0, "the full path gives the envelopes");
  for (const table of full.envelopes) {
    const other = condensed.envelopes.find((candidate) => candidate.kind === table.kind &&
      candidate.quantity === table.quantity && candidate.elementType === table.elementType);
    const what = `${table.kind} ${table.elementType} ${table.quantity}`;
    assert.ok(other, `the condensed path gives the envelope of ${what}`);
    assert.deepEqual([...other.ids], [...table.ids], `the rows of ${what}`);
    assert.deepEqual([...other.nodeIds], [...table.nodeIds], `the grid points of ${what}`);
    const largest = Math.max(...Array.from(table.values, Math.abs), 1e-30);
    table.values.forEach((value, k) =>
      assert.ok(Math.abs(other.values[k] - value) < MIXED_SAME_RESULT * largest,
        `${what}, row ${k}: ${other.values[k]} and ${value}`));
  }
}

const UNIT_BY_UNIT = { condensation: { units: "all", memoryMode: "low_memory" }, keepsCaseTables: false };
const envelopesOf = (kind, quantities, mode) => quantities.map((quantity) => ({ kind, quantity, mode }));

for (const connection of ["springs", "rotationSprings", "fastener"])
  test(`condensed statics of mixed hybrid plates recovered unit by unit connection=${connection}`, () => {
    withMixedComponents(overlappingPlates(connection), (model) => {
      const envelopes = [...envelopesOf("displacements", ["t1", "t3", "r1", "r2", "r3"], "absMax"),
        ...envelopesOf("singlePointConstrForces", ["t3", "r2"], "absMax"), ...envelopesOf("stress", ["vonMises"], "max")];
      const full = solver.run(model, "statics", { envelopes, keepsCaseTables: false });
      const condensed = solver.run(model, "statics", { ...UNIT_BY_UNIT, envelopes });
      assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
      assert.equal(condensed.condensation.unitByUnit, true, "the units recovered unit by unit");
      assertSameEnvelopes(full, condensed);
    });
  });

// Two blocks of SolidHexa in MixedHybridFormulation, the second 0.5 beyond the first along x, joined by springs between
// their facing corners, the first held on its far face, the second loaded on its far face in two subcases.
function solidBlockPair() {
  const id = (block, i, j, k) => block * 1000 + (k * 2 + j) * 2 + i + 1;
  const bulk = [];
  const sharpEdges = [];
  for (const block of [0, 1]) {
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j)
        for (let i = 0; i <= 1; ++i) bulk.push(`Node;${id(block, i, j, k)};;${real(1.5 * block + i)};${real(j)};${real(k)}`);
    const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
    bulk.push(`SolidHexa;${block + 1};1;${corners.map(([i, j, k]) => id(block, i, j, k)).join(";")}`);
    // The twelve edges of the block are sharp.
    for (let t = 0; t <= 1; ++t)
      for (let u = 0; u <= 1; ++u)
        sharpEdges.push([id(block, 0, t, u), id(block, 1, t, u)], [id(block, t, 0, u), id(block, t, 1, u)],
          [id(block, t, u, 0), id(block, t, u, 1)]);
  }
  bulk.push("SolidProp;1;1", "IsoMat;1;1000.;;0.3;1.");
  let spring = 500;
  const held = [];
  for (let j = 0; j <= 1; ++j)
    for (let k = 0; k <= 1; ++k) {
      held.push(id(0, 0, j, k));
      for (let c = 1; c <= 3; ++c) bulk.push(`SpringDofInline;${spring++};100.;${id(0, 1, j, k)};${c};${id(1, 0, j, k)};${c}`);
      bulk.push(`PointForceLoad;2;${id(1, 1, j, k)};;1.;1.;0.;${real(0.2 * (j - k))}`,
        `PointForceLoad;4;${id(1, 1, j, k)};;1.;0.;${real(0.3 + 0.1 * j)};0.5`);
    }
  bulk.push(`SinglePointConstrList;1;123;${held.join(";")}`);
  const cases = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "SubcaseStart;2", "SinglePointConstrSelection;1", "StaticLoadSelection;4"];
  return { sharpEdges, text: ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "StressOutput;All", ...cases, "[model]", ...bulk, "Parameter;AutoSingularityConstr;Yes", ""].join("\n") };
}

test("condensed statics of mixed hybrid solids recovered unit by unit gives the grid point stresses", () => {
  const { text, sharpEdges } = solidBlockPair();
  oor.writeFile("/work/condensation.oos", text);
  const deck = oor.readModel("/work/condensation.oos", { strict: true });
  const model = solver.modelFromReader(oor, deck, { condensation: true, elementFormulation: "MixedHybridFormulation",
    formulationGeometry: { sharpEdges } });
  try {
    const envelopes = [...envelopesOf("displacements", ["t1", "t2", "t3"], "absMax"),
      ...envelopesOf("stress", ["oxx", "txy", "vonMises"], "max")];
    const full = solver.run(model, "statics", { envelopes, keepsCaseTables: false });
    const condensed = solver.run(model, "statics", { ...UNIT_BY_UNIT, envelopes });
    assert.equal(condensed.condensation.unitByUnit, true, "the units recovered unit by unit");
    assert.ok(full.envelopes.some((table) => table.elementType === "SolidGridPoints"), "the grid point stresses");
    assertSameEnvelopes(full, condensed);
  } finally {
    model.delete();
    deck.delete();
  }
});

// The plates free, joined by RigidBodyStar, under inertia relief on FreeBodySupport DOFs of the first plate: the condensed
// path solves its interface on six farthest translations and brings the displacements back to the gauge of the
// FreeBodySupport DOFs, the entity unknowns of the shells with the formulation's rigid fields (assembly::addRigidMotion).
function freePlatesUnderRelief() {
  return overlappingPlates("rigidBodyStar").replace(/SinglePointConstrSelection;[13]\n/g, "")
    .replace("[model]\n", "[model]\nParameter;InertiaRelief;-1\nFreeBodySupport;1;123;7;23;3;3\n");
}

test("condensed statics of free mixed hybrid plates under inertia relief is the full path's", () => {
  withMixedComponents(freePlatesUnderRelief(), (model) => {
    const full = solver.run(model, "statics");
    const condensed = solver.run(model, "statics", { condensation: "all" });
    assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
    assert.ok(relativeDifference(full, condensed, "displacements") < MIXED_SAME_RESULT, "displacements");
    assert.ok(relativeDifference(full, condensed, "stress", "elementTables") < MIXED_SAME_RESULT, "stress");
  });
});

test("condensed statics of free mixed hybrid plates under inertia relief recovered unit by unit", () => {
  withMixedComponents(freePlatesUnderRelief(), (model) => {
    const envelopes = [...envelopesOf("displacements", ["t1", "t3", "r1", "r2"], "absMax"),
      ...envelopesOf("stress", ["vonMises"], "max")];
    const full = solver.run(model, "statics", { envelopes, keepsCaseTables: false });
    const condensed = solver.run(model, "statics", { ...UNIT_BY_UNIT, envelopes });
    assert.equal(condensed.condensation.unitByUnit, true, "the units recovered unit by unit");
    assertSameEnvelopes(full, condensed);
  });
});

test("envelopes are asked of a static analysis only", () => {
  withMixedComponents(overlappingPlates("springs"), (model) => {
    assert.throws(() => solver.run(model, "modes", { envelopes: envelopesOf("displacements", ["t1"], "max") }),
      /envelopes are offered for statics only/);
  });
});

// The values of every table of a kind of two runs, compared to the bit.
function assertSameBits(first, second, kind, tables = "nodalTables") {
  const of = (run) => run[tables].filter((table) => table.kind === kind).flatMap((table) => [...table.values]);
  assert.deepEqual(of(second), of(first), kind);
}

test("condensed statics of mixed hybrid plates takes its units from the cache", () => {
  const text = overlappingPlates("springs");
  withMixedComponents(text, (model) => {
    const cache = solver.condensationCache();
    try {
      const settings = { condensation: { units: "all", cache } };
      const first = solver.run(model, "statics", settings);
      assert.equal(cache.statistics().misses, 2, "the first run condenses both units");
      const second = solver.run(model, "statics", settings);
      const statistics = cache.statistics();
      assert.equal(statistics.memoryHits, 2, JSON.stringify(statistics));
      assert.equal(statistics.misses, 2, JSON.stringify(statistics));
      for (const kind of ["displacements", "singlePointConstrForces"]) assertSameBits(first, second, kind);
      assertSameBits(first, second, "stress", "elementTables");
      // The same deck in ClassicFormulation: other matrices, other keys.
      withComponents(text, (classic) => {
        solver.run(classic, "statics", settings);
        assert.equal(cache.statistics().misses, 4, JSON.stringify(cache.statistics()));
      });
      // The same deck with another geometry declaration (a facet), which the reader's stamps do not see: the
      // declarations of the model enter the key of every unit, both are condensed again.
      oor.writeFile("/work/condensation.oos", text);
      const deck = oor.readModel("/work/condensation.oos", { strict: true });
      const faceted = solver.modelFromReader(oor, deck, { condensation: true,
        elementFormulation: "MixedHybridFormulation", formulationGeometry: { rawElements: [1] } });
      try {
        solver.run(faceted, "statics", settings);
        assert.equal(cache.statistics().misses, 6, JSON.stringify(cache.statistics()));
      } finally {
        faceted.delete();
        deck.delete();
      }
    } finally {
      cache.delete();
    }
  });
});

// The eigenvalues of the first subcase of two runs, against the largest.
function eigenvalueDifference(full, condensed) {
  const roots = (run) => run.eigenvalues[0].rows.map((row) => row.eigenvalue);
  const [a, b] = [roots(full), roots(condensed)];
  assert.equal(b.length, a.length, "the same count of roots");
  const largest = Math.max(...a.map(Math.abs));
  return Math.max(...a.map((value, k) => Math.abs(value - b[k]))) / largest;
}

// Craig-Bampton keeps the fixed-interface modes of each unit beside its boundary: with more modes than a unit has
// interior unknowns, every one is kept and the roots are those of the full path to the accuracy of the units' modes
// (3e-8 here; fewer modes leave a part of the root: 3e-5 with 40 modes on plates of 2 × 2 elements, whose edges carry
// many unknowns).
const CRAIG_BAMPTON_MODES = 1000;
const CRAIG_BAMPTON_DIFFERENCE = 1e-6;

// The plate pair of n × n ShellQuad4 each (one for the mixed hybrid formulation: its units small enough to keep every
// mode within the budget of a case), its subcases those of an eigenvalue analysis: two vibration subcases, or a compressive preload and its
// buckling.
function eigenPlatePair(analysis, n = 1) {
  const text = platePair("springs", n).replace("SinglePointConstrSelection;3", "SinglePointConstrSelection;1");
  if (analysis === "modes")
    return text.replace("SolutionSequence;101", "SolutionSequence;103")
      .replace("StaticLoadSelection;2", "EigenMethodSelection;3").replace("StaticLoadSelection;4", "EigenMethodSelection;3")
      + "LanczosEigenMethod;3;;;3\n";
  return text.replace("SolutionSequence;101", "SolutionSequence;105")
    .replace(/PointForceLoad;4;(\d+);;10\./g, "PointForceLoad;4;$1;;-10.")
    .replace("StaticLoadSelection;4", "StaticLoadSelection;4\nSubcaseStart;3\nSinglePointConstrSelection;1\n" +
      "EigenMethodSelection;3\nStaticSubcaseSelection;2") + "LanczosEigenMethod;3;;;2\n";
}

for (const analysis of ["modes", "buckling"])
  test(`Craig-Bampton ${analysis} of mixed hybrid plates are those of the full path`, () => {
    withMixedComponents(eigenPlatePair(analysis), (model) => {
      const full = solver.run(model, analysis);
      const condensed = solver.run(model, analysis,
        { condensation: { units: "all", modeCount: CRAIG_BAMPTON_MODES } });
      assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
      assert.deepEqual(condensed.condensation.condensedUnits, [1, 2]);
      const difference = eigenvalueDifference(full, condensed);
      assert.ok(difference < CRAIG_BAMPTON_DIFFERENCE, `roots ${difference}`);
    });
  });

// A wing of two mixed hybrid plates (root and tip, each 2 × 2 ShellQuad4), joined across a gap by springs, trimmed on
// one lifting panel: the condensed path keeps the spline's grid points on the boundaries of their units.
function twoPlateWing() {
  const n = 2;
  const id = (plate, i, j) => plate * 100 + j * (n + 1) + i + 1;
  const bulk = [];
  const ids = [];
  for (const plate of [0, 1])
    for (let j = 0; j <= n; ++j)
      for (let i = 0; i <= n; ++i) {
        ids.push(id(plate, i, j));
        bulk.push(`Node;${id(plate, i, j)};;${real(i / n)};${real(plate * 1.2 + j / n)};0.`);
      }
  let element = 0;
  for (const plate of [0, 1])
    for (let j = 0; j < n; ++j)
      for (let i = 0; i < n; ++i)
        bulk.push(`ShellQuad4;${++element};1;${id(plate, i, j)};${id(plate, i + 1, j)};${id(plate, i + 1, j + 1)};` +
          `${id(plate, i, j + 1)}`);
  let spring = 500;
  for (let i = 0; i <= n; ++i)
    for (let c = 1; c <= 5; ++c)
      bulk.push(`SpringDofInline;${spring++};1.e4;${id(0, i, n)};${c};${id(1, i, 0)};${c}`);
  bulk.push(`SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, i) => id(0, i, 0)).join(";")}`,
    "ShellProp;1;1;0.1;1;;1", "IsoMat;1;1.e6;;0.3;1.", "AeroStaticParams;0;0;1.;2.2;2.2",
    "AeroLiftingPanelProp1;10", "AeroLiftingPanel1;100;10;0;2;2;0;0;1;0.;0.;0.;1.;0.;2.2;0.;1.",
    `IdListSet1;50;${ids.join(";")}`, "AeroSurfaceSpline1;60;100;100;103;50", "AeroRigidTrimVariable;70;AngleOfAttack",
    "AeroTrim;5;0.1;10.;AngleOfAttack;0.05");
  return ["[executive]", "SolutionSequence;144", "[case]", "DisplacementOutput;All", "SinglePointConstrForceOutput;All",
    "StressOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "TrimSelection;5", "[model]", ...bulk,
    ""].join("\n");
}

test("a condensed trim of a wing of two mixed hybrid plates is the full path's", () => {
  withMixedComponents(twoPlateWing(), (model) => {
    const full = solver.run(model, "aeroelastic");
    assert.ok(!full.diagnostics.some((d) => d.severity === "error"), JSON.stringify(full.diagnostics));
    const condensed = solver.run(model, "aeroelastic", { condensation: "all" });
    assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
    assert.deepEqual(condensed.condensation.condensedUnits, [1, 2]);
    for (const kind of ["displacements", "singlePointConstrForces"])
      assert.ok(relativeDifference(full, condensed, kind) < MIXED_SAME_RESULT, kind);
    assert.ok(relativeDifference(full, condensed, "stress", "elementTables") < MIXED_SAME_RESULT, "stress");
  });
});

// The grid point forces of a model of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md section
// 10.1) on the condensed path: the table of the full path, row by row. `connection` joins the plates: springs (on the
// rotations: observer relations, the formulation's rows) or a fastener (a relation of a card on the shells' unknowns).
// The second case is a combination of the two subcases, whose constraint sets differ: its balance is closed by the
// formulation's row.
function forceRowsOf(run) {
  return run.gridPointForces.flatMap((table) => Array.from(table.nodeIds, (node, r) => ({
    key: `${table.subcase}/${node}/${table.elementIds[r]}/${table.names[table.nameIndices[r]]}`,
    name: table.names[table.nameIndices[r]], values: Array.from(table.values.slice(6 * r, 6 * r + 6)) })));
}

for (const connection of ["rotationSprings", "fastener"])
  test(`the grid point forces of condensed mixed hybrid plates are the full path's connection=${connection}`, () => {
    const text = overlappingPlates(connection).replace("DisplacementOutput;All", "DisplacementOutput;All\nNodeForceOutput;All")
      .replace("[model]", "SubcaseCombination;5\nSubcaseCombinationFactors;1.5,-0.5\n[model]");
    withMixedComponents(text, (model) => {
      const full = solver.run(model, "statics");
      const condensed = solver.run(model, "statics", { condensation: "all" });
      assert.ok(!condensed.diagnostics.some((d) => d.severity === "error"), JSON.stringify(condensed.diagnostics));
      const [a, b] = [forceRowsOf(full), forceRowsOf(condensed)];
      assert.equal(full.gridPointForces.length, 3, "a table per subcase and one of the combination");
      const scale = Math.max(...a.flatMap((row) => row.values.map(Math.abs)));
      // The balance closes on the full path, at every grid point of every case.
      for (const row of a.filter((entry) => entry.name === "total"))
        for (const value of row.values) assert.ok(Math.abs(value) < 1e-8 * scale, `${row.key}: total ${value}`);
      const other = new Map(b.map((row) => [row.key, row.values]));
      assert.equal(b.length, a.length, "the same rows");
      for (const row of a) {
        const values = other.get(row.key);
        assert.ok(values, `the condensed path gives the row ${row.key}`);
        for (let c = 0; c < 6; ++c)
          assert.ok(Math.abs(values[c] - row.values[c]) < MIXED_SAME_RESULT * scale,
            `${row.key} component ${c + 1}: ${values[c]}, the full path's ${row.values[c]}`);
      }
    });
  });
