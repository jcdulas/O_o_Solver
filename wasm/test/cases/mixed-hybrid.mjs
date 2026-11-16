// The mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md): the switch that puts the whole model in it,
// the refusals of the choices that contradict it, and the geometry declarations it reads.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

// A clamped plate of nx × 1 four-node shells in the XY plane, with a tip force, and the given case records.
function plate({ nx = 2, caseRecords = [], modelRecords = [] } = {}) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", ...caseRecords, "SubcaseStart;1",
    "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  const id = (i, j) => j * (nx + 1) + i + 1;
  for (let j = 0; j <= 1; ++j)
    for (let i = 0; i <= nx; ++i) lines.push(`Node;${id(i, j)};;${real(i / nx)};${real(0.2 * j)};0.`);
  for (let i = 0; i < nx; ++i) lines.push(`ShellQuad4;${i + 1};1;${id(i, 0)};${id(i + 1, 0)};${id(i + 1, 1)};${id(i, 1)}`);
  lines.push("ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.");
  lines.push(`SinglePointConstrList;1;123456;${id(0, 0)}`, `SinglePointConstrList;1;123456;${id(0, 1)}`);
  lines.push(`PointForceLoad;2;${id(nx, 0)};;1.;0.;0.;1.`, ...modelRecords);
  return lines.join("\n") + "\n";
}

function readDeck(text, name) {
  const file = `/work/${name}.oos`;
  oor.writeFile(file, text);
  return oor.readModel(file);
}

function withModel(text, name, options, action) {
  const deck = readDeck(text, name);
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try { return action(model); } finally { model.delete(); }
  } finally { deck.delete(); }
}

const refusalCodes = (model) => model.checkSupport("statics").refusals.map((d) => d.code);

test("the formulation of the whole model is ClassicFormulation by default and by the ElementFormulation parameter", () => {
  withModel(plate(), "mixed-default", {}, (model) => assert.deepEqual(refusalCodes(model), []));
  withModel(plate({ caseRecords: ["SubcaseParameter;ElementFormulation;ClassicFormulation"] }), "mixed-classic", {},
    (model) => assert.deepEqual(refusalCodes(model), []));
  withModel(plate(), "mixed-classic-api", { elementFormulation: "ClassicFormulation" },
    (model) => assert.deepEqual(refusalCodes(model), []));
});

test("an unknown formulation of the model, or one given in a subcase, is refused", () => {
  withModel(plate({ caseRecords: ["SubcaseParameter;ElementFormulation;NOSUCH"] }), "mixed-unknown", {},
    (model) => {
      const refusals = model.checkSupport("statics").refusals;
      assert.deepEqual(refusals.map((d) => d.code), ["unknown-formulation"]);
      assert.match(refusals[0].text, /NOSUCH/);
      assert.match(refusals[0].text, /MixedHybridFormulation/);
    });
  const perSubcase = plate().replace("SubcaseStart;1", "SubcaseStart;1\nSubcaseParameter;ElementFormulation;ClassicFormulation");
  withModel(perSubcase, "mixed-subcase", {}, (model) => assert.deepEqual(refusalCodes(model), ["unknown-formulation"]));
  // The program's choice is checked at the call.
  const deck = readDeck(plate(), "mixed-api-unknown");
  try {
    assert.throws(() => solver.modelFromReader(oor, deck, { elementFormulation: "NOSUCH" }), /MixedHybridFormulation/);
  } finally { deck.delete(); }
});

test("the mixed hybrid formulation is chosen for the model, never for one card", () => {
  withModel(plate({ caseRecords: ["SubcaseParameter;ShellQuad4Formulation;MixedHybridFormulation"] }), "mixed-card", {},
    (model) => assert.ok(refusalCodes(model).includes("formulation-per-card")));
  withModel(plate(), "mixed-card-api", { elementFormulations: { ShellQuad4: "MixedHybridFormulation" } },
    (model) => assert.ok(refusalCodes(model).includes("formulation-per-card")));
});

test("under the mixed hybrid formulation a card of the formulation cannot take another one", () => {
  withModel(plate({ caseRecords: ["SubcaseParameter;ElementFormulation;MixedHybridFormulation",
    "SubcaseParameter;ShellQuad4Formulation;ClassicFormulation"] }), "mixed-mix", { formulationGeometry: {} },
  (model) => {
    const codes = refusalCodes(model);
    assert.ok(codes.includes("formulation-mix"), codes.join(", "));
  });
});

test("the geometry declarations are checked when they are given", () => {
  const deck = readDeck(plate(), "mixed-geometry");
  try {
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { rawEdges: [[1, 2, 3]] } }),
      /each edge is \[g1, g2\]/);
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { rawEdges: [[4, 4]] } }),
      /two different grid points/);
    const model = solver.modelFromReader(oor, deck,
      { formulationGeometry: { rawEdges: [[2, 1], [1, 2]], rawElements: [1], sharpEdges: [] } });
    model.delete();
  } finally { deck.delete(); }
});

// ---------------------------------------------------------------------------------------------------------------------
// Line elements alone (spec section 3.2): their moments of degree k make a cantilever of any number of elements exact,
// as the ClassicFormulation bar is.

const BEAM = { length: 2.0, area: 0.01, i1: 1.0e-6, i2: 2.0e-6, j: 1.5e-6, young: 7.0e10, poisson: 0.3 };
const SHEAR = BEAM.young / (2 * (1 + BEAM.poisson));

// A cantilever of `count` Bar2 (or Rod2) along the unit direction `axis`, clamped at node 1, with the tip load
// `load` = [Fx, Fy, Fz, Mx, My, Mz] in the basic system; the orientation vector of the bars is `orientation`.
function cantilever({ card = "Bar2", count = 1, axis = [1, 0, 0], orientation = [0, 1, 0], load }) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "ElemForceOutput;All", "[model]"];
  for (let i = 0; i <= count; ++i) {
    const s = (BEAM.length * i) / count;
    lines.push(`Node;${i + 1};;${real(s * axis[0])};${real(s * axis[1])};${real(s * axis[2])}`);
  }
  for (let i = 0; i < count; ++i)
    lines.push(card === "Bar2" ? `Bar2;${i + 1};1;${i + 1};${i + 2};${real(orientation[0])};;${real(orientation[1])};${real(orientation[2])}`
      : `Rod2;${i + 1};1;${i + 1};${i + 2}`);
  lines.push(card === "Bar2" ? `BarProp;1;1;${real(BEAM.area)};${real(BEAM.i1)};${real(BEAM.i2)};${real(BEAM.j)}`
    : `RodProp;1;1;${real(BEAM.area)};${real(BEAM.j)}`);
  lines.push(`IsoMat;1;${real(BEAM.young)};;${real(BEAM.poisson)};2700.`);
  lines.push("SinglePointConstrList;1;123456;1");
  const [fx, fy, fz, mx, my, mz] = load;
  if (fx || fy || fz) lines.push(`PointForceLoad;2;${count + 1};;1.;${real(fx)};${real(fy)};${real(fz)}`);
  if (mx || my || mz) lines.push(`PointMomentLoad;2;${count + 1};;1.;${real(mx)};${real(my)};${real(mz)}`);
  return lines.join("\n") + "\n";
}

function tip(run, node) {
  const table = run.nodalTables.find((t) => t.kind === "displacements" && t.subcase === 1);
  const row = table.ids.indexOf(node);
  return Array.from(table.values.slice(row * 6, row * 6 + 6));
}

// Each component within tolerance times the largest expected magnitude (at least `floor`: a table of an unloaded
// element holds round-off only).
function assertClose(actual, expected, tolerance, what, floor = 1e-30) {
  const scale = Math.max(...expected.map(Math.abs), floor);
  for (let k = 0; k < expected.length; ++k)
    assert.ok(Math.abs(actual[k] - expected[k]) <= tolerance * scale,
      `${what}: component ${k + 1} is ${actual[k]}, expected ${expected[k]}`);
}

const P = 100.0;
const CANTILEVER_CASES = [
  { name: "stretched", load: [P, 0, 0, 0, 0, 0],
    expected: [P * BEAM.length / (BEAM.young * BEAM.area), 0, 0, 0, 0, 0] },
  { name: "bent in plane 1", load: [0, P, 0, 0, 0, 0],
    expected: [0, P * BEAM.length ** 3 / (3 * BEAM.young * BEAM.i1), 0, 0, 0,
      P * BEAM.length ** 2 / (2 * BEAM.young * BEAM.i1)] },
  { name: "bent in plane 2", load: [0, 0, P, 0, 0, 0],
    expected: [0, 0, P * BEAM.length ** 3 / (3 * BEAM.young * BEAM.i2), 0,
      -P * BEAM.length ** 2 / (2 * BEAM.young * BEAM.i2), 0] },
  { name: "twisted", load: [0, 0, 0, P, 0, 0], expected: [0, 0, 0, P * BEAM.length / (SHEAR * BEAM.j), 0, 0] },
  { name: "under an end couple", load: [0, 0, 0, 0, 0, P],
    expected: [0, P * BEAM.length ** 2 / (2 * BEAM.young * BEAM.i1), 0, 0, 0, P * BEAM.length / (BEAM.young * BEAM.i1)] },
];

for (const count of [1, 3])
  for (const { name, load, expected } of CANTILEVER_CASES)
    test(`a cantilever of ${count} Bar2 in MixedHybridFormulation ${name} is the beam theory's`, () => {
      const text = cantilever({ count, load }).replace("ElemForceOutput;All\n", "");
      withModel(text, `mixed-bar-${count}`, { elementFormulation: "MixedHybridFormulation" }, (model) => {
        const run = solver.run(model, "statics");
        assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
        // Exact but for the stiff shear of a bar without shear flexibility (k G A = 1e4 E A): 1e-7 here.
        assertClose(tip(run, count + 1), expected, 1e-6, `load ${load}`);
      });
    });

test("the Bar2 of MixedHybridFormulation gives the end forces of the ClassicFormulation bar", () => {
  const text = cantilever({ count: 2, axis: [0.6, 0.8, 0], orientation: [0, 0, 1], load: [30, -40, 50, 7, -3, 2] });
  const forces = (options) => withModel(text, "mixed-bar-forces", options, (model) => {
    const run = solver.run(model, "statics");
    const table = run.elementTables.find((t) => t.elementType === "Bar2" && t.kind === "forces");
    assert.ok(table, "a Bar2 force table");
    return Array.from(table.values);
  });
  const classic = forces({});
  const mixed = forces({ elementFormulation: "MixedHybridFormulation" });
  assertClose(mixed, classic, 1e-6, "forces");
});

test("a Rod2 in MixedHybridFormulation stretches as E A / L and twists as G J / L", () => {
  // Along x: a chain of rods has no stiffness across it, which the automatic constraint of singular components holds
  // in the displacement systems' axes (as for the ClassicFormulation rod).
  const P = 100.0;
  const axis = [1, 0, 0];
  const run = (load) => withModel(cantilever({ card: "Rod2", count: 2, axis, load }), "mixed-rod",
    { elementFormulation: "MixedHybridFormulation" }, (model) => solver.run(model, "statics"));
  const stretched = tip(run(axis.map((c) => P * c).concat([0, 0, 0])), 3);
  const u = P * BEAM.length / (BEAM.young * BEAM.area);
  assertClose(stretched.slice(0, 3), axis.map((c) => u * c), 1e-9, "stretch");
  const twisted = tip(run([0, 0, 0].concat(axis.map((c) => P * c))), 3);
  const phi = P * BEAM.length / (SHEAR * BEAM.j);
  assertClose(twisted.slice(3, 6), axis.map((c) => phi * c), 1e-9, "twist");
});

// A cantilever column of `count` Bar2 along x (BEAM's section and material, alpha 1e-5), clamped at node 1, as the deck
// of an analysis: `solution` 101 (with a uniform temperature rise `deltaT`), 103 or 105 (an axial compression at the
// tip as the preload of the buckling subcase).
function column({ count, solution, deltaT = 0 }) {
  const caseLines = solution === 105
    ? ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "SubcaseStart;2",
      "SinglePointConstrSelection;1", "EigenMethodSelection;3", "StaticSubcaseSelection;1"]
    : solution === 103 ? ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"]
    : ["TemperatureSelection;7;TemperatureLoadPart", "SubcaseStart;1", "SinglePointConstrSelection;1",
      "StaticLoadSelection;2", "DisplacementOutput;All", "ElemForceOutput;All"];
  const lines = ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseLines, "[model]"];
  for (let i = 0; i <= count; ++i) lines.push(`Node;${i + 1};;${real((BEAM.length * i) / count)};0.;0.`);
  for (let i = 0; i < count; ++i) lines.push(`Bar2;${i + 1};1;${i + 1};${i + 2};0.;;1.;0.`);
  lines.push(`BarProp;1;1;${real(BEAM.area)};${real(BEAM.i1)};${real(BEAM.i2)};${real(BEAM.j)}`);
  lines.push(`IsoMat;1;${real(BEAM.young)};;${real(BEAM.poisson)};2700.;1.e-5;0.`, "SinglePointConstrList;1;123456;1");
  if (solution === 105) lines.push(`PointForceLoad;2;${count + 1};;1.;-1.;0.;0.`);
  if (solution === 101) {
    lines.push(`PointForceLoad;2;${count + 1};;1.;0.;0.;0.`);
    for (let i = 0; i <= count; ++i) lines.push(`NodeTemperatureLoad;7;${i + 1};${real(deltaT)}`);
  }
  if (solution !== 101) lines.push("LanczosEigenMethod;3;;;2");
  return lines.join("\n") + "\n";
}

test("a free cantilever of Bar2 in MixedHybridFormulation expands without stress under a temperature rise", () => {
  const deltaT = 50.0;
  withModel(column({ count: 2, solution: 101, deltaT }), "mixed-thermal", { elementFormulation: "MixedHybridFormulation" },
    (model) => {
      const run = solver.run(model, "statics");
      assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
      assertClose(tip(run, 3), [1.0e-5 * deltaT * BEAM.length, 0, 0, 0, 0, 0], 1e-9, "free expansion");
      const forces = run.elementTables.find((t) => t.elementType === "Bar2" && t.kind === "forces");
      const largest = Math.max(...Array.from(forces.values).map(Math.abs));
      assert.ok(largest < 1e-6 * BEAM.young * BEAM.area * 1.0e-5 * deltaT, `no force: ${largest}`);
    });
});

test("a cantilever of Bar2 in MixedHybridFormulation vibrates at the beam's first bending frequency", () => {
  withModel(column({ count: 4, solution: 103 }), "mixed-modes", { elementFormulation: "MixedHybridFormulation" },
    (model) => {
      const run = solver.run(model, "modes");
      const first = run.eigenvalues[0].rows[0].eigenvalue;
      // ω1² = 1.8751⁴ E I / (ρ A L⁴), plane 1 (I1, the smaller inertia) bends first.
      const expected = 1.87510407 ** 4 * BEAM.young * BEAM.i1 / (2700 * BEAM.area * BEAM.length ** 4);
      // The mass of the formulation has the rotary inertia of the section, which the Euler-Bernoulli value leaves out:
      // it lowers the frequency by about 1e-4 here (r/L = 0.005).
      assert.ok(Math.abs(first / expected - 1) < 3e-4, `first eigenvalue ${first}, beam ${expected}`);
    });
});

test("a cantilever column of Bar2 in MixedHybridFormulation buckles at Euler's load", () => {
  withModel(column({ count: 4, solution: 105 }), "mixed-buckling", { elementFormulation: "MixedHybridFormulation" },
    (model) => {
      const run = solver.run(model, "buckling");
      const factor = run.eigenvalues[0].rows[0].eigenvalue;
      // P_cr = π² E I / (4 L²) for a unit compression.
      const expected = Math.PI ** 2 * BEAM.young * BEAM.i1 / (4 * BEAM.length ** 2);
      assert.ok(Math.abs(factor / expected - 1) < 1e-4, `buckling factor ${factor}, Euler ${expected}`);
    });
});

test("the Beam2 of MixedHybridFormulation gives the station forces of the ClassicFormulation beam", () => {
  const text = cantilever({ count: 2, axis: [0.6, 0.8, 0], orientation: [0, 0, 1], load: [30, -40, 50, 7, -3, 2] })
    .replace(/^Bar2;(\d+);1;/gm, "Beam2;$1;1;")
    .replace(/^BarProp;.*$/m, `BeamProp;1;1;${real(BEAM.area)};${real(BEAM.i1)};${real(BEAM.i2)};;${real(BEAM.j)}`);
  const forces = (options) => withModel(text, "mixed-beam-forces", options, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const table = run.elementTables.find((t) => t.elementType === "Beam2" && t.kind === "forces");
    assert.ok(table, "a Beam2 force table");
    return { values: Array.from(table.values), displacement: tip(run, 3) };
  });
  const classic = forces({});
  const mixed = forces({ elementFormulation: "MixedHybridFormulation" });
  assertClose(mixed.displacement, classic.displacement, 1e-6, "tip displacement");
  assertClose(mixed.values, classic.values, 1e-6, "station forces");
});

for (const loadKind of ["uniform", "partial", "concentrated", "couple"])
  test(`a Bar2 of MixedHybridFormulation under a ${loadKind} BeamDistributedLoad moves and loads as the ClassicFormulation bar`, () => {
    const lines = {
      uniform: ["BeamDistributedLoad;2;1;ForceY;FractionalLength;0.;-30.;1.;-30.",
        "BeamDistributedLoad;2;2;ForceY;FractionalLength;0.;-30.;1.;-30."],
      partial: ["BeamDistributedLoad;2;2;ForceZ;FractionalLength;0.2;10.;0.7;-25."],
      concentrated: ["BeamDistributedLoad;2;1;ForceY;FractionalLength;0.4;80.;0.4;"],
      couple: ["BeamDistributedLoad;2;2;MomentX;FractionalLength;0.;5.;1.;5."],
    }[loadKind];
    const text = cantilever({ count: 2, axis: [0.6, 0.8, 0], orientation: [0, 0, 1], load: [0, 0, 0, 0, 0, 0] })
      .replace("ElemForceOutput;All", "ElemForceOutput;All\nStressOutput;All") + lines.join("\n") + "\n";
    const results = (options) => withModel(text, "mixed-bar-line-loads", options, (model) => {
      const run = solver.run(model, "statics");
      assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
      const tables = run.elementTables.filter((t) => t.elementType.startsWith("Bar2"))
        .map((t) => `${t.elementType}/${t.kind}:` + Array.from(t.values).map((v) => v.toPrecision(6)).join(","));
      return { displacement: tip(run, 3), tables: run.elementTables.filter((t) => t.elementType.startsWith("Bar2")) };
    });
    const classic = results({});
    const mixed = results({ elementFormulation: "MixedHybridFormulation" });
    assertClose(mixed.displacement, classic.displacement, 1e-6, "tip displacement");
    assert.equal(mixed.tables.length, classic.tables.length, "the same tables");
    for (const table of classic.tables) {
      const other = mixed.tables.find((t) => t.elementType === table.elementType && t.kind === table.kind);
      assert.ok(other, `${table.elementType} ${table.kind}`);
      assertClose(Array.from(other.values), Array.from(table.values), 1e-5, `${table.elementType} ${table.kind}`, 1e-3);
    }
  });

// A propped beam with a hinge (the pin flags of a Bar2: section 4.2): the same displacements and forces as the
// ClassicFormulation bar, whose single elements are exact here.
for (const variant of ["hinge"])
  test(`a Bar2 of MixedHybridFormulation with ${variant} moves and loads as the ClassicFormulation bar`, () => {
    const second = "Bar2;1;1;1;2;0.;;0.;1.;;;56";
    const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
      "StaticLoadSelection;2", "DisplacementOutput;All", "ElemForceOutput;All", "[model]",
      "Node;1;;0.;0.;0.", "Node;2;;1.;0.2;0.", "Node;3;;2.;0.;0.3", second, "Bar2;2;1;2;3;0.;;0.;1.",
      `BarProp;1;1;${real(BEAM.area)};${real(BEAM.i1)};${real(BEAM.i2)};${real(BEAM.j)}`,
      `IsoMat;1;${real(BEAM.young)};;${real(BEAM.poisson)};2700.`, "SinglePointConstrList;1;123456;1",
      "SinglePointConstrList;1;123;3", "PointForceLoad;2;2;;1.;10.;-60.;35.", "PointMomentLoad;2;2;;1.;0.;4.;0."]
      .join("\n") + "\n";
    const results = (options) => withModel(text, `mixed-bar-${variant}`, options, (model) => {
      const run = solver.run(model, "statics");
      assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
      const table = run.elementTables.find((t) => t.elementType === "Bar2" && t.kind === "forces");
      return { displacement: tip(run, 2), forces: Array.from(table.values) };
    });
    const classic = results({});
    const mixed = results({ elementFormulation: "MixedHybridFormulation" });
    assertClose(mixed.displacement, classic.displacement, 1e-6, "displacement of the middle grid point");
    assertClose(mixed.forces, classic.forces, 1e-6, "forces", 1e-3);
  });

// A cantilever Bar2 whose axis is offset from its grid points in the plane of the load: the formulation's eccentric
// axis (the offset couples the axial strain to the curvatures) gives the ClassicFormulation's rigid offsets exactly.
// Across the offset they differ: the formulation keeps the torsion and the shear about the edge (spec section 3.6).
test("an offset Bar2 of MixedHybridFormulation bends as the ClassicFormulation bar in the plane of its offset", () => {
  const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", "Node;1;;0.;0.;0.", "Node;2;;2.;0.;0.",
    "Bar2;1;1;1;2;0.;;0.;1.;;;;0.;0.05;0.;0.;0.05;0.",
    `BarProp;1;1;${real(BEAM.area)};${real(BEAM.i1)};${real(BEAM.i2)};${real(BEAM.j)}`,
    `IsoMat;1;${real(BEAM.young)};;${real(BEAM.poisson)};2700.`, "SinglePointConstrList;1;123456;1",
    "PointForceLoad;2;2;;1.;100.;-60.;0.", "PointMomentLoad;2;2;;1.;0.;0.;4."].join("\n") + "\n";
  const displacement = (options) => withModel(text, "mixed-bar-offset", options,
    (model) => tip(solver.run(model, "statics"), 2));
  assertClose(displacement({ elementFormulation: "MixedHybridFormulation" }), displacement({}), 1e-6, "tip");
});

// ---------------------------------------------------------------------------------------------------------------------
// Shells (spec section 3.1): their displacement of degree k holds the affine membrane fields and the quadratic
// deflections exactly, so that a uniform stretch and a pure bending imposed by the supports are exact on any mesh.

const STRIP = { length: 2.0, width: 0.4, thickness: 0.01, young: 7.0e10 };
const SHELL_OPTIONS = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };

// A strip of nx × ny shells of `shape` (ShellQuad4, ShellTria3: each quadrilateral cut along its diagonal, ShellQuad8,
// ShellTria6) in the XY plane, STRIP.length along x and STRIP.width along y, lifted by z(x, y), with the constraint and
// load records `records(node, nx, ny)`: `node(i, j)` its corner grid point at column i and row j; `points` every grid
// point with its position.
function strip({ shape = "ShellQuad4", nx = 4, ny = 2, poisson = 0.3, z = () => 0, records = () => [], extra = [] }) {
  const m = shape === "ShellQuad8" || shape === "ShellTria6" ? 2 : 1;
  const at = (a, b) => b * (m * nx + 1) + a + 1;
  const node = (i, j) => at(m * i, m * j);
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "[model]"];
  const points = [];
  for (let b = 0; b <= m * ny; ++b)
    for (let a = 0; a <= m * nx; ++a) {
      if (shape === "ShellQuad8" && a % 2 === 1 && b % 2 === 1) continue;
      const x = (STRIP.length * a) / (m * nx);
      const y = (STRIP.width * b) / (m * ny);
      points.push({ id: at(a, b), x, y, z: z(x, y), corner: a % m === 0 && b % m === 0 });
      lines.push(`Node;${at(a, b)};;${real(x)};${real(y)};${real(z(x, y))}`);
    }
  let element = 0;
  for (let j = 0; j < ny; ++j)
    for (let i = 0; i < nx; ++i) {
      const [a, b] = [m * i, m * j];
      if (shape === "ShellQuad4")
        lines.push(`ShellQuad4;${++element};1;${node(i, j)};${node(i + 1, j)};${node(i + 1, j + 1)};${node(i, j + 1)}`);
      else if (shape === "ShellTria3")
        lines.push(`ShellTria3;${++element};1;${node(i, j)};${node(i + 1, j)};${node(i + 1, j + 1)}`,
          `ShellTria3;${++element};1;${node(i, j)};${node(i + 1, j + 1)};${node(i, j + 1)}`);
      else if (shape === "ShellQuad8")
        lines.push(`ShellQuad8;${++element};1;${[at(a, b), at(a + 2, b), at(a + 2, b + 2), at(a, b + 2),
          at(a + 1, b), at(a + 2, b + 1), at(a + 1, b + 2), at(a, b + 1)].join(";")}`);
      else
        lines.push(`ShellTria6;${++element};1;${[at(a, b), at(a + 2, b), at(a + 2, b + 2),
          at(a + 1, b), at(a + 2, b + 1), at(a + 1, b + 1)].join(";")}`,
        `ShellTria6;${++element};1;${[at(a, b), at(a + 2, b + 2), at(a, b + 2),
          at(a + 1, b + 1), at(a + 1, b + 2), at(a, b + 1)].join(";")}`);
    }
  lines.push(`ShellProp;1;1;${real(STRIP.thickness)};1;;1`, `IsoMat;1;${real(STRIP.young)};;${real(poisson)};2700.`);
  lines.push(...records(node, nx, ny), ...extra);
  return { text: lines.join("\n") + "\n", node, nx, ny, points };
}

function nodal(run, kind, node) {
  const table = run.nodalTables.find((t) => t.kind === kind && t.subcase === 1);
  assert.ok(table, `a ${kind} table`);
  const row = table.ids.indexOf(node);
  return row < 0 ? [0, 0, 0, 0, 0, 0] : Array.from(table.values.slice(row * 6, row * 6 + 6));
}

function runShells(text, name) {
  return withModel(text, name, SHELL_OPTIONS, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return run;
  });
}

// The Bar2 stiffeners along the edge y = 0 of a strip (one per element side), their axis on the shells' mid-surface,
// bending out of the plane of the strip in their plane 1.
const STIFFENER = { area: 2.0e-4, i1: 3.0e-9, i2: 1.0e-9, j: 2.0e-9 };
function stiffeners(node, nx) {
  const records = [`BarProp;2;1;${real(STIFFENER.area)};${real(STIFFENER.i1)};${real(STIFFENER.i2)};${real(STIFFENER.j)}`];
  for (let i = 0; i < nx; ++i) records.push(`Bar2;${100 + i};2;${node(i, 0)};${node(i + 1, 0)};0.;;0.;1.`);
  return records;
}

const STRETCH_CASES = ["ShellQuad4", "ShellTria3", "ShellQuad8", "ShellTria6"].map((shape) => ({ shape, stiffened: false }))
  .concat([{ shape: "ShellQuad4", stiffened: true }]);

for (const { shape, stiffened } of STRETCH_CASES)
  test(`a strip of ${shape} in MixedHybridFormulation${stiffened ? " stiffened by Bar2 on its edge" : ""} stretched by its supports takes the uniform uniaxial stress`, () => {
    const delta = 1.0e-3;
    const poisson = 0.3;
    const model = strip({ shape, poisson, records: (node, nx, ny) => {
      const records = [`SinglePointConstrList;1;2;${node(0, 0)}`];
      for (let j = 0; j <= ny; ++j)
        records.push(`SinglePointConstrList;1;1;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};1;${real(delta)}`);
      for (let j = 0; j <= ny; ++j) for (let i = 0; i <= nx; ++i) records.push(`SinglePointConstrList;1;3;${node(i, j)}`);
      return stiffened ? records.concat(stiffeners(node, nx)) : records;
    } });
    const run = runShells(model.text, `mixed-shell-stretch-${shape}`);
    for (const { id, x, y } of model.points)
      assertClose(nodal(run, "displacements", id).slice(0, 3),
        [(delta * x) / STRIP.length, (-poisson * delta * y) / STRIP.length, 0], 1e-8, `grid ${id}`, delta);
    let pull = 0;
    for (let j = 0; j <= model.ny; ++j) pull += nodal(run, "singlePointConstrForces", model.node(model.nx, j))[0];
    const area = STRIP.thickness * STRIP.width + (stiffened ? STIFFENER.area : 0);
    const expected = (STRIP.young * area * delta) / STRIP.length;
    assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
  });

for (const { shape, stiffened } of STRETCH_CASES)
  test(`a strip of ${shape} in MixedHybridFormulation${stiffened ? " stiffened by Bar2 on its edge" : ""} bent by its supports takes the uniform moment`, () => {
    // Without Poisson's ratio the clamped end does not stop the anticlastic curvature: w = θ x² / (2 L) exactly.
    const theta = 1.0e-3;
    const model = strip({ shape, poisson: 0, records: (node, nx, ny) => {
      const records = [];
      for (let j = 0; j <= ny; ++j)
        records.push(`SinglePointConstrList;1;123456;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};5;${real(-theta)}`);
      return stiffened ? records.concat(stiffeners(node, nx)) : records;
    } });
    const run = runShells(model.text, `mixed-shell-bend-${shape}`);
    for (const { id, x } of model.points)
      assertClose(nodal(run, "displacements", id).slice(0, 3), [0, 0, (theta * x * x) / (2 * STRIP.length)], 1e-8,
        `grid ${id}`, theta);
    let moment = 0;
    for (let j = 0; j <= model.ny; ++j) moment += nodal(run, "singlePointConstrForces", model.node(model.nx, j))[4];
    const stiffness = (STRIP.young * STRIP.thickness ** 3 * STRIP.width) / 12 + (stiffened ? STRIP.young * STIFFENER.i1 : 0);
    const expected = -(stiffness * theta) / STRIP.length;
    assert.ok(Math.abs(moment / expected - 1) < 1e-8, `moment ${moment}, expected ${expected}`);
  });

test("a free warped plate of ShellQuad4 in MixedHybridFormulation has six rigid-body modes", () => {
  // Every element of the hyperbolic paraboloid z = 0.2 (x − 1)(y − 0.2) / 0.4 is warped.
  const model = strip({ nx: 3, ny: 2, z: (x, y) => (0.2 * (x - 1) * (y - 0.2)) / 0.4 });
  const text = model.text.replace("SolutionSequence;101", "SolutionSequence;103")
    .replace(/SubcaseStart;1\nSinglePointConstrSelection;1\n/, "SubcaseStart;1\nEigenMethodSelection;3\n") +
    "LanczosEigenMethod;3;;;8\n";
  withModel(text, "mixed-shell-free", SHELL_OPTIONS, (model) => {
    const run = solver.run(model, "modes");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    assert.ok(values.length >= 7, `${values.length} eigenvalues`);
    for (let k = 0; k < 6; ++k)
      assert.ok(Math.abs(values[k]) < 1e-6 * values[6], `eigenvalue ${k + 1} is ${values[k]}, the first elastic ${values[6]}`);
  });
});

// The rows of an element table as objects of its columns, with their element and grid point (0 for the centre).
function rowsOf(run, elementType, kind) {
  const table = run.elementTables.find((t) => t.elementType === elementType && t.kind === kind && t.subcase === 1);
  assert.ok(table, `a ${elementType} ${kind} table`);
  const width = table.components.length;
  return Array.from(table.ids, (id, r) => {
    const row = { id, nodeId: table.nodeIds[r], layer: table.layers[r] };
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    return row;
  });
}

// The corner rows of the four-node quadrilaterals and three-node triangles are written on request (CornerResults).
const WITH_ELEMENT_OUTPUT =
  "SinglePointConstrForceOutput;All\nElemForceOutput;All;CornerResults\nStressOutput;All;CornerResults";

for (const shape of ["ShellQuad4", "ShellQuad8"])
  test(`the ${shape} of MixedHybridFormulation write the forces and stresses of a uniform stretch at their centre and corners`, () => {
    const delta = 1.0e-3;
    const model = strip({ shape, poisson: 0.3, records: (node, nx, ny) => {
      const records = [`SinglePointConstrList;1;2;${node(0, 0)}`];
      for (let j = 0; j <= ny; ++j)
        records.push(`SinglePointConstrList;1;1;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};1;${real(delta)}`);
      for (let j = 0; j <= ny; ++j) for (let i = 0; i <= nx; ++i) records.push(`SinglePointConstrList;1;3;${node(i, j)}`);
      return records;
    } });
    const run = runShells(model.text.replace("SinglePointConstrForceOutput;All", WITH_ELEMENT_OUTPUT),
      `mixed-shell-stretch-rows-${shape}`);
    const force = (STRIP.young * STRIP.thickness * delta) / STRIP.length;
    const forces = rowsOf(run, shape, "forces");
    assert.equal(forces.length, model.nx * model.ny * 5, "the centre and the four corners of each element");
    for (const row of forces)
      assertClose([row.nx, row.ny, row.nxy, row.mx / STRIP.thickness, row.my / STRIP.thickness,
        row.mxy / STRIP.thickness, row.qx, row.qy], [force, 0, 0, 0, 0, 0, 0, 0], 1e-8,
      `forces of element ${row.id} at ${row.nodeId}`);
    for (const row of rowsOf(run, shape, "stress"))
      assertClose([row.oxx, row.oyy, row.txy], [force / STRIP.thickness, 0, 0], 1e-8,
        `stresses of element ${row.id} at ${row.nodeId}, fibre ${row.layer}`);
  });

test("the ShellQuad4 of MixedHybridFormulation write the moments of a pure bending", () => {
  const theta = 1.0e-3;
  const model = strip({ poisson: 0, records: (node, nx, ny) => {
    const records = [];
    for (let j = 0; j <= ny; ++j)
      records.push(`SinglePointConstrList;1;123456;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};5;${real(-theta)}`);
    return records;
  } });
  const run = runShells(model.text.replace("SinglePointConstrForceOutput;All", WITH_ELEMENT_OUTPUT),
    "mixed-shell-bend-rows");
  // The force table writes −∫σ z dz = D κ, κx = ∂²w/∂x² = θ / L.
  const moment = (STRIP.young * STRIP.thickness ** 3 * theta) / (12 * STRIP.length);
  for (const row of rowsOf(run, "ShellQuad4", "forces"))
    assertClose([row.nx * STRIP.thickness, row.ny * STRIP.thickness, row.nxy * STRIP.thickness, row.mx, row.my, row.mxy,
      row.qx * STRIP.thickness, row.qy * STRIP.thickness], [0, 0, 0, moment, 0, 0, 0, 0], 1e-8,
    `forces of element ${row.id} at ${row.nodeId}`);
  // The fibre Z1 = −t/2 stretches: σ = E (t / 2) θ / L.
  for (const row of rowsOf(run, "ShellQuad4", "stress"))
    assertClose([row.oxx, row.oyy, row.txy], [(row.layer === 1 ? 1 : -1) * STRIP.young * STRIP.thickness * theta /
      (2 * STRIP.length), 0, 0], 1e-8, `stresses of element ${row.id} at ${row.nodeId}, fibre ${row.layer}`);
});

test("the ShellQuad4 of MixedHybridFormulation write the centre forces of a cantilever strip as the ClassicFormulation one", () => {
  // Without Poisson's ratio the strip is a beam: m_x = F (L − x) / b and q_x = F / b per unit width at every centre,
  // which both formulations give; the signs of the shear forces and of the moments are those of the result tables.
  const force = 50.0;
  const model = strip({ poisson: 0, records: (node, nx, ny) => {
    const records = [];
    for (let j = 0; j <= ny; ++j) {
      records.push(`SinglePointConstrList;1;123456;${node(0, j)}`);
      records.push(`PointForceLoad;2;${node(nx, j)};;${real(force / ny / (j === 0 || j === ny ? 2 : 1))};0.;0.;1.`);
    }
    return records;
  } });
  const text = model.text.replace("SinglePointConstrForceOutput;All", "ElemForceOutput;All")
    .replace("SinglePointConstrSelection;1\n", "SinglePointConstrSelection;1\nStaticLoadSelection;2\n");
  const centres = (options) => withModel(text, "mixed-shell-cantilever-rows", options, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return rowsOf(run, "ShellQuad4", "forces").filter((row) => row.nodeId === 0);
  });
  const classic = centres({});
  const mixed = centres(SHELL_OPTIONS);
  assert.equal(mixed.length, classic.length);
  for (let k = 0; k < classic.length; ++k) {
    const scale = force / STRIP.width;
    assert.ok(Math.abs(mixed[k].qx - classic[k].qx) < 0.02 * scale, `qx of element ${mixed[k].id}: ${mixed[k].qx}, ${classic[k].qx}`);
    assert.ok(Math.abs(mixed[k].mx - classic[k].mx) < 0.02 * scale * STRIP.length,
      `mx of element ${mixed[k].id}: ${mixed[k].mx}, ${classic[k].mx}`);
  }
});

// A cantilever strip without Poisson's ratio is a Timoshenko beam whose deflection under a uniform load is a quartic:
// the shells hold it exactly, w(L) = p L⁴ / (8 D) + p L² / (2 κ G t) per unit width (D = E t³ / 12, G = E / 2,
// κ = 0.833333 the default TS/T of ShellProp).
const SHEAR_FACTOR = 0.833333;
function cantileverStrip({ shape = "ShellQuad4", loads }) {
  return strip({ shape, poisson: 0, records: (node, nx, ny) => {
    const records = [];
    for (let j = 0; j <= ny; ++j) records.push(`SinglePointConstrList;1;123456;${node(0, j)}`);
    return records.concat(loads(nx, ny));
  } });
}
const loadCase = (text) => text.replace("SinglePointConstrSelection;1\n", "SinglePointConstrSelection;1\nStaticLoadSelection;2\n");

for (const variant of ["ShellPressureLoad", "FacePressureLoad", "FacePressureLoad along a direction"])
  test(`a cantilever strip of ShellQuad4 in MixedHybridFormulation under a uniform ${variant} bends as the Timoshenko beam`, () => {
    const p = 200.0;
    const model = cantileverStrip({ loads: (nx, ny) => {
      const count = nx * ny;
      if (variant === "ShellPressureLoad") return [`ShellPressureLoad;2;${real(p)};1;Through;${count}`];
      const records = [];
      for (let e = 1; e <= count; ++e)
        records.push(variant === "FacePressureLoad" ? `FacePressureLoad;2;${e};${real(p)}`
          : `FacePressureLoad;2;${e};${real(p)};;;;;;0;0.;0.;2.`);
      return records;
    } });
    const run = runShells(loadCase(model.text), "mixed-shell-pressure");
    const d = (STRIP.young * STRIP.thickness ** 3) / 12;
    const g = STRIP.young / 2;
    const expected = (p * STRIP.length ** 4) / (8 * d) + (p * STRIP.length ** 2) / (2 * SHEAR_FACTOR * g * STRIP.thickness);
    for (let j = 0; j <= model.ny; ++j) {
      const w = nodal(run, "displacements", model.node(model.nx, j))[2];
      assert.ok(Math.abs(w / expected - 1) < 1e-8, `tip deflection ${w}, expected ${expected}`);
    }
  });

test("a free strip of ShellQuad4 in MixedHybridFormulation expands without stress under a temperature linear in x", () => {
  // ε_x = ε_y = a x with a = α ΔT_L / L is compatible: u = a (x² − y²) / 2, v = a x y, held at (0, 0) and by v at
  // (L, 0) only.
  const alpha = 1.0e-5;
  const rise = 60.0;
  const model = strip({ poisson: 0.3, records: (node, nx, ny) => {
    const records = [`SinglePointConstrList;1;12;${node(0, 0)}`, `SinglePointConstrList;1;2;${node(nx, 0)}`];
    for (let j = 0; j <= ny; ++j)
      for (let i = 0; i <= nx; ++i)
        records.push(`SinglePointConstrList;1;3;${node(i, j)}`, `NodeTemperatureLoad;7;${node(i, j)};${real((rise * i) / nx)}`);
    return records;
  } });
  const text = model.text.replace(";2700.\n", `;2700.;${real(alpha)};0.\n`)
    .replace("[case]\n", "[case]\nTemperatureSelection;7;TemperatureLoadPart\n")
    .replace("SinglePointConstrForceOutput;All", "SinglePointConstrForceOutput;All\nElemForceOutput;All;CornerResults");
  const run = runShells(text, "mixed-shell-thermal");
  const a = (alpha * rise) / STRIP.length;
  for (const { id, x, y } of model.points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [(a * (x * x - y * y)) / 2, a * x * y, 0], 1e-8, `grid ${id}`,
      a * STRIP.length ** 2);
  const scale = STRIP.young * STRIP.thickness * alpha * rise;
  for (const row of rowsOf(run, "ShellQuad4", "forces"))
    for (const name of ["nx", "ny", "nxy"])
      assert.ok(Math.abs(row[name]) < 1e-8 * scale, `${name} of element ${row.id} at ${row.nodeId}: ${row[name]}`);
});

test("a cantilever strip of ShellQuad4 in MixedHybridFormulation buckles at Euler's load", () => {
  const model = cantileverStrip({ loads: (nx, ny) => {
    const records = [];
    for (let j = 0; j <= ny; ++j)
      records.push(`PointForceLoad;2;${(j + 1) * (nx + 1)};;${real(1 / ny / (j === 0 || j === ny ? 2 : 1))};-1.;0.;0.`);
    return records;
  } });
  const text = model.text.replace("SolutionSequence;101", "SolutionSequence;105")
    .replace(/SubcaseStart;1\nSinglePointConstrSelection;1\n/, "SubcaseStart;1\nSinglePointConstrSelection;1\nStaticLoadSelection;2\n" +
      "SubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\nStaticSubcaseSelection;1\n") + "LanczosEigenMethod;3;;;2\n";
  withModel(text, "mixed-shell-buckling", SHELL_OPTIONS, (model) => {
    const run = solver.run(model, "buckling");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const factor = run.eigenvalues[0].rows[0].eigenvalue;
    const inertia = (STRIP.width * STRIP.thickness ** 3) / 12;
    const expected = (Math.PI ** 2 * STRIP.young * inertia) / (4 * STRIP.length ** 2);
    assert.ok(Math.abs(factor / expected - 1) < 1e-3, `buckling factor ${factor}, Euler ${expected}`);
  });
});

// The Scordelis-Lo roof (MacNeal and Harder, "A proposed standard set of problems to test finite element accuracy",
// Finite Elements in Analysis and Design 1, 1985): a cylindrical roof of radius 25 and length 50 over ±40°, t = 0.25,
// E = 4.32e8, ν = 0, under its weight of 90 per unit area, on diaphragms at its curved ends. The vertical
// displacement at the middle of a free edge is 0.3024 in that reference. A quarter of the roof, nx × nφ ShellQuad4 on
// the cylinder (the formulation reads the curved surface from its grid points), held by symmetry along its middle
// section x = 0 (u_x, θ_y, θ_z) and along its crown φ = 0 (u_y, θ_x, θ_z): the rotations held at grid points of shells
// hold the rotation of the edges about their tangent.
function scordelisLo({ nx = 4, nphi = 4 }) {
  const radius = 25.0;
  const half = 25.0;
  const spread = (40 * Math.PI) / 180;
  const node = (i, j) => j * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  for (let j = 0; j <= nphi; ++j)
    for (let i = 0; i <= nx; ++i) {
      const phi = (spread * j) / nphi;
      lines.push(`Node;${node(i, j)};;${real((half * i) / nx)};${real(radius * Math.sin(phi))};${real(radius * Math.cos(phi))}`);
    }
  let element = 0;
  for (let j = 0; j < nphi; ++j)
    for (let i = 0; i < nx; ++i)
      lines.push(`ShellQuad4;${++element};1;${node(i, j)};${node(i + 1, j)};${node(i + 1, j + 1)};${node(i, j + 1)}`);
  lines.push("ShellProp;1;1;0.25;1;;1", "IsoMat;1;4.32e8;;0.;360.", "GravityLoad;2;;1.;0.;0.;-1.");
  for (let j = 0; j <= nphi; ++j)
    lines.push(`SinglePointConstrList;1;23;${node(nx, j)}`, `SinglePointConstrList;1;156;${node(0, j)}`);
  for (let i = 0; i <= nx; ++i) lines.push(`SinglePointConstrList;1;246;${node(i, 0)}`);
  return { text: lines.join("\n") + "\n", freeEdgeMiddle: node(0, nphi) };
}

test("the Scordelis-Lo roof of ShellQuad4 in MixedHybridFormulation sags as the reference value", () => {
  const model = scordelisLo({});
  const run = runShells(model.text, "mixed-shell-roof");
  const sag = -nodal(run, "displacements", model.freeEdgeMiddle)[2];
  assert.ok(Math.abs(sag / 0.3024 - 1) < 0.01, `sag ${sag}, reference 0.3024`);
});

// ---------------------------------------------------------------------------------------------------------------------
// Solids (spec section 3.3): their boundary displacement and their stresses of degree 4 hold the uniform and the
// linear stress states exactly, so that a uniform stretch and a pure bending imposed by the supports are exact for
// every shape.

const BLOCK = { length: 2.0, height: 1.0, width: 1.0, young: 2.0e11 };

// A block of nx cells along x (BLOCK.length) of SolidHexa, of two SolidPenta, of six SolidTetra around the diagonal
// c0-c6, or of six SolidPyramid on the faces of the cell with an apex at its centre, and the constraint and load
// records `records(node, nx)`: node(i, j, k) its grid point at the corner (i, j, k) of the lattice (j, k = 0 or 1).
// `points` every grid point with its position.
function solidBlock({ shape, nx = 1, poisson = 0.3, records = () => [] }) {
  const node = (i, j, k) => (k * 2 + j) * (nx + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "StressOutput;All", "[model]"];
  const points = [];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i <= nx; ++i) {
        const p = { id: node(i, j, k), x: (BLOCK.length * i) / nx, y: BLOCK.height * j, z: BLOCK.width * k };
        points.push(p);
        lines.push(`Node;${p.id};;${real(p.x)};${real(p.y)};${real(p.z)}`);
      }
  let element = 0;
  let centre = 4 * (nx + 1);
  for (let i = 0; i < nx; ++i) {
    const c = [node(i, 0, 0), node(i + 1, 0, 0), node(i + 1, 1, 0), node(i, 1, 0),
      node(i, 0, 1), node(i + 1, 0, 1), node(i + 1, 1, 1), node(i, 1, 1)];
    const cells = {
      SolidHexa: [c],
      SolidPenta: [[c[0], c[1], c[2], c[4], c[5], c[6]], [c[0], c[2], c[3], c[4], c[6], c[7]]],
      SolidTetra: [[c[0], c[1], c[2], c[6]], [c[0], c[2], c[3], c[6]], [c[0], c[3], c[7], c[6]],
        [c[0], c[7], c[4], c[6]], [c[0], c[4], c[5], c[6]], [c[0], c[5], c[1], c[6]]],
    }[shape];
    if (shape === "SolidPyramid") {
      const apex = ++centre;
      const p = { id: apex, x: (BLOCK.length * (i + 0.5)) / nx, y: BLOCK.height / 2, z: BLOCK.width / 2 };
      points.push(p);
      lines.push(`Node;${p.id};;${real(p.x)};${real(p.y)};${real(p.z)}`);
      // Each face of the cell ordered so that its normal points to the apex.
      for (const base of [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
        lines.push(`SolidPyramid;${++element};1;${base.map((b) => c[b]).join(";")};${apex}`);
    } else
      for (const cell of cells) lines.push(`${shape};${++element};1;${cell.join(";")}`);
  }
  lines.push("SolidProp;1;1", `IsoMat;1;${real(BLOCK.young)};;${real(poisson)};7800.`);
  lines.push(...records(node, nx));
  // The twelve edges of the block are sharp: the skin is read as a smooth surface across every other edge.
  const sharpEdges = [];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i < nx; ++i) sharpEdges.push([node(i, j, k), node(i + 1, j, k)]);
  for (const i of [0, nx])
    for (let t = 0; t <= 1; ++t) sharpEdges.push([node(i, 0, t), node(i, 1, t)], [node(i, t, 0), node(i, t, 1)]);
  const options = { elementFormulation: "MixedHybridFormulation", formulationGeometry: { sharpEdges } };
  return { text: lines.join("\n") + "\n", node, nx, points, options };
}

function runSolids(model, name) {
  return withModel(model.text, name, model.options, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return run;
  });
}

for (const shape of ["SolidHexa", "SolidPenta", "SolidTetra", "SolidPyramid"])
  test(`a block of ${shape} in MixedHybridFormulation stretched by its supports takes the uniform uniaxial stress`, () => {
    const delta = 1.0e-3;
    const poisson = 0.3;
    const model = solidBlock({ shape, poisson, records: (node, nx) => {
      const records = [`SinglePointConstrList;1;23;${node(0, 0, 0)}`, `SinglePointConstrList;1;3;${node(0, 1, 0)}`,
        `SinglePointConstrList;1;2;${node(0, 0, 1)}`];
      for (let k = 0; k <= 1; ++k)
        for (let j = 0; j <= 1; ++j)
          records.push(`SinglePointConstrList;1;1;${node(0, j, k)}`, `SinglePointConstr;1;${node(nx, j, k)};1;${real(delta)}`);
      return records;
    } });
    const run = runSolids(model, `mixed-solid-stretch-${shape}`);
    const strain = delta / BLOCK.length;
    for (const { id, x, y, z } of model.points)
      assertClose(nodal(run, "displacements", id), [strain * x, -poisson * strain * y, -poisson * strain * z, 0, 0, 0],
        1e-8, `grid ${id}`, delta);
    let pull = 0;
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j) pull += nodal(run, "singlePointConstrForces", model.node(model.nx, j, k))[0];
    const expected = BLOCK.young * BLOCK.height * BLOCK.width * strain;
    assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
    const rows = rowsOf(run, shape, "stress");
    assert.ok(rows.length > 0, "stress rows");
    for (const row of rows)
      assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz], [BLOCK.young * strain, 0, 0, 0, 0, 0], 1e-8,
        `stresses of element ${row.id} at ${row.nodeId}`);
    // The table of the grid points (spec section 10.3): every grid point of the solids, the mean of their corners.
    const grid = rowsOf(run, "SolidGridPoints", "stress");
    assert.deepEqual(grid.map((row) => row.id).sort((a, b) => a - b), model.points.map((p) => p.id).sort((a, b) => a - b));
    for (const row of grid)
      assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz], [BLOCK.young * strain, 0, 0, 0, 0, 0], 1e-8,
        `stresses at grid point ${row.id}`);
  });

test("a block of SolidHexa in MixedHybridFormulation bent by its supports takes the linear stress of a pure bending", () => {
  // u = −κ x y', v = κ (x² + ν (y'² − z'²)) / 2, w = κ ν y' z' (y', z' from the centre of the section): the supports
  // give u on both ends, and v, w at two grid points of the end x = 0.
  const kappa = 1.0e-3;
  const poisson = 0.3;
  const field = (x, y, z) => {
    const yc = y - BLOCK.height / 2;
    const zc = z - BLOCK.width / 2;
    return [-kappa * x * yc, (kappa * (x * x + poisson * (yc * yc - zc * zc))) / 2, kappa * poisson * yc * zc];
  };
  const model = solidBlock({ shape: "SolidHexa", nx: 2, poisson, records: (node, nx) => {
    const at = (i, j, k) => field((BLOCK.length * i) / nx, BLOCK.height * j, BLOCK.width * k);
    const records = [];
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j)
        for (const i of [0, nx]) records.push(`SinglePointConstr;1;${node(i, j, k)};1;${real(at(i, j, k)[0])}`);
    records.push(`SinglePointConstr;1;${node(0, 0, 0)};2;${real(at(0, 0, 0)[1])}`,
      `SinglePointConstr;1;${node(0, 0, 0)};3;${real(at(0, 0, 0)[2])}`,
      `SinglePointConstr;1;${node(0, 1, 0)};3;${real(at(0, 1, 0)[2])}`);
    return records;
  } });
  const run = runSolids(model, "mixed-solid-bend");
  for (const { id, x, y, z } of model.points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), field(x, y, z), 1e-8, `grid ${id}`, kappa);
  for (const row of rowsOf(run, "SolidHexa", "stress")) {
    const point = row.nodeId === 0 ? null : model.points.find((p) => p.id === row.nodeId);
    const y = point ? point.y : BLOCK.height / 2;
    assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz],
      [-BLOCK.young * kappa * (y - BLOCK.height / 2), 0, 0, 0, 0, 0], 1e-8,
      `stresses of element ${row.id} at ${row.nodeId}`, BLOCK.young * kappa * BLOCK.height);
  }
});

test("a free block of SolidHexa in MixedHybridFormulation has six rigid-body modes", () => {
  const model = solidBlock({ shape: "SolidHexa" });
  const text = model.text.replace("SolutionSequence;101", "SolutionSequence;103")
    .replace(/SubcaseStart;1\nSinglePointConstrSelection;1\n/, "SubcaseStart;1\nEigenMethodSelection;3\n") +
    "LanczosEigenMethod;3;;;8\n";
  withModel(text, "mixed-solid-free", model.options, (model) => {
    const run = solver.run(model, "modes");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    assert.ok(values.length >= 7, `${values.length} eigenvalues`);
    for (let k = 0; k < 6; ++k)
      assert.ok(Math.abs(values[k]) < 1e-6 * values[6], `eigenvalue ${k + 1} is ${values[k]}, the first elastic ${values[6]}`);
  });
});

test("the weight of a strip of ShellQuad4 in MixedHybridFormulation is its mass with that of its NSM cards", () => {
  // Under a uniform gravity the supports carry the weight (ρ t + m) L b g, m the mass per area of the NSM set.
  const added = 5.0;
  const gravity = 9.81;
  for (const withNsm of [false, true]) {
    const model = cantileverStrip({ loads: () => [`GravityLoad;2;;${real(gravity)};0.;0.;-1.`].concat(
      withNsm ? [`NonStructMass;20;ShellProp;1;${real(added)}`] : []) });
    const text = loadCase(model.text).replace("StaticLoadSelection;2\n",
      withNsm ? "StaticLoadSelection;2\nNonStructMassSelection;20\n" : "StaticLoadSelection;2\n");
    const run = runShells(text, "mixed-shell-nsm");
    let carried = 0;
    for (let j = 0; j <= model.ny; ++j) carried += nodal(run, "singlePointConstrForces", model.node(0, j))[2];
    const expected = (2700 * STRIP.thickness + (withNsm ? added : 0)) * STRIP.length * STRIP.width * gravity;
    assert.ok(Math.abs(carried / expected - 1) < 1e-9, `carried ${carried}, expected ${expected} (NSM ${withNsm})`);
  }
});

test("the geometry report of MixedHybridFormulation names the fold of an angle section, rounded unless declared raw", () => {
  // Two legs of nx ShellQuad4 at 90 degrees along the x axis: the leg z = 0 and the leg y = 0, clamped at x = 0.
  const nx = 2;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  const fold = (i) => i + 1;
  const flat = (i) => nx + 2 + i;
  const upright = (i) => 2 * (nx + 1) + 1 + i;
  for (let i = 0; i <= nx; ++i) {
    const x = real(i / nx);
    lines.push(`Node;${fold(i)};;${x};0.;0.`, `Node;${flat(i)};;${x};0.3;0.`, `Node;${upright(i)};;${x};0.;0.3`);
  }
  for (let i = 0; i < nx; ++i)
    lines.push(`ShellQuad4;${i + 1};1;${fold(i)};${fold(i + 1)};${flat(i + 1)};${flat(i)}`,
      `ShellQuad4;${nx + i + 1};1;${fold(i)};${upright(i)};${upright(i + 1)};${fold(i + 1)}`);
  lines.push("ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.");
  for (const g of [fold(0), flat(0), upright(0)]) lines.push(`SinglePointConstrList;1;123456;${g}`);
  lines.push(`PointForceLoad;2;${fold(nx)};;1.;0.;0.;-1.`);
  const rawEdges = [];
  for (let i = 0; i < nx; ++i) rawEdges.push([fold(i), fold(i + 1)]);
  const diagnostics = (formulationGeometry) => withModel(lines.join("\n") + "\n", "mixed-shell-report",
    { elementFormulation: "MixedHybridFormulation", formulationGeometry }, (model) => {
      const run = solver.run(model, "statics");
      assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
      return run.diagnostics;
    });
  // Declared raw: the fold is kept, a kink of 90 degrees along each of its edges.
  const declared = diagnostics({ rawEdges });
  const report = declared.find((d) => d.code === "geometry-report");
  assert.ok(report, JSON.stringify(declared.map((d) => d.code)));
  assert.match(report.text, /2 declared raw/);
  const kinks = declared.filter((d) => d.code === "geometry-kink");
  assert.equal(kinks.length, nx, JSON.stringify(kinks));
  for (const kink of kinks) assert.match(kink.text, /differ by up to (89\.9|90\.0)\d* degrees \(declared raw\)/);
  assert.equal(declared.filter((d) => d.code === "geometry-rounding").length, 0, "no rounding of the legs");
  // Not declared: the reading rounds the fold, about 45 degrees from the plane of each leg along it.
  const rounded = diagnostics({}).filter((d) => d.code === "geometry-rounding");
  assert.ok(rounded.length >= nx, JSON.stringify(rounded));
  assert.match(rounded[0].text, /turns by up to (4[0-9]|5[0-9])\.\d+ degrees/);
});

for (const relief of ["-1", "-2"])
  test(`a free strip of ShellQuad4 in MixedHybridFormulation under a uniform pressure balanced by inertia relief ${relief} does not deform`, () => {
    // The pressure p and the inertia −ρ t a of the uniform acceleration that balances it cancel at every point: the
    // consistent mass and the consistent pressure load of the shells make that exact.
    const model = strip({ poisson: 0.3, records: (node, nx, ny) => {
      const records = [`ShellPressureLoad;2;100.;1;Through;${nx * ny}`, `Parameter;InertiaRelief;${relief}`];
      if (relief === "-1")
        records.push(`FreeBodySupport;${node(0, 0)};123;${node(nx, 0)};23;${node(0, ny)};3`);
      return records;
    } });
    const text = model.text.replace("SinglePointConstrSelection;1\n", "StaticLoadSelection;2\n")
      .replace("SinglePointConstrForceOutput;All", "ElemForceOutput;All");
    const run = runShells(text, `mixed-shell-relief${relief}`);
    // The scale: the deflection of the strip clamped at one end under that pressure, p L⁴ / (8 D).
    const scale = (100.0 * STRIP.length ** 4) / (8 * (STRIP.young * STRIP.thickness ** 3) / 12);
    for (const { id } of model.points) {
      const u = nodal(run, "displacements", id).slice(0, 3);
      assert.ok(Math.max(...u.map(Math.abs)) < 1e-9 * scale, `grid ${id}: ${u}`);
    }
    for (const row of rowsOf(run, "ShellQuad4", "forces"))
      assert.ok(Math.abs(row.mx) + Math.abs(row.my) < 1e-9 * 100.0 * STRIP.length ** 2, `element ${row.id}: ${row.mx}`);
  });

// A plate wing of n × n ShellQuad4 (1 × 2, clamped along its root y = 0) under the aerodynamic load of a lifting panel
// of 2 × 2 boxes splined to every grid point, trimmed at a given angle of attack.
function trimWing(n) {
  const deck = oor.createModel();
  deck.setSolution(144);
  deck.caseControl.global.set("SinglePointConstrSelection", 1);
  deck.caseControl.global.set("TrimSelection", 5);
  deck.caseControl.global.set("DisplacementOutput", "All");
  const id = (i, j) => j * (n + 1) + i + 1;
  const ids = [];
  const xs = [];
  const ys = [];
  for (let j = 0; j <= n; ++j)
    for (let i = 0; i <= n; ++i) {
      ids.push(id(i, j));
      xs.push(i / n);
      ys.push((2 * j) / n);
    }
  deck.addCardBatch("Node", { id: ids, position1: xs, position2: ys, position3: 0 });
  const cards = [];
  let element = 0;
  for (let j = 0; j < n; ++j)
    for (let i = 0; i < n; ++i)
      cards.push({ type: "ShellQuad4", fields: [++element, 1, id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)] });
  for (let i = 0; i <= n; ++i) cards.push({ type: "SinglePointConstrList", fields: [1, 123456, id(i, 0)] });
  cards.push(
    { type: "ShellProp", fields: [1, 1, 0.1, 1, null, 1] },
    { type: "IsoMat", fields: [1, 1e6, null, 0.3, 1] },
    { type: "AeroStaticParams", fields: [0, 0, 1, 2, 2] },
    { type: "AeroLiftingPanelProp1", fields: [10] },
    { type: "AeroLiftingPanel1", fields: [100, 10, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 1] },
    { type: "IdListSet1", fields: [50, ...ids] },
    { type: "AeroSurfaceSpline1", fields: [60, 100, 100, 103, 50] },
    { type: "AeroRigidTrimVariable", fields: [70, "AngleOfAttack"] },
    { type: "AeroTrim", fields: [5, 0.1, 10, "AngleOfAttack", 0.05] });
  deck.addCards(cards);
  deck.applyDefaults();
  return { deck, tip: id(n, n) };
}

test("a plate wing of ShellQuad4 in MixedHybridFormulation trims as the ClassicFormulation one on a fine mesh", () => {
  const tipOf = ({ deck, tip }, options) => {
    try {
      const model = solver.modelFromReader(oor, deck, options);
      try {
        const run = solver.run(model, "aeroelastic");
        assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
        const table = run.nodalTables.find((t) => t.kind === "displacements");
        const row = table.ids.indexOf(tip);
        return table.values[row * 6 + 2];
      } finally { model.delete(); }
    } finally { deck.delete(); }
  };
  const classic = tipOf(trimWing(8), {});
  // At least 4 × 4: the spline reads the structure at its grid points only, too few on a coarser mesh.
  const mixed = tipOf(trimWing(4), SHELL_OPTIONS);
  assert.ok(Number.isFinite(mixed) && Math.abs(mixed / classic - 1) < 0.01, `tip ${mixed}, classic ${classic}`);
});

for (const reversed of [false, true])
  test(`a PointsPressureLoad on the shells of MixedHybridFormulation${reversed ? " given in the other order" : ""} acts through their pressure facet`, () => {
    // P times the vector area of the order of its grid points: along the normal of the shells, or against it.
    const p = 200.0;
    const model = cantileverStrip({ loads: (nx, ny) => {
      const node = (i, j) => j * (nx + 1) + i + 1;
      const records = [];
      for (let j = 0; j < ny; ++j)
        for (let i = 0; i < nx; ++i) {
          const face = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
          records.push(`PointsPressureLoad;2;${real(p)};${(reversed ? face.reverse() : face).join(";")}`);
        }
      return records;
    } });
    const run = runShells(loadCase(model.text), "mixed-shell-points-pressure");
    const d = (STRIP.young * STRIP.thickness ** 3) / 12;
    const g = STRIP.young / 2;
    const expected = (reversed ? -1 : 1) * ((p * STRIP.length ** 4) / (8 * d) +
      (p * STRIP.length ** 2) / (2 * SHEAR_FACTOR * g * STRIP.thickness));
    for (let j = 0; j <= model.ny; ++j) {
      const w = nodal(run, "displacements", model.node(model.nx, j))[2];
      assert.ok(Math.abs(w / expected - 1) < 1e-8, `tip deflection ${w}, expected ${expected}`);
    }
  });

test("a PointsPressureLoad on the end face of a block of SolidHexa in MixedHybridFormulation pulls it uniformly", () => {
  // The face x = L in the order whose vector area points out of the block: P > 0 pulls, σx = P.
  const p = 1.0e6;
  const poisson = 0.3;
  const model = solidBlock({ shape: "SolidHexa", poisson, records: (node, nx) => {
    const records = [`SinglePointConstrList;1;23;${node(0, 0, 0)}`, `SinglePointConstrList;1;3;${node(0, 1, 0)}`,
      `SinglePointConstrList;1;2;${node(0, 0, 1)}`,
      `PointsPressureLoad;2;${real(p)};${node(nx, 0, 0)};${node(nx, 1, 0)};${node(nx, 1, 1)};${node(nx, 0, 1)}`];
    for (let k = 0; k <= 1; ++k) for (let j = 0; j <= 1; ++j) records.push(`SinglePointConstrList;1;1;${node(0, j, k)}`);
    return records;
  } });
  model.text = model.text.replace("SinglePointConstrSelection;1\n", "SinglePointConstrSelection;1\nStaticLoadSelection;2\n");
  const run = runSolids(model, "mixed-solid-points-pressure");
  const strain = p / BLOCK.young;
  for (const { id, x, y, z } of model.points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [strain * x, -poisson * strain * y, -poisson * strain * z],
      1e-8, `grid ${id}`, strain * BLOCK.length);
});

// ---------------------------------------------------------------------------------------------------------------------
// Ties (spec section 3.4): the solids meet the shells and the lines along their edges and faces.

const SKIN = { thickness: 0.02 };
const STIFFENING_BAR = { area: 0.05, i1: 2.0e-4, i2: 2.0e-4, j: 3.0e-4 };

// The records of a block stretched by its supports (the uniaxial state of the stretch tests of the solids).
function stretchRecords(node, nx, delta) {
  const records = [`SinglePointConstrList;1;23;${node(0, 0, 0)}`, `SinglePointConstrList;1;3;${node(0, 1, 0)}`,
    `SinglePointConstrList;1;2;${node(0, 0, 1)}`];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      records.push(`SinglePointConstrList;1;1;${node(0, j, k)}`, `SinglePointConstr;1;${node(nx, j, k)};1;${real(delta)}`);
  return records;
}

for (const partner of ["a shell on its top face", "a Bar2 along its top edge"])
  test(`a block of SolidHexa in MixedHybridFormulation with ${partner} stretched by its supports takes the uniform stress`, () => {
    const delta = 1.0e-3;
    const poisson = 0.3;
    const model = solidBlock({ shape: "SolidHexa", nx: 2, poisson, records: (node, nx) => {
      const records = stretchRecords(node, nx, delta);
      if (partner.startsWith("a shell")) {
        records.push(`ShellProp;2;1;${real(SKIN.thickness)};1;;1`);
        for (let i = 0; i < nx; ++i)
          records.push(`ShellQuad4;${100 + i};2;${node(i, 0, 1)};${node(i + 1, 0, 1)};${node(i + 1, 1, 1)};${node(i, 1, 1)}`);
      } else {
        records.push(`BarProp;2;1;${real(STIFFENING_BAR.area)};${real(STIFFENING_BAR.i1)};${real(STIFFENING_BAR.i2)};${real(STIFFENING_BAR.j)}`);
        for (let i = 0; i < nx; ++i) records.push(`Bar2;${100 + i};2;${node(i, 0, 1)};${node(i + 1, 0, 1)};0.;;0.;1.`);
      }
      return records;
    } });
    const run = runSolids(model, `mixed-tie-stretch-${partner.split(" ")[1]}`);
    const strain = delta / BLOCK.length;
    for (const { id, x, y, z } of model.points)
      assertClose(nodal(run, "displacements", id).slice(0, 3), [strain * x, -poisson * strain * y, -poisson * strain * z],
        1e-8, `grid ${id}`, delta);
    let pull = 0;
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j) pull += nodal(run, "singlePointConstrForces", model.node(model.nx, j, k))[0];
    const added = partner.startsWith("a shell") ? SKIN.thickness * BLOCK.width : STIFFENING_BAR.area;
    const expected = BLOCK.young * (BLOCK.height * BLOCK.width + added) * strain;
    assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
  });

test("a free block of SolidHexa in MixedHybridFormulation with a shell fin on its edge has six rigid-body modes and no mechanism", () => {
  // The fin stands on the edge y = 0, z = W of the block, in the plane y = 0: its rotation about that edge follows the
  // rotation of the solid (tie 4); left free, it would turn about the edge as a hinge, a seventh mode of zero energy.
  const model = solidBlock({ shape: "SolidHexa", records: (node, nx) => {
    const tip = 4 * (nx + 1);
    return ["ShellProp;2;1;0.05;1;;1", `Node;${tip + 1};;0.;0.;1.5`, `Node;${tip + 2};;${real(BLOCK.length)};0.;1.5`,
      `ShellQuad4;100;2;${node(0, 0, 1)};${node(nx, 0, 1)};${tip + 2};${tip + 1}`];
  } });
  const text = model.text.replace("SolutionSequence;101", "SolutionSequence;103")
    .replace(/SubcaseStart;1\nSinglePointConstrSelection;1\n/, "SubcaseStart;1\nEigenMethodSelection;3\n") +
    "LanczosEigenMethod;3;;;8\n";
  withModel(text, "mixed-tie-fin", model.options, (model) => {
    const run = solver.run(model, "modes");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    assert.ok(values.length >= 7, `${values.length} eigenvalues`);
    for (let k = 0; k < 6; ++k)
      assert.ok(Math.abs(values[k]) < 1e-6 * values[6], `eigenvalue ${k + 1} is ${values[k]}, the first elastic ${values[6]}`);
    assert.ok(values[6] > 1e-3 * values[7], `the seventh eigenvalue ${values[6]} is a mechanism (eighth ${values[7]})`);
  });
});
