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
      /formulationGeometry\.rawEdges: each item is \[g1, g2\]/);
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { sharpEdges: [[1]] } }),
      /formulationGeometry\.sharpEdges: each item is \[g1, g2\] or \[g1, g2, element\]/);
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { lineContinuations: [[1, 2]] } }),
      /formulationGeometry\.lineContinuations: each item is \[g1, g, g2\]/);
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { lineContinuations: [[1, 2, 1]] } }),
      /three different grid points/);
    assert.throws(() => solver.modelFromReader(oor, deck, { formulationGeometry: { lineEnds: [[2, 2]] } }),
      /two different grid points/);
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

for (const shape of ["SolidHexa", "SolidTetra"])
  test(`the edges of a block of ${shape} in MixedHybridFormulation are sharp by the fold angle of the model as when they are declared`, () => {
    const delta = 1.0e-3;
    const model = solidBlock({ shape, nx: 2, records: (node, nx) => {
      const records = [`SinglePointConstrList;1;23;${node(0, 0, 0)}`, `SinglePointConstrList;1;3;${node(0, 1, 0)}`,
        `SinglePointConstrList;1;2;${node(0, 0, 1)}`];
      for (let k = 0; k <= 1; ++k)
        for (let j = 0; j <= 1; ++j)
          records.push(`SinglePointConstrList;1;1;${node(0, j, k)}`, `SinglePointConstr;1;${node(nx, j, k)};1;${real(delta)}`);
      return records;
    } });
    const undeclared = { ...model, options: { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} } };
    const declared = runSolids(model, `mixed-solid-declared-${shape}`);
    const byAngle = runSolids(undeclared, `mixed-solid-angle-${shape}`);
    for (const { id } of model.points)
      assertClose(nodal(byAngle, "displacements", id), nodal(declared, "displacements", id), 1e-12, `grid ${id}`, delta);
    assert.deepEqual(byAngle.diagnostics.filter((d) => d.severity !== "information"), []);
    assert.match(byAngle.diagnostics.find((d) => d.code === "geometry-report").text,
      /the skin of the solids of MixedHybridFormulation has \d+ edges between two faces, 16 sharp \(16 by the fold angle of the model alone\)$/);
    // The table of the edges read: the sixteen edges of the block are sharp, by the angle or by the declaration.
    const key = (a, b) => `${Math.min(a, b)}-${Math.max(a, b)}`;
    const sharp = new Set(model.options.formulationGeometry.sharpEdges.map(([a, b]) => key(a, b)));
    for (const [candidate, origin] of [[undeclared, "Angle"], [model, "Declared"]])
      withModel(candidate.text, `mixed-solid-edges-${shape}-${origin}`, candidate.options, (built) => {
        const edges = built.surfaceEdges();
        assert.ok(edges.grid1.length > 0 && edges.surface.every((surface) => surface === "SolidSkin"));
        for (let row = 0; row < edges.grid1.length; ++row) {
          const isBlockEdge = sharp.has(key(edges.grid1[row], edges.grid2[row]));
          assert.ok(edges.grid1[row] < edges.grid2[row]);
          assert.equal(edges.state[row], isBlockEdge ? "Sharp" : "Smooth", `edge ${edges.grid1[row]}-${edges.grid2[row]}`);
          assert.equal(edges.origin[row], isBlockEdge ? origin : "Angle");
          assert.ok(Math.abs(edges.angle[row] - (isBlockEdge ? 90 : 0)) < 1e-5, `angle ${edges.angle[row]}`);
        }
      });
  });

// The pure bending of a block: u = −κ x y', v = κ (x² + ν (y'² − z'²)) / 2, w = κ ν y' z' (y', z' from the centre of
// the section), and the supports that impose it: u on both ends, and v, w at two grid points of the end x = 0.
const BENT_BLOCK = { kappa: 1.0e-3, poisson: 0.3 };
function bentBlockField(x, y, z) {
  const { kappa, poisson } = BENT_BLOCK;
  const yc = y - BLOCK.height / 2;
  const zc = z - BLOCK.width / 2;
  return [-kappa * x * yc, (kappa * (x * x + poisson * (yc * yc - zc * zc))) / 2, kappa * poisson * yc * zc];
}
function bentBlockRecords(node, nx) {
  const at = (i, j, k) => bentBlockField((BLOCK.length * i) / nx, BLOCK.height * j, BLOCK.width * k);
  const records = [];
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (const i of [0, nx]) records.push(`SinglePointConstr;1;${node(i, j, k)};1;${real(at(i, j, k)[0])}`);
  records.push(`SinglePointConstr;1;${node(0, 0, 0)};2;${real(at(0, 0, 0)[1])}`,
    `SinglePointConstr;1;${node(0, 0, 0)};3;${real(at(0, 0, 0)[2])}`,
    `SinglePointConstr;1;${node(0, 1, 0)};3;${real(at(0, 1, 0)[2])}`);
  return records;
}

test("a block of SolidHexa in MixedHybridFormulation bent by its supports takes the linear stress of a pure bending", () => {
  const { kappa, poisson } = BENT_BLOCK;
  const field = bentBlockField;
  const model = solidBlock({ shape: "SolidHexa", nx: 2, poisson, records: bentBlockRecords });
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

test("the geometry report of MixedHybridFormulation names the fold of an angle section: sharp by the fold angle, raw or rounded as declared", () => {
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
  // Not declared: the fold angle of the model makes the fold sharp, and the legs stay flat.
  const byAngle = diagnostics({});
  assert.match(byAngle.find((d) => d.code === "geometry-report").text,
    /0 declared raw, 2 sharp \(2 by the fold angle of the model alone\)/);
  const sharpKinks = byAngle.filter((d) => d.code === "geometry-kink");
  assert.equal(sharpKinks.length, nx, JSON.stringify(sharpKinks));
  for (const kink of sharpKinks)
    assert.match(kink.text, /differ by up to (89\.9|90\.0)\d* degrees \(sharp by the fold angle of the model\)/);
  assert.equal(byAngle.filter((d) => d.code === "geometry-rounding").length, 0, "no rounding of the legs");
  // a fold of 90 degrees is far above the angle: nothing asks whether it is a coarse curve
  assert.equal(byAngle.filter((d) => d.code === "geometry-fold-by-angle").length, 0);
  // Declared smooth: the reading rounds the fold, about 45 degrees from the plane of each leg along it.
  const rounded = diagnostics({ smoothEdges: rawEdges }).filter((d) => d.code === "geometry-rounding");
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

test("a block of SolidHexa and a separate strip of ShellQuad4 in MixedHybridFormulation stretched by their supports take their uniform stresses", () => {
  // The two parts share no grid point: the elements of each family read their own entities on the whole g set (the
  // estimate of the output volume and the recovery give them the entity values too).
  const delta = 1.0e-3;
  const poisson = 0.3;
  const lift = 3.0; // the strip lies in the plane z = lift, above the block
  const stripNode = (i, j) => 200 + 3 * j + i; // a strip of 2 × 1 shells, STRIP.length by STRIP.width
  const stripPoints = [];
  const model = solidBlock({ shape: "SolidHexa", poisson, records: (node, nx) => {
    const records = stretchRecords(node, nx, delta);
    records.push(`ShellProp;2;2;${real(STRIP.thickness)};2;;2`, `IsoMat;2;${real(STRIP.young)};;${real(poisson)};2700.`);
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i <= 2; ++i) {
        const point = { id: stripNode(i, j), x: (STRIP.length * i) / 2, y: STRIP.width * j };
        stripPoints.push(point);
        records.push(`Node;${point.id};;${real(point.x)};${real(point.y)};${real(lift)}`,
          `SinglePointConstrList;1;3;${point.id}`);
      }
    for (let i = 0; i < 2; ++i)
      records.push(`ShellQuad4;${100 + i};2;${stripNode(i, 0)};${stripNode(i + 1, 0)};${stripNode(i + 1, 1)};${stripNode(i, 1)}`);
    records.push(`SinglePointConstrList;1;2;${stripNode(0, 0)}`);
    for (let j = 0; j <= 1; ++j)
      records.push(`SinglePointConstrList;1;1;${stripNode(0, j)}`, `SinglePointConstr;1;${stripNode(2, j)};1;${real(delta)}`);
    return records;
  } });
  const run = runSolids(model, "mixed-block-and-strip");
  const strain = delta / BLOCK.length;
  for (const { id, x, y, z } of model.points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [strain * x, -poisson * strain * y, -poisson * strain * z],
      1e-8, `grid ${id}`, delta);
  for (const row of rowsOf(run, "SolidHexa", "stress"))
    assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz], [BLOCK.young * strain, 0, 0, 0, 0, 0], 1e-8,
      `stresses of element ${row.id} at ${row.nodeId}`);
  const stripStrain = delta / STRIP.length;
  for (const { id, x, y } of stripPoints)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [stripStrain * x, -poisson * stripStrain * y, 0], 1e-8,
      `grid ${id}`, delta);
  const shellRows = rowsOf(run, "ShellQuad4", "stress");
  assert.ok(shellRows.length > 0, "shell stress rows");
  for (const row of shellRows)
    assertClose([row.oxx, row.oyy, row.txy], [STRIP.young * stripStrain, 0, 0], 1e-8,
      `stresses of element ${row.id} at ${row.nodeId}, fibre ${row.layer}`);
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

// ---------------------------------------------------------------------------------------------------------------------
// Membrane-only and bending-only shells (spec section 4.1): the membrane alone has no moments, no transverse shear and
// no rotation about its edges; the bending alone has no membrane stiffness. The formulation holds the entity directions
// no element stiffens (the normal of a flat membrane, the plane of a flat bending-only plate), the singularity processor
// those of the grid points.

const FULL_SECTION = `ShellProp;1;1;${real(STRIP.thickness)};1;;1`;
const MEMBRANE_ONLY = `ShellProp;1;1;${real(STRIP.thickness)};;;`;
const BENDING_ONLY = `ShellProp;1;;${real(STRIP.thickness)};1;;1`;
function withSection(model, section) {
  assert.ok(model.text.includes(FULL_SECTION), "the strip's shell property");
  return { ...model, text: model.text.replace(FULL_SECTION, section) };
}
const heldDirections = (run) => run.diagnostics.find((d) => d.code === "unstiffened-directions-held");

for (const { shape, stiffened } of STRETCH_CASES)
  test(`a strip of membrane-only ${shape} in MixedHybridFormulation${stiffened ? " stiffened by Bar2 on its edge" : ""} stretched by its supports takes the uniform uniaxial stress`, () => {
    // No normal displacement held: the grid points' by the singularity processor, the edges' and interiors' by the
    // formulation. The stiffeners are beams that the membrane does not hold out of its plane: their ends hold their
    // deflection and one end their twist.
    const delta = 1.0e-3;
    const poisson = 0.3;
    const model = withSection(strip({ shape, poisson, records: (node, nx, ny) => {
      const records = [`SinglePointConstrList;1;2;${node(0, 0)}`];
      for (let j = 0; j <= ny; ++j)
        records.push(`SinglePointConstrList;1;1;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};1;${real(delta)}`);
      return stiffened
        ? records.concat(stiffeners(node, nx), [`SinglePointConstrList;1;34;${node(0, 0)}`, `SinglePointConstrList;1;3;${node(nx, 0)}`])
        : records;
    } }), MEMBRANE_ONLY);
    const run = runShells(model.text, `mixed-membrane-stretch-${shape}`);
    assert.ok(heldDirections(run), JSON.stringify(run.diagnostics.map((d) => d.code)));
    for (const { id, x, y } of model.points)
      assertClose(nodal(run, "displacements", id).slice(0, 3),
        [(delta * x) / STRIP.length, (-poisson * delta * y) / STRIP.length, 0], 1e-8, `grid ${id}`, delta);
    let pull = 0;
    for (let j = 0; j <= model.ny; ++j) pull += nodal(run, "singlePointConstrForces", model.node(model.nx, j))[0];
    const area = STRIP.thickness * STRIP.width + (stiffened ? STIFFENER.area : 0);
    const expected = (STRIP.young * area * delta) / STRIP.length;
    assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
  });

for (const shape of ["ShellQuad4", "ShellTria3", "ShellQuad8", "ShellTria6"])
  test(`a strip of bending-only ${shape} in MixedHybridFormulation bent by its supports takes the uniform moment`, () => {
    // No in-plane displacement held but at the clamped end.
    const theta = 1.0e-3;
    const model = withSection(strip({ shape, poisson: 0, records: (node, nx, ny) => {
      const records = [];
      for (let j = 0; j <= ny; ++j)
        records.push(`SinglePointConstrList;1;123456;${node(0, j)}`, `SinglePointConstr;1;${node(nx, j)};5;${real(-theta)}`);
      return records;
    } }), BENDING_ONLY);
    const run = runShells(model.text, `mixed-bending-bend-${shape}`);
    assert.ok(heldDirections(run), JSON.stringify(run.diagnostics.map((d) => d.code)));
    for (const { id, x } of model.points)
      assertClose(nodal(run, "displacements", id).slice(0, 3), [0, 0, (theta * x * x) / (2 * STRIP.length)], 1e-8,
        `grid ${id}`, theta);
    let moment = 0;
    for (let j = 0; j <= model.ny; ++j) moment += nodal(run, "singlePointConstrForces", model.node(model.nx, j))[4];
    const expected = -(STRIP.young * STRIP.thickness ** 3 * STRIP.width * theta) / (12 * STRIP.length);
    assert.ok(Math.abs(moment / expected - 1) < 1e-8, `moment ${moment}, expected ${expected}`);
  });

test("a cantilever strip of bending-only ShellQuad4 in MixedHybridFormulation under a uniform pressure bends as the Timoshenko beam with no membrane force", () => {
  const p = 200.0;
  const model = withSection(cantileverStrip({ loads: (nx, ny) => [`ShellPressureLoad;2;${real(p)};1;Through;${nx * ny}`] }),
    BENDING_ONLY);
  const text = loadCase(model.text).replace("SinglePointConstrForceOutput;All",
    "SinglePointConstrForceOutput;All\nElemForceOutput;All;CornerResults");
  const run = runShells(text, "mixed-bending-pressure");
  const d = (STRIP.young * STRIP.thickness ** 3) / 12;
  const g = STRIP.young / 2;
  const expected = (p * STRIP.length ** 4) / (8 * d) + (p * STRIP.length ** 2) / (2 * SHEAR_FACTOR * g * STRIP.thickness);
  for (let j = 0; j <= model.ny; ++j) {
    const w = nodal(run, "displacements", model.node(model.nx, j))[2];
    assert.ok(Math.abs(w / expected - 1) < 1e-8, `tip deflection ${w}, expected ${expected}`);
  }
  const rows = rowsOf(run, "ShellQuad4", "forces");
  assert.ok(rows.length > 0, "force rows");
  for (const row of rows)
    for (const name of ["nx", "ny", "nxy"]) assert.equal(row[name], 0, `${name} of element ${row.id} at ${row.nodeId}`);
});

test("an octant of a sphere of membrane-only ShellTria3 in MixedHybridFormulation under internal pressure takes the membrane stress", () => {
  // Radius R, held by symmetry on its three edges (u_x = 0 on x = 0, u_y = 0 on y = 0, u_z = 0 on z = 0): N = p R / 2
  // in every direction, the radial displacement p R² (1 − ν) / (2 E t) everywhere. A membrane sphere has no
  // inextensional motion, so no mechanism (an open membrane cylinder would have them: its rings ovalise freely).
  const radius = 1.0;
  const thickness = 0.01;
  const young = 7.0e10;
  const poisson = 0.3;
  const pressure = 1.0e5;
  const n = 6;
  // Grid (i, j), i + j ≤ n: the point of the sphere along i X + j Y + (n − i − j) Z.
  const ids = new Map();
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  const points = [];
  for (let i = 0; i <= n; ++i)
    for (let j = 0; i + j <= n; ++j) {
      const v = [i, j, n - i - j];
      const length = Math.hypot(...v);
      const p = v.map((c) => (radius * c) / length);
      const id = ids.size + 1;
      ids.set(`${i},${j}`, id);
      points.push({ id, p });
      lines.push(`Node;${id};;${real(p[0])};${real(p[1])};${real(p[2])}`);
      if (i === 0) lines.push(`SinglePointConstrList;1;1;${id}`);
      if (j === 0) lines.push(`SinglePointConstrList;1;2;${id}`);
      if (i + j === n) lines.push(`SinglePointConstrList;1;3;${id}`);
    }
  const node = (i, j) => ids.get(`${i},${j}`);
  let element = 0;
  // Their normal points outward: the internal pressure acts along it.
  for (let i = 0; i < n; ++i)
    for (let j = 0; i + j < n; ++j) {
      lines.push(`ShellTria3;${++element};1;${node(i, j)};${node(i + 1, j)};${node(i, j + 1)}`);
      if (i + j < n - 1) lines.push(`ShellTria3;${++element};1;${node(i + 1, j)};${node(i + 1, j + 1)};${node(i, j + 1)}`);
    }
  lines.push(`ShellProp;1;1;${real(thickness)};;;`, `IsoMat;1;${real(young)};;${real(poisson)};2700.`,
    `ShellPressureLoad;2;${real(pressure)};1;Through;${element}`);
  const run = runShells(lines.join("\n") + "\n", "mixed-membrane-sphere");
  const expected = (pressure * radius ** 2 * (1 - poisson)) / (2 * young * thickness);
  for (const { id, p } of points) {
    const u = nodal(run, "displacements", id);
    const radial = (u[0] * p[0] + u[1] * p[1] + u[2] * p[2]) / radius;
    assert.ok(Math.abs(radial / expected - 1) < 1e-3, `radial displacement of grid ${id}: ${radial}, expected ${expected}`);
  }
});

test("a membrane-only and a bending-only ShellQuad4 on the same grid points are refused in MixedHybridFormulation", () => {
  // Two shells on one face would put two shells on one side of its edges: one property with both parts is the way.
  const model = strip({ nx: 1, ny: 1 });
  const text = model.text.replace(FULL_SECTION,
    [MEMBRANE_ONLY, BENDING_ONLY.replace("ShellProp;1;", "ShellProp;2;"), `ShellQuad4;2;2;1;2;4;3`].join("\n"));
  withModel(text, "mixed-overlay", SHELL_OPTIONS, (model) =>
    assert.throws(() => solver.run(model, "statics"), /ShellQuad4 2: .*corner grids of element 1: coincident shells/));
});

// ---------------------------------------------------------------------------------------------------------------------
// The points of the elements of the formulation that other elements read (spec section 5.10): a fastener that pierces a
// shell reads its displacement and rotation over the footprint of its cross-section about the point, a glued
// contact the displacement of a shell or of a solid face at the point, over all their unknowns. On fields the
// elements hold exactly (a pure bending, quadratic) a glued point moves exactly as the field, where the corner grid
// points alone would give their bilinear interpolation, and a fastener reads the mean of the field over its
// footprint.

// A strip bent by its supports (w = θ x² / (2 L), the rotation −θ x / L about y) and the records `extra` added.
const BENT_STRIP_ROTATION = 1.0e-3;
function bentStrip(extra) {
  return strip({ poisson: 0, records: (node, nx, ny) => {
    const records = [];
    for (let j = 0; j <= ny; ++j)
      records.push(`SinglePointConstrList;1;123456;${node(0, j)}`,
        `SinglePointConstr;1;${node(nx, j)};5;${real(-BENT_STRIP_ROTATION)}`);
    return records.concat(extra);
  } });
}
const bentStripDeflection = (x) => (BENT_STRIP_ROTATION * x * x) / (2 * STRIP.length);

test("a Fastener from a bent strip of ShellQuad4 in MixedHybridFormulation reads the motion of the shell over its footprint", () => {
  // A plate above element 2 of the strip (x 0.5 to 1, y 0 to 0.2), joined to it by a fastener at (0.6, 0.07) with GA
  // and GB given: the plate carries nothing, so the fastener carries nothing and GA moves with the shell there. The
  // fastener reads the mean, over the square of the area of its cross-section (its side a = (√π / 2) D), of the
  // motion of the shell carried rigidly to the point, w(X) + w'(X) (x − X) = w(x) − w'' (x − X)² / 2 for this
  // deflection of constant curvature w'' = θ / L: the deflection there less w'' a² / 24 (the mean of (x − X)² over
  // the side is a² / 12), and the rotation of the shell at the point, which is linear.
  const x = 0.6;
  const diameter = 0.1;
  const side = (Math.sqrt(Math.PI) / 2) * diameter;
  const deflection = bentStripDeflection(x) - (BENT_STRIP_ROTATION / STRIP.length) * side ** 2 / 24;
  const model = bentStrip(["Node;101;;0.5;0.;0.05", "Node;102;;1.;0.;0.05", "Node;103;;1.;0.2;0.05",
    "Node;104;;0.5;0.2;0.05", "ShellQuad4;100;2;101;102;103;104", "ShellProp;2;1;0.01;1;;1",
    `Node;110;;${real(x)};0.07;0.025`, `Node;111;;${real(x)};0.07;0.`, `Node;112;;${real(x)};0.07;0.05`,
    "Fastener;201;3;Elements;2;100;110;111;112",
    `FastenerProp;3;${real(diameter)};0;1;1.e5;1.e5;1.e5;1.e3;1.e3;1.e3`]);
  const run = runShells(model.text, "mixed-fastener-bent");
  const slope = (BENT_STRIP_ROTATION * x) / STRIP.length;
  assertClose(nodal(run, "displacements", 111), [0, 0, deflection, 0, -slope, 0], 1e-8, "GA", BENT_STRIP_ROTATION);
  // GB follows the plate, which follows the fastener rigidly: u_B = u_A + θ_A × (x_B − x_A).
  assertClose(nodal(run, "displacements", 112), [-0.05 * slope, 0, deflection, 0, -slope, 0], 1e-8, "GB",
    BENT_STRIP_ROTATION);
});

const GLUE_RECORDS = ["ContactBody;1;ThreeDimensional;Deformable;1", "ContactBody;2;ThreeDimensional;Deformable;2",
  "ContactTable;1;;;;;;Secondary;2;;;;;1;;;;;;;;;;;;;;;;;;;;;;;;;Primary;1;1"];
const withContact = (text) => text.replace("[case]\n", "[case]\nContactSelection;1\n");

test("a plate glued on a bent strip of ShellQuad4 in MixedHybridFormulation moves with the shell at its glued points", () => {
  // A very soft plate on element 2 of the strip, its four corners glued to the strip's surface.
  const corners = [[301, 0.6, 0.07], [302, 0.9, 0.07], [303, 0.9, 0.15], [304, 0.6, 0.15]];
  const model = bentStrip([...corners.map(([id, x, y]) => `Node;${id};;${real(x)};${real(y)};0.`),
    "ShellQuad4;300;2;301;302;303;304", "ShellProp;2;2;0.01;2;;2", "IsoMat;2;7.;;0.;2700.",
    "ContactElemSurface;1;1;Through;8", "ContactElemSurface;2;300", ...GLUE_RECORDS]);
  const run = runShells(withContact(model.text), "mixed-glue-strip");
  assert.ok(run.diagnostics.some((d) => d.code === "glued-pair" && /4 grid points tied/.test(d.text)),
    JSON.stringify(run.diagnostics.map((d) => d.code)));
  for (const [id, x] of corners)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [0, 0, bentStripDeflection(x)], 1e-8, `grid ${id}`,
      BENT_STRIP_ROTATION);
});

test("a plate glued on a face of a bent block of SolidHexa in MixedHybridFormulation moves with the solid at its glued points", () => {
  // A very soft plate on the face z = W of the first cell, its four corners glued to the face.
  const corners = [[301, 0.3, 0.2], [302, 0.8, 0.2], [303, 0.8, 0.7], [304, 0.3, 0.7]];
  const model = solidBlock({ shape: "SolidHexa", nx: 2, poisson: BENT_BLOCK.poisson, records: (node, nx) =>
    bentBlockRecords(node, nx).concat(corners.map(([id, x, y]) => `Node;${id};;${real(x)};${real(y)};${real(BLOCK.width)}`),
      ["ShellQuad4;300;2;301;302;303;304", "ShellProp;2;2;0.01;2;;2", "IsoMat;2;2.;;0.;7800.",
        "ContactPropertyRegion;1;1", "ContactElemSurface;2;300", ...GLUE_RECORDS]) });
  const run = runSolids({ ...model, text: withContact(model.text) }, "mixed-glue-block");
  for (const [id, x, y] of corners)
    assertClose(nodal(run, "displacements", id).slice(0, 3), bentBlockField(x, y, BLOCK.width), 1e-8, `grid ${id}`,
      BENT_BLOCK.kappa);
});

test("two free plates of ShellQuad4 in MixedHybridFormulation joined by a Fastener have six rigid-body modes", () => {
  // The motion of the piercing points is exact for every rigid motion of the shells: the fastener stores no energy.
  const lines = ["[executive]", "SolutionSequence;103", "[case]", "SubcaseStart;1", "EigenMethodSelection;3", "[model]",
    "Node;1;;0.;0.;0.", "Node;2;;1.;0.;0.", "Node;3;;1.;1.;0.2", "Node;4;;0.;1.;0.", "Node;5;;0.;0.;0.1",
    "Node;6;;1.;0.;0.1", "Node;7;;1.;1.;0.3", "Node;8;;0.;1.;0.1", "Node;9;;0.3;0.6;0.08", "Node;10;;0.3;0.6;0.06",
    "Node;11;;0.3;0.6;0.16", "ShellQuad4;1;1;1;2;3;4", "ShellQuad4;2;1;5;6;7;8", "ShellProp;1;1;0.01;1;;1",
    "IsoMat;1;7.e10;;0.3;2700.", "Fastener;101;3;Elements;1;2;9;10;11",
    "FastenerProp;3;0.1;0;1;1.e6;1.e6;1.e6;1.e3;1.e3;1.e3", "LanczosEigenMethod;3;;;8"];
  withModel(lines.join("\n") + "\n", "mixed-fastener-free", SHELL_OPTIONS, (model) => {
    const run = solver.run(model, "modes");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    assert.ok(values.length >= 7, `${values.length} eigenvalues`);
    for (let k = 0; k < 6; ++k)
      assert.ok(Math.abs(values[k]) < 1e-6 * values[6], `eigenvalue ${k + 1} is ${values[k]}, the first elastic ${values[6]}`);
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// Design derivatives (docs/mixed-hybrid-formulation-spec.md section 5.9): the semi-analytic sensitivities difference the
// facets of the mixed hybrid elements on the snapshots of the design point moved by one field (the formulation's model
// built again on each), and solve on the analysis' factors over its entities.

// A cantilever strip of n ShellQuad4 along x (0.1 thick), held at x = 0, its tip loaded by `load`, in the analysis
// `solution` (105: subcase 2 the buckling of the preload of subcase 1), with design variable 1 (the thickness) and, with `shape`, variable 3 (grid points 5 and 6 moved out of the
// plane: from 0, so that the strip the gradient is taken at is warped, see the shape test).
function designStrip({ n = 2, solution = 101, load = "1.;0.;0.;1.", shape = false, extra = [] } = {}) {
  const records = [];
  for (let i = 0; i <= n; ++i)
    for (let j = 0; j <= 1; ++j) records.push(`Node;${i * 2 + j + 1};;${real(i / n)};${real(j)};0.`);
  for (let i = 0; i < n; ++i) records.push(`ShellQuad4;${i + 1};1;${i * 2 + 1};${i * 2 + 3};${i * 2 + 4};${i * 2 + 2}`);
  records.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "SinglePointConstrList;1;123456;1;2",
    `PointForceLoad;2;${2 * n + 1};;${load}`, `PointForceLoad;2;${2 * n + 2};;${load}`, "LanczosEigenMethod;3;;;2",
    ...extra);
  if (shape) records.push("DesignVar;3;MOVE;0.;-0.1;0.1", "DesignNodeMove;3;5;0;1.;0.;0.;1.",
    "DesignNodeMove;3;6;0;1.;0.;0.;1.");
  else records.push("DesignVar;1;THICK;0.1;0.05;0.2", "DesignPropRelation1;1;ShellProp;1;thickness;;;;1;1.");
  const cases = solution === 101 ? ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2"]
    : solution === 105 ? ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "SubcaseStart;2",
      "SinglePointConstrSelection;1", "EigenMethodSelection;3", "StaticSubcaseSelection;1"]
    : ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"];
  return { tip: 2 * n + 1, text: ["[executive]", `SolutionSequence;${solution}`, "[case]", ...cases,
    "DisplacementOutput;All", "StressOutput;All", "[model]", ...records, ""].join("\n") };
}

// The native gradient of every response (by each of `methods`; [undefined]: the default) against the central
// differences of the analysis (by default a step of 1e-4 of the value of each variable), relative to the largest of
// each response's gradient.
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

const BOTH_METHODS = ["direct", "adjoint"];

function withDesign({ text }, responses, settings, action) {
  oor.writeFile("/work/mixed-design.oos", text);
  const deck = oor.readModel("/work/mixed-design.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck, { responses }, settings, SHELL_OPTIONS);
    try { action(problem); } finally { problem.delete(); }
  } finally { deck.delete(); }
}

const GRADIENT_TOLERANCE = 1e-5;
// A shape gradient against the differences of the analysis, whose first response is a small difference of large
// displacements: 2e-5 off at the step 1e-4 (its extrapolated differences agree to 1e-5).
const SHAPE_GRADIENT_TOLERANCE = 1e-4;

test("the thickness gradients of a static strip in MixedHybridFormulation are those of the analysis", () => {
  // Response 12 reads the rotation of a grid point of the shells (an excluded DOF filled from them): its adjoint takes
  // its weight on the unknowns the rotation is filled from (FormulationConstraints::moveExcludedWeights).
  const strip = designStrip();
  withDesign(strip, [{ id: 10, type: "Displacement", attributeA: 3, targets: [strip.tip] },
    { id: 11, type: "Stress", attributeA: 9, propertyType: "Elements", targets: [1] },
    { id: 12, type: "Displacement", attributeA: 5, targets: [strip.tip] }], {},
  (problem) => assertGradients(problem, [0.1], GRADIENT_TOLERANCE, undefined, BOTH_METHODS));
});

test("the thickness gradient of the first mode of a ShellQuad4 in MixedHybridFormulation is the analysis'", () => {
  withDesign(designStrip({ n: 1, solution: 103 }), [{ id: 14, type: "Eigenvalue", attributeA: 1 }],
    { analyses: { modes: [1] } }, (problem) => assertGradients(problem, [0.1], GRADIENT_TOLERANCE));
});

test("the thickness gradient of the buckling factor of a ShellQuad4 strip in MixedHybridFormulation is the analysis'", () => {
  // The factor's derivative through the preload takes the product φᵀ K_d(e_k) φ of the shells in one pass per element
  // (differentialStiffnessPreloadGradient) rather than K_d of each of their hundred or so DOFs.
  withDesign(designStrip({ solution: 105, load: "1.;-1.;0.;0." }),
    [{ id: 14, type: "BucklingFactor", attributeA: 1, tracksMode: true, subcases: [2] }],
    { analyses: { buckling: [2] } }, (problem) => assertGradients(problem, [0.1], GRADIENT_TOLERANCE));
});

test("the inertia gradient of the buckling factor of a Bar2 column in MixedHybridFormulation is the analysis'", () => {
  // The lines have no product of K_d along a shape: the derivative through the preload forms K_d of each of their
  // DOFs, which are few.
  const text = column({ count: 2, solution: 105 }) + `DesignVar;1;INERTIA;${real(BEAM.i1)};0.5e-6;2.e-6\n` +
    "DesignPropRelation1;1;BarProp;1;bendingInertia1;;;;1;1.\n";
  oor.writeFile("/work/mixed-design.oos", text);
  const deck = oor.readModel("/work/mixed-design.oos", { strict: true });
  try {
    const problem = solver.optimizationFromReader(oor, deck,
      { responses: [{ id: 14, type: "BucklingFactor", attributeA: 1, tracksMode: true, subcases: [2] }] },
      { analyses: { buckling: [2] } }, { elementFormulation: "MixedHybridFormulation" });
    try { assertGradients(problem, [BEAM.i1], GRADIENT_TOLERANCE); } finally { problem.delete(); }
  } finally { deck.delete(); }
});

test("the shape gradient of a strip in MixedHybridFormulation reaches the elements beside the grid points it moves",
  () => {
    // Element 1 has none of the grid points 5 and 6, but the normals of its edge at x = 1/3 are read from element 2.
    // The strip is warped where the gradient is taken (0.05): on the flat strip, a straight boundary that a move bends
    // switches the reading of the surface (SurfaceReconstruction's straight boundary), and the matrices of the
    // elements are not smooth there.
    const strip = designStrip({ n: 3, shape: true, load: "1.;0.;0.;1." });
    withDesign(strip, [{ id: 10, type: "Displacement", attributeA: 3, targets: [strip.tip] },
      { id: 11, type: "Displacement", attributeA: 1, targets: [strip.tip + 1] }], {},
    (problem) => assertGradients(problem, [0.05], SHAPE_GRADIENT_TOLERANCE, [1e-4]));
  });

// A shape that moves the relations of the formulation (docs/mixed-hybrid-formulation-spec.md section 5.15): a spring
// on the rotation of grid point 5, or a moment there, makes that rotation an observer relation of the shells' unknowns,
// whose coefficients move with the grid points; the rotations of the other grid points are excluded DOFs filled from
// the shells. Responses 11 and 12 read such rotations.
for (const [name, extra] of [["spring", ["SpringDofInline;50;10.;5;4;;"]], ["moment", ["PointMomentLoad;2;5;;0.3;1.;0.;0."]]])
  test(`the shape gradients of a strip in MixedHybridFormulation with an observer relation (${name}) are the analysis'`,
    () => {
      const strip = designStrip({ n: 2, shape: true, extra });
      withDesign(strip, [{ id: 10, type: "Displacement", attributeA: 3, targets: [strip.tip] },
        { id: 11, type: "Displacement", attributeA: 4, targets: [5] },
        { id: 12, type: "Displacement", attributeA: 5, targets: [strip.tip] },
        { id: 13, type: "Stress", attributeA: 9, propertyType: "Elements", targets: [2] }], {},
      (problem) => assertGradients(problem, [0.05], SHAPE_GRADIENT_TOLERANCE, [1e-4], BOTH_METHODS));
    });

test("the shape gradients of the roots of a strip in MixedHybridFormulation with an observer relation are the analysis'",
  () => {
    // The roots of the reduced problem change with the relations: the projection takes D·(K − λ·M)·φ (EigenItems).
    const strip = designStrip({ n: 2, shape: true, solution: 103, extra: ["SpringDofInline;50;10.;5;4;;"] });
    withDesign(strip, [{ id: 14, type: "Eigenvalue", attributeA: 1 }, { id: 15, type: "Eigenvalue", attributeA: 2 }],
      { analyses: { modes: [1] } }, (problem) => assertGradients(problem, [0.05], SHAPE_GRADIENT_TOLERANCE, [1e-4]));
  });

test("the shape gradient of the buckling factor of a strip in MixedHybridFormulation with an observer relation is the analysis'",
  () => {
    const strip = designStrip({ n: 2, shape: true, solution: 105, load: "1.;-1.;0.;0.05",
      extra: ["SpringDofInline;50;10.;5;4;;"] });
    withDesign(strip, [{ id: 14, type: "BucklingFactor", attributeA: 1, tracksMode: true, subcases: [2] }],
      { analyses: { buckling: [2] } }, (problem) => assertGradients(problem, [0.05], SHAPE_GRADIENT_TOLERANCE, [1e-4]));
  });

// ---------------------------------------------------------------------------------------------------------------------
// External units (docs/mixed-hybrid-formulation-spec.md section 5.9): a zone's elements replaced by matrices on the
// grid points of its contour. The rotations of the shells' grid points the matrices act on become observer relations.

// Two plates of n × n ShellQuad4 side by side, A on x in [0, 1] held along x = 0, B on x in [1, 2] loaded along x = 2,
// with their own grid points on x = 1 joined by springs on components 1 to 5 between coincident points (stiffnesses
// that no rigid motion strains: an external unit's K* must not). Gives the deck and the external unit whose matrix is
// that of the springs.
function springJoinedPlates(n = 2) {
  const id = (plate, i, j) => plate * 100 + i * (n + 1) + j + 1;
  const bulk = [];
  for (const plate of [0, 1])
    for (let i = 0; i <= n; ++i)
      for (let j = 0; j <= n; ++j) bulk.push(`Node;${id(plate, i, j)};;${real(plate + i / n)};${real(j / n)};0.`);
  let element = 1;
  for (const plate of [0, 1])
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        bulk.push(`ShellQuad4;${element++};1;${id(plate, i, j)};${id(plate, i + 1, j)};${id(plate, i + 1, j + 1)};` +
          `${id(plate, i, j + 1)}`);
  bulk.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.");
  const springs = []; // [ID, grid of A, grid of B, component]
  for (let j = 0; j <= n; ++j)
    for (let c = 1; c <= 5; ++c) {
      springs.push([500 + springs.length, id(0, n, j), id(1, 0, j), c]);
      const [spring, a, b] = springs[springs.length - 1];
      bulk.push(`SpringDofInline;${spring};1000.;${a};${c};${b};${c}`);
    }
  bulk.push(`SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, j) => id(0, 0, j)).join(";")}`);
  for (let j = 0; j <= n; ++j) bulk.push(`PointForceLoad;2;${id(1, n, j)};;1.;0.;0.3;1.`);
  const text = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "[model]", ...bulk, ""].join("\n");
  // K* of the springs: k on the diagonal of both DOFs of a spring, −k between them.
  const labels = [];
  const rowOf = new Map();
  for (const [, a, b, c] of springs)
    for (const grid of [a, b])
      if (!rowOf.has(`${grid}:${c}`)) {
        rowOf.set(`${grid}:${c}`, labels.length);
        labels.push([grid, c]);
      }
  const size = labels.length;
  const stiffness = new Array(size * size).fill(0);
  for (const [, a, b, c] of springs) {
    const [p, q] = [rowOf.get(`${a}:${c}`), rowOf.get(`${b}:${c}`)];
    stiffness[p * size + p] += 1000;
    stiffness[q * size + q] += 1000;
    stiffness[p * size + q] -= 1000;
    stiffness[q * size + p] -= 1000;
  }
  const grids = [...new Set(labels.map(([grid]) => grid))];
  const unit = { zone: { id: 1, name: "springs", elements: springs.map(([spring]) => spring),
    boundary: grids.map((grid) => ({ grid, components: "12345" })) }, labels, stiffness };
  return { text, unit, firstPlateGrid: (i, j) => id(0, i, j) };
}

test("an external unit replaces the springs between two plates of MixedHybridFormulation", () => {
  const { text, unit } = springJoinedPlates();
  const model = readDeck(text, "mixed-external");
  try {
    const shells = solver.modelFromReader(oor, model, SHELL_OPTIONS);
    try {
      const full = solver.run(shells, "statics");
      const external = solver.run(shells, "statics", { externalUnits: [unit] });
      assert.ok(!external.diagnostics.some((d) => d.severity === "error"), JSON.stringify(external.diagnostics));
      for (const kind of ["displacements", "singlePointConstrForces"]) {
        const [a, b] = [full, external].map((run) => run.nodalTables.find((table) => table.kind === kind));
        const largest = Math.max(...Array.from(a.values).map(Math.abs));
        a.values.forEach((value, k) => assert.ok(Math.abs(value - b.values[k]) < 1e-6 * largest, `${kind} ${k}`));
      }
    } finally { shells.delete(); }
  } finally { model.delete(); }
});

test("an external unit whose zone cuts the shells of MixedHybridFormulation is refused", () => {
  // The zone of element 1 of plate A: its edges with elements 2 and 3 are unknowns of both sides.
  const { text, firstPlateGrid } = springJoinedPlates();
  const contour = [firstPlateGrid(1, 0), firstPlateGrid(1, 1), firstPlateGrid(0, 1)];
  const labels = contour.flatMap((grid) => [1, 2, 3].map((c) => [grid, c]));
  const stiffness = labels.flatMap((_, i) => labels.map((__, j) => (i === j ? 1 : 0)));
  const unit = { zone: { id: 2, name: "one shell", elements: [1], boundary: contour.map((grid) => ({ grid,
    components: "123" })) }, labels, stiffness };
  const model = readDeck(text, "mixed-external-cut");
  try {
    const shells = solver.modelFromReader(oor, model, SHELL_OPTIONS);
    try {
      assert.throws(() => solver.run(shells, "statics", { externalUnits: [unit] }), /external-unit-formulation-contour/);
    } finally { shells.delete(); }
  } finally { model.delete(); }
});

// ---------------------------------------------------------------------------------------------------------------------
// Superelements (docs/mixed-hybrid-formulation-spec.md section 5.9): a part of its own model joined to the residual
// structure at grid points, reduced on them.

// Plate A (2 × 2 ShellQuad4 on x in [0, 1], held along x = 0) and plate B (2 × 2 on x in [0.5, 1.5], loaded along
// x = 1.5), the points of the lines x = 0.5 and x = 1 of each plate coincident and joined by springs on components 1 to
// `springComponents` (two lines: no hinge). With `shared`, B is on x in [1, 2] and has A's grid points of the line
// x = 1 (no spring): the plates share the edges of that line. Gives the decks of the whole model, of the residual
// structure (A, the springs and B's points they reach) and of the part (B), and the part's boundary.
function superelementPlates({ springComponents = 3, shared = false, solution = 101 } = {}) {
  const n = 2;
  const idA = (i, j) => i * (n + 1) + j + 1;
  const idB = (i, j) => (shared && i === 0 ? idA(n, j) : 100 + i * (n + 1) + j + 1);
  const xB = shared ? 1 : 1 - 1 / n;
  const nodesA = [];
  const nodesB = [];
  for (let i = 0; i <= n; ++i)
    for (let j = 0; j <= n; ++j) {
      nodesA.push(`Node;${idA(i, j)};;${real(i / n)};${real(j / n)};0.`);
      if (!shared || i > 0) nodesB.push(`Node;${idB(i, j)};;${real(xB + i / n)};${real(j / n)};0.`);
    }
  const shells = (id, first) => {
    const records = [];
    let element = first;
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        records.push(`ShellQuad4;${element++};1;${id(i, j)};${id(i + 1, j)};${id(i + 1, j + 1)};${id(i, j + 1)}`);
    return records;
  };
  const boundary = [];
  const springs = [];
  if (shared) for (let j = 0; j <= n; ++j) boundary.push(idA(n, j));
  else
    for (const [ia, ib] of [[n - 1, 0], [n, 1]])
      for (let j = 0; j <= n; ++j) {
        boundary.push(idB(ib, j));
        for (let c = 1; c <= springComponents; ++c)
          springs.push(`SpringDofInline;${500 + springs.length};1000.;${idA(ia, j)};${c};${idB(ib, j)};${c}`);
      }
  const common = ["ShellProp;1;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1."];
  const held = `SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, j) => idA(0, j)).join(";")}`;
  const loads = Array.from({ length: n + 1 }, (_, j) => `PointForceLoad;2;${idB(n, j)};;1.;0.;0.3;1.`);
  const selection = solution === 103 ? ["EigenMethodSelection;3"] : ["StaticLoadSelection;2"];
  const method = solution === 103 ? ["LanczosEigenMethod;3;;;3"] : [];
  const deck = (records) => ["[executive]", `SolutionSequence;${solution}`, "[case]", "DisplacementOutput;All",
    "SubcaseStart;1", "SinglePointConstrSelection;1", ...selection, "[model]", ...records, ...method, ""].join("\n");
  const boundaryNodes = nodesB.filter((record) => boundary.includes(Number(record.split(";")[1])));
  return {
    boundary,
    whole: deck([...nodesA, ...nodesB, ...shells(idA, 1), ...shells(idB, 101), ...springs, ...common, held, ...loads]),
    residual: deck([...nodesA, ...boundaryNodes, ...shells(idA, 1), ...springs, ...common, held,
      `PointForceLoad;2;${idA(1, 1)};;0.;0.;0.;1.`]),
    // The part's own SPC set 1 holds a scalar point only (the Case Control selects it).
    part: deck([...(shared ? nodesA.slice(-(n + 1)) : []), ...nodesB, ...shells(idB, 101), ...common, ...loads,
      "ScalarPoint;999", "SinglePointConstrList;1;0;999"]),
  };
}

function withSuperelementModels(plates, action) {
  const decks = [];
  const models = [];
  try {
    for (const name of ["whole", "residual", "part"]) {
      decks.push(readDeck(plates[name], `mixed-superelement-${name}`));
      models.push(solver.modelFromReader(oor, decks[decks.length - 1], SHELL_OPTIONS));
    }
    return action(...models);
  } finally {
    models.forEach((model) => model.delete());
    decks.forEach((deck) => deck.delete());
  }
}

test("a superelement of MixedHybridFormulation gives the displacements of the whole model", () => {
  const plates = superelementPlates();
  withSuperelementModels(plates, (whole, residual, part) => {
    const full = solver.run(whole, "statics").nodalTables.find((table) => table.kind === "displacements");
    const run = solver.runSuperelements(residual, [{ id: 1, model: part,
      boundary: plates.boundary.map((point) => [point, point]) }], "statics");
    assert.deepEqual(run.superelements.map((entry) => entry.id), [0, 1]);
    const largest = Math.max(...Array.from(full.values).map(Math.abs));
    for (const entry of run.superelements) {
      const table = entry.nodalTables.find((candidate) => candidate.kind === "displacements");
      for (let r = 0; r < table.ids.length; ++r) {
        // The rotation of a boundary point is the mean of the part's shells there: the residual structure, which has
        // no shell there, takes it from the part's recovery.
        const row = Array.from(full.ids).indexOf(table.ids[r]);
        if (row < 0) continue; // the part's own scalar point
        for (let c = 0; c < 6; ++c)
          assert.ok(Math.abs(table.values[r * 6 + c] - full.values[row * 6 + c]) < 1e-8 * largest,
            `superelement ${entry.id}, grid point ${table.ids[r]}, component ${c + 1}`);
      }
    }
  });
});

test("the mode shapes of a superelement of MixedHybridFormulation take the part's rotations at its boundary", () => {
  // The rotation of a boundary point is the mean of the part's shells there; the residual structure, which has no
  // shell there, takes it in each mode from the part's mode shape.
  const plates = superelementPlates({ solution: 103 });
  withSuperelementModels(plates, (whole, residual, part) => {
    const run = solver.runSuperelements(residual, [{ id: 1, model: part,
      boundary: plates.boundary.map((point) => [point, point]) }], "modes");
    const shapesOf = (id) => run.superelements.find((entry) => entry.id === id).nodalTables
      .filter((table) => table.kind === "eigenvectors");
    const [residualShapes, partShapes] = [shapesOf(0), shapesOf(1)];
    assert.ok(residualShapes.length >= 1 && residualShapes.length === partShapes.length,
      `${residualShapes.length} and ${partShapes.length} mode shapes`);
    residualShapes.forEach((shape, mode) => {
      const other = partShapes[mode];
      const largest = Math.max(...Array.from(shape.values).map(Math.abs));
      for (const point of plates.boundary) {
        const [r, q] = [Array.from(shape.ids).indexOf(point), Array.from(other.ids).indexOf(point)];
        for (let c = 0; c < 6; ++c)
          assert.ok(Math.abs(shape.values[r * 6 + c] - other.values[q * 6 + c]) < 1e-10 * largest,
            `mode ${mode + 1}, grid point ${point}, component ${c + 1}: ${shape.values[r * 6 + c]} and ` +
            `${other.values[q * 6 + c]}`);
      }
      const rotations = plates.boundary.flatMap((point) => {
        const r = Array.from(shape.ids).indexOf(point);
        return [3, 4, 5].map((c) => Math.abs(shape.values[r * 6 + c]));
      });
      assert.ok(Math.max(...rotations) > 1e-6 * largest, `mode ${mode + 1}: the boundary does not turn`);
    });
  });
});

test("a superelement whose shell rotations the residual structure stiffens is refused", () => {
  const plates = superelementPlates({ springComponents: 5 });
  withSuperelementModels(plates, (whole, residual, part) => {
    assert.throws(() => solver.runSuperelements(residual, [{ id: 1, model: part,
      boundary: plates.boundary.map((point) => [point, point]) }], "statics"), /superelement-boundary-rotation/);
  });
});

test("a superelement that shares the edges of shells of MixedHybridFormulation with the residual is refused", () => {
  const plates = superelementPlates({ shared: true });
  withSuperelementModels(plates, (whole, residual, part) => {
    assert.throws(() => solver.runSuperelements(residual, [{ id: 1, model: part,
      boundary: plates.boundary.map((point) => [point, point]) }], "statics"), /superelement-boundary-formulation/);
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// Supports and relations at the grid points of shells (docs/mixed-hybrid-formulation-spec.md section 5.6): the shells
// have no rotation at their grid points, so what a support holds there, or what a rigid element moves, is read along
// the edges whose two grid points have it, and tied to the mean rotation of the shells where no edge reads it.

// A strip with a static load set selected.
function loadedStrip(options) {
  const model = strip(options);
  return { ...model,
    text: model.text.replace("SinglePointConstrSelection;1", "SinglePointConstrSelection;1\nStaticLoadSelection;2") };
}

// The largest difference between the displacements of the grid points `ids` in two runs (their first `count`
// components), relative to the largest displacement.
function displacementGap(a, b, ids, count = 6) {
  let [gap, scale] = [0, 0];
  for (const id of ids) {
    const [ua, ub] = [nodal(a, "displacements", id), nodal(b, "displacements", id)];
    for (let c = 0; c < count; ++c) {
      gap = Math.max(gap, Math.abs(ua[c] - ub[c]));
      scale = Math.max(scale, Math.abs(ua[c]));
    }
  }
  return gap / scale;
}

const clampedEnd = (node, ny) => Array.from({ length: ny + 1 }, (_, j) => `SinglePointConstrList;1;123456;${node(0, j)}`);

test("an edge of ShellQuad4 in MixedHybridFormulation tied to a RigidBodyStar moves as under the same enforced motion", () => {
  // The grid points of the end x = L depend on grid point 900 above its middle, which is moved by w along z and
  // turned by θ about y: they move by u = u_900 + θ × r, r = (0, y − W / 2, −h), that is u_x = −h θ, u_z = w. The
  // edges between them follow by the map that carries the same values enforced at the grid points themselves.
  const [theta, w, h] = [1.0e-2, 2.0e-3, 0.05];
  const rigid = strip({ nx: 3, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    `Node;900;;${real(STRIP.length)};${real(STRIP.width / 2)};${real(h)}`,
    `RigidBodyStar;800;900;123456;${Array.from({ length: ny + 1 }, (_, j) => node(nx, j)).join(";")}`,
    "SinglePointConstrList;1;1246;900", `SinglePointConstr;1;900;3;${real(w)}`,
    `SinglePointConstr;1;900;5;${real(theta)}`) });
  const enforced = strip({ nx: 3, records: (node, nx, ny) => {
    const records = clampedEnd(node, ny);
    for (let j = 0; j <= ny; ++j)
      records.push(`SinglePointConstrList;1;246;${node(nx, j)}`,
        `SinglePointConstr;1;${node(nx, j)};1;${real(-h * theta)}`, `SinglePointConstr;1;${node(nx, j)};3;${real(w)}`,
        `SinglePointConstr;1;${node(nx, j)};5;${real(theta)}`);
    return records;
  } });
  const [a, b] = [runShells(rigid.text, "mixed-rigid-edge"), runShells(enforced.text, "mixed-enforced-edge")];
  assert.ok(!a.diagnostics.some((d) => d.code === "rotation-not-transmitted"), "no rotation left out");
  const gap = displacementGap(a, b, enforced.points.map((point) => point.id));
  assert.ok(gap < 1e-9, `the rigid element and the enforced motion differ by ${gap}`);
  // The force and the moment that hold grid point 900 are those of the enforced end about it.
  const total = [0, 0, 0, 0, 0, 0];
  for (let j = 0; j <= enforced.ny; ++j) {
    const id = enforced.node(enforced.nx, j);
    const f = nodal(b, "singlePointConstrForces", id);
    const r = [0, enforced.points.find((point) => point.id === id).y - STRIP.width / 2, -h];
    const arm = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    for (let c = 0; c < 3; ++c) { total[c] += f[c]; total[3 + c] += f[3 + c] + arm[c]; }
  }
  assertClose(nodal(a, "singlePointConstrForces", 900), total, 1e-7, "the forces at grid point 900");
});

test("a rigid element joined to ShellQuad4 of MixedHybridFormulation at one grid point turns with their mean rotation", () => {
  // A force F at grid point 900, h above grid point g of the shells and tied to it by a RigidBodyStar, works on the
  // shells as the force F and the moment (x_900 − x_g) × F at g: the rotation of the rigid element is the mean
  // rotation of the shells at g, on which a moment given there acts (spec section 5.4).
  const h = 0.05;
  const [fx, fz] = [30.0, 2.0];
  const rigid = loadedStrip({ nx: 3, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    `Node;900;;${real(STRIP.length)};${real(STRIP.width / 2)};${real(h)}`,
    `RigidBodyStar;800;900;123456;${node(nx, 1)}`, `PointForceLoad;2;900;;1.;${real(fx)};0.;${real(fz)}`) });
  const direct = loadedStrip({ nx: 3, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    `PointForceLoad;2;${node(nx, 1)};;1.;${real(fx)};0.;${real(fz)}`,
    `PointMomentLoad;2;${node(nx, 1)};;1.;0.;${real(h * fx)};0.`) });
  const [a, b] = [runShells(rigid.text, "mixed-rigid-point"), runShells(direct.text, "mixed-direct-point")];
  assert.ok(!a.diagnostics.some((d) => d.code === "rotation-not-transmitted"), "the rotation is transmitted");
  const gap = displacementGap(a, b, direct.points.map((point) => point.id));
  assert.ok(gap < 1e-8, `the rigid element and the loads at its grid point differ by ${gap}`);
  // Grid point 900 turns with the shells.
  const g = direct.node(direct.nx, 1);
  assertClose(nodal(a, "displacements", 900).slice(3), nodal(b, "displacements", g).slice(3), 1e-8,
    "the rotation of grid point 900");
});

test("a plate of ShellQuad4 in MixedHybridFormulation clamped at one grid point is held, with the reactions of the loads", () => {
  // The six components held at one corner hold the plate as a free body: its translations there, and the mean
  // rotation of the shells there about the three axes (no other grid point holds a rotation).
  const model = loadedStrip({ nx: 3, records: (node, nx, ny) => [`SinglePointConstrList;1;123456;${node(0, 0)}`,
    `PointForceLoad;2;${node(nx, ny)};;1.;0.;0.;1.`, `PointForceLoad;2;${node(nx, 0)};;1.;0.;1.;0.`] });
  const run = runShells(model.text, "mixed-clamp-point");
  // F = −ΣF; M = −Σ r × F about the corner: r × F = (W, −L, 0) for the force along z, (0, 0, L) for the one along y.
  assertClose(nodal(run, "singlePointConstrForces", model.node(0, 0)),
    [0, -1, -1, -STRIP.width, STRIP.length, -STRIP.length], 1e-7, "the reactions at the corner");
});

test("the rotation about the normal held at every grid point of a plate of MixedHybridFormulation stiffens nothing", () => {
  // The support written for shells without drilling stiffness: the shells of the formulation have no rotation of
  // their own about their normal, and the rotation of their membrane field stays free.
  const records = (drilling) => (node, nx, ny) => {
    const lines = clampedEnd(node, ny).concat(`PointForceLoad;2;${node(nx, ny)};;1.;0.;40.;1.`);
    if (drilling)
      for (let j = 0; j <= ny; ++j)
        for (let i = 1; i <= nx; ++i) lines.push(`SinglePointConstrList;1;6;${node(i, j)}`);
    return lines;
  };
  const free = loadedStrip({ nx: 3, records: records(false) });
  const held = loadedStrip({ nx: 3, records: records(true) });
  const [a, b] = [runShells(free.text, "mixed-drilling-free"), runShells(held.text, "mixed-drilling-held")];
  const gap = displacementGap(a, b, free.points.map((point) => point.id), 5);
  assert.ok(gap < 1e-10, `the supports of the rotation about the normal change the displacements by ${gap}`);
});

test("a line of grid points clamped inside a strip of ShellQuad4 in MixedHybridFormulation clamps the shells along it", () => {
  // The grid points of the section x = L / 2 hold their six components: the edges between them are held as the
  // edges of a boundary are, so the half beyond them does not move under a load on the first half, which bends as a
  // cantilever of length L / 2 (without Poisson's ratio: P (L / 2)³ / (3 E I) under the end force P).
  const force = 1.0;
  const model = loadedStrip({ nx: 4, poisson: 0, records: (node, nx, ny) => {
    const lines = [];
    for (let j = 0; j <= ny; ++j)
      lines.push(`SinglePointConstrList;1;123456;${node(nx / 2, j)}`,
        `PointForceLoad;2;${node(0, j)};;1.;0.;0.;${real(force / (ny + 1))}`);
    return lines;
  } });
  const run = runShells(model.text, "mixed-clamp-line");
  const inertia = (STRIP.width * STRIP.thickness ** 3) / 12;
  const expected = (force * (STRIP.length / 2) ** 3) / (3 * STRIP.young * inertia);
  const tip = nodal(run, "displacements", model.node(0, 1))[2];
  assert.ok(Math.abs(tip / expected - 1) < 0.02, `deflection of the loaded end ${tip}, of the cantilever ${expected}`);
  for (const { id, x } of model.points)
    if (x > STRIP.length / 2)
      assertClose(nodal(run, "displacements", id), [0, 0, 0, 0, 0, 0], 1e-10, `grid ${id} beyond the clamped line`,
        expected);
});

// ---------------------------------------------------------------------------------------------------------------------
// Grid point force balance (docs/mixed-hybrid-formulation-spec.md section 10.1): the forces on the unknowns of the
// edges, faces and interiors are carried to the grid points of their entities with their resultant, so that the rows
// of a grid point sum to zero, the rows of an element have no resultant, and the rows of the grid points of a free
// body give the forces on it.

const FIXED_FORCE_ROWS = ["appliedLoad", "singlePointConstrForce", "multiPointConstrForce", "total", "elementFormulation"];

// The rows of a grid point force table as { node, element, name, values }.
function forceRows(table) {
  return Array.from(table.nodeIds, (node, r) => ({ node, element: table.elementIds[r],
    name: table.names[table.nameIndices[r]], values: Array.from(table.values.slice(6 * r, 6 * r + 6)) }));
}

// The resultant of `rows` about `centre` (force, then moment), `positionOf` giving the position of a grid point.
function resultantOf(rows, positionOf, centre = [0, 0, 0]) {
  const total = [0, 0, 0, 0, 0, 0];
  for (const { node, values } of rows) {
    const p = positionOf(node).map((x, c) => x - centre[c]);
    const arm = [p[1] * values[2] - p[2] * values[1], p[2] * values[0] - p[0] * values[2],
      p[0] * values[1] - p[1] * values[0]];
    for (let c = 0; c < 3; ++c) { total[c] += values[c]; total[3 + c] += values[3 + c] + arm[c]; }
  }
  return total;
}

// A cantilever strip curved along x (z = c x²: its elements are not flat), stiffened by Bar2 on one edge, under a
// pressure and a force F at a corner of its free end; `extra` more records.
const BALANCE_FORCE = [3, 2, 1];
function balanceStrip(extra = () => []) {
  const curve = 0.15;
  const model = loadedStrip({ nx: 3, z: (x) => curve * x * x, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    stiffeners(node, nx), `PointForceLoad;2;${node(nx, ny)};;1.;${BALANCE_FORCE.map(real).join(";")}`,
    Array.from({ length: nx * ny }, (_, e) => `ShellPressureLoad;2;50.;${e + 1}`), extra(node, nx, ny)) });
  const positions = new Map(model.points.map((point) => [point.id, [point.x, point.y, point.z]]));
  return { ...model, text: model.text.replace("DisplacementOutput;All", "DisplacementOutput;All\nNodeForceOutput;All"),
    positionOf: (id) => positions.get(id) };
}

test("the grid point forces of a curved stiffened strip in MixedHybridFormulation balance at every grid point", () => {
  const model = balanceStrip();
  const run = runShells(model.text, "mixed-balance");
  const rows = forceRows(run.gridPointForces[0]);
  const scale = Math.max(...rows.flatMap((row) => row.values.map(Math.abs)));
  for (const row of rows.filter((entry) => entry.name === "total"))
    assertClose(row.values, [0, 0, 0, 0, 0, 0], 1e-8, `the total at grid point ${row.node}`, scale);
  // The rows of one element are the forces it applies to its grid points: they have no resultant.
  const elements = rows.filter((row) => !FIXED_FORCE_ROWS.includes(row.name));
  assert.ok(elements.some((row) => row.name === "Bar2") && elements.some((row) => row.name === "ShellQuad4"));
  for (const element of new Set(elements.map((row) => row.element)))
    assertClose(resultantOf(elements.filter((row) => row.element === element), model.positionOf), [0, 0, 0, 0, 0, 0],
      1e-8, `the resultant of element ${element}`, scale);
  // The supports hold the loads.
  const applied = resultantOf(rows.filter((row) => row.name === "appliedLoad"), model.positionOf);
  const supports = resultantOf(rows.filter((row) => row.name === "singlePointConstrForce"), model.positionOf);
  assertClose(supports, applied.map((value) => -value), 1e-8, "the resultant of the supports");
});

test("the grid point forces of a strip in MixedHybridFormulation give the forces through a section", () => {
  // The free body beyond the section x = L / 3 (the grid points of the columns i ≥ 1 and the elements between
  // them): the elements before the section apply, at the grid points of the section, minus the loads on the body.
  const model = balanceStrip();
  const run = runShells(model.text, "mixed-section");
  const rows = forceRows(run.gridPointForces[0]);
  const cut = STRIP.length / 3;
  const isBeyond = (id) => model.positionOf(id)[0] > cut - 1e-9;
  const isOnSection = (id) => Math.abs(model.positionOf(id)[0] - cut) < 1e-9;
  // The elements before the section: those with a grid point not beyond it.
  const before = new Set(rows.filter((row) => !FIXED_FORCE_ROWS.includes(row.name) && !isBeyond(row.node))
    .map((row) => row.element));
  const through = resultantOf(rows.filter((row) => before.has(row.element) && !FIXED_FORCE_ROWS.includes(row.name) &&
    isOnSection(row.node)), model.positionOf);
  const loads = resultantOf(rows.filter((row) => row.name === "appliedLoad" && isBeyond(row.node)), model.positionOf);
  assertClose(through, loads.map((value) => -value), 1e-8, "the forces through the section");
});

test("the grid point forces of MixedHybridFormulation count the relations of the formulation at their grid points", () => {
  // A moment at a grid point of shells acts on their mean rotation there (an observer relation, spec section 5.4):
  // its row is the moment, the formulation's row what the relation gives the shells around.
  const model = balanceStrip((node, nx) => [`PointMomentLoad;2;${node(nx, 1)};;1.;0.;5.;0.`]);
  const run = runShells(model.text, "mixed-balance-moment");
  const rows = forceRows(run.gridPointForces[0]);
  const scale = Math.max(...rows.flatMap((row) => row.values.map(Math.abs)));
  assert.ok(rows.some((row) => row.name === "elementFormulation"), "a row of the formulation's relations");
  for (const row of rows.filter((entry) => entry.name === "total"))
    assertClose(row.values, [0, 0, 0, 0, 0, 0], 1e-8, `the total at grid point ${row.node}`, scale);
  // The relation moves the moment without adding a force: its rows have no resultant.
  assertClose(resultantOf(rows.filter((row) => row.name === "elementFormulation"), model.positionOf),
    [0, 0, 0, 0, 0, 0], 1e-8, "the resultant of the formulation's rows", scale);
});

// ---------------------------------------------------------------------------------------------------------------------
// What the validation suite found (wasm/test/validation): each case below failed before its correction.

// A plate of n × n ShellQuad4 or a block of n SolidHexa turned by 30 degrees about z, every grid point in the
// rectangular displacement system 5 turned with it (its axis 1 along the first side): a model whose supports are
// along no basic axis.
const TURN = Math.PI / 6;
const turned = (x, y) => [x * Math.cos(TURN) - y * Math.sin(TURN), x * Math.sin(TURN) + y * Math.cos(TURN)];
const TURNED_FRAME = `RectFrame2;5;;0.;0.;0.;0.;0.;1.;${real(Math.cos(TURN))};${real(Math.sin(TURN))};0.`;

test("the supports of shells of MixedHybridFormulation in a displacement system turned from the basic one hold the edges", () => {
  // A stretch along the first side: component 1 held on the side s = 0 and enforced on the side s = 1. The edges
  // hold the direction their two grid points share (spec section 5.6), so the state is the uniform one:
  // u1 = δ s, u2 = −ν δ r, and the pull is E t δ (the plate is a unit square).
  const [n, delta, young, poisson, thickness] = [2, 1.0e-3, 7.0e10, 0.3, 0.01];
  const id = (i, j) => j * (n + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "[model]", TURNED_FRAME];
  for (let j = 0; j <= n; ++j)
    for (let i = 0; i <= n; ++i) {
      const [x, y] = turned(i / n, j / n);
      lines.push(`Node;${id(i, j)};;${real(x)};${real(y)};0.;5`);
    }
  for (let j = 0; j < n; ++j)
    for (let i = 0; i < n; ++i)
      lines.push(`ShellQuad4;${j * n + i + 1};1;${id(i, j)};${id(i + 1, j)};${id(i + 1, j + 1)};${id(i, j + 1)}`);
  lines.push(`ShellProp;1;1;${real(thickness)};1;;1`, `IsoMat;1;${real(young)};;${real(poisson)};2700.`);
  for (let j = 0; j <= n; ++j)
    lines.push(`SinglePointConstrList;1;13;${id(0, j)}`, `SinglePointConstrList;1;3;${id(n, j)}`,
      `SinglePointConstr;1;${id(n, j)};1;${real(delta)}`);
  lines.push(`SinglePointConstrList;1;2;${id(0, 0)}`);
  const run = runShells(lines.join("\n") + "\n", "mixed-turned-shells");
  assert.ok(!run.diagnostics.some((d) => d.severity === "warning"), JSON.stringify(run.diagnostics));
  for (let j = 0; j <= n; ++j)
    for (let i = 0; i <= n; ++i)
      assertClose(nodal(run, "displacements", id(i, j)).slice(0, 3), [(delta * i) / n, (-poisson * delta * j) / n, 0],
        1e-8, `grid ${id(i, j)}`, delta);
  let pull = 0;
  for (let j = 0; j <= n; ++j) pull += nodal(run, "singlePointConstrForces", id(n, j))[0];
  assert.ok(Math.abs(pull / (young * thickness * delta) - 1) < 1e-8, `pull ${pull}`);
});

test("the supports of a SolidHexa of MixedHybridFormulation in a turned displacement system hold its faces", () => {
  // One hexahedron 2 × 1 × 1.5 turned by 30 degrees, stretched between its end faces (component 1 of the turned
  // system): the uniaxial state, pull E ε b c.
  const [a, b, c, delta, young] = [2.0, 1.0, 1.5, 1.0e-3, 2.0e11];
  const corners = [[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0], [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]];
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "[model]", TURNED_FRAME];
  corners.forEach(([x, y, z], k) => {
    const [bx, by] = turned(x, y);
    lines.push(`Node;${k + 1};;${real(bx)};${real(by)};${real(z)};5`);
  });
  lines.push("SolidHexa;1;1;1;2;3;4;5;6;7;8", "SolidProp;1;1", `IsoMat;1;${real(young)};;0.3;7800.`);
  for (const k of [1, 4, 5, 8]) lines.push(`SinglePointConstrList;1;1;${k}`);
  for (const k of [2, 3, 6, 7]) lines.push(`SinglePointConstr;1;${k};1;${real(delta)}`);
  lines.push("SinglePointConstrList;1;23;1", "SinglePointConstrList;1;3;4", "SinglePointConstrList;1;2;5");
  const edges = [[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7], [7, 8], [8, 5], [1, 5], [2, 6], [3, 7], [4, 8]];
  const model = { text: lines.join("\n") + "\n",
    options: { elementFormulation: "MixedHybridFormulation", formulationGeometry: { sharpEdges: edges } } };
  const run = runSolids(model, "mixed-turned-solid");
  assert.ok(!run.diagnostics.some((d) => d.severity === "warning"), JSON.stringify(run.diagnostics));
  let pull = 0;
  for (const k of [2, 3, 6, 7]) pull += nodal(run, "singlePointConstrForces", k)[0];
  const expected = (young * delta * b * c) / a;
  assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
});

test("a couple at a grid point of a Rod2 on an edge of ShellQuad4 in MixedHybridFormulation bends the shells", () => {
  // A rod on an edge of shells stiffens no rotation of its grid points in bending: there the rotation follows the
  // shells (spec section 5.4), so a couple about y at the end of the rod bends the strip as it does without the rod
  // (whose section is too small to stiffen it): tip deflection −M L² / (2 E I) of the strip, without Poisson's ratio.
  const moment = 1.0;
  const rods = (node, nx) => ["RodProp;2;1;1.e-8;2.e-14"].concat(
    Array.from({ length: nx }, (_, i) => `Rod2;${100 + i};2;${node(i, 0)};${node(i + 1, 0)}`));
  const model = loadedStrip({ nx: 3, ny: 1, poisson: 0, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    rods(node, nx), `PointMomentLoad;2;${node(nx, 0)};;1.;0.;${real(moment / 2)};0.`,
    `PointMomentLoad;2;${node(nx, 1)};;1.;0.;${real(moment / 2)};0.`) });
  const run = runShells(model.text, "mixed-rod-couple");
  const inertia = (STRIP.width * STRIP.thickness ** 3) / 12;
  const expected = -(moment * STRIP.length ** 2) / (2 * STRIP.young * inertia);
  for (const j of [0, 1]) {
    const deflection = nodal(run, "displacements", model.node(model.nx, j))[2];
    assert.ok(Math.abs(deflection / expected - 1) < 0.02, `deflection ${deflection}, of the strip ${expected}`);
  }
});

test("the inertia of a point mass at a grid point of ShellQuad4 in MixedHybridFormulation acts on the modes", () => {
  // A mass with a rotary inertia at the middle of the free end of a cantilever strip: the first torsion mode turns it,
  // and its frequency falls when the inertia is given (it was ignored: the rotation of the grid point had no
  // unknown and nothing told the formulation that the mass uses it).
  const frequencies = (mass) => {
    const model = strip({ nx: 2, ny: 2, records: (node, nx, ny) => clampedEnd(node, ny).concat(
      `PointMassInertia;900;${node(nx, 1)};;${mass}`, "LanczosEigenMethod;3;;;3") });
    const text = model.text.replace("[executive]\nSolutionSequence;101", "[executive]\nSolutionSequence;103")
      .replace("SinglePointConstrSelection;1", "SinglePointConstrSelection;1\nEigenMethodSelection;3");
    return withModel(text, "mixed-point-inertia", SHELL_OPTIONS, (m) => {
      const run = solver.run(m, "modes");
      assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
      return run.eigenvalues[0].rows.map((row) => row.radians);
    });
  };
  const bare = frequencies("0.5;0.;0.;0.");
  const withInertia = frequencies("0.5;0.;0.;0.;0.5;;0.5;;;0.5");
  // An inertia lowers every frequency, and by much that of a mode that turns the grid point.
  for (let k = 0; k < bare.length; ++k)
    assert.ok(withInertia[k] <= bare[k] * (1 + 1e-9), `mode ${k + 1}: ${withInertia[k]} with the inertia, ${bare[k]}`);
  assert.ok(withInertia.some((frequency, k) => frequency < 0.8 * bare[k]),
    `with the inertia ${withInertia}, without ${bare}`);
});

test("the support forces of a curved clamped edge of ShellQuad4 in MixedHybridFormulation balance the loads in moment", () => {
  // The clamped end of the strip is an arc (x = c y (W − y) on it): the reactions of the unknowns of its edges are
  // carried to its grid points with their resultant (assembly::EntityForceCarry), so the table balances the force
  // and its moment about the origin; the applied load table gives the pressure whole.
  const [bow, pressure] = [0.5, 100.0];
  const model = loadedStrip({ nx: 2, ny: 2, records: (node, nx, ny) => clampedEnd(node, ny).concat(
    `PointForceLoad;2;${node(nx, ny)};;1.;1.;2.;3.`,
    Array.from({ length: nx * ny }, (_, e) => `ShellPressureLoad;2;${real(pressure)};${e + 1}`)) });
  // Bend the clamped end in the plane of the strip.
  const bent = model.points.map((point) =>
    ({ ...point, x: point.x + bow * point.y * (STRIP.width - point.y) * (1 - point.x / STRIP.length) }));
  let text = model.text.replace("DisplacementOutput;All", "DisplacementOutput;All\nAppliedLoadOutput;All");
  for (const point of bent)
    text = text.replace(new RegExp(`Node;${point.id};;[^\\n]*`), `Node;${point.id};;${real(point.x)};${real(point.y)};0.`);
  const run = runShells(text, "mixed-curved-clamp");
  const positions = new Map(bent.map((point) => [point.id, [point.x, point.y, 0]]));
  const resultant = (kind) => {
    const table = run.nodalTables.find((t) => t.kind === kind);
    return resultantOf(Array.from(table.ids, (node, r) => ({ node, values: Array.from(table.values.slice(6 * r, 6 * r + 6)) })),
      (id) => positions.get(id));
  };
  const [applied, supports] = [resultant("appliedLoads"), resultant("singlePointConstrForces")];
  assertClose(supports, applied.map((value) => -value), 1e-8, "the support forces against the applied loads");
  // The pressure acts along z on the area of the strip, which the bend of its end keeps to first order in the bow.
  assert.ok(Math.abs(applied[1] - 2) < 1e-9 && Math.abs(applied[0] - 1) < 1e-9, `applied force ${applied}`);
  assert.ok(applied[2] > 3 + 0.9 * pressure * STRIP.length * STRIP.width, `the pressure is in the table: ${applied[2]}`);
});

test("a declared edge of MixedHybridFormulation that is no edge of the model is refused", () => {
  const run = (geometry, name) => withModel(plate(), name,
    { elementFormulation: "MixedHybridFormulation", formulationGeometry: geometry }, (model) => solver.run(model, "statics"));
  // Grid points 1 and 3 of the plate are joined by no shell, and element 99 is no shell of the edge 1-2.
  assert.throws(() => run({ rawEdges: [[1, 3]] }, "mixed-raw-unknown"), /geometry-declaration-unknown/);
  assert.throws(() => run({ sharpEdges: [[1, 3]] }, "mixed-sharp-unknown"), /geometry-declaration-unknown/);
  assert.throws(() => run({ smoothEdges: [[1, 3]] }, "mixed-smooth-unknown"), /geometry-declaration-unknown/);
  assert.throws(() => run({ sharpEdges: [[1, 2, 99]] }, "mixed-side-unknown"),
    /geometry-declaration-unknown.*found element 99 for the edge between grid points 1 and 2/s);
  assert.throws(() => run({ lineEnds: [[1, 3]] }, "mixed-end-unknown"), /geometry-declaration-unknown/);
  assert.throws(() => run({ lineContinuations: [[1, 2, 4]] }, "mixed-line-unknown"), /geometry-declaration-unknown/);
  for (const [geometry, name] of [[{ rawEdges: [[1, 2]] }, "mixed-raw-known"],
    [{ sharpEdges: [[1, 2]] }, "mixed-sharp-known"], [{ smoothEdges: [[1, 2, 1]] }, "mixed-side-known"],
    [{ lineEnds: [[1, 2]] }, "mixed-end-known"]]) {
    const accepted = run(geometry, name);
    assert.ok(!accepted.diagnostics.some((d) => d.severity === "error"), JSON.stringify(accepted.diagnostics));
  }
});

test("geometry declarations of MixedHybridFormulation that give one edge two states are refused", () => {
  const run = (geometry, name) => withModel(plate(), name,
    { elementFormulation: "MixedHybridFormulation", formulationGeometry: geometry }, (model) => solver.run(model, "statics"));
  assert.throws(() => run({ smoothEdges: [[1, 2]], sharpEdges: [[2, 1]] }, "mixed-conflict-states"),
    /geometry-declaration-conflict.*declared both smooth and sharp/s);
  assert.throws(() => run({ rawEdges: [[1, 2]], sharpEdges: [[1, 2]] }, "mixed-conflict-raw"),
    /geometry-declaration-conflict.*declared raw and also smooth or sharp/s);
  assert.throws(() => run({ lineEnds: [[1, 2]], lineContinuations: [[1, 2, 3]] }, "mixed-conflict-line"),
    /geometry-declaration-conflict.*declared both ended and continued at grid point 2/s);
});

test("checkSupport lists what MixedHybridFormulation refuses of a model before it is run", () => {
  // The refusals of the formulation's model (here a declared edge that is no edge) are those of the run.
  const codes = (geometry, name) => withModel(plate(), name,
    { elementFormulation: "MixedHybridFormulation", formulationGeometry: geometry },
    (model) => model.checkSupport("statics").refusals.map((refusal) => refusal.code));
  assert.ok(codes({ rawEdges: [[1, 3]] }, "mixed-check-raw").includes("geometry-declaration-unknown"));
  assert.ok(codes({ sharpEdges: [[1, 3]] }, "mixed-check-sharp").includes("geometry-declaration-unknown"));
  assert.ok(codes({ smoothEdges: [[1, 2]], sharpEdges: [[1, 2]] }, "mixed-check-conflict")
    .includes("geometry-declaration-conflict"));
  assert.deepEqual(codes({ rawEdges: [[1, 2]] }, "mixed-check-known"), []);
});

test("the stresses of SolidHexa in MixedHybridFormulation under their weight take the particular stresses of the body force", () => {
  // A bar along x held on its end x = 0 under a gravity g along x, without Poisson's ratio: σxx = ρ g (L − x) and
  // u = ρ g (L x − x² / 2) / E. The stresses are linear and the displacements quadratic: exact once the particular
  // stresses of the body force are added to those of the unknowns (spec section 10.3).
  const [gravity, density, nx] = [9.81, 7800.0, 2];
  const model = solidBlock({ shape: "SolidHexa", nx, poisson: 0, records: (node) => {
    const records = [`GravityLoad;2;;${real(gravity)};1.;0.;0.`, `SinglePointConstrList;1;23;${node(0, 0, 0)}`,
      `SinglePointConstrList;1;3;${node(0, 1, 0)}`, `SinglePointConstrList;1;2;${node(0, 0, 1)}`];
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j) records.push(`SinglePointConstrList;1;1;${node(0, j, k)}`);
    return records;
  } });
  model.text = model.text.replace("SinglePointConstrSelection;1", "SinglePointConstrSelection;1\nStaticLoadSelection;2");
  const run = runSolids(model, "mixed-solid-weight");
  const weight = density * gravity;
  const tip = (weight * BLOCK.length ** 2) / (2 * BLOCK.young);
  for (const { id, x } of model.points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [(weight * (BLOCK.length * x - (x * x) / 2)) / BLOCK.young, 0, 0],
      1e-9, `grid ${id}`, tip);
  const stressAt = (x) => [weight * (BLOCK.length - x), 0, 0, 0, 0, 0];
  for (const row of rowsOf(run, "SolidHexa", "stress")) {
    const x = row.nodeId === 0 ? (BLOCK.length * (row.id - 0.5)) / nx : model.points.find((p) => p.id === row.nodeId).x;
    assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz], stressAt(x), 1e-9,
      `stresses of element ${row.id} at ${row.nodeId}`, weight * BLOCK.length);
  }
  for (const row of rowsOf(run, "SolidGridPoints", "stress"))
    assertClose([row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz], stressAt(model.points.find((p) => p.id === row.id).x),
      1e-9, `stresses at grid point ${row.id}`, weight * BLOCK.length);
});

test("a curved edge of ShellQuad4 in MixedHybridFormulation that follows a rigid element turns with it", () => {
  // A roof of 2 × 2 shells whose end x = 0.6 (an arc of three grid points) follows grid point 100 by a rigid element,
  // and nothing else holds it: turning grid point 100 turns the roof as a rigid body, u = θ × (x − x100), without
  // strain and without reaction. The edges of the arc read the relation of their grid points; on a curved edge
  // that needs the rotations too (u_edge = u_rod + θ × δ, δ the offset of the arc from its chord; spec section 5.6).
  const [rise, half, angle] = [0.0603, 0.342, 0.01];
  const centre = [0.7, 0.1, 0.1];
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "[model]"];
  const points = [];
  for (let i = 0; i <= 2; ++i)
    for (let j = 0; j <= 2; ++j) {
      const point = { id: 3 * i + j + 1, x: 0.3 * i, y: half * (j - 1), z: j === 1 ? 0 : -rise };
      points.push(point);
      lines.push(`Node;${point.id};;${real(point.x)};${real(point.y)};${real(point.z)}`);
    }
  lines.push("ShellQuad4;1;1;1;4;5;2", "ShellQuad4;2;1;2;5;6;3", "ShellQuad4;3;1;4;7;8;5", "ShellQuad4;4;1;5;8;9;6",
    "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.", `Node;100;;${centre.map(real).join(";")}`,
    "RigidBodyStar;900;100;123;7;8;9", "SinglePointConstrList;1;12346;100", `SinglePointConstr;1;100;5;${real(angle)}`);
  const run = runShells(lines.join("\n") + "\n", "mixed-curved-followed-edge");
  let reach = 0;
  for (const point of points) reach = Math.max(reach, angle * Math.hypot(point.x - centre[0], point.z - centre[2]));
  for (const { id, x, z } of points)
    assertClose(nodal(run, "displacements", id).slice(0, 3), [angle * (z - centre[2]), 0, -angle * (x - centre[0])], 1e-8,
      `grid ${id}`, reach);
  // The membrane stiffness E t times the motion is the scale of the reaction a hidden link to the ground would give.
  const reaction = nodal(run, "singlePointConstrForces", 100);
  const scale = 7.0e10 * 0.01 * reach;
  assert.ok(Math.max(...reaction.map(Math.abs)) < 1e-8 * scale, `reaction ${reaction}, scale ${scale}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// What real models found.

test("a sandwich of 2D orthotropic plies without transverse shear moduli is solved in both formulations", () => {
  // A cantilever plate of 4 × 4 ShellQuad4, 2.5 thicknesses wide each, with an unsymmetric sandwich (four plies,
  // a core, two plies) whose plies give no G1Z and G2Z: the laminate is rigid in transverse shear. ClassicFormulation
  // took the general matrix of the bending section as its transverse shear there, whose Poisson term made the
  // stiffness of such elements indefinite (negative pivots); it takes that matrix without that term. The two
  // formulations then agree on the deflection under the corner force (1.8645e-4 and 1.9013e-4; ClassicFormulation
  // gives 1.8913e-4 on 8 × 8).
  const [n, side] = [4, 2.5];
  const id = (i, j) => j * (n + 1) + i + 1;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  for (let j = 0; j <= n; ++j)
    for (let i = 0; i <= n; ++i) lines.push(`Node;${id(i, j)};;${real(side * i)};${real(side * j)};0.`);
  for (let j = 0; j < n; ++j)
    for (let i = 0; i < n; ++i)
      lines.push(`ShellQuad4;${j * n + i + 1};1;${id(i, j)};${id(i + 1, j)};${id(i + 1, j + 1)};${id(i, j + 1)}`);
  const plies = [[1, 0.1, 0], [1, 0.1, 45], [1, 0.1, -45], [1, 0.1, 90], [2, 0.5, 0], [1, 0.05, 90], [1, 0.05, 0]];
  lines.push(`CompProp;1;;0.;;;0.;0.;;${plies.map(([m, t, a]) => `${m};${real(t)};${real(a)};Yes`).join(";")}`,
    "OrthoMat2D;1;1.3e11;9.e9;0.3;5.e9;;;1600.", "IsoMat;2;1.e8;;0.3;50.");
  for (let j = 0; j <= n; ++j) lines.push(`SinglePointConstrList;1;123456;${id(0, j)}`);
  lines.push(`PointForceLoad;2;${id(n, n)};;1.e4;0.;0.;1.`);
  const text = lines.join("\n") + "\n";
  const deflection = (options) => withModel(text, "sandwich-without-shear-moduli", options, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return nodal(run, "displacements", id(n, n))[2];
  });
  const [classic, mixed] = [deflection({}), deflection(SHELL_OPTIONS)];
  assert.ok(Math.abs(classic / mixed - 1) < 0.03, `deflection ${classic} in ClassicFormulation, ${mixed} in MixedHybridFormulation`);
});
