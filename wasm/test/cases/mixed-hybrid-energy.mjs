// The mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md): the strain energy density of its elements (the
// energy over the volume of the element, whatever the corner it starts from), and the factor diagonal ratio test of
// the unknowns of its edges under a stiff spring on the rotation of a grid point of shells.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

const MIXED = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };
const CLASSIC = {};
const SUBTOTAL_ELEMENT_ID = 100000000;

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

let deckCount = 0;

// Reads `text` as a model file, builds the model with `options` and runs its statics.
function statics(text, options) {
  const file = `/work/mixed-energy-${++deckCount}.oos`;
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try { return solver.run(model, "statics"); } finally { model.delete(); }
  } finally { deck.delete(); }
}

// The statics of `text`, which must run without an error diagnostic.
function solved(text, options) {
  const run = statics(text, options);
  assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
  return run;
}

// The strain energy rows of the elements of `elementType`: element ID → { energy, density }.
function energyRows(run, elementType) {
  const table = run.elementTables.find((t) => t.kind === "strainEnergy" && t.elementType === elementType);
  assert.ok(table, `a strain energy table of ${elementType}`);
  const width = table.components.length;
  const rows = new Map();
  for (let r = 0; r < table.ids.length; ++r)
    if (table.ids[r] !== SUBTOTAL_ELEMENT_ID)
      rows.set(table.ids[r], { energy: table.values[r * width], density: table.values[r * width + 2] });
  return rows;
}

// The volume the strain energy table divides the energy of an element by.
const volumeOf = (row) => row.energy / row.density;

function assertRatio(actual, expected, tolerance, what) {
  assert.ok(Math.abs(actual / expected - 1) <= tolerance,
    `${what}: ${actual}, expected ${expected} (ratio ${actual / expected}, tolerance ${tolerance})`);
}

function displacement(run, node) {
  const table = run.nodalTables.find((t) => t.kind === "displacements" && t.subcase === 1);
  const row = table.ids.indexOf(node);
  return Array.from(table.values.slice(row * 6, row * 6 + 6));
}

const CASE_RECORDS = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
  "ElemStrainEnergyOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "[model]"];

// A plate of one ShellQuad4 and two ShellTria3 of thickness PLATE.thickness, clamped along x = 0 and loaded at a
// corner: the quadrilateral is PLATE.side square, each triangle half of that square. `lift` raises one grid point out
// of the plane (a warped quadrilateral), `firstCorner` turns the grid points of every element (the corner it starts
// from), `motion` moves the whole model rigidly: { rotation: a 3 × 3 matrix by rows, translation }.
const PLATE = { side: 0.5, thickness: 0.01 };
function plate({ lift = 0, firstCorner = 0, motion = null } = {}) {
  const place = (p) => motion === null ? p
    : motion.rotation.map((row, i) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + motion.translation[i]);
  const turn = (v) => motion === null ? v : motion.rotation.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
  const from = (grids) => grids.map((_, k) => grids[(k + firstCorner) % grids.length]).join(";");
  const s = PLATE.side;
  const points = [[0, 0, 0], [s, 0, 0], [2 * s, 0, 0], [0, s, 0], [s, s, lift], [2 * s, s, 0]];
  const lines = [...CASE_RECORDS];
  points.forEach((p, i) => lines.push(`Node;${i + 1};;${place(p).map(real).join(";")}`));
  lines.push(`ShellQuad4;1;1;${from([1, 2, 5, 4])}`, `ShellTria3;2;1;${from([2, 3, 6])}`,
    `ShellTria3;3;1;${from([2, 6, 5])}`);
  lines.push(`ShellProp;1;1;${real(PLATE.thickness)};1;;1`, "IsoMat;1;7.e10;;0.3;2700.");
  lines.push("SinglePointConstrList;1;123456;1;4", `PointForceLoad;2;6;;1.;${turn([30, 0, 1]).map(real).join(";")}`);
  return lines.join("\n") + "\n";
}

test("the strain energy density of the shells of MixedHybridFormulation is their energy over their volume", () => {
  const quadVolume = PLATE.side * PLATE.side * PLATE.thickness;
  for (const options of [CLASSIC, MIXED]) {
    const run = solved(plate(), options);
    assertRatio(volumeOf(energyRows(run, "ShellQuad4").get(1)), quadVolume, 1e-10, "volume of the quadrilateral");
    for (const [id, row] of energyRows(run, "ShellTria3"))
      assertRatio(volumeOf(row), quadVolume / 2, 1e-10, `volume of triangle ${id}`);
  }
});

test("the strain energy density of a warped shell of MixedHybridFormulation does not change with its first corner nor with a rigid motion", () => {
  const rows = (deck) => {
    const run = solved(deck, MIXED);
    return [energyRows(run, "ShellQuad4").get(1), ...energyRows(run, "ShellTria3").values()];
  };
  const lift = 0.04;
  const reference = rows(plate({ lift }));
  // A rotation by 0.7 rad about the unit axis a = (2, −1, 2) / 3, R = c I + s [a]× + (1 − c) a aᵀ, then a translation.
  const [c, s, [ax, ay, az]] = [Math.cos(0.7), Math.sin(0.7), [2 / 3, -1 / 3, 2 / 3]];
  const rotation = [[c + (1 - c) * ax * ax, (1 - c) * ax * ay - s * az, (1 - c) * ax * az + s * ay],
    [(1 - c) * ay * ax + s * az, c + (1 - c) * ay * ay, (1 - c) * ay * az - s * ax],
    [(1 - c) * az * ax - s * ay, (1 - c) * az * ay + s * ax, c + (1 - c) * az * az]];
  const variants = { "from the second corner": { lift, firstCorner: 1 },
    "from the third corner": { lift, firstCorner: 2 },
    "moved rigidly": { lift, motion: { rotation, translation: [3, -4, 5] } } };
  for (const [name, variant] of Object.entries(variants)) {
    const moved = rows(plate(variant));
    reference.forEach((row, e) => {
      assertRatio(volumeOf(moved[e]), volumeOf(row), 1e-10, `volume of element ${e + 1} ${name}`);
      assertRatio(moved[e].density, row.density, 1e-6, `density of element ${e + 1} ${name}`);
    });
  }
  // The warped quadrilateral is a little larger than the flat one.
  const flat = PLATE.side * PLATE.side * PLATE.thickness;
  assert.ok(volumeOf(reference[0]) > flat && volumeOf(reference[0]) < 1.01 * flat, `volume ${volumeOf(reference[0])}`);
});

// A cantilever strip of two ShellQuad4 (STRIP.length by STRIP.width) with a Rod2 on each element side of its edge
// y = 0, pulled and bent at its tip, and beside it a cantilever Bar2 alone on its own grid points.
const STRIP = { length: 1.0, width: 0.2, thickness: 0.01 };
const LINE = { rodArea: 2.0e-4, barArea: 5.0e-4, barLength: 0.6 };
function stiffenedStrip() {
  const lines = [...CASE_RECORDS];
  const id = (i, j) => 3 * j + i + 1;
  for (let j = 0; j <= 1; ++j)
    for (let i = 0; i <= 2; ++i)
      lines.push(`Node;${id(i, j)};;${real((STRIP.length * i) / 2)};${real(STRIP.width * j)};0.`);
  lines.push("Node;7;;0.;1.;0.", `Node;8;;${real(LINE.barLength)};1.;0.`);
  for (let i = 0; i < 2; ++i) {
    lines.push(`ShellQuad4;${i + 1};1;${id(i, 0)};${id(i + 1, 0)};${id(i + 1, 1)};${id(i, 1)}`);
    lines.push(`Rod2;${11 + i};2;${id(i, 0)};${id(i + 1, 0)}`);
  }
  lines.push("Bar2;21;3;7;8;0.;;1.;0.");
  lines.push(`ShellProp;1;1;${real(STRIP.thickness)};1;;1`, `RodProp;2;1;${real(LINE.rodArea)};1.e-9`);
  lines.push(`BarProp;3;1;${real(LINE.barArea)};2.e-8;1.e-8;1.5e-8`, "IsoMat;1;7.e10;;0.3;2700.");
  lines.push(`SinglePointConstrList;1;123456;${id(0, 0)};${id(0, 1)};7`);
  lines.push(`PointForceLoad;2;${id(2, 0)};;1.;100.;0.;1.`, "PointForceLoad;2;8;;1.;100.;0.;1.");
  return lines.join("\n") + "\n";
}

test("the strain energy density of the lines of MixedHybridFormulation is their energy over area times length", () => {
  for (const options of [CLASSIC, MIXED]) {
    const run = solved(stiffenedStrip(), options);
    const rodVolume = (LINE.rodArea * STRIP.length) / 2;
    for (const [id, row] of energyRows(run, "Rod2"))
      assertRatio(volumeOf(row), rodVolume, 1e-10, `volume of the rod ${id} on an edge of shells`);
    const bar = energyRows(run, "Bar2").get(21);
    assertRatio(volumeOf(bar), LINE.barArea * LINE.barLength, 1e-10, "volume of the bar alone");
    for (const [id, row] of energyRows(run, "ShellQuad4"))
      assertRatio(volumeOf(row), (STRIP.length * STRIP.width * STRIP.thickness) / 2, 1e-10, `volume of shell ${id}`);
  }
});

// One SolidHexa of BLOCK.length × BLOCK.height × BLOCK.width (its twelve edges sharp: a straight element), held at
// its face x = 0 and pulled at the other.
const BLOCK = { length: 2.0, height: 1.0, width: 0.5 };
function block() {
  const lines = [...CASE_RECORDS];
  const id = (i, j, k) => 4 * i + 2 * k + j + 1;
  for (let i = 0; i <= 1; ++i)
    for (let k = 0; k <= 1; ++k)
      for (let j = 0; j <= 1; ++j)
        lines.push(`Node;${id(i, j, k)};;${real(BLOCK.length * i)};${real(BLOCK.height * j)};${real(BLOCK.width * k)}`);
  const face = (i) => [id(i, 0, 0), id(i, 1, 0), id(i, 1, 1), id(i, 0, 1)];
  lines.push(`SolidHexa;1;1;${face(0).join(";")};${face(1).join(";")}`, "SolidProp;1;1", "IsoMat;1;2.e11;;0.3;7800.");
  lines.push(`SinglePointConstrList;1;123;${face(0).join(";")}`);
  for (const grid of face(1)) lines.push(`PointForceLoad;2;${grid};;1.;1000.;0.;0.`);
  const sharpEdges = [];
  for (let t = 0; t <= 1; ++t)
    for (let u = 0; u <= 1; ++u)
      sharpEdges.push([id(0, t, u), id(1, t, u)], [id(t, 0, u), id(t, 1, u)], [id(t, u, 0), id(t, u, 1)]);
  return { text: lines.join("\n") + "\n", options: { elementFormulation: "MixedHybridFormulation",
    formulationGeometry: { sharpEdges } } };
}

test("the strain energy density of a SolidHexa of MixedHybridFormulation is its energy over its volume", () => {
  const { text, options } = block();
  const volume = BLOCK.length * BLOCK.height * BLOCK.width;
  for (const formulation of [CLASSIC, options])
    assertRatio(volumeOf(energyRows(solved(text, formulation), "SolidHexa").get(1)), volume, 1e-10, "volume");
});

// Two plates of 1 × JOINT.rows ShellQuad4 (square cells of side JOINT.cell) in the plane z = 0, the second beside the
// first: the grid points of their common side are doubled and joined pair by pair by springs of stiffness `stiffness`
// on the components `components`. The first plate is clamped at its far side, the second loaded across its plane at
// its far side.
const JOINT = { rows: 2, cell: 1 / 3, thickness: 0.002 };
function joinedPlates({ stiffness, components }) {
  const lines = CASE_RECORDS.filter((record) => record !== "ElemStrainEnergyOutput;All");
  const id = (plateIndex, i, j) => 100 * plateIndex + 2 * j + i + 1;
  const rows = Array.from({ length: JOINT.rows + 1 }, (_, j) => j);
  for (let plateIndex = 0; plateIndex <= 1; ++plateIndex) {
    for (const j of rows)
      for (let i = 0; i <= 1; ++i)
        lines.push(`Node;${id(plateIndex, i, j)};;${real(JOINT.cell * (plateIndex + i))};${real(JOINT.cell * j)};0.`);
    for (let j = 0; j < JOINT.rows; ++j)
      lines.push(`ShellQuad4;${10 * plateIndex + j + 1};1;${id(plateIndex, 0, j)};${id(plateIndex, 1, j)};`
        + `${id(plateIndex, 1, j + 1)};${id(plateIndex, 0, j + 1)}`);
  }
  lines.push(`ShellProp;1;1;${real(JOINT.thickness)};1;;1`, "IsoMat;1;7.e10;;0.3;2700.");
  let spring = 500;
  for (const j of rows)
    for (const c of components)
      lines.push(`SpringDofInline;${++spring};${real(stiffness)};${id(0, 1, j)};${c};${id(1, 0, j)};${c}`);
  lines.push(`SinglePointConstrList;1;123456;${rows.map((j) => id(0, 0, j)).join(";")}`);
  for (const j of rows) lines.push(`PointForceLoad;2;${id(1, 1, j)};;1.;0.;0.;1.`);
  return { text: lines.join("\n") + "\n", tip: id(1, 1, 0) };
}

test("stiff springs on the rotations of grid points of shells of MixedHybridFormulation pass the factor diagonal ratio test", () => {
  // The rotation of a grid point of shells is no unknown: it is a combination of the unknowns of the shells around
  // it, and a spring on it raises the diagonal of every one of them (by its stiffness times the square of the
  // coefficient), which is no mechanism. With the springs ten times softer the ratio test passed already: the stiffer
  // springs give the same deflection, a little smaller.
  const deflection = (stiffness) => {
    const { text, tip } = joinedPlates({ stiffness, components: [1, 2, 3, 4, 5] });
    return displacement(solved(text, MIXED), tip)[2];
  };
  const [stiff, softer] = [deflection(1.0e8), deflection(1.0e7)];
  assert.ok(stiff < softer, `deflections ${stiff} (stiff springs) and ${softer}`);
  assertRatio(stiff, softer, 1e-5, "deflection of the tip");
});

test("a hinge between two plates of MixedHybridFormulation is still a mechanism at an unknown of an element beside stiff rotational springs", () => {
  // Springs on the translations and on the rotation about x only: the second plate turns freely about the common
  // side (the y axis). The pivot that vanishes is that of an unknown inside one of its shells.
  const { text } = joinedPlates({ stiffness: 1.0e8, components: [1, 2, 3, 4] });
  assert.throws(() => statics(text, MIXED), (error) => {
    assert.match(error.message, /factor diagonal ratio test/);
    assert.match(error.message, /DOF \d+ of the interior of element 1[12]/);
    assert.doesNotMatch(error.message, /of the edge/);
    return true;
  });
});

// A cantilever plate of 6 × 1 ShellQuad4 (length 6, width 0.2, thickness 0.1) bent out of its plane by its weight, at
// degree 2 of the formulation. The largest ratio of its factorisation is 1.4e7, at an unknown of the interior of the
// element of its tip: the diagonal of such an unknown is a bending stiffness at the scale of its element, far above
// the stiffness of the plate seen from it, and the plate has no mechanism (its deflection is that of the beam).
function cantileverPlate(maxRatio) {
  const [nx, length, width] = [6, 6.0, 0.2];
  const lines = CASE_RECORDS.filter((record) => record !== "ElemStrainEnergyOutput;All");
  const id = (i, j) => 1 + i + j * (nx + 1);
  for (let j = 0; j <= 1; ++j)
    for (let i = 0; i <= nx; ++i) lines.push(`Node;${id(i, j)};;${real((length * i) / nx)};${real(width * j)};0.`);
  for (let i = 0; i < nx; ++i)
    lines.push(`ShellQuad4;${i + 1};1;${id(i, 0)};${id(i + 1, 0)};${id(i + 1, 1)};${id(i, 1)}`);
  lines.push("ShellProp;1;1;0.1;1;;1", "IsoMat;1;1.e7;;0.3;1.", `SinglePointConstrList;1;123456;${id(0, 0)};${id(0, 1)}`,
    "GravityLoad;2;;1.;0.;0.;-1.", "Parameter;AutoSingularityConstr;Yes", "Parameter;MixedHybridDegree;2");
  if (maxRatio !== undefined) lines.push(`Parameter;MaxDiagonalRatio;${real(maxRatio)}`);
  return { text: lines.join("\n") + "\n", tip: id(nx, 0) };
}

test("the unknowns of the edges and interiors of MixedHybridFormulation may have ten times MaxDiagonalRatio", () => {
  // The deflection of the beam: q L⁴ / (8 E I) with q = ρ g b t and I = b t³ / 12, plus the shear deflection.
  const beam = (1.0 * 0.2 * 0.1 * 6.0 ** 4) / (8 * 1.0e7 * (0.2 * 0.1 ** 3) / 12);
  const { text, tip } = cantileverPlate();
  assertRatio(-displacement(solved(text, MIXED), tip)[2], beam, 0.01, "deflection of the tip at the default limit");
  // The ratio is 1.4e7: it passes ten times a limit of 2e6, and fails ten times a limit of 1e6.
  assertRatio(-displacement(solved(cantileverPlate(2.0e6).text, MIXED), tip)[2], beam, 0.01, "deflection of the tip");
  assert.throws(() => statics(cantileverPlate(1.0e6).text, MIXED), (error) => {
    assert.match(error.message, /factor diagonal ratio test/);
    assert.match(error.message, /of the (interior of element|edge between grid points)/);
    return true;
  });
});
