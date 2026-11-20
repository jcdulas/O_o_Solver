// Validation of the line elements of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md, sections
// 3.2 and 13.3): Rod2, Tube2, Bar2 and Beam2 alone and at the end of shells and solids, against closed forms, with the
// criteria of section 13.6 (support.mjs, CRITERIA). The structures are placed in a skew position in space: no bar
// lies along an axis of the basic system unless the case says so.
//
// Every case prints the ratios it measures ("ratio ..."), whether it passes or not.
//
// Run: node --test wasm/test/validation/lines.test.mjs

import assert from "node:assert/strict";
import { CLASSIC, CRITERIA, MIXED, nodal, prepared, real, runModel, surfaceRecords, test } from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// Helpers

// The measurements of a case: each is printed, and the case fails at its end when one is out of its criterion.
function measurements() {
  const failures = [];
  const note = (ok, text) => {
    console.log(`    ${ok ? "ok  " : "FAIL"} ${text}`);
    if (!ok) failures.push(text);
  };
  return {
    // |actual / expected − 1| ≤ tolerance.
    ratio(what, actual, expected, tolerance) {
      const value = actual / expected;
      note(Math.abs(value - 1) <= tolerance, `ratio ${what}: ${value.toFixed(8)} (${actual.toPrecision(8)} for ` +
        `${expected.toPrecision(8)}, criterion ${tolerance})`);
    },
    // |value| ≤ tolerance × scale.
    small(what, value, scale, tolerance) {
      note(Math.abs(value) <= tolerance * Math.abs(scale), `small ${what}: ${(Math.abs(value) / Math.abs(scale))
        .toExponential(2)} of the scale (criterion ${tolerance})`);
    },
    // Every component of `actual` within tolerance × the largest magnitude of `expected`.
    vector(what, actual, expected, tolerance) {
      const scale = Math.max(...expected.map(Math.abs));
      const worst = Math.max(...expected.map((value, k) => Math.abs(actual[k] - value)));
      note(worst <= tolerance * scale, `vector ${what}: off by ${(worst / scale).toExponential(2)} of its largest ` +
        `component ${scale.toPrecision(6)} (criterion ${tolerance})`);
    },
    truth(what, ok) { note(Boolean(ok), what); },
    done() { assert.equal(failures.length, 0, failures.join("\n")); },
  };
}

// The text of a model file, with the declarations of its geometry (`prepared`).
function deck(solution, caseLines, modelLines) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseLines, "[model]",
    ...prepared(modelLines), ""].join("\n");
}

const STATIC_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
  "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "ElemForceOutput;All", "StressOutput;All"];
const MODES_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"];
const BUCKLING_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "SubcaseStart;2",
  "SinglePointConstrSelection;1", "EigenMethodSelection;3", "StaticSubcaseSelection;1"];
const eigenMethod = (count) => `LanczosEigenMethod;3;;;${count}`;

// The line elements have no shell and no solid: the model needs no geometry declaration.
const LINES = { elementFormulation: "MixedHybridFormulation" };

// The diagnostics of a run that are not information.
function warningsOf(run) {
  return run.diagnostics.filter((d) => d.severity !== "info" && d.severity !== "information").map((d) => d.text);
}

// The rows of an element table as objects of its columns.
function rowsOf(run, elementType, kind, subcase = 1) {
  const table = run.elementTables.find((t) => t.elementType === elementType && t.kind === kind &&
    t.subcase === subcase);
  assert.ok(table, `a ${elementType} ${kind} table of subcase ${subcase}`);
  const width = table.components.length;
  return Array.from(table.ids, (id, r) => {
    const row = { id, nodeId: table.nodeIds[r] };
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    return row;
  });
}

// Vectors.
const add = (p, q) => p.map((value, k) => value + q[k]);
const sub = (p, q) => p.map((value, k) => value - q[k]);
const scale = (p, factor) => p.map((value) => value * factor);
const dot = (p, q) => p.reduce((sum, value, k) => sum + value * q[k], 0);
const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
const unit = (p) => scale(p, 1 / Math.sqrt(dot(p, p)));
const fields = (p) => p.map(real).join(";");

// A skew frame: the rotation Rz(0.6) Ry(−0.4) Rx(0.3) and an origin. place(p) is the basic position of the local
// point p, turn(v) the basic components of the local vector v, local(v) the local components of a basic vector.
const SKEW = (() => {
  const [a, b, c] = [0.6, -0.4, 0.3];
  const [ca, sa, cb, sb, cc, sc] = [Math.cos(a), Math.sin(a), Math.cos(b), Math.sin(b), Math.cos(c), Math.sin(c)];
  const axes = [[ca * cb, sa * cb, -sb], [ca * sb * sc - sa * cc, sa * sb * sc + ca * cc, cb * sc],
    [ca * sb * cc + sa * sc, sa * sb * cc - ca * sc, cb * cc]]; // the local axes x, y, z in the basic system
  const origin = [0.3, -0.2, 0.5];
  const turn = (v) => add(add(scale(axes[0], v[0]), scale(axes[1], v[1])), scale(axes[2], v[2]));
  return { axes, origin, turn, place: (p) => add(origin, turn(p)), local: (v) => axes.map((axis) => dot(axis, v)) };
})();

// The material and the section of most cases: steel, a rectangle 0.03 (along the orientation vector) × 0.02.
const E = 2.1e11;
const NU = 0.3;
const G = E / (2 * (1 + NU));
const RHO = 7850.0;
const ALPHA = 1.2e-5;
const MATERIAL = `IsoMat;1;${real(E)};;${real(NU)};${real(RHO)};${real(ALPHA)};0.`;
// I1 is the inertia of the bending in plane 1 (the plane of the axis and of the orientation vector), I2 that of the
// bending in plane 2.
const SECTION = { area: 6.0e-4, i1: 4.5e-8, i2: 2.0e-8, j: 4.7e-8 };
// The recovery points C, D, E, F: the corners of the rectangle (y along the orientation vector, z across).
const POINTS = [[0.015, 0.01], [-0.015, 0.01], [-0.015, -0.01], [0.015, -0.01]];
function barProp(id, { area = SECTION.area, i1 = SECTION.i1, i2 = SECTION.i2, j = SECTION.j, k1 = "", k2 = "" } = {}) {
  const points = POINTS.map(([y, z]) => `${real(y)};${real(z)}`).join(";");
  return `BarProp;${id};1;${real(area)};${real(i1)};${real(i2)};${real(j)};;${points};${k1 === "" ? "" : real(k1)};` +
    `${k2 === "" ? "" : real(k2)}`;
}

// A straight line of `count` elements from the local point `from` to the local point `to` in the skew frame, its
// grid points numbered from `first`; the orientation vector is the local vector `orientation`. With
// `displacementFrame` (a displacement system along the axes of the skew frame) the grid points take it, and the
// orientation vector is written in it: an orientation vector is read in the displacement system of the first grid
// point of its element.
function straightLine({ card = "Bar2", from, to, count, first = 1, firstElement = 1, property = 1,
  orientation = [0, 1, 0], frame = SKEW, displacementFrame = "" }) {
  const lines = [];
  for (let k = 0; k <= count; ++k) {
    const p = frame.place(add(from, scale(sub(to, from), k / count)));
    lines.push(`Node;${first + k};;${fields(p)}${displacementFrame === "" ? "" : `;${displacementFrame}`}`);
  }
  const v = displacementFrame === "" ? frame.turn(orientation) : orientation;
  for (let k = 0; k < count; ++k) {
    const record = `${card};${firstElement + k};${property};${first + k};${first + k + 1}`;
    lines.push(card === "Rod2" || card === "Tube2" ? record
      : `${record};${real(v[0])};;${real(v[1])};${real(v[2])}`);
  }
  return lines;
}

const force = (set, node, vector, frame = SKEW) => `PointForceLoad;${set};${node};;1.;${fields(frame.turn(vector))}`;
const moment = (set, node, vector, frame = SKEW) => `PointMomentLoad;${set};${node};;1.;${fields(frame.turn(vector))}`;

// The displacement of a grid point in the local axes of the skew frame: [u, v, w, θx, θy, θz].
function localDisplacement(run, node, subcase = 1, frame = SKEW) {
  const u = nodal(run, "displacements", node, subcase);
  return [...frame.local(u.slice(0, 3)), ...frame.local(u.slice(3, 6))];
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Cantilevers in every direction of space, oriented by a vector and by a grid point

// The directions of the axis and the orientation vectors (basic components; none is along the other).
const DIRECTIONS = [
  { axis: [1, 0, 0], orientation: [0, 1, 0] },
  { axis: [0, 1, 0], orientation: [0, 0, 1] },
  { axis: [0, 0, 1], orientation: [1, 0, 0] },
  { axis: [0, 0, -1], orientation: [1, 1, 0] },
  { axis: [1, 1, 0], orientation: [0, 0, 1] },
  { axis: [1, -2, 2], orientation: [0.3, 1, 0.2] },
  { axis: [-3, 1, -2], orientation: [1, 0.5, -1] },
  { axis: [0.2, 0.1, 5], orientation: [-1, 2, 0.1] },
];

for (const card of ["Bar2", "Beam2"])
  for (const orientedBy of ["a vector", "a grid point"])
    test(`cantilevers of two ${card} in eight directions of space, oriented by ${orientedBy}: the beam theory`, () => {
      // A cantilever of length L along x of its own axes (y along the orientation vector), under the end forces
      // (Fx, Fy, Fz) and the end torque Mx: u = Fx L / (E A), v = Fy L³ / (3 E I1), w = Fz L³ / (3 E I2),
      // θx = Mx L / (G J), θy = −Fz L² / (2 E I2), θz = Fy L² / (2 E I1). At the clamp the moments are Fy L and Fz L,
      // and the stress at the point (y, z) of the section is −M1 y / I1 − M2 z / I2 (bending) and Fx / A (axial).
      const length = 1.2;
      const [fx, fy, fz, mx] = [4000.0, 60.0, -35.0, 25.0];
      const check = measurements();
      for (const { axis, orientation } of DIRECTIONS) {
        const x = unit(axis);
        const y = unit(sub(orientation, scale(x, dot(orientation, x))));
        const z = cross(x, y);
        const frame = { place: (p) => scale(x, p[0]), turn: (v) => add(add(scale(x, v[0]), scale(y, v[1])),
          scale(z, v[2])), local: (v) => [dot(x, v), dot(y, v), dot(z, v)] };
        const lines = straightLine({ card, from: [0, 0, 0], to: [length, 0, 0], count: 2, frame,
          orientation: frame.local(orientation) });
        if (orientedBy === "a grid point") {
          // The grid point 9 gives the orientation: the vector from the first end of each element to it.
          lines.push(`Node;9;;${fields(scale(orientation, 0.7))}`, "SinglePointConstrList;1;123456;9");
          for (let k = 0; k < lines.length; ++k)
            if (lines[k].startsWith(`${card};1;`)) lines[k] = `${card};1;1;1;2;;9`;
          // The second element starts at the grid point 2: its vector to the grid point 9 is another one in the
          // same plane.
          for (let k = 0; k < lines.length; ++k)
            if (lines[k].startsWith(`${card};2;`)) lines[k] = `${card};2;1;2;3;;9`;
        }
        lines.push(card === "Bar2" ? barProp(1)
          : `BeamProp;1;1;${real(SECTION.area)};${real(SECTION.i1)};${real(SECTION.i2)};;${real(SECTION.j)};;` +
            `${POINTS.map(([py, pz]) => `${real(py)};${real(pz)}`).join(";")};;0.;0.`,
        MATERIAL, "SinglePointConstrList;1;123456;1", force(2, 3, [fx, fy, fz], frame),
        moment(2, 3, [mx, 0, 0], frame));
        const run = runModel(deck(101, STATIC_CASE, lines), LINES);
        const name = `axis ${axis}`;
        const expected = [(fx * length) / (E * SECTION.area), (fy * length ** 3) / (3 * E * SECTION.i1),
          (fz * length ** 3) / (3 * E * SECTION.i2), (mx * length) / (G * SECTION.j),
          (-fz * length ** 2) / (2 * E * SECTION.i2), (fy * length ** 2) / (2 * E * SECTION.i1)];
        const tip = localDisplacement(run, 3, 1, frame);
        check.vector(`${name}: translations of the end`, tip.slice(0, 3), expected.slice(0, 3), CRITERIA.exact);
        check.vector(`${name}: rotations of the end`, tip.slice(3), expected.slice(3), CRITERIA.exact);
        const [m1, m2] = [fy * length, fz * length];
        const bending = POINTS.map(([py, pz]) => (-m1 * py) / SECTION.i1 - (m2 * pz) / SECTION.i2);
        if (card === "Bar2") {
          const forces = rowsOf(run, "Bar2", "forces")[0];
          check.vector(`${name}: forces of the element at the clamp`, [forces.bendingMomentA1, forces.bendingMomentA2,
            forces.bendingMomentB1, forces.bendingMomentB2, forces.shear1, forces.shear2, forces.axial, forces.torque],
          [m1, m2, m1 / 2, m2 / 2, fy, fz, fx, mx], CRITERIA.exact);
          const stress = rowsOf(run, "Bar2", "stress")[0];
          check.vector(`${name}: bending stresses at C, D, E, F at the clamp`, [stress.s1a, stress.s2a, stress.s3a,
            stress.s4a], bending, CRITERIA.exact);
          check.ratio(`${name}: axial stress`, stress.axial, fx / SECTION.area, CRITERIA.exact);
        } else {
          const forces = rowsOf(run, "Beam2", "forces")[0];
          check.vector(`${name}: forces of the element at the clamp`, [forces.bendingMoment1, forces.bendingMoment2,
            forces.shear1, forces.shear2, forces.axialForce, forces.totalTorque], [m1, m2, fy, fz, fx, mx],
          CRITERIA.exact);
          const stress = rowsOf(run, "Beam2", "stress")[0];
          check.vector(`${name}: stresses at C, D, E, F at the clamp`, [stress.sxc, stress.sxd, stress.sxe,
            stress.sxf], bending.map((value) => value + fx / SECTION.area), CRITERIA.exact);
        }
        const warnings = warningsOf(run);
        check.truth(`${name}: no warning: ${warnings}`, warnings.length === 0);
      }
      check.done();
    });

test("a short cantilever of Bar2 with shear factors K1 and K2: the shear deflection of Timoshenko's beam", () => {
  // v = F L³ / (3 E I) + F L / (K G A) in each plane (S. P. Timoshenko, Strength of Materials, part 1).
  // L = 0.15 for a depth of 0.03: the shear makes 3 % of the deflection in plane 1.
  const length = 0.15;
  const [k1, k2] = [0.85, 0.6];
  const [fy, fz] = [500.0, 300.0];
  const lines = [...straightLine({ from: [0, 0, 0], to: [length, 0, 0], count: 2 }), barProp(1, { k1, k2 }), MATERIAL,
    "SinglePointConstrList;1;123456;1", force(2, 3, [0, fy, fz])];
  const run = runModel(deck(101, STATIC_CASE, lines), LINES);
  const check = measurements();
  const tip = localDisplacement(run, 3);
  check.ratio("deflection in plane 1", tip[1], (fy * length ** 3) / (3 * E * SECTION.i1) +
    (fy * length) / (k1 * G * SECTION.area), CRITERIA.exact);
  check.ratio("deflection in plane 2", tip[2], (fz * length ** 3) / (3 * E * SECTION.i2) +
    (fz * length) / (k2 * G * SECTION.area), CRITERIA.exact);
  check.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 2. Frames

test("a portal frame with fixed feet under a side force: sway and moment at the feet", () => {
  // Two columns of height h (inertia Ic) and a beam of span L (inertia Ib), in a skew plane, a force P along the beam
  // at its first end. With k = (Ib / L) / (Ic / h) the slope-deflection equations give the sway
  // P h³ (3 k + 2) / (12 E Ic (6 k + 1)) and the moment P h (3 k + 1) / (2 (6 k + 1)) at each foot. The equations take
  // the members as inextensible: the area is such that their stretching adds 24 Ic / (A L²) = 5e-7 to the sway.
  const [h, span, p] = [1.5, 2.0, 1000.0];
  const column = { area: 0.6, i1: 4.5e-8, i2: 2.0e-8, j: 4.7e-8 };
  const beam = { area: 0.6, i1: 9.0e-8, i2: 2.0e-8, j: 4.7e-8 };
  const lines = [
    ...straightLine({ from: [0, 0, 0], to: [0, h, 0], count: 2, first: 1, firstElement: 1, orientation: [1, 0, 0] }),
    ...straightLine({ from: [span, 0, 0], to: [span, h, 0], count: 2, first: 11, firstElement: 11,
      orientation: [1, 0, 0] }),
    `Node;21;;${fields(SKEW.place([span / 2, h, 0]))}`,
    `Bar2;21;2;3;21;${fields(SKEW.turn([0, 1, 0])).replace(";", ";;")}`,
    `Bar2;22;2;21;13;${fields(SKEW.turn([0, 1, 0])).replace(";", ";;")}`,
    barProp(1, column), barProp(2, beam), MATERIAL, "SinglePointConstrList;1;123456;1;11", force(2, 3, [p, 0, 0])];
  const run = runModel(deck(101, STATIC_CASE, lines), LINES);
  const check = measurements();
  const k = beam.i1 / span / (column.i1 / h);
  check.ratio("sway of the beam", localDisplacement(run, 3)[0],
    (p * h ** 3 * (3 * k + 2)) / (12 * E * column.i1 * (6 * k + 1)), CRITERIA.exact);
  const feet = rowsOf(run, "Bar2", "forces").filter((row) => row.id === 1 || row.id === 11);
  for (const foot of feet)
    check.ratio(`moment at the foot of element ${foot.id}`, Math.abs(foot.bendingMomentA1),
      (p * h * (3 * k + 1)) / (2 * (6 * k + 1)), CRITERIA.exact);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

test("a bent cantilever in space under a force across its plane: bending and torsion", () => {
  // A cantilever of length a, then at a right angle a second leg of length b, a force P across their plane at the
  // end: the first leg bends and twists by the torque P b, the second bends:
  // w = P a³ / (3 E I) + P a b² / (G J) + P b³ / (3 E I) (Castigliano's theorem).
  const [a, b, p] = [1.0, 0.6, 200.0];
  const lines = [
    ...straightLine({ from: [0, 0, 0], to: [a, 0, 0], count: 2, first: 1, firstElement: 1, orientation: [0, 0, 1] }),
    `Node;4;;${fields(SKEW.place([a, b / 2, 0]))}`, `Node;5;;${fields(SKEW.place([a, b, 0]))}`,
    `Bar2;3;1;3;4;${fields(SKEW.turn([0, 0, 1])).replace(";", ";;")}`,
    `Bar2;4;1;4;5;${fields(SKEW.turn([0, 0, 1])).replace(";", ";;")}`,
    barProp(1), MATERIAL, "SinglePointConstrList;1;123456;1", force(2, 5, [0, 0, p])];
  const run = runModel(deck(101, STATIC_CASE, lines), LINES);
  const check = measurements();
  check.ratio("deflection of the end", localDisplacement(run, 5)[2], (p * a ** 3) / (3 * E * SECTION.i1) +
    (p * a * b * b) / (G * SECTION.j) + (p * b ** 3) / (3 * E * SECTION.i1), CRITERIA.exact);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

test("a ring of 120 Bar2 pulled apart by two forces, its supports in a skew displacement system", () => {
  // A ring of radius R in a skew plane, of straight bars oriented by the grid point of its centre, pulled by two
  // opposite forces F on a diameter: the diameter along the forces grows by F R³ (π/4 − 2/π) / (E I) + π F R / (4 E A)
  // and the diameter across them shrinks by F R³ (2/π − 1/2) / (E I) − F R / (2 E A) (Timoshenko, Strength of
  // Materials, part 1, with the stretching of the ring by Castigliano's theorem). The bars are the
  // chords of the ring: 120 of them are 1e-3 off the circle. Every grid point has a rectangular displacement system
  // along the axes of the ring, in which the supports and the forces are stated.
  const [radius, count, f] = [0.5, 120, 100.0];
  const [o, z, x] = [SKEW.place([0, 0, 0]), SKEW.place([0, 0, 1]), SKEW.place([1, 0, 0])];
  const lines = [`RectFrame2;5;;${fields(o)};${fields(z)};${fields(x)}`, `Node;1000;;${fields(o)};5`,
    "SinglePointConstrList;1;123456;1000"];
  for (let k = 0; k < count; ++k) {
    const angle = (2 * Math.PI * k) / count;
    lines.push(`Node;${k + 1};;${fields(SKEW.place([radius * Math.cos(angle), radius * Math.sin(angle), 0]))};5`,
      `Bar2;${k + 1};1;${k + 1};${((k + 1) % count) + 1};;1000`, `SinglePointConstrList;1;345;${k + 1}`);
  }
  const [east, north, west, south] = [1, count / 4 + 1, count / 2 + 1, (3 * count) / 4 + 1];
  lines.push(barProp(1), MATERIAL, `SinglePointConstrList;1;2;${east};${west}`, `SinglePointConstrList;1;1;${north}`,
    `PointForceLoad;2;${north};5;${real(f)};0.;1.;0.`, `PointForceLoad;2;${south};5;${real(f)};0.;-1.;0.`);
  const run = runModel(deck(101, STATIC_CASE, lines), LINES);
  const check = measurements();
  const bend = (f * radius ** 3) / (E * SECTION.i1);
  const stretch = (f * radius) / (E * SECTION.area);
  const along = nodal(run, "displacements", north)[1] - nodal(run, "displacements", south)[1];
  const across = nodal(run, "displacements", east)[0] - nodal(run, "displacements", west)[0];
  check.ratio("growth of the diameter along the forces", along, (Math.PI / 4 - 2 / Math.PI) * bend +
    (Math.PI * stretch) / 4, CRITERIA.displacement);
  check.ratio("shrinking of the diameter across the forces", -across, (2 / Math.PI - 0.5) * bend - stretch / 2,
    CRITERIA.displacement);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

test("a helical spring of 180 Bar2 under a force along its axis", () => {
  // A helix of radius R, of n turns at the pitch angle a, of round wire of diameter d, clamped at one end and pulled
  // along its axis at the other (the ends reach the axis by arms a thousand times stiffer). Every section carries the
  // torque F R cos a and the bending moment F R sin a: the spring stretches by
  // F R² L (cos² a / (G J) + sin² a / (E I)), L = 2 π R n / cos a the length of the wire (A. M. Wahl, Mechanical
  // Springs, 1963; Timoshenko, Strength of Materials, part 1). The bars have no shear
  // flexibility, and the stretching of the wire adds F L sin² a / (E A), 1e-5 of the whole. The bars are the chords of
  // the helix: 60 per turn are 2e-3 off it.
  const [radius, turns, pitch, diameter, perTurn, f] = [0.05, 3, (6 * Math.PI) / 180, 0.008, 60, 50.0];
  const wire = { area: (Math.PI * diameter ** 2) / 4, i1: (Math.PI * diameter ** 4) / 64,
    i2: (Math.PI * diameter ** 4) / 64, j: (Math.PI * diameter ** 4) / 32 };
  const arm = { area: 1.0e2 * wire.area, i1: 1.0e3 * wire.i1, i2: 1.0e3 * wire.i2, j: 1.0e3 * wire.j };
  const count = turns * perTurn;
  const point = (k) => {
    const angle = (2 * Math.PI * k) / perTurn;
    return [radius * Math.cos(angle), radius * Math.sin(angle), radius * angle * Math.tan(pitch)];
  };
  const lines = [];
  for (let k = 0; k <= count; ++k) lines.push(`Node;${k + 1};;${fields(SKEW.place(point(k)))}`);
  for (let k = 0; k < count; ++k) {
    // The orientation: from the middle of the bar to the axis.
    const middle = scale(add(point(k), point(k + 1)), 0.5);
    const inward = fields(SKEW.turn([-middle[0], -middle[1], 0])).replace(";", ";;");
    lines.push(`Bar2;${k + 1};1;${k + 1};${k + 2};${inward}`);
  }
  const top = point(count);
  lines.push(`Node;1001;;${fields(SKEW.place([0, 0, 0]))}`, `Node;1002;;${fields(SKEW.place([0, 0, top[2]]))}`,
    `Bar2;1001;2;1001;1;${fields(SKEW.turn([0, 0, 1])).replace(";", ";;")}`,
    `Bar2;1002;2;${count + 1};1002;${fields(SKEW.turn([0, 0, 1])).replace(";", ";;")}`,
    barProp(1, wire), barProp(2, arm), MATERIAL, "SinglePointConstrList;1;123456;1001", force(2, 1002, [0, 0, f]));
  const run = runModel(deck(101, STATIC_CASE, lines), LINES);
  const check = measurements();
  const length = (2 * Math.PI * radius * turns) / Math.cos(pitch);
  const expected = f * radius ** 2 * length * (Math.cos(pitch) ** 2 / (G * wire.j) +
    Math.sin(pitch) ** 2 / (E * wire.i1)) + (f * length * Math.sin(pitch) ** 2) / (E * wire.area);
  check.ratio("stretch of the spring", localDisplacement(run, 1002)[2], expected, CRITERIA.displacement);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 3. Trusses

for (const card of ["Rod2", "Tube2"])
  test(`a tripod of ${card} under a force along its axis and a force across it (two subcases)`, () => {
    // Three rods of length L from feet on a circle to an apex on its axis, at the angle θ from the axis. Under a
    // force P along the axis each rod carries P / (3 cos θ) and the apex moves by P L / (3 E A cos² θ); under a force
    // H across the axis, toward the first foot, the first rod carries 2 H / (3 sin θ) and the apex moves by
    // 2 H L / (3 E A sin² θ) (the stiffness of the three rods across the axis is 3 E A sin² θ / (2 L)).
    const [circle, height, p, h] = [0.6, 1.0, 9000.0, 4000.0];
    const length = Math.hypot(circle, height);
    const [cosine, sine] = [height / length, circle / length];
    const area = card === "Rod2" ? 3.0e-4 : (Math.PI * (0.04 ** 2 - 0.034 ** 2)) / 4;
    const lines = [`Node;4;;${fields(SKEW.place([0, 0, height]))}`, "SinglePointConstrList;1;456;4"];
    for (let k = 0; k < 3; ++k) {
      const angle = (2 * Math.PI * k) / 3;
      lines.push(`Node;${k + 1};;${fields(SKEW.place([circle * Math.cos(angle), circle * Math.sin(angle), 0]))}`,
        `${card};${k + 1};1;${k + 1};4`, `SinglePointConstrList;1;123456;${k + 1}`);
    }
    lines.push(card === "Rod2" ? `RodProp;1;1;${real(area)};1.e-8` : "TubeProp;1;1;0.04;0.003", MATERIAL,
      force(2, 4, [0, 0, -p]), force(3, 4, [h, 0, 0]));
    const caseLines = ["DisplacementOutput;All", "ElemForceOutput;All", "StressOutput;All", "SubcaseStart;1",
      "SinglePointConstrSelection;1", "StaticLoadSelection;2", "SubcaseStart;2", "SinglePointConstrSelection;1",
      "StaticLoadSelection;3"];
    const run = runModel(deck(101, caseLines, lines), LINES);
    const check = measurements();
    check.vector("force along the axis: displacement of the apex", localDisplacement(run, 4, 1).slice(0, 3),
      [0, 0, (-p * length) / (3 * E * area * cosine ** 2)], CRITERIA.exact);
    for (const row of rowsOf(run, card, "forces", 1))
      check.ratio(`force along the axis: force of rod ${row.id}`, row.axial, -p / (3 * cosine), CRITERIA.exact);
    for (const row of rowsOf(run, card, "stress", 1))
      check.ratio(`force along the axis: stress of rod ${row.id}`, row.axial, -p / (3 * cosine * area),
        CRITERIA.exact);
    check.vector("force across the axis: displacement of the apex", localDisplacement(run, 4, 2).slice(0, 3),
      [(2 * h * length) / (3 * E * area * sine ** 2), 0, 0], CRITERIA.exact);
    // The first foot is on the side the force points to: its rod is compressed.
    check.ratio("force across the axis: force of the first rod", rowsOf(run, card, "forces", 2)[0].axial,
      (-2 * h) / (3 * sine), CRITERIA.exact);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. Sections that vary, neutral axis, end offsets, pin flags

// The integral of f over [a, b] by a composite five-point Gauss rule on 64 intervals.
function integrate(f, a, b) {
  const points = [0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640, -0.9061798459386640];
  const weights = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891,
    0.2369268850561891];
  const pieces = 64;
  let sum = 0;
  for (let piece = 0; piece < pieces; ++piece) {
    const [lower, upper] = [a + ((b - a) * piece) / pieces, a + ((b - a) * (piece + 1)) / pieces];
    points.forEach((x, k) => { sum += (weights[k] * f((lower + upper) / 2 + ((upper - lower) * x) / 2) *
      (upper - lower)) / 2; });
  }
  return sum;
}

test("a cantilever of two tapered Beam2 with stations: the beam whose section varies linearly", () => {
  // Each Beam2 goes from the section of its end A to a section of half the area and inertias at its end B, the
  // properties linear between the two stations: under end forces the deflections are F ∫ (L − x)² / (E I(x)) dx, the
  // stretch P ∫ dx / (E A(x)) and the twist T ∫ dx / (G J(x)) (the unit load theorem). The shear factors are zero:
  // no shear flexibility.
  const length = 1.2;
  const ratio = 0.5;
  const [fx, fy, fz, mx] = [4000.0, 60.0, -35.0, 25.0];
  const lines = [...straightLine({ card: "Beam2", from: [0, 0, 0], to: [length, 0, 0], count: 2 }),
    `BeamProp;1;1;${real(SECTION.area)};${real(SECTION.i1)};${real(SECTION.i2)};;${real(SECTION.j)};;No;1.;` +
      `${real(SECTION.area * ratio)};${real(SECTION.i1 * ratio)};${real(SECTION.i2 * ratio)};;` +
      `${real(SECTION.j * ratio)};;;0.;0.`,
    MATERIAL, "SinglePointConstrList;1;123456;1", force(2, 3, [fx, fy, fz]), moment(2, 3, [mx, 0, 0])];
  // The property at x: linear from 1 to `ratio` over each half of the cantilever.
  const factor = (x) => {
    const s = (x % (length / 2)) / (length / 2);
    return 1 + (ratio - 1) * s;
  };
  const over = (weight) => integrate((x) => weight(x) / factor(x), 0, length / 2) +
    integrate((x) => weight(x) / factor(x), length / 2, length);
  const expected = [(fx * over(() => 1)) / (E * SECTION.area), (fy * over((x) => (length - x) ** 2)) / (E * SECTION.i1),
    (fz * over((x) => (length - x) ** 2)) / (E * SECTION.i2), (mx * over(() => 1)) / (G * SECTION.j)];
  const check = measurements();
  for (const [name, options] of [["MixedHybridFormulation", LINES], ["ClassicFormulation", CLASSIC]]) {
    const run = runModel(deck(101, STATIC_CASE, lines), options);
    const tip = localDisplacement(run, 3);
    const criterion = name === "MixedHybridFormulation" ? CRITERIA.displacement : 1;
    check.ratio(`${name}: stretch`, tip[0], expected[0], criterion);
    check.ratio(`${name}: deflection in plane 1`, tip[1], expected[1], criterion);
    check.ratio(`${name}: deflection in plane 2`, tip[2], expected[2], criterion);
    check.ratio(`${name}: twist`, tip[3], expected[3], criterion);
  }
  check.done();
});

test("a cantilever Beam2 whose neutral axis is off its shear centre: an end force along the axis bends it", () => {
  // The neutral axis lies at the distance e from the axis of the grid points (the shear centre), in plane 1. A force
  // P along the axis at the end grid point acts at the distance e from the neutral axis: the beam bends under the
  // moment P e. The end moves by u = P L / (E A) + P e² L / (E I1) along the axis and by P e L² / (2 E I1) across it,
  // and turns by P e L / (E I1). A force F across the axis in plane 1 gives F L³ / (3 E I1) and moves the end by
  // F e L² / (2 E I1) along the axis (Maxwell's theorem).
  const [length, e, p, f] = [1.2, 0.012, 4000.0, 60.0];
  const property = ["BeamProp", 1, 1, real(SECTION.area), real(SECTION.i1), real(SECTION.i2), "", real(SECTION.j),
    "", ...new Array(8).fill(""), "", "0.", "0.", ...new Array(6).fill(""), ...new Array(4).fill(""), real(e), "0.",
    real(e), "0."].join(";");
  const model = (loads) => deck(101, STATIC_CASE, [...straightLine({ card: "Beam2", from: [0, 0, 0],
    to: [length, 0, 0], count: 2 }), property, MATERIAL, "SinglePointConstrList;1;123456;1", ...loads]);
  const check = measurements();
  const bent = (p * e * length ** 2) / (2 * E * SECTION.i1);
  const pulled = localDisplacement(runModel(model([force(2, 3, [p, 0, 0])]), LINES), 3);
  check.ratio("force along the axis: stretch", pulled[0], (p * length) / (E * SECTION.area) +
    (p * e * e * length) / (E * SECTION.i1), CRITERIA.exact);
  check.ratio("force along the axis: deflection (magnitude)", Math.abs(pulled[1]), bent, CRITERIA.exact);
  check.ratio("force along the axis: rotation (magnitude)", Math.abs(pulled[5]), (p * e * length) / (E * SECTION.i1),
    CRITERIA.exact);
  const pushed = localDisplacement(runModel(model([force(2, 3, [0, f, 0])]), LINES), 3);
  check.ratio("force across the axis: deflection", pushed[1], (f * length ** 3) / (3 * E * SECTION.i1),
    CRITERIA.exact);
  check.ratio("force across the axis: stretch (magnitude)", Math.abs(pushed[0]),
    (f * e * length ** 2) / (2 * E * SECTION.i1), CRITERIA.exact);
  // The two cross terms are equal with the sign of Maxwell's theorem.
  check.ratio("the cross terms are reciprocal", pulled[1] / p, pushed[0] / f, CRITERIA.exact);
  const classic = localDisplacement(runModel(model([force(2, 3, [p, 0, 0])]), CLASSIC), 3);
  check.vector("force along the axis: the displacements of ClassicFormulation", pulled.slice(0, 3),
    classic.slice(0, 3), CRITERIA.exact);
  check.done();
});

// A cantilever of one Bar2 of length L whose axis is offset by e along its orientation vector at both ends (WA, WB,
// in the displacement system of the grid points).
function offsetCantilever(loads, length, e) {
  const [a, b] = [SKEW.place([0, 0, 0]), SKEW.place([length, 0, 0])];
  const w = fields(SKEW.turn([0, e, 0]));
  return deck(101, STATIC_CASE, [`Node;1;;${fields(a)}`, `Node;2;;${fields(b)}`,
    `Bar2;1;1;1;2;${fields(SKEW.turn([0, 1, 0])).replace(";", ";;")};;;;${w};${w}`, barProp(1), MATERIAL,
    "SinglePointConstrList;1;123456;1", ...loads]);
}

test("a cantilever Bar2 with end offsets, loaded in the plane of the offsets: the rigid arms", () => {
  // The axis is at the distance e from the line of the grid points, in plane 1. A force P along the axis at the end
  // grid point bends the bar under the moment P e: u = P L / (E A) + P e² L / (E I1), deflection P e L² / (2 E I1),
  // rotation P e L / (E I1). A force F across the axis in plane 1 and a moment M about the normal of plane 1:
  // v = F L³ / (3 E I1) + M L² / (2 E I1).
  const [length, e, p, f, m] = [1.2, 0.02, 4000.0, 60.0, 15.0];
  const check = measurements();
  const pulled = localDisplacement(runModel(offsetCantilever([force(2, 2, [p, 0, 0])], length, e), LINES), 2);
  check.ratio("force along the axis: stretch", pulled[0], (p * length) / (E * SECTION.area) +
    (p * e * e * length) / (E * SECTION.i1), CRITERIA.exact);
  check.ratio("force along the axis: deflection (magnitude)", Math.abs(pulled[1]),
    (p * e * length ** 2) / (2 * E * SECTION.i1), CRITERIA.exact);
  check.ratio("force along the axis: rotation (magnitude)", Math.abs(pulled[5]), (p * e * length) / (E * SECTION.i1),
    CRITERIA.exact);
  const bent = localDisplacement(runModel(offsetCantilever([force(2, 2, [0, f, 0]), moment(2, 2, [0, 0, m])], length,
    e), LINES), 2);
  check.ratio("force and moment in plane 1: deflection", bent[1], (f * length ** 3) / (3 * E * SECTION.i1) +
    (m * length ** 2) / (2 * E * SECTION.i1), CRITERIA.exact);
  check.done();
});

test("a cantilever Bar2 with end offsets, loaded across the plane of the offsets (known limit)", () => {
  // KNOWN LIMIT (spec section 3.6: the offset of the axis is in the section law, not a rigid arm; the torsion and the
  // shear stay about the edge). A force F across plane 1 at the end grid point: the ClassicFormulation bar is twisted
  // by the torque F e of its rigid arm, w = F L³ / (3 E I2) + F e² L / (G J) and θx = F e L / (G J); the bar of the
  // formulation is not, w = F L³ / (3 E I2) and θx = 0. Both are checked as the spec states them.
  const [length, e, f] = [1.2, 0.02, 35.0];
  const text = offsetCantilever([force(2, 2, [0, 0, f])], length, e);
  const check = measurements();
  const bending = (f * length ** 3) / (3 * E * SECTION.i2);
  const twist = (f * e * length) / (G * SECTION.j);
  const classic = localDisplacement(runModel(text, CLASSIC), 2);
  check.ratio("ClassicFormulation: deflection with the twist of the arm", classic[2], bending + twist * e,
    CRITERIA.exact);
  check.ratio("ClassicFormulation: twist (magnitude)", Math.abs(classic[3]), twist, CRITERIA.exact);
  const mixed = localDisplacement(runModel(text, LINES), 2);
  check.ratio("MixedHybridFormulation: deflection without the twist of the arm", mixed[2], bending, CRITERIA.exact);
  check.small("MixedHybridFormulation: twist", mixed[3], twist, CRITERIA.exact);
  console.log(`    info ratio deflection of MixedHybridFormulation over ClassicFormulation: ` +
    `${(mixed[2] / classic[2]).toFixed(6)}`);
  check.done();
});

test("pin flags of Bar2 alone: a hinge in each plane, an axial release and a torsion release", () => {
  // A beam of length L clamped at both ends, of two elements, loaded at its middle grid point. The first element
  // releases components at its end B, the middle grid point (PB):
  //   56: a hinge; each half is a cantilever under half the force: F L³ / (48 E I) in each plane;
  //   1: the first half takes no axial force: u = P L / (2 E A);
  //   4: the first half takes no torque: θ = T L / (2 G J).
  const length = 2.0;
  const [p, fy, fz, t] = [5000.0, 80.0, -50.0, 30.0];
  const model = (pins, loads) => {
    const lines = straightLine({ from: [0, 0, 0], to: [length, 0, 0], count: 2 });
    const first = lines.findIndex((line) => line.startsWith("Bar2;1;"));
    lines[first] += `;;;${pins}`;
    return deck(101, STATIC_CASE, [...lines, barProp(1), MATERIAL, "SinglePointConstrList;1;123456;1;3", ...loads]);
  };
  const check = measurements();
  const hinged = runModel(model("56", [force(2, 2, [0, fy, fz])]), LINES);
  const middle = localDisplacement(hinged, 2);
  check.ratio("hinge: deflection in plane 1", middle[1], (fy * length ** 3) / (48 * E * SECTION.i1), CRITERIA.exact);
  check.ratio("hinge: deflection in plane 2", middle[2], (fz * length ** 3) / (48 * E * SECTION.i2), CRITERIA.exact);
  const forces = rowsOf(hinged, "Bar2", "forces");
  check.small("hinge: moments of the first element at the hinge", Math.hypot(forces[0].bendingMomentB1,
    forces[0].bendingMomentB2), (fy * length) / 4, CRITERIA.exact);
  check.ratio("hinge: moment in plane 1 at the first clamp (magnitude)", Math.abs(forces[0].bendingMomentA1),
    (fy * length) / 4, CRITERIA.exact);
  const sliding = runModel(model("1", [force(2, 2, [p, 0, 0])]), LINES);
  check.ratio("axial release: displacement", localDisplacement(sliding, 2)[0], (p * length) / (2 * E * SECTION.area),
    CRITERIA.exact);
  check.small("axial release: axial force of the first element", rowsOf(sliding, "Bar2", "forces")[0].axial, p,
    CRITERIA.exact);
  const turning = runModel(model("4", [moment(2, 2, [t, 0, 0])]), LINES);
  check.ratio("torsion release: twist", localDisplacement(turning, 2)[3], (t * length) / (2 * G * SECTION.j),
    CRITERIA.exact);
  for (const run of [hinged, sliding, turning])
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 5. Loads along the lines, temperatures, deformations

test("BeamDistributedLoad on a cantilever of two Bar2: uniform, linear, partial, along a basic axis, a couple", () => {
  // A cantilever of length L (two elements) in the skew frame, one load set per subcase:
  //   1. a uniform force q in plane 1 (ForceYAlongLength, FractionalLength): q L⁴ / (8 E I1), clamp moment q L² / 2;
  //   2. a force in plane 2 that grows linearly from 0 at the clamp to q at the end (ActualLength):
  //      11 q L⁴ / (120 E I2);
  //   3. a uniform force q in plane 1 on a part [a, b] of the second element: q (L (b³ − a³) − (b⁴ − a⁴) / 4) /
  //      (6 E I1) (the end deflection x² (3 L − x) / (6 E I) of a force at x, integrated);
  //   4. a uniform force q along the basic z axis (ForceZ): its components c on the axes of the bar give
  //      q cx L² / (2 E A), q cy L⁴ / (8 E I1), q cz L⁴ / (8 E I2);
  //   5. a uniform couple m about the axis (MomentXAlongLength): m L² / (2 G J).
  // (Timoshenko, Strength of Materials, part 1, chapter 5, for the deflections of the cantilever.)
  const length = 1.2;
  const half = length / 2;
  const [q, m] = [300.0, 40.0];
  const loads = [
    `BeamDistributedLoad;21;1;ForceYAlongLength;FractionalLength;0.;${real(q)};1.;${real(q)}`,
    `BeamDistributedLoad;21;2;ForceYAlongLength;FractionalLength;0.;${real(q)};1.;${real(q)}`,
    `BeamDistributedLoad;22;1;ForceZAlongLength;ActualLength;0.;0.;${real(half)};${real(q / 2)}`,
    `BeamDistributedLoad;22;2;ForceZAlongLength;ActualLength;0.;${real(q / 2)};${real(half)};${real(q)}`,
    `BeamDistributedLoad;23;2;ForceYAlongLength;FractionalLength;0.2;${real(q)};0.7;${real(q)}`,
    `BeamDistributedLoad;24;1;ForceZ;FractionalLength;0.;${real(q)};1.;${real(q)}`,
    `BeamDistributedLoad;24;2;ForceZ;FractionalLength;0.;${real(q)};1.;${real(q)}`,
    `BeamDistributedLoad;25;1;MomentXAlongLength;FractionalLength;0.;${real(m)};1.;${real(m)}`,
    `BeamDistributedLoad;25;2;MomentXAlongLength;FractionalLength;0.;${real(m)};1.;${real(m)}`];
  const caseLines = ["DisplacementOutput;All", "ElemForceOutput;All"];
  for (let subcase = 1; subcase <= 5; ++subcase)
    caseLines.push(`SubcaseStart;${subcase}`, "SinglePointConstrSelection;1", `StaticLoadSelection;${20 + subcase}`);
  const lines = [...straightLine({ from: [0, 0, 0], to: [length, 0, 0], count: 2 }), barProp(1), MATERIAL,
    "SinglePointConstrList;1;123456;1", ...loads];
  const run = runModel(deck(101, caseLines, lines), LINES);
  const check = measurements();
  check.ratio("uniform force: deflection of the end", localDisplacement(run, 3, 1)[1],
    (q * length ** 4) / (8 * E * SECTION.i1), CRITERIA.exact);
  // Under a load along it a Bar2 writes its forces at stations (Bar2LoadStations): the first is the clamp.
  check.ratio("uniform force: moment at the clamp (magnitude)", Math.abs(rowsOf(run, "Bar2LoadStations", "forces", 1)[0]
    .bendingMoment1), (q * length ** 2) / 2, CRITERIA.exact);
  check.ratio("linear force: deflection of the end", localDisplacement(run, 3, 2)[2],
    (11 * q * length ** 4) / (120 * E * SECTION.i2), CRITERIA.exact);
  const [a, b] = [half + 0.2 * half, half + 0.7 * half];
  check.ratio("partial force: deflection of the end", localDisplacement(run, 3, 3)[1],
    (q * (length * (b ** 3 - a ** 3) - (b ** 4 - a ** 4) / 4)) / (6 * E * SECTION.i1), CRITERIA.exact);
  const c = SKEW.local([0, 0, 1]);
  check.vector("force along the basic z axis: displacement of the end", localDisplacement(run, 3, 4).slice(0, 3),
    [(q * c[0] * length ** 2) / (2 * E * SECTION.area), (q * c[1] * length ** 4) / (8 * E * SECTION.i1),
      (q * c[2] * length ** 4) / (8 * E * SECTION.i2)], CRITERIA.exact);
  check.ratio("uniform couple: twist of the end", localDisplacement(run, 3, 5)[3],
    (m * length ** 2) / (2 * G * SECTION.j), CRITERIA.exact);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

// A line of length L of two elements under the temperatures of LineElemTemperatureLoad: the rise ΔT and the gradients
// T'y and T'z across the section, the same at both ends; clamped at its first end, or at both ends (`held`).
function heatedLine(card, held, rise, gy, gz) {
  const property = card === "Bar2" ? barProp(1)
    : `BeamProp;1;1;${real(SECTION.area)};${real(SECTION.i1)};${real(SECTION.i2)};;${real(SECTION.j)}`;
  return deck(101, ["TemperatureSelection;7;TemperatureLoadPart", "SubcaseStart;1",
    "SinglePointConstrSelection;1", "DisplacementOutput;All", "ElemForceOutput;All"],
  [...straightLine({ card, from: [0, 0, 0], to: [1.2, 0, 0], count: 2 }), property, MATERIAL,
    `SinglePointConstrList;1;123456;1${held ? ";3" : ""}`,
    ...[1, 2].map((element) => `LineElemTemperatureLoad;7;${element};${real(rise)};${real(rise)};${real(gy)};` +
      `${real(gy)};${real(gz)};${real(gz)}`)]);
}

for (const card of ["Bar2", "Beam2"])
  test(`a temperature rise on two ${card}: free and held`, () => {
    // Clamped at one end: u = α ΔT L and no force. Clamped at both ends: no displacement, N = −E A α ΔT.
    const [length, rise] = [1.2, 40.0];
    const axialName = card === "Bar2" ? "axial" : "axialForce";
    const scaleForce = E * SECTION.area * ALPHA * rise;
    const check = measurements();
    const free = runModel(heatedLine(card, false, rise, 0, 0), LINES);
    check.vector("free end: displacement", localDisplacement(free, 3).slice(0, 3), [ALPHA * rise * length, 0, 0],
      CRITERIA.exact);
    for (const row of rowsOf(free, card, "forces"))
      check.small(`free end: axial force of element ${row.id}`, row[axialName], scaleForce, CRITERIA.exact);
    const held = runModel(heatedLine(card, true, rise, 0, 0), LINES);
    check.small("both ends held: displacement of the middle", Math.max(...localDisplacement(held, 2).slice(0, 3)
      .map(Math.abs)), ALPHA * rise * length, CRITERIA.exact);
    check.ratio("both ends held: axial force", rowsOf(held, card, "forces")[0][axialName], -scaleForce,
      CRITERIA.exact);
    check.done();
  });

for (const card of ["Bar2", "Beam2"])
  test(`temperature gradients across two ${card} (LineElemTemperatureLoad): free and held`, () => {
    // Clamped at one end: v = −α T'y L² / 2, w = −α T'z L² / 2 (the hot side lengthens: the line bends away from it),
    // no force. Clamped at both ends: no displacement, moments E I1 α T'y and E I2 α T'z (B. A. Boley and
    // J. H. Weiner, Theory of Thermal Stresses). Spec section 7 lists the gradients of the lines in the
    // thermal facet.
    const [length, gy, gz] = [1.2, 300.0, -200.0];
    const momentNames = card === "Bar2" ? ["bendingMomentA1", "bendingMomentA2"] : ["bendingMoment1", "bendingMoment2"];
    const check = measurements();
    const free = runModel(heatedLine(card, false, 0, gy, gz), LINES);
    check.vector("free end: displacement", localDisplacement(free, 3).slice(0, 3), [0,
      (-ALPHA * gy * length ** 2) / 2, (-ALPHA * gz * length ** 2) / 2], CRITERIA.exact);
    const held = runModel(heatedLine(card, true, 0, gy, gz), LINES);
    const row = rowsOf(held, card, "forces")[0];
    check.ratio("both ends held: moment in plane 1 (magnitude)", Math.abs(row[momentNames[0]]),
      E * SECTION.i1 * ALPHA * Math.abs(gy), CRITERIA.exact);
    check.ratio("both ends held: moment in plane 2 (magnitude)", Math.abs(row[momentNames[1]]),
      E * SECTION.i2 * ALPHA * Math.abs(gz), CRITERIA.exact);
    check.done();
  });

for (const card of ["Rod2", "Bar2", "Beam2"])
  test(`AxialDeformationLoad on two ${card}: free and held`, () => {
    // The first element of a line of two elements of length l is given the deformation δ. Clamped at one end: the
    // other end moves by δ, no force. Clamped at both ends: the middle moves by δ / 2 and the line is compressed by
    // E A δ / (2 l).
    const [length, delta] = [1.2, 3.0e-4];
    const area = card === "Rod2" ? 3.0e-4 : SECTION.area;
    const property = card === "Rod2" ? `RodProp;1;1;${real(area)};1.e-8` : card === "Bar2" ? barProp(1)
      : `BeamProp;1;1;${real(SECTION.area)};${real(SECTION.i1)};${real(SECTION.i2)};;${real(SECTION.j)}`;
    // A rod has no stiffness across its axis: its free grid points are held there, in a displacement system along
    // the rod (the automatic constraint of the singular components would hold basic axes, which are skew to it).
    const frame = `RectFrame2;5;;${fields(SKEW.place([0, 0, 0]))};${fields(SKEW.place([0, 0, 1]))};` +
      `${fields(SKEW.place([1, 0, 0]))}`;
    const model = (held) => deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "DeformationLoadSelection;4",
      "DisplacementOutput;All", "ElemForceOutput;All"],
    [frame, ...straightLine({ card, from: [0, 0, 0], to: [length, 0, 0], count: 2, displacementFrame: 5 }), property,
      MATERIAL, `SinglePointConstrList;1;123456;1${held ? ";3" : ""}`,
      ...(card === "Rod2" ? ["SinglePointConstrList;1;23456;2;3"] : []), `AxialDeformationLoad;4;1;${real(delta)}`]);
    const axialName = card === "Beam2" ? "axialForce" : "axial";
    const check = measurements();
    // The displacements are in the displacement system of the grid points, along the line.
    const free = runModel(model(false), LINES);
    check.vector("free end: displacement", nodal(free, "displacements", 3).slice(0, 3), [delta, 0, 0],
      CRITERIA.exact);
    check.small("free end: axial force", rowsOf(free, card, "forces")[0][axialName], (E * area * delta) / length,
      CRITERIA.exact);
    const held = runModel(model(true), LINES);
    check.vector("both ends held: displacement of the middle", nodal(held, "displacements", 2).slice(0, 3),
      [delta / 2, 0, 0], CRITERIA.exact);
    for (const row of rowsOf(held, card, "forces"))
      check.ratio(`both ends held: axial force of element ${row.id}`, row[axialName], (-E * area * delta) / length,
        CRITERIA.exact);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 6. Modes and buckling

// A rectangular displacement system along the axes of the skew frame: the supports of a line in a skew direction
// are stated in it (grid points with line elements hold what they state: spec section 5.6).
const SKEW_FRAME = `RectFrame2;5;;${fields(SKEW.place([0, 0, 0]))};${fields(SKEW.place([0, 0, 1]))};` +
  `${fields(SKEW.place([1, 0, 0]))}`;

// The frequency (ω²) of the mode of wavenumber k of a simply supported beam with shear flexibility and rotary
// inertia (S. P. Timoshenko, Vibration Problems in Engineering): the lower root of
//   (κ G A k² − ρ A ω²)(E I k² + κ G A − ρ I ω²) − (κ G A)² k² = 0;
// without shear flexibility (κ G A infinite) it is Rayleigh's ω² = E I k⁴ / (ρ (A + I k²)).
//
// The line element of the formulation is not this beam in its mass: its rotation is the slope of its axis (spec
// section 3.2: θ = ψ t + t × du/ds, the shear flexibility in the complementary energy of the shear force), so that
// the rotary inertia of the section works on the slope and not on the rotation of the section. Its own frequency is
//   ω² = E I k⁴ / ((1 + E I k² / (κ G A)) ρ (A + I k²))   (lineEigenvalue),
// the static stiffness of Timoshenko's beam with the rotary inertia on the slope: the same without shear
// flexibility, lower with it by a part that grows as (I k² / A)(E / κ G) (the reference measures 0.99999, 0.99980 and
// 0.99905 on the first three modes of its beam: its document 34, section 11.5).
function lineEigenvalue(k, inertia, shearFactor) {
  const flexibility = shearFactor === 0 ? 0 : (E * inertia * k * k) / (shearFactor * G * SECTION.area);
  return (E * inertia * k ** 4) / ((1 + flexibility) * RHO * (SECTION.area + inertia * k * k));
}
function beamEigenvalue(k, inertia, shearFactor) {
  if (shearFactor === 0) return (E * inertia * k ** 4) / (RHO * (SECTION.area + inertia * k * k));
  const s = shearFactor * G * SECTION.area;
  const [m0, m2] = [RHO * SECTION.area, RHO * inertia];
  const [qa, qb, qc] = [m0 * m2, -(s * k * k * m2 + m0 * (E * inertia * k * k + s)), s * E * inertia * k ** 4];
  return (2 * qc) / (-qb + Math.sqrt(qb * qb - 4 * qa * qc));
}

for (const { name, length, shearFactor } of [
  { name: "a slender beam without shear flexibility (Euler-Bernoulli with rotary inertia)", length: 1.0,
    shearFactor: 0 },
  { name: "a short beam with shear factors (Timoshenko)", length: 0.3, shearFactor: 0.85 }])
  test(`modes of ${name} of 16 Bar2 on simple supports: bending in both planes, torsion, stretching`, () => {
    // The beam is held at its first end (three translations and the twist) and across its axis at its second end.
    // Bending with n half-waves in each plane: beamEigenvalue(n π / L). Torsion, the twist held at one end:
    // ω = (2 n − 1) π / (2 L) √(G J / (ρ Ip)), Ip = I1 + I2. Stretching, one end held: ω = (2 n − 1) π / (2 L) √(E / ρ)
    // (R. D. Blevins, Formulas for Natural Frequency and Mode Shape, 1979, chapter 8). The modes are compared in
    // the order of their frequencies; the bending modes counted have at least four elements per half-wave.
    const count = 16;
    const lines = [SKEW_FRAME, ...straightLine({ from: [0, 0, 0], to: [length, 0, 0], count, displacementFrame: 5 }),
      barProp(1, shearFactor === 0 ? {} : { k1: shearFactor, k2: shearFactor }), MATERIAL,
      "SinglePointConstrList;1;1234;1", `SinglePointConstrList;1;23;${count + 1}`, eigenMethod(14)];
    const run = runModel(deck(103, MODES_CASE, lines), LINES, "modes");
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    const expected = [];
    for (let n = 1; n <= 6; ++n) {
      const k = (n * Math.PI) / length;
      for (const [plane, inertia] of [[1, SECTION.i1], [2, SECTION.i2]])
        expected.push({ name: `bending in plane ${plane}, ${n} half-waves`,
          value: beamEigenvalue(k, inertia, shearFactor), own: lineEigenvalue(k, inertia, shearFactor),
          counted: n <= 4 });
    }
    for (let n = 1; n <= 2; ++n) {
      const k = ((2 * n - 1) * Math.PI) / (2 * length);
      const torsion = (k * k * G * SECTION.j) / (RHO * (SECTION.i1 + SECTION.i2));
      const stretching = (k * k * E) / RHO;
      expected.push({ name: `torsion ${n}`, value: torsion, own: torsion, counted: true },
        { name: `stretching ${n}`, value: stretching, own: stretching, counted: true });
    }
    expected.sort((p, q) => p.value - q.value);
    const check = measurements();
    for (let mode = 0; mode < 10; ++mode) {
      const { name: modeName, value, own, counted } = expected[mode];
      const ratio = Math.sqrt(values[mode] / value);
      // KNOWN LIMIT of the short beam (the rotary inertia on the slope, above): the modes whose frequency of the
      // formulation is more than the criterion below Timoshenko's are counted against the formulation's own
      // closed form, and their ratio to Timoshenko's is printed (0.9973 to 0.9886 here).
      const isOwn = Math.abs(Math.sqrt(own / value) - 1) > CRITERIA.frequency / 2;
      if (!counted || isOwn) console.log(`    info ratio frequency ${mode + 1}, ${modeName}: ${ratio.toFixed(6)}`);
      if (counted && !isOwn) check.ratio(`frequency ${mode + 1}, ${modeName}`, ratio, 1, CRITERIA.frequency);
      if (counted && isOwn)
        check.ratio(`frequency ${mode + 1}, ${modeName}, over the formulation's own closed form (known limit)`,
          Math.sqrt(values[mode] / own), 1, CRITERIA.frequency);
    }
    for (const row of run.eigenvalues[0].rows.slice(0, 10))
      check.ratio(`generalised mass of mode ${row.mode}`, row.generalizedMass, 1, 1e-8);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

test("a free beam of 16 Bar2: six rigid-body modes and the free-free bending frequencies", () => {
  // Free-free bending: ω = λ² √(E I / (ρ A)) / L² with λ = 4.73004 (Blevins, Formulas for Natural Frequency and Mode
  // Shape, chapter 8), less the rotary inertia by Rayleigh's quotient with that mode shape:
  // ω² / ω0² = 1 / (1 + (I / A) ∫φ'² / ∫φ²), ∫φ'² / ∫φ² = 49.4808 / L² for the first free-free mode. Slender (L = 2):
  // the correction is 1e-3 of ω².
  const length = 2.0;
  const count = 16;
  const lines = [...straightLine({ from: [0, 0, 0], to: [length, 0, 0], count }), barProp(1), MATERIAL,
    eigenMethod(8)];
  const run = runModel(deck(103, ["SubcaseStart;1", "EigenMethodSelection;3"], lines), LINES, "modes");
  const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
  const check = measurements();
  for (let mode = 0; mode < 6; ++mode)
    check.small(`rigid-body frequency ${mode + 1}`, Math.sqrt(Math.abs(values[mode])), Math.sqrt(values[6]), 1e-3);
  const lambda = 4.730040745;
  const slope = 49.4808 / length ** 2;
  for (const [mode, inertia, plane] of [[6, SECTION.i2, 2], [7, SECTION.i1, 1]]) {
    const thin = (lambda ** 4 * E * inertia) / (RHO * SECTION.area * length ** 4);
    check.ratio(`first bending frequency in plane ${plane}`, Math.sqrt(values[mode] /
      (thin / (1 + (inertia / SECTION.area) * slope))), 1, CRITERIA.frequency);
  }
  check.done();
});

// The smallest positive buckling factors of a run, sorted.
function bucklingFactors(text, options) {
  const run = runModel(text, options, "buckling");
  return { run, factors: run.eigenvalues[0].rows.map((row) => row.eigenvalue).filter((value) => value > 0)
    .sort((p, q) => p - q) };
}

// The four classical columns: P_cr = c E I / L² (Timoshenko and Gere, Theory of Elastic Stability, chapter 2). The
// second coefficient of each is that of its second mode in the same plane.
const COLUMNS = [
  { name: "pinned at both ends", first: "1234", second: "23", coefficients: [Math.PI ** 2, 4 * Math.PI ** 2] },
  { name: "clamped and free", first: "123456", second: "", coefficients: [Math.PI ** 2 / 4, (9 * Math.PI ** 2) / 4] },
  { name: "clamped and pinned", first: "123456", second: "23", coefficients: [4.493409458 ** 2, 7.725251837 ** 2] },
  { name: "clamped at both ends", first: "123456", second: "23456",
    coefficients: [4 * Math.PI ** 2, (2 * 4.493409458) ** 2] },
];

for (const { name, first, second, coefficients } of COLUMNS)
  test(`buckling of a column of 8 Bar2 ${name} (Euler)`, () => {
    // A column in the skew direction under a unit compression at its second end; its supports in the displacement
    // system along it. Both planes buckle: the expected loads are those of the first two modes of each plane.
    const [length, count] = [1.5, 8];
    const lines = [SKEW_FRAME, ...straightLine({ from: [0, 0, 0], to: [length, 0, 0], count, displacementFrame: 5 }),
      barProp(1), MATERIAL, `SinglePointConstrList;1;${first};1`,
      ...(second ? [`SinglePointConstrList;1;${second};${count + 1}`] : []),
      `PointForceLoad;2;${count + 1};5;1.;-1.;0.;0.`, eigenMethod(4)];
    const { run, factors } = bucklingFactors(deck(105, BUCKLING_CASE, lines), LINES);
    const expected = [];
    for (const inertia of [SECTION.i1, SECTION.i2])
      for (const coefficient of coefficients) expected.push((coefficient * E * inertia) / length ** 2);
    expected.sort((p, q) => p - q);
    const check = measurements();
    // The first mode of each plane has eight elements per half-wave or more; the third load (a second mode, four
    // elements per half-wave or less) is printed.
    for (let mode = 0; mode < 2; ++mode)
      check.ratio(`critical load ${mode + 1}`, factors[mode], expected[mode], CRITERIA.criticalLoad);
    console.log(`    info ratio critical load 3: ${(factors[2] / expected[2]).toFixed(6)}`);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

test("buckling of a portal frame with fixed feet under a force on each column: the sway mode", () => {
  // Two columns of height h (inertia Ic) and a beam of span L (inertia Ib) in a skew plane, a force P down each
  // column. In the sway mode the beam bends in antisymmetry and holds the top of each column with the stiffness
  // 6 E Ib / L: the column, clamped at its foot and free to sway, buckles at P = (k h)² E Ic / h² with
  // tan(k h) = −k h Ic L / (6 Ib h), π/2 < k h < π (Timoshenko and Gere, Theory of Elastic Stability, chapter 2).
  // The frame is held in its plane (the displacement across it and the rotations in it, in the skew displacement
  // system), and its members do not stretch enough to matter (I / (A h²) = 3e-7).
  const [h, span] = [1.5, 2.0];
  const column = { area: 0.06, i1: 4.5e-8, i2: 2.0e-8, j: 4.7e-8 };
  const beam = { area: 0.06, i1: 9.0e-8, i2: 2.0e-8, j: 4.7e-8 };
  const lines = [SKEW_FRAME,
    ...straightLine({ from: [0, 0, 0], to: [0, h, 0], count: 4, first: 1, firstElement: 1, orientation: [1, 0, 0],
      displacementFrame: 5 }),
    ...straightLine({ from: [span, 0, 0], to: [span, h, 0], count: 4, first: 11, firstElement: 11,
      orientation: [1, 0, 0], displacementFrame: 5 }),
    ...straightLine({ from: [0, h, 0], to: [span, h, 0], count: 4, first: 21, firstElement: 21, property: 2,
      displacementFrame: 5 }).filter((line) => !/^Node;2[15];/.test(line))
      .map((line) => line.replace(/^(Bar2;21;2);21;/, "$1;5;").replace(/^(Bar2;24;2;24);25;/, "$1;15;")),
    barProp(1, column), barProp(2, beam), MATERIAL, "SinglePointConstrList;1;123456;1;11",
    "SinglePointConstrList;1;345;2;Through;5", "SinglePointConstrList;1;345;12;Through;15",
    "SinglePointConstrList;1;345;22;Through;24",
    "PointForceLoad;2;5;5;1.;0.;-1.;0.", "PointForceLoad;2;15;5;1.;0.;-1.;0.", eigenMethod(3)];
  const { run, factors } = bucklingFactors(deck(105, BUCKLING_CASE, lines), LINES);
  // The root of tan(x) + x Ic L / (6 Ib h) = 0 between π/2 and π, by bisection.
  const slope = (column.i1 * span) / (6 * beam.i1 * h);
  let [lower, upper] = [Math.PI / 2 + 1e-9, Math.PI];
  for (let step = 0; step < 100; ++step) {
    const middle = (lower + upper) / 2;
    if (Math.tan(middle) + middle * slope < 0) lower = middle;
    else upper = middle;
  }
  const check = measurements();
  check.ratio("critical load of the sway mode", factors[0], (lower ** 2 * E * column.i1) / h ** 2,
    CRITERIA.criticalLoad);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 7. Lines that end on shells and on solids (spec section 3.4, tie 6, and the pinned end on a solid)

test("three Bar2 that continue a cantilever strip of ShellQuad4: their rotation is that of the shells (tie 6)", () => {
  // A strip L1 × b of 4 × 2 shells without Poisson's ratio, clamped at x = 0; from each of the three grid points of
  // its end a bar goes on along x for L2, with a force at its end (the bars and the forces shared 1/4, 1/2, 1/4).
  // Each bar turns with the mean rotation of the shells at its grid point (tie 6), which the displacement table
  // gives: its end deflects by w + θ L2 + F L2³ / (3 E I) from the deflection w and the rotation θ of that grid
  // point (the slope is −θy). The clamp of the strip carries the moment of the forces. Away from the junction the
  // strip is the beam under the force F and the couple F L2 at its end: w(x) = F x² (3 L1 − x) / (6 E Is) +
  // F L2 x² / (2 E Is) + F x / (κ G b t). At the junction each bar puts a concentrated moment on the shells, where the
  // rotation converges slowly: the deflection of the ends of the bars is printed against the beam's, not counted.
  const [l1, width, thickness, l2, f] = [1.0, 0.4, 0.01, 0.5, 20.0];
  const young = 7.0e10;
  const rigidity = (young * width * thickness ** 3) / 12;
  const bar = { area: 4.0e-4, i1: (2 * rigidity) / E, i2: 1.0e-8, j: 1.0e-8 }; // the three bars: twice the strip
  const node = (i, j) => 1 + j * 5 + i;
  const lines = ["ShellProp;1;2;0.01;2;;2", `IsoMat;2;${real(young)};;0.;2700.`, MATERIAL];
  for (let j = 0; j <= 2; ++j)
    for (let i = 0; i <= 4; ++i) lines.push(`Node;${node(i, j)};;${real((l1 * i) / 4)};${real((width * j) / 2)};0.`);
  for (let j = 0; j < 2; ++j)
    for (let i = 0; i < 4; ++i)
      lines.push(`ShellQuad4;${1 + j * 4 + i};1;${[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
        .join(";")}`);
  for (let j = 0; j <= 2; ++j) {
    const share = j === 1 ? 0.5 : 0.25;
    lines.push(`SinglePointConstrList;1;123456;${node(0, j)}`,
      `Node;${100 + j};;${real(l1 + l2)};${real((width * j) / 2)};0.`,
      `Bar2;${100 + j};${10 + j};${node(4, j)};${100 + j};0.;;0.;1.`,
      barProp(10 + j, { area: bar.area * share, i1: bar.i1 * share, i2: bar.i2 * share, j: bar.j * share }),
      `PointForceLoad;2;${100 + j};;${real(f * share)};0.;0.;1.`);
  }
  const run = runModel(deck(101, STATIC_CASE, lines), MIXED);
  const check = measurements();
  const beam = (x) => (f * x * x * (3 * l1 - x)) / (6 * rigidity) + (f * l2 * x * x) / (2 * rigidity) +
    (f * x) / ((0.833333 * young * width * thickness) / 2);
  for (let j = 0; j <= 2; ++j) {
    const share = j === 1 ? 0.5 : 0.25;
    const root = nodal(run, "displacements", node(4, j));
    const end = nodal(run, "displacements", 100 + j);
    check.ratio(`bar ${j + 1}: deflection of its end from the motion of its grid point of shells`, end[2],
      root[2] - root[4] * l2 + (f * share * l2 ** 3) / (3 * E * bar.i1 * share), CRITERIA.exact);
    check.ratio(`bar ${j + 1}: rotation of its end`, end[4], root[4] - (f * share * l2 ** 2) / (2 * E * bar.i1 * share),
      CRITERIA.exact);
    check.ratio(`strip: deflection at mid-length, row ${j}`, nodal(run, "displacements", node(2, j))[2], beam(l1 / 2),
      CRITERIA.displacement);
    const expected = beam(l1) + (f * l1 ** 2 / (2 * rigidity) + (f * l2 * l1) / rigidity) * l2 +
      (f * l2 ** 3) / (3 * E * bar.i1);
    console.log(`    info ratio deflection of the end of bar ${j + 1} over the beam's: ` +
      `${(end[2] / expected).toFixed(6)}`);
  }
  let clampMoment = 0;
  let clampForce = 0;
  for (let j = 0; j <= 2; ++j) {
    const reaction = nodal(run, "singlePointConstrForces", node(0, j));
    clampForce += reaction[2];
    clampMoment += reaction[4];
  }
  check.ratio("force at the clamp", clampForce, -f, CRITERIA.invariant);
  check.ratio("moment at the clamp", clampMoment, f * (l1 + l2), CRITERIA.invariant);
  const others = warningsOf(run);
  check.truth(`no warning: ${others}`, others.length === 0);
  check.done();
});

test("a Bar2 from a grid point of a SolidHexa to a pinned support: the end on the solid is pinned", () => {
  // A stiff block (a SolidHexa a hundred times stiffer than the bar's material) held on its base, a bar from one of
  // its top corners to a support that holds its three translations and its twist, a force across the bar at its
  // middle. A grid point of solids has no rotation: the bar is pinned there, and it is the beam on simple supports,
  // F L³ / (48 E I) in each plane (clamped on the solid it would deflect 0.44 of that).
  const [side, length] = [0.2, 1.0];
  const [fy, fz] = [80.0, -50.0];
  const lines = [];
  const corner = (i, j, k) => 1 + i + 2 * j + 4 * k;
  for (let k = 0; k <= 1; ++k)
    for (let j = 0; j <= 1; ++j)
      for (let i = 0; i <= 1; ++i)
        lines.push(`Node;${corner(i, j, k)};;${real(side * i)};${real(side * j)};${real(side * k)}`);
  lines.push(`SolidHexa;1;5;${[corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 0), corner(0, 0, 1),
    corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)].join(";")}`, "SolidProp;5;3",
  `IsoMat;3;${real(100 * E)};;0.3;7850.`,
  `Node;11;;${real(side + length / 2)};${real(side)};${real(side)}`,
  `Node;12;;${real(side + length)};${real(side)};${real(side)}`,
  `Bar2;11;1;${corner(1, 1, 1)};11;0.;;1.;0.`, "Bar2;12;1;11;12;0.;;1.;0.", barProp(1), MATERIAL,
  `SinglePointConstrList;1;123;${[corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 0)].join(";")}`,
  "SinglePointConstrList;1;1234;12", `PointForceLoad;2;11;;1.;0.;${real(fy)};${real(fz)}`);
  const sharpEdges = [];
  for (const t of [0, 1])
    for (const u of [0, 1])
      sharpEdges.push([corner(0, t, u), corner(1, t, u)], [corner(t, 0, u), corner(t, 1, u)],
        [corner(t, u, 0), corner(t, u, 1)]);
  // The twelve edges of the cube are declared sharp in the model file.
  lines.push(...surfaceRecords({ sharpEdges }));
  const run = runModel(deck(101, STATIC_CASE.filter((line) => line !== "StressOutput;All"), lines), MIXED);
  const check = measurements();
  const middle = nodal(run, "displacements", 11);
  check.ratio("deflection in plane 1", middle[1], (fy * length ** 3) / (48 * E * SECTION.i1), CRITERIA.displacement);
  check.ratio("deflection in plane 2", middle[2], (fz * length ** 3) / (48 * E * SECTION.i2), CRITERIA.displacement);
  const forces = rowsOf(run, "Bar2", "forces")[0];
  check.small("moments of the bar at the solid", Math.hypot(forces.bendingMomentA1, forces.bendingMomentA2),
    (fy * length) / 4, CRITERIA.exact);
  console.log(`    info diagnostics: ${warningsOf(run).join(" | ").slice(0, 300)}`);
  check.done();
});
