// Validation of the shells of the ClassicFormulation: the ShellQuad4 of docs/shell-quad4-formulation.md, the
// ShellTria3 of docs/shell-tria3-formulation.md and the ShellQuad8 and ShellTria6 of
// docs/shell-quad8-tria6-formulation.md, through the whole path (model file, reader, modelFromReader, run).
// Each case is run with one quadrilateral per cell (ShellQuad4, ShellQuad8) and with two triangles on the grid
// points of each cell (ShellTria3, ShellTria6: cut along the diagonal 1-3, 2-4, or alternating), against a closed
// form or a classical value. The criteria are the bounds each element meets, with a margin: the documents give the
// measured values, and each case writes its own in the output of the test.
//
// Run: node --test wasm/test/validation/classic-shells.test.mjs   (Node.js 24; needs ../O_o_Reader/build/wasm/dist)

import assert from "node:assert/strict";
import { CLASSIC, real, runModel, nodal, test } from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// Decks and meshes

// The meshes of a case: one quadrilateral per cell, or two triangles. `family` names the bounds of a case.
const QUAD4 = { name: "ShellQuad4", shape: "ShellQuad4", family: "quad4" };
const TRIA3_24 = { name: "ShellTria3 on the diagonal 2-4", shape: "ShellTria3", family: "tria3", diagonal: "24" };
const QUAD8 = { name: "ShellQuad8", shape: "ShellQuad8", family: "quad8" };
const TRIA6_13 = { name: "ShellTria6 on the diagonal 1-3", shape: "ShellTria6", family: "tria6", diagonal: "13" };
const MESHES = [
  QUAD4,
  { name: "ShellTria3 on the diagonal 1-3", shape: "ShellTria3", family: "tria3", diagonal: "13" },
  TRIA3_24,
  { name: "ShellTria3 on alternating diagonals", shape: "ShellTria3", family: "tria3", diagonal: "alternate" },
  QUAD8,
  TRIA6_13,
  { name: "ShellTria6 on alternating diagonals", shape: "ShellTria6", family: "tria6", diagonal: "alternate" },
];
const isQuad = (mesh) => mesh.family === "quad4";

// The grid points at the middle of the sides along the lines of the last mesh built, each with the two corner grid
// points of its side: `completed` gives them the supports their two corners share.
let middles = [];

// A structured mesh of nu × nv cells on the surface `position(a, b)`, a and b in [0, 1]. Returns the cards, `node(i,
// j)` the grid point of column i and row j of the cells, `points` every grid point with its parameters, and the
// number of elements. The grid points of the middles of the sides (ShellQuad8, ShellTria6) are on the surface.
function grid({ nu, nv, position, shape, diagonal }) {
  const m = shape === "ShellQuad8" || shape === "ShellTria6" ? 2 : 1;
  const ids = new Map();
  const lines = [];
  const points = [];
  // A ShellQuad8 has no grid point at the centre of its cell.
  const isUsed = (a, b) => shape !== "ShellQuad8" || a % 2 === 0 || b % 2 === 0;
  for (let b = 0; b <= m * nv; ++b)
    for (let a = 0; a <= m * nu; ++a) {
      if (!isUsed(a, b)) continue;
      const p = position(a / (m * nu), b / (m * nv));
      const id = ids.size + 1;
      ids.set(`${a},${b}`, id);
      points.push({ id, a: a / (m * nu), b: b / (m * nv) });
      lines.push(`Node;${id};;${real(p[0])};${real(p[1])};${real(p[2])}`);
    }
  const at = (a, b) => ids.get(`${a},${b}`);
  const node = (i, j) => at(m * i, m * j);
  middles = [];
  for (let b = 0; b <= m * nv; ++b)
    for (let a = 0; a <= m * nu; ++a) {
      if (m === 1 || !isUsed(a, b)) continue;
      if (a % 2 === 1 && b % 2 === 0) middles.push([at(a, b), at(a - 1, b), at(a + 1, b)]);
      if (a % 2 === 0 && b % 2 === 1) middles.push([at(a, b), at(a, b - 1), at(a, b + 1)]);
    }
  let count = 0;
  const emit = (grids) => lines.push(`${shape};${++count};1;${grids.join(";")}`);
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < nu; ++i) {
      const cell = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      const corner = (k) => node(cell[k][0], cell[k][1]);
      const middle = (k, l) => at(cell[k][0] + cell[l][0], cell[k][1] + cell[l][1]);
      const triangle = (k, l, n) => m === 1 ? [corner(k), corner(l), corner(n)]
        : [corner(k), corner(l), corner(n), middle(k, l), middle(l, n), middle(n, k)];
      if (shape === "ShellQuad4") emit([0, 1, 2, 3].map(corner));
      else if (shape === "ShellQuad8")
        emit([...[0, 1, 2, 3].map(corner), middle(0, 1), middle(1, 2), middle(2, 3), middle(3, 0)]);
      else if (diagonal === "13" || (diagonal === "alternate" && (i + j) % 2 === 0)) {
        emit(triangle(0, 1, 2));
        emit(triangle(0, 2, 3));
      } else {
        emit(triangle(0, 1, 3));
        emit(triangle(1, 2, 3));
      }
    }
  return { lines, node, points, count: () => count, nu, nv };
}

// The cards of a model of the last mesh built, with, for each grid point at the middle of a side along a line of the
// mesh, the supports its two corner grid points share (a case gives its supports at the corner grid points).
function completed(model) {
  const held = new Map();
  for (const line of model) {
    const fields = line.split(";");
    if (fields[0] !== "SinglePointConstrList") continue;
    for (const id of fields.slice(3)) held.set(Number(id), (held.get(Number(id)) ?? "") + fields[2]);
  }
  const added = [];
  for (const [id, first, last] of middles) {
    const [a, b] = [held.get(first) ?? "", held.get(last) ?? ""];
    const shared = [...new Set(a)].filter((component) => b.includes(component)).sort().join("");
    if (shared) added.push(`SinglePointConstrList;1;${shared};${id}`);
  }
  return [...model, ...added];
}

// `SinglePointConstrList` records of set 1 holding `components` at the grid points `ids`.
const hold = (components, ids) => ids.map((id) => `SinglePointConstrList;1;${components};${id}`);

// The text of a static deck of one subcase (supports 1, loads 2).
function staticDeck(model) {
  return ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", ...completed(model)].join("\n") + "\n";
}

// The text of a modal deck (supports 1, `count` modes).
function modalDeck(model, count) {
  return ["[executive]", "SolutionSequence;103", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "EigenMethodSelection;3", "[model]", ...completed(model), `LanczosEigenMethod;3;;;${count}`].join("\n") +
    "\n";
}

// An isotropic section: `shear` false leaves the shear material of the property blank (no transverse shear
// flexibility).
const section = ({ thickness, young, poisson, density = 0, shear = true }) => [
  `ShellProp;1;1;${real(thickness)};1${shear ? ";;1" : ""}`,
  `IsoMat;1;${real(young)};;${real(poisson)};${real(density)}`];

// Checks |value / expected − 1| ≤ tolerance and writes the ratio in the output of the test.
function ratio(t, what, value, expected, tolerance) {
  const measured = value / expected;
  t.diagnostic(`${what}: ratio ${measured.toFixed(6)} (criterion ${tolerance})`);
  assert.ok(Math.abs(measured - 1) <= tolerance, `${what}: ratio ${measured.toFixed(6)}, criterion ${tolerance}`);
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. A cantilever strip: the bending of a beam

const STRIP = { length: 10, width: 1, thickness: 0.1, young: 1e6 };

// The strip of n × `across` cells, clamped at X = 0. `load` "shear": a unit force along Z shared by the grid points
// of the tip; "inPlane": the same force along Y; "twist": a unit torque about X as two forces on the corners of the
// tip. Returns the mean deflection of the tip along the force, or the rotation of the tip about X from the
// deflections of its corners.
function strip({ mesh, n, across = 1, load = "shear", shear = true, length = STRIP.length }) {
  const g = grid({ ...mesh, nu: n, nv: across, position: (a, b) => [length * a, STRIP.width * b, 0] });
  const tip = Array.from({ length: across + 1 }, (_, j) => g.node(n, j));
  const model = [...g.lines, ...section({ ...STRIP, poisson: 0, shear }),
    ...hold("123456", Array.from({ length: across + 1 }, (_, j) => g.node(0, j)))];
  if (load === "shear" || load === "inPlane")
    tip.forEach((id, j) => model.push(`PointForceLoad;2;${id};;${real((j === 0 || j === across ? 0.5 : 1) / across)};${
      load === "shear" ? "0.;0.;1." : "0.;1.;0."}`));
  else
    model.push(`PointForceLoad;2;${tip[0]};;${real(1 / STRIP.width)};0.;0.;-1.`,
      `PointForceLoad;2;${tip[across]};;${real(1 / STRIP.width)};0.;0.;1.`);
  const run = runModel(staticDeck(model), CLASSIC);
  const deflections = tip.map((id) => nodal(run, "displacements", id)[load === "inPlane" ? 1 : 2]);
  return load === "twist" ? (deflections[across] - deflections[0]) / STRIP.width
    : deflections.reduce((sum, w) => sum + w, 0) / deflections.length;
}

// The tip deflection of the beam: P L³ / (3 E I) + P L / (κ G A), κ = 5/6 (ν = 0: G = E / 2), or without its shear
// term.
function beamDeflection(length, shear = true) {
  const { width, thickness, young } = STRIP;
  const bending = length ** 3 / (3 * young * width * thickness ** 3 / 12);
  return bending + (shear ? length / ((5 / 6) * (young / 2) * width * thickness) : 0);
}

for (const mesh of MESHES)
  test(`a cantilever strip of ${mesh.name} bends as the beam with any number of elements`, (t) => {
    // The ShellQuad4 gives the deflection of the beam exactly (document section 14.5). Two ShellTria3 give it within
    // 0.03 % on these cells, longer than wide (10, 5, 2.5 and 1.25 by 1). The ShellQuad8 gives it within 6e-5. The
    // deflection of a ShellTria6 is quadratic along the beam: 0.751, 0.942, 0.997 and 1.001 of the cubic of the beam
    // with 1, 2, 4 and 8 cells (three quarters with one cell, as one quadratic beam element).
    const bounds = { quad4: [1e-8, 1e-8, 1e-8, 1e-8], tria3: [1e-3, 1e-3, 1e-3, 1e-3], quad8: [1e-4, 1e-4, 1e-4, 1e-4],
      tria6: [0.26, 0.065, 0.006, 0.003] }[mesh.family];
    [1, 2, 4, 8].forEach((n, k) =>
      ratio(t, `${n} cells along`, strip({ mesh, n }), beamDeflection(STRIP.length), bounds[k]));
  });

for (const mesh of MESHES)
  test(`a cantilever strip of ${mesh.name} without transverse shear flexibility bends as the beam`, (t) => {
    // The section without shear material of a ShellQuad4 or a ShellTria3 takes a shear stiffness 1e4 times its
    // bending one (c_K): the shear deflection left is below 1e-4 of the bending one. That of a ShellQuad8 or a
    // ShellTria6 keeps the shear of its bending material (docs/shell-quad8-tria6-formulation.md, section 6), a
    // shear deflection of 7.5e-5 of the bending one here: 1.0001 for the ShellQuad8; 0.751 with one cell (above)
    // and 0.997 to 0.998 with four for the ShellTria6.
    const bounds = { quad4: [2e-4, 2e-4], tria3: [1e-3, 1e-3], quad8: [2e-4, 2e-4],
      tria6: [0.26, 0.006] }[mesh.family];
    [1, 4].forEach((n, k) => ratio(t, `${n} cells along`, strip({ mesh, n, shear: false }),
      beamDeflection(STRIP.length, false), bounds[k]));
  });

for (const mesh of MESHES.filter((m) => m.family === "tria3"))
  test(`a cantilever strip of ${mesh.name} wider than its cells are long stays within its bound`, (t) => {
    // The triangles of a cell wider than long are softer than the beam (section 6 of the ShellTria3 document): the
    // deflection is above the beam's by 0.4 % on one cell of 2 by 1, 3.1 % of 1 by 1 and 10.1 % of 0.5 by 1, and
    // by 0.03 %, 0.16 % and 0.44 % at most for four cells along.
    const cases = [{ cell: 2, n: 1, bound: 0.006 }, { cell: 1, n: 1, bound: 0.04 }, { cell: 0.5, n: 1, bound: 0.12 },
      { cell: 2, n: 4, bound: 0.001 }, { cell: 1, n: 4, bound: 0.003 }, { cell: 0.5, n: 4, bound: 0.008 }];
    for (const { cell, n, bound } of cases) {
      const length = cell * n * STRIP.width;
      ratio(t, `${n} cells of ${cell} by 1`, strip({ mesh, n, length }), beamDeflection(length), bound);
    }
  });

test("a cantilever of one ShellQuad8 wider than long is soft by the mode of its reduced rule", (t) => {
  // An element alone keeps one mode of its bending without energy (docs/shell-quad8-tria6-formulation.md, section
  // 4), which the clamped side holds the less the wider the element: 1.004, 1.062 and 1.90 of the deflection of the
  // beam for one cell of 2 by 1, 1 by 1 and 0.5 by 1; four cells along share no such mode: 1.000, 1.001, 1.017.
  const cases = [{ cell: 2, n: 1, low: 1.0, high: 1.01 }, { cell: 1, n: 1, low: 1.04, high: 1.08 },
    { cell: 0.5, n: 1, low: 1.8, high: 2.0 }, { cell: 2, n: 4, low: 0.999, high: 1.001 },
    { cell: 1, n: 4, low: 0.999, high: 1.003 }, { cell: 0.5, n: 4, low: 1.0, high: 1.03 }];
  for (const { cell, n, low, high } of cases) {
    const length = cell * n * STRIP.width;
    const measured = strip({ mesh: QUAD8, n, length }) / beamDeflection(length);
    t.diagnostic(`${n} cells of ${cell} by 1: ratio ${measured.toFixed(6)} (between ${low} and ${high})`);
    assert.ok(measured >= low && measured <= high, `${n} cells of ${cell} by 1: ratio ${measured.toFixed(6)}`);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
// 2. The same strip in torsion

for (const mesh of MESHES)
  test(`a strip of ${mesh.name} twists as the bar of thin rectangular section`, (t) => {
    // The rotation of the tip over T L / (G J), J = b t³ / 3. The bar of section b × t has J = b t³ / 3 (1 − 0.63
    // t / b): 1.067 here, less the restraint of the clamped end. Square cells, one, two and four across the width.
    const reference = STRIP.length / ((STRIP.young / 2) * STRIP.width * STRIP.thickness ** 3 / 3);
    // ShellQuad4: 1.006, 1.028, 1.036. ShellTria3: 1.130, 1.071, 1.043: softer on a strip one cell wide.
    // ShellQuad8: 0.996, 1.025, 1.050. ShellTria6: 0.988 to 0.991, 1.009 to 1.011, 1.030 to 1.032.
    const bounds = { quad4: [0.02, 0.04, 0.05], tria3: [0.14, 0.08, 0.05], quad8: [0.02, 0.04, 0.06],
      tria6: [0.02, 0.03, 0.05] }[mesh.family];
    [1, 2, 4].forEach((across, k) => {
      const rotation = strip({ mesh, n: 10 * across, across, load: "twist" });
      ratio(t, `${10 * across} x ${across} cells`, rotation, reference, bounds[k]);
    });
  });

// ---------------------------------------------------------------------------------------------------------------------
// 3. The same strip bent in its plane

for (const mesh of MESHES)
  test(`a strip of ${mesh.name} bent in its plane`, (t) => {
    // The tip deflection over that of the beam, P L³ / (3 E I) + 6 P L / (5 G A), I = t b³ / 12. The enhanced
    // strains of the ShellQuad4 bend with one element across the depth (0.936, 0.983, 0.995 for 2, 4 and 8 cells
    // along). The membrane strains of a ShellTria3 are constant: one cell across the depth is far too stiff (0.19 to
    // 0.21 for 8 cells along), and the deflection comes with the number of cells across (0.82 to 0.84 with 4,
    // 0.95 to 0.96 with 8, on square cells). This is the limit of the three-node triangle, whatever its
    // formulation: where a part bends in its plane, quadrilaterals, or several triangles across its depth. The
    // quadratic membrane of the ShellQuad8 and of the ShellTria6 bends with one element across the depth (0.989 to
    // 0.999 with 4 cells along, 0.997 to 0.999 with 8).
    const { length, width, thickness, young } = STRIP;
    const beam = length ** 3 / (3 * young * thickness * width ** 3 / 12) +
      1.2 * length / ((young / 2) * width * thickness);
    const cases = {
      quad4: [{ n: 2, across: 1, low: 0.92, high: 1.0 }, { n: 4, across: 1, low: 0.97, high: 1.0 },
        { n: 8, across: 1, low: 0.99, high: 1.0 }],
      tria3: [{ n: 8, across: 1, low: 0.15, high: 0.25 }, { n: 40, across: 4, low: 0.80, high: 0.86 },
        { n: 80, across: 8, low: 0.93, high: 0.97 }],
      quad8: [{ n: 4, across: 1, low: 0.98, high: 1.005 }, { n: 8, across: 1, low: 0.99, high: 1.005 }],
      tria6: [{ n: 4, across: 1, low: 0.98, high: 1.005 }, { n: 8, across: 1, low: 0.99, high: 1.005 }],
    }[mesh.family];
    for (const { n, across, low, high } of cases) {
      const measured = strip({ mesh, n, across, load: "inPlane" }) / beam;
      t.diagnostic(`${n} x ${across} cells: ratio ${measured.toFixed(6)} (between ${low} and ${high})`);
      assert.ok(measured >= low && measured <= high, `${n} x ${across} cells: ratio ${measured.toFixed(6)}`);
    }
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. A square plate under a pressure

const PLATE = { side: 1, young: 1e7, poisson: 0.3, thickness: 0.01, pressure: 1 };

// The plate of n × n cells, simply supported (hard: the deflection and the rotation of the normal along the edge
// held) or clamped, under a uniform pressure. Returns the deflection of its centre over that of the thin plate,
// α q a⁴ / D with α = 0.004062 (simply supported) or 0.001265 (clamped) (Timoshenko and Woinowsky-Krieger, Theory of
// plates and shells, 1959, tables 8 and 35).
function plateDeflection({ mesh, n, support }) {
  const g = grid({ ...mesh, nu: n, nv: n, position: (a, b) => [PLATE.side * a, PLATE.side * b, 0] });
  const model = [...g.lines, ...section(PLATE)];
  for (let j = 0; j <= n; ++j)
    for (let i = 0; i <= n; ++i) {
      const [alongY, alongX] = [i === 0 || i === n, j === 0 || j === n];
      if (!alongX && !alongY) continue;
      model.push(...hold(support === "clamped" ? "123456" : `123${alongY ? "4" : ""}${alongX ? "5" : ""}`,
        [g.node(i, j)]));
    }
  model.push(`ShellPressureLoad;2;${real(PLATE.pressure)};1;Through;${g.count()}`);
  const run = runModel(staticDeck(model), CLASSIC);
  const rigidity = PLATE.young * PLATE.thickness ** 3 / (12 * (1 - PLATE.poisson ** 2));
  const coefficient = support === "clamped" ? 0.001265 : 0.004062;
  return nodal(run, "displacements", g.node(n / 2, n / 2))[2] /
    (coefficient * PLATE.pressure * PLATE.side ** 4 / rigidity);
}

for (const support of ["simple", "clamped"])
  for (const mesh of MESHES)
    test(`a ${support === "simple" ? "simply supported" : "clamped"} thin plate of ${mesh.name} under a pressure`,
      (t) => {
        // a / t = 100. The ShellQuad4 comes from above (1.017 and 1.006 simply supported, 1.059 and 1.021 clamped),
        // the ShellTria3 from below (0.955 to 0.976, then 0.990 to 0.997). ShellQuad8: 1.0005 and 1.0006, 0.995 and
        // 1.002. ShellTria6, from above: 1.007 to 1.011 and 1.003, 1.016 to 1.029 and 1.011 to 1.012 (0.846 to 0.901
        // and 0.985 to 0.991 clamped with the strains of its displacements).
        const bounds = { quad4: { 8: 0.07, 16: 0.03 }, tria3: { 8: 0.055, 16: 0.015 }, quad8: { 8: 0.01, 16: 0.005 },
          tria6: { 8: 0.04, 16: 0.015 } }[mesh.family];
        for (const n of [8, 16]) ratio(t, `${n} x ${n} cells`, plateDeflection({ mesh, n, support }), 1, bounds[n]);
      });

// ---------------------------------------------------------------------------------------------------------------------
// 5. Curved shells

// The barrel vault under its own weight (A. C. Scordelis, K. S. Lo, Computer analysis of cylindrical shells, J. Am.
// Concrete Inst. 61 (1964)): radius 25, length 50, 40° on each side of the crown, t = 0.25, E = 4.32e8, ν = 0, a
// weight of 90 per unit area, rigid diaphragms at the ends. A quarter of the vault; the vertical deflection of the
// middle of the free edge is 0.3024 (the value the finite element literature converges to).
function vault({ mesh, n }) {
  const [radius, length, angle] = [25, 50, (40 * Math.PI) / 180];
  const g = grid({ ...mesh, nu: n, nv: n,
    position: (a, b) => [radius * Math.sin(angle * a), (length / 2) * b, radius * Math.cos(angle * a)] });
  const model = [...g.lines, ...section({ thickness: 0.25, young: 4.32e8, poisson: 0, density: 360 }),
    "GravityLoad;2;;1.;0.;0.;-1."];
  for (let k = 0; k <= n; ++k) {
    model.push(...hold("13", [g.node(k, 0)]));   // the diaphragm at Y = 0
    model.push(...hold("246", [g.node(k, n)]));  // the plane of symmetry at the middle of the length
    model.push(...hold("156", [g.node(0, k)]));  // the plane of symmetry of the crown
  }
  const run = runModel(staticDeck(model), CLASSIC);
  return -nodal(run, "displacements", g.node(n, n))[2];
}

for (const mesh of MESHES)
  test(`the barrel vault under its weight, of ${mesh.name}`, (t) => {
    // Membrane-dominated: the constant membrane strains of the ShellTria3 converge more slowly (0.86 to 0.88, then
    // 0.96 to 0.97) than the enhanced strains of the ShellQuad4 (1.014, 1.006). ShellQuad8: 1.001, 0.999.
    // ShellTria6: 1.000 to 1.003, 0.997 to 0.998 (0.977 to 0.987 at 8 x 8 with the strains of its displacements).
    const bounds = { quad4: { 8: 0.02, 16: 0.01 }, tria3: { 8: 0.15, 16: 0.05 }, quad8: { 8: 0.005, 16: 0.005 },
      tria6: { 8: 0.01, 16: 0.006 } }[mesh.family];
    for (const n of [8, 16]) ratio(t, `${n} x ${n} cells`, vault({ mesh, n }), 0.3024, bounds[n]);
  });

// The pinched hemisphere with a hole of 18° (R. H. MacNeal, R. L. Harder, A proposed standard set of problems to test
// finite element accuracy, Finite Elem. Anal. Des. 1 (1985)): radius 10, t = 0.04, E = 6.825e7, ν = 0.3, two pairs
// of opposite unit forces on the equator. A quarter; the displacement under a force is 0.0940.
function hemisphere({ mesh, n }) {
  const [radius, top] = [10, (72 * Math.PI) / 180];
  const g = grid({ ...mesh, nu: n, nv: n, position: (a, b) => {
    const [longitude, latitude] = [(Math.PI / 2) * a, top * b];
    return [radius * Math.cos(latitude) * Math.cos(longitude), radius * Math.cos(latitude) * Math.sin(longitude),
      radius * Math.sin(latitude)];
  } });
  const model = [...g.lines, ...section({ thickness: 0.04, young: 6.825e7, poisson: 0.3 })];
  for (let j = 0; j <= n; ++j) {
    model.push(...hold("246", [g.node(0, j)]));  // the plane Y = 0
    model.push(...hold("156", [g.node(n, j)]));  // the plane X = 0
  }
  model.push(...hold("3", [g.node(n / 2, 0)]), `PointForceLoad;2;${g.node(0, 0)};;1.;1.;0.;0.`,
    `PointForceLoad;2;${g.node(n, 0)};;1.;0.;-1.;0.`);
  const run = runModel(staticDeck(model), CLASSIC);
  return nodal(run, "displacements", g.node(0, 0))[0];
}

for (const mesh of MESHES)
  test(`the pinched hemisphere of ${mesh.name}`, (t) => {
    // The rotation about the normal is held by the drilling term alone. ShellQuad4: 1.013 and 0.999; ShellTria3:
    // 0.996 to 0.998, then 0.988 to 0.990. ShellQuad8: 0.992 and 0.997. ShellTria6: 0.933 to 0.952, then 0.988 to
    // 0.991 (0.38 to 0.44 and 0.88 to 0.91 with the strains of its displacements, which lock).
    const bounds = mesh.family === "tria6" ? { 8: 0.08, 16: 0.02 } : { 8: 0.02, 16: 0.015 };
    for (const n of [8, 16]) ratio(t, `${n} x ${n} cells`, hemisphere({ mesh, n }), 0.0940, bounds[n]);
  });

// The twisted beam (MacNeal and Harder): length 12, width 1.1, twisted by 90 degrees from its clamped root to its tip,
// E = 29e6, ν = 0.22, t = 0.32, a unit force at its tip. The tip moves by 0.005424 along a force in the plane of the
// tip section and by 0.001754 along a force across it. Every quadrilateral is warped.
function twistedBeam({ mesh, nx, across, inPlane, parameters = [] }) {
  const g = grid({ ...mesh, nu: nx, nv: across, position: (a, b) => {
    const [offset, angle] = [1.1 * (b - 0.5), (Math.PI / 2) * a];
    return [12 * a, offset * Math.cos(angle), offset * Math.sin(angle)];
  } });
  const model = [...g.lines, ...section({ thickness: 0.32, young: 29e6, poisson: 0.22 }), ...parameters,
    ...hold("123456", Array.from({ length: across + 1 }, (_, j) => g.node(0, j)))];
  // The tip section lies along Z: a force in its plane is along Z.
  for (let j = 0; j <= across; ++j)
    model.push(`PointForceLoad;2;${g.node(nx, j)};;${real((j === 0 || j === across ? 0.5 : 1) / across)};${
      inPlane ? "0.;0.;1." : "0.;1.;0."}`);
  const run = runModel(staticDeck(model), CLASSIC);
  return nodal(run, "displacements", g.node(nx, across / 2))[inPlane ? 2 : 1];
}

for (const mesh of MESHES)
  test(`the twisted beam of ${mesh.name}`, (t) => {
    // ShellQuad4: 0.996 and 0.987, then 0.998 and 0.994. Triangles: 0.990 to 0.999 and 0.842 to 0.860, then 0.993
    // to 0.996 and 0.925 to 0.936: the force across the tip bends the beam in the plane of its elements at the root,
    // where the triangles need more cells across the width (section 6 of the ShellTria3 document, item 4). Without
    // the bending term of their drilling stiffness (c_n) the triangles soften with the refinement of this mesh.
    const meshes = [{ nx: 12, across: 2 }, { nx: 24, across: 4 }];
    // ShellQuad8 and ShellTria6: 0.998 to 0.999 on both meshes.
    const bounds = { quad4: { inPlane: [0.01, 0.01], across: [0.015, 0.01] },
      tria3: { inPlane: [0.025, 0.015], across: [0.18, 0.09] },
      quad8: { inPlane: [0.005, 0.005], across: [0.005, 0.005] },
      tria6: { inPlane: [0.005, 0.005], across: [0.005, 0.005] } }[mesh.family];
    meshes.forEach(({ nx, across }, k) => {
      ratio(t, `${nx} x ${across} cells, force in the plane of the tip`,
        twistedBeam({ mesh, nx, across, inPlane: true }), 0.005424, bounds.inPlane[k]);
      ratio(t, `${nx} x ${across} cells, force across the tip`, twistedBeam({ mesh, nx, across, inPlane: false }),
        0.001754, bounds.across[k]);
    });
  });

// The normals of the grid points and the drilling factor (ShellDirectors.hpp), on the twisted beam of
// triangles under the force in the plane of its tip.
test("the twisted beam of ShellTria3: the normals shared at the grid points and the drilling factor", (t) => {
  const mesh = TRIA3_24;
  const displacement = (nx, across, parameters) =>
    twistedBeam({ mesh, nx, across, inPlane: true, parameters }) / 0.005424;
  // With the normal of each element (PARAM ShellNormalAngle -1) the rotation about the normal of a triangle is a
  // bending rotation for its neighbours, which hold it by their bending alone: the beam softens as its mesh is
  // refined. With the shared normals (the default) it converges.
  const own = [displacement(12, 2, ["Parameter;ShellNormalAngle;-1."]),
    displacement(24, 4, ["Parameter;ShellNormalAngle;-1."])];
  const shared = [displacement(12, 2, []), displacement(24, 4, [])];
  t.diagnostic(`each element's own normal: ${own.map((v) => v.toFixed(4)).join(", ")}; shared normals: ${
    shared.map((v) => v.toFixed(4)).join(", ")}`);
  assert.ok(own[0] > 1.1 && own[1] > own[0] + 0.1, `own normals soften with the refinement: ${own}`);
  assert.ok(Math.abs(shared[0] - 1) < 0.025 && Math.abs(shared[1] - 1) < 0.015, `shared normals converge: ${shared}`);
  // With shared normals the drilling factor is a stabilisation: 0 (no term), the default (100) and 400 give the same
  // displacement within 1e-4.
  for (const stiffness of ["0.", "400."]) {
    const value = displacement(24, 4, [`Parameter;ShellDrillingStiffness;${stiffness}`]);
    t.diagnostic(`ShellDrillingStiffness ${stiffness}: ${value.toFixed(6)} (default ${shared[1].toFixed(6)})`);
    assert.ok(Math.abs(value / shared[1] - 1) < 1e-4, `ShellDrillingStiffness ${stiffness}: ${value}`);
  }
});

test("the twisted beam of ShellTria3: normals given at the grid points by ShellNormal cards", (t) => {
  const mesh = TRIA3_24;
  const [nx, across] = [24, 4];
  // The normal of the twisted surface at the grid point of column i and row j: (s k, sin φ, −cos φ) made unit, s the
  // distance from the axis, φ the twist at that station and k its rate.
  const cards = (scale) => {
    const lines = [];
    for (let j = 0; j <= across; ++j)
      for (let i = 0; i <= nx; ++i) {
        const [offset, angle, rate] = [1.1 * (j / across - 0.5), (Math.PI / 2) * (i / nx), Math.PI / 24];
        const n = [offset * rate, Math.sin(angle), -Math.cos(angle)];
        const length = Math.hypot(...n);
        lines.push(`ShellNormal;${1 + j * (nx + 1) + i};;${n.map((c) => real((scale * c) / length)).join(";")}`);
      }
    return lines;
  };
  const displacement = (parameters) => twistedBeam({ mesh, nx, across, inPlane: true, parameters }) / 0.005424;
  const own = displacement(["Parameter;ShellNormalAngle;-1."]);
  // The normals of the surface given at every grid point, without the shared normals: the beam converges as with
  // them. Zero vectors with the shared normals: every element keeps its own normal.
  const given = displacement(["Parameter;ShellNormalAngle;-1.", ...cards(1)]);
  const none = displacement(cards(0));
  t.diagnostic(`own normals ${own.toFixed(4)}, normals given ${given.toFixed(4)}, zero vectors ${none.toFixed(4)}`);
  assert.ok(Math.abs(given - 1) < 0.015, `normals given: ${given}`);
  assert.ok(Math.abs(none / own - 1) < 1e-9, `zero vectors leave the elements their own normals: ${none}, ${own}`);
});

test("a mass with an inertia on a grid point that shells alone hold is pointed out in a modal analysis", () => {
  // A cantilever plate with a PointMassInertia at the middle of its free end: the rotation of that grid point about
  // the normal of the plate has the drilling stabilisation of the shells for only stiffness.
  const [nx, ny] = [8, 4];
  const g = grid({ ...QUAD4, nu: nx, nv: ny, position: (a, b) => [a, 0.5 * b, 0] });
  const model = (extra) => [...g.lines, ...section({ thickness: 0.01, young: 7e10, poisson: 0.3, density: 2700 }),
    `PointMassInertia;900;${g.node(nx, ny / 2)};;5.;0.;0.;0.05;0.02;;0.02;;;0.03`,
    ...hold("123456", Array.from({ length: ny + 1 }, (_, j) => g.node(0, j))), ...extra];
  const notices = (extra) => runModel(modalDeck(model(extra), 4), CLASSIC, "modes").diagnostics
    .filter((d) => d.code === "inertia-on-shell-normal");
  const told = notices([]);
  assert.equal(told.length, 1, "one notice");
  assert.match(told[0].message ?? told[0].text, /900/);
  // Without the drilling term the automatic constraint removes the rotation, and the inertia about it with it; a
  // bar on the grid point holds the rotation.
  assert.equal(notices(["Parameter;ShellDrillingStiffness;0."]).length, 0, "no notice without the drilling term");
  assert.equal(notices([`Bar2;901;7;${g.node(nx, ny / 2)};${g.node(nx - 1, ny / 2)};0.;;0.;1.`,
    "BarProp;7;1;1.e-4;1.e-9;1.e-9;1.e-9"]).length, 0, "no notice with a bar on the grid point");
});

// The ShellQuad8 and the ShellTria6 take the normals of their grid points and their drilling factor as the ShellQuad4
// and the ShellTria3 do (ShellDirectors.hpp), on the pinched hemisphere.
for (const mesh of [QUAD8, TRIA6_13])
  test(`the pinched hemisphere of ${mesh.name}: the normals of the grid points and the drilling factor`, (t) => {
    const n = 8;
    const [radius, top] = [10, (72 * Math.PI) / 180];
    const direction = (a, b) => {
      const [longitude, latitude] = [(Math.PI / 2) * a, top * b];
      return [Math.cos(latitude) * Math.cos(longitude), Math.cos(latitude) * Math.sin(longitude), Math.sin(latitude)];
    };
    const displacement = (extra) => {
      const g = grid({ ...mesh, nu: n, nv: n, position: (a, b) => direction(a, b).map((c) => radius * c) });
      const model = [...g.lines, ...section({ thickness: 0.04, young: 6.825e7, poisson: 0.3 }), ...extra(g)];
      for (let j = 0; j <= n; ++j)
        model.push(...hold("246", [g.node(0, j)]), ...hold("156", [g.node(n, j)]));
      model.push(...hold("3", [g.node(n / 2, 0)]), `PointForceLoad;2;${g.node(0, 0)};;1.;1.;0.;0.`,
        `PointForceLoad;2;${g.node(n, 0)};;1.;0.;-1.;0.`);
      return nodal(runModel(staticDeck(model), CLASSIC), "displacements", g.node(0, 0))[0] / 0.0940;
    };
    // The normals of the sphere given at every grid point, and zero vectors (every element keeps its own normal).
    const cards = (scale) => (g) => g.points.map(({ id, a, b }) =>
      `ShellNormal;${id};;${direction(a, b).map((c) => real(scale * c)).join(";")}`);
    const shared = displacement(() => []);
    const own = displacement(() => ["Parameter;ShellNormalAngle;-1."]);
    const given = displacement((g) => ["Parameter;ShellNormalAngle;-1.", ...cards(1)(g)]);
    const none = displacement(cards(0));
    t.diagnostic(`shared normals ${shared.toFixed(6)}, own normals ${own.toFixed(6)}, normals given ${
      given.toFixed(6)}, zero vectors ${none.toFixed(6)}`);
    // The own normals of two quadratic elements at a grid point differ by the interpolation error of their
    // surfaces, small here: the three sets of normals give the displacement within 1 %.
    assert.ok(Math.abs(own / shared - 1) < 0.01, `own normals: ${own}, shared: ${shared}`);
    assert.ok(Math.abs(given / shared - 1) < 0.01, `normals given: ${given}, shared: ${shared}`);
    assert.ok(Math.abs(none / own - 1) < 1e-9, `zero vectors leave the elements their own normals: ${none}, ${own}`);
    assert.ok(given !== own, "the normals given are read");
    // The drilling factor is a stabilisation: 0 (no term: the automatic constraint removes the rotation about the
    // normal), the default (100) and 400 give the same displacement within 1e-4.
    for (const stiffness of ["0.", "400."]) {
      const value = displacement(() => [`Parameter;ShellDrillingStiffness;${stiffness}`]);
      t.diagnostic(`ShellDrillingStiffness ${stiffness}: ${value.toFixed(6)} (default ${shared.toFixed(6)})`);
      assert.ok(Math.abs(value / shared - 1) < 1e-4, `ShellDrillingStiffness ${stiffness}: ${value}`);
    }
  });

// A free quarter of a cylinder under a uniform temperature rise expands by a homothety, without stress: flat
// quadrilaterals, quadrilaterals warped by moving their grid points on the cylinder, triangles, and the curved
// ShellQuad8 and ShellTria6 (2e-5 to 9e-5 of α ΔT off the homothety before their dilatation was taken apart).
for (const mesh of MESHES)
  for (const warped of isQuad(mesh) ? [false, true] : [false])
    test(`a free quarter cylinder of ${warped ? "warped " : ""}${mesh.name} expands freely under a temperature ` +
      "rise", (t) => {
      const [n, young, expansion, rise] = [8, 7e10, 1.2e-5, 80];
      // A deterministic move of the inner grid points by up to a fifth of a cell, on the cylinder.
      const moved = (a, b, salt) => {
        const inner = a > 0 && a < 1 && b > 0 && b < 1;
        const s = Math.sin(12.9898 * a * n + 78.233 * b * n + 37.719 * salt) * 43758.5453;
        return inner && warped ? (0.2 / n) * (2 * (s - Math.floor(s)) - 1) : 0;
      };
      const points = new Map(); // by grid point: the mesh numbers its grid points in the order it asks for them
      const g = grid({ ...mesh, nu: n, nv: n, position: (a, b) => {
        const [x, angle] = [a + moved(a, b, 1), (Math.PI / 2) * (b + moved(a, b, 2))];
        const p = [x, Math.sin(angle), Math.cos(angle)];
        points.set(points.size + 1, p);
        return p;
      } });
      const model = [...g.lines, "ShellProp;1;1;0.01;1;;1", `IsoMat;1;${real(young)};;0.3;0.;${real(expansion)};0.`,
        ...hold("123", [g.node(0, 0)]), ...hold("23", [g.node(n, 0)]),
        `SinglePointConstr;1;${g.node(0, n)};3;${real(-expansion * rise)}`, `DefaultTemperatureLoad;7;${real(rise)}`];
      const text = ["[executive]", "SolutionSequence;101", "[case]", "TemperatureSelection;7;TemperatureLoadPart",
        "SubcaseStart;1", "SinglePointConstrSelection;1", "DisplacementOutput;All", "[model]",
        ...completed(model)].join("\n") +
        "\n";
      const run = runModel(text, CLASSIC);
      // The grid point (0, 0, 1) is held: u = α ΔT (x, y, z − 1).
      let worst = 0;
      for (const [id, p] of points) {
        const u = nodal(run, "displacements", id);
        const expected = [p[0], p[1], p[2] - 1].map((c) => expansion * rise * c);
        worst = Math.max(worst, ...expected.map((value, k) => Math.abs(u[k] - value)));
      }
      const error = worst / (expansion * rise);
      t.diagnostic(`displacements off the homothety by ${error.toExponential(2)} of α ΔT (criterion 1e-8)`);
      assert.ok(error <= 1e-8, `displacements off the homothety by ${error.toExponential(2)} of α ΔT`);
    });

// ---------------------------------------------------------------------------------------------------------------------
// 6. Natural frequencies

for (const mesh of MESHES)
  test(`the first frequencies of a simply supported thin plate of ${mesh.name}`, (t) => {
    // ω = π² (m² + n²) / a² √(D / (ρ t)) for the thin plate; a / t = 100 lowers them by less than 1e-3. 16 × 16
    // cells, the coupled mass. ShellQuad4: 0.998, 1.005, 1.005, 1.001; ShellTria3: 1.002 to 1.026; ShellQuad8:
    // 0.994 to 0.996; ShellTria6: 0.992 to 0.997.
    const n = 16;
    const density = 1;
    const g = grid({ ...mesh, nu: n, nv: n, position: (a, b) => [PLATE.side * a, PLATE.side * b, 0] });
    const model = [...g.lines, ...section({ ...PLATE, density }), "Parameter;CoupledMass;1"];
    for (let j = 0; j <= n; ++j)
      for (let i = 0; i <= n; ++i) {
        const onEdge = i === 0 || i === n || j === 0 || j === n;
        model.push(...hold(onEdge ? "1236" : "126", [g.node(i, j)]));
      }
    const run = runModel(modalDeck(model, 4), CLASSIC, "modes");
    const rows = run.eigenvalues[0].rows;
    const rigidity = PLATE.young * PLATE.thickness ** 3 / (12 * (1 - PLATE.poisson ** 2));
    const unit = (Math.PI / PLATE.side) ** 2 * Math.sqrt(rigidity / (density * PLATE.thickness));
    [2, 5, 5, 8].forEach((sum, k) => ratio(t, `frequency ${k + 1}`, Math.sqrt(rows[k].eigenvalue), sum * unit,
      sum === 2 ? 0.01 : 0.03));
  });
