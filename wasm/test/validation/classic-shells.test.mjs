// Validation of the shells of the ClassicFormulation: the ShellQuad4 of docs/shell-quad4-formulation.md and the
// ShellTria3 of docs/shell-tria3-formulation.md, through the whole path (model file, reader, modelFromReader, run).
// Each case is run with ShellQuad4 and with two ShellTria3 on the grid points of each quadrilateral (cut along the
// diagonal 1-3, 2-4, or alternating), against a closed form or a classical value. The criteria are the bounds each
// element meets, with a margin: section 6 of docs/shell-tria3-formulation.md gives the measured values, and each
// case writes its own in the output of the test.
//
// Run: node --test wasm/test/validation/classic-shells.test.mjs   (Node.js 24; needs ../O_o_Reader/build/wasm/dist)

import assert from "node:assert/strict";
import { CLASSIC, real, runModel, nodal, test } from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// Decks and meshes

// The meshes of a case: one ShellQuad4 per cell, or two ShellTria3.
const MESHES = [
  { name: "ShellQuad4", shape: "ShellQuad4" },
  { name: "ShellTria3 on the diagonal 1-3", shape: "ShellTria3", diagonal: "13" },
  { name: "ShellTria3 on the diagonal 2-4", shape: "ShellTria3", diagonal: "24" },
  { name: "ShellTria3 on alternating diagonals", shape: "ShellTria3", diagonal: "alternate" },
];
const isQuad = (mesh) => mesh.shape === "ShellQuad4";

// A structured mesh of nu × nv cells on the surface `position(a, b)`, a and b in [0, 1]. Returns the cards, `node(i,
// j)` the grid point of column i and row j, and the number of elements.
function grid({ nu, nv, position, shape, diagonal }) {
  const node = (i, j) => 1 + j * (nu + 1) + i;
  const lines = [];
  for (let j = 0; j <= nv; ++j)
    for (let i = 0; i <= nu; ++i) {
      const p = position(i / nu, j / nv);
      lines.push(`Node;${node(i, j)};;${real(p[0])};${real(p[1])};${real(p[2])}`);
    }
  let count = 0;
  const emit = (grids) => lines.push(`${shape};${++count};1;${grids.join(";")}`);
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < nu; ++i) {
      const q = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
      if (shape === "ShellQuad4") emit(q);
      else if (diagonal === "13" || (diagonal === "alternate" && (i + j) % 2 === 0)) {
        emit([q[0], q[1], q[2]]);
        emit([q[0], q[2], q[3]]);
      } else {
        emit([q[0], q[1], q[3]]);
        emit([q[1], q[2], q[3]]);
      }
    }
  return { lines, node, count: () => count, nu, nv };
}

// `SinglePointConstrList` records of set 1 holding `components` at the grid points `ids`.
const hold = (components, ids) => ids.map((id) => `SinglePointConstrList;1;${components};${id}`);

// The text of a static deck of one subcase (supports 1, loads 2).
function staticDeck(model) {
  return ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", ...model].join("\n") + "\n";
}

// The text of a modal deck (supports 1, `count` modes).
function modalDeck(model, count) {
  return ["[executive]", "SolutionSequence;103", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "EigenMethodSelection;3", "[model]", ...model, `LanczosEigenMethod;3;;;${count}`].join("\n") + "\n";
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
    // 0.03 % on these cells, longer than wide (10, 5, 2.5 and 1.25 by 1).
    for (const n of [1, 2, 4, 8])
      ratio(t, `${n} cells along`, strip({ mesh, n }), beamDeflection(STRIP.length), isQuad(mesh) ? 1e-8 : 1e-3);
  });

for (const mesh of MESHES)
  test(`a cantilever strip of ${mesh.name} without transverse shear flexibility bends as the beam`, (t) => {
    // The section without shear material takes a shear stiffness 1e4 times its bending one (c_K): the shear
    // deflection left is below 1e-4 of the bending one.
    for (const n of [1, 4])
      ratio(t, `${n} cells along`, strip({ mesh, n, shear: false }), beamDeflection(STRIP.length, false),
        isQuad(mesh) ? 2e-4 : 1e-3);
  });

for (const mesh of MESHES.filter((m) => !isQuad(m)))
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

// ---------------------------------------------------------------------------------------------------------------------
// 2. The same strip in torsion

for (const mesh of MESHES)
  test(`a strip of ${mesh.name} twists as the bar of thin rectangular section`, (t) => {
    // The rotation of the tip over T L / (G J), J = b t³ / 3. The bar of section b × t has J = b t³ / 3 (1 − 0.63
    // t / b): 1.067 here, less the restraint of the clamped end. Square cells, one, two and four across the width.
    const reference = STRIP.length / ((STRIP.young / 2) * STRIP.width * STRIP.thickness ** 3 / 3);
    // ShellQuad4: 1.006, 1.028, 1.036. Triangles: 1.130, 1.071, 1.043: softer on a strip one cell wide.
    const bounds = isQuad(mesh) ? [0.02, 0.04, 0.05] : [0.14, 0.08, 0.05];
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
    // formulation: where a part bends in its plane, quadrilaterals, or several triangles across its depth.
    const { length, width, thickness, young } = STRIP;
    const beam = length ** 3 / (3 * young * thickness * width ** 3 / 12) +
      1.2 * length / ((young / 2) * width * thickness);
    const cases = isQuad(mesh)
      ? [{ n: 2, across: 1, low: 0.92, high: 1.0 }, { n: 4, across: 1, low: 0.97, high: 1.0 },
        { n: 8, across: 1, low: 0.99, high: 1.0 }]
      : [{ n: 8, across: 1, low: 0.15, high: 0.25 }, { n: 40, across: 4, low: 0.80, high: 0.86 },
        { n: 80, across: 8, low: 0.93, high: 0.97 }];
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
        // the triangles from below (0.955 to 0.976, then 0.990 to 0.997).
        const bounds = isQuad(mesh) ? { 8: 0.07, 16: 0.03 } : { 8: 0.055, 16: 0.015 };
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
    // Membrane-dominated: the constant membrane strains of the triangles converge more slowly (0.86 to 0.88, then
    // 0.96 to 0.97) than the enhanced strains of the ShellQuad4 (1.014, 1.006).
    const bounds = isQuad(mesh) ? { 8: 0.02, 16: 0.01 } : { 8: 0.15, 16: 0.05 };
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
    // The rotation about the normal is held by the drilling term alone. ShellQuad4: 1.013 and 0.999; triangles:
    // 0.996 to 0.998, then 0.988 to 0.990.
    const bounds = { 8: 0.02, 16: 0.015 };
    for (const n of [8, 16]) ratio(t, `${n} x ${n} cells`, hemisphere({ mesh, n }), 0.0940, bounds[n]);
  });

// The twisted beam (MacNeal and Harder): length 12, width 1.1, twisted by 90 degrees from its clamped root to its tip,
// E = 29e6, ν = 0.22, t = 0.32, a unit force at its tip. The tip moves by 0.005424 along a force in the plane of the
// tip section and by 0.001754 along a force across it. Every quadrilateral is warped.
function twistedBeam({ mesh, nx, across, inPlane }) {
  const g = grid({ ...mesh, nu: nx, nv: across, position: (a, b) => {
    const [offset, angle] = [1.1 * (b - 0.5), (Math.PI / 2) * a];
    return [12 * a, offset * Math.cos(angle), offset * Math.sin(angle)];
  } });
  const model = [...g.lines, ...section({ thickness: 0.32, young: 29e6, poisson: 0.22 }),
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
    const bounds = isQuad(mesh) ? { inPlane: [0.01, 0.01], across: [0.015, 0.01] }
      : { inPlane: [0.02, 0.015], across: [0.17, 0.08] };
    meshes.forEach(({ nx, across }, k) => {
      ratio(t, `${nx} x ${across} cells, force in the plane of the tip`,
        twistedBeam({ mesh, nx, across, inPlane: true }), 0.005424, bounds.inPlane[k]);
      ratio(t, `${nx} x ${across} cells, force across the tip`, twistedBeam({ mesh, nx, across, inPlane: false }),
        0.001754, bounds.across[k]);
    });
  });

// ---------------------------------------------------------------------------------------------------------------------
// 6. Natural frequencies

for (const mesh of MESHES)
  test(`the first frequencies of a simply supported thin plate of ${mesh.name}`, (t) => {
    // ω = π² (m² + n²) / a² √(D / (ρ t)) for the thin plate; a / t = 100 lowers them by less than 1e-3. 16 × 16
    // cells, the coupled mass. ShellQuad4: 0.998, 1.005, 1.005, 1.001; triangles: 1.002 to 1.026.
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
