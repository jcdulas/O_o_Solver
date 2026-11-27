// Validation of the forces read as tractions along the outline of the shells of the mixed hybrid formulation, and
// of the warning for the edge layer of a soft support (docs/mixed-hybrid-formulation-spec.md, sections 3.6, 5.6 and
// 7). Each case goes through the whole path (model file, reader, modelFromReader, run).
//
// Run: node --test wasm/test/validation/edge-forces.test.mjs   (Node.js 24; needs ../O_o_Model/build/wasm/dist)

import assert from "node:assert/strict";
import { CRITERIA, MIXED, CLASSIC, real, runModel, nodal, innerEdges, prepared, surfaceRecords, assertRatio, test }
  from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// The plate: aluminium, 10 mm thick (SI units).

const [YOUNG, POISSON, DENSITY, THICKNESS] = [7e10, 0.3, 2700, 0.01];
const SHEAR_FACTOR = 5 / 6;
const BENDING = YOUNG * THICKNESS ** 3 / (12 * (1 - POISSON ** 2));
const SHEAR_MODULUS = YOUNG / (2 * (1 + POISSON));

// The grid points and the quadrilaterals of a rectangle a × b cut along the lines `xs` and `ys`: id(i, j) the grid
// point of column i and row j.
function rectangle(xs, ys) {
  const id = (i, j) => j * xs.length + i + 1;
  const [nodes, elements] = [[], []];
  ys.forEach((y, j) => xs.forEach((x, i) => nodes.push({ id: id(i, j), p: [x, y, 0], i, j })));
  for (let j = 0; j + 1 < ys.length; ++j)
    for (let i = 0; i + 1 < xs.length; ++i) elements.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
  return { xs, ys, id, nodes, elements, columns: xs.length - 1, rows: ys.length - 1 };
}

// `count` equal cells over `length`, with a row of width `strip` cut out of the first and the last when given.
function lines(length, count, strip = 0) {
  const out = Array.from({ length: count + 1 }, (_, k) => (k * length) / count);
  if (strip > 0) out.push(strip, length - strip);
  return out.sort((a, b) => a - b);
}

// Half the lengths of the cells on both sides of line k: what a uniform traction puts at its grid point.
function tributary(positions, k) {
  const before = k > 0 ? positions[k] - positions[k - 1] : 0;
  const after = k + 1 < positions.length ? positions[k + 1] - positions[k] : 0;
  return (before + after) / 2;
}

// The model file of `mesh` (its nodes, with p, and its quadrilaterals) with the case and model records given.
function deck({ mesh, solution, caseRecords, modelRecords, shear = true, geometry = {} }) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseRecords, "[model]",
    ...mesh.nodes.map(({ id, p }) => `Node;${id};;${real(p[0])};${real(p[1])};${real(p[2])}`),
    ...mesh.elements.map((corners, k) => `ShellQuad4;${k + 1};1;${corners.join(";")}`),
    `ShellProp;1;1;${THICKNESS};1${shear ? ";;1" : ""}`, `IsoMat;1;${real(YOUNG)};;${POISSON};${real(DENSITY)}`,
    ...modelRecords, ...prepared(surfaceRecords(geometry)), ""].join("\n");
}

// The supports of a simply supported plate: the deflection on the four sides, with the rotation about the normal of
// each side when `hard`; and the in-plane rigid motions, held at the middles of the sides.
function simpleSupports(mesh, hard) {
  const [m, n, id] = [mesh.columns, mesh.rows, mesh.id];
  const records = [];
  for (let j = 0; j <= n; ++j) records.push(`SinglePointConstrList;1;${hard ? "34" : "3"};${id(0, j)};${id(m, j)}`);
  for (let i = 0; i <= m; ++i) records.push(`SinglePointConstrList;1;${hard ? "35" : "3"};${id(i, 0)};${id(i, n)}`);
  records.push(`SinglePointConstrList;1;1;${id(m / 2, 0)};${id(m / 2, n)}`,
    `SinglePointConstrList;1;2;${id(0, n / 2)};${id(m, n / 2)}`);
  return records;
}

const firstEigenvalue = (run, analysis) => run.eigenvalues.find((t) => t.analysis === analysis).rows[0];
const codesOf = (run) => run.diagnostics.map((d) => d.code);

// ---------------------------------------------------------------------------------------------------------------------
// Buckling under a compression given as forces at the grid points of two sides

// A square plate compressed along x by `load` per length, in buckling, with hard or soft simple supports. `loading`:
// "forces" (the consistent forces of the traction at the grid points of the sides x = 0 and x = a) or "displacement"
// (the two sides moved in by the shortening of that traction).
function compressedPlate({ count, strip = 0, hard = true, loading }) {
  const [side, load] = [1, 1000];
  const mesh = rectangle(lines(side, count, strip), lines(side, count, strip));
  const [m, n, id] = [mesh.columns, mesh.rows, mesh.id];
  const records = simpleSupports(mesh, hard);
  for (let j = 0; j <= n; ++j) {
    if (loading === "forces") {
      const force = real(load * tributary(mesh.ys, j));
      records.push(`PointForceLoad;4;${id(0, j)};;${force};1.;0.;0.`, `PointForceLoad;4;${id(m, j)};;${force};-1.;0.;0.`);
    } else {
      const shortening = (load * side) / (2 * YOUNG * THICKNESS); // each side: N = E t (2 d / a)
      records.push(`SinglePointConstr;1;${id(0, j)};1;${shortening};${id(m, j)};1;${-shortening}`);
    }
  }
  records.push("LanczosEigenMethod;3;;;1");
  const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1",
    ...(loading === "forces" ? ["StaticLoadSelection;4"] : []), "SubcaseStart;2", "SinglePointConstrSelection;1",
    "EigenMethodSelection;3", "StaticSubcaseSelection;1", "DisplacementOutput;All"];
  const run = runModel(deck({ mesh, solution: 105, caseRecords, modelRecords: records }), MIXED, "buckling");
  return { run, criticalLoad: firstEigenvalue(run, "buckling").eigenvalue * load };
}

// The critical load of the hard simply supported square plate with transverse shear, one half-wave each way: the
// thin-plate value 4 π² D / a² over 1 + D k² / (κ G t), k² = 2 π² / a² (the Navier solution of the
// Reissner-Mindlin plate; Reddy, Theory and Analysis of Elastic Plates and Shells, 2nd ed., 2007).
const CRITICAL_LOAD = (4 * Math.PI ** 2 * BENDING) /
  (1 + (BENDING * 2 * Math.PI ** 2) / (SHEAR_FACTOR * SHEAR_MODULUS * THICKNESS));

for (const count of [2, 4])
  test(`buckling of a plate compressed by forces at the grid points of two sides, ${count} x ${count} ShellQuad4`,
    (t) => {
      const forces = compressedPlate({ count, loading: "forces" });
      const displacement = compressedPlate({ count, loading: "displacement" });
      t.diagnostic(`ratio to the closed form ${(forces.criticalLoad / CRITICAL_LOAD).toFixed(6)}, to the moved sides ` +
        `${(forces.criticalLoad / displacement.criticalLoad).toFixed(9)}`);
      // The forces are read as the uniform traction they stand for: the preload is the uniform compression.
      assertRatio(forces.criticalLoad, displacement.criticalLoad, CRITERIA.exact, "forces against moved sides");
      assertRatio(forces.criticalLoad, CRITICAL_LOAD, CRITERIA.criticalLoad / 10, "critical load");
      assert.ok(codesOf(forces.run).includes("forces-read-along-edges"), "the reading of the forces is told");
    });

// ---------------------------------------------------------------------------------------------------------------------
// Uniform states given as forces at the grid points of the four sides

// A rectangle of 4 × 2 quadrilaterals (cells of 0.3 by 0.25) under tractions on its four sides given as their
// consistent forces at the grid points; `traction(side)` is the traction of a side ("x0", "xa", "y0", "yb").
function loadedRectangle(traction, supports) {
  const mesh = rectangle(lines(1.2, 4), lines(0.5, 2));
  const [m, n, id] = [mesh.columns, mesh.rows, mesh.id];
  const forces = new Map(mesh.nodes.map((node) => [node.id, [0, 0]]));
  const add = (node, side, length) => {
    const [fx, fy] = traction(side);
    const force = forces.get(node);
    force[0] += fx * length;
    force[1] += fy * length;
  };
  for (let j = 0; j <= n; ++j) {
    add(id(0, j), "x0", tributary(mesh.ys, j));
    add(id(m, j), "xa", tributary(mesh.ys, j));
  }
  for (let i = 0; i <= m; ++i) {
    add(id(i, 0), "y0", tributary(mesh.xs, i));
    add(id(i, n), "yb", tributary(mesh.xs, i));
  }
  const records = [...supports(mesh)];
  // the plate does not bend: its deflection is held at the four corners
  records.push(`SinglePointConstrList;1;3;${id(0, 0)};${id(m, 0)};${id(0, n)};${id(m, n)}`);
  for (const [node, [fx, fy]] of forces)
    if (fx !== 0 || fy !== 0) records.push(`PointForceLoad;2;${node};;1.;${real(fx)};${real(fy)};0.`);
  const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All"];
  return { mesh, run: runModel(deck({ mesh, solution: 101, caseRecords, modelRecords: records }), MIXED) };
}

// The largest difference between the in-plane displacements of the grid points and `field(x, y)`, over `scale`.
function worstInPlane({ mesh, run }, field, scale) {
  let worst = 0;
  for (const { id, p } of mesh.nodes) {
    const [u, wanted] = [nodal(run, "displacements", id), field(p[0], p[1])];
    worst = Math.max(worst, Math.abs(u[0] - wanted[0]) / scale, Math.abs(u[1] - wanted[1]) / scale);
  }
  return worst;
}

test("biaxial tractions given as forces at the grid points of the four sides are a uniform state", (t) => {
  const [nx, ny] = [1000, -400];
  const outward = { x0: [-nx, 0], xa: [nx, 0], y0: [0, -ny], yb: [0, ny] };
  const solved = loadedRectangle((side) => outward[side], (mesh) => [
    `SinglePointConstrList;1;1;${mesh.id(2, 0)};${mesh.id(2, 2)}`,
    `SinglePointConstrList;1;2;${mesh.id(0, 1)};${mesh.id(4, 1)}`]);
  const strainX = (nx - POISSON * ny) / (YOUNG * THICKNESS);
  const strainY = (ny - POISSON * nx) / (YOUNG * THICKNESS);
  const worst = worstInPlane(solved, (x, y) => [strainX * (x - 0.6), strainY * (y - 0.25)], strainX * 0.6);
  t.diagnostic(`largest error of the displacements ${worst.toExponential(2)}`);
  // The force of a corner is shared between its two sides, each continuing its own traction.
  assert.ok(worst <= CRITERIA.exact, `error ${worst}`);
});

test("a shear flow given as forces at the grid points of the four sides is a uniform shear", (t) => {
  const flow = 1000;
  const shear = { x0: [0, -flow], xa: [0, flow], y0: [-flow, 0], yb: [flow, 0] };
  const solved = loadedRectangle((side) => shear[side], (mesh) => [
    `SinglePointConstrList;1;12;${mesh.id(2, 1)}`, `SinglePointConstrList;1;2;${mesh.id(0, 1)}`]);
  const angle = flow / (SHEAR_MODULUS * THICKNESS);
  const worst = worstInPlane(solved, (x, y) => [angle * (y - 0.25), 0], angle * 0.25);
  t.diagnostic(`largest error of the displacements ${worst.toExponential(2)}`);
  assert.ok(worst <= CRITERIA.exact, `error ${worst}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// The resultant of the tractions along curved edges

// Forces at the grid points of a curved side of `mesh` (`loaded`, each with its force), the grid points `clamped`
// held: the support forces and the applied load table have the resultant force and moment of the forces given.
function checkResultants(t, mesh, clamped, loaded, armScale) {
  const records = [`SinglePointConstrList;1;123456;${clamped.map((node) => node.id).join(";")}`,
    ...loaded.map(({ node, force }) => `PointForceLoad;2;${node.id};;1.;${force.map(real).join(";")}`)];
  const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "AppliedLoadOutput;All"];
  const run = runModel(deck({ mesh, solution: 101, caseRecords, modelRecords: records,
    geometry: { smoothEdges: innerEdges(mesh.elements) } }), MIXED);
  // the resultant at the origin of the rows of a nodal table: [Fx, Fy, Fz, Mx, My, Mz]
  const resultant = (rows) => {
    const sum = [0, 0, 0, 0, 0, 0];
    for (const { p, f } of rows) {
      for (let k = 0; k < 3; ++k) sum[k] += f[k];
      sum[3] += p[1] * f[2] - p[2] * f[1] + (f[3] ?? 0);
      sum[4] += p[2] * f[0] - p[0] * f[2] + (f[4] ?? 0);
      sum[5] += p[0] * f[1] - p[1] * f[0] + (f[5] ?? 0);
    }
    return sum;
  };
  const given = resultant(loaded.map(({ node, force }) => ({ p: node.p, f: force })));
  const tableResultant = (kind) => resultant(mesh.nodes.map((node) => ({ p: node.p, f: nodal(run, kind, node.id) })));
  const [reactions, applied] = [tableResultant("singlePointConstrForces"), tableResultant("appliedLoads")];
  const forceScale = loaded.reduce((sum, { force }) => sum + Math.hypot(...force), 0);
  let [worstReaction, worstApplied] = [0, 0];
  for (let k = 0; k < 6; ++k) {
    const scale = k < 3 ? forceScale : forceScale * armScale;
    worstReaction = Math.max(worstReaction, Math.abs(reactions[k] + given[k]) / scale);
    worstApplied = Math.max(worstApplied, Math.abs(applied[k] - given[k]) / scale);
  }
  t.diagnostic(`resultant of the support forces ${worstReaction.toExponential(2)}, of the applied loads ` +
    `${worstApplied.toExponential(2)}`);
  assert.ok(codesOf(run).includes("forces-read-along-edges"), "the forces are read along the edges");
  assert.ok(worstReaction <= CRITERIA.invariant, `support forces: ${worstReaction}`);
  assert.ok(worstApplied <= CRITERIA.invariant, `applied loads: ${worstApplied}`);
}

// A mesh of 4 × 2 quadrilaterals whose grid point (i, j) is at `place(i / 4, j / 2)`.
function mapped(place) {
  const mesh = rectangle(lines(1, 4), lines(1, 2));
  for (const node of mesh.nodes) node.p = place(node.p[0], node.p[1]);
  return mesh;
}

test("forces at the grid points of the rim of a quarter cylinder keep their resultant force and moment", (t) => {
  // radius 1, length 0.5: the side z = 0 clamped, the rim z = 0.5 loaded along the axis, the radius and the tangent
  const mesh = mapped((u, v) => {
    const angle = (u * Math.PI) / 2;
    return [Math.cos(angle), Math.sin(angle), 0.5 * v];
  });
  const loaded = mesh.nodes.filter((node) => node.j === 2).map((node) => {
    const [radial, tangent] = [[node.p[0], node.p[1], 0], [-node.p[1], node.p[0], 0]];
    const [alongAxis, alongRadius, alongTangent] = [10 + 3 * node.i, 5 * (node.i + 1), 7 - 2 * node.i];
    return { node, force: [0, 1, 2].map((k) =>
      (k === 2 ? alongAxis : 0) + alongRadius * radial[k] + alongTangent * tangent[k]) };
  });
  checkResultants(t, mesh, mesh.nodes.filter((node) => node.j === 0), loaded, 1);
});

test("forces at the grid points of the inner arc of a flat ring sector keep their resultant force and moment", (t) => {
  // radii 0.5 and 1 over a quarter turn, in the plane z = 0: the outer arc clamped, the inner one loaded in the plane
  const mesh = mapped((u, v) => {
    const [angle, radius] = [(u * Math.PI) / 2, 0.5 + 0.5 * v];
    return [radius * Math.cos(angle), radius * Math.sin(angle), 0];
  });
  const loaded = mesh.nodes.filter((node) => node.j === 0).map((node) => {
    const [radial, tangent] = [[2 * node.p[0], 2 * node.p[1], 0], [-2 * node.p[1], 2 * node.p[0], 0]];
    const [alongRadius, alongTangent] = [4 + node.i, 9 - 2 * node.i];
    return { node, force: [0, 1, 2].map((k) => alongRadius * radial[k] + alongTangent * tangent[k]) };
  });
  checkResultants(t, mesh, mesh.nodes.filter((node) => node.j === 2), loaded, 1);
});

test("a force alone on the outline stays a force at its grid point", () => {
  const mesh = rectangle(lines(1, 4), lines(1, 4));
  const records = [`SinglePointConstrList;1;123456;${[0, 1, 2, 3, 4].map((j) => mesh.id(0, j)).join(";")}`,
    // two forces on the side x = a, with a grid point without force between them: no edge has both its grid points
    `PointForceLoad;2;${mesh.id(4, 1)};;1.;0.;0.;5.`, `PointForceLoad;2;${mesh.id(4, 3)};;1.;0.;0.;5.`];
  const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "AppliedLoadOutput;All"];
  const run = runModel(deck({ mesh, solution: 101, caseRecords, modelRecords: records }), MIXED);
  assert.ok(!codesOf(run).includes("forces-read-along-edges"), "nothing is read along the edges");
  assert.equal(nodal(run, "appliedLoads", mesh.id(4, 1))[2], 5);
  assert.equal(nodal(run, "appliedLoads", mesh.id(4, 2))[2], 0);
});

// ---------------------------------------------------------------------------------------------------------------------
// The edge layer of a soft support

// The simply supported square plate under a pressure of 1 kPa: its run and its centre deflection.
function pressedPlate({ count, strip = 0, hard = false, shear = true, formulation = MIXED }) {
  const mesh = rectangle(lines(1, count, strip), lines(1, count, strip));
  const records = [...simpleSupports(mesh, hard), ...mesh.elements.map((_, k) => `ShellPressureLoad;2;1000.;${k + 1}`)];
  const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All"];
  const run = runModel(deck({ mesh, solution: 101, caseRecords, modelRecords: records, shear }), formulation);
  return { run, deflection: Math.abs(nodal(run, "displacements", mesh.id(mesh.columns / 2, mesh.rows / 2))[2]) };
}

test("a soft support on elements deeper than four thicknesses is told, and only there", () => {
  const warned = (options) => codesOf(pressedPlate(options).run).includes("edge-layer-not-resolved");
  const soft = pressedPlate({ count: 4 }).run.diagnostics.find((d) => d.code === "edge-layer-not-resolved");
  assert.ok(soft, "a soft support on elements 25 thicknesses deep");
  assert.equal(soft.severity, "warning");
  assert.match(soft.message, /^16 outline edges .* up to 25 thicknesses deep/);
  assert.ok(!warned({ count: 4, hard: true }), "a hard support has no such layer");
  assert.ok(!warned({ count: 4, shear: false }), "a section without transverse shear has no layer");
  assert.ok(!warned({ count: 4, strip: THICKNESS }), "a row one thickness wide along the sides resolves the layer");
  assert.ok(!warned({ count: 4, formulation: CLASSIC }), "the warning is the formulation's");
});

// The limits of the soft simply supported plate (side 1 m): its centre deflection under 1 kPa, its first frequency
// and its critical load, computed by this formulation at degree 4 on 8 × 8 elements with two rows of elements, 20
// and 10 mm wide, along each side (64 × 64 equal elements give the same four digits, and the ClassicFormulation
// reaches them at 256 × 256). They are 0.91 % above, 0.48 % and 0.94 % below the thin-plate values of the hard
// support: the edge layer of the soft support (section 3.6).
const SOFT_LIMITS = { deflection: 0.63949e-3, frequency: 48.17577, criticalLoad: 250.69784e3 };

test("2 x 2 elements with a row one thickness wide along each side reach the limits of the soft support", (t) => {
  const [count, strip] = [2, THICKNESS];
  const deflection = pressedPlate({ count, strip }).deflection;
  const mesh = rectangle(lines(1, count, strip), lines(1, count, strip));
  const modes = runModel(deck({ mesh, solution: 103, modelRecords: [...simpleSupports(mesh, false),
    "LanczosEigenMethod;3;;;1"], caseRecords: ["SubcaseStart;1", "SinglePointConstrSelection;1",
    "EigenMethodSelection;3", "DisplacementOutput;All"] }), MIXED, "modes");
  const frequency = firstEigenvalue(modes, "modes").cycles;
  const criticalLoad = compressedPlate({ count, strip, hard: false, loading: "forces" }).criticalLoad;
  t.diagnostic(`ratios to the limits: deflection ${(deflection / SOFT_LIMITS.deflection).toFixed(6)}, frequency ` +
    `${(frequency / SOFT_LIMITS.frequency).toFixed(6)}, critical load ` +
    `${(criticalLoad / SOFT_LIMITS.criticalLoad).toFixed(6)}`);
  assertRatio(deflection, SOFT_LIMITS.deflection, 1e-4, "centre deflection");
  assertRatio(frequency, SOFT_LIMITS.frequency, 1e-4, "first frequency");
  assertRatio(criticalLoad, SOFT_LIMITS.criticalLoad, 1e-4, "critical load");
});
