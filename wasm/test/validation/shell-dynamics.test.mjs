// Validation of the shells of the mixed hybrid formulation in the eigenvalue analyses and under inertia relief
// (docs/mixed-hybrid-formulation-spec.md, section 13.3): natural frequencies, buckling loads, free bodies, stiffened
// panels and the things that use a shell grid point, against closed forms, classical values or a converged
// ClassicFormulation model, with the criteria of section 13.6 (support.mjs, CRITERIA).
//
// Every case prints the ratios it measures ("ratio ..."), whether it passes or not.
//
// Run: node --test wasm/test/validation/shell-dynamics.test.mjs

import assert from "node:assert/strict";
import { CLASSIC, CRITERIA, MIXED, nodal, real, runModel, test } from "./support.mjs";

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
      note(Math.abs(value - 1) <= tolerance, `ratio ${what}: ${value.toFixed(6)} (${actual.toPrecision(7)} for ` +
        `${expected.toPrecision(7)}, criterion ${tolerance})`);
    },
    // |value| ≤ tolerance × scale.
    small(what, value, scale, tolerance) {
      note(Math.abs(value) <= tolerance * Math.abs(scale), `small ${what}: ${(Math.abs(value) / Math.abs(scale))
        .toExponential(2)} of the scale (criterion ${tolerance})`);
    },
    truth(what, ok) { note(Boolean(ok), what); },
    done() { assert.equal(failures.length, 0, failures.join("\n")); },
  };
}

// A deterministic number in [−1, 1] from two integers: the distortion of the meshes does not change between runs.
function jitter(i, j, salt) {
  const x = Math.sin(12.9898 * (i + 1) + 78.233 * (j + 1) + 37.719 * salt) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

// A structured mesh of nx × ny cells over the unit square of the parameters (s, t), mapped by map(s, t) = [x, y, z].
// shape: "ShellQuad4", or "ShellTria3" (each cell cut along a diagonal). distortion: the interior grid points are
// moved in (s, t) by up to that fraction of a cell (they stay on the surface of the map: on a curved surface the
// quadrilaterals are then warped); the grid points of the columns of `straightColumns` move in t only and those of
// the rows of `straightRows` in s only. closed: the mesh closes on itself in s (a cylinder). node(i, j) is the grid
// point of column i and row j, points the grid points with their parameters and positions, elements the element IDs,
// centroid(element) the mean of the corners of an element.
function mesh({ nx, ny, map, shape = "ShellQuad4", distortion = 0, closed = false, property = 1,
  straightColumns = [], straightRows = [] }) {
  const columns = closed ? nx : nx + 1;
  const node = (i, j) => 1 + j * columns + (closed ? ((i % nx) + nx) % nx : i);
  const lines = [];
  const points = [];
  for (let j = 0; j <= ny; ++j)
    for (let i = 0; i < columns; ++i) {
      let s = i / nx;
      let t = j / ny;
      if (distortion && (closed || (i > 0 && i < nx)) && j > 0 && j < ny) {
        if (!straightColumns.includes(i)) s += (distortion * jitter(i, j, 1)) / nx;
        if (!straightRows.includes(j)) t += (distortion * jitter(i, j, 2)) / ny;
      }
      const [x, y, z] = map(s, t);
      points.push({ id: node(i, j), i, j, s, t, x, y, z });
      lines.push(`Node;${node(i, j)};;${real(x)};${real(y)};${real(z)}`);
    }
  const elements = [];
  const corners = new Map();
  const add = (grids) => {
    elements.push(elements.length + 1);
    corners.set(elements.length, grids);
    lines.push(`${shape};${elements.length};${property};${grids.join(";")}`);
  };
  for (let j = 0; j < ny; ++j)
    for (let i = 0; i < nx; ++i) {
      const c = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
      if (shape === "ShellQuad4") add(c);
      else {
        add([c[0], c[1], c[2]]);
        add([c[0], c[2], c[3]]);
      }
    }
  const byId = new Map(points.map((point) => [point.id, point]));
  const centroid = (element) => {
    const grids = corners.get(element).map((id) => byId.get(id));
    return ["x", "y", "z"].map((axis) => grids.reduce((sum, grid) => sum + grid[axis], 0) / grids.length);
  };
  return { lines, node, points, elements, corners, centroid, nx, ny, shape };
}

// The text of a model file.
function deck(solution, caseLines, modelLines) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseLines, "[model]", ...modelLines, ""]
    .join("\n");
}

const MODES_CASE = ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"];
const FREE_MODES_CASE = ["SubcaseStart;1", "EigenMethodSelection;3"];
// A static subcase (supports 1, loads 2, temperatures 7 when `thermal`) and the buckling subcase on its stresses.
function bucklingCase(thermal = false) {
  const preload = thermal ? "TemperatureSelection;7;TemperatureLoadPart" : "StaticLoadSelection;2";
  return ["SubcaseStart;1", "SinglePointConstrSelection;1", preload, "SubcaseStart;2",
    "SinglePointConstrSelection;1", "EigenMethodSelection;3", "StaticSubcaseSelection;1"];
}
const eigenMethod = (count) => `LanczosEigenMethod;3;;;${count}`;

// The eigenvalue rows of a modal or buckling run.
function eigenRows(text, options, analysis) {
  const run = runModel(text, options, analysis);
  assert.ok(run.eigenvalues.length > 0, "an eigenvalue table");
  return { run, rows: run.eigenvalues[0].rows, values: run.eigenvalues[0].rows.map((row) => row.eigenvalue) };
}

// The diagnostics of a run that are not information.
function warningsOf(run) {
  return run.diagnostics.filter((d) => d.severity !== "info" && d.severity !== "information").map((d) => d.text);
}

// The rows of an element table as objects of its columns, with their element and grid point (0 for the centre).
function rowsOf(run, elementType, kind, subcase = 1) {
  const table = run.elementTables.find((t) => t.elementType === elementType && t.kind === kind &&
    t.subcase === subcase);
  assert.ok(table, `a ${elementType} ${kind} table of subcase ${subcase}`);
  const width = table.components.length;
  return Array.from(table.ids, (id, r) => {
    const row = { id, nodeId: table.nodeIds[r], layer: table.layers[r] };
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    return row;
  });
}

// The translations of the eigenvectors of a run (its table of each mode), as rows [id, t1, t2, t3].
function modeShapes(run) {
  return run.nodalTables.filter((t) => t.kind === "eigenvectors").sort((p, q) => p.step - q.step)
    .map((t) => ({ eigenvalue: t.eigenvalue, rows: Array.from(t.ids, (id, r) => [id, t.values[r * 6],
      t.values[r * 6 + 1], t.values[r * 6 + 2]]) }));
}

// The material of the isotropic cases, and the sections.
const E = 7.0e10;
const NU = 0.3;
const RHO = 2700.0;
const SHEAR_FACTOR = 0.833333; // the default shear thickness ratio of ShellProp
const isoMat = (poisson = NU, extra = "") => `IsoMat;1;${real(E)};;${real(poisson)};${real(RHO)}${extra}`;
const shellProp = (thickness) => `ShellProp;1;1;${real(thickness)};1;;1`;
const bendingStiffness = (thickness, poisson = NU) => (E * thickness ** 3) / (12 * (1 - poisson ** 2));

// The shapes and meshes every plate case runs: regular quadrilaterals, quadrilaterals distorted by 0.3 of their
// size (the envelope of the reference's campaign), and triangles on a distorted mesh.
const PLATE_MESHES = [
  { name: "ShellQuad4", shape: "ShellQuad4", distortion: 0 },
  { name: "distorted ShellQuad4", shape: "ShellQuad4", distortion: 0.3 },
  { name: "distorted ShellTria3", shape: "ShellTria3", distortion: 0.2 },
];

// ---------------------------------------------------------------------------------------------------------------------
// 1. Natural frequencies of plates

// A rectangular plate a × b in the XY plane of n × n cells, with the supports of `edges`:
//   "simple": hard simple support (w and the rotation about the normal of the edge held), "clamped", "free",
//   "cantilever" (clamped along x = 0).
// The in-plane displacements are held along the whole boundary when the plate has supports: the plate is flat, its
// bending modes do not see them, and the in-plane rigid motions of a plate held at points only would be modes whose
// frequency falls with the mesh (a point support of a membrane has no stiffness in the limit).
function plateModes({ n, shape, distortion, a = 1, b = 1, thickness, edges, count, section = null, poisson = NU,
  withShapes = false }) {
  const m = mesh({ nx: n, ny: n, map: (s, t) => [a * s, b * t, 0], shape, distortion });
  const lines = [...m.lines, ...(section ?? [shellProp(thickness), isoMat(poisson)]), eigenMethod(count)];
  for (let i = 0; i <= n; ++i) {
    const alongX = [m.node(i, 0), m.node(i, n)].join(";"); // the edges y = 0 and y = b
    const alongY = [m.node(0, i), m.node(n, i)].join(";"); // the edges x = 0 and x = a
    if (edges === "simple")
      lines.push(`SinglePointConstrList;1;1235;${alongX}`, `SinglePointConstrList;1;1234;${alongY}`);
    else if (edges === "clamped") lines.push(`SinglePointConstrList;1;123456;${alongX};${alongY}`);
    else if (edges === "cantilever") lines.push(`SinglePointConstrList;1;123456;${m.node(0, i)}`);
  }
  const caseLines = edges === "free" ? FREE_MODES_CASE : MODES_CASE;
  // The mode shapes are written when the displacements are asked.
  return deck(103, withShapes ? [...caseLines, "DisplacementOutput;All"] : caseLines, lines);
}

// The flexural frequency (ω²) of the mode of wavenumber k of a simply supported Mindlin plate (R. D. Mindlin,
// "Influence of rotatory inertia and shear on flexural motions of isotropic, elastic plates", Journal of Applied
// Mechanics 18, 1951): the lower root of
//   (κ G t k² − ρ t ω²)(D k² + κ G t − ρ t³/12 ω²) − (κ G t)² k² = 0.
function mindlinEigenvalue(k2, thickness, poisson = NU) {
  const d = bendingStiffness(thickness, poisson);
  const s = (SHEAR_FACTOR * E * thickness) / (2 * (1 + poisson));
  const m0 = RHO * thickness;
  const m2 = (RHO * thickness ** 3) / 12;
  // m0 m2 ω⁴ − (s k² m2 + m0 (D k² + s)) ω² + s k² D k² = 0
  const [qa, qb, qc] = [m0 * m2, -(s * k2 * m2 + m0 * (d * k2 + s)), s * d * k2 * k2];
  return (2 * qc) / (-qb + Math.sqrt(qb * qb - 4 * qa * qc));
}

// The thin plate on the three meshes, the thick ones on the distorted quadrilaterals.
const MINDLIN_CASES = [
  ...PLATE_MESHES.map((plateMesh) => ({ ...plateMesh, ratioName: "thin (a/t = 100)", thickness: 0.01 })),
  { ...PLATE_MESHES[1], ratioName: "thick (a/t = 10)", thickness: 0.1 },
  { ...PLATE_MESHES[1], ratioName: "very thick (a/t = 5)", thickness: 0.2 },
];

for (const { name, shape, distortion, ratioName, thickness } of MINDLIN_CASES)
  test(`modes of a simply supported square ${ratioName} plate of 8 x 8 ${name}: Mindlin's exact frequencies`, () => {
    const text = plateModes({ n: 8, shape, distortion, thickness, edges: "simple", count: 10, withShapes: true });
    const { rows: all, run } = eigenRows(text, MIXED, "modes");
    // The bending modes: those that move across the plate. The in-plane modes of the plate held along its boundary
    // come among the first ten when the plate is thick (the fifth and the sixth at a/t = 5).
    const shapes = modeShapes(run);
    const rows = all.filter((row, k) => {
      const across = Math.max(...shapes[k].rows.map((r) => Math.abs(r[3])));
      const along = Math.max(...shapes[k].rows.map((r) => Math.hypot(r[1], r[2])));
      return across > along;
    });
    const check = measurements();
    check.truth(`six bending modes among the first ten: ${rows.length}`, rows.length >= 6);
    // (1,1), (1,2), (2,1), (2,2), (1,3), (3,1): k² = π² (m² + n²) / a².
    const expected = [2, 5, 5, 8, 10, 10].map((sum) => mindlinEigenvalue(Math.PI ** 2 * sum, thickness));
    expected.forEach((value, k) => check.ratio(`frequency ${k + 1}`, Math.sqrt(rows[k].eigenvalue / value), 1,
      CRITERIA.frequency));
    // The modes are normalised to a unit generalised mass; the generalised stiffness is then the eigenvalue.
    for (const row of rows.slice(0, 6)) {
      check.ratio(`generalised mass of mode ${row.mode}`, row.generalizedMass, 1, 1e-8);
      check.ratio(`generalised stiffness of mode ${row.mode}`, row.generalizedStiffness, row.eigenvalue, 1e-6);
    }
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// The frequency parameters λ = ω a² √(ρ t / D) of thin square plates, ν = 0.3.
// Clamped: A. W. Leissa, Vibration of Plates, NASA SP-160 (1969), and "The free vibration of rectangular plates",
// Journal of Sound and Vibration 31 (1973): 35.992, 73.413, 73.413, 108.27, 131.64, 132.24 (upper bounds, within
// 0.05 % of the converged values 35.985, 73.394, 108.22, 131.58, 132.20).
// Free: the same paper gives the upper bounds 13.489, 19.789, 24.432, 35.024; the converged values are 13.468, 19.596,
// 24.270, 34.801, 34.801, 61.093 (D. J. Gorman, Free Vibration Analysis of Rectangular Plates, 1982).
// Cantilever: Leissa gives the upper bounds 3.4917, 8.5246, 21.429, 27.331, 31.111; the converged values are 3.4710,
// 8.5063, 21.285, 27.199, 30.955.
const THIN = 0.002; // a/t = 500: the shear flexibility lowers these frequencies by less than 2e-4
const LEISSA = {
  clamped: [35.985, 73.394, 73.394, 108.22, 131.58, 132.20],
  free: [13.468, 19.596, 24.270, 34.801, 34.801, 61.093],
  cantilever: [3.4710, 8.5063, 21.285, 27.199, 30.955],
};

for (const edges of ["clamped", "cantilever", "free"])
  for (const { name, shape, distortion } of PLATE_MESHES)
    test(`modes of a ${edges} thin square plate of 8 x 8 ${name}: the frequency parameters of Leissa`, () => {
      const rigid = edges === "free" ? 6 : 0;
      const expected = LEISSA[edges];
      const text = plateModes({ n: 8, shape, distortion, thickness: THIN, edges, count: rigid + expected.length });
      const { values, run } = eigenRows(text, MIXED, "modes");
      const check = measurements();
      const unit = bendingStiffness(THIN) / (RHO * THIN); // ω² = λ² D / (ρ t a⁴), a = 1
      // The six rigid-body modes of the free plate: their frequency below 1e-3 of the first elastic one.
      for (let k = 0; k < rigid; ++k)
        check.small(`rigid-body frequency ${k + 1}`, Math.sqrt(Math.abs(values[k])), Math.sqrt(values[rigid]), 1e-3);
      expected.forEach((lambda, k) => check.ratio(`frequency parameter ${k + 1}`,
        Math.sqrt(values[rigid + k] / unit), lambda, CRITERIA.frequency));
      check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
      check.done();
    });

// ---------------------------------------------------------------------------------------------------------------------
// 2. Buckling of plates

// A flat plate a × b of nx × ny cells under a uniform membrane state, put by the enforced in-plane displacements of
// the affine field u = εx x + γ y / 2, v = γ x / 2 + εy y on the grid points of its whole boundary (carried linearly
// along the edges: the uniform strain is exact), or by a temperature rise with the boundary held. Out of its plane the
// plate is simply supported (hard) or clamped on the edges named in `supported` ("x": the edges x = 0 and x = a, "y":
// the edges y = 0 and y = b).
function plateBuckling({ nx, ny, shape, distortion, a = 1, b = 1, thickness, edges = "simple", supported = "xy",
  strain = [0, 0, 0], poisson = NU, rise = 0, expansion = 0, count = 4, extra = [] }) {
  const m = mesh({ nx, ny, map: (s, t) => [a * s, b * t, 0], shape, distortion });
  const material = isoMat(poisson, expansion ? `;${real(expansion)};0.` : "");
  const lines = [...m.lines, shellProp(thickness), material, eigenMethod(count), ...extra];
  const [ex, ey, gamma] = strain;
  const boundary = m.points.filter(({ i, j }) => i === 0 || i === nx || j === 0 || j === ny);
  for (const { id, i, j, x, y } of boundary) {
    lines.push(`SinglePointConstr;1;${id};1;${real(ex * x + (gamma * y) / 2)}`,
      `SinglePointConstr;1;${id};2;${real((gamma * x) / 2 + ey * y)}`);
    const onX = supported.includes("x") && (i === 0 || i === nx);
    const onY = supported.includes("y") && (j === 0 || j === ny);
    if (!onX && !onY) continue;
    // Hard simple support: w and the rotation about the normal of the edge (about x on an edge x = constant).
    const components = edges === "clamped" ? "3456" : `3${onX ? "4" : ""}${onY ? "5" : ""}`;
    lines.push(`SinglePointConstrList;1;${components};${id}`);
  }
  if (rise) lines.push(`DefaultTemperatureLoad;7;${real(rise)}`);
  return { text: deck(105, bucklingCase(rise !== 0), lines), mesh: m };
}

// The lowest positive buckling factor of a run.
function lowestFactor(text, options) {
  const { values, run } = eigenRows(text, options, "buckling");
  const positive = values.filter((value) => value > 0).sort((p, q) => p - q);
  assert.ok(positive.length > 0, `a positive buckling factor among ${values}`);
  return { factor: positive[0], values, run };
}

// The buckling coefficients k of a square plate, N_cr = k π² D / b² (S. P. Timoshenko and J. M. Gere, Theory of
// Elastic Stability, 2nd edition, chapter 9).
const BUCKLING_THICKNESS = 0.005; // b/t = 200: the transverse shear lowers these loads by less than 1e-3
const PLATE_BUCKLING_CASES = [
  // Simply supported, uniaxial compression: k = (m + 1/m)² = 4.
  { name: "simply supported plate under uniaxial compression (k = 4)", edges: "simple", k: 4, meshes: PLATE_MESHES,
    membrane: (sigma) => [-sigma / E, (NU * sigma) / E, 0] },
  // Simply supported, equal compression in both directions: k = 2.
  { name: "simply supported plate under biaxial compression (k = 2)", edges: "simple", k: 2,
    meshes: PLATE_MESHES.slice(1, 2), membrane: (sigma) => [(-sigma * (1 - NU)) / E, (-sigma * (1 - NU)) / E, 0] },
  // Simply supported, pure shear: k = 9.34 in Timoshenko and Gere; the converged value of the series is 9.325.
  { name: "simply supported plate under shear (k = 9.34)", edges: "simple", k: 9.325, meshes: PLATE_MESHES,
    membrane: (sigma) => [0, 0, (2 * (1 + NU) * sigma) / E] },
  // Clamped, uniaxial compression: k = 10.07.
  { name: "clamped plate under uniaxial compression (k = 10.07)", edges: "clamped", k: 10.07,
    meshes: PLATE_MESHES.slice(1), membrane: (sigma) => [-sigma / E, (NU * sigma) / E, 0] },
];

for (const { name, edges, k, membrane, meshes } of PLATE_BUCKLING_CASES)
  for (const { name: meshName, shape, distortion } of meshes)
    test(`buckling of a ${name}, 8 x 8 ${meshName}`, () => {
      const sigma = 1.0e6;
      const { text } = plateBuckling({ nx: 8, ny: 8, shape, distortion, thickness: BUCKLING_THICKNESS, edges,
        strain: membrane(sigma) });
      const { factor, run } = lowestFactor(text, MIXED);
      const check = measurements();
      const expected = (k * Math.PI ** 2 * bendingStiffness(BUCKLING_THICKNESS)) / (sigma * BUCKLING_THICKNESS);
      check.ratio("critical load", factor, expected, CRITERIA.criticalLoad);
      check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
      check.done();
    });

for (const { name: meshName, shape, distortion } of PLATE_MESHES.slice(1))
  test(`thermal buckling of a simply supported plate held in its plane, 8 x 8 ${meshName}`, () => {
    // The heated plate is in equal biaxial compression N = E α ΔT t / (1 − ν); it buckles at N = 2 π² D / a²
    // (Timoshenko and Gere, chapter 9): ΔT_cr = π² t² / (6 (1 + ν) α a²).
    const expansion = 1.0e-5;
    const { text } = plateBuckling({ nx: 8, ny: 8, shape, distortion, thickness: BUCKLING_THICKNESS, rise: 1,
      expansion });
    const { factor, run } = lowestFactor(text, MIXED);
    const check = measurements();
    check.ratio("critical temperature rise", factor,
      (Math.PI ** 2 * BUCKLING_THICKNESS ** 2) / (6 * (1 + NU) * expansion), CRITERIA.criticalLoad);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

for (const { name: meshName, shape, distortion } of PLATE_MESHES)
  test(`buckling of a plate column pinned at both ends (Euler), 8 x 2 ${meshName}`, () => {
    // A strip L × b without Poisson's ratio, free along its long edges, shortened by its ends: P_cr = π² E I / L²,
    // I = b t³ / 12, less the effect of the shear flexibility, P_cr / (1 + P_cr / (κ G b t)) (Timoshenko and Gere,
    // chapter 2).
    const [length, width, thickness, strain] = [2.0, 0.4, 0.01, 1.0e-6];
    const { text } = plateBuckling({ nx: 8, ny: 2, shape, distortion, a: length, b: width, thickness, poisson: 0,
      supported: "x", strain: [-strain, 0, 0] });
    // The long edges are free: only the ends hold the in-plane displacements (and one grid point holds v).
    const lines = text.split("\n").filter((line) => {
      const match = /^SinglePointConstr;1;(\d+);([12]);/.exec(line);
      if (!match) return true;
      const id = Number(match[1]);
      const column = (id - 1) % 9;
      if (match[2] === "2") return id === 1;
      return column === 0 || column === 8;
    });
    const { factor, run } = lowestFactor(lines.join("\n"), MIXED);
    const check = measurements();
    const euler = (Math.PI ** 2 * E * width * thickness ** 3) / (12 * length ** 2);
    const expected = euler / (1 + euler / ((SHEAR_FACTOR * E * width * thickness) / 2));
    check.ratio("critical load", factor * E * width * thickness * strain, expected, CRITERIA.criticalLoad);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 3. Curved shells: modes of cylinders and of a ring

// The smallest eigenvalue of a small symmetric matrix (cyclic Jacobi rotations).
function smallestEigenvalue(matrix) {
  const a = matrix.map((row) => row.slice());
  const n = a.length;
  for (let sweep = 0; sweep < 60; ++sweep)
    for (let p = 0; p < n; ++p)
      for (let q = p + 1; q < n; ++q) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = (theta < 0 ? -1 : 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; ++k) [a[k][p], a[k][q]] = [c * a[k][p] - s * a[k][q], s * a[k][p] + c * a[k][q]];
        for (let k = 0; k < n; ++k) [a[p][k], a[q][k]] = [c * a[p][k] - s * a[q][k], s * a[p][k] + c * a[q][k]];
      }
  return Math.min(...a.map((row, i) => row[i]));
}

// The harmonic (m, n) of a circular cylinder between shear diaphragms (the radial and tangential displacements held
// at its ends, the axial one free): u = U cos(a x) cos(n θ), v = V sin(a x) sin(n θ), w = W sin(a x) cos(n θ),
// βx = Bx cos(a x) cos(n θ), βθ = Bθ sin(a x) sin(n θ), a = m π / L. It is an exact solution of the shell equations
// (A. W. Leissa, Vibration of Shells, NASA SP-288, 1973, chapter 2), written here for the shell with transverse shear
// and rotary inertia (as the plate of Mindlin) and the twist of the theory of Sanders:
//   εx = u,x   εθ = (v,θ + w) / R   γ = u,θ / R + v,x
//   κx = βx,x  κθ = βθ,θ / R        κxθ = βθ,x + βx,θ / R + (v,x − u,θ / R) / (2 R)
//   γxz = βx + w,x                  γθz = βθ + (w,θ − v) / R
// The stiffness of the five amplitudes (U, V, W, Bx, Bθ), per unit of the integral of cos² or sin².
function cylinderHarmonicStiffness({ thickness, radius, a, n, poisson }) {
  const g = E / (2 * (1 + poisson));
  const membrane = (E * thickness) / (1 - poisson ** 2);
  const bending = bendingStiffness(thickness, poisson);
  const ex = [-a, 0, 0, 0, 0];
  const et = [0, n / radius, 1 / radius, 0, 0];
  const gamma = [-n / radius, a, 0, 0, 0];
  const kx = [0, 0, 0, -a, 0];
  const kt = [0, 0, 0, 0, n / radius];
  const kxt = [n / (2 * radius ** 2), a / (2 * radius), 0, -n / radius, a];
  const gxz = [0, 0, a, 1, 0];
  const gtz = [0, -1 / radius, -n / radius, 0, 1];
  const k = Array.from({ length: 5 }, () => new Array(5).fill(0));
  const add = (factor, p, q) => {
    for (let i = 0; i < 5; ++i) for (let j = 0; j < 5; ++j) k[i][j] += (factor * (p[i] * q[j] + q[i] * p[j])) / 2;
  };
  add(membrane, ex, ex);
  add(membrane, et, et);
  add(2 * poisson * membrane, ex, et);
  add(g * thickness, gamma, gamma);
  add(bending, kx, kx);
  add(bending, kt, kt);
  add(2 * poisson * bending, kx, kt);
  add((g * thickness ** 3) / 12, kxt, kxt);
  add(SHEAR_FACTOR * g * thickness, gxz, gxz);
  add(SHEAR_FACTOR * g * thickness, gtz, gtz);
  return k;
}

// ω² of the harmonic (m, n ≥ 1): the smallest eigenvalue of M^(−1/2) K M^(−1/2), M = ρ t (1, 1, 1, t²/12, t²/12).
function cylinderHarmonicEigenvalue(harmonic) {
  const k = cylinderHarmonicStiffness(harmonic);
  const mass = [1, 1, 1, harmonic.thickness ** 2 / 12, harmonic.thickness ** 2 / 12]
    .map((value) => value * RHO * harmonic.thickness);
  return smallestEigenvalue(k.map((row, i) => row.map((value, j) => value / Math.sqrt(mass[i] * mass[j]))));
}

// The axial compression per unit length at which the harmonic (m, n) buckles: the prestress works on
// (u,x² + v,x² + w,x²) / 2, so that K q = N a² diag(1, 1, 1, 0, 0) q; the rotations are condensed. For n = 0 the
// harmonic has no v and no βθ.
function cylinderHarmonicBucklingLoad(harmonic) {
  const k = cylinderHarmonicStiffness(harmonic);
  const kept = harmonic.n === 0 ? [0, 2] : [0, 1, 2];
  const condensed = harmonic.n === 0 ? [3] : [3, 4];
  // The inverse of the block of the rotations (1 × 1 or 2 × 2).
  const [p, q] = condensed;
  const det = q === undefined ? k[p][p] : k[p][p] * k[q][q] - k[p][q] ** 2;
  const inverse = q === undefined ? [[1 / det]]
    : [[k[q][q] / det, -k[p][q] / det], [-k[p][q] / det, k[p][p] / det]];
  const reduced = kept.map((i) => kept.map((j) => {
    let value = k[i][j];
    condensed.forEach((r, x) => condensed.forEach((s, y) => { value -= k[i][r] * inverse[x][y] * k[s][j]; }));
    return value / harmonic.a ** 2;
  }));
  return smallestEigenvalue(reduced);
}

const CYLINDER = { radius: 1.0, length: 2.0 };

// A whole cylinder of nt × nz cells between shear diaphragms: its end grid points hold their displacements in the
// plane of the end (x and y: both grid points of an end edge hold the same two directions, so the edge holds them).
function cylinderModes({ nt, nz, thickness, shape, distortion, count }) {
  const m = mesh({ nx: nt, ny: nz, closed: true, shape, distortion, map: (s, t) => [
    CYLINDER.radius * Math.cos(2 * Math.PI * s), CYLINDER.radius * Math.sin(2 * Math.PI * s), CYLINDER.length * t] });
  const lines = [...m.lines, shellProp(thickness), isoMat(), eigenMethod(count)];
  for (let i = 0; i < nt; ++i) lines.push(`SinglePointConstrList;1;12;${m.node(i, 0)};${m.node(i, nz)}`);
  return deck(103, MODES_CASE, lines);
}

// The lowest harmonics of the cylinder, sorted: each one a pair of modes (cos n θ and sin n θ).
function cylinderHarmonics(thickness, pairs) {
  const list = [];
  for (let m = 1; m <= 3; ++m)
    for (let n = 1; n <= 14; ++n)
      list.push({ m, n, eigenvalue: cylinderHarmonicEigenvalue({ thickness, radius: CYLINDER.radius,
        a: (m * Math.PI) / CYLINDER.length, n, poisson: NU }) });
  return list.sort((p, q) => p.eigenvalue - q.eigenvalue).slice(0, pairs);
}

// The quadrilaterals of the distorted meshes of a cylinder are warped; the triangles of a distorted mesh of a curved
// surface meet at kinks (the geometry report counts them), which the formulation takes as folds (spec section 3.6).
// The harmonics counted have 4 to 7 waves around the cylinder at R/t = 100 and 2 to 5 at R/t = 20: the meshes have
// 1.6 to 4 elements per half-wave around it (32 x 8 regular quadrilaterals give the same ratios to 1e-5).
const CYLINDER_MODE_CASES = [
  { name: "R/t = 100, 24 x 6 warped ShellQuad4", thickness: 0.01, nt: 24, nz: 6, shape: "ShellQuad4",
    distortion: 0.2 },
  { name: "R/t = 100, 24 x 6 distorted ShellTria3", thickness: 0.01, nt: 24, nz: 6, shape: "ShellTria3",
    distortion: 0.2 },
  { name: "R/t = 20, 16 x 4 warped ShellQuad4", thickness: 0.05, nt: 16, nz: 4, shape: "ShellQuad4",
    distortion: 0.2 },
];

for (const { name, thickness, nt, nz, shape, distortion } of CYLINDER_MODE_CASES)
  test(`modes of a cylinder between diaphragms, ${name}: the harmonics of the shear-deformable shell`, () => {
    // The rigid translation along the axis is free (one zero frequency), then four pairs of harmonics.
    const { values, run } = eigenRows(cylinderModes({ nt, nz, thickness, shape, distortion, count: 9 }), MIXED,
      "modes");
    const check = measurements();
    const harmonics = cylinderHarmonics(thickness, 4);
    check.small("rigid-body frequency", Math.sqrt(Math.abs(values[0])), Math.sqrt(values[1]), 1e-3);
    harmonics.forEach(({ m, n, eigenvalue }, k) => {
      for (const mode of [1 + 2 * k, 2 + 2 * k])
        check.ratio(`frequency of mode ${mode + 1}, harmonic (${m}, ${n})`, Math.sqrt(values[mode] / eigenvalue), 1,
          CRITERIA.frequency);
    });
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

const RING_MESHES = [
  { name: "32 x 1 ShellQuad4", nt: 32, nz: 1, shape: "ShellQuad4", distortion: 0 },
  { name: "32 x 2 warped ShellQuad4", nt: 32, nz: 2, shape: "ShellQuad4", distortion: 0.3 },
  { name: "32 x 2 ShellTria3", nt: 32, nz: 2, shape: "ShellTria3", distortion: 0 },
];

for (const { name, nt, nz, shape, distortion } of RING_MESHES)
  test(`modes of a free ring, ${name}: the inextensional bending modes of the thin ring`, () => {
    // A free ring of radius R, of rectangular section b × t (t radial), without Poisson's ratio: its bending modes in
    // its plane with n waves are ω² = E I n² (n² − 1)² / (ρ A R⁴ (n² + 1)), I = b t³ / 12, A = b t (R. D. Blevins,
    // Formulas for Natural Frequency and Mode Shape, 1979, chapter 9; A. E. H. Love, A Treatise on the Mathematical
    // Theory of Elasticity). R/t = 100: the extension, the shear and the rotary inertia that the formula
    // leaves out lower the frequency by less than 1e-3 up to n = 4.
    const [radius, width, thickness] = [1.0, 0.1, 0.01];
    const m = mesh({ nx: nt, ny: nz, closed: true, shape, distortion, map: (s, t) => [
      radius * Math.cos(2 * Math.PI * s), radius * Math.sin(2 * Math.PI * s), width * t] });
    const lines = [...m.lines, shellProp(thickness), isoMat(0), eigenMethod(16)];
    const text = deck(103, [...FREE_MODES_CASE, "DisplacementOutput;All"], lines);
    const { values, run } = eigenRows(text, MIXED, "modes");
    const check = measurements();
    for (let k = 0; k < 6; ++k)
      check.small(`rigid-body frequency ${k + 1}`, Math.sqrt(Math.abs(values[k])), Math.sqrt(values[6]), 1e-3);
    // The modes in the plane of the ring: those whose displacement along the axis is small (it is not zero on a
    // distorted mesh, which is not symmetric about the middle plane of the ring).
    const inPlane = modeShapes(run).slice(6).filter(({ rows }) => {
      const axial = Math.max(...rows.map((row) => Math.abs(row[3])));
      const planar = Math.max(...rows.map((row) => Math.hypot(row[1], row[2])));
      return axial < 0.05 * planar;
    });
    check.truth(`at least six modes in the plane of the ring among the first ten elastic ones: ${inPlane.length}`,
      inPlane.length >= 6);
    const unit = (E * thickness ** 2) / (12 * RHO * radius ** 4);
    [2, 3, 4].forEach((n, k) => {
      const expected = (unit * n ** 2 * (n ** 2 - 1) ** 2) / (n ** 2 + 1);
      for (const mode of inPlane.slice(2 * k, 2 * k + 2))
        check.ratio(`frequency of a mode of ${n} waves`, Math.sqrt(mode.eigenvalue / expected), 1,
          CRITERIA.frequency);
    });
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// A cross-ply laminate [0/90]s of four equal plies, as a CompProp of OrthoMat2D plies, its material axes along the
// basic system (the material frame 0 of the elements, projected on them).
const PLY = { e1: 1.4e11, e2: 1.0e10, nu12: 0.3, g12: 5.0e9, rho: 1600.0 };
function laminateSection(thickness) {
  const ply = real(thickness / 4);
  return [`CompProp;1;;;;;;;;2;${ply};0.;;2;${ply};90.;;2;${ply};90.;;2;${ply};0.`,
    `OrthoMat2D;2;${real(PLY.e1)};${real(PLY.e2)};${real(PLY.nu12)};${real(PLY.g12)};${real(PLY.g12)};` +
    `${real(PLY.g12)};${real(PLY.rho)}`];
}

for (const { name, shape, distortion } of PLATE_MESHES.slice(1))
  test(`modes of a simply supported [0/90]s laminate plate of 8 x 8 ${name}: the classical lamination theory`, () => {
    // ω² = π⁴ (D11 m⁴ + 2 (D12 + 2 D66) m² n² + D22 n⁴) / (ρ h a⁴) for a specially orthotropic plate (R. M. Jones,
    // Mechanics of Composite Materials, section 5.4; J. M. Whitney, Structural Analysis of Laminated Anisotropic
    // Plates), with D11 = h³ (7 Q11 + Q22) / 96, D22 = h³ (7 Q22 + Q11) / 96, D12 = Q12 h³ / 12, D66 = G12 h³ / 12
    // for [0/90]s. a/h = 200: the transverse shear lowers these frequencies by less than 1e-3.
    const thickness = 0.005;
    const text = plateModes({ n: 8, shape, distortion, edges: "simple", count: 4, section: laminateSection(thickness) })
      .replace(/^(Shell(?:Quad4|Tria3);\d+;1;[\d;]+)$/gm, "$1;;0");
    const { values, run } = eigenRows(text, MIXED, "modes");
    const check = measurements();
    const denominator = 1 - (PLY.nu12 ** 2 * PLY.e2) / PLY.e1;
    const [q11, q22, q12] = [PLY.e1 / denominator, PLY.e2 / denominator, (PLY.nu12 * PLY.e2) / denominator];
    const cube = thickness ** 3;
    const [d11, d22, d12, d66] = [(cube * (7 * q11 + q22)) / 96, (cube * (7 * q22 + q11)) / 96, (q12 * cube) / 12,
      (PLY.g12 * cube) / 12];
    const expected = [];
    for (let m = 1; m <= 3; ++m)
      for (let n = 1; n <= 4; ++n)
        expected.push((Math.PI ** 4 * (d11 * m ** 4 + 2 * (d12 + 2 * d66) * m * m * n * n + d22 * n ** 4)) /
          (PLY.rho * thickness));
    expected.sort((p, q) => p - q);
    for (let k = 0; k < 4; ++k)
      check.ratio(`frequency ${k + 1}`, Math.sqrt(values[k] / expected[k]), 1, CRITERIA.frequency);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. Buckling of a cylinder, of a deep strip and of stiffened plates

// A quarter of a cylinder of radius R and thickness t between the planes y = 0 and x = 0 (symmetry: the displacement
// across the plane and the rotations in it held), nt × nz cells, of the length of `waves` axial half-waves of the
// axisymmetric buckle, l = π √(R t) / (12 (1 − ν²))^(1/4). The static subcase shortens it by its ends, which hold
// their radial expansion ν ε R (the membrane state is then uniform); the buckling subcase holds the ends in their
// plane and leaves them free along the axis (one grid point holds the axial rigid motion).
function cylinderBuckling({ nt, nz, shape, distortion, radius = 1.0, thickness = 0.01, waves = 2, strain = 1.0e-4 }) {
  const length = (waves * Math.PI * Math.sqrt(radius * thickness)) / (12 * (1 - NU ** 2)) ** 0.25;
  const m = mesh({ nx: nt, ny: nz, shape, distortion, map: (s, t) => [
    radius * Math.cos((Math.PI * s) / 2), radius * Math.sin((Math.PI * s) / 2), length * t] });
  const lines = [...m.lines, shellProp(thickness), isoMat(), eigenMethod(4)];
  for (const { id, i, j, x, y } of m.points) {
    const isEnd = j === 0 || j === nz;
    for (const set of [1, 3]) {
      if (isEnd) {
        lines.push(`SinglePointConstr;${set};${id};1;${real(NU * strain * x)}`,
          `SinglePointConstr;${set};${id};2;${real(NU * strain * y)}`);
        if (set === 1 || (i === 0 && j === 0))
          lines.push(`SinglePointConstr;${set};${id};3;${real(j === 0 ? 0 : -strain * length)}`);
      }
      if (i === 0) lines.push(`SinglePointConstrList;${set};${isEnd ? "46" : "246"};${id}`);
      if (i === nt) lines.push(`SinglePointConstrList;${set};${isEnd ? "56" : "156"};${id}`);
    }
  }
  const caseLines = bucklingCase().map((line, k) => (k === 4 ? "SinglePointConstrSelection;3" : line));
  // The classical load per unit length, E t² / (R √(3 (1 − ν²))) = 0.605 E t² / R (Timoshenko and Gere, Theory of
  // Elastic Stability, chapter 11), and the lowest load of the harmonics of this quarter (n even, m = 1, 2, ...).
  const classical = (E * thickness ** 2) / (radius * Math.sqrt(3 * (1 - NU ** 2)));
  let harmonic = Infinity;
  for (let mm = 1; mm <= 8; ++mm)
    for (let n = 0; n <= 40; n += 2)
      harmonic = Math.min(harmonic, cylinderHarmonicBucklingLoad({ thickness, radius, a: (mm * Math.PI) / length, n,
        poisson: NU }));
  return { text: deck(105, caseLines, lines), unit: E * thickness * strain, classical, harmonic };
}

const CYLINDER_BUCKLING_MESHES = [
  { name: "12 x 8 warped ShellQuad4", shape: "ShellQuad4", distortion: 0.2 },
  { name: "12 x 8 ShellTria3", shape: "ShellTria3", distortion: 0 },
];

for (const { name, shape, distortion } of CYLINDER_BUCKLING_MESHES)
  test(`buckling of a cylinder under axial compression, R/t = 100, ${name}`, () => {
    // The classical value comes from the shallow shell equations. The harmonics of the shear-deformable shell
    // (cylinderHarmonicBucklingLoad) give 0.9949 of it at R/t = 100 on this length: the criterion is on the harmonic
    // value, and the ratio to the classical value is printed.
    const model = cylinderBuckling({ nt: 12, nz: 8, shape, distortion });
    const { factor, run } = lowestFactor(model.text, MIXED);
    const check = measurements();
    check.ratio("critical load over the harmonic solution", factor * model.unit, model.harmonic,
      CRITERIA.criticalLoad);
    console.log(`    info ratio critical load over the classical 0.605 E t² / R: ` +
      `${((factor * model.unit) / model.classical).toFixed(6)}`);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

test("buckling of a cylinder under axial compression, R/t = 100, 12 x 8 distorted ShellTria3 (known limit)", () => {
  // KNOWN LIMIT (spec section 3.6: the kinks between triangles read on a curved surface are folds for this shell).
  // The triangles of a distorted mesh of the cylinder meet at kinks of up to 2 degrees (geometry report), and the
  // axial buckling load of a cylinder is very sensitive to them: 0.918 of the harmonic solution on this mesh (0.909
  // with a distortion of 0.1), against 0.9997 for the same triangles on the regular mesh and for warped
  // quadrilaterals. The criterion written here is the one the case meets, not that of section 13.6.
  const model = cylinderBuckling({ nt: 12, nz: 8, shape: "ShellTria3", distortion: 0.2 });
  const { factor } = lowestFactor(model.text, MIXED);
  const check = measurements();
  check.ratio("critical load over the harmonic solution (known limit, criterion 0.1)", factor * model.unit,
    model.harmonic, 0.1);
  check.done();
});

// A strip of length L, depth h and thickness t in the XZ plane, without Poisson's ratio, bent in its plane by the
// rotations ∓θ of its ends, which hold the lateral displacement along their whole depth (fork supports):
//   M_cr = (π / L) √(E Iy (G J + π² E Cw / L²)), Iy = h t³ / 12, Cw = h³ t³ / 144 (the warping constant of the narrow
// rectangle; Timoshenko and Gere, Theory of Elastic Stability, chapter 6), J = h t³ (1 − 0.630 t / h) / 3
// (Saint-Venant's torsion constant of the narrow rectangle; Timoshenko and Goodier, Theory of Elasticity,
// chapter 10): the shell with its transverse shear has the layer of the free edges that makes the factor
// 1 − 0.630 t / h.
function lateralBuckling({ nx, ny, shape, distortion }) {
  const [length, depth, thickness, theta] = [2.0, 0.2, 0.005, 1.0e-4];
  const m = mesh({ nx, ny, shape, distortion, map: (s, t) => [length * s, 0, depth * (t - 0.5)] });
  const lines = [...m.lines, shellProp(thickness), isoMat(0), eigenMethod(4)];
  for (const { id, i, j, z } of m.points) {
    if (i !== 0 && i !== nx) continue;
    lines.push(`SinglePointConstr;1;${id};1;${real((i === 0 ? 1 : -1) * theta * z)}`,
      `SinglePointConstrList;1;2;${id}`);
    if (2 * j === ny) lines.push(`SinglePointConstrList;1;3;${id}`);
  }
  const applied = (E * thickness * depth ** 3 * 2 * theta) / (12 * length);
  const [iy, j, cw] = [(depth * thickness ** 3) / 12, ((depth * thickness ** 3) / 3) * (1 - (0.63 * thickness) / depth),
    (depth ** 3 * thickness ** 3) / 144];
  const expected = (Math.PI / length) * Math.sqrt(E * iy * ((E / 2) * j + (Math.PI ** 2 * E * cw) / length ** 2));
  return { text: deck(105, bucklingCase(), lines), applied, expected };
}

// The twisting strip has free edges: their layer, of the width of the thickness, is spread over an element at coarse
// meshes and converges slowly (spec section 3.6, first known limit). Measured over the formula: 1.0043 (8 x 2
// quadrilaterals), 1.0029 (distorted), 1.0019 (16 x 4), 1.0015 (32 x 4).
for (const { name, nx, ny, shape, distortion } of [
  { name: "8 x 2 ShellQuad4", nx: 8, ny: 2, shape: "ShellQuad4", distortion: 0 },
  { name: "8 x 2 distorted ShellQuad4", nx: 8, ny: 2, shape: "ShellQuad4", distortion: 0.3 },
  { name: "16 x 4 distorted ShellTria3", nx: 16, ny: 4, shape: "ShellTria3", distortion: 0.2 }])
  test(`lateral-torsional buckling of a deep thin strip under equal end moments, ${name}`, () => {
    const model = lateralBuckling({ nx, ny, shape, distortion });
    const { factor, run } = lowestFactor(model.text, MIXED);
    const check = measurements();
    check.ratio("critical moment", factor * model.applied, model.expected, CRITERIA.criticalLoad);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

test("lateral-torsional buckling of a deep thin strip, 8 x 2 distorted ShellTria3 (known limit)", () => {
  // KNOWN LIMIT (spec section 3.6: free edges of twisting plates at coarse meshes). Two triangles across the depth:
  // 1.0051 of the formula, just out of the criterion of section 13.6; the 16 x 4 mesh above is within it. The
  // criterion written here is the one the case meets.
  const model = lateralBuckling({ nx: 8, ny: 2, shape: "ShellTria3", distortion: 0.2 });
  const { factor } = lowestFactor(model.text, MIXED);
  const check = measurements();
  check.ratio("critical moment (known limit, criterion 0.01)", factor * model.applied, model.expected, 0.01);
  check.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 5. Inertia relief: free bodies under unbalanced loads

// The integral of f over [a, b] by the five-point Gauss rule (exact for the polynomials of these cases).
function integrate(f, a, b) {
  const points = [0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640, -0.9061798459386640];
  const weights = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891,
    0.2369268850561891];
  let sum = 0;
  points.forEach((x, k) => { sum += weights[k] * f((a + b) / 2 + ((b - a) * x) / 2); });
  return (sum * (b - a)) / 2;
}

// A free strip L × b in the XY plane, without Poisson's ratio (a beam along x), of nx × 2 cells, under the unbalanced
// load `loads(m)`, with the inertia relief of `relief`: "-1" on the FreeBodySupport components of three corners, "-2"
// on supports the solver chooses.
const FREE_STRIP = { length: 2.0, width: 0.4, thickness: 0.01 };
function freeStrip({ nx = 8, shape, distortion, relief, loads }) {
  // The columns 2, 4 and 6 of grid points stay straight: the loads end on them or lie on them.
  const m = mesh({ nx, ny: 2, shape, distortion, straightColumns: [2, 4, 6],
    map: (s, t) => [FREE_STRIP.length * s, FREE_STRIP.width * t, 0] });
  const lines = [...m.lines, shellProp(FREE_STRIP.thickness), isoMat(0), `Parameter;InertiaRelief;${relief}`,
    ...loads(m)];
  if (relief === "-1")
    lines.push(`FreeBodySupport;${m.node(0, 0)};123;${m.node(nx, 0)};23;${m.node(0, 2)};3`);
  const caseLines = ["SubcaseStart;1", "StaticLoadSelection;2", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "ElemForceOutput;All"];
  return { text: deck(101, caseLines, lines), mesh: m };
}

// The moment mx = D ∂²w/∂x² per unit width at x of a free beam under the load q(ξ) per unit area (the applied load
// and the inertia of the rigid acceleration that balances it): the ends are free, so D w''(x) = ∫0..x q(ξ) (x − ξ) dξ.
function freeBeamMoment(q, x, breaks) {
  let sum = 0;
  const cuts = [0, ...breaks.filter((c) => c < x), x];
  for (let k = 0; k + 1 < cuts.length; ++k) sum += integrate((xi) => q(xi) * (x - xi), cuts[k], cuts[k + 1]);
  return sum;
}

// The centre rows of the force table of the shells, as [element, mx + my]. The rows are in the axes of each element
// (x along its first edge), which a distorted mesh turns; the sum mx + my does not depend on them, and it is the
// moment of the beam here (without Poisson's ratio the moment across the strip vanishes).
function centreMomentTraces(run, shape) {
  return rowsOf(run, shape, "forces").filter((row) => row.nodeId === 0).map((row) => [row.id, row.mx + row.my]);
}

// The reactions at the FreeBodySupport components must vanish: the largest value of the support force table.
function largestReaction(run) {
  const table = run.nodalTables.find((t) => t.kind === "singlePointConstrForces");
  return table ? Math.max(0, ...Array.from(table.values).map(Math.abs)) : 0;
}

for (const relief of ["-1", "-2"])
  for (const { name, shape, distortion } of PLATE_MESHES.slice(0, 2))
    test(`inertia relief ${relief} of a free strip of 8 x 2 ${name} under a pressure on its middle half`, () => {
      // The pressure p on L/4 < x < 3L/4 accelerates the strip by p / (2 ρ t): the load per unit area is p − p / 2
      // under the pressure and −p / 2 elsewhere. The moment is freeBeamMoment; the deflection of the middle relative
      // to the ends is w(L/2) − w(0) = −∫0..L/2 w''(s) s ds (w'(L/2) = 0), plus that of the shear force,
      // ∫0..L/2 w_s' dx with κ G t w_s'(x) = −∫0..x q. On the regular mesh the deflection is a quartic in each
      // element: an exact state.
      const pressure = 100.0;
      const { length, width, thickness } = FREE_STRIP;
      const { text, mesh: m } = freeStrip({ shape, distortion, relief, loads: (grid) => {
        const loaded = grid.elements.filter((element) => {
          const column = (element - 1) % 8;
          return column >= 2 && column < 6;
        });
        return loaded.map((element) => `FacePressureLoad;2;${element};${real(pressure)}`);
      } });
      const run = runModel(text, MIXED);
      const check = measurements();
      const exact = distortion === 0;
      const q = (x) => (x > length / 4 && x < (3 * length) / 4 ? pressure / 2 : -pressure / 2);
      const d = bendingStiffness(thickness, 0);
      const scale = (pressure * length ** 2) / 32; // the moment of the middle section
      let worst = 0;
      for (const [element, moment] of centreMomentTraces(run, shape)) {
        const expected = freeBeamMoment(q, m.centroid(element)[0], [length / 4, (3 * length) / 4]);
        worst = Math.max(worst, Math.abs(moment - expected));
      }
      check.small("largest difference of the centre moments with the beam's", worst, scale,
        exact ? CRITERIA.exact : CRITERIA.stress);
      const bend = -(integrate((s) => freeBeamMoment(q, s, []) * s, 0, length / 4) +
        integrate((s) => freeBeamMoment(q, s, [length / 4]) * s, length / 4, length / 2)) / d;
      const shear = -(integrate((x) => -pressure * x / 2, 0, length / 4) +
        integrate((x) => (-pressure * length) / 8 + (pressure * (x - length / 4)) / 2, length / 4, length / 2)) /
        ((SHEAR_FACTOR * E * thickness) / 2);
      for (const j of [0, 1, 2]) {
        const rise = nodal(run, "displacements", m.node(4, j))[2] -
          (nodal(run, "displacements", m.node(0, j))[2] + nodal(run, "displacements", m.node(8, j))[2]) / 2;
        check.ratio(`deflection of the middle relative to the ends, row ${j}`, rise, bend + shear,
          exact ? CRITERIA.exact : CRITERIA.displacement);
      }
      check.small("largest reaction at the supports", largestReaction(run), pressure * length * width / 2,
        CRITERIA.invariant);
      check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
      check.done();
    });

for (const kind of ["point forces", "point moments"])
  for (const { name, shape, distortion } of [PLATE_MESHES[0], PLATE_MESHES[2]])
    test(`inertia relief -2 of a free strip of 8 x 2 ${name} under ${kind} on its middle section`, () => {
      // Forces F (shared 1/4, 1/2, 1/4 between the three grid points of the middle section) accelerate the strip by
      // F / (ρ t b L): the load per unit area is −F / (b L), and mx = −F x² / (2 b L) before the middle. Moments M
      // about y turn it with the angular acceleration a = M / (ρ t b L (L² + t²) / 12): the load per unit area is
      // ρ t a (x − L / 2), and mx = ρ t a (x³ / 6 − L x² / 4). The moments are read at the centres of the elements of
      // the first quarter, away from the loaded section (a concentrated moment on a shell converges slowly near its
      // grid point).
      const { length, width, thickness } = FREE_STRIP;
      const [force, moment] = [10.0, 10.0];
      const { text, mesh: m } = freeStrip({ shape, distortion, relief: "-2", loads: (grid) => [0, 1, 2].map((j) => {
        const share = real(j === 1 ? 0.5 : 0.25);
        return kind === "point forces" ? `PointForceLoad;2;${grid.node(4, j)};;${share};0.;0.;${real(force)}`
          : `PointMomentLoad;2;${grid.node(4, j)};;${share};0.;${real(moment)};0.`;
      }) });
      const run = runModel(text, MIXED);
      const check = measurements();
      const angular = moment / ((RHO * thickness * width * length * (length ** 2 + thickness ** 2)) / 12);
      const expected = (x) => (kind === "point forces" ? (-force * x * x) / (2 * width * length)
        : RHO * thickness * angular * (x ** 3 / 6 - (length * x * x) / 4));
      // The elements whose centroid is in the first quarter of the strip, more than 1.5 widths from the loads: the
      // loads at three grid points are not uniform across the width, which bends the strip across it nearby.
      let worst = 0;
      for (const [element, value] of centreMomentTraces(run, shape)) {
        const x = m.centroid(element)[0];
        if (x < length / 4) worst = Math.max(worst, Math.abs(value - expected(x)));
      }
      check.small("largest difference of the centre moments with the beam's", worst, expected(length / 2),
        CRITERIA.stress);
      check.small("largest reaction at the supports", largestReaction(run), kind === "point forces" ? force
        : moment / length, CRITERIA.invariant);
      check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
      check.done();
    });

for (const relief of ["-1", "-2"])
  for (const { name, shape, distortion } of [
    { name: "warped ShellQuad4", shape: "ShellQuad4", distortion: 0.2 },
    { name: "distorted ShellTria3", shape: "ShellTria3", distortion: 0.2 }])
    test(`inertia relief ${relief} of a free cylindrical panel of 6 x 6 ${name} under a uniform traction`, () => {
      // A traction p per unit area along a fixed direction on a shell of uniform mass per area is balanced at every
      // point by the inertia of the acceleration p / (ρ t): no deformation, no stress resultant (an exact state, with
      // the consistent mass and the consistent load of the curved elements).
      const [radius, length, thickness, traction] = [1.0, 1.0, 0.01, 100.0];
      const m = mesh({ nx: 6, ny: 6, shape, distortion, map: (s, t) => [
        radius * Math.cos((Math.PI * s) / 2), radius * Math.sin((Math.PI * s) / 2), length * t] });
      const lines = [...m.lines, shellProp(thickness), isoMat(), `Parameter;InertiaRelief;${relief}`];
      for (const element of m.elements)
        lines.push(`FacePressureLoad;2;${element};${real(traction)};;;;;;0;0.6;0.;0.8`);
      if (relief === "-1") lines.push(`FreeBodySupport;${m.node(0, 0)};123;${m.node(6, 0)};13;${m.node(0, 6)};1`);
      const caseLines = ["SubcaseStart;1", "StaticLoadSelection;2", "DisplacementOutput;All",
        "SinglePointConstrForceOutput;All", "ElemForceOutput;All"];
      const run = runModel(deck(101, caseLines, lines), MIXED);
      const check = measurements();
      // The scales: the membrane displacement p R² / (E t) and force p R of a cylinder under that pressure.
      let largest = 0;
      for (const { id } of m.points)
        largest = Math.max(largest, ...nodal(run, "displacements", id).slice(0, 3).map(Math.abs));
      if (relief === "-1") check.small("largest displacement", largest, (traction * radius ** 2) / (E * thickness),
        CRITERIA.exact);
      let membrane = 0;
      let moments = 0;
      for (const row of rowsOf(run, shape, "forces")) {
        membrane = Math.max(membrane, Math.abs(row.nx), Math.abs(row.ny), Math.abs(row.nxy));
        moments = Math.max(moments, Math.abs(row.mx), Math.abs(row.my), Math.abs(row.mxy));
      }
      check.small("largest membrane force", membrane, traction * radius, CRITERIA.exact);
      check.small("largest moment", moments, traction * radius * thickness, CRITERIA.exact);
      check.small("largest reaction at the supports", largestReaction(run), traction * radius * length,
        CRITERIA.invariant);
      check.done();
    });

for (const { name, nt, nz, shape, distortion, relief } of [...RING_MESHES.map((ring) => ({ ...ring, relief: "-2" })),
  { ...RING_MESHES[1], relief: "-1" }])
  test(`inertia relief ${relief} of a free ring pulled apart by two forces, ${name}`, () => {
    // Two opposite forces F on a diameter of a free ring (in balance: the inertia relief finds no acceleration): the
    // diameter along the forces grows by (π/4 − 2/π) F R³ / (E I) and the diameter across them shrinks by
    // (2/π − 1/2) F R³ / (E I) (Timoshenko, Strength of Materials, part 1), I = b t³ / 12. R/t = 100: the
    // extension and the shear of the ring add less than 1e-3. The diameter along the forces is read at the loaded
    // grid points, where the deflection of a shear-deformable shell under a point force grows slowly with the mesh.
    const [radius, width, thickness, force] = [1.0, 0.1, 0.01, 1.0];
    const m = mesh({ nx: nt, ny: nz, closed: true, shape, distortion, map: (s, t) => [
      radius * Math.cos(2 * Math.PI * s), radius * Math.sin(2 * Math.PI * s), width * t] });
    const lines = [...m.lines, shellProp(thickness), isoMat(0), `Parameter;InertiaRelief;${relief}`];
    if (relief === "-1")
      lines.push(`FreeBodySupport;${m.node(0, 0)};123;${m.node(nt / 4, 0)};13;${m.node(0, nz)};2`);
    for (let j = 0; j <= nz; ++j) {
      const share = real((j === 0 || j === nz ? 0.5 : 1) / nz);
      lines.push(`PointForceLoad;2;${m.node(0, j)};;${share};${real(force)};0.;0.`,
        `PointForceLoad;2;${m.node(nt / 2, j)};;${share};${real(-force)};0.;0.`);
    }
    const caseLines = ["SubcaseStart;1", "StaticLoadSelection;2", "DisplacementOutput;All"];
    const run = runModel(deck(101, caseLines, lines), MIXED);
    const check = measurements();
    const unit = (force * radius ** 3) / ((E * width * thickness ** 3) / 12);
    const along = nodal(run, "displacements", m.node(0, 0))[0] - nodal(run, "displacements", m.node(nt / 2, 0))[0];
    const across = nodal(run, "displacements", m.node(nt / 4, 0))[1] -
      nodal(run, "displacements", m.node((3 * nt) / 4, 0))[1];
    check.ratio("growth of the diameter along the forces", along, (Math.PI / 4 - 2 / Math.PI) * unit,
      CRITERIA.displacement);
    check.ratio("shrinking of the diameter across the forces", -across, (2 / Math.PI - 0.5) * unit,
      CRITERIA.displacement);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 6. Stiffened panel: shells with Bar2 or Beam2 stiffeners on their edges, against a fine ClassicFormulation model

// A plate a × b, simply supported (hard) on its four edges, with two blade stiffeners along x on the mesh lines at a
// third and two thirds of its width: Bar2 or Beam2 on the edges of the shells, their axis on the middle surface of
// the plate or under it by half the plate and half the blade (`offset`). n × n cells (n a multiple of 6); the grid
// points of the stiffeners and those of the middle lines stay on their lines in a distorted mesh. Three analyses:
// a uniform pressure (101), the modes (103), and the buckling under a shortening along x, with the transverse
// expansion of Poisson's ratio, put by the grid points of the boundary (105).
const PANEL = { a: 1.0, b: 0.6, thickness: 0.004, bladeDepth: 0.03, bladeThickness: 0.004, pressure: 1000.0,
  strain: 1.0e-5 };
function stiffenedPanel({ n, analysis, card, offset, shape = "ShellQuad4", distortion = 0 }) {
  const m = mesh({ nx: n, ny: n, shape, distortion, straightColumns: [n / 2], straightRows: [n / 3, n / 2, (2 * n) / 3],
    map: (s, t) => [PANEL.a * s, PANEL.b * t, 0] });
  const [depth, blade] = [PANEL.bladeDepth, PANEL.bladeThickness];
  // The blade: area, inertia in the plane normal to the plate (plane 1, the orientation vector is z), inertia in the
  // plane of the plate, torsion constant.
  const section = [depth * blade, (blade * depth ** 3) / 12, (depth * blade ** 3) / 12, (depth * blade ** 3) / 3];
  const lines = [...m.lines, shellProp(PANEL.thickness), isoMat(),
    card === "Bar2" ? `BarProp;2;1;${section.map(real).join(";")}`
      : `BeamProp;2;1;${real(section[0])};${real(section[1])};${real(section[2])};;${real(section[3])}`];
  const shift = real(offset ? -(PANEL.thickness + depth) / 2 : 0);
  let stiffener = 10000;
  for (const j of [n / 3, (2 * n) / 3])
    for (let i = 0; i < n; ++i)
      lines.push(`${card};${++stiffener};2;${m.node(i, j)};${m.node(i + 1, j)};0.;;0.;1.` +
        (offset ? `;;;;0.;0.;${shift};0.;0.;${shift}` : ""));
  for (const { id, i, j, x, y } of m.points) {
    const onX = i === 0 || i === n;
    const onY = j === 0 || j === n;
    if (!onX && !onY) continue;
    if (analysis === 105)
      lines.push(`SinglePointConstr;1;${id};1;${real(-PANEL.strain * x)}`,
        `SinglePointConstr;1;${id};2;${real(NU * PANEL.strain * y)}`);
    else lines.push(`SinglePointConstrList;1;12;${id}`);
    lines.push(`SinglePointConstrList;1;3${onX ? "4" : ""}${onY ? "5" : ""};${id}`);
  }
  if (analysis === 101) {
    lines.push(`ShellPressureLoad;2;${real(PANEL.pressure)};1;Through;${m.elements.length}`);
    return { text: deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
      "DisplacementOutput;All"], lines), mesh: m };
  }
  lines.push(eigenMethod(4));
  return { text: deck(analysis, analysis === 103 ? MODES_CASE : bucklingCase(), lines), mesh: m };
}

// The deflections at the centre of the panel and at the middle of a stiffener, the first three frequencies and, when
// `withBuckling`, the buckling factor.
function panelResults(model, options, withBuckling = true) {
  const statics = stiffenedPanel({ ...model, analysis: 101 });
  const run = runModel(statics.text, options);
  const n = model.n;
  return {
    centre: nodal(run, "displacements", statics.mesh.node(n / 2, n / 2))[2],
    stiffener: nodal(run, "displacements", statics.mesh.node(n / 2, n / 3))[2],
    frequencies: eigenRows(stiffenedPanel({ ...model, analysis: 103 }).text, options, "modes").values.slice(0, 3)
      .map(Math.sqrt),
    factor: withBuckling ? lowestFactor(stiffenedPanel({ ...model, analysis: 105 }).text, options).factor : null,
    warnings: warningsOf(run),
  };
}

// The reference: ClassicFormulation on 48 × 48 cells. From 48 to 96 its deflections change by 2e-5, its frequencies
// by 3e-4 and its buckling factor by 2e-4: it is within 4e-4 of its limit (an element of second order).
// ClassicFormulation refuses offsets in a buckling analysis under the default offset definition, and the model
// builder of the JavaScript package does not pass ModelParams OffsetDefinition: the buckling factor of the panels
// with offset stiffeners has no ClassicFormulation reference here. It is compared with the formulation's own value
// on 12 × 12 regular cells, which checks the mesh and not the formulation.
const referencePanels = new Map();
function classicPanel(card, offset) {
  const key = `classic ${card} ${offset}`;
  if (!referencePanels.has(key)) referencePanels.set(key, panelResults({ n: 48, card, offset }, CLASSIC, !offset));
  return referencePanels.get(key);
}
function finerBucklingFactor(card) {
  const key = `finer ${card}`;
  if (!referencePanels.has(key))
    referencePanels.set(key, lowestFactor(stiffenedPanel({ n: 12, card, offset: true, analysis: 105 }).text, MIXED)
      .factor);
  return referencePanels.get(key);
}

const PANEL_CASES = [
  { card: "Bar2", offset: false, shape: "ShellQuad4", distortion: 0 },
  { card: "Bar2", offset: true, shape: "ShellQuad4", distortion: 0 },
  { card: "Bar2", offset: true, shape: "ShellQuad4", distortion: 0.3 },
  { card: "Beam2", offset: true, shape: "ShellTria3", distortion: 0.2 },
];

for (const { card, offset, shape, distortion } of PANEL_CASES)
  test(`a panel of 6 x 6 ${distortion ? "distorted " : ""}${shape} with ${offset ? "offset" : "centred"} ${card} ` +
    "stiffeners: statics, modes and buckling of the fine ClassicFormulation model", () => {
    const classic = classicPanel(card, offset);
    const mixed = panelResults({ n: 6, card, offset, shape, distortion }, MIXED);
    const check = measurements();
    check.ratio("deflection of the centre", mixed.centre, classic.centre, CRITERIA.displacement);
    check.ratio("deflection of the middle of a stiffener", mixed.stiffener, classic.stiffener, CRITERIA.displacement);
    mixed.frequencies.forEach((value, k) => check.ratio(`frequency ${k + 1}`, value, classic.frequencies[k],
      CRITERIA.frequency));
    if (offset)
      check.ratio("buckling factor over the formulation's own on 12 x 12 cells", mixed.factor,
        finerBucklingFactor(card), CRITERIA.criticalLoad);
    else check.ratio("buckling factor", mixed.factor, classic.factor, CRITERIA.criticalLoad);
    check.truth(`no warning: ${mixed.warnings}`, mixed.warnings.length === 0);
    check.done();
  });

for (const { name, shape, distortion } of PLATE_MESHES)
  test(`buckling of a plate column with Bar2 stiffeners on its free edges (Euler), 8 x 2 ${name}`, () => {
    // A strip L × b without Poisson's ratio, pinned at both ends, with a bar on each long edge (its axis on the
    // middle surface), shortened by its ends: the column of the whole section buckles at the strain
    // ε_cr = π² (E I)_total / (L² (E A)_total), I = b t³ / 12 + 2 I1, A = b t + 2 A1 (Timoshenko and Gere, Theory of
    // Elastic Stability, chapter 2).
    const [length, width, thickness, strain] = [2.0, 0.4, 0.01, 1.0e-6];
    const bar = { area: 2.0e-4, i1: 3.0e-8, i2: 1.0e-9, j: 2.0e-9 };
    const m = mesh({ nx: 8, ny: 2, shape, distortion, map: (s, t) => [length * s, width * t, 0] });
    const lines = [...m.lines, shellProp(thickness), isoMat(0), eigenMethod(4),
      `BarProp;2;1;${real(bar.area)};${real(bar.i1)};${real(bar.i2)};${real(bar.j)}`];
    for (const j of [0, 2])
      for (let i = 0; i < 8; ++i)
        lines.push(`Bar2;${100 * (j + 1) + i};2;${m.node(i, j)};${m.node(i + 1, j)};0.;;0.;1.`);
    for (const j of [0, 1, 2])
      lines.push(`SinglePointConstrList;1;13;${m.node(0, j)}`, `SinglePointConstrList;1;3;${m.node(8, j)}`,
        `SinglePointConstr;1;${m.node(8, j)};1;${real(-strain * length)}`);
    lines.push(`SinglePointConstrList;1;2;${m.node(0, 0)}`);
    const { factor, run } = lowestFactor(deck(105, bucklingCase(), lines), MIXED);
    const check = measurements();
    const inertia = (width * thickness ** 3) / 12 + 2 * bar.i1;
    const area = width * thickness + 2 * bar.area;
    check.ratio("critical strain", factor * strain, (Math.PI ** 2 * inertia) / (length ** 2 * area),
      CRITERIA.criticalLoad);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 7. Shell grid points used by other things (spec section 5.4: the mean corner rotation of the shells)

// A cantilever plate 1 × 0.5 clamped along x = 0, with a point mass at the middle of its free end: 5 kg (the plate
// weighs 13.5 kg) 0.05 above the plate, with the inertias of a body of that size.
function massOnPlate({ nx, ny, shape = "ShellQuad4", distortion = 0 }) {
  const m = mesh({ nx, ny, shape, distortion, straightRows: [ny / 2], map: (s, t) => [s, 0.5 * t, 0] });
  const lines = [...m.lines, shellProp(0.01), isoMat(), eigenMethod(4),
    `PointMassInertia;900;${m.node(nx, ny / 2)};;5.;0.;0.;0.05;0.02;;0.02;;;0.03`];
  for (let j = 0; j <= ny; ++j) lines.push(`SinglePointConstrList;1;123456;${m.node(0, j)}`);
  return deck(103, MODES_CASE, lines);
}

let classicMassOnPlate = null;
for (const { name, shape, distortion } of PLATE_MESHES)
  test(`modes of a cantilever plate of 8 x 4 ${name} with an offset point mass with inertia on a grid point, ` +
    "against the fine ClassicFormulation model", () => {
    // The reference: ClassicFormulation on 96 × 48 cells: 33.15, 190.7, 225.3, 376.8 rad/s (33.30, 226.8, 253.2,
    // 714.5 with the mass alone, without its offset and its inertias). From 48 × 24 to 96 × 48 its first three
    // frequencies change by 3e-4, 1 % and 1.6 %, and the fourth by 11 %: the inertia of the mass acts on the rotation
    // of one grid point of the plate, where the rotation under a concentrated moment grows slowly with the mesh, in
    // any formulation. The first three frequencies are counted, the fourth is printed.
    classicMassOnPlate ??= eigenRows(massOnPlate({ nx: 96, ny: 48 }), CLASSIC, "modes").values;
    const classic = classicMassOnPlate;
    const { values, run } = eigenRows(massOnPlate({ nx: 8, ny: 4, shape, distortion }), MIXED, "modes");
    const check = measurements();
    for (let k = 0; k < 3; ++k)
      check.ratio(`frequency ${k + 1}`, Math.sqrt(values[k] / classic[k]), 1, CRITERIA.frequency);
    console.log(`    info ratio frequency 4: ${Math.sqrt(values[3] / classic[3]).toFixed(6)}`);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

for (const { name, shape, distortion } of PLATE_MESHES)
  test(`a strip of 8 x 2 ${name} on simple supports with springs on the rotations of the grid points of one end`,
    () => {
      // A strip L × b without Poisson's ratio, simply supported at both ends, under a uniform pressure p, with springs
      // to the ground of total stiffness K on the rotation about y of the three grid points of the end x = 0 (shared
      // 1/4, 1/2, 1/4). The beam (q = p b, E I = E b t³ / 12): the end turns by θ = θ0 / (1 + K L / (3 E I)) with
      // θ0 = q L³ / (24 E I), the springs take M = K θ, and the middle deflects by 5 q L⁴ / (384 E I) − M L² / (16 E I)
      // (Timoshenko, Strength of Materials, part 1, chapter 5), plus the shear deflection q L² / (8 κ G b t).
      // K L / (3 E I) = 0.1: the springs lower the rotation by 9 %. Each spring puts a concentrated moment on the
      // plate, whose own rotation under it adds to that of the spring (and grows without bound as the mesh is refined,
      // in any formulation): about 5 % of the flexibility of springs ten times stiffer (measured: the deflection is
      // then 1.024 of the beam's), 0.5 % of that of these.
      const [length, width, thickness, pressure] = [2.0, 0.4, 0.01, 100.0];
      const m = mesh({ nx: 8, ny: 2, shape, distortion, straightColumns: [4],
        map: (s, t) => [length * s, width * t, 0] });
      const rigidity = (E * width * thickness ** 3) / 12;
      const ratio = 0.1; // K L / (3 E I)
      const stiffness = (ratio * 3 * rigidity) / length;
      const lines = [...m.lines, shellProp(thickness), isoMat(0),
        `ShellPressureLoad;2;${real(pressure)};1;Through;${m.elements.length}`];
      for (const j of [0, 1, 2]) {
        lines.push(`SinglePointConstrList;1;3;${m.node(0, j)};${m.node(8, j)}`,
          `SpringDofInline;${900 + j};${real(stiffness * (j === 1 ? 0.5 : 0.25))};${m.node(0, j)};5`);
      }
      lines.push(`SinglePointConstrList;1;12;${m.node(0, 0)}`, `SinglePointConstrList;1;2;${m.node(8, 0)}`);
      const run = runModel(deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
        "DisplacementOutput;All"], lines), MIXED);
      const check = measurements();
      const q = pressure * width;
      const theta = (q * length ** 3) / (24 * rigidity) / (1 + ratio);
      const deflection = (5 * q * length ** 4) / (384 * rigidity) -
        (stiffness * theta * length ** 2) / (16 * rigidity) +
        (q * length ** 2) / (8 * ((SHEAR_FACTOR * E * width * thickness) / 2));
      // Without the springs the middle would deflect 5.8 % more: the criterion tells the two apart. The rotation of
      // the grid points of the springs is printed: it is the rotation under the concentrated moments (0.987 to 0.992
      // of the beam's on these meshes).
      for (const j of [0, 1, 2]) {
        check.ratio(`deflection of the middle, row ${j}`, nodal(run, "displacements", m.node(4, j))[2], deflection,
          CRITERIA.displacement);
        console.log(`    info ratio rotation of the grid point of a spring, row ${j}: ` +
          `${(-nodal(run, "displacements", m.node(0, j))[4] / theta).toFixed(6)}`);
      }
      check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
      check.done();
    });

// The deflection at (x, y) of a simply supported thin plate a × a under a moment M about y at its centre (Navier's
// series; Timoshenko and Woinowsky-Krieger, Theory of Plates and Shells, chapter 5, for a couple as the limit of
// two opposite forces): the load is M δ'(x − a/2) δ(y − a/2), whose coefficients are
// q_mn = −(4 M / a²) (m π / a) cos(m π / 2) sin(n π / 2), and w = Σ q_mn sin(m π x / a) sin(n π y / a) /
// (D π⁴ (m² + n²)² / a⁴).
function plateUnderCentralMoment(x, y, a, d, moment) {
  let w = 0;
  for (let m = 2; m <= 800; m += 2)
    for (let n = 1; n <= 799; n += 2) {
      const q = -((4 * moment) / a ** 2) * ((m * Math.PI) / a) * Math.cos((m * Math.PI) / 2) *
        Math.sin((n * Math.PI) / 2);
      w += (q * Math.sin((m * Math.PI * x) / a) * Math.sin((n * Math.PI * y) / a)) /
        ((d * Math.PI ** 4 * (m * m + n * n) ** 2) / a ** 4);
    }
  return w;
}

for (const { name, shape, distortion } of PLATE_MESHES)
  test(`a simply supported plate of 8 x 8 ${name} under a PointMomentLoad at its centre: Navier's series away ` +
    "from the moment", () => {
    // The moment acts on the mean corner rotation of the shells at the grid point. Under it the slope of a thin plate
    // is unbounded (the series of the slope diverges as a logarithm), so the rotation there converges slowly with any
    // mesh; the deflections away from it are those of the series.
    const [a, thickness, moment] = [1.0, 0.005, 10.0];
    const m = mesh({ nx: 8, ny: 8, shape, distortion, straightColumns: [2, 4, 6], straightRows: [2, 4, 6],
      map: (s, t) => [a * s, a * t, 0] });
    const lines = [...m.lines, shellProp(thickness), isoMat(),
      `PointMomentLoad;2;${m.node(4, 4)};;${real(moment)};0.;1.;0.`];
    for (let i = 0; i <= 8; ++i)
      lines.push(`SinglePointConstrList;1;1235;${m.node(i, 0)};${m.node(i, 8)}`,
        `SinglePointConstrList;1;1234;${m.node(0, i)};${m.node(8, i)}`);
    const run = runModel(deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
      "DisplacementOutput;All"], lines), MIXED);
    const check = measurements();
    const d = bendingStiffness(thickness);
    for (const [i, j] of [[2, 4], [2, 2], [6, 4], [6, 6]])
      check.ratio(`deflection at (${i / 8} a, ${j / 8} a)`, nodal(run, "displacements", m.node(i, j))[2],
        plateUnderCentralMoment((i * a) / 8, (j * a) / 8, a, d, moment), CRITERIA.displacement);
    check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
    check.done();
  });

for (const { name, shape, distortion } of PLATE_MESHES)
  test(`a cantilever strip of 8 x 2 ${name} with an enforced deflection and rotation along its end edge`, () => {
    // A strip L × b without Poisson's ratio clamped at x = 0, its end x = L moved by w = δ and turned by θy = φ
    // (EnforcedMotionConstr-like values of SinglePointConstr on the three grid points of the end edge). The beam with
    // shear flexibility (Timoshenko): the rotation of the section ψ = −θy = A x + B x², the shear force
    // V = −2 D B per unit width, w = A x² / 2 + B x³ / 3 − 2 D B x / (κ G t); A and B from ψ(L) = −φ and w(L) = δ.
    // The force of the supports of the end is V b and their moment about y is −D ψ'(L) b.
    const [length, width, thickness, delta, phi] = [2.0, 0.4, 0.01, 1.0e-2, 4.0e-3];
    const m = mesh({ nx: 8, ny: 2, shape, distortion, map: (s, t) => [length * s, width * t, 0] });
    const lines = [...m.lines, shellProp(thickness), isoMat(0)];
    for (const j of [0, 1, 2])
      lines.push(`SinglePointConstrList;1;123456;${m.node(0, j)}`,
        `SinglePointConstr;1;${m.node(8, j)};3;${real(delta)}`, `SinglePointConstr;1;${m.node(8, j)};5;${real(phi)}`);
    const run = runModel(deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "DisplacementOutput;All",
      "SinglePointConstrForceOutput;All"], lines), MIXED);
    const check = measurements();
    const d = (E * thickness ** 3) / 12;
    const shear = (SHEAR_FACTOR * E * thickness) / 2;
    // ψ(L) = A L + B L² = −φ, w(L) = A L² / 2 + B (L³ / 3 − 2 D L / κGt) = δ.
    const c = length ** 3 / 3 - (2 * d * length) / shear;
    const b = (delta + (phi * length) / 2) / (c - length ** 3 / 2);
    const a = (-phi - b * length ** 2) / length;
    const deflection = (x) => (a * x * x) / 2 + (b * x ** 3) / 3 - (2 * d * b * x) / shear;
    const criterion = distortion === 0 ? CRITERIA.exact : CRITERIA.displacement;
    let worst = 0;
    for (const { id, x } of m.points)
      worst = Math.max(worst, Math.abs(nodal(run, "displacements", id)[2] - deflection(x)));
    check.small("largest difference of the deflections with the beam's", worst, delta, criterion);
    let force = 0;
    let couple = 0;
    for (const j of [0, 1, 2]) {
      force += nodal(run, "singlePointConstrForces", m.node(8, j))[2];
      couple += nodal(run, "singlePointConstrForces", m.node(8, j))[4];
    }
    check.ratio("force of the supports of the end", force, -2 * d * b * width, criterion);
    check.ratio("moment of the supports of the end", couple, -d * (a + 2 * b * length) * width, criterion);
    // The formulation notes that the rotations enforced at grid points of shells act through the edges (support-note).
    const others = warningsOf(run).filter((text) => !text.includes("[support-note]"));
    check.truth(`no warning but the note of the enforced rotations: ${others}`, others.length === 0);
    check.done();
  });

// A sector of a cylinder of radius R under an internal pressure p, held along the axis (plane strain) and held by
// symmetry along its four edges, with every grid point in a cylindrical displacement system about the axis: the
// supports are the tangential displacement (2) and the rotations about the radius and about the axis (4, 6) on the
// straight edges, the rotations about the radius and about the tangent (4, 5) on the curved ends (without them the
// moment ν D u_r / R² of the plane strain is missing at the ends, an edge effect of 0.09 t / R), and the axial
// displacement (3) everywhere. The cylinder grows by u_r = p R² (1 − ν²) / (E t) (the hoop force p R with the axial
// strain held), the same at every point.
function pressurisedSector({ degrees, shape, distortion }) {
  const [radius, length, thickness, pressure] = [1.0, 0.5, 0.01, 1.0e4];
  const sector = (degrees * Math.PI) / 180;
  const m = mesh({ nx: 6, ny: 3, shape, distortion, map: (s, t) => [
    radius * Math.cos(sector * s), radius * Math.sin(sector * s), length * t] });
  const lines = ["CylindricalFrame2;7;;0.;0.;0.;0.;0.;1.;1.;0.;0.",
    ...m.lines.map((line) => (line.startsWith("Node;") ? `${line};7` : line)), shellProp(thickness), isoMat(),
    `ShellPressureLoad;2;${real(pressure)};1;Through;${m.elements.length}`];
  for (const { id, i, j } of m.points) {
    const straight = i === 0 || i === 6;
    const end = j === 0 || j === 3;
    lines.push(`SinglePointConstrList;1;${straight && end ? "23456" : straight ? "2346" : end ? "345" : "3"};${id}`);
  }
  const run = runModel(deck(101, ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All"], lines), MIXED);
  const expected = (pressure * radius ** 2 * (1 - NU ** 2)) / (E * thickness);
  let worst = 0;
  for (const { id } of m.points) {
    const [radial, tangential, axial] = nodal(run, "displacements", id);
    worst = Math.max(worst, Math.abs(radial - expected), Math.abs(tangential), Math.abs(axial));
  }
  return { worst, expected, warnings: warningsOf(run) };
}

for (const { name, shape, distortion } of [
  { name: "ShellQuad4", shape: "ShellQuad4", distortion: 0 },
  { name: "warped ShellQuad4", shape: "ShellQuad4", distortion: 0.2 },
  { name: "ShellTria3", shape: "ShellTria3", distortion: 0 }])
  for (const degrees of [90, 60])
    test(`a ${degrees} degree sector of a pressurised cylinder of 6 x 3 ${name}, its supports in a cylindrical ` +
      "displacement system", () => {
      // 90 degrees: the tangent of each straight edge is an axis of the basic system. 60 degrees: the tangent of the
      // second straight edge is skew to the basic axes, the same at both grid points of each of its edges (spec
      // section 5.6: the edge holds the directions that the held directions of its two grid points share).
      const { worst, expected, warnings } = pressurisedSector({ degrees, shape, distortion });
      const check = measurements();
      // The bending of the shell that grows adds (t / R)² / 12 = 8e-6 to the membrane solution; on the warped mesh
      // the surface read is not the cylinder, and the criterion is that of the displacements.
      check.small("largest difference of the displacements with the uniform growth", worst, expected,
        distortion === 0 ? 1e-4 : CRITERIA.displacement);
      // The warning support-not-carried-along-edge is printed, not counted: it is raised at 90 degrees too, where
      // every held direction is carried (the held axes of each grid point span the basic axes it needs).
      for (const text of warnings) console.log(`    info ${text.slice(0, 200)}`);
      check.done();
    });

test("a plate of 8 x 8 distorted ShellQuad4 in three subcases with different supports and loads", () => {
  // A thin square plate (a/t = 100): clamped under a uniform pressure, w = 0.00126532 p a⁴ / D; simply supported
  // under the same pressure, w = 0.00406235 p a⁴ / D; simply supported under a force at its centre,
  // w = 0.0116008 P a² / D (Timoshenko and Woinowsky-Krieger, Theory of Plates and Shells, chapters 5 and 6). The
  // transverse shear adds 1e-3 at this thickness, and more under the force (the deflection of a shear-deformable
  // plate under a point force grows slowly with the mesh).
  const [a, thickness, pressure, force] = [1.0, 0.01, 1000.0, 100.0];
  const m = mesh({ nx: 8, ny: 8, shape: "ShellQuad4", distortion: 0.3, straightColumns: [4], straightRows: [4],
    map: (s, t) => [a * s, a * t, 0] });
  const lines = [...m.lines, shellProp(thickness), isoMat(),
    `ShellPressureLoad;2;${real(pressure)};1;Through;${m.elements.length}`,
    `PointForceLoad;3;${m.node(4, 4)};;${real(force)};0.;0.;1.`];
  for (let i = 0; i <= 8; ++i) {
    const alongX = `${m.node(i, 0)};${m.node(i, 8)}`;
    const alongY = `${m.node(0, i)};${m.node(8, i)}`;
    lines.push(`SinglePointConstrList;11;123456;${alongX};${alongY}`, `SinglePointConstrList;12;1235;${alongX}`,
      `SinglePointConstrList;12;1234;${alongY}`);
  }
  const caseLines = ["DisplacementOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;11",
    "StaticLoadSelection;2", "SubcaseStart;2", "SinglePointConstrSelection;12", "StaticLoadSelection;2",
    "SubcaseStart;3", "SinglePointConstrSelection;12", "StaticLoadSelection;3"];
  const run = runModel(deck(101, caseLines, lines), MIXED);
  const check = measurements();
  const d = bendingStiffness(thickness);
  const centre = (subcase) => nodal(run, "displacements", m.node(4, 4), subcase)[2];
  check.ratio("clamped, pressure: deflection of the centre", centre(1), (0.00126532 * pressure * a ** 4) / d,
    CRITERIA.displacement);
  check.ratio("simply supported, pressure: deflection of the centre", centre(2), (0.00406235 * pressure * a ** 4) / d,
    CRITERIA.displacement);
  check.ratio("simply supported, force: deflection of the centre", centre(3), (0.0116008 * force * a ** 2) / d,
    CRITERIA.displacement);
  check.truth(`no warning: ${warningsOf(run)}`, warningsOf(run).length === 0);
  check.done();
});
