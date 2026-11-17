// The reading of the surface of the shells of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md,
// sections 4.4 and 10.5) and what its report says: a corner of the outline that no raw edge declares is rounded, with a
// warning that names its edges, and refused where the rounding leaves an element no angle; the kinks and roundings
// along edges that are not declared raw are warnings; nothing depends on the corner an element starts from.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

const MIXED = "MixedHybridFormulation";
// The results of a model must not depend on the corner its elements start from.
const INVARIANCE = 1e-8;

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

let deckCount = 0;

// Reads `text` as a model file, builds the model with the declarations `formulationGeometry` and runs `action(model)`.
function withModel(text, formulationGeometry, action) {
  const file = `/work/mixed-geometry-${++deckCount}.oos`;
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  try {
    const model = solver.modelFromReader(oor, deck, { elementFormulation: MIXED, formulationGeometry });
    try { return action(model); } finally { model.delete(); }
  } finally { deck.delete(); }
}

// The static run of `text`; an error diagnostic fails the case.
function solve(text, formulationGeometry) {
  return withModel(text, formulationGeometry, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return run;
  });
}

function table(run, kind) {
  const found = run.nodalTables.find((t) => t.kind === kind);
  assert.ok(found, `a ${kind} table`);
  return found;
}

// The sum of the support forces along z, with its sign changed: the pressure the plate carries.
function carried(run) {
  const forces = table(run, "singlePointConstrForces");
  let sum = 0;
  for (let row = 0; row < forces.ids.length; ++row) sum -= forces.values[6 * row + 2];
  return sum;
}

// The largest difference between two tables of the same grid points, over the largest value of the first.
function relativeDifference(first, second) {
  assert.deepEqual(Array.from(first.ids), Array.from(second.ids));
  let scale = 0, difference = 0;
  for (let i = 0; i < first.values.length; ++i) {
    assert.ok(Number.isFinite(first.values[i]) && Number.isFinite(second.values[i]), `value ${i} of ${first.kind}`);
    scale = Math.max(scale, Math.abs(first.values[i]));
    difference = Math.max(difference, Math.abs(first.values[i] - second.values[i]));
  }
  return difference / scale;
}

// The three orders of the grid points of a triangle that keep its orientation.
const turns = (grids) => [0, 1, 2].map((k) => [grids[k], grids[(k + 1) % 3], grids[(k + 2) % 3]]);

test("the curve that a mid-side grid point gives an outline edge of MixedHybridFormulation is not reported as a rounding", () => {
  // One ShellQuad8 on the unit square whose edge 2-3 bulges by 0.2 through its mid-side grid point: the parabola
  // leaves its chord by atan(4 × 0.2) = 38.7 degrees, as the model says.
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]",
    "Node;1;;0.;0.;0.", "Node;2;;1.;0.;0.", "Node;3;;1.;1.;0.", "Node;4;;0.;1.;0.",
    "Node;5;;0.5;0.;0.", "Node;6;;1.2;0.5;0.", "Node;7;;0.5;1.;0.", "Node;8;;0.;0.5;0.",
    "ShellQuad8;1;1;1;2;3;4;5;6;7;8", "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.",
    "SinglePointConstrList;1;123456;1;4", "PointForceLoad;2;2;;1.;0.;0.;1."];
  const run = solve(lines.join("\n") + "\n", {});
  assert.deepEqual(run.diagnostics.filter((d) => d.severity !== "information").map((d) => d.code), []);
  assert.doesNotMatch(run.diagnostics.find((d) => d.code === "geometry-report").text, /outline/);
});

// Two legs at 90 degrees along the x axis, one shell each: the leg z = 0 and the leg y = 0, clamped at x = 0, a force
// at the end of the fold. ShellQuad4, or ShellQuad8 with the mid-side grid points at the middle of their edges.
function angleSection(shape) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  const corners = { 1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 0.3, 0], 4: [0, 0.3, 0], 5: [1, 0, 0.3], 6: [0, 0, 0.3] };
  const node = (id, x) => lines.push(`Node;${id};;${x.map(real).join(";")}`);
  for (const [id, x] of Object.entries(corners)) node(id, x);
  const legs = [[1, 2, 3, 4], [1, 6, 5, 2]];
  const middles = new Map();
  let next = 10;
  legs.forEach((leg, e) => {
    const grids = [...leg];
    if (shape === "ShellQuad8")
      for (let k = 0; k < 4; ++k) {
        const [a, b] = [leg[k], leg[(k + 1) % 4]];
        const key = `${Math.min(a, b)}-${Math.max(a, b)}`;
        if (!middles.has(key)) {
          middles.set(key, ++next);
          node(next, corners[a].map((v, c) => 0.5 * (v + corners[b][c])));
        }
        grids.push(middles.get(key));
      }
    lines.push(`${shape};${e + 1};1;${grids.join(";")}`);
  });
  lines.push("ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.", "SinglePointConstrList;1;123456;1;4;6",
    "PointForceLoad;2;2;;1.;0.;0.;-1.");
  return lines.join("\n") + "\n";
}

test("the geometry report of MixedHybridFormulation warns of the rounding and of the kink of a fold that is not declared raw", () => {
  const named = (run) => run.diagnostics.filter((d) => d.code === "geometry-kink" || d.code === "geometry-rounding");
  // Not declared: the reading rounds the fold of the ShellQuad4, 45 degrees from the plane of each leg along it.
  const rounded = named(solve(angleSection("ShellQuad4"), {}));
  assert.ok(rounded.length >= 1, JSON.stringify(rounded));
  for (const d of rounded) assert.deepEqual([d.code, d.severity], ["geometry-rounding", "warning"], d.text);
  assert.match(rounded[0].text, /the surface read along the edge between grid points 1 and 2 turns by up to 4[0-9]\.\d+ degrees from the plane of the corners of its shells: a fold that the declarations do not name is rounded there/);
  assert.match(rounded[0].text, /declare raw the edges along the fold \(formulationGeometry\.rawEdges\) or the shells \(formulationGeometry\.rawElements\)/);
  // Declared raw: the kink of the fold is expected.
  const declared = named(solve(angleSection("ShellQuad4"), { rawEdges: [[1, 2]] }));
  assert.deepEqual(declared.map((d) => [d.code, d.severity]), [["geometry-kink", "information"]]);
  assert.match(declared[0].text, /differ by up to (89\.9|90\.0)\d* degrees \(declared raw\)$/);
  // Mid-side grid points make every edge of the ShellQuad8 straight: the legs are flat and the fold is a kink, which
  // the formulation takes as a fold although nothing declares it.
  const kinked = named(solve(angleSection("ShellQuad8"), {}));
  assert.deepEqual(kinked.map((d) => [d.code, d.severity]), [["geometry-kink", "warning"]], JSON.stringify(kinked));
  assert.match(kinked[0].text, /the normals of the 2 shells along the edge between grid points 1 and 2 differ by up to (89\.9|90\.0)\d* degrees and the edge is not declared raw/);
  assert.match(kinked[0].text, /declare the edge raw \(formulationGeometry\.rawEdges\)/);
  assert.match(kinked[0].text, /give the shells mid-side nodes, or declare the shells raw \(formulationGeometry\.rawElements/);
});

// A flat plate whose outline turns into the plate at grid point 7 (cells dx × dy), under a unit pressure, clamped at
// grid points 2 and 3:
//   10 - 11 - 12     the cell 2-3-7-6 is cut in two ShellTria3 along 2-7, written from the corners `first` and
//    |    |    |     `second`; the cells above are ShellQuad4. The outline turns between its edges 3-7 and 7-8, and
//    6 -  7 -  8     three elements have grid point 7: the reading takes the outline round the corner unless one of
//    |  / |          these edges is declared raw.
//    2 -  3
function cornerPlate({ dx = 1, dy = 1, first = [2, 3, 7], second = [2, 7, 6] } = {}) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "[model]"];
  const at = { 2: [1, 0], 3: [2, 0], 6: [1, 1], 7: [2, 1], 8: [3, 1], 10: [1, 2], 11: [2, 2], 12: [3, 2] };
  for (const [id, [i, j]] of Object.entries(at)) lines.push(`Node;${id};;${real(i * dx)};${real(j * dy)};0.`);
  lines.push(`ShellTria3;1;1;${first.join(";")}`, `ShellTria3;2;1;${second.join(";")}`, "ShellQuad4;3;1;6;7;11;10",
    "ShellQuad4;4;1;7;8;12;11", "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.",
    "SinglePointConstrList;1;123456;2;3", "ShellPressureLoad;2;1.;1;2;3;4");
  return lines.join("\n") + "\n";
}

const CORNER_EDGES = [[3, 7], [7, 8]];
// The two triangles of cornerPlate written from each of their three corners.
const TRIANGLE_STARTS = turns([2, 3, 7]).map((first, k) => [first, turns([2, 7, 6])[k]]);
const outlineRoundings = (run) =>
  run.diagnostics.filter((d) => d.code === "geometry-rounding" && /outline/.test(d.text));

test("a corner of the outline that the reading closes on a triangle of MixedHybridFormulation is refused, from every starting corner", () => {
  // Square cells: the outline edge 3-7, rounded, leaves grid point 7 along the diagonal 7-2 of the first triangle,
  // which has no angle left there. Its normal at that corner is 0 / 0: the results were NaN, or not, with the corner
  // the triangle started from.
  for (const [first, second] of TRIANGLE_STARTS)
    withModel(cornerPlate({ first, second }), {}, (model) => {
      // the run stops before the assembly, on the one problem of the first triangle
      let refusal = "";
      try { solver.run(model, "statics"); } catch (error) { refusal = error.message; }
      assert.match(refusal, /the model has 1 problem /, `${first} / ${second}`);
      assert.match(refusal, /error \[degenerate-geometry\] ShellTria3 1: the edges of the shell from grid point 7 /);
      assert.match(refusal, /are tangent to each other at grid point 7 as the mesh is read, or cross there/);
      assert.match(refusal, /declare raw \(formulationGeometry\.rawEdges\) the edges read curved at grid point 7: /);
      assert.match(refusal, /: the edge between grid points 3 and 7, the edge between grid points 7 and 8$/m);
    });
});

test("a corner of the outline of MixedHybridFormulation declared raw is kept: the plate carries its area, from every starting corner", () => {
  let reference = null;
  for (const [first, second] of TRIANGLE_STARTS) {
    const run = solve(cornerPlate({ first, second }), { rawEdges: CORNER_EDGES });
    assert.ok(Math.abs(carried(run) - 3) < 1e-9, `carried ${carried(run)}, area 3`);
    assert.deepEqual(run.diagnostics.filter((d) => d.severity !== "information"), []);
    reference ??= run;
    for (const kind of ["displacements", "singlePointConstrForces"]) {
      const difference = relativeDifference(table(reference, kind), table(run, kind));
      assert.ok(difference < INVARIANCE, `${kind} from ${first} / ${second}: ${difference}`);
    }
  }
});

test("a rounded corner of the outline of MixedHybridFormulation that leaves its shells an angle is solved with warnings that name its edges", () => {
  // Cells 2 × 1: the rounded edge 3-7 stays short of the diagonal 7-2. The plate is solved, without the area between
  // the two rounded edges and their chords (5.0365 of 6), and the report names the edge that leaves its chord most
  // (the other, at 26.57 degrees, is below the angle a smooth outline may reach).
  let reference = null;
  for (const [first, second] of TRIANGLE_STARTS) {
    const run = solve(cornerPlate({ dx: 2, first, second }), {});
    assert.ok(Math.abs(carried(run) - 5.036504591) < 1e-8, `carried ${carried(run)}`);
    const warnings = outlineRoundings(run);
    assert.deepEqual(warnings.map((d) => d.severity), ["warning"], JSON.stringify(run.diagnostics));
    const texts = warnings.map((d) => d.text).join(" | ");
    assert.match(texts, /along the edge between grid points 7 and 8 is read as a curve that leaves the chord of the edge by up to 63\.43\d+ degrees/);
    for (const warning of warnings)
      assert.match(warning.text, /declare raw the outline edges that meet there \(formulationGeometry\.rawEdges\)/);
    const report = run.diagnostics.find((d) => d.code === "geometry-report");
    assert.match(report.text, /the outline read leaves the chord of an edge by up to 63\.43\d+ degrees, along the edge between grid points 7 and 8/);
    assert.match(report.text, /0 kinks and 0 roundings along edges that the reading takes as smooth, and of 1 rounded edges of the outline/);
    reference ??= run;
    for (const kind of ["displacements", "singlePointConstrForces"]) {
      const difference = relativeDifference(table(reference, kind), table(run, kind));
      assert.ok(difference < INVARIANCE, `${kind} from ${first} / ${second}: ${difference}`);
    }
  }
  // Declared raw: the area, and nothing to name.
  const declared = solve(cornerPlate({ dx: 2 }), { rawEdges: CORNER_EDGES });
  assert.ok(Math.abs(carried(declared) - 6) < 1e-9, `carried ${carried(declared)}, area 6`);
  assert.deepEqual(declared.diagnostics.filter((d) => d.severity !== "information"), []);
  assert.doesNotMatch(declared.diagnostics.find((d) => d.code === "geometry-report").text, /outline/);
});
