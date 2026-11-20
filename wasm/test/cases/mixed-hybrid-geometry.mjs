// The reading of the surface of the shells of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md,
// sections 4.4 and 10.5) and what its report says: the fold angle of the model makes a fold sharp where nothing
// declares it; a corner of the outline that no declaration keeps is rounded, with a warning that names its edges, and
// refused where the rounding leaves an element no angle; the kinks and roundings along edges read smooth are warnings;
// nothing depends on the corner an element starts from.

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

test("the geometry report of MixedHybridFormulation says what a fold is read as: sharp by the fold angle or as declared, rounded or kinked when declared smooth", () => {
  const named = (run) => run.diagnostics.filter((d) => d.code === "geometry-kink" || d.code === "geometry-rounding");
  // Not declared: the fold angle of the model makes the fold sharp, and its kink is expected.
  for (const shape of ["ShellQuad4", "ShellQuad8"]) {
    const byAngle = named(solve(angleSection(shape), {}));
    assert.deepEqual(byAngle.map((d) => [d.code, d.severity]), [["geometry-kink", "information"]], shape);
    assert.match(byAngle[0].text, /differ by up to (89\.9|90\.0)\d* degrees \(sharp by the fold angle of the model\)$/);
  }
  // Declared sharp or raw: the same kink, with the declaration named.
  for (const [geometry, reason] of [[{ sharpEdges: [[1, 2]] }, "declared sharp"], [{ rawEdges: [[1, 2]] }, "declared raw"]]) {
    const declared = named(solve(angleSection("ShellQuad4"), geometry));
    assert.deepEqual(declared.map((d) => [d.code, d.severity]), [["geometry-kink", "information"]]);
    assert.match(declared[0].text, new RegExp(`differ by up to (89\\.9|90\\.0)\\d* degrees \\(${reason}\\)$`));
  }
  // Declared smooth: the reading rounds the fold of the ShellQuad4, 45 degrees from the plane of each leg along it.
  const rounded = named(solve(angleSection("ShellQuad4"), { smoothEdges: [[1, 2]] }));
  assert.ok(rounded.length >= 1, JSON.stringify(rounded));
  for (const d of rounded) assert.deepEqual([d.code, d.severity], ["geometry-rounding", "warning"], d.text);
  assert.match(rounded[0].text, /the surface read along the edge between grid points 1 and 2 turns by up to 4[0-9]\.\d+ degrees from the plane of the corners of its shells: a fold shallower than the fold angle of the model is rounded there/);
  assert.match(rounded[0].text, /declare sharp the edges along the fold \(formulationGeometry\.sharpEdges\), or raw when the fold is straight \(formulationGeometry\.rawEdges, or the shells: formulationGeometry\.rawElements\)/);
  // Mid-side grid points make every edge of the ShellQuad8 straight: declared smooth, the legs stay flat and the fold
  // is a kink, which the formulation takes as a fold whatever the declaration says.
  const kinked = named(solve(angleSection("ShellQuad8"), { smoothEdges: [[1, 2]] }));
  assert.deepEqual(kinked.map((d) => [d.code, d.severity]), [["geometry-kink", "warning"]], JSON.stringify(kinked));
  assert.match(kinked[0].text, /the normals of the 2 shells along the edge between grid points 1 and 2 differ by up to (89\.9|90\.0)\d* degrees and the edge is read smooth/);
  assert.match(kinked[0].text, /declare the edge sharp \(formulationGeometry\.sharpEdges\), or raw when the fold is straight \(formulationGeometry\.rawEdges\)/);
  assert.match(kinked[0].text, /give the shells mid-side nodes, or declare the shells raw \(formulationGeometry\.rawElements/);
});

test("the SurfaceEdge, SurfaceFacet and SurfaceLine cards of a deck declare the geometry of MixedHybridFormulation, and the program wins where both name an edge", () => {
  const named = (run) => run.diagnostics.filter((d) => d.code === "geometry-kink" || d.code === "geometry-rounding");
  const refusal = (text, geometry) => withModel(text, geometry, (model) => {
    try { solver.run(model, "statics"); } catch (error) { return error.message; }
    return "";
  });
  const angle = angleSection("ShellQuad4");
  // no declaration at all: refused, and the message names the cards and the parameter
  const missing = refusal(angle, undefined);
  assert.match(missing, /geometry-declarations-missing/);
  assert.match(missing, /SurfaceEdge, SurfaceFacet and SurfaceLine cards of the deck/);
  assert.match(missing, /the model parameter GeometryDeclarations with the value Given/);
  // the parameter alone is the empty declaration: the fold angle decides
  const prepared = named(solve(angle + "Parameter;GeometryDeclarations;Given\n", undefined));
  assert.match(prepared[0].text, /\(sharp by the fold angle of the model\)$/);
  // the cards: raw, smooth (the fold is rounded), sharp for one side
  assert.match(named(solve(angle + "SurfaceEdge;1;1;2;Raw\n", undefined))[0].text, /\(declared raw\)$/);
  assert.match(named(solve(angle + "SurfaceEdge;1;2;1;Sharp;2\n", undefined))[0].text, /\(declared sharp\)$/);
  const smooth = named(solve(angle + "SurfaceEdge;1;1;2;Smooth\n", undefined));
  assert.deepEqual([smooth[0].code, smooth[0].severity], ["geometry-rounding", "warning"]);
  // a facet: the four edges of the first shell are raw
  assert.match(solve(angle + "SurfaceFacet;1\n", undefined).diagnostics.find((d) => d.code === "geometry-report").text,
    /4 declared raw/);
  // the program wins on the edge it names: sharp, not the smooth of the deck, and no conflict
  const won = named(solve(angle + "SurfaceEdge;1;1;2;Smooth\n", { sharpEdges: [[1, 2]] }));
  assert.deepEqual(won.map((d) => [d.code, d.severity]), [["geometry-kink", "information"]]);
  assert.match(won[0].text, /\(declared sharp\)$/);
  // two cards of the deck that give one edge two states are refused, and so is a card that names no edge
  assert.match(refusal(angle + "SurfaceEdge;1;1;2;Smooth\nSurfaceEdge;2;2;1;Sharp\n", undefined),
    /geometry-declaration-conflict/);
  assert.match(refusal(angle + "SurfaceEdge;1;1;3;Sharp\n", undefined), /geometry-declaration-unknown/);
  assert.match(refusal(angle + "SurfaceLine;1;1;3\n", undefined), /geometry-declaration-unknown/);
  // the lines of the corner of the outline ended by two cards: the plate carries its area
  const ended = solve(cornerPlate() + "SurfaceLine;1;3;7\nSurfaceLine;2;8;7\n", undefined);
  assert.ok(Math.abs(carried(ended) - 3) < 1e-9, `carried ${carried(ended)}, area 3`);
  assert.deepEqual(ended.diagnostics.filter((d) => d.severity !== "information"), []);
  // a continuation in the deck: the outline goes on round the corner, as the reading takes it without a declaration
  const continued = refusal(cornerPlate() + "SurfaceLine;1;3;7;8\n", undefined);
  assert.match(continued, /degenerate-geometry/);
});

test("the table of the edges read by MixedHybridFormulation gives the state of every edge, where it comes from and its angles", () => {
  const edgesOf = (text, geometry, options) => {
    const file = `/work/mixed-geometry-edges-${++deckCount}.oos`;
    oor.writeFile(file, text);
    const deck = oor.readModel(file);
    try {
      const model = solver.modelFromReader(oor, deck, options ?? { elementFormulation: MIXED, formulationGeometry: geometry });
      try { return model.surfaceEdges(); } finally { model.delete(); }
    } finally { deck.delete(); }
  };
  const rowOf = (edges, a, b) => {
    for (let row = 0; row < edges.grid1.length; ++row)
      if (edges.grid1[row] === a && edges.grid2[row] === b)
        return { surface: edges.surface[row], state: edges.state[row], origin: edges.origin[row],
          angle: edges.angle[row], kink: edges.kink[row], rounding: edges.rounding[row] };
    return null;
  };
  const text = angleSection("ShellQuad4");
  // the fold 1-2 between the two legs, and the outline
  const byAngle = edgesOf(text, {});
  assert.equal(byAngle.grid1.length, 7, "seven edges");
  const fold = rowOf(byAngle, 1, 2);
  assert.deepEqual([fold.surface, fold.state, fold.origin], ["Shells", "Sharp", "Angle"]);
  assert.ok(Math.abs(fold.angle - 90) < 1e-6 && Math.abs(fold.kink - 90) < 1e-4 && fold.rounding < 1e-6, JSON.stringify(fold));
  const outline = rowOf(byAngle, 2, 3);
  assert.deepEqual([outline.state, outline.origin, outline.angle, outline.kink], ["Smooth", "Outline", 0, 0]);
  // declared: raw, sharp, and smooth, which rounds the fold by 45 degrees
  assert.deepEqual([rowOf(edgesOf(text, { rawEdges: [[1, 2]] }), 1, 2)].map((r) => [r.state, r.origin]), [["Raw", "Declared"]]);
  assert.deepEqual([rowOf(edgesOf(text, { sharpEdges: [[1, 2]] }), 1, 2)].map((r) => [r.state, r.origin]), [["Sharp", "Declared"]]);
  const smooth = rowOf(edgesOf(text, { smoothEdges: [[1, 2]] }), 1, 2);
  assert.deepEqual([smooth.state, smooth.origin], ["Smooth", "Declared"]);
  assert.ok(Math.abs(smooth.angle - 90) < 1e-6 && smooth.rounding > 40 && smooth.rounding < 50, JSON.stringify(smooth));
  // a raw element: its edges that nothing else declares
  const facet = edgesOf(text, { rawElements: [1] });
  assert.deepEqual([rowOf(facet, 1, 2)].map((r) => [r.state, r.origin]), [["Raw", "RawElement"]]);
  // without any declaration the model is refused by the analyses, and the table still shows what the angle decides
  assert.deepEqual([rowOf(edgesOf(text, undefined), 1, 2)].map((r) => [r.state, r.origin]), [["Sharp", "Angle"]]);
  // another formulation reads no surface from the mesh
  const classic = edgesOf(text, undefined, {});
  assert.deepEqual([classic.grid1.length, classic.state.length, classic.angle.length], [0, 0, 0]);
});

// A strip of `cells` ShellQuad4 round a quarter of a cylinder of radius 1 along x, clamped along one generator and
// pushed along the other; `angle`: the model's ShellNormalAngle, when it is given.
function quarterCylinder(cells, angle) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]"];
  if (angle !== undefined) lines.push(`Parameter;ShellNormalAngle;${real(angle)}`);
  const node = (i, j) => 1 + 2 * j + i;
  for (let j = 0; j <= cells; ++j) {
    const theta = (0.5 * Math.PI * j) / cells;
    for (const i of [0, 1]) lines.push(`Node;${node(i, j)};;${real(0.5 * i)};${Math.sin(theta)};${Math.cos(theta)}`);
  }
  for (let j = 0; j < cells; ++j)
    lines.push(`ShellQuad4;${j + 1};1;${node(0, j)};${node(1, j)};${node(1, j + 1)};${node(0, j + 1)}`);
  lines.push("ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.", `SinglePointConstrList;1;123456;${node(0, 0)};${node(1, 0)}`,
    `PointForceLoad;2;${node(0, cells)};;1.;0.;0.;-1.`);
  return { text: lines.join("\n") + "\n", generators: Array.from({ length: cells - 1 }, (_, k) => [node(0, k + 1), node(1, k + 1)]) };
}

test("a curved mesh of MixedHybridFormulation coarser than the fold angle is read as facets, with a warning for each of its edges, unless declared smooth or given a larger angle", () => {
  const byAngle = (run) => run.diagnostics.filter((d) => d.code === "geometry-fold-by-angle");
  const tip = (run, model) => {
    const displacements = table(run, "displacements");
    const row = Array.from(displacements.ids).indexOf(1 + 2 * model.cells);
    return displacements.values[6 * row + 2];
  };
  // three cells of 30 degrees: above the default angle of 20 degrees
  const coarse = { ...quarterCylinder(3), cells: 3 };
  const faceted = solve(coarse.text, {});
  const warnings = byAngle(faceted);
  assert.deepEqual(warnings.map((d) => d.severity), ["warning", "warning"], JSON.stringify(faceted.diagnostics));
  assert.match(warnings[0].text, /differ by 30\.0\d* degrees at a grid point, more than the fold angle of the model \(ShellNormalAngle\)/);
  assert.match(warnings[0].text, /declare the edge smooth \(formulationGeometry\.smoothEdges\) or refine the mesh/);
  assert.match(faceted.diagnostics.find((d) => d.code === "geometry-report").text,
    /2 sharp \(2 by the fold angle of the model alone\).*2 edges are sharp by the fold angle alone with an angle under 40\.0\d* degrees/);
  // declared smooth, or under a larger angle: the same round surface, and no warning
  const declared = solve(coarse.text, { smoothEdges: coarse.generators });
  const wider = solve(quarterCylinder(3, 45).text, {});
  for (const run of [declared, wider]) {
    assert.deepEqual(byAngle(run), []);
    assert.deepEqual(run.diagnostics.filter((d) => d.severity !== "information"), []);
  }
  assert.ok(Math.abs(tip(wider, coarse) / tip(declared, coarse) - 1) < 1e-12, "the angle or the declaration: one reading");
  assert.ok(Math.abs(tip(faceted, coarse) / tip(declared, coarse) - 1) > 1e-3, "facets are another structure");
  // declared sharp: the facets are what the model means, and nothing is warned
  const sharp = solve(coarse.text, { sharpEdges: coarse.generators });
  assert.deepEqual(byAngle(sharp), []);
  assert.ok(Math.abs(tip(sharp, coarse) / tip(faceted, coarse) - 1) < 1e-12, "declared sharp: the facets of the angle");
  // six cells of 15 degrees: under the angle, round without a declaration
  const fine = solve(quarterCylinder(6).text, {});
  assert.deepEqual(fine.diagnostics.filter((d) => d.severity !== "information"), []);
  assert.match(fine.diagnostics.find((d) => d.code === "geometry-report").text, /0 sharp/);
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

test("a corner of the outline of MixedHybridFormulation whose line is declared ended is kept, from every starting corner", () => {
  // The line of the outline edge 3-7 ends at grid point 7, and so does that of the edge 8-7: no edge is raw, and the
  // plate carries its area as with the two edges declared raw.
  const raw = solve(cornerPlate(), { rawEdges: CORNER_EDGES });
  for (const [first, second] of TRIANGLE_STARTS) {
    const run = solve(cornerPlate({ first, second }), { lineEnds: [[3, 7], [8, 7]] });
    assert.ok(Math.abs(carried(run) - 3) < 1e-9, `carried ${carried(run)}, area 3`);
    assert.deepEqual(run.diagnostics.filter((d) => d.severity !== "information"), []);
    assert.match(run.diagnostics.find((d) => d.code === "geometry-report").text, /0 declared raw/);
    for (const kind of ["displacements", "singlePointConstrForces"]) {
      const difference = relativeDifference(table(raw, kind), table(run, kind));
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

// ---------------------------------------------------------------------------------------------------------------------
// A mesh transition on a boundary that goes straight on (spec section 4.4): three quadrilaterals at a grid point of
// a straight outline or junction are not a re-entrant corner.

// A flat skin of 3 × 2 unit quadrilaterals in the plane z = 0 with a flat rib under its middle line (the plane
// y = 0, 1.2 deep), clamped at the end x = 3, under its weight. `transition`: the rib has one element across its
// depth at the clamped end and two at the free end, with three quadrilaterals at grid point 3 of the junction.
function ribbedSkin(transition) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", "ShellProp;1;1;0.01;1;;1",
    "IsoMat;1;7.e10;;0.3;2700.", "GravityLoad;2;;9.81;0.;0.;-1."];
  const node = (id, x, y, z) => lines.push(`Node;${id};;${real(x)};${real(y)};${real(z)}`);
  const quad = (id, corners) => lines.push(`ShellQuad4;${id};1;${corners.join(";")}`);
  // the skin: grid point 1 + 3 i + j at (i, j − 1, 0)
  const skin = (i, j) => 1 + 3 * i + j;
  for (let i = 0; i <= 3; ++i)
    for (let j = 0; j <= 2; ++j) node(skin(i, j), i, j - 1, 0);
  for (let i = 0; i < 3; ++i)
    for (let j = 0; j < 2; ++j) quad(1 + 2 * i + j, [skin(i, j), skin(i + 1, j), skin(i + 1, j + 1), skin(i, j + 1)]);
  const top = (i) => skin(i, 1);
  const held = [skin(3, 0), skin(3, 1), skin(3, 2)];
  if (transition) {
    node(20, 0, 0, -0.6); node(21, 0, 0, -1.2); node(22, 0.5, 0, -0.6); node(23, 0.5, 0, -1.2);
    node(24, 1, 0, -1.2); node(25, 2, 0, -1.2); node(26, 3, 0, -1.2);
    quad(11, [20, 22, top(1), top(0)]); quad(12, [21, 23, 22, 20]); quad(13, [top(1), 22, 23, 24]);
    quad(14, [24, 25, top(2), top(1)]); quad(15, [25, 26, top(3), top(2)]);
    held.push(26);
  } else {
    for (let i = 0; i <= 3; ++i) { node(20 + 2 * i, i, 0, -0.6); node(21 + 2 * i, i, 0, -1.2); }
    for (let i = 0; i < 3; ++i) {
      quad(11 + 2 * i, [20 + 2 * i, 22 + 2 * i, top(i + 1), top(i)]);
      quad(12 + 2 * i, [21 + 2 * i, 23 + 2 * i, 22 + 2 * i, 20 + 2 * i]);
    }
    held.push(26, 27);
  }
  for (const id of held) lines.push(`SinglePointConstrList;1;123456;${id}`);
  return lines.join("\n") + "\n";
}

test("a mesh transition of the rib on its junction with a skin of MixedHybridFormulation leaves the junction straight", () => {
  // Everything is plane: no kink, no rounding, and the skin answers as it does over the regular rib. Read as a
  // re-entrant corner of the rib, the three quadrilaterals at the junction bent it towards the mesh line inside the
  // rib (kinks of 27 degrees between coplanar shells of the skin, a deflection seven times too small).
  const deflection = (transition) => {
    const run = solve(ribbedSkin(transition), {});
    const warnings = run.diagnostics.filter((d) => d.severity === "warning");
    assert.deepEqual(warnings.map((d) => d.code), [], JSON.stringify(warnings));
    const displacements = table(run, "displacements");
    return displacements.values[6 * Array.from(displacements.ids).indexOf(1) + 2];
  };
  const [regular, withTransition] = [deflection(false), deflection(true)];
  assert.ok(Math.abs(withTransition / regular - 1) < 0.02,
    `corner deflection ${withTransition} with the transition, ${regular} over the regular rib`);
});

test("a re-entrant corner of three ShellQuad4 of MixedHybridFormulation beside a triangle of another surface is kept", () => {
  // An L of three unit quadrilaterals in the plane z = 0 (its re-entrant corner at grid point 5, the origin) with a
  // web of four triangles through the line y = 0. The side of the L has quadrilaterals only at grid point 5: it is read
  // as between quadrilaterals (each edge of the outline goes on into the L), whatever the web is made of, and the
  // L carries the pressure on its area of 3. Read by the rule of the grid points that have a triangle, the outline
  // went round the corner (a rounding of 42 degrees).
  const pressure = 100.0;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "SinglePointConstrForceOutput;All", "[model]",
    "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700."];
  // grid point 1 + 3 i + j at (i − 1, j − 1, 0); the quadrant x < 0, y > 0 has no element
  const id = (i, j) => 1 + 3 * i + j;
  for (let i = 0; i <= 2; ++i)
    for (let j = 0; j <= 2; ++j)
      if (!(i === 0 && j === 2)) lines.push(`Node;${id(i, j)};;${real(i - 1)};${real(j - 1)};0.`);
  let element = 0;
  for (const [i, j] of [[0, 0], [1, 0], [1, 1]]) {
    lines.push(`ShellQuad4;${++element};1;${id(i, j)};${id(i + 1, j)};${id(i + 1, j + 1)};${id(i, j + 1)}`,
      `ShellPressureLoad;2;${real(pressure)};${element}`);
  }
  // the web: two triangles under the line y = 0 and two above it, so that both edges of the L along it are junctions
  const [centre, before, after] = [id(1, 1), id(0, 1), id(2, 1)];
  lines.push("Node;20;;0.;0.;-1.", "Node;21;;0.;0.;1.", `ShellTria3;11;1;${centre};${after};20`,
    `ShellTria3;12;1;${centre};20;${before}`, `ShellTria3;13;1;${centre};${before};21`,
    `ShellTria3;14;1;${centre};21;${after}`);
  for (const j of [0, 1, 2]) lines.push(`SinglePointConstrList;1;123456;${id(2, j)}`);
  const run = solve(lines.join("\n") + "\n", {});
  const warnings = run.diagnostics.filter((d) => d.severity === "warning");
  assert.deepEqual(warnings.map((d) => d.code), [], JSON.stringify(warnings));
  assert.ok(Math.abs(carried(run) / (3 * pressure) - 1) < 1e-9, `the L carries ${carried(run)}`);
});
