// Validation of the fastener elements in the two element formulations (ClassicFormulation and
// MixedHybridFormulation): Fastener with FastenerProp, Weld with WeldProp, and the same joints written with a Bushing
// between two grid points tied to the skins by RigidInterpolation, through the whole path (model file, reader,
// modelFromReader, run). Every model is synthetic: two sheets a small gap apart, joined by the fasteners.
//
// What is checked, each case in both formulations:
//   1. one fastener between two stiff plates: the six stiffnesses, the force table, a rigid motion;
//   2. where the fastener pierces the shells (grid point, inside, edge, near a corner) and what it pierces
//      (ShellQuad4, ShellTria3, ShellQuad8, ShellTria6, flat and warped);
//   3. every way of locating a fastener and of giving its system, and the forms that are refused;
//   4. a lap joint with a row of fasteners against the one-dimensional model of two bars joined by springs;
//   5. two skins riveted together in bending, between two separate plates and the monolithic plate;
//   6. two concentric cylindrical skins under pressure;
//   7. a stiffened panel whose stringer flange is fastened to the skin;
//   8. the forms of the Weld card against the short beam of its diameter and material;
//   9. one joint written with a Fastener, a Weld and a Bushing;
//  10. modes, 11. buckling, 12. the streamed path and a uniform temperature rise;
//  13. a flange bolted to a skin along one of its edges, written with fasteners and with springs between the grid
//      points.
// The closed forms are given with each case. The criteria are the bounds each case meets, with a margin, and each
// case writes what it measures in the output of the test ("ratio ...", "small ..."). A case the solver does not meet
// keeps its criterion and is marked "todo" with the measured value.
//
// A point load on a shell is singular: the displacement in the plane under a point force and the rotation under a
// point moment grow without bound as the elements resolve them. So a Fastener, and a Weld of the forms PartPatch
// and ElemPatch, is tied to each shell over its cross-section: a square of the area of the circle of FastenerProp or
// WeldProp D (its side 0.886 D) about the piercing point, its sides along those of the pierced element. The
// ClassicFormulation reads the shell at the four corners of the square, the auxiliary points, each on its element:
// the piercing point takes their mean translation, the rotation of the shell about the axes of its plane (the
// rotations of the grid points, by the shape functions) and the rotation of the sheet in its plane (from the
// translations of the corners); the MixedHybridFormulation takes the mean of the motion of the shell over the
// square. Both follow the normal fibre of the shell, as a fastener through the thickness does; the rotation of the
// piercing point is not the slope of the sheet between its grid points (section 13). The joint then stops
// softening once the cells are as small as the fastener. On cells of the pitch of the fasteners the
// MixedHybridFormulation, whose elements resolve the sheet between the fasteners, gives what the
// ClassicFormulation gives on cells several times smaller (section 7). Where a case compares meshes, each
// formulation has the mesh stated with the case (the MixedHybridFormulation the coarser one: it costs more for each
// element).
//
// Run: node --test wasm/test/validation/fasteners.test.mjs   (Node.js 24; needs ../O_o_Reader/build/wasm/dist)

import assert from "node:assert/strict";
import { CLASSIC, MIXED, nodal, oor, real, runModel, solver, test, withModel } from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// Helpers

const FORMULATIONS = [{ name: "ClassicFormulation", options: CLASSIC }, { name: "MixedHybridFormulation",
  options: MIXED }];

// The side of the square that stands for a fastener of diameter 1 on the shells (the same area): √π / 2.
const FOOTPRINT_SIDE_PER_DIAMETER = Math.sqrt(Math.PI) / 2;

const add = (a, b) => a.map((value, i) => value + b[i]);
const sub = (a, b) => a.map((value, i) => value - b[i]);
const scale = (factor, a) => a.map((value) => factor * value);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => scale(1 / norm(a), a);

// The measurements of a case: each is written in the output of the test, and the case fails at its end when one is
// out of its criterion.
function measurements(t) {
  const failures = [];
  const note = (ok, text) => {
    t.diagnostic(`${ok ? "ok  " : "FAIL"} ${text}`);
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
    small(what, value, size, tolerance) {
      note(Math.abs(value) <= tolerance * Math.abs(size), `small ${what}: ${(Math.abs(value) / Math.abs(size))
        .toExponential(2)} of the scale (criterion ${tolerance})`);
    },
    // low ≤ value ≤ high.
    between(what, value, low, high) {
      note(value >= low && value <= high, `between ${what}: ${value.toPrecision(7)} in [${low.toPrecision(7)}, ` +
        `${high.toPrecision(7)}]`);
    },
    truth(what, ok) { note(Boolean(ok), what); },
    done() { assert.equal(failures.length, 0, failures.join("\n")); },
  };
}

// The cards of a model and the positions of its grid points.
class Deck {
  constructor() {
    this.cards = [];
    this.positions = new Map();
    this.selections = []; // case control lines before the subcases (the selection of a set of constraint equations)
  }

  node(id, position) {
    this.positions.set(id, position);
    this.cards.push(`Node;${id};;${position.map(real).join(";")}`);
    return id;
  }

  // A card: integers and words as they are, reals written by q().
  add(...fields) { this.cards.push(fields.join(";")); }

  // A force `force` (three components in the basic system) at grid point `id`, in the load set `set`.
  force(set, id, force) { this.add("PointForceLoad", set, id, "", real(1), ...force.map(real)); }

  // The supports `components` of the grid points `ids`, in the set 1.
  hold(components, ids) { this.add("SinglePointConstrList", 1, components, ...ids); }

  // The text of a static model: each subcase is [subcase, support set, load set or 0, temperature set or 0].
  statics(subcases = [[1, 1, 2, 0]]) {
    const lines = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
      "SinglePointConstrForceOutput;All", "AppliedLoadOutput;All", "ElemForceOutput;All", ...this.selections];
    for (const [subcase, supports, loads, temperatures] of subcases) {
      lines.push(`SubcaseStart;${subcase}`);
      if (supports) lines.push(`SinglePointConstrSelection;${supports}`);
      if (loads) lines.push(`StaticLoadSelection;${loads}`);
      if (temperatures) lines.push(`TemperatureSelection;${temperatures};TemperatureLoadPart`);
    }
    return [...lines, "[model]", ...this.cards, ""].join("\n");
  }

  // The text of a modal model of `count` modes, held by the support set `supports` (0: free).
  modes(count, supports = 0) {
    return ["[executive]", "SolutionSequence;103", "[case]", "SubcaseStart;1",
      ...(supports ? [`SinglePointConstrSelection;${supports}`] : []), "EigenMethodSelection;3", "[model]",
      ...this.cards, `LanczosEigenMethod;3;;;${count}`, ""].join("\n");
  }

  // The text of a buckling model: the preload of the load set 2 on the support set 1, then `count` factors.
  buckling(count) {
    return ["[executive]", "SolutionSequence;105", "[case]", "DisplacementOutput;All", "SubcaseStart;1",
      "SinglePointConstrSelection;1", "StaticLoadSelection;2", "SubcaseStart;2", "SinglePointConstrSelection;1",
      "EigenMethodSelection;3", "StaticSubcaseSelection;1", "[model]", ...this.cards,
      `LanczosEigenMethod;3;;;${count}`, ""].join("\n");
  }
}

const q = real;

// The shapes of the sheets: one quadrilateral per cell, or two triangles (cut along the diagonal 1-3 of the cell,
// or along alternating diagonals).
const QUAD4 = { name: "ShellQuad4", shape: "ShellQuad4" };
const TRIA3 = { name: "ShellTria3", shape: "ShellTria3", diagonal: "alternate" };
const QUAD8 = { name: "ShellQuad8", shape: "ShellQuad8" };
const TRIA6 = { name: "ShellTria6", shape: "ShellTria6", diagonal: "alternate" };

// A structured sheet of nx × ny cells on the surface `position(a, b)`, a and b in [0, 1], of the shape `shape` (above).
// The grid points take the IDs after `firstNode`, the elements the IDs after `firstElement`. Returns `node(i, j)`
// the corner grid point of column i and row j, `element(i, j)` the elements of a cell, `nodes` and `elements` every ID.
// The grid points of the middles of the sides (ShellQuad8, ShellTria6) are on the surface.
function sheet(deck, { nx, ny, position, shape = QUAD4, property = 1, firstNode = 0, firstElement = 0 }) {
  const card = shape.shape;
  const m = card === "ShellQuad8" || card === "ShellTria6" ? 2 : 1;
  // A ShellQuad8 has no grid point at the centre of its cell.
  const isUsed = (a, b) => card !== "ShellQuad8" || a % 2 === 0 || b % 2 === 0;
  const ids = new Map();
  let lastNode = firstNode;
  for (let b = 0; b <= m * ny; ++b)
    for (let a = 0; a <= m * nx; ++a)
      if (isUsed(a, b)) ids.set(`${a},${b}`, deck.node(++lastNode, position(a / (m * nx), b / (m * ny))));
  const at = (a, b) => ids.get(`${a},${b}`);
  const node = (i, j) => at(m * i, m * j);
  const cells = new Map();
  const elements = [];
  let lastElement = firstElement;
  const emit = (i, j, grids) => {
    deck.add(card, ++lastElement, property, ...grids);
    elements.push(lastElement);
    cells.set(`${i},${j}`, [...(cells.get(`${i},${j}`) ?? []), lastElement]);
  };
  for (let j = 0; j < ny; ++j)
    for (let i = 0; i < nx; ++i) {
      const cell = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      const corner = (k) => node(cell[k][0], cell[k][1]);
      const middle = (k, l) => at(cell[k][0] + cell[l][0], cell[k][1] + cell[l][1]);
      const triangle = (k, l, n) => m === 1 ? [corner(k), corner(l), corner(n)]
        : [corner(k), corner(l), corner(n), middle(k, l), middle(l, n), middle(n, k)];
      if (card === "ShellQuad4") emit(i, j, [0, 1, 2, 3].map(corner));
      else if (card === "ShellQuad8")
        emit(i, j, [...[0, 1, 2, 3].map(corner), middle(0, 1), middle(1, 2), middle(2, 3), middle(3, 0)]);
      else if (shape.diagonal === "13" || (shape.diagonal === "alternate" && (i + j) % 2 === 0)) {
        emit(i, j, triangle(0, 1, 2));
        emit(i, j, triangle(0, 2, 3));
      } else {
        emit(i, j, triangle(0, 1, 3));
        emit(i, j, triangle(1, 2, 3));
      }
    }
  return { node, nx, ny, nodes: [...ids.values()], elements, element: (i, j) => cells.get(`${i},${j}`),
    corners: [node(0, 0), node(nx, 0), node(nx, ny), node(0, ny)], lastNode, lastElement };
}

// The rows of the element table `kind` of the card `type` in `subcase`: a Map from the element ID to an object of its
// components.
function elementRows(run, type, kind = "forces", subcase = 1) {
  const table = run.elementTables.find((t) => t.elementType === type && t.kind === kind && t.subcase === subcase);
  assert.ok(table, `a ${kind} table of ${type} in subcase ${subcase}`);
  const width = table.components.length;
  const rows = new Map();
  table.ids.forEach((id, r) => {
    const row = {};
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    rows.set(id, row);
  });
  return rows;
}

// The six forces of a Fastener in its system: fx, fy, fz, mx, my, mz.
const FASTENER_FORCES = ["fx", "fy", "fz", "mx", "my", "mz"];
const fastenerForce = (rows, id) => FASTENER_FORCES.map((name) => rows.get(id)[name]);

// The resultant (three forces, three moments about the origin) of the nodal table `kind` of forces, the grid points
// placed by `deck` (no displacement system in these models).
function resultant(run, deck, kind, subcase = 1, only = null) {
  const table = run.nodalTables.find((t) => t.kind === kind && t.subcase === subcase);
  assert.ok(table, `a ${kind} table of subcase ${subcase}`);
  const total = [0, 0, 0, 0, 0, 0];
  table.ids.forEach((id, row) => {
    if (only && !only.has(id)) return;
    const force = Array.from(table.values.slice(row * 6, row * 6 + 3));
    const moment = add(cross(deck.positions.get(id), force), Array.from(table.values.slice(row * 6 + 3, row * 6 + 6)));
    for (let c = 0; c < 3; ++c) {
      total[c] += force[c];
      total[3 + c] += moment[c];
    }
  });
  return total;
}

// The axes of a fastener from A to B when its property gives no system (the rule of FastenerConnection.hpp): x from
// A to B, y the basic axis least aligned with x made perpendicular to it (the first of X, Y, Z wins a tie), z = x × y.
// Returns [x, y, z] in the basic system.
function defaultAxes(pointA, pointB) {
  const x = unit(sub(pointB, pointA));
  let least = 0;
  for (let k = 1; k < 3; ++k) if (Math.abs(x[k]) < Math.abs(x[least])) least = k;
  const basic = [0, 0, 0];
  basic[least] = 1;
  const y = unit(sub(basic, scale(x[least], x)));
  return [x, y, cross(x, y)];
}

// The components of `vector` (basic) on `axes`, and the vector (basic) of `components` on `axes`.
const toAxes = (axes, vector) => axes.map((axis) => dot(axis, vector));
const fromAxes = (axes, components) => axes.reduce((sum, axis, k) => add(sum, scale(components[k], axis)), [0, 0, 0]);

// The solution of the system a·x = b (Gauss elimination with pivoting; a few unknowns).
function solveLinear(a, b) {
  const n = b.length;
  const rows = a.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; ++i) {
    let pivot = i;
    for (let r = i + 1; r < n; ++r) if (Math.abs(rows[r][i]) > Math.abs(rows[pivot][i])) pivot = r;
    [rows[i], rows[pivot]] = [rows[pivot], rows[i]];
    for (let r = 0; r < n; ++r) {
      if (r === i) continue;
      const factor = rows[r][i] / rows[i][i];
      for (let c = i; c <= n; ++c) rows[r][c] -= factor * rows[i][c];
    }
  }
  return rows.map((row, i) => row[n] / row[i]);
}

// Forces at the grid points `ids` of `deck` whose resultant is the force `force` and the moment `moment` about the
// point `about`: f_i = a + b × r_i, r_i from the centroid c of the points, with n·a = force and J·b the moment about
// c, J = Σ (|r_i|²·I − r_i·r_iᵀ) (the forces of least norm).
function wrenchForces(deck, ids, force, moment, about) {
  const points = ids.map((id) => deck.positions.get(id));
  const centroid = scale(1 / points.length, points.reduce(add, [0, 0, 0]));
  const arms = points.map((point) => sub(point, centroid));
  const inertia = [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
    arms.reduce((sum, r) => sum + (i === j ? dot(r, r) : 0) - r[i] * r[j], 0)));
  const b = solveLinear(inertia, add(moment, cross(sub(about, centroid), force)));
  return arms.map((r) => add(scale(1 / ids.length, force), cross(b, r)));
}

// Those forces written in the load set `set`.
function applyWrench(deck, set, ids, force, moment, about) {
  wrenchForces(deck, ids, force, moment, about).forEach((f, i) => deck.force(set, ids[i], f));
}

// The largest difference between the translations of the grid points `ids` in `run` and the field `expected(x)`,
// over the largest expected translation.
function fieldError(run, deck, ids, expected, subcase = 1) {
  let worst = 0;
  let size = 0;
  for (const id of ids) {
    const wanted = expected(deck.positions.get(id));
    const found = nodal(run, "displacements", id, subcase).slice(0, 3);
    worst = Math.max(worst, norm(sub(found, wanted)));
    size = Math.max(size, norm(wanted));
  }
  return worst / size;
}

// A model that must be refused: returns the text of what the reader and the solver say of it (the diagnostics of the
// reader, the refusals of checkSupport, the error diagnostics of the run or the message of what it throws).
function refusalOf(text, options, analysis = "statics") {
  const file = "/work/fasteners-refused.oos";
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  const said = { reader: deck.diagnostics().map((d) => `${d.code}: ${d.message}`), solver: [] };
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try {
      said.solver.push(...model.checkSupport(analysis).refusals.map((d) => d.text));
      said.solver.push(...solver.run(model, analysis).diagnostics.filter((d) => d.severity === "error")
        .map((d) => d.text));
    } finally { model.delete(); }
  } catch (error) {
    said.solver.push(String(error.message));
  } finally { deck.delete(); }
  return said;
}

// ---------------------------------------------------------------------------------------------------------------------
// Two flat plates a gap apart

const ALUMINIUM = { young: 7.0e10, poisson: 0.3, density: 2700, expansion: 2.3e-5 };

// The stiffnesses of a rivet of some 4 mm through sheets of 2 mm, of the order of what is measured (N/m along the
// fastener and in the two shear directions, N·m/rad about them); the three shear and rotation values differ on
// purpose, so that a wrong axis shows.
const RIVET = { kt: [3.0e7, 1.2e7, 1.5e7], kr: [150, 80, 60] };

// Two square plates of side `side`, the lower one (property 1, grid points from 1, elements from 1) at Z = 0, the
// upper one (property 2, grid points from 1001, elements from 1001) at Z = gap, of nx × nx cells each. Returns the
// deck and the two sheets.
function twoPlates({ side = 0.12, gap = 0.004, nx = 3, thickness = 0.002, material = ALUMINIUM } = {}) {
  const deck = new Deck();
  const surface = (z) => (a, b) => [side * a, side * b, z];
  const lower = sheet(deck, { nx, ny: nx, position: surface(0), property: 1 });
  const upper = sheet(deck, { nx, ny: nx, position: surface(gap), property: 2, firstNode: 1000,
    firstElement: 1000 });
  deck.add("ShellProp", 1, 1, q(thickness), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(thickness), 1, "", 1);
  deck.add("IsoMat", 1, q(material.young), "", q(material.poisson), q(material.density), q(material.expansion));
  return { deck, lower, upper, side, gap };
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. One fastener between two stiff plates
//
// The lower plate is held at every grid point and the upper plate hangs on the fastener alone, so the fastener
// carries the whole load whatever the plates do: its forces are the resultant of the load at the middle p of A-B, in
// its system E (statics, exact). The plates are 10 mm of aluminium and the fastener is soft (KT of some 1e3 N/m, KR of
// some N·m/rad), so the upper plate moves as a rigid body on the six springs at p:
//     u(x) = E·(f/KT) + (E·(m/KR)) × (x − p),   f, m the force and the moment of the load at p in the system E.
// The plates are not rigid: their flexibility under the load of the fastener is the tolerance stated.

const SINGLE = { kt: [4000, 2000, 3000], kr: [3, 1, 2], location: [0.052, 0.066] };

// The joint of this section: the fastener 5001 at `location` through two plates of 3 × 3 cells, its grid point GS at
// the middle of the gap.
function singleJoint({ thickness = 0.01, mass = 0, kt = SINGLE.kt, kr = SINGLE.kr } = {}) {
  const joint = twoPlates({ thickness });
  const [x, y] = SINGLE.location;
  joint.middle = [x, y, joint.gap / 2];
  joint.axes = defaultAxes([x, y, 0], [x, y, joint.gap]);
  joint.deck.node(9001, joint.middle);
  joint.deck.add("Fastener", 5001, 7, "Property", 1, 2, 9001);
  joint.deck.add("FastenerProp", 7, q(0.004), "", "", ...kt.map(q), ...kr.map(q), q(mass));
  return joint;
}

for (const { name, options } of FORMULATIONS)
  test(`1. one fastener between two stiff plates: each stiffness in turn gives F/K (${name})`, (t) => {
    const joint = singleJoint();
    const { deck, lower, upper, middle, axes } = joint;
    deck.hold(123456, lower.nodes);
    // Six subcases: a force of 100 N along each axis of the fastener through p, then a moment of 1 N·m about each.
    const wrenches = [0, 1, 2, 3, 4, 5].map((c) => [0, 1, 2, 3, 4, 5].map((k) => (k === c ? (c < 3 ? 100 : 1) : 0)));
    wrenches.forEach((wrench, c) => applyWrench(deck, 11 + c, upper.corners, fromAxes(axes, wrench.slice(0, 3)),
      fromAxes(axes, wrench.slice(3)), middle));
    const run = runModel(deck.statics(wrenches.map((_, c) => [1 + c, 1, 11 + c, 0])), options);
    const m = measurements(t);
    const names = ["KT1", "KT2", "KT3", "KR1", "KR2", "KR3"];
    wrenches.forEach((wrench, c) => {
      const subcase = 1 + c;
      // The force table is the load (statics): the forces against the force of the case (or its moment over the
      // side of the plate), the moments against its moment (or its force times the side). Measured: classic 8e-10,
      // mixed 4e-7 (the plates are 1e5 to 1e6 times stiffer than the springs).
      const forces = fastenerForce(elementRows(run, "Fastener", "forces", subcase), 5001);
      const difference = forces.map((f, k) => Math.abs(f - wrench[k]));
      const [forceSize, momentSize] = c < 3 ? [wrench[c], wrench[c] * joint.side] : [wrench[c] / joint.side, wrench[c]];
      m.small(`${names[c]}: the fastener forces less the load`, Math.max(...difference.slice(0, 3)), forceSize, 1e-6);
      m.small(`${names[c]}: the fastener moments less the load`, Math.max(...difference.slice(3)), momentSize, 1e-6);
      // The upper plate moves as a rigid body on the springs. Measured: 3.1e-4 (classic) and 4.1e-4 (mixed) under
      // the force along the fastener, which bends the plates; 1.4e-4 at most for the others.
      const slide = fromAxes(axes, wrench.slice(0, 3).map((f, k) => f / SINGLE.kt[k]));
      const turn = fromAxes(axes, wrench.slice(3).map((f, k) => f / SINGLE.kr[k]));
      const error = fieldError(run, deck, upper.corners, (x) => add(slide, cross(turn, sub(x, middle))), subcase);
      m.small(`${names[c]}: the motion of the upper plate less F/K`, error, 1, 1e-3);
    });
    m.done();
  });

// A rigid motion of the whole joint: the grid points of the lower plate are given the translations of the motion
// u(x) = a + ω × x (their rotations are left to the plate), the upper plate hangs on the fastener. The upper plate
// follows and the fastener carries nothing. The forces are compared with those the same motion would give if the
// upper plate stayed behind, K·|u|. Plates of 2 mm and the stiffnesses of a rivet.
for (const { name, options } of FORMULATIONS)
  test(`1. one fastener between two plates: a rigid motion gives no fastener force (${name})`, (t) => {
    const { deck, lower, upper, middle } = singleJoint({ thickness: 0.002, kt: RIVET.kt, kr: RIVET.kr });
    const shift = [1.0e-3, -2.0e-3, 1.5e-3];
    const turn = [2.0e-3, 1.0e-3, -3.0e-3];
    const motion = (x) => add(shift, cross(turn, x));
    for (const id of lower.nodes)
      motion(deck.positions.get(id)).forEach((value, c) => deck.add("SinglePointConstr", 1, id, c + 1, q(value)));
    const run = runModel(deck.statics([[1, 1, 0, 0]]), options);
    const m = measurements(t);
    // Measured: classic 4e-12, mixed 3e-11.
    m.small("the motion of the upper plate less the rigid motion", fieldError(run, deck, upper.nodes, motion), 1,
      1e-8);
    const forces = fastenerForce(elementRows(run, "Fastener"), 5001);
    const carried = norm(motion(middle));
    // Measured: classic 5e-15 and 8e-11 (forces, moments), mixed 9e-13 and 4e-10.
    m.small("the fastener forces", Math.max(...forces.slice(0, 3).map(Math.abs)),
      Math.min(...RIVET.kt) * carried, 1e-8);
    m.small("the fastener moments", Math.max(...forces.slice(3).map(Math.abs)), Math.min(...RIVET.kr) * norm(turn),
      1e-8);
    m.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 2. Where the fastener pierces the shells
//
// Two plates of 2 mm, the lower one clamped on its outline, the upper one hanging on one rivet at the centre of the
// plates. The fastener does not move: the lines of the two meshes are shifted under it (x = a + s·4a(1 − a)/n in each
// direction), so that it pierces the plates at a grid point (s = 0), inside an element, on an edge or near a corner
// of an element. Four loads on the corners of the upper plate, each a pure force or moment at the middle p of the
// fastener: a shear force along X, a pull along the fastener, a moment about Y, a torque about the fastener. The
// compliance of the joint seen by the plates is the work of the load, Σ f_i·u_i, which holds the springs of the
// fastener (F²/K) and the flexibility of the two plates around it.
//
// The fastener system is given (MCID 0, MFLAG 1) so that the case does not depend on the default axes.
//
// What the meshes can tell: under the pull the compliance converges (ClassicFormulation ShellQuad4: 0.118 to 0.123
// at 8 cells a side, 0.124 to 0.125 at 16, 0.125 to 0.126 at 32; MixedHybridFormulation 0.1263 to 0.1269 at 4), and
// the spread across the piercing points is the discretisation error of the plates. Under the shear force, the moment
// and the torque a fastener tied at a point would have no converged value (a point force in the plane of a sheet and
// a point moment on a plate are logarithmically singular); tied over the square of its cross-section (3.5 mm for
// this fastener of 4 mm) it has one, which the plates reach when their cells are as small as the square. Until
// then the compliance inside an element grows some 10 % each time the cells are halved, and the joint must not
// change more than that with the place of the piercing point. Measured on ShellQuad4, at a grid point and inside an
// element, under the shear force (×10⁻³): ClassicFormulation on 8, 16, 32, 64 and 128 cells a side (cells of 15 to
// 0.9 mm) 1.39 and 1.23, 1.43 and 1.38, 1.44 and 1.49, 1.62 and 1.50, 1.68 and 1.59; MixedHybridFormulation on 4,
// 8, 16 and 32 cells 1.63 and 1.41, 1.68 and 1.56, 1.70 and 1.66, 1.73 and 1.72 (under the moment, ×10⁻³: 25.3 and
// 22.0, 25.4 and 24.4, 25.1 and 26.0, 28.2 and 25.7, 29.1 and 27.1; and 28.6 and 24.4, 29.0 and 26.6, 28.8 and
// 28.1, 29.0 and 28.4). On cells smaller than the square the compliance of the ClassicFormulation still grows by 4
// to 12 % at a halving: its four corners load the sheet at four points, a quarter of the load each, and the
// rotation of the shell is read from the few grid points around them. Criterion: the largest compliance over the
// smallest, across the four places, within 10 % under the pull and within PIERCING_CRITERIA under the other loads,
// the ClassicFormulation on 8 cells a side and the MixedHybridFormulation on 4 (where each resolves the bending of
// the plates under the pull).

const PIERCED_AT = [{ name: "at a grid point", shift: [0, 0] }, { name: "inside an element", shift: [0.35, 0.6] },
  { name: "on an edge", shift: [0.5, 0] }, { name: "near a corner", shift: [0.1, 0.1] }];
const PIERCING_LOADS = [{ name: "shear force", force: [100, 0, 0], moment: [0, 0, 0] },
  { name: "pull", force: [0, 0, 100], moment: [0, 0, 0] }, { name: "moment", force: [0, 0, 0], moment: [0, 1, 0] },
  { name: "torque", force: [0, 0, 0], moment: [0, 0, 1] }];
const PIERCED_SHAPES = [{ shape: QUAD4, warp: 0 }, { shape: QUAD4, warp: 0.2 }, { shape: TRIA3, warp: 0 },
  { shape: QUAD8, warp: 0 }, { shape: QUAD8, warp: 0.2 }, { shape: TRIA6, warp: 0 }];
const PIERCING_CELLS = { ClassicFormulation: 8, MixedHybridFormulation: 4 };
const PIERCING_SPREAD = 0.10;
// The criteria by load of PIERCING_LOADS (shear force, pull, moment, torque).
const PIERCING_CRITERIA = { ClassicFormulation: [0.15, PIERCING_SPREAD, 0.20, 0.05],
  MixedHybridFormulation: [0.20, PIERCING_SPREAD, 0.25, 0.05] };

// The compliances of the joint (one per load of PIERCING_LOADS) and its fastener forces, the fastener piercing the
// plates of `cells` cells a side at the place `shift` of a cell.
function piercedJoint({ shape, warp, shift, cells }, options) {
  const lines = (s) => (a) => a + (s / cells) * 4 * a * (1 - a);
  const deck = new Deck();
  const side = 0.12;
  const gap = 0.004;
  const surface = (z) => (a, b) => {
    const [x, y] = [side * lines(-shift[0])(a), side * lines(-shift[1])(b)];
    return [x, y, z + warp * (x - side / 2) * (y - side / 2) / side];
  };
  const lower = sheet(deck, { nx: cells, ny: cells, position: surface(0), shape, property: 1 });
  const upper = sheet(deck, { nx: cells, ny: cells, position: surface(gap), shape, property: 2, firstNode: 5000,
    firstElement: 5000 });
  deck.add("ShellProp", 1, 1, q(0.002), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(0.002), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density));
  const middle = [side / 2, side / 2, gap / 2];
  deck.add("Fastener", 9001, 7, "Property", 1, 2, "", "", "", ...middle.map(q));
  // In the basic system: KT along X, Y, Z are the shear stiffnesses 2 and 3 and the stiffness along the fastener.
  deck.add("FastenerProp", 7, q(0.004), 0, 1, q(RIVET.kt[1]), q(RIVET.kt[2]), q(RIVET.kt[0]), q(RIVET.kr[1]),
    q(RIVET.kr[2]), q(RIVET.kr[0]));
  deck.hold(123456, lower.nodes.filter((id) => {
    const [x, y] = deck.positions.get(id);
    return Math.min(x, y, side - x, side - y) < 1e-9;
  }));
  const loads = PIERCING_LOADS.map(({ force, moment }, c) => {
    const forces = wrenchForces(deck, upper.corners, force, moment, middle);
    forces.forEach((f, i) => deck.force(11 + c, upper.corners[i], f));
    return forces;
  });
  const run = runModel(deck.statics(loads.map((_, c) => [1 + c, 1, 11 + c, 0])), options);
  return {
    compliances: loads.map((forces, c) => forces.reduce((work, f, i) =>
      work + dot(f, nodal(run, "displacements", upper.corners[i], 1 + c).slice(0, 3)), 0)),
    forces: loads.map((_, c) => fastenerForce(elementRows(run, "Fastener", "forces", 1 + c), 9001)),
  };
}

// The joints of this section are solved once for the cases that read them.
const piercedResults = new Map();
function piercedCase(formulation, { shape, warp }) {
  const key = `${formulation.name} ${shape.name} ${warp}`;
  if (!piercedResults.has(key))
    piercedResults.set(key, PIERCED_AT.map(({ shift }) => piercedJoint({ shape, warp, shift,
      cells: PIERCING_CELLS[formulation.name] }, formulation.options)));
  return piercedResults.get(key);
}

const spreadOf = (values) => Math.max(...values) / Math.min(...values) - 1;
const piercedName = ({ shape, warp }) => `${warp ? "warped " : ""}${shape.name}`;

for (const formulation of FORMULATIONS)
  for (const pierced of PIERCED_SHAPES) {
    const label = `${piercedName(pierced)}, ${formulation.name}`;
    test(`2. a fastener that pierces ${label}: the forces balance the load and the pull does not see the place`,
      (t) => {
        const results = piercedCase(formulation, pierced);
        const m = measurements(t);
        // The fastener carries the load: in the basic system, the force and the moment of the case at p. Measured:
        // 5e-10 at most. A warped ShellQuad8 on the shifted lines is not the hyperbolic paraboloid exactly (the
        // surface is biquadratic in the parameters), so the middle of the fastener is up to 1.1e-6 m from the point
        // p the loads are written about: the moment of the shear force differs by that arm (9e-6 of F times the
        // side, measured), and the criterion of the moments is 1e-4 there.
        const curved = pierced.warp !== 0 && pierced.shape === QUAD8;
        PIERCED_AT.forEach((place, k) => PIERCING_LOADS.forEach(({ name, force, moment }, c) => {
          const wanted = [...force, ...moment];
          const size = c < 2 ? 100 : 1 / 0.12;
          const difference = results[k].forces[c].map((f, i) => Math.abs(f - wanted[i]) / (i < 3 ? size : size * 0.12));
          m.small(`${place.name}, ${name}: the fastener forces less the load`, Math.max(...difference.slice(0, 3)), 1,
            1e-7);
          m.small(`${place.name}, ${name}: the fastener moments less the load`, Math.max(...difference.slice(3)), 1,
            curved ? 1e-4 : 1e-7);
        }));
        // Measured: classic 0.040 (ShellQuad4), 0.039 (warped), 0.039 (ShellTria3), 0.002 (ShellQuad8, warped
        // ShellQuad8 and ShellTria6); mixed 0.006 at most.
        const pulls = results.map((result) => result.compliances[1]);
        t.diagnostic(`compliances under the pull: ${pulls.map((value) => value.toPrecision(5)).join(", ")}`);
        m.small("the spread of the compliance under the pull across the piercing points", spreadOf(pulls), 1,
          PIERCING_SPREAD);
        m.done();
      });
    test(`2. a fastener that pierces ${label}: the shear, moment and torque compliances do not see the place`,
      (t) => {
        const results = piercedCase(formulation, pierced);
        const m = measurements(t);
        // Measured, mixed (4 cells a side, a square of 3.5 mm in cells of 30 mm): 0.13 to 0.15 (shear force), 0.18
        // to 0.20 (moment), 0.02 to 0.03 (torque), the largest at a grid point, where four elements meet. The
        // spread narrows with the mesh (ShellQuad4, shear force: 0.15, 0.08, 0.02 and 0.02 on 4, 8, 16 and 32
        // cells), where the shell read at the point alone gave 0.57, 0.70 and 0.88 (the compliance at a grid point
        // growing by a fifth at each refinement).
        // Measured, classic: 0.11 to 0.14 (shear force) and 0.12 to 0.16 (moment) for the ShellQuad4 and the
        // ShellTria3, 0.02 to 0.06 and 0.03 to 0.07 for the ShellQuad8 and the ShellTria6; 0.018 (torque) at most.
        // The largest at a grid point, where the rotation of the shell is that of one grid point, on which a
        // moment bends the sheet more than on the four of an element.
        for (const c of [0, 2, 3]) {
          const values = results.map((result) => result.compliances[c]);
          t.diagnostic(`compliances under the ${PIERCING_LOADS[c].name}: ${values.map((value) =>
            value.toPrecision(5)).join(", ")}`);
          m.small(`the spread of the compliance under the ${PIERCING_LOADS[c].name} across the piercing points`,
            spreadOf(values), 1, PIERCING_CRITERIA[formulation.name][c]);
        }
        m.done();
      });
  }

// The two formulations on the same joints, the fastener inside an element (each formulation on its mesh). The pull
// converges and must agree; the others are recorded against the 10 % that halving the cells changes.
for (const pierced of PIERCED_SHAPES)
  test(`2. a fastener that pierces ${piercedName(pierced)} inside an element: the two formulations agree`, (t) => {
    const [classic, mixed] = FORMULATIONS.map((formulation) => piercedCase(formulation, pierced)[1].compliances);
    const m = measurements(t);
    // Measured: pull 1.00 to 1.08 (the classic plates of 8 cells are still up to 8 % too stiff under a point load),
    // shear force 0.99 to 1.18, moment 0.97 to 1.13, torque 0.997 to 1.005.
    PIERCING_LOADS.forEach(({ name }, c) => m.ratio(`the compliance under the ${name}, mixed over classic`, mixed[c],
      classic[c], c === 1 ? 0.10 : 0.20));
    m.done();
  });

// A fastener at a grid point of a joint that has the symmetries of a square.
//
// Two square plates of 4 × 4 cells, the lower one clamped on its outline, joined at their centre, a grid point of
// both, by a rivet whose two shear stiffnesses are the same; the upper plate is lifted by four equal forces at its
// corners. The joint has the symmetries of the square: the four corners lift alike.
for (const { name, options } of FORMULATIONS) {
  // ClassicFormulation: the four corners of the square of the fastener lie in the four elements around the grid
  // point, and the corners of the plate lift by 1.1642 mm each. Tied to the grid points of one of these elements,
  // the slope of the plate that lifts was taken for a rotation: 0.29, 1.22, 1.22 and 2.15 mm.
  // MixedHybridFormulation: the square of the fastener lies in the four elements around the grid point, a point of
  // it on a line between two of them read on both, and the corners lift by 1.2688 mm each (read in one of them:
  // 1.14, 1.27, 1.27 and 1.39 mm).
  test(`2. a fastener at a grid point of both plates of a symmetric joint: the four corners lift alike (${name})`,
    (t) => {
      const { deck, lower, upper, side, gap } = twoPlates({ nx: 4, gap: 0.002 });
      deck.add("Fastener", 5001, 7, "Property", 1, 2, "", "", "", q(side / 2), q(side / 2), q(gap / 2));
      deck.add("FastenerProp", 7, q(0.004), 0, 1, q(1.2e7), q(1.2e7), q(3.0e7), q(80), q(80), q(150));
      deck.hold(123456, lower.nodes.filter((id) => {
        const [x, y] = deck.positions.get(id);
        return Math.min(x, y, side - x, side - y) < 1e-9;
      }));
      for (const id of upper.corners) deck.force(2, id, [0, 0, 25]);
      const run = runModel(deck.statics(), options);
      const lifts = upper.corners.map((id) => nodal(run, "displacements", id)[2]);
      t.diagnostic(`the corners lift by ${lifts.map((value) => (1000 * value).toFixed(4)).join(", ")} mm`);
      const m = measurements(t);
      m.small("the largest lift over the smallest, less 1", Math.max(...lifts) / Math.min(...lifts) - 1, 1, 1e-6);
      m.done();
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Every way of locating a fastener, and its system
//
// One joint: two plates of 2 mm, the lower one clamped along X = 0, the upper one pulled and lifted at its far corners,
// the fastener at (0.052, 0.066) with the stiffnesses of a rivet. The piercing points are the same whatever the card
// says of them (the projection of the location on the lower plate, then the normal to the upper plate), so the
// results are the same to the rounding of the solution.

const LOCATED = [0.052, 0.066];

// The joint of this section, the fastener written by `write(deck, pierced)` (`pierced`: the elements the fastener
// pierces in the lower and in the upper plate) and its property by `property` (the fields after the diameter).
function locatedJoint(write, property = ["", "", ...RIVET.kt.map(q), ...RIVET.kr.map(q)]) {
  const joint = twoPlates();
  const { deck, lower, upper, gap } = joint;
  write(deck, [lower.element(1, 1)[0], upper.element(1, 1)[0]], { lower, upper, gap });
  deck.add("FastenerProp", 7, q(0.004), ...property);
  deck.hold(123456, [0, 1, 2, 3].map((j) => lower.node(0, j)));
  for (const j of [0, 3]) deck.force(2, upper.node(3, j), [25, 12.5, 6.25]);
  return joint;
}

const [X0, Y0] = LOCATED;
const GAP = 0.004;
const LOCATIONS = [
  { name: "by property, GS at the middle of the gap", write: (deck) => {
    deck.node(9001, [X0, Y0, GAP / 2]);
    deck.add("Fastener", 5001, 7, "Property", 1, 2, 9001);
  } },
  { name: "by property, GS on the lower plate", write: (deck) => {
    deck.node(9001, [X0, Y0, 0]);
    deck.add("Fastener", 5001, 7, "Property", 1, 2, 9001);
  } },
  { name: "by property, XS YS ZS", write: (deck) =>
    deck.add("Fastener", 5001, 7, "Property", 1, 2, "", "", "", q(X0), q(Y0), q(GAP / 2)) },
  { name: "by element, GS", write: (deck, pierced) => {
    deck.node(9001, [X0, Y0, GAP / 2]);
    deck.add("Fastener", 5001, 7, "Elements", ...pierced, 9001);
  } },
  { name: "by element, XS YS ZS", write: (deck, pierced) =>
    deck.add("Fastener", 5001, 7, "Elements", ...pierced, "", "", "", q(X0), q(Y0), q(GAP / 2)) },
  { name: "by element, the elements named being neighbours of the pierced ones", write: (deck, pierced, { lower,
    upper }) => deck.add("Fastener", 5001, 7, "Elements", lower.element(0, 1)[0], upper.element(2, 1)[0], "", "", "",
    q(X0), q(Y0), q(GAP / 2)) },
  { name: "by property, GA and GB given", write: (deck) => {
    deck.node(9002, [X0, Y0, 0]);
    deck.node(9003, [X0, Y0, GAP]);
    deck.add("Fastener", 5001, 7, "Property", 1, 2, "", 9002, 9003);
  } },
  { name: "by element, GA and GB given", write: (deck, pierced) => {
    deck.node(9002, [X0, Y0, 0]);
    deck.node(9003, [X0, Y0, GAP]);
    deck.add("Fastener", 5001, 7, "Elements", ...pierced, "", 9002, 9003);
  } },
  { name: "by property, GS, GA and GB given", write: (deck) => {
    deck.node(9001, [X0, Y0, GAP / 2]);
    deck.node(9002, [X0, Y0, 0]);
    deck.node(9003, [X0, Y0, GAP]);
    deck.add("Fastener", 5001, 7, "Property", 1, 2, 9001, 9002, 9003);
  } },
];

// The results a joint is compared on: the translations of the far corners of the upper plate and the fastener forces.
function jointResults(joint, options) {
  const run = runModel(joint.deck.statics(), options);
  return { run, motion: [0, 3].flatMap((j) => nodal(run, "displacements", joint.upper.node(3, j)).slice(0, 3)),
    forces: fastenerForce(elementRows(run, "Fastener"), 5001) };
}

// The largest difference between two lists of values over the largest value of the first.
const worstDifference = (first, second) =>
  Math.max(...first.map((value, i) => Math.abs(value - second[i]))) / Math.max(...first.map(Math.abs));

for (const { name, options } of FORMULATIONS)
  test(`3. every way of locating a fastener gives the same joint (${name})`, (t) => {
    const m = measurements(t);
    const reference = jointResults(locatedJoint(LOCATIONS[0].write), options);
    // The load at the middle of the fastener, in its system (x along Z, y along X, z along Y): statics.
    const load = [12.5, 50, 25];
    m.small("the fastener forces less the load", Math.max(...reference.forces.slice(0, 3).map((f, k) =>
      Math.abs(f - load[k]))), 50, 1e-8);
    for (const location of LOCATIONS.slice(1)) {
      const joint = locatedJoint(location.write);
      const results = jointResults(joint, options);
      // Measured: 0 to 8e-13 in both formulations.
      m.small(`${location.name}: the motion of the upper plate less that of the first form`,
        worstDifference(reference.motion, results.motion), 1, 1e-9);
      m.small(`${location.name}: the fastener forces less those of the first form`,
        worstDifference(reference.forces, results.forces), 1, 1e-9);
    }
    m.done();
  });

// GA and GB given are grid points of the model: they move with the plates they pierce, and the relative motion of B
// and A at the middle p of the fastener, in its system, is the force over the stiffness:
//     E^T·((u_B + θ_B × (p − x_B)) − (u_A + θ_A × (p − x_A))) = f/KT,   E^T·(θ_B − θ_A) = m/KR.
// The plates are flexible here (2 mm): the relation holds whatever they do.
for (const { name, options } of FORMULATIONS)
  test(`3. the relative motion of the given GA and GB is the fastener force over its stiffness (${name})`, (t) => {
    const joint = locatedJoint(LOCATIONS[6].write);
    const { run, forces } = jointResults(joint, options);
    const axes = defaultAxes([X0, Y0, 0], [X0, Y0, GAP]);
    const [a, b] = [9002, 9003].map((id) => nodal(run, "displacements", id));
    const half = [0, 0, GAP / 2];
    const slide = toAxes(axes, sub(add(b.slice(0, 3), cross(b.slice(3), scale(-1, half))),
      add(a.slice(0, 3), cross(a.slice(3), half))));
    const turn = toAxes(axes, sub(b.slice(3), a.slice(3)));
    const m = measurements(t);
    // Measured: 1e-9 at most in both formulations.
    forces.slice(0, 3).forEach((f, k) => m.ratio(`the slide along axis ${k + 1} over f/KT`, slide[k],
      f / RIVET.kt[k], 1e-6));
    forces.slice(3).forEach((f, k) => m.ratio(`the turn about axis ${k + 1} over m/KR`, turn[k], f / RIVET.kr[k],
      1e-6));
    m.done();
  });

// The fastener system: the default (x from A to B, here Z; y the basic axis least aligned with it, X; z = x × y, Y)
// against the same axes given by a coordinate system with MFLAG = 1, and against the basic system (MCID 0, MFLAG 1)
// with the stiffnesses in its order. The forces are given in the system of the property.
for (const { name, options } of FORMULATIONS)
  test(`3. the fastener system given by MCID with MFLAG 1 gives the joint of the default axes (${name})`, (t) => {
    const m = measurements(t);
    const write = LOCATIONS[0].write;
    const reference = jointResults(locatedJoint(write), options);
    // The system 31: origin, a point on its z-axis (basic Y), a point in its x-z plane (basic Z): x = Z, y = X, z = Y.
    const sameAxes = locatedJoint((deck, ...rest) => {
      write(deck, ...rest);
      deck.add("RectFrame2", 31, "", q(0), q(0), q(0), q(0), q(1), q(0), q(0), q(0), q(1));
    }, [31, 1, ...RIVET.kt.map(q), ...RIVET.kr.map(q)]);
    const same = jointResults(sameAxes, options);
    m.small("MCID 31 of the default axes: the motion of the upper plate", worstDifference(reference.motion,
      same.motion), 1, 1e-9);
    m.small("MCID 31 of the default axes: the fastener forces", worstDifference(reference.forces, same.forces), 1,
      1e-9);
    // The basic system: KT1 along X, KT2 along Y, KT3 along Z, so the stiffnesses of the axes 2, 3, 1 of the default.
    const turned = (values) => [values[1], values[2], values[0]];
    const basic = jointResults(locatedJoint(write, [0, 1, ...turned(RIVET.kt).map(q), ...turned(RIVET.kr).map(q)]),
      options);
    m.small("MCID 0 with the stiffnesses in its order: the motion of the upper plate",
      worstDifference(reference.motion, basic.motion), 1, 1e-9);
    m.small("MCID 0 with the stiffnesses in its order: the fastener forces", worstDifference(
      [...turned(reference.forces.slice(0, 3)), ...turned(reference.forces.slice(3))], basic.forces), 1, 1e-9);
    m.done();
  });

// The forms that are refused, each with what the diagnostic must say. The reader refuses a card without a location;
// the solver the rest, before the assembly.
const REFUSED = [
  { name: "MCID 0 with MFLAG 0 (a relative fastener system)", property: [0, 0, ...RIVET.kt.map(q),
    ...RIVET.kr.map(q)], code: "unsupported-field", says: /relative fastener system is not supported/ },
  { name: "a cylindrical MCID", frame: ["CylindricalFrame2", 32, "", q(0), q(0), q(0), q(0), q(0), q(1), q(1), q(0),
    q(0)], property: [32, 1, ...RIVET.kt.map(q), ...RIVET.kr.map(q)], code: "unsupported-field",
    says: /expected a rectangular coordinate system/ },
  { name: "an MCID the model does not define", property: [99, 1, ...RIVET.kt.map(q), ...RIVET.kr.map(q)],
    code: "missing-coordinate-system", says: /found 99/ },
  { name: "a thermal expansion of the fastener", property: ["", "", ...RIVET.kt.map(q), ...RIVET.kr.map(q), q(0),
    q(0), q(1.0e-5)], code: "unsupported-field", says: /thermal expansion of a fastener is not supported/ },
  { name: "a coincidence length", property: ["", "", ...RIVET.kt.map(q), ...RIVET.kr.map(q), q(0), q(0), q(0), q(0),
    q(0.001)], code: "unsupported-field", says: /coincidence length of a fastener is not supported/ },
  { name: "a property that is not a FastenerProp", fastener: ["Fastener", 5001, 8, "Property", 1, 2, 9001],
    code: "missing-property", says: /expected a FastenerProp for property 8/ },
  { name: "a location off the plates", location: [0.4, Y0, GAP / 2], code: "missing-point",
    says: /no piercing point on side A.*falls on none of the shells of property 1/ },
  { name: "the two sides on the same plate", fastener: ["Fastener", 5001, 7, "Property", 1, 1, 9001],
    code: "degenerate-geometry", says: /expected distinct piercing points A and B/ },
  { name: "an element that is not a shell of the model", fastener: ["Fastener", 5001, 7, "Elements", 777, 888, 9001],
    code: "missing-point", says: /element 777 is not a shell element/ },
];

for (const { name, options } of FORMULATIONS)
  test(`3. the fastener forms that are not supported are refused with their reason (${name})`, (t) => {
    const m = measurements(t);
    for (const refused of REFUSED) {
      const joint = locatedJoint((deck) => {
        deck.node(9001, refused.location ?? [X0, Y0, GAP / 2]);
        deck.add(...(refused.fastener ?? ["Fastener", 5001, 7, "Property", 1, 2, 9001]));
        if (refused.frame) deck.add(...refused.frame);
      }, refused.property);
      const said = refusalOf(joint.deck.statics(), options).solver.join("\n");
      m.truth(`${refused.name}: refused with [${refused.code}] and its reason`,
        said.includes(`[${refused.code}]`) && refused.says.test(said) && /Fastener 5001|FastenerProp 7/.test(said));
    }
    m.done();
  });

// A Fastener without a location (no GS, no GA, no XS YS ZS; GB alone is not a location) is refused by the reader, which
// names the fields; the solver then refuses the model.
const UNLOCATED = [["without GS, GA and XS YS ZS", ["Fastener", 5001, 7, "Property", 1, 2]],
  ["with GB alone", ["Fastener", 5001, 7, "Property", 1, 2, "", "", 1022]]];
for (const [form, card] of UNLOCATED) {
  test(`3. a Fastener ${form} is refused by the reader, which names the location fields`, (t) => {
    const said = refusalOf(locatedJoint((deck) => deck.add(...card)).deck.statics(), CLASSIC);
    const m = measurements(t);
    m.truth(`the reader: ${said.reader.join(" | ")}`, said.reader.some((text) =>
      /card\.missing-field/.test(text) && /the location is required: spotNode, nodeA or locationX/.test(text)));
    m.truth("the solver refuses the model", said.solver.length > 0);
    m.done();
  });
  // The solver does not have the card (the reader did not read it): its refusal says so and sends to the
  // diagnostics of the reader, which name the card and its fields.
  test(`3. a Fastener ${form}: the diagnostic of the solver sends to those of the reader`, (t) => {
    const said = refusalOf(locatedJoint((deck) => deck.add(...card)).deck.statics(), CLASSIC);
    t.diagnostic(`the solver: ${said.solver.join(" | ").slice(0, 400)}`);
    assert.ok(said.solver.some((text) => /not read as a card/.test(text) && /deck\.diagnostics\(\)/.test(text)),
      "the diagnostic of the solver says that a record was not read and where its reason is");
  });
}

// The default axes of a fastener along a basic axis. The rule (FastenerConnection.hpp): y is the basic axis least
// aligned with x, the first of X, Y, Z winning a tie. A fastener between two plates normal to Z has x along Z and a
// tie between X and Y, so y is X and z is Y wherever the fastener is and however it is located. The upper plate is
// pulled by 100 N along X through the middle of the fastener: fy = 100 and fz = 0 (statics).
const AXIS_CASES = [
  { name: "GS at (0.052, 0.066)", at: [0.052, 0.066], form: "GS" },
  { name: "GS at (0.1001, 0.0333)", at: [0.1001, 0.0333], form: "GS" },
  { name: "GA alone at (0.052, 0.066)", at: [0.052, 0.066], form: "GA" },
  { name: "GA alone at (0.07, 0.05)", at: [0.07, 0.05], form: "GA" },
];

function axisJoint({ at, form }) {
  const joint = twoPlates();
  const { deck, lower, upper, gap } = joint;
  deck.node(9001, [at[0], at[1], form === "GS" ? gap / 2 : 0]);
  deck.add("Fastener", 5001, 7, "Property", 1, 2, ...(form === "GS" ? [9001] : ["", 9001]));
  deck.add("FastenerProp", 7, q(0.004), "", "", ...RIVET.kt.map(q), ...RIVET.kr.map(q));
  deck.hold(123456, lower.nodes);
  applyWrench(deck, 2, upper.corners, [100, 0, 0], [0, 0, 0], [at[0], at[1], gap / 2]);
  return joint;
}

// (Two components of x within FASTENER_AXIS_TIE of each other are a tie: decided by their round-off, three of these
// four fasteners had y along Y.)
test("3. the default axes of a fastener along Z are the same wherever it is: y along X", (t) => {
  const m = measurements(t);
  for (const axisCase of AXIS_CASES) {
    const forces = fastenerForce(elementRows(runModel(axisJoint(axisCase).deck.statics(), CLASSIC), "Fastener"), 5001);
    m.truth(`${axisCase.name}: fy ${forces[1].toFixed(3)}, fz ${forces[2].toFixed(3)} for 100 and 0`,
      Math.abs(forces[1] - 100) < 1e-6 && Math.abs(forces[2]) < 1e-6);
  }
  m.done();
});

// GA given alone (B is where the normal of the lower plate at GA crosses the upper plate): the joint of the other
// forms.
for (const { name, options } of FORMULATIONS)
  test(`3. a fastener located by GA alone gives the same joint (${name})`, (t) => {
    const m = measurements(t);
    const reference = jointResults(locatedJoint(LOCATIONS[0].write), options);
    const results = jointResults(locatedJoint((deck) => {
      deck.node(9002, [X0, Y0, 0]);
      deck.add("Fastener", 5001, 7, "Property", 1, 2, "", 9002);
    }), options);
    m.small("the motion of the upper plate less that of the first form", worstDifference(reference.motion,
      results.motion), 1, 1e-9);
    m.small("the fastener forces less those of the first form", worstDifference(reference.forces, results.forces), 1,
      1e-9);
    m.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. A lap joint with a row of fasteners
//
// Two strips of width w and thickness t, the lower one clamped at X = 0, the upper one pulled by P at its far end,
// joined over their overlap by n fasteners a pitch p apart. With ν = 0, the strips one cell wide and every motion but
// the one along X held, each strip is a bar of stiffness EA = E·w·t and the joint is the one-dimensional model of
// two bars joined by springs (the classical model of the load transfer of a multi-row joint: the end fasteners carry
// more than the others, and the more so the stiffer the fasteners). Its unknowns are the displacements a_i and b_i
// of the two bars at the fastener i, i = 1..n:
//     lower bar:  (EA/c)·a_1·[i = 1] + (EA/p)·(2a_i − a_(i−1) − a_(i+1)) − K·(b_i − a_i) = 0,
//     upper bar:  (EA/p)·(2b_i − b_(i−1) − b_(i+1)) + K·(b_i − a_i) = P·[i = n],
// c the length of the lower bar from the clamp to the first fastener (the terms of a bar that is not there left
// out), and the load of the fastener i is F_i = K·(b_i − a_i), Σ F_i = P. Solved here by Gauss elimination.

// The shares F_i/P of the n fasteners of stiffness `stiffness` in the one-dimensional model above.
function lapShares({ n, barStiffness, leadStiffness, stiffness }) {
  const size = 2 * n;
  const matrix = Array.from({ length: size }, () => new Array(size).fill(0));
  const load = new Array(size).fill(0);
  const spring = (i, j, k) => {
    matrix[i][i] += k;
    matrix[j][j] += k;
    matrix[i][j] -= k;
    matrix[j][i] -= k;
  };
  matrix[0][0] += leadStiffness;
  for (let i = 0; i < n; ++i) {
    if (i + 1 < n) {
      spring(i, i + 1, barStiffness);
      spring(n + i, n + i + 1, barStiffness);
    }
    spring(i, n + i, stiffness);
  }
  load[size - 1] = 1;
  const u = solveLinear(matrix, load);
  return Array.from({ length: n }, (_, i) => stiffness * (u[n + i] - u[i]));
}

// The shares of the same joint in the ClassicFormulation, which ties each fastener to its strip over its footprint:
// the bars cut as the strips (cells of half a pitch: a bar element of stiffness 2·EA/p between two lines of the
// mesh), and each fastener reading its bar, and loading it, at the two points a distance d either side of its line,
// half each (d the half side of the square of the fastener; across the width the square changes nothing: a strip
// moves alike over its width). A bar element gives u(±d) = (1 − r)·u_0 + r·u_(±1), r = d over the length of a cell,
// so the fastener reads (1 − r)·u_0 + (r/2)·(u_(−1) + u_(+1)). With d = 0 it is the model above.
function lapSharesOverFootprint({ n, barStiffness, stiffness, reach }) {
  const cells = 2 + 2 * (n - 1) + 1;
  const lines = cells + 1; // the lines of the mesh of a strip: the unknowns of the lower bar, then of the upper one
  const size = 2 * lines;
  const matrix = Array.from({ length: size }, () => new Array(size).fill(0));
  const load = new Array(size).fill(0);
  const spring = (i, j, k) => {
    matrix[i][i] += k;
    matrix[j][j] += k;
    matrix[i][j] -= k;
    matrix[j][i] -= k;
  };
  for (let k = 0; k < cells; ++k) {
    spring(k, k + 1, 2 * barStiffness);
    spring(lines + k, lines + k + 1, 2 * barStiffness);
  }
  // The fastener i is on the line 2 + 2i of the lower strip and 1 + 2i of the upper one, which starts a cell later.
  const readings = Array.from({ length: n }, (_, i) => {
    const reading = new Array(size).fill(0);
    for (const [line, sign] of [[2 + 2 * i, -1], [lines + 1 + 2 * i, 1]]) {
      reading[line] += sign * (1 - reach);
      reading[line - 1] += sign * reach / 2;
      reading[line + 1] += sign * reach / 2;
    }
    return reading;
  });
  for (const reading of readings)
    for (let a = 0; a < size; ++a)
      for (let b = 0; b < size; ++b) matrix[a][b] += stiffness * reading[a] * reading[b];
  // The clamp of the lower bar at its first line, the load at the last line of the upper one.
  for (let k = 0; k < size; ++k) matrix[0][k] = matrix[k][0] = 0;
  matrix[0][0] = 1;
  load[size - 1] = 1;
  const u = solveLinear(matrix, load);
  return readings.map((reading) => stiffness * dot(reading, u));
}

const LAP = { width: 0.008, thickness: 0.002, pitch: 0.024, gap: 0.002, load: 1000, young: 7.0e10,
  diameter: 0.004 };
// The stiffness of a strip over one pitch, EA/p: 4.67e7 N/m. A flexible fastener has a tenth of it, a stiff one ten
// times it.
LAP.barStiffness = LAP.young * LAP.width * LAP.thickness / LAP.pitch;
const LAP_FASTENERS = [{ name: "flexible", stiffness: 0.1 * LAP.barStiffness },
  { name: "stiff", stiffness: 10 * LAP.barStiffness }];

// The lap joint of n fasteners: cells of half a pitch, the fasteners on the lines between two cells at the middle of
// the width (so that each loads its strip evenly across the width), the first one a pitch from the clamp, the strips
// going half a pitch beyond the end fasteners. Returns the shares F_i/P.
function lapJoint({ n, stiffness }, options) {
  const { width, thickness, pitch, gap, load, young } = LAP;
  const deck = new Deck();
  const h = pitch / 2;
  const cells = 2 + 2 * (n - 1) + 1; // a pitch of lead, the pitches between the fasteners, half a pitch of edge
  const length = cells * h;
  const lower = sheet(deck, { nx: cells, ny: 1, position: (a, b) => [length * a, width * b, 0], property: 1 });
  const upper = sheet(deck, { nx: cells, ny: 1, position: (a, b) => [h + length * a, width * b, gap], property: 2,
    firstNode: 1000, firstElement: 1000 });
  deck.add("ShellProp", 1, 1, q(thickness), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(thickness), 1, "", 1);
  deck.add("IsoMat", 1, q(young), "", q(0), q(0));
  // The basic system: KT1 along X is the shear stiffness of the row.
  deck.add("FastenerProp", 7, q(LAP.diameter), 0, 1, q(stiffness), q(stiffness), q(stiffness));
  for (let i = 0; i < n; ++i)
    deck.add("Fastener", 5001 + i, 7, "Property", 1, 2, "", "", "", q(pitch + i * pitch), q(width / 2), q(gap / 2));
  deck.hold(23456, [...lower.nodes, ...upper.nodes]);
  deck.hold(1, [lower.node(0, 0), lower.node(0, 1)]);
  for (const j of [0, 1]) deck.force(2, upper.node(cells, j), [load / 2, 0, 0]);
  const rows = elementRows(runModel(deck.statics(), options), "Fastener");
  return Array.from({ length: n }, (_, i) => rows.get(5001 + i).fx / load);
}

// The ClassicFormulation against the bars cut as the strips, their fasteners read over their footprints (the half
// side of the square of a fastener of 4 mm, 1.77 mm, in cells of 12 mm): to round-off, 2e-14 of P. Against the bars
// joined at points its stiff fasteners are 0.03 to 0.06 of P off: three of them carry 0.510, −0.020 and 0.510 of
// P for 0.478, 0.043 and 0.478. A fastener tied over its footprint holds the mean motion of its bar over the
// footprint, which under its own load is less than the motion at its middle, by about F·d/(2·EA) for each bar: it
// holds its bar more stiffly than one tied at a point. That counts beside a fastener as stiff as these (1/K is
// 0.1·p/EA, d/EA 0.074·p/EA), where the bar strains under the end fasteners and leaves the inner ones a small
// load of the other sign; it does not count beside a flexible one (4e-4 of P at most).
// The MixedHybridFormulation against the bars joined at points: 4e-4 with flexible fasteners. With stiff fasteners
// it is off that model by 1e-4 for n = 2, 0.059 for n = 3 (shares 0.449, 0.102, 0.449 for 0.478, 0.043, 0.478)
// and 0.035 for n = 5: the sheet, which its elements resolve
// across the width of the strip, gives way around a fastener of 4 mm by an amount the bars of either model do not
// have (the shell read at the point alone, without the footprint, gave 0.156 and 0.109). Criterion there: 0.08 of P.
for (const { name, options } of FORMULATIONS)
  for (const n of [2, 3, 5])
    for (const fastener of LAP_FASTENERS) {
      const isClassic = name === "ClassicFormulation";
      const sharesCriterion = !isClassic && fastener.name === "stiff" && n > 2 ? 0.08 : isClassic ? 1e-8 : 5e-3;
      test(`4. a lap joint with a row of ${n} ${fastener.name} fasteners: the share of each against the two bars ` +
        `on springs (${name})`, (t) => {
        const m = measurements(t);
        const reach = 0.5 * FOOTPRINT_SIDE_PER_DIAMETER * LAP.diameter / (LAP.pitch / 2);
        const expected = isClassic
          ? lapSharesOverFootprint({ n, barStiffness: LAP.barStiffness, stiffness: fastener.stiffness, reach })
          : lapShares({ n, barStiffness: LAP.barStiffness, leadStiffness: LAP.barStiffness,
            stiffness: fastener.stiffness });
        const measured = lapJoint({ n, stiffness: fastener.stiffness }, options);
        t.diagnostic(`shares ${measured.map((v) => v.toFixed(4)).join(", ")} for ` +
          expected.map((v) => v.toFixed(4)).join(", "));
        m.small("the sum of the shares less 1", measured.reduce((a, b) => a + b, 0) - 1, 1, 1e-8);
        measured.forEach((share, i) => m.small(`the share of fastener ${i + 1} less that of the model`,
          share - expected[i], 1, sharesCriterion));
        // The classical result: the end fasteners carry the most (n = 2: the two fasteners share the load equally,
        // whatever their stiffness).
        if (n > 2)
          m.truth("the end fasteners carry more than the others", Math.min(measured[0], measured[n - 1]) >
            Math.max(...measured.slice(1, n - 1)));
        m.done();
      });
    }

// The end fasteners carry more of the load as the fasteners stiffen: the model gives 0.344 then 0.478 for n = 3, 0.235
// then 0.477 for n = 5.
for (const { name, options } of FORMULATIONS)
  test(`4. a lap joint: the end fasteners carry more with stiff fasteners than with flexible ones (${name})`, (t) => {
    const m = measurements(t);
    for (const n of [3, 5]) {
      const [flexible, stiff] = LAP_FASTENERS.map(({ stiffness }) => lapJoint({ n, stiffness }, options));
      t.diagnostic(`${n} fasteners: the first carries ${flexible[0].toFixed(4)} then ${stiff[0].toFixed(4)}`);
      m.truth(`${n} fasteners: the end shares grow with the stiffness`, stiff[0] > flexible[0] &&
        stiff[n - 1] > flexible[n - 1]);
    }
    m.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 5. Two skins riveted together in bending
//
// Two superposed strips of thickness t (their mid-surfaces a distance d = t apart), clamped at X = 0 and loaded at
// their tips by P across the thickness, joined by a row of fasteners a pitch p apart. With ν = 0 each strip is a
// beam of EA1 = E·b·t and EI1 = E·b·t³/12. Without shear connection the two bend separately, EI0 = 2·EI1 and
// w = P·L³/(3·EI0); with a rigid connection they are the beam of thickness 2t, four times stiffer. Between the two,
// the fasteners smeared as a shear stiffness k = K/p per unit length, the theory of partial interaction (N. M.
// Newmark, C. P. Siess, I. M. Viest, "Tests and analysis of composite beams with incomplete interaction", Proceedings
// of the Society for Experimental Stress Analysis 9(1), 1951) gives the axial force N of each strip:
//     N'' − α²·N = −k·d·M/EI0,   α² = k·(2/EA1 + d²/EI0),   M = P·(L − x),   N'(0) = 0 (no slip at the clamp),
//     N(L) = 0,   so N = β·(L − x) − (β/α)·sinh(α(L − x))/cosh(αL),   β = k·d·P/(α²·EI0),
// the curvature w'' = (M − N·d)/EI0 and the tip deflection w(L) = ∫ (L − x)·w'' dx.

const SKINS = { length: 0.24, width: 0.02, thickness: 0.002, young: 7.0e10, load: 1, cells: 24, density: 2700 };
SKINS.ea = SKINS.young * SKINS.width * SKINS.thickness;
SKINS.ei = 2 * SKINS.young * SKINS.width * SKINS.thickness ** 3 / 12;
SKINS.separate = SKINS.load * SKINS.length ** 3 / (3 * SKINS.ei);

// The tip deflection of the two strips joined by fasteners of shear stiffness `stiffness` a pitch `pitch` apart,
// by the theory of partial interaction above (Simpson's rule on 2000 intervals).
function partialInteraction({ stiffness, pitch }) {
  const { length, thickness: d, ea, ei, load } = SKINS;
  const k = stiffness / pitch;
  const alpha = Math.sqrt(k * (2 / ea + d * d / ei));
  const beta = k * d * load / (alpha * alpha * ei);
  // sinh(α(L − x))/cosh(αL) written with decaying exponentials, so that a stiff connection does not overflow.
  const decay = (x) => (Math.exp(-alpha * x) - Math.exp(-alpha * (2 * length - x))) /
    (1 + Math.exp(-2 * alpha * length));
  const curvature = (x) => (load * (length - x) - d * beta * (length - x - decay(x) / alpha)) / ei;
  const steps = 2000;
  let sum = 0;
  for (let i = 0; i <= steps; ++i) {
    const x = length * i / steps;
    sum += (i === 0 || i === steps ? 1 : i % 2 ? 4 : 2) * (length - x) * curvature(x);
  }
  return sum * length / steps / 3;
}

// The two strips of this section with `count` fasteners of shear stiffness `stiffness` along X, a pitch L/count
// apart at the middle of the width, each at the centre of a cell (`onEdges`: half a cell further, on the line
// between two cells); the fasteners hold the strips together across the thickness (1e9 N/m). `cells`: the cells
// along the strips.
function rivetedSkins({ stiffness, count, density = 0, onEdges = false, cells = SKINS.cells }) {
  const { length, width, thickness, young } = SKINS;
  const deck = new Deck();
  const lower = sheet(deck, { nx: cells, ny: 1, position: (a, b) => [length * a, width * b, 0], property: 1 });
  const upper = sheet(deck, { nx: cells, ny: 1, position: (a, b) => [length * a, width * b, thickness], property: 2,
    firstNode: 1000, firstElement: 1000 });
  deck.add("ShellProp", 1, 1, q(thickness), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(thickness), 1, "", 1);
  deck.add("IsoMat", 1, q(young), "", q(0), q(density));
  // The basic system: KT1 along X the shear stiffness, KT2 along Y the same, KT3 along Z across the thickness.
  deck.add("FastenerProp", 7, q(0.004), 0, 1, q(stiffness), q(stiffness), q(1.0e9));
  const pitch = length / count;
  const cell = length / cells;
  for (let i = 0; i < count; ++i) {
    const centre = (Math.floor((i + 0.5) * pitch / cell - 1e-9) + 0.5) * cell;
    deck.add("Fastener", 5001 + i, 7, "Property", 1, 2, "", "", "", q(centre + (onEdges ? cell / 2 : 0)),
      q(width / 2), q(thickness / 2));
  }
  deck.hold(123456, [lower.node(0, 0), lower.node(0, 1), upper.node(0, 0), upper.node(0, 1)]);
  return { deck, lower, upper, pitch, tips: [lower.node(cells, 0), lower.node(cells, 1), upper.node(cells, 0),
    upper.node(cells, 1)] };
}

// The tip deflection of the riveted strips under the load P shared by the four grid points of the tips.
function skinsDeflection(joint, options) {
  const skins = rivetedSkins(joint);
  for (const id of skins.tips) skins.deck.force(2, id, [0, 0, SKINS.load / 4]);
  const run = runModel(skins.deck.statics(), options);
  return skins.tips.reduce((sum, id) => sum + nodal(run, "displacements", id)[2], 0) / 4;
}

const SKIN_STIFFNESSES = [1.0e3, 3.0e5, 3.0e6, 1.0e9];

for (const { name, options } of FORMULATIONS)
  test(`5. two skins riveted together in bending: from two plates to the monolithic one with the fastener ` +
    `stiffness (${name})`, (t) => {
    const m = measurements(t);
    const { separate } = SKINS;
    const deflections = SKIN_STIFFNESSES.map((stiffness) => skinsDeflection({ stiffness, count: 12 }, options));
    // Measured, the deflection over the theory for K = 1e3, 3e5, 3e6, 1e9: 1.0001, 1.0115, 1.0070, 1.0010 (classic),
    // 1.0002, 1.0122, 1.0118, 1.0097 (mixed): the fasteners are a pitch apart, the theory smears them. The limits:
    // 0.9977 of two separate plates (the soft fasteners still hold a little), 1.0021 and 1.0107 of the monolithic one.
    SKIN_STIFFNESSES.forEach((stiffness, i) => {
      const theory = partialInteraction({ stiffness, pitch: SKINS.length / 12 });
      t.diagnostic(`K = ${stiffness.toExponential(0)}: deflection over that of two separate plates ` +
        `${(deflections[i] / separate).toFixed(4)}, partial interaction ${(theory / separate).toFixed(4)}`);
      m.ratio(`K = ${stiffness.toExponential(0)}: the tip deflection over the theory of partial interaction`,
        deflections[i], theory, 0.02);
    });
    m.ratio("soft fasteners: the tip deflection over that of two separate plates", deflections[0], separate, 5e-3);
    m.ratio("stiff fasteners: the tip deflection over that of the monolithic plate",
      deflections[deflections.length - 1], separate / 4, 0.02);
    m.truth("the deflection falls as the fasteners stiffen", deflections.every((value, i) =>
      i === 0 || value < deflections[i - 1]));
    m.truth("the deflection stays between the monolithic plate and the two separate plates",
      deflections.every((value) => value > 0.999 * separate / 4 && value < 1.001 * separate));
    m.done();
  });

for (const { name, options } of FORMULATIONS)
  test(`5. two skins riveted together in bending: stiffer with a finer pitch of the same fasteners (${name})`,
    (t) => {
      const m = measurements(t);
      const counts = [6, 12, 24];
      const stiffness = 1.0e6;
      const deflections = counts.map((count) => skinsDeflection({ stiffness, count }, options));
      // Measured, the deflection over the theory for 6, 12 and 24 fasteners: 1.0142, 1.0116, 1.0000 (classic),
      // 1.0169, 1.0140, 1.0017 (mixed).
      counts.forEach((count, i) => {
        const theory = partialInteraction({ stiffness, pitch: SKINS.length / count });
        t.diagnostic(`${count} fasteners: deflection over that of two separate plates ` +
          `${(deflections[i] / SKINS.separate).toFixed(4)}, partial interaction ` +
          (theory / SKINS.separate).toFixed(4));
        m.ratio(`${count} fasteners: the tip deflection over the theory of partial interaction`, deflections[i],
          theory, 0.02);
      });
      m.truth("the deflection falls as the pitch is refined", deflections[1] < deflections[0] &&
        deflections[2] < deflections[1]);
      m.done();
    });

// The monolithic plate is a bound: joined at points only, the two skins cannot be stiffer than it. The fasteners are
// here on the lines between two cells (a pitch of two cells), where the ClassicFormulation reads the rotation of a
// sheet about the line from the two cells (from one of them only, the pair came out at 0.9676 of the monolithic
// deflection).
for (const { name, options } of FORMULATIONS) {
  test(`5. two skins riveted by stiff fasteners on the lines between cells are not stiffer than the monolithic ` +
    `plate (${name})`, (t) => {
    const m = measurements(t);
    const deflection = skinsDeflection({ stiffness: 1.0e9, count: 12, onEdges: true }, options);
    // Measured: mixed 1.0195.
    m.between("the tip deflection over that of the monolithic plate", deflection / (SKINS.separate / 4), 0.999, 1.03);
    m.done();
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. Curved skins
//
// Two concentric cylindrical panels of axis Y (radii 0.496 and 0.500, 60 degrees of arc from 20 to 80 degrees about
// Y, 0.3 long, 2 mm thick), joined by nine rivets along the normal: each has its GA and GB given, grid points on the
// two cylinders on the same radius, so that its axis is the radius exactly. The outer skin is held in translation
// along its two straight edges; the inner skin hangs on the rivets and carries a pressure that pulls it away from the
// outer one. The meshes: 21 × 15 cells in the ClassicFormulation, whose plane facets need them, 7 × 5 in the
// MixedHybridFormulation, which reads the cylinder from the grid points; no rivet falls on a line of these meshes.
//
// The default axes of a rivet at the angle θ are x = (cos θ, 0, sin θ) (the radius), y = Y (the basic axis least
// aligned with x: no tie here) and z = x × y = (−sin θ, 0, cos θ) (the hoop direction).
//
// Checks: the support forces balance the loads; the rivets carry the load of the inner skin (the sum of their forces
// and moments at the middles of the rivets, in the basic system, is the opposite of the resultant of the pressure);
// on the mesh of quadrilaterals along the generators, which has the two symmetries of the panel, the rivets that are
// images of one another carry the same forces; the meshes of quadrilaterals, of warped quadrilaterals (the lines
// along the axis slanted inside the panel, the outline kept) and of triangles, and the two formulations, give the
// rivet loads of a fine mesh of quadrilaterals of the ClassicFormulation (49 × 35), to the discretisation error
// stated with the case.

const CURVED = { radius: 0.5, gap: 0.004, from: 20, to: 80, length: 0.3, pressure: 1.0e4,
  kt: [3.0e7, 1.2e7, 1.2e7], kr: [150, 80, 80], across: [0.2, 0.5, 0.8], along: [0.14, 0.5, 0.86] };

// The two skins of nu × nv cells and their rivets. `skew`: the lines along the axis slanted by this many cells from
// one end of the panel to the other at the middle of the arc, and not at all on the straight edges (their
// quadrilaterals are then warped). `load(deck, inner, outer)` writes the loads.
function curvedSkins({ cells: [nu, nv], shape = QUAD4, skew = 0, load }) {
  const { radius, gap, from, to, length, kt, kr, across, along } = CURVED;
  const deck = new Deck();
  const angle = (a, b) => (from + (to - from) * (a + skew * Math.sin(Math.PI * a) * (b - 0.5) / nu)) * Math.PI / 180;
  const cylinder = (r) => (a, b) => [r * Math.cos(angle(a, b)), length * b, r * Math.sin(angle(a, b))];
  const inner = sheet(deck, { nx: nu, ny: nv, position: cylinder(radius - gap), shape, property: 1 });
  const outer = sheet(deck, { nx: nu, ny: nv, position: cylinder(radius), shape, property: 2, firstNode: 2000,
    firstElement: 2000 });
  deck.add("ShellProp", 1, 1, q(0.002), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(0.002), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density), q(ALUMINIUM.expansion));
  deck.add("FastenerProp", 7, q(0.004), "", "", ...kt.map(q), ...kr.map(q));
  const rivets = [];
  along.forEach((b, j) => across.forEach((a, i) => {
    const id = 5001 + i + 3 * j;
    const theta = (from + (to - from) * a) * Math.PI / 180;
    const [pointA, pointB] = [radius - gap, radius].map((r) => [r * Math.cos(theta), length * b, r * Math.sin(theta)]);
    deck.node(8000 + id, pointA);
    deck.node(9000 + id, pointB);
    deck.add("Fastener", id, 7, "Property", 1, 2, "", 8000 + id, 9000 + id);
    rivets.push({ id, i, j, middle: scale(0.5, add(pointA, pointB)), axes: defaultAxes(pointA, pointB) });
  }));
  // The two straight edges of the outer skin.
  deck.hold(123, outer.nodes.filter((id) => {
    const [x, , z] = deck.positions.get(id);
    const theta = Math.atan2(z, x) * 180 / Math.PI;
    return Math.abs(theta - from) < 1e-6 || Math.abs(theta - to) < 1e-6;
  }));
  load?.(deck, inner, outer);
  return { deck, inner, outer, rivets };
}

// The pressure on the inner skin: positive along the normal of its elements, which points to the axis.
const innerPressure = (deck, inner) => deck.add("ShellPressureLoad", 2, q(CURVED.pressure), ...inner.elements);

// The sum of the forces of the rivets and of their moments about the origin, in the basic system.
function rivetResultant(rows, rivets) {
  const total = [0, 0, 0, 0, 0, 0];
  for (const rivet of rivets) {
    const values = fastenerForce(rows, rivet.id);
    const force = fromAxes(rivet.axes, values.slice(0, 3));
    const moment = add(fromAxes(rivet.axes, values.slice(3)), cross(rivet.middle, force));
    for (let c = 0; c < 3; ++c) {
      total[c] += force[c];
      total[3 + c] += moment[c];
    }
  }
  return total;
}

// The curved skins under the pressure, solved once for the cases that read them.
const CURVED_CELLS = { ClassicFormulation: [21, 15], MixedHybridFormulation: [7, 5] };
const curvedResults = new Map();
function curvedCase(formulation, mesh, cells = CURVED_CELLS[formulation.name]) {
  const key = `${formulation.name} ${mesh.name} ${cells}`;
  if (!curvedResults.has(key)) {
    const skins = curvedSkins({ cells, shape: mesh.shape, skew: mesh.skew, load: innerPressure });
    const run = runModel(skins.deck.statics(), formulation.options);
    curvedResults.set(key, { ...skins, run, rows: elementRows(run, "Fastener") });
  }
  return curvedResults.get(key);
}

const CURVED_MESHES = [{ name: "ShellQuad4 along the generators", shape: QUAD4, skew: 0 },
  { name: "warped ShellQuad4 on slanted lines", shape: QUAD4, skew: 1 },
  { name: "ShellTria3", shape: TRIA3, skew: 0 }];

for (const formulation of FORMULATIONS)
  for (const mesh of CURVED_MESHES)
    test(`6. two cylindrical skins of ${mesh.name} riveted along the normal: equilibrium (${formulation.name})`,
      (t) => {
        const { deck, inner, run, rows, rivets } = curvedCase(formulation, mesh);
        const m = measurements(t);
        const applied = resultant(run, deck, "appliedLoads");
        const supports = resultant(run, deck, "singlePointConstrForces");
        const force = norm(applied.slice(0, 3));
        // The arm of the moments: the radius.
        const sizes = [force, force, force, force * CURVED.radius, force * CURVED.radius, force * CURVED.radius];
        t.diagnostic(`the load: ${applied.map((value) => value.toPrecision(6)).join(", ")}`);
        // Measured: 3e-13 (classic), 1e-11 (mixed).
        m.small("the support forces plus the loads", Math.max(...applied.map((value, c) =>
          Math.abs(value + supports[c]) / sizes[c])), 1, 1e-8);
        // The inner skin: the pressure it carries against the forces of the rivets. Measured: 3e-13 (classic),
        // 2e-11 (mixed).
        const onInner = resultant(run, deck, "appliedLoads", 1, new Set(inner.nodes));
        const carried = rivetResultant(rows, rivets);
        m.small("the forces of the rivets plus the load of the inner skin", Math.max(...onInner.map((value, c) =>
          Math.abs(value + carried[c]) / sizes[c])), 1, 1e-7);
        m.done();
      });

// The images of a rivet in the two planes of symmetry of the panel: across the arc (i → 2 − i) the hoop force fz and
// the moments about the radius and the axis change sign, along the axis (j → 2 − j) the axial force fy and the moments
// about the radius and the hoop direction do.
for (const formulation of FORMULATIONS)
  test(`6. two cylindrical skins of ShellQuad4 along the generators: the rivet forces have the symmetries of the ` +
    `panel (${formulation.name})`, (t) => {
    const { rows, rivets } = curvedCase(formulation, CURVED_MESHES[0]);
    const m = measurements(t);
    const pull = Math.max(...rivets.map((rivet) => Math.abs(fastenerForce(rows, rivet.id)[0])));
    const twist = Math.max(...rivets.map((rivet) => Math.max(...fastenerForce(rows, rivet.id).slice(3).map(Math.abs))));
    const images = [{ name: "across the arc", of: (r) => r.i + 3 * r.j !== 2 - r.i + 3 * r.j && r.i < 1 ?
      5001 + 2 - r.i + 3 * r.j : 0, signs: [1, 1, -1, -1, -1, 1] },
    { name: "along the axis", of: (r) => r.j < 1 ? 5001 + r.i + 3 * (2 - r.j) : 0, signs: [1, -1, 1, -1, 1, -1] }];
    for (const image of images) {
      let worst = 0;
      for (const rivet of rivets) {
        const other = image.of(rivet);
        if (!other) continue;
        const [first, second] = [fastenerForce(rows, rivet.id), fastenerForce(rows, other)];
        first.forEach((value, c) => {
          worst = Math.max(worst, Math.abs(value - image.signs[c] * second[c]) / (c < 3 ? pull : twist));
        });
      }
      // Measured: 3e-12 (classic), 7e-11 (mixed).
      m.small(`the forces of a rivet less those of its image ${image.name}`, worst, 1, 1e-7);
    }
    m.done();
  });

// The same panel on the three meshes and in the two formulations, against the ClassicFormulation on 49 × 35
// quadrilaterals: the pull of each rivet (fx) and its shear (fy and fz together), over the largest of the reference,
// and the mean radial displacement of the four corners of the inner skin.
//
// The reference: pulls of 142.6, 34.5, 142.6 (end rows) and 158.1, 35.3, 158.1 N (middle row), shears of 286.9, 6.3,
// 286.9 and 301.7, 0, 301.7 N, corners at −7.33 mm. It is itself known to some 4 % on the shears: on 28 × 20,
// 35 × 25, 42 × 30 and 49 × 35 cells those of the end rivets of the end rows are 294.1, 291.7, 299.4 and 286.9 N,
// those of the middle row 293.5, 295.2, 302.6 and 301.7 N.
// Measured (pulls, shears, corners): ClassicFormulation on 21 × 15 cells 0.052, 0.038, 0.948 (quadrilaterals),
// 0.065, 0.056, 0.950 (warped), 0.053, 0.031, 0.942 (triangles); MixedHybridFormulation on 7 × 5 cells 0.057, 0.118,
// 0.952 (quadrilaterals), 0.048, 0.126, 0.963 (warped), 0.059, 0.120, 0.957 (triangles): its shears are 322.5 and
// 324.1 N on these cells, 310.5 and 301.5 N on 14 × 10, 296.5 and 304.1 N on 21 × 15.
const CURVED_AGREEMENT = { ClassicFormulation: { pulls: 0.10, shears: 0.10, corners: 0.08 },
  MixedHybridFormulation: { pulls: 0.08, shears: 0.15, corners: 0.06 } };

test("6. two cylindrical skins riveted along the normal: the meshes and the formulations agree", (t) => {
  const m = measurements(t);
  const reading = (formulation, mesh, cells) => {
    const { deck, inner, run, rows, rivets } = curvedCase(formulation, mesh, cells);
    const radial = (id) => {
      const [x, , z] = deck.positions.get(id);
      const u = nodal(run, "displacements", id);
      return (u[0] * x + u[2] * z) / Math.hypot(x, z);
    };
    return { pulls: rivets.map((rivet) => fastenerForce(rows, rivet.id)[0]),
      shears: rivets.map((rivet) => Math.hypot(...fastenerForce(rows, rivet.id).slice(1, 3))),
      corners: inner.corners.reduce((sum, id) => sum + radial(id), 0) / 4 };
  };
  const reference = reading(FORMULATIONS[0], CURVED_MESHES[0], [49, 35]);
  const [largestPull, largestShear] = [Math.max(...reference.pulls), Math.max(...reference.shears)];
  t.diagnostic(`reference: pulls ${reference.pulls.map((v) => v.toFixed(1)).join(", ")}; shears ` +
    `${reference.shears.map((v) => v.toFixed(1)).join(", ")}; corners ${reference.corners.toExponential(4)}`);
  for (const formulation of FORMULATIONS)
    for (const mesh of CURVED_MESHES) {
      const other = reading(formulation, mesh);
      const what = `${mesh.name}, ${formulation.name}`;
      const criteria = CURVED_AGREEMENT[formulation.name];
      m.small(`${what}: the pulls of the rivets less the reference`, Math.max(...other.pulls.map((value, k) =>
        Math.abs(value - reference.pulls[k]))), largestPull, criteria.pulls);
      m.small(`${what}: the shears of the rivets less the reference`, Math.max(...other.shears.map((value, k) =>
        Math.abs(value - reference.shears[k]))), largestShear, criteria.shears);
      m.ratio(`${what}: the radial displacement of the corners of the inner skin`, other.corners, reference.corners,
        criteria.corners);
    }
  m.done();
});

// A rivet on a line of the mesh, in a panel that keeps its two symmetries: 14 × 10 quadrilaterals, where the three
// rivets of the middle column lie on the line of the middle of the arc and the rivet at the centre on a grid point.
// The rivets that are images of one another across the arc must still carry the same loads.
// The square of such a rivet lies in the tangent plane of the skin at the point (the mean of the normals of the
// elements that hold it), on the elements of both sides of the line. Measured: 4e-12.
// (Tied to one of the two elements of its line, the lowest ID, the two end rivets of a row carried shears 4 % apart.)
test("6. two cylindrical skins of 14 x 10 ShellQuad4 with rivets on a line of the mesh: the rivet forces keep the " +
  "symmetry across the arc (ClassicFormulation)", (t) => {
  const { rows, rivets } = curvedCase(FORMULATIONS[0], CURVED_MESHES[0], [14, 10]);
  const m = measurements(t);
  for (const j of [0, 1, 2]) {
    const [left, right] = [0, 2].map((i) => fastenerForce(rows, rivets[i + 3 * j].id));
    m.ratio(`row ${j + 1}: the pull of the first rivet over that of the last`, left[0], right[0], 1e-6);
    m.ratio(`row ${j + 1}: the shear of the first rivet over that of the last`, Math.hypot(left[1], left[2]),
      Math.hypot(right[1], right[2]), 1e-6);
  }
  m.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 7. A stiffened panel
//
// A flat skin 2 mm thick, of cells of 25 mm (8 × 4 of them: 0.2 × 0.1 m; or 12 × 8: 0.3 × 0.2 m), with a stringer
// along X at the middle of its width: a flange of 50 mm (two cells, 2 mm thick, its mid-surface 2 mm above that of
// the skin) and a web of 30 mm standing on the middle line of the flange. The skin is held in translation along
// X = 0 and across the thickness along its other end. Two loads: a pressure on the skin, and a pull of 100 N along X
// on the end of the stringer.
//
// The flange is joined to the skin either by a fastener at the centre of each of its cells (a pitch of one cell in
// both directions), or by a rigid link from each grid point of the skin under the flange to the grid point of the
// flange above it (RigidBar: the flange merged to the grid points of the skin, at its distance).
//   - With fasteners a thousand times stiffer than rivets, the panel is the merged one, to the tolerance stated.
//   - With rivets, the stringer is a free body that the fasteners alone hold: the sum of their forces and moments
//     (given in the basic system by MCID 0) is the resultant of what is applied to the stringer: nothing under the
//     pressure on the skin, the pull in the second load.

const PANEL = { cell: 0.025, gap: 0.002, web: 0.03, pressure: 1000, pull: 100,
  rivet: { kt: [1.2e7, 1.2e7, 3.0e7], kr: [80, 80, 150] } };

// The panel of nx × ny cells, its flange joined by `joint`: "links", or the stiffnesses { kt, kr } of its fasteners
// in the basic system. Each cell is meshed by `cut` × `cut` elements (1: one element a cell); the fasteners stay at
// the centres of the cells. Returns the deck, the grid points read (the middle of the skin and the top of the web
// at its middle), the fasteners with their middles, the grid points of the end of the stringer, which carry the
// pull, and the length of the panel.
function stiffenedPanel(joint, [nx, ny], cut = 1) {
  const { cell, gap, web, pressure, pull } = PANEL;
  const [length, width] = [nx * cell, ny * cell];
  const [mx, my] = [nx * cut, ny * cut];
  const deck = new Deck();
  const skin = sheet(deck, { nx: mx, ny: my, position: (a, b) => [length * a, width * b, 0], property: 1 });
  const flange = sheet(deck, { nx: mx, ny: 2 * cut,
    position: (a, b) => [length * a, width / 2 + cell * (2 * b - 1), gap], property: 2, firstNode: 20000,
    firstElement: 20000 });
  const top = Array.from({ length: mx + 1 }, (_, i) => deck.node(40001 + i, [length * i / mx, width / 2, gap + web]));
  for (let i = 0; i < mx; ++i)
    deck.add("ShellQuad4", 40001 + i, 3, flange.node(i, cut), flange.node(i + 1, cut), top[i + 1], top[i]);
  for (const property of [1, 2, 3]) deck.add("ShellProp", property, 1, q(0.002), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density));
  const fasteners = [];
  if (joint === "links") {
    let id = 50000;
    for (let i = 0; i <= mx; ++i)
      for (let j = 0; j <= 2 * cut; ++j)
        deck.add("RigidBar", ++id, skin.node(i, (ny / 2 - 1) * cut + j), flange.node(i, j), 123456);
  } else {
    deck.add("FastenerProp", 7, q(0.004), 0, 1, ...joint.kt.map(q), ...joint.kr.map(q));
    for (let i = 0; i < nx; ++i)
      for (const j of [0, 1]) {
        const middle = [length * (i + 0.5) / nx, width / 2 + cell * (j - 0.5), gap / 2];
        fasteners.push({ id: 60001 + 2 * i + j, middle });
        deck.add("Fastener", 60001 + 2 * i + j, 7, "Property", 1, 2, "", "", "", ...middle.map(q));
      }
  }
  deck.hold(123, Array.from({ length: my + 1 }, (_, j) => skin.node(0, j)));
  deck.hold(3, Array.from({ length: my + 1 }, (_, j) => skin.node(mx, j)));
  deck.add("ShellPressureLoad", 11, q(pressure), ...skin.elements);
  const end = [flange.node(mx, 0), flange.node(mx, cut), flange.node(mx, 2 * cut), top[mx]];
  for (const id of end) deck.force(12, id, [pull / 4, 0, 0]);
  return { deck, fasteners, end, length, read: { skin: skin.node(mx / 2, my / 2), web: top[mx / 2] } };
}

// The panel solved under its two loads: the deflections of the skin and of the web under the pressure, the mean
// displacement along X of the four grid points that carry the pull, and the run.
const panelResults = new Map();
function panelCase(formulation, name, joint, cells = [8, 4], cut = 1) {
  const key = `${formulation.name} ${name} ${cells} ${cut}`;
  if (!panelResults.has(key)) {
    const panel = stiffenedPanel(joint, cells, cut);
    const run = runModel(panel.deck.statics([[1, 1, 11, 0], [2, 1, 12, 0]]), formulation.options);
    panelResults.set(key, { ...panel, run, values: {
      "deflection of the skin under the pressure": nodal(run, "displacements", panel.read.skin, 1)[2],
      "deflection of the top of the web under the pressure": nodal(run, "displacements", panel.read.web, 1)[2],
      "displacement of the end of the stringer under the pull": panel.end.reduce((sum, id) =>
        sum + nodal(run, "displacements", id, 2)[0], 0) / 4,
    } });
  }
  return panelResults.get(key);
}
const STIFF_FASTENERS = { kt: PANEL.rivet.kt.map((k) => 1000 * k), kr: PANEL.rivet.kr.map((k) => 1000 * k) };

// The meshes of the comparison with the merged stringer: 12 × 8 cells in the ClassicFormulation, 8 × 4 in the
// MixedHybridFormulation (its cost).
const PANEL_CELLS = { ClassicFormulation: [12, 8], MixedHybridFormulation: [8, 4] };

// With elements as large as the pitch of the fasteners, the ClassicFormulation is close to the merged panel: an
// element tied at its centre in translation and in rotation has little shape left to part from the one it is
// fastened to. It is a property of that mesh, not of the joint: between two fasteners the skin and the flange are
// two sheets, which part from each other (nothing presses them together in a linear analysis), and a mesh that
// resolves them shows it. The deflection of the skin under the pressure, fastened over merged, on the panel of
// 8 × 4 cells with each cell cut in 1, 2, 3, 4, 6 and 8: 1.43, 2.70, 2.31, 2.69, 2.68, 2.72 with these stiff
// fasteners; 3.12, 4.04, 3.89, 4.08, 4.10, 4.14 with rivets (the MixedHybridFormulation, with each cell cut in 1,
// 2 and 3: 2.69, 3.48, 3.34; 4.35, 4.58, 4.53).
test("7. a stringer fastened to a skin by stiff fasteners at every cell is close to the stringer merged to the " +
  "skin, on elements of the pitch of the fasteners (ClassicFormulation)", (t) => {
  const formulation = FORMULATIONS[0];
  const cells = PANEL_CELLS[formulation.name];
  const merged = panelCase(formulation, "links", "links", cells);
  const fastened = panelCase(formulation, "stiff", STIFF_FASTENERS, cells);
  const m = measurements(t);
  // Never stiffer than the merged panel, and within a quarter of it. Measured on these 12 × 8 cells: 1.147, 1.161
  // and 1.025.
  for (const [what, value] of Object.entries(fastened.values))
    m.between(`the ${what}: fastened over merged`, value / merged.values[what], 1.0, 1.25);
  m.done();
});

for (const formulation of FORMULATIONS)
  test(`7. a stringer riveted to a skin: the rivets carry what is applied to the stringer (${formulation.name})`,
    (t) => {
      const { run, fasteners, end, deck, values, length } = panelCase(formulation, "rivets", PANEL.rivet);
      const merged = panelCase(formulation, "links", "links");
      const m = measurements(t);
      for (const [what, value] of Object.entries(values))
        t.diagnostic(`the ${what}: riveted over merged ${(value / merged.values[what]).toFixed(4)}`);
      // The pull on the stringer: 100 N along X at the four grid points of its end.
      const applied = [0, 0, 0, 0, 0, 0];
      for (const id of end) {
        const moment = cross(deck.positions.get(id), [PANEL.pull / 4, 0, 0]);
        applied[0] += PANEL.pull / 4;
        for (let c = 0; c < 3; ++c) applied[3 + c] += moment[c];
      }
      [[1, [0, 0, 0, 0, 0, 0], "the pressure on the skin"], [2, applied, "the pull on the stringer"]].forEach(
        ([subcase, wanted, what]) => {
          const rows = elementRows(run, "Fastener", "forces", subcase);
          const total = [0, 0, 0, 0, 0, 0];
          let size = 0;
          for (const { id, middle } of fasteners) {
            const values6 = fastenerForce(rows, id);
            const moment = add(values6.slice(3), cross(middle, values6.slice(0, 3)));
            for (let c = 0; c < 3; ++c) {
              total[c] += values6[c];
              total[3 + c] += moment[c];
            }
            size = Math.max(size, norm(values6.slice(0, 3)));
          }
          t.diagnostic(`${what}: the largest fastener force is ${size.toPrecision(4)} N`);
          // Measured: 8e-14 (classic), 2e-11 (mixed).
          m.small(`${what}: the sum of the forces of the rivets less what is applied to the stringer`,
            norm(sub(total.slice(0, 3), wanted.slice(0, 3))), size, 1e-7);
          m.small(`${what}: the sum of the moments of the rivets less what is applied to the stringer`,
            norm(sub(total.slice(3), wanted.slice(3))), size * length, 1e-7);
        });
      m.done();
    });

// The two formulations on the panel of 8 × 4 cells, under the pressure (the pull is carried by four point forces,
// under which a sheet of the MixedHybridFormulation sinks more than one of the ClassicFormulation: not compared):
// merged, where no fastener is read, then fastened.
const PRESSURE_VALUES = ["deflection of the skin under the pressure", "deflection of the top of the web under the " +
  "pressure"];

test("7. a stiffened panel with the stringer merged to the skin: the two formulations agree", (t) => {
  const m = measurements(t);
  const [classic, mixed] = FORMULATIONS.map((formulation) => panelCase(formulation, "links", "links").values);
  // Measured: 1.050 and 1.052 on these 8 × 4 cells (1.021 on 12 × 8): the discretisation of the panel.
  for (const what of PRESSURE_VALUES) m.ratio(`the ${what}, mixed over classic`, mixed[what], classic[what], 0.08);
  m.done();
});

// The fastened panel: the MixedHybridFormulation on cells of the pitch of the fasteners against the
// ClassicFormulation with each cell cut in 8 × 8 (cells of 3 mm, the size of the fasteners), which resolves the
// sheets between the fasteners as the elements of the MixedHybridFormulation do. Measured, mixed over classic:
// 1.03 and 1.03 with rivets; 0.97 and 0.96 with the stiff fasteners, where the MixedHybridFormulation has not
// converged (the series of the case above: it softens by a quarter once its cells are cut in 2). On the same cells
// of the pitch the two formulations are 1.5 (rivets) and 2.0 (stiff fasteners) apart: the ClassicFormulation is
// close to the merged panel there.
test("7. a stiffened panel with the stringer fastened to the skin: the MixedHybridFormulation on cells of the " +
  "pitch against the ClassicFormulation on cells cut in eight", (t) => {
  const m = measurements(t);
  for (const [name, joint, criterion] of [["rivets", PANEL.rivet, 0.06], ["stiff", STIFF_FASTENERS, 0.08]]) {
    const classic = panelCase(FORMULATIONS[0], name, joint, [8, 4], 8).values;
    const mixed = panelCase(FORMULATIONS[1], name, joint).values;
    for (const what of PRESSURE_VALUES)
      m.ratio(`${name === "stiff" ? "stiff fasteners" : name}: the ${what}, mixed over classic`, mixed[what],
        classic[what], criterion);
  }
  m.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 8. Weld
//
// A Weld is a shear-deformable beam of circular section between its piercing points (WeldElement.hpp): diameter D,
// E and G of the material of its property, length L (the distance of the piercing points, or the mean thickness of
// the two sheets when that is longer, kept between LDMIN·D and LDMAX·D), at the middle of A-B. A uniform beam of
// length L is six uncoupled springs on the relative motion of its ends carried to its middle p (the flexibility of
// the beam about its middle is diagonal):
//     along the beam   EA/L,                          about the beam   GJ/L,
//     across it        1/(L³/(12·EI) + L/(k·A·G)),    about a diameter EI/L,
// A = πD²/4, I = πD⁴/64, J = 2I, k = 6(1 + ν)/(7 + 6ν) (Cowper's coefficient of a circle; the shear term is the
// 12·EI/(L³·(1 + Φ)) of the header, Φ = 12·EI/(k·A·G·L²)). So the checks of the Fastener apply with these
// stiffnesses: with A and B grid points of the model, the relative motion of B and A at p is the force of the weld
// over them, whatever the sheets do.
//
// The joint: two sheets of 2 mm in contact (their mid-surfaces 2 mm apart, the length of the beam), the lower one
// clamped along X = 0, the upper one loaded at its corners by six loads, each a force of 100 N or a moment of 1 N·m
// at p along an axis of the basic system. The weld is at (0.066, 0.052), inside an element of each sheet of 3 × 3
// cells, or at the centre of the sheets, a grid point of a sheet of 4 × 4 cells, for the forms that join a grid point.
//
// The force table of a Weld has the layout of a bar (the axial force, the torque, the two shears and the two bending
// moments at each end). Its two transverse axes follow the default axes of a fastener, so the table is compared with
// the load through what does not depend on them: the axial force, the torque, the resultant shear and the resultant
// bending moment at the middle.

const WELD = { diameter: 0.005, young: ALUMINIUM.young, poisson: ALUMINIUM.poisson, gap: 0.002, thickness: 0.002,
  side: 0.12, inside: [0.066, 0.052], centre: [0.06, 0.06] };

// The four stiffnesses of the springs of a weld of length `length` and modulus `young`: along, across, about the
// beam, about a diameter.
function weldSprings(length, young = WELD.young) {
  const { diameter, poisson } = WELD;
  const area = Math.PI * diameter ** 2 / 4;
  const inertia = Math.PI * diameter ** 4 / 64;
  const shear = young / (2 * (1 + poisson));
  const coefficient = 6 * (1 + poisson) / (7 + 6 * poisson);
  return { along: young * area / length,
    across: 1 / (length ** 3 / (12 * young * inertia) + length / (coefficient * area * shear)),
    twist: shear * 2 * inertia / length, bend: young * inertia / length };
}

const TRIA3_13 = { name: "ShellTria3", shape: "ShellTria3", diagonal: "13" };

// The two sheets of this section, the weld at `at`. `lowerShape`, `upperShape`: their elements (QUAD4 or TRIA3_13);
// `lowerCells`, `upperCells`: their cells a side; `held`: the grid points of the lower sheet that are clamped
// ("edge": along X = 0, "all"); `lowerThickness`: the thickness of the lower sheet; `weldYoung`: the modulus of the
// material of the weld (the material 2). Returns what the forms of the card name: the elements pierced (`elementA`,
// `elementB`), their grid points (`faceA`, `faceB`), the grid points at the weld (`gridA`, `gridB`: 0 when the sheet
// has none there), the points A and B and the middle of the weld.
function weldSheets({ lowerShape = QUAD4, upperShape = QUAD4, lowerCells = 3, upperCells = 3, at = WELD.inside,
  held = "edge", lowerThickness = WELD.thickness, weldYoung = WELD.young } = {}) {
  const { side, thickness, gap } = WELD;
  const deck = new Deck();
  const lower = sheet(deck, { nx: lowerCells, ny: lowerCells, position: (a, b) => [side * a, side * b, 0],
    shape: lowerShape, property: 1 });
  const upper = sheet(deck, { nx: upperCells, ny: upperCells, position: (a, b) => [side * a, side * b, gap],
    shape: upperShape, property: 2, firstNode: 1000, firstElement: 1000 });
  deck.add("ShellProp", 1, 1, q(lowerThickness), 1, "", 1);
  deck.add("ShellProp", 2, 1, q(thickness), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density));
  deck.add("IsoMat", 2, q(weldYoung), "", q(WELD.poisson), q(0));
  deck.hold(123456, held === "all" ? lower.nodes : Array.from({ length: lowerCells + 1 }, (_, j) =>
    lower.node(0, j)));
  // The element of a sheet the weld falls in, and its grid points. A weld on a line of the mesh is given the cell
  // before the line; in a cell cut along its diagonal 1-3, the first triangle (corners 1, 2, 3) holds the points
  // below the diagonal and those on it.
  const pierced = (plate, cells, shape) => {
    const size = side / cells;
    const [i, j] = at.map((x) => Math.min(cells - 1, Math.ceil(x / size - 1e-9) - 1));
    const [xi, eta] = [at[0] / size - i, at[1] / size - j];
    const corners = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) => plate.node(a, b));
    if (shape === QUAD4) return { element: plate.element(i, j)[0], face: corners };
    const first = xi >= eta - 1e-9;
    return { element: plate.element(i, j)[first ? 0 : 1], face: first ? corners.slice(0, 3)
      : [corners[0], corners[2], corners[3]] };
  };
  const gridAt = (plate, cells) => {
    const [i, j] = at.map((x) => x * cells / side);
    return Math.abs(i - Math.round(i)) < 1e-9 && Math.abs(j - Math.round(j)) < 1e-9
      ? plate.node(Math.round(i), Math.round(j)) : 0;
  };
  const [a, b] = [pierced(lower, lowerCells, lowerShape), pierced(upper, upperCells, upperShape)];
  return { deck, lower, upper, weldYoung, elementA: a.element, elementB: b.element, faceA: a.face, faceB: b.face,
    gridA: gridAt(lower, lowerCells), gridB: gridAt(upper, upperCells), pointA: [...at, 0], pointB: [...at, gap],
    middle: [...at, gap / 2] };
}

// The six loads of this section on the corners of the upper sheet, in the load sets 11 to 16.
const WELD_LOADS = [0, 1, 2, 3, 4, 5].map((c) => ({ force: [0, 1, 2].map((k) => (k === c ? 100 : 0)),
  moment: [0, 1, 2].map((k) => (k + 3 === c ? 1 : 0)) }));

// A weld written by `write(sheets)` (the fields of the Weld card after its property) on the sheets `sheets`, solved
// under the six loads. `property`: the fields of the WeldProp after its diameter. Returns, for each load, the
// translations of the corners of the upper sheet, what the force table says that does not depend on the transverse
// axes, and the motions of the grid points `ends` = [A, B] when given.
function weldResults(sheets, write, options, { ends = null, property = [] } = {}) {
  const { deck, upper, middle } = sheets;
  deck.add("Weld", 5001, 9, ...write(sheets));
  deck.add("WeldProp", 9, 2, q(WELD.diameter), ...property);
  WELD_LOADS.forEach(({ force, moment }, c) => applyWrench(deck, 11 + c, upper.corners, force, moment, middle));
  const run = runModel(deck.statics(WELD_LOADS.map((_, c) => [1 + c, 1, 11 + c, 0])), options);
  return WELD_LOADS.map((_, c) => {
    const row = elementRows(run, "Weld", "forces", 1 + c).get(5001);
    return { corners: upper.corners.flatMap((id) => nodal(run, "displacements", id, 1 + c).slice(0, 3)),
      table: [row.axial, Math.abs(row.torque), Math.hypot(row.shear1, row.shear2),
        Math.hypot(row.bendingMomentA1 + row.bendingMomentB1, row.bendingMomentA2 + row.bendingMomentB2) / 2],
      ends: ends ? ends.map((id) => (id ? nodal(run, "displacements", id, 1 + c) : [0, 0, 0, 0, 0, 0])) : null };
  });
}

// Checks one weld between the points A and B (their motions in `results[c].ends`) against the springs of its beam
// of length `length`, and its force table against the loads: the measurements of `m` under the name `what`.
function checkWeldBeam(m, what, sheets, results, length, tolerance = 1e-6) {
  const springs = weldSprings(length, sheets.weldYoung);
  const axis = unit(sub(sheets.pointB, sheets.pointA));
  let [worstMotion, worstTable] = [0, 0];
  WELD_LOADS.forEach(({ force, moment }, c) => {
    const [a, b] = results[c].ends;
    // The relative motion at the middle p, and what the springs give: along and across the beam.
    const slide = sub(add(b.slice(0, 3), cross(b.slice(3), sub(sheets.middle, sheets.pointB))),
      add(a.slice(0, 3), cross(a.slice(3), sub(sheets.middle, sheets.pointA))));
    const turn = sub(b.slice(3), a.slice(3));
    const split = (vector) => [scale(dot(vector, axis), axis), sub(vector, scale(dot(vector, axis), axis))];
    const [forceAlong, forceAcross] = split(force);
    const [momentAlong, momentAcross] = split(moment);
    const wantedSlide = add(scale(1 / springs.along, forceAlong), scale(1 / springs.across, forceAcross));
    const wantedTurn = add(scale(1 / springs.twist, momentAlong), scale(1 / springs.bend, momentAcross));
    worstMotion = Math.max(worstMotion, c < 3 ? norm(sub(slide, wantedSlide)) / norm(wantedSlide)
      : norm(sub(turn, wantedTurn)) / norm(wantedTurn));
    // The force table: axial force, torque, shear and bending moment at the middle.
    const wanted = [dot(force, axis), Math.abs(dot(moment, axis)), norm(forceAcross), norm(momentAcross)];
    const sizes = c < 3 ? [100, 100 * WELD.side, 100, 100 * WELD.side] : [1 / WELD.side, 1, 1 / WELD.side, 1];
    worstTable = Math.max(worstTable, ...results[c].table.map((value, k) => Math.abs(value - wanted[k]) / sizes[k]));
  });
  m.small(`${what}: the slide and the turn of B on A at the middle less F/K, the worst of the six loads`,
    worstMotion, 1, tolerance);
  m.small(`${what}: the force table less the load, the worst of the six loads`, worstTable, 1, 1e-7);
}

// The largest difference between the results of two welds (the corners of the upper sheet and the force tables,
// each over its largest value under the load).
function weldDifference(first, second) {
  let worst = 0;
  first.forEach((result, c) => {
    worst = Math.max(worst, worstDifference(result.corners, second[c].corners),
      worstDifference(result.table, second[c].table));
  });
  return worst;
}

// The forms whose points A and B are grid points of the model: GA and GB given to a patch form, and AlignedNodes,
// which joins a grid point of each sheet directly.
const givenEnds = ({ deck, pointA, pointB }) => [deck.node(9002, pointA), deck.node(9003, pointB)];
const WELDS_BETWEEN_POINTS = [
  { name: "PartPatch with GA and GB", sheets: {}, write: (sheets) => ["", "PartPatch", ...givenEnds(sheets), "", 1,
    2], ends: () => [9002, 9003] },
  { name: "ElemPatch with GA and GB", sheets: {}, write: (sheets) => ["", "ElemPatch", ...givenEnds(sheets), "",
    sheets.elementA, sheets.elementB], ends: () => [9002, 9003] },
  { name: "AlignedNodes", sheets: { lowerCells: 4, upperCells: 4, at: WELD.centre },
    write: ({ gridA, gridB }) => ["", "AlignedNodes", gridA, gridB], ends: ({ gridA, gridB }) => [gridA, gridB] },
];

for (const { name, options } of FORMULATIONS)
  test(`8. a weld between two grid points is the beam of its diameter, material and length (${name})`, (t) => {
    const m = measurements(t);
    // Measured: 3e-12 between given grid points and 2e-9 for AlignedNodes (classic), 6e-11 and 2e-10 (mixed), on
    // the motions and on the force table alike.
    for (const form of WELDS_BETWEEN_POINTS) {
      const sheets = weldSheets(form.sheets);
      const results = weldResults(sheets, form.write, options, { ends: form.ends(sheets) });
      checkWeldBeam(m, form.name, sheets, results, WELD.gap);
    }
    m.done();
  });

// LDMIN: a weld shorter than LDMIN·D takes that length, at the middle of A-B with rigid arms to A and B. Here
// LDMIN = 1.2 makes the beam 6 mm long between sheets 2 mm apart.
for (const { name, options } of FORMULATIONS)
  test(`8. a weld shorter than LDMIN times its diameter is the beam of that length (${name})`, (t) => {
    const m = measurements(t);
    const form = WELDS_BETWEEN_POINTS[1];
    const sheets = weldSheets();
    const results = weldResults(sheets, form.write, options, { ends: form.ends(sheets),
      property: ["", "", q(1.2), q(5)] });
    checkWeldBeam(m, "LDMIN 1.2", sheets, results, 1.2 * WELD.diameter);
    m.done();
  });

// The patch-to-patch forms pierce the same elements at the same points as the ElemPatch above. PartPatch and
// ElemPatch tie the weld over its cross-section: the same joint. An ElemIdPatch names the one element each side
// is tied to: in the ClassicFormulation its points move with the grid points of those elements, not with the corners
// of the square of the weld, and the joint is that of the other forms to the difference of the two ties (a weld of
// 5 mm inside elements of 40 mm: 1.8e-3 of the results); the MixedHybridFormulation reads the shell over the square
// for every form.
const SAME_WELDS = [
  { name: "PartPatch, GS", write: ({ deck, middle }) => [deck.node(9001, middle), "PartPatch", "", "", "", 1, 2] },
  { name: "PartPatch, XS YS ZS", write: ({ middle }) => ["", "PartPatch", "", "", "", 1, 2, ...middle.map(q)] },
  { name: "ElemPatch, GS", write: ({ deck, middle, elementA, elementB }) => [deck.node(9001, middle), "ElemPatch", "",
    "", "", elementA, elementB] },
  { name: "ElemPatch, XS YS ZS", write: ({ middle, elementA, elementB }) => ["", "ElemPatch", "", "", "", elementA,
    elementB, ...middle.map(q)] },
  { name: "ElemIdPatch", write: ({ deck, middle, elementA, elementB }) => [deck.node(9001, middle), "ElemIdPatch", "",
    "", "", elementA, elementB] },
];
const byElements = SAME_WELDS[4].write;
// A NodeIdPatch names the faces by their grid points, the first face padded to eight.
const byPoints = (shapes) => ({ deck, middle, faceA, faceB }) => [deck.node(9001, middle), "NodeIdPatch", "", "",
  shapes, "", ...faceA, ...new Array(8 - faceA.length).fill(""), ...faceB];

for (const { name, options } of FORMULATIONS)
  test(`8. the patch-to-patch forms of a weld that name properties or elements give the same joint (${name})`,
    (t) => {
      const m = measurements(t);
      const reference = weldResults(weldSheets(), WELDS_BETWEEN_POINTS[1].write, options);
      // Measured: 4e-11 (classic), 9e-11 (mixed).
      for (const form of SAME_WELDS) {
        const isOneElement = form.name === "ElemIdPatch" && name === "ClassicFormulation";
        m.small(`${form.name}: the results less those of the ElemPatch with GA and GB`,
          weldDifference(reference, weldResults(weldSheets(), form.write, options)), 1, isOneElement ? 0.01 : 1e-8);
      }
      m.done();
    });

// The faces of a NodeIdPatch are the elements whose grid points it names (four: a quadrilateral, three: a triangle
// of a sheet of ShellTria3): the joint of the ElemIdPatch that names those elements.
const FACE_WELDS = [{ name: "QuadQuadPatch", lowerShape: QUAD4, upperShape: QUAD4 },
  { name: "TriaTriaPatch", lowerShape: TRIA3_13, upperShape: TRIA3_13 },
  { name: "QuadTriaPatch", lowerShape: QUAD4, upperShape: TRIA3_13 },
  { name: "TriaQuadPatch", lowerShape: TRIA3_13, upperShape: QUAD4 }];

for (const { name, options } of FORMULATIONS) {
  const limit = name === "MixedHybridFormulation" ? { todo: "a face given by grid points keeps the interpolation " +
    "of the ClassicFormulation over the translations of those grid points (specification, section 5.10), while " +
    "the ElemIdPatch reads the shell over the footprint of the weld: the corners of the upper sheet move up to " +
    "0.17 (QuadQuadPatch) and 0.20 (TriaTriaPatch) of their largest translation differently" }
    : {};
  test(`8. a weld NodeIdPatch between two faces is the ElemIdPatch of the elements of those faces (${name})`, limit,
    (t) => {
      const m = measurements(t);
      // Measured (classic): 0. The MixedHybridFormulation is run on the first two (QuadTriaPatch and TriaQuadPatch
      // measured 1.01 and 1.52).
      for (const form of name === "MixedHybridFormulation" ? FACE_WELDS.slice(0, 2) : FACE_WELDS) {
        const shapes = { lowerShape: form.lowerShape, upperShape: form.upperShape };
        m.small(`NodeIdPatch ${form.name}: the results less those of the ElemIdPatch`,
          weldDifference(weldResults(weldSheets(shapes), byElements, options),
            weldResults(weldSheets(shapes), byPoints(form.name), options)), 1, 1e-8);
      }
      m.done();
    });
}

// The point-to-patch forms: the grid point GS, here the grid point of the upper sheet at the centre, is the end B
// itself, and A is where it projects on the element or the face named in the lower sheet. To know the motion of A,
// the lower sheet is 20 mm thick and clamped at every grid point, and the weld is of a material a thousand times
// softer than the sheets (E = 7e7): A does not move, to the flexibility of the lower sheet under the weld, and the
// motion of GS carried to the middle of the weld is the load over the springs of the beam (its length is the
// distance of A and B: these forms read no thickness).
const POINT_WELDS = [
  { name: "ElemIdPatch without SHIDB", lowerShape: QUAD4, write: ({ gridB, elementA }) => [gridB, "ElemIdPatch", "",
    "", "", elementA] },
  { name: "NodeIdPatch QuadPatch", lowerShape: QUAD4, write: ({ gridB, faceA }) => [gridB, "NodeIdPatch", "", "",
    "QuadPatch", "", ...faceA] },
  { name: "ElemIdPatch without SHIDB on a ShellTria3", lowerShape: TRIA3_13, write: ({ gridB, elementA }) => [gridB,
    "ElemIdPatch", "", "", "", elementA] },
  { name: "NodeIdPatch TriaPatch", lowerShape: TRIA3_13, write: ({ gridB, faceA }) => [gridB, "NodeIdPatch", "", "",
    "TriaPatch", "", ...faceA] },
];
// Measured: 4e-9 in the ClassicFormulation, whose A is the mean of grid points that are held. In the
// MixedHybridFormulation an ElemIdPatch reads the shell at A, which moves with the unknowns of the sheet that are not
// grid points: 5.0e-4 on the quadrilateral and 5.5e-6 on the triangle, and 6.2e-5 between the two forms (the
// NodeIdPatch keeps the tie to the grid points).
const POINT_WELD_TOLERANCE = { ClassicFormulation: 1e-6, MixedHybridFormulation: 2e-3 };

for (const { name, options } of FORMULATIONS)
  test(`8. the point-to-patch forms of a weld are the beam from the patch to the grid point (${name})`, (t) => {
    const m = measurements(t);
    const all = POINT_WELDS.map((form) => {
      const sheets = weldSheets({ lowerShape: form.lowerShape, upperCells: 4, at: WELD.centre, held: "all",
        lowerThickness: 0.02, weldYoung: 7.0e7 });
      const results = weldResults(sheets, form.write, options, { ends: [0, sheets.gridB] });
      checkWeldBeam(m, form.name, sheets, results, WELD.gap, POINT_WELD_TOLERANCE[name]);
      return results;
    });
    m.small("NodeIdPatch QuadPatch: the results less those of the ElemIdPatch without SHIDB",
      weldDifference(all[0], all[1]), 1, POINT_WELD_TOLERANCE[name]);
    m.small("NodeIdPatch TriaPatch: the results less those of the ElemIdPatch without SHIDB on the triangle",
      weldDifference(all[2], all[3]), 1, POINT_WELD_TOLERANCE[name]);
    m.done();
  });

// A rigid motion of the welded sheets: the grid points of the lower sheet are given the translations of the motion
// u(x) = a + ω × x, the upper sheet hangs on the weld, whose beam (LDMIN = 1.2: 6 mm) is longer than the distance of
// A and B and so has rigid arms. The weld carries nothing: its forces against those of the same motion of one end
// alone, K·|u|.
for (const { name, options } of FORMULATIONS)
  test(`8. a weld with rigid arms: a rigid motion gives no weld force (${name})`, (t) => {
    const sheets = weldSheets();
    const { deck, lower, upper, middle } = sheets;
    deck.add("Weld", 5001, 9, ...SAME_WELDS[3].write(sheets));
    deck.add("WeldProp", 9, 2, q(WELD.diameter), "", "", q(1.2), q(5));
    const shift = [1.0e-3, -2.0e-3, 1.5e-3];
    const turn = [2.0e-3, 1.0e-3, -3.0e-3];
    const motion = (x) => add(shift, cross(turn, x));
    for (const id of lower.nodes)
      motion(deck.positions.get(id)).forEach((value, c) => deck.add("SinglePointConstr", 2, id, c + 1, q(value)));
    const run = runModel(deck.statics([[1, 2, 0, 0]]), options);
    const m = measurements(t);
    const springs = weldSprings(1.2 * WELD.diameter);
    const row = elementRows(run, "Weld").get(5001);
    // Measured: 2e-12 (classic), 6e-11 (mixed) on the motion; 2e-11 and 5e-10 on the forces.
    m.small("the motion of the upper sheet less the rigid motion", fieldError(run, deck, upper.nodes, motion), 1,
      1e-8);
    m.small("the axial force and the shears of the weld", Math.max(Math.abs(row.axial), Math.abs(row.shear1),
      Math.abs(row.shear2)), springs.across * norm(motion(middle)), 1e-7);
    m.small("the torque and the bending moments of the weld", Math.max(Math.abs(row.torque),
      Math.abs(row.bendingMomentA1), Math.abs(row.bendingMomentA2), Math.abs(row.bendingMomentB1),
      Math.abs(row.bendingMomentB2)), springs.twist * norm(turn), 1e-7);
    m.done();
  });

// The forms of the Weld that are refused.
test("8. the weld forms that are not supported are refused with their reason", (t) => {
  const m = measurements(t);
  const refused = [
    { name: "a weld system (MCID)", weld: (s) => ["", "ElemPatch", "", "", 0, s.elementA, s.elementB,
      ...s.middle.map(q)], property: [9, 2, q(WELD.diameter)], code: "unsupported-field",
    says: /a weld system from a coordinate system is not supported/ },
    { name: "a property that is not defined", weld: (s) => SAME_WELDS[3].write(s), property: [8, 2,
      q(WELD.diameter)], code: "missing-property", says: /expected the ID of a WeldProp, found 9/ },
    { name: "a location off the sheets", weld: (s) => ["", "PartPatch", "", "", "", 1, 2, q(0.5), q(0.05), q(0.001)],
      property: [9, 2, q(WELD.diameter)], code: "missing-point", says: /no piercing point on side A/ },
  ];
  for (const form of refused) {
    const sheets = weldSheets();
    sheets.deck.add("Weld", 5001, 9, ...form.weld(sheets));
    sheets.deck.add("WeldProp", ...form.property);
    const said = [];
    for (const { options } of FORMULATIONS) said.push(refusalOf(sheets.deck.statics(), options).solver.join("\n"));
    t.diagnostic(`${form.name}: ${said[0].split("\n").find((line) => line.includes("error [")) ?? said[0]}`
      .slice(0, 300));
    m.truth(`${form.name}: refused with [${form.code}] and its reason in both formulations`, said.every((text) =>
      text.includes(`[${form.code}]`) && form.says.test(text) && text.includes("Weld 5001")));
  }
  m.done();
});

// ---------------------------------------------------------------------------------------------------------------------
// 9. One joint written three ways
//
// The joint of section 8 (the weld inside an element of each sheet, the lower sheet clamped along X = 0, the six
// loads on the corners of the upper sheet), written with:
//   - a Weld of 5 mm of the material of the sheets;
//   - a Fastener at the same place with the stiffnesses of the springs of that beam (section 8): the beam is those
//     six springs at the middle of A-B, exactly;
//   - a Bushing with the same stiffnesses in the basic system between two grid points at A and B, each tied to its
//     sheet as the ClassicFormulation ties a fastener, written by hand with constraint equations. The square of the
//     area of the cross-section about the point (its sides along X and Y, the sides of the elements) has its four
//     corners in the element the point lies in; at the corner k, at (dx_k, dy_k) = (±h, ±h) from the point, the
//     shape functions of the element are N_kj (bilinear: (1 − ξ)(1 − η), ξ(1 − η), ξη, (1 − ξ)η). The grid point at
//     the end of the Bushing takes
//       the mean translation of the corners,                    u = Σ_j (¼ Σ_k N_kj)·u_j;
//       the mean rotation of the shell about X and Y,           θx = Σ_j (¼ Σ_k N_kj)·θx_j, the same for θy;
//       the rotation of the corners about Z (the least squares  θz = Σ_k (dx_k·v_k − dy_k·u_k) / (8·h²),
//       rigid motion of their translations),                    u_k = Σ_j N_kj·u_j, v_k = Σ_j N_kj·v_j.
// The first two are the same matrices in both formulations. The third is the tie of the ClassicFormulation (the
// same result there); the MixedHybridFormulation reads the shell over the square for the Fastener and the Weld, and
// the motion of the grid points for the constraint equations: a different, softer or stiffer, tie.

// The Fastener of this section: the springs of the beam of the weld in the basic system (the weld is along Z).
function equivalentFastener(sheets) {
  const springs = weldSprings(WELD.gap);
  sheets.deck.add("Fastener", 5001, 7, "Elements", sheets.elementA, sheets.elementB, "", "", "",
    ...sheets.middle.map(q));
  sheets.deck.add("FastenerProp", 7, q(WELD.diameter), 0, 1, q(springs.across), q(springs.across), q(springs.along),
    q(springs.bend), q(springs.bend), q(springs.twist));
}

// The Bushing of this section, between the grid points 9002 (A) and 9003 (B).
function equivalentBushing(sheets) {
  const { deck, lower, upper, pointA, pointB } = sheets;
  const springs = weldSprings(WELD.gap);
  deck.node(9002, pointA);
  deck.node(9003, pointB);
  // The frame 0: the basic system. The spring is at the middle of the two grid points (the default).
  deck.add("Bushing", 5001, 7, 9002, 9003, "", "", "", "", 0);
  deck.add("BushProp", 7, "Stiffness", q(springs.across), q(springs.across), q(springs.along), q(springs.bend),
    q(springs.bend), q(springs.twist));
  const size = WELD.side / 3;
  const half = 0.5 * FOOTPRINT_SIDE_PER_DIAMETER * WELD.diameter;
  deck.selections.push("MultiPointConstrSelection;10");
  [[lower, 9002], [upper, 9003]].forEach(([plate, point]) => {
    const grids = [[1, 1], [2, 1], [2, 2], [1, 2]].map(([i, j]) => plate.node(i, j));
    // The corners of the square: their place from the point, and the shape functions of the element there.
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => {
      const [xi, eta] = [WELD.inside[0] + sx * half, WELD.inside[1] + sy * half].map((x) => x / size - 1);
      return { dx: sx * half, dy: sy * half, n: [(1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta] };
    });
    // One equation: −(component of the point) + Σ coefficient·(component of a grid point) = 0.
    const equation = (component, terms) => deck.add("MultiPointConstr", 10, point, component, q(-1),
      ...terms.flatMap(([grid, of, coefficient]) => [grid, of, q(coefficient)]));
    const mean = grids.map((_, j) => corners.reduce((sum, corner) => sum + corner.n[j] / 4, 0));
    for (const component of [1, 2, 3, 4, 5]) equation(component, grids.map((grid, j) => [grid, component, mean[j]]));
    equation(6, grids.flatMap((grid, j) => [
      [grid, 2, corners.reduce((sum, corner) => sum + corner.n[j] * corner.dx, 0) / (8 * half ** 2)],
      [grid, 1, -corners.reduce((sum, corner) => sum + corner.n[j] * corner.dy, 0) / (8 * half ** 2)]]));
  });
}

// The translations of the corners of the upper sheet under the six loads, and the force and the moment the
// connector carries at the middle (in the basic system), read from the table of `card`.
function threeWays(write, card, options) {
  const sheets = weldSheets();
  const { deck, upper, middle } = sheets;
  write(sheets);
  WELD_LOADS.forEach(({ force, moment }, c) => applyWrench(deck, 11 + c, upper.corners, force, moment, middle));
  const run = runModel(deck.statics(WELD_LOADS.map((_, c) => [1 + c, 1, 11 + c, 0])), options);
  return WELD_LOADS.map((_, c) => ({
    corners: upper.corners.flatMap((id) => nodal(run, "displacements", id, 1 + c).slice(0, 3)),
    carried: card === "Fastener" ? fastenerForce(elementRows(run, "Fastener", "forces", 1 + c), 5001) : null }));
}

const cornerDifference = (first, second) => Math.max(...first.map((result, c) =>
  worstDifference(result.corners, second[c].corners)));

for (const { name, options } of FORMULATIONS)
  test(`9. one joint written with a Weld and with a Fastener of the stiffnesses of its beam (${name})`, (t) => {
    const m = measurements(t);
    const weld = threeWays((sheets) => {
      sheets.deck.add("Weld", 5001, 9, ...SAME_WELDS[3].write(sheets));
      sheets.deck.add("WeldProp", 9, 2, q(WELD.diameter));
    }, "Weld", options);
    const fastener = threeWays(equivalentFastener, "Fastener", options);
    // Measured: 8e-13 (classic), 7e-12 (mixed).
    m.small("the corners of the upper sheet: the Fastener less the Weld", cornerDifference(weld, fastener), 1, 1e-6);
    // The Fastener carries the load (in the basic system, its system).
    WELD_LOADS.forEach(({ force, moment }, c) => m.small(`load ${c + 1}: the forces of the Fastener less the load`,
      Math.max(...fastener[c].carried.map((value, k) => Math.abs(value - [...force, ...moment][k]) /
        ((c < 3) === (k < 3) ? 1 : c < 3 ? WELD.side : 1 / WELD.side))), c < 3 ? 100 : 1, 1e-7));
    m.done();
  });

for (const { name, options } of FORMULATIONS) {
  const limit = name === "MixedHybridFormulation" ? { todo: "the corners of the upper sheet move up to 17 % " +
    "differently: the Fastener reads the displacement and the rotation of the shell over its footprint, the " +
    "constraint equations the motion of the four grid points of the element (the tie of the ClassicFormulation)" }
    : {};
  test(`9. one joint written with a Fastener and with a Bushing tied by constraint equations (${name})`, limit, (t) => {
    const m = measurements(t);
    const fastener = threeWays(equivalentFastener, "Fastener", options);
    const bushing = threeWays(equivalentBushing, "Bushing", options);
    WELD_LOADS.forEach((_, c) => t.diagnostic(`load ${c + 1}: the corners differ by ` +
      worstDifference(fastener[c].corners, bushing[c].corners).toExponential(2)));
    // Measured (classic): 2e-12.
    m.small("the corners of the upper sheet: the Bushing less the Fastener", cornerDifference(fastener, bushing), 1,
      1e-6);
    m.done();
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 10. Modes
//
// The mass of a fastener (FastenerProp MASS) is lumped half at A and half at B. Three checks of it:
//   - under gravity the supports carry the weight of the plates and of the fastener, and the fastener itself the
//     weight of the upper plate and of its own half at B;
//   - the joint free in space has six rigid-body modes at zero frequency and no other (no mechanism in the joint);
//   - a stiff plate on one soft fastener at its centre is a mass on a spring along the fastener:
//     f = sqrt(KT1/(m + MASS/2))/(2π), m the mass of the plate.

const FASTENER_MASS = 0.2;
const GRAVITY = 9.81;

for (const { name, options } of FORMULATIONS)
  test(`10. the mass of a fastener weighs on the supports, half of it through the fastener (${name})`, (t) => {
    const { deck, lower, side } = singleJoint({ thickness: 0.002, kt: RIVET.kt, kr: RIVET.kr, mass: FASTENER_MASS });
    deck.hold(123456, lower.nodes);
    deck.add("GravityLoad", 2, "", q(GRAVITY), q(0), q(0), q(-1));
    const run = runModel(deck.statics(), options);
    const plate = ALUMINIUM.density * side * side * 0.002;
    const m = measurements(t);
    // Measured: 1.000000 in both formulations.
    m.ratio("the vertical support force over the weight of the two plates and of the fastener",
      resultant(run, deck, "singlePointConstrForces")[2], (2 * plate + FASTENER_MASS) * GRAVITY, 1e-8);
    // The fastener is along Z: fx is its axial force, in compression under the upper plate.
    m.ratio("the axial force of the fastener over the weight of the upper plate and of half the fastener",
      fastenerForce(elementRows(run, "Fastener"), 5001)[0], -(plate + FASTENER_MASS / 2) * GRAVITY, 1e-8);
    m.done();
  });

// The natural frequencies (Hz) of the `count` first modes of a modal model.
const frequencies = (text, options) => runModel(text, options, "modes").eigenvalues[0].rows.map((row) => row.cycles);

for (const { name, options } of FORMULATIONS)
  test(`10. two plates joined by one fastener, free in space: six rigid-body modes and no mechanism (${name})`, (t) => {
    const { deck } = singleJoint({ thickness: 0.002, kt: RIVET.kt, kr: RIVET.kr, mass: FASTENER_MASS });
    const found = frequencies(deck.modes(8), options);
    t.diagnostic(`frequencies: ${found.map((f) => f.toPrecision(4)).join(", ")} Hz`);
    const m = measurements(t);
    // Measured: the sixth frequency is 3e-7 (classic) and 2e-6 (mixed) of the seventh.
    m.small("the sixth frequency against the seventh", found[5], found[6], 1e-3);
    m.truth("the seventh mode is elastic (above 10 Hz)", found[6] > 10);
    m.done();
  });

for (const { name, options } of FORMULATIONS)
  test(`10. a stiff plate on one soft fastener with a mass: the frequency along the fastener (${name})`, (t) => {
    const m = measurements(t);
    const plate = ALUMINIUM.density * 0.12 * 0.12 * 0.01;
    for (const mass of [0, FASTENER_MASS]) {
      // The fastener at the centre of the plates, 10 mm thick, with the soft springs of section 1.
      const joint = twoPlates({ thickness: 0.01 });
      const { deck, lower, side, gap } = joint;
      deck.add("Fastener", 5001, 7, "Property", 1, 2, "", "", "", q(side / 2), q(side / 2), q(gap / 2));
      deck.add("FastenerProp", 7, q(0.004), "", "", ...SINGLE.kt.map(q), ...SINGLE.kr.map(q), q(mass));
      deck.hold(123456, lower.nodes);
      const found = frequencies(deck.modes(6, 1), options);
      const expected = Math.sqrt(SINGLE.kt[0] / (plate + mass / 2)) / (2 * Math.PI);
      const nearest = found.reduce((best, f) => (Math.abs(f - expected) < Math.abs(best - expected) ? f : best));
      t.diagnostic(`MASS ${mass}: frequencies ${found.map((f) => f.toPrecision(5)).join(", ")} Hz, expected ` +
        `${expected.toPrecision(5)} Hz along the fastener`);
      // Measured: 0.99999 (classic), 0.99996 (mixed): the plate is not quite rigid.
      m.ratio(`MASS ${mass}: the frequency of the mode along the fastener`, nearest, expected, 1e-3);
    }
    m.done();
  });

// The first bending frequency of the riveted skins of section 5 (clamped at X = 0): between that of two separate
// strips, f = (1.8751²/(2π·L²))·sqrt(E·t²/(12·ρ)) (the first mode of a cantilever beam: S. Timoshenko, Vibration
// Problems in Engineering), and that of the monolithic strip of thickness 2t, twice as high, rising with the
// stiffness of the fasteners.
for (const { name, options } of FORMULATIONS)
  test(`10. two riveted skins: the first bending frequency rises from two plates to the monolithic one (${name})`,
    (t) => {
      const m = measurements(t);
      const { length, thickness, young, density } = SKINS;
      const separate = 1.8751 ** 2 / (2 * Math.PI * length ** 2) * Math.sqrt(young * thickness ** 2 / (12 * density));
      const stiffnesses = [1.0e3, 1.0e6, 1.0e9];
      // Eight cells and eight fasteners along the strips.
      const found = stiffnesses.map((stiffness) => frequencies(rivetedSkins({ stiffness, count: 8, cells: 8,
        density }).deck.modes(1, 1), options)[0]);
      t.diagnostic(`frequencies ${found.map((f) => f.toFixed(3)).join(", ")} Hz for K = 1e3, 1e6, 1e9; two ` +
        `separate strips ${separate.toFixed(3)} Hz, monolithic ${(2 * separate).toFixed(3)} Hz`);
      // Measured: 0.9936 and 0.9938 (classic: eight cells, the mass at the grid points), 1.0007 and 0.9919 (mixed).
      m.ratio("soft fasteners: the frequency over that of two separate strips", found[0], separate, 0.01);
      m.ratio("stiff fasteners: the frequency over that of the monolithic strip", found[2], 2 * separate, 0.02);
      m.truth("the frequency rises with the stiffness of the fasteners", found[0] < found[1] && found[1] < found[2]);
      m.between("the frequency with fasteners of 1e6 N/m, over that of two separate strips", found[1] / separate,
        1.05, 1.95);
      m.done();
    });

// ---------------------------------------------------------------------------------------------------------------------
// 11. Buckling
//
// The riveted skins of section 5 as a column: clamped at X = 0, compressed at their tips. A fastener has no
// differential stiffness, so the critical load only sees the stiffness it gives the pair: between that of two
// separate strips, P = π²·EI0/(4·L²) (Euler's load of a clamped-free column), and that of the monolithic strip, four
// times as high.
for (const { name, options } of FORMULATIONS)
  test(`11. two riveted skins in compression buckle between two plates and the monolithic one (${name})`, (t) => {
    const m = measurements(t);
    const separate = Math.PI ** 2 * SKINS.ei / (4 * SKINS.length ** 2);
    const stiffnesses = [1.0e3, 1.0e6, 1.0e9];
    const found = stiffnesses.map((stiffness) => {
      // Eight cells and eight fasteners along the strips.
      const skins = rivetedSkins({ stiffness, count: 8, cells: 8 });
      // A compression of 1 N in all: the factor is the critical load.
      for (const id of skins.tips) skins.deck.force(2, id, [-0.25, 0, 0]);
      return runModel(skins.deck.buckling(1), options, "buckling").eigenvalues[0].rows[0].eigenvalue;
    });
    t.diagnostic(`critical loads ${found.map((f) => f.toFixed(2)).join(", ")} N for K = 1e3, 1e6, 1e9; two separate ` +
      `strips ${separate.toFixed(2)} N, monolithic ${(4 * separate).toFixed(2)} N`);
    // Measured: 1.0048 and 1.0042 (classic), 1.0016 and 0.9904 (mixed).
    m.ratio("soft fasteners: the critical load over that of two separate strips", found[0], separate, 0.01);
    m.ratio("stiff fasteners: the critical load over that of the monolithic strip", found[2], 4 * separate, 0.03);
    m.truth("the critical load rises with the stiffness of the fasteners", found[0] < found[1] &&
      found[1] < found[2]);
    m.between("the critical load with fasteners of 1e6 N/m, over that of two separate strips", found[1] / separate,
      1.1, 3.9);
    m.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 12. Other paths
//
// The streamed statics: the joint of section 3, solved without a factor of the whole model, against the full path.
for (const { name, options } of FORMULATIONS)
  test(`12. the streamed path of a riveted joint is the full path (${name})`, (t) => {
    const joint = locatedJoint(LOCATIONS[0].write);
    const [full, streamed] = withModel(joint.deck.statics(), options, (model) =>
      [solver.run(model, "statics"), solver.run(model, "statics", { streamed: true })]);
    for (const run of [full, streamed])
      assert.equal(run.diagnostics.filter((d) => d.severity === "error").length, 0, JSON.stringify(run.diagnostics));
    const motions = (run) => joint.upper.nodes.flatMap((id) => nodal(run, "displacements", id).slice(0, 3));
    const forces = (run) => fastenerForce(elementRows(run, "Fastener"), 5001);
    const m = measurements(t);
    // Measured: 1e-13 and 7e-13 (classic), 6e-12 and 2e-11 (mixed).
    m.small("the translations of the upper plate: streamed less full", worstDifference(motions(full),
      motions(streamed)), 1, 1e-8);
    m.small("the fastener forces: streamed less full", worstDifference(forces(full), forces(streamed)), 1, 1e-8);
    m.done();
  });

// A uniform temperature rise of a free joint of one material: the two plates grow alike about the same point, the
// fasteners (four, with no thermal expansion of their own) see no relative motion and carry nothing. The lower plate
// is held without restraint (three translations at a corner, two at another, one at a third). The forces are
// compared with those the growth of one plate alone over the pitch of the fasteners would give, K·α·ΔT·p.
for (const { name, options } of FORMULATIONS)
  test(`12. a uniform temperature rise of a free riveted joint gives no fastener force (${name})`, (t) => {
    const { deck, lower, upper, side, gap } = twoPlates();
    const rise = 100;
    const pitch = side / 2;
    deck.add("FastenerProp", 7, q(0.004), "", "", ...RIVET.kt.map(q), ...RIVET.kr.map(q));
    const rivets = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]].map(([a, b], k) => {
      deck.add("Fastener", 5001 + k, 7, "Property", 1, 2, "", "", "", q(side * a + 0.003), q(side * b - 0.002),
        q(gap / 2));
      return 5001 + k;
    });
    deck.hold(123, [lower.node(0, 0)]);
    deck.hold(23, [lower.node(3, 0)]);
    deck.hold(3, [lower.node(0, 3)]);
    deck.add("DefaultTemperatureLoad", 4, q(rise));
    const run = runModel(deck.statics([[1, 1, 0, 4]]), options);
    const m = measurements(t);
    const growth = ALUMINIUM.expansion * rise;
    // Both plates grow about the corner that is held. Measured: 7e-13 (classic), 3e-12 (mixed).
    m.small("the translations of the two plates less the free growth", fieldError(run, deck, [...lower.nodes,
      ...upper.nodes], (x) => [growth * x[0], growth * x[1], 0]), 1, 1e-7);
    const rows = elementRows(run, "Fastener");
    const worst = (from, to) => Math.max(...rivets.flatMap((id) => fastenerForce(rows, id).slice(from, to)
      .map(Math.abs)));
    // Measured: 4e-14 and 7e-15 (classic), 3e-13 and 3e-14 (mixed).
    m.small("the forces of the fasteners", worst(0, 3), Math.min(...RIVET.kt) * growth * pitch, 1e-7);
    m.small("the moments of the fasteners", worst(3, 6), Math.min(...RIVET.kt) * growth * pitch * gap, 1e-7);
    m.done();
  });

// ---------------------------------------------------------------------------------------------------------------------
// 13. A flange bolted to a skin along one of its edges
//
// A skin of 8 × 4 cells of 30 mm, 2 mm thick, clamped on its outline, and a flange one cell wide lying 2 mm above
// it along X, from the middle line of the skin to the next line of its mesh. The flange is bolted to the skin along
// its first edge, a bolt at each grid point of that edge (a pitch of one cell), and lifted by its other edge: it
// bends as a cantilever from the line of the bolts, which pry on the skin. The usual mesh of the cap of a spar or
// of a stiffener in an aircraft model: one element across, the bolts on the grid points of an edge.
//
// The bolts are written twice with the same six stiffnesses in the basic system: as a Bushing between the grid
// point of the skin and that of the flange, and as a Fastener that pierces the two sheets there. The two are the
// same joint to the size of the tie (a grid point, or the square of the fastener).
//
// What this guards: a piercing point takes the rotation of the shell, that of the grid points, not the slope of the
// sheet between its grid points. Read as that slope, the rotation of a bolt on the edge of the flange is the lift
// of the other edge over the width of the flange, and the arm and the rotational springs of each bolt tie that
// other edge to the skin: the flange no longer bends. Measured, the lift of the edge with fasteners over that with
// bushings: 0.965 with rivets and 0.930 with bolts a hundred times stiffer (read as the slope of the sheet: 0.615
// and 0.211). The two ways of writing a bolt do not agree better than that: a Bushing loads each sheet at a grid
// point, a Fastener the skin over its square. Neither lift is that of a resolved flange: with rivets it is
// 2.72e-5 m on these cells and 3.46e-5, 3.99e-5 and 4.70e-5 with each cell cut in 2, 4 and 8 (bushings: 2.82e-5
// to 5.28e-5); the MixedHybridFormulation gives 3.80e-5 on these cells and 3.98e-5 with each cell cut in 2.

const FLANGE = { cell: 0.03, nx: 8, ny: 4, gap: 0.002, lift: 10 };

// The flange bolted by `joint` ("bushings" or "fasteners"), the stiffnesses of a rivet times `stiffness`. Returns
// the deck and the grid points of the lifted edge.
function boltedFlange(joint, stiffness) {
  const { cell, nx, ny, gap, lift } = FLANGE;
  const [length, width] = [nx * cell, ny * cell];
  const deck = new Deck();
  const skin = sheet(deck, { nx, ny, position: (a, b) => [length * a, width * b, 0], property: 1 });
  const flange = sheet(deck, { nx, ny: 1, position: (a, b) => [length * a, width / 2 + cell * b, gap], property: 2,
    firstNode: 20000, firstElement: 20000 });
  for (const property of [1, 2]) deck.add("ShellProp", property, 1, q(0.002), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density));
  // In the basic system: across the bolt along X and Y, along it (Z), then the rotations about them.
  const springs = [RIVET.kt[1], RIVET.kt[2], RIVET.kt[0], RIVET.kr[1], RIVET.kr[2], RIVET.kr[0]].map((k) =>
    q(stiffness * k));
  for (let i = 0; i <= nx; ++i) {
    const [below, above] = [skin.node(i, ny / 2), flange.node(i, 0)];
    if (joint === "bushings") deck.add("Bushing", 5001 + i, 7, below, above, "", "", "", "", 0);
    else deck.add("Fastener", 5001 + i, 7, "Property", 1, 2, "", "", "", q(length * i / nx), q(width / 2), q(gap / 2));
  }
  if (joint === "bushings") deck.add("BushProp", 7, "Stiffness", ...springs);
  else deck.add("FastenerProp", 7, q(0.004), 0, 1, ...springs);
  deck.hold(123456, skin.nodes.filter((id) => {
    const [x, y] = deck.positions.get(id);
    return Math.min(x, y, length - x, width - y) < 1e-9;
  }));
  const lifted = Array.from({ length: nx + 1 }, (_, i) => flange.node(i, 1));
  lifted.forEach((id, i) => deck.force(2, id, [0, 0, lift / nx * (i === 0 || i === nx ? 0.5 : 1)]));
  return { deck, lifted };
}

for (const [name, stiffness] of [["rivets", 1], ["bolts a hundred times stiffer", 100]])
  test(`13. a flange bolted along one edge by ${name}: fasteners against springs between the grid points ` +
    "(ClassicFormulation)", (t) => {
    const m = measurements(t);
    const lift = (joint) => {
      const { deck, lifted } = boltedFlange(joint, stiffness);
      const run = runModel(deck.statics(), CLASSIC);
      return lifted.reduce((sum, id) => sum + nodal(run, "displacements", id)[2], 0) / lifted.length;
    };
    const [bushings, fasteners] = [lift("bushings"), lift("fasteners")];
    t.diagnostic(`the mean lift of the edge: ${bushings.toExponential(4)} (bushings), ${fasteners.toExponential(4)} ` +
      "(fasteners)");
    m.ratio("the lift of the edge, fasteners over bushings", fasteners, bushings, 0.10);
    m.done();
  });
