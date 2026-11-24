// Robustness and consistency of MixedHybridFormulation on small assembled models (specification
// docs/mixed-hybrid-formulation-spec.md, sections 5 and 13.4). These checks need no closed form: global equilibrium,
// the same results on every path (full, condensed, streamed, envelopes), invariance under renumbering, rigid motion
// and unit scaling, determinism, the classic formulation on a finer mesh as a yardstick, rigid-body modes and mass,
// robustness of the input, and the behaviour of the interface.
//
// A case that fails here is a finding on the solver, kept as it is: the summary printed by the last case gives every
// check with its largest difference.
//
// Run: node --test wasm/test/validation/assemblies.test.mjs   (Node.js 24; needs ../O_o_Model/build/wasm/dist)
// OOS_ASSEMBLIES_DUMP=<folder> writes the model file of every failing comparison there, with its options.

import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { oor, solver, real, withModel, CRITERIA, prepared, surfaceRecords, test, unprepared, withSurface }
  from "./support.mjs";

const MIXED_NAME = "MixedHybridFormulation";
// The relative difference below which two results are one (round-off), section 13.6.
const SAME = CRITERIA.invariant;
// The agreement asked between the mixed formulation and the classic one on a finer mesh.
const YARDSTICK = 0.05;
const DUMP_FOLDER = process.env.OOS_ASSEMBLIES_DUMP ?? "";

// ---------------------------------------------------------------------------------------------------------------------
// Small vector helpers.

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => scale(a, 1 / norm(a));
const apply = (m, a) => [dot(m[0], a), dot(m[1], a), dot(m[2], a)];
const transpose = (m) => [0, 1, 2].map((i) => [m[0][i], m[1][i], m[2][i]]);
const IDENTITY_MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// The rotation of `angle` radians about the unit vector `axis` (Rodrigues' formula).
function rotationAbout(axis, angle) {
  const [x, y, z] = unit(axis);
  const [c, s] = [Math.cos(angle), Math.sin(angle)];
  const t = 1 - c;
  return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
}

// A deterministic generator of numbers in [0, 1) (mulberry32), for the shuffles.
function randomOf(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(items, seed) {
  const random = randomOf(seed);
  const result = [...items];
  for (let k = result.length - 1; k > 0; --k) {
    const other = Math.floor(random() * (k + 1));
    [result[k], result[other]] = [result[other], result[k]];
  }
  return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// A model as tagged records. A generator writes each field of a card with what it is (a grid point ID, an element ID,
// a position, a direction, a quantity with its dimension), so that one generator gives the same assembly renumbered,
// moved rigidly, in other units, with its elements started from another corner and its cards in another order.

const grid = (id) => ({ kind: "grid", id });
const element = (id) => ({ kind: "element", id });
const position = (p) => ({ kind: "position", vector: p });
// A direction is turned with the model; `exponent` scales it as a length (an offset) and `gap` leaves a blank field
// after its first component (the orientation of a bar: x, the alternative grid point, y, z).
const direction = (v, { exponent = 0, gap = false } = {}) => ({ kind: "direction", vector: v, exponent, gap });
// A quantity of dimension force^a × length^exponent × time^c: only the length changes in the unit variants.
const quantity = (value, exponent = 0) => ({ kind: "quantity", value, exponent });
const q = quantity;

// A variant of an assembly: how its IDs, positions and units are written. The assembly itself is the same.
const BASE_VARIANT = { gridId: (id) => id, elementId: (id) => id, rotation: IDENTITY_MATRIX, translation: [0, 0, 0],
  lengthScale: 1, cornerShift: 0, shuffleSeed: 0 };

class Assembly {
  constructor(variant = {}) {
    this.variant = { ...BASE_VARIANT, ...variant };
    this.executive = [];
    this.caseLines = [];
    this.records = [];
    this.points = new Map(); // grid point ID (of the base variant) → position (base variant)
    this.frames = new Map(); // displacement frame ID → { kind, origin, axis, inPlane } (base variant)
    this.displacementFrame = new Map(); // grid point ID → displacement frame ID
    // The declarations of the geometry, in the IDs of the base variant: raw and sharp edges and line ends as pairs
    // of grid points (a line end: the line of the edge of the two ends at the second one; a sharp edge may name, as
    // a third item, the one shell that ends there at a junction), raw elements.
    this.rawEdges = [];
    this.rawElements = [];
    this.sharpEdges = [];
    this.lineEnds = [];
    this.keys = {}; // named grid points and values the checks read
    this.mass = 0; // the mass computed by hand, summed by the generator
    this.size = 1; // a length of the model, for the scales of the moments
    this.counts = {}; // elements per card
    this.resultants = new Map(); // load set → the resultant of its forces and couples about the origin
    this.stated = new Map(); // subcase → { force, moment (or null: not computed by hand) } of its loads
    this.internalForce = new Map(); // subcase → the scale of the internal forces of a subcase without loads
  }

  // A force or a couple at a grid point, added to the resultant of its load set.
  force(set, node, f) {
    this.add("PointForceLoad", set, grid(node), "", q(1), direction(f));
    this.addResultant(set, f, cross(this.points.get(node), f));
  }

  couple(set, node, m) {
    this.add("PointMomentLoad", set, grid(node), "", q(1, 1), direction(m));
    this.addResultant(set, [0, 0, 0], m);
  }

  addResultant(set, f, m) {
    const total = this.resultants.get(set) ?? [0, 0, 0, 0, 0, 0];
    this.resultants.set(set, total.map((value, c) => value + (c < 3 ? f[c] : m[c - 3])));
  }

  // States the loads of `subcase` as the resultant of load set `set`.
  state(subcase, set) {
    const total = this.resultants.get(set);
    this.stated.set(subcase, { force: total.slice(0, 3), moment: total.slice(3) });
  }

  add(name, ...fields) {
    this.records.push([name, ...fields]);
    if (/^(Shell|Solid|Bar2|Beam2|Rod2|Spring|Bushing|Fastener|Weld)/.test(name) && !/Prop/.test(name))
      this.counts[name] = (this.counts[name] ?? 0) + 1;
  }

  node(id, p, frame = "") {
    this.points.set(id, p);
    if (frame !== "") this.displacementFrame.set(id, frame);
    this.add("Node", grid(id), "", position(p), frame);
  }

  // A rectangular or cylindrical frame through three points: its origin, a point of its z axis and a point of its
  // xz plane.
  frame(id, kind, origin, onAxis, inPlane) {
    this.frames.set(id, { kind, origin, onAxis, inPlane });
    this.add(kind === "cylindrical" ? "CylindricalFrame2" : "RectFrame2", id, 0, position(origin), position(onAxis),
      position(inPlane));
  }

  // The axes of the displacement system of a grid point of the base variant, as the rows of a matrix (basic → local).
  localAxes(id) {
    const frame = this.frames.get(this.displacementFrame.get(id));
    if (!frame) return IDENTITY_MATRIX;
    const z = unit(sub(frame.onAxis, frame.origin));
    if (frame.kind === "cylindrical") {
      const radial = sub(this.points.get(id), frame.origin);
      const r = unit(sub(radial, scale(z, dot(radial, z))));
      return [r, cross(z, r), z];
    }
    const inPlane = sub(frame.inPlane, frame.origin);
    const x = unit(sub(inPlane, scale(z, dot(inPlane, z))));
    return [x, cross(z, x), z];
  }

  fieldText(field) {
    const v = this.variant;
    if (field === undefined || field === null || field === "") return [""];
    if (typeof field !== "object") return [String(field)];
    switch (field.kind) {
    case "grid": return [String(v.gridId(field.id))];
    case "element": return [String(v.elementId(field.id))];
    case "position": return add(apply(v.rotation, field.vector), v.translation).map((c) => real(c * v.lengthScale));
    case "direction": {
      const parts = apply(v.rotation, field.vector).map((c) => real(c * v.lengthScale ** field.exponent));
      return field.gap ? [parts[0], "", parts[1], parts[2]] : parts;
    }
    case "quantity": return [real(field.value * v.lengthScale ** field.exponent)];
    default: throw new Error(`unknown field ${JSON.stringify(field)}`);
    }
  }

  // The model file. Its model section holds the declarations of the geometry in the IDs of the variant, as a
  // preprocessor writes them (the ClassicFormulation does not read them).
  text() {
    const lines = this.records.map((record) => record.flatMap((field) => this.fieldText(field)).join(";")
      .replace(/;+$/, ""));
    const edge = ([a, b, shell]) => [this.variant.gridId(a), this.variant.gridId(b),
      ...(shell === undefined ? [] : [this.variant.elementId(shell)])];
    lines.push(...surfaceRecords({ rawEdges: this.rawEdges.map(edge), sharpEdges: this.sharpEdges.map(edge),
      rawElements: this.rawElements.map((id) => this.variant.elementId(id)), lineEnds: this.lineEnds.map(edge) }));
    const ordered = this.variant.shuffleSeed ? shuffled(lines, this.variant.shuffleSeed) : lines;
    return ["[executive]", ...this.executive, "[case]", ...this.caseLines, "[model]", ...prepared(ordered), ""]
      .join("\n");
  }

  // The options of modelFromReader: the formulation.
  options(formulation = MIXED_NAME, extra = {}) {
    return formulation === MIXED_NAME ? { elementFormulation: MIXED_NAME, ...extra } : { ...extra };
  }

  elementCount() { return Object.values(this.counts).reduce((a, b) => a + b, 0); }
}

// The corners of a shell started `shift` corners further (the same element, the same normal).
const turned = (nodes, shift) => nodes.map((_, k) => nodes[(k + shift) % nodes.length]);

// The edges of each solid shape in the order of the mid-side nodes of its card, as pairs of corner indices.
const SOLID_EDGES = {
  SolidTetra: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
  SolidPyramid: [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
  SolidPenta: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5], [5, 3]],
  SolidHexa: [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]],
};
// One turn of each shape onto itself (about its axis), as the corner each corner takes.
const SOLID_TURN = { SolidTetra: [1, 2, 0, 3], SolidPyramid: [1, 2, 3, 0, 4], SolidPenta: [1, 2, 0, 4, 5, 3],
  SolidHexa: [1, 2, 3, 0, 5, 6, 7, 4] };
const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

// Adds a solid of `card` on `corners`, turned `shift` times, with the mid-side nodes `mids` (edge key → grid point)
// that lie on its edges: a partial set where only some edges have one.
function addSolid(model, card, id, property, corners, mids = new Map()) {
  let nodes = corners;
  for (let turn = 0; turn < model.variant.cornerShift % 3; ++turn) nodes = SOLID_TURN[card].map((k) => nodes[k]);
  const middle = SOLID_EDGES[card].map(([a, b]) => mids.get(edgeKey(nodes[a], nodes[b])));
  const last = middle.findLastIndex((m) => m !== undefined);
  model.add(card, element(id), property, ...nodes.map(grid),
    ...middle.slice(0, last + 1).map((m) => (m ? grid(m) : "")));
}

// ---------------------------------------------------------------------------------------------------------------------
// Running and reading.

let dumpCount = 0;

// Writes the model file of a failing check where OOS_ASSEMBLIES_DUMP says, with its options.
function dump(name, text, options, note = "") {
  if (!DUMP_FOLDER) return;
  mkdirSync(DUMP_FOLDER, { recursive: true });
  const stem = path.join(DUMP_FOLDER, `${String(++dumpCount).padStart(2, "0")}-${name.replace(/[^A-Za-z0-9]+/g, "-")}`);
  writeFileSync(`${stem}.oos`, text);
  writeFileSync(`${stem}.options.json`, JSON.stringify({ options, note }, null, 1));
}

const errorsOf = (run) => run.diagnostics.filter((d) => d.severity === "error");

// Builds the model of `assembly` and runs `action(model)`.
function withAssembly(assembly, formulation, action, extra = {}) {
  return withModel(assembly.text(), assembly.options(formulation, extra), action);
}

// Runs an analysis of `assembly`; an error diagnostic fails the case. The time of the run is kept in `timings`.
const timings = [];
function runAssembly(assembly, formulation = MIXED_NAME, analysis = "statics", settings = undefined, extra = {}) {
  return withAssembly(assembly, formulation, (model) => {
    const start = performance.now();
    const run = solver.run(model, analysis, settings);
    const seconds = (performance.now() - start) / 1000;
    timings.push({ model: assembly.name ?? "", formulation, analysis, settings: settingsLabel(settings), seconds,
      elements: assembly.elementCount() });
    assert.equal(errorsOf(run).length, 0, JSON.stringify(errorsOf(run)));
    return run;
  }, extra);
}

function settingsLabel(settings) {
  if (!settings) return "full";
  if (settings.streamed) return "streamed";
  if (settings.condensation) {
    const c = settings.condensation;
    return `condensed${typeof c === "object" && c.memoryMode ? ` ${c.memoryMode}` : ""}` +
      (settings.keepsCaseTables === false ? " unit by unit" : "");
  }
  return "full";
}

const subcasesOf = (run) => [...new Set(run.nodalTables.map((t) => t.subcase))].sort((a, b) => a - b);
const nodalTable = (run, kind, subcase, step = 0) =>
  run.nodalTables.find((t) => t.kind === kind && t.subcase === subcase && t.step === step);

// The six values of each grid point of a nodal table, by ID.
function rowsById(table) {
  const rows = new Map();
  table.ids.forEach((id, row) => rows.set(id, Array.from(table.values.slice(row * 6, row * 6 + 6))));
  return rows;
}

// Every value of every table of a run is a finite number.
function assertFinite(run, what) {
  for (const table of [...run.nodalTables, ...run.elementTables])
    table.values.forEach((value, k) => assert.ok(Number.isFinite(value) || /^ms|margin|strainEnergyDensity/i.test(
      table.components[k % table.components.length]),
    `${what}: ${table.kind} ${table.elementType ?? ""} of subcase ${table.subcase} holds ${value} ` +
      `(${table.components[k % table.components.length]})`));
}

// The resultant of a nodal table of forces about the origin, in the basic system: [Fx, Fy, Fz, Mx, My, Mz], and the
// largest force and moment of the table (the scale of its round-off).
function resultant(assembly, table) {
  const total = [0, 0, 0, 0, 0, 0];
  let [largestForce, largestMoment] = [0, 0];
  table.ids.forEach((id, row) => {
    const local = transpose(assembly.localAxes(id));
    const force = apply(local, Array.from(table.values.slice(row * 6, row * 6 + 3)));
    const moment = apply(local, Array.from(table.values.slice(row * 6 + 3, row * 6 + 6)));
    const arm = assembly.points.get(id) ?? [0, 0, 0];
    const about = add(cross(arm, force), moment);
    for (let c = 0; c < 3; ++c) { total[c] += force[c]; total[3 + c] += about[c]; }
    largestForce = Math.max(largestForce, norm(force));
    largestMoment = Math.max(largestMoment, norm(moment), norm(cross(arm, force)));
  });
  return { total, largestForce, largestMoment };
}

// Global equilibrium of a static subcase: the applied loads and the support forces sum to zero in force and in moment
// about the origin. Returns the largest residuals, in force and in moment, against the largest force of the two
// tables (and the largest moment, or that force times the size of the model). A subcase without loads (temperatures)
// has the scale of its internal forces instead, which the assembly states: its reactions may be round-off only.
function equilibriumResiduals(assembly, run, subcase) {
  const applied = resultant(assembly, nodalTable(run, "appliedLoads", subcase));
  const reactions = resultant(assembly, nodalTable(run, "singlePointConstrForces", subcase));
  return residualsOf(assembly, applied, reactions.total, subcase);
}

// The residuals of `loads` (a resultant with its scales) and of the six `balancing` values that should cancel it.
function residualsOf(assembly, loads, balancing, subcase) {
  const forceScale = Math.max(loads.largestForce, norm(balancing.slice(0, 3)), assembly.internalForce.get(subcase) ?? 0,
    Number.MIN_VALUE);
  const momentScale = Math.max(loads.largestMoment, forceScale * assembly.size);
  let [force, moment] = [0, 0];
  for (let c = 0; c < 3; ++c) {
    force = Math.max(force, Math.abs(loads.total[c] + balancing[c]) / forceScale);
    moment = Math.max(moment, Math.abs(loads.total[3 + c] + balancing[3 + c]) / momentScale);
  }
  return { force, moment };
}

// A resultant stated by hand, as the resultant of a table is given (the scales: its own norms).
function statedResultant(stated) {
  const moment = stated.moment ?? [0, 0, 0];
  return { total: [...stated.force, ...moment], largestForce: norm(stated.force), largestMoment: norm(moment) };
}

// The strain energy of a subcase: the sum of the rows of every strain energy table (the subtotal rows left out).
const SUBTOTAL_ID = 100000000;
function strainEnergy(run, subcase) {
  let total = 0;
  for (const table of run.elementTables.filter((t) => t.kind === "strainEnergy" && t.subcase === subcase)) {
    const column = table.components.indexOf("strainEnergy");
    table.ids.forEach((id, row) => {
      if (id !== SUBTOTAL_ID) total += table.values[row * table.components.length + column];
    });
  }
  return total;
}

// Half the work of the applied loads of a subcase on its displacements (both tables in the displacement systems).
function halfWork(run, subcase) {
  const loads = rowsById(nodalTable(run, "appliedLoads", subcase));
  const moves = rowsById(nodalTable(run, "displacements", subcase));
  let work = 0;
  for (const [id, load] of loads) {
    const move = moves.get(id);
    if (move) for (let c = 0; c < 6; ++c) work += load[c] * move[c];
  }
  return work / 2;
}

// ---------------------------------------------------------------------------------------------------------------------
// Comparing two runs table by table: every nodal and element table, the rows matched by their IDs (grid point,
// element, place, layer, station), every component against the largest value of its kind in the table.

// The kind of quantity of a component of an element table, with its dimension as a power of the length (forces
// fixed): the components of one class are compared against the largest of the class in the table. null: left out (the
// principal angles turn by 180 degrees at a point of equal principal values; the margins are ratios of allowables).
function componentClass(table, component) {
  if (/angle/i.test(component) || /^ms[A-Z]/.test(component) || /^sm[at]$/.test(component)) return null;
  // The direction cosines of the principal stresses of a solid: undefined at equal principal values.
  if (/^o[123][xyz]$/.test(component)) return null;
  if (table.elementType === "Fastener" && table.kind !== "strainEnergy")
    return /^moment|^m[xyz]$/.test(component) ? { name: "moment", exponent: 1 } : { name: "force", exponent: 0 };
  // A bushing writes its forces and couples as stresses, its stretches and rotations as strains.
  if (table.elementType === "Bushing" && table.kind === "stress")
    return /^r/.test(component) ? { name: "moment", exponent: 1 } : { name: "force", exponent: 0 };
  if (table.elementType === "Bushing" && table.kind === "strain")
    return /^r/.test(component) ? { name: "rotation", exponent: 0 } : { name: "stretch", exponent: 1 };
  // The strain table of a shell holds membrane strains (flag 0) and curvatures (flag -1): the flag has no unit.
  if (component === "fiberCurvature") return { name: "flag", exponent: 0 };
  if (component === "fiberDistance") return { name: "fibre", exponent: 1 };
  if (table.kind === "strainEnergy") {
    if (component === "percent") return { name: "percent", exponent: 0 };
    return component === "strainEnergy" ? { name: "energy", exponent: 1 } : { name: "energyDensity", exponent: -2 };
  }
  if (table.kind === "stress") return { name: "stress", exponent: -2 };
  if (table.kind === "strain") return { name: "strain", exponent: 0 };
  if (table.kind === "forces") {
    const shell = /^Shell/.test(table.elementType);
    if (/moment|torque|torsion|^m[xyz]$|^mxy$/i.test(component)) return { name: "moment", exponent: shell ? 0 : 1 };
    return { name: "force", exponent: shell ? -1 : 0 };
  }
  return { name: table.kind, exponent: 0 };
}

const tableName = (table) => `${table.kind} ${table.elementType ?? ""} subcase ${table.subcase} step ${table.step}`;
const sameTable = (a, b) => a.kind === b.kind && (a.elementType ?? "") === (b.elementType ?? "") &&
  a.subcase === b.subcase && a.step === b.step;

// The key of row `row` of an element table, its element ID mapped by `elementId` and its grid point by `gridId`.
function elementRowKey(table, row, elementId = (id) => id, gridId = (id) => id) {
  const id = table.ids[row] === SUBTOTAL_ID ? SUBTOTAL_ID : elementId(table.ids[row]);
  const place = table.nodeIds[row] === 0 ? 0 : gridId(table.nodeIds[row]);
  const station = table.positions && table.positions.length ? table.positions[row].toFixed(6) : "";
  // The grid point stresses of the solids have the grid point as their element ID.
  const mapped = table.elementType === "SolidGridPoints" ? gridId(table.ids[row]) : id;
  return `${mapped}|${table.elementType === "SolidGridPoints" ? mapped : place}|${table.layers[row]}|${station}`;
}

// The largest relative difference between the tables of `reference` and of `other`, with what it is. `map` says how
// the second run is read: `gridId` and `elementId` (the IDs of the second run from those of the first),
// `nodal(id, values, kind)` (the six values expected in the second run from those of the first), `lengthScale` (the
// unit of length of the second run in that of the first), `invariantsOnly(table)` (the tables whose components are in
// axes that the map moves: only their invariants are compared), `skip(table)`.
function compareRuns(reference, other, map = {}) {
  const gridId = map.gridId ?? ((id) => id);
  const elementId = map.elementId ?? ((id) => id);
  const lengthScale = map.lengthScale ?? 1;
  const size = map.size ?? 1;
  let worst = { difference: 0, what: "nothing compared", byKind: new Map() };
  let kindNow = "";
  // Keeps the largest difference, and the largest of each kind of table.
  const note = (difference, what) => {
    if (!(difference <= worst.difference)) worst = { ...worst, difference, what };
    if (!(difference <= (worst.byKind.get(kindNow)?.difference ?? -1))) worst.byKind.set(kindNow, { difference, what });
  };
  const forceFloor = map.forceFloor ?? (() => 0);

  for (const table of reference.nodalTables) {
    if (map.skip?.(table)) continue;
    kindNow = table.kind;
    const match = other.nodalTables.find((t) => sameTable(t, table));
    if (!match) { note(Infinity, `${tableName(table)} is missing`); continue; }
    const rows = rowsById(match);
    if (rows.size !== table.ids.length)
      note(Infinity, `${tableName(table)}: ${rows.size} rows for ${table.ids.length}`);
    // Translations (forces) and rotations (moments) have their own scales; a table of rotations that are zero is
    // compared against the translations over the size of the model.
    const moments = table.kind !== "displacements" && table.kind !== "eigenvectors";
    const expected = [];
    let [first, second] = [0, 0];
    table.ids.forEach((id, row) => {
      let values = Array.from(table.values.slice(row * 6, row * 6 + 6));
      if (map.nodal) values = map.nodal(id, values, table.kind);
      expected.push(values);
      first = Math.max(first, ...values.slice(0, 3).map(Math.abs));
      second = Math.max(second, ...values.slice(3).map(Math.abs));
    });
    const sizeNow = size * lengthScale;
    // A table of forces that are round-off (the reactions of a free thermal expansion) has the scale of the internal
    // forces of its subcase.
    if (moments) first = Math.max(first, forceFloor(table.subcase));
    second = Math.max(second, moments ? first * sizeNow : first / sizeNow);
    table.ids.forEach((id, row) => {
      const actual = rows.get(gridId(id));
      if (!actual) { note(Infinity, `${tableName(table)}: no row of grid point ${gridId(id)}`); return; }
      for (let c = 0; c < 6; ++c) {
        const scaleOf = Math.max(c < 3 ? first : second, Number.MIN_VALUE);
        note(Math.abs(actual[c] - expected[row][c]) / scaleOf,
          `${tableName(table)}, grid point ${id}, ${table.components[c]}: ${actual[c]} for ${expected[row][c]}`);
      }
    });
  }

  for (const table of reference.elementTables) {
    if (map.skip?.(table)) continue;
    kindNow = table.kind;
    const match = other.elementTables.find((t) => sameTable(t, table));
    if (!match) { note(Infinity, `${tableName(table)} is missing`); continue; }
    const width = table.components.length;
    if (match.components.join() !== table.components.join()) {
      note(Infinity, `${tableName(table)}: other columns`);
      continue;
    }
    if (match.ids.length !== table.ids.length)
      note(Infinity, `${tableName(table)}: ${match.ids.length} rows for ${table.ids.length}`);
    const rowOf = new Map();
    for (let row = 0; row < match.ids.length; ++row) rowOf.set(elementRowKey(match, row), row);
    const classes = table.components.map((component) => componentClass(table, component));
    const invariantsOnly = map.invariantsOnly?.(table) ?? false;
    const largest = new Map();
    const curvatureFlag = table.kind === "strain" ? table.components.indexOf("fiberCurvature") : -1;
    for (let k = 0; k < table.values.length; ++k) {
      const cls = classes[k % width];
      if (!cls || !Number.isFinite(table.values[k])) continue;
      const row = Math.floor(k / width);
      const name = cls.name === "strain" && curvatureFlag >= 0 && table.values[row * width + curvatureFlag] === -1
        ? "curvature" : cls.name;
      largest.set(name, Math.max(largest.get(name) ?? 0, Math.abs(table.values[k])));
    }
    for (let row = 0; row < table.ids.length; ++row) {
      // A row of curvatures (flag -1) is one over a length.
      const curvature = curvatureFlag >= 0 && table.values[row * width + curvatureFlag] === -1;
      const other = rowOf.get(elementRowKey(table, row, elementId, gridId));
      if (other === undefined) {
        note(Infinity, `${tableName(table)}: no row ${elementRowKey(table, row, elementId, gridId)}`);
        continue;
      }
      for (let c = 0; c < width; ++c) {
        const cls = classes[c];
        if (!cls) continue;
        if (invariantsOnly && !/vonMises|^o?max|^o?min|major|minor|principal|pressure|strainEnergy|percent/i.test(
          table.components[c])) continue;
        const bent = curvature && cls.name === "strain";
        const exponent = bent ? -1 : cls.exponent;
        const expected = table.values[row * width + c] * lengthScale ** exponent;
        const actual = match.values[other * width + c];
        if (!Number.isFinite(expected) && !Number.isFinite(actual)) continue;
        const scaleOf = Math.max((largest.get(bent ? "curvature" : cls.name) ?? 0) * lengthScale ** exponent,
          Number.MIN_VALUE);
        kindNow = table.components[c] === "strainEnergyDensity" ? "strainEnergyDensity" : table.kind;
        note(Math.abs(actual - expected) / scaleOf, `${tableName(table)}, element ${table.ids[row]} at ` +
          `${table.nodeIds[row]}, layer ${table.layers[row]}, ${table.components[c]}: ${actual} for ${expected}`);
      }
    }
  }
  return worst;
}

// The results of the cases, printed at the end: check, model, largest relative difference.
const summary = [];
function record(check, model, value, tolerance, what = "") {
  const pass = value <= tolerance;
  summary.push({ check, model, value, tolerance, pass, what: pass ? "" : what });
  return pass;
}

// Records a check and fails the case when it is over its tolerance.
function expectWithin(check, model, value, tolerance, what = "") {
  const pass = record(check, model, value, tolerance, what);
  assert.ok(pass, `${check} of ${model}: ${value} over ${tolerance}${what ? ` (${what})` : ""}`);
}

// ---------------------------------------------------------------------------------------------------------------------
// The case section of the decks.

// Every output the format offers. The strain energy is asked in the first two subcases only: it forms every element
// matrix again for each subcase, which is most of the time of a run.
const EVERY_OUTPUT = ["DisplacementOutput;All", "SinglePointConstrForceOutput;All", "MultiPointConstrForceOutput;All",
  "AppliedLoadOutput;All", "ElemForceOutput;All;CornerResults", "StressOutput;All;CornerResults",
  "StrainOutput;All;CornerResults"];
const ENERGY_SUBCASES = [1, 2];

// The lines of static subcases: each is [subcase, constraint set, load set or null, temperature set or null].
function staticSubcases(subcases) {
  const lines = [];
  for (const [subcase, constraints, loads, temperatures] of subcases) {
    lines.push(`SubcaseStart;${subcase}`);
    if (ENERGY_SUBCASES.includes(subcase)) lines.push("ElemStrainEnergyOutput;All");
    if (constraints) lines.push(`SinglePointConstrSelection;${constraints}`);
    if (loads) lines.push(`StaticLoadSelection;${loads}`);
    if (temperatures) lines.push(`TemperatureSelection;${temperatures};TemperatureLoadPart`);
  }
  return lines;
}

const EIGEN_METHOD = 3;

// Sets the sections of `model` for an analysis: "statics" with `subcases`; "modes" held by `constraints` (0: free)
// with `modes` roots; "buckling" of the preload [constraints, loads] with `modes` roots.
function setAnalysis(model, analysis, { subcases = [], constraints = 0, loads = 0, modes = 4, global = [] } = {}) {
  if (analysis === "statics") {
    model.executive = ["SolutionSequence;101"];
    model.caseLines = [...EVERY_OUTPUT, ...global, ...staticSubcases(subcases)];
  } else if (analysis === "modes") {
    model.executive = ["SolutionSequence;103"];
    model.caseLines = ["DisplacementOutput;All", ...global, "SubcaseStart;1",
      ...(constraints ? [`SinglePointConstrSelection;${constraints}`] : []), `EigenMethodSelection;${EIGEN_METHOD}`];
  } else {
    model.executive = ["SolutionSequence;105"];
    model.caseLines = ["DisplacementOutput;All", ...global,
      ...staticSubcases([[1, constraints, loads, null]]).filter((line) => !/StrainEnergy/.test(line)),
      "SubcaseStart;2", `SinglePointConstrSelection;${constraints}`, `EigenMethodSelection;${EIGEN_METHOD}`,
      "StaticSubcaseSelection;1"];
  }
  if (analysis !== "statics") model.add("LanczosEigenMethod", EIGEN_METHOD, "", "", modes);
}

// The length of the curve c(s), s in [a, b], by Simpson's rule on `steps` intervals.
function curveLength(curve, a, b, steps = 400) {
  const h = (b - a) / steps;
  const speed = (s) => norm(scale(sub(curve(s + 1e-6), curve(s - 1e-6)), 1 / 2e-6));
  let total = speed(a) + speed(b);
  for (let k = 1; k < steps; ++k) total += (k % 2 ? 4 : 2) * speed(a + k * h);
  return (total * h) / 3;
}

// ---------------------------------------------------------------------------------------------------------------------
// Assembly A: a curved stiffened panel. A panel of nx × ny cells on a cylinder (or on a twisted barrel: every
// quadrilateral warped), its last column of cells cut into triangles; three bands of properties (a plain skin, a
// laminate, a thicker offset skin with a non-structural mass); Bar2 stiffeners with offsets along a line of edges
// and a Beam2 ring across; a tripod of bars alone and a rod standing on the panel with a mass at its apex; a
// separate frame of bars under the panel joined to it by a bushing, springs, a rigid element with dependent
// translations and a multipoint constraint; a rigid spider that carries a load; springs to grounded grid points.

const PANEL = { length: 1.2, radius: 1.5, arc: (50 * Math.PI) / 180, nx: 4, ny: 3, twist: (10 * Math.PI) / 180,
  bulge: 0.08, skin: 0.003, thick: 0.005, thickOffset: 0.0015, thickNsm: 0.8, ply: 0.00025,
  plyAngles: [0, 45, -45, 90, 90, -45, 45, 0], stiffenerOffset: 0.012, apexHeight: 0.3, frameDepth: 0.2 };
const ALUMINIUM = { young: 7.0e10, poisson: 0.33, density: 2700, expansion: 2.3e-5 };
const STEEL = { young: 2.1e11, poisson: 0.3, density: 7800, expansion: 1.2e-5 };
const PLY = { young1: 1.3e11, young2: 9.0e9, poisson: 0.3, shear: 5.0e9, shear1z: 5.0e9, shear2z: 3.0e9,
  density: 1600, expansion1: 1.0e-6, expansion2: 3.0e-5 };
const STIFFENER = { area: 1.2e-4, i1: 6.0e-9, i2: 2.0e-9, j: 1.0e-9 };
const RING = { area: 1.0e-4, i1: 4.0e-9, i2: 4.0e-9, j: 2.0e-9 };
const STRUT = { area: 8.0e-5, i: 3.0e-9, j: 6.0e-9 };
const TIE_ROD = { area: 5.0e-5, j: 1.0e-9 };
const FRAME_BAR = { area: 2.0e-4, i: 2.0e-8, j: 4.0e-8 };
const GRAVITY = { magnitude: 9.81, direction: [0.3, -0.5, -1.0] };

function addMaterials(model) {
  model.add("IsoMat", 1, q(ALUMINIUM.young, -2), "", q(ALUMINIUM.poisson), q(ALUMINIUM.density, -4),
    q(ALUMINIUM.expansion), q(0));
  model.add("IsoMat", 2, q(STEEL.young, -2), "", q(STEEL.poisson), q(STEEL.density, -4), q(STEEL.expansion), q(0));
  model.add("OrthoMat2D", 3, q(PLY.young1, -2), q(PLY.young2, -2), q(PLY.poisson), q(PLY.shear, -2),
    q(PLY.shear1z, -2), q(PLY.shear2z, -2), q(PLY.density, -4), q(PLY.expansion1), q(PLY.expansion2), q(0));
}

// The orientation vector of a bar is given in the basic system, whatever the displacement systems of its grid points.
const BASIC_ORIENTATION = "OffsetsBasicGlobalGlobal";

// The laminate of property `id`: the plies of PANEL.plyAngles, each with its stresses asked.
function addLaminate(model, id) {
  const plies = PANEL.plyAngles.flatMap((angle) => [3, q(PANEL.ply, 1), q(angle), "Yes"]);
  model.add("CompProp", id, "", "", "", "", "", "", "", ...plies);
}

function addBarProp(model, id, material, section) {
  const [c, d] = [0.01, 0.004]; // the stress points, at the corners of a 20 × 8 mm box
  model.add("BarProp", id, material, q(section.area, 2), q(section.i1 ?? section.i, 4), q(section.i2 ?? section.i, 4),
    q(section.j, 4), "", q(c, 1), q(d, 1), q(-c, 1), q(d, 1), q(-c, 1), q(-d, 1), q(c, 1), q(-d, 1));
}

// `shape`: "cylinder" (exact geometry, flat quadrilaterals) or "warped"; `refine`: each cell cut into refine × refine;
// `analysis`: "statics", "modes" (clamped), "freeModes", "buckling"; `frames`: a displacement system on every grid
// point; `part`: "all", or "panel" (the panel with its stiffeners and its tripod only).
// `joints`: "mixed" (a bushing, springs, a rigid element and a multipoint constraint between the frame and the panel)
// or "elastic" (bushings and springs only).
function panelAssembly({ variant = {}, shape = "warped", refine = 1, analysis = "statics", frames = false,
  part = "all", modes = 4, without = [], joints = "mixed" } = {}) {
  const model = new Assembly(variant);
  const has = (feature) => !without.includes(feature);
  model.name = `panel ${shape}${refine > 1 ? ` x${refine}` : ""}`;
  model.size = PANEL.length;
  const [nx, ny] = [PANEL.nx * refine, PANEL.ny * refine];
  const N = (i, j) => 1 + i + j * (nx + 1);
  const key = (i, j) => N(i * refine, j * refine);
  const warped = shape === "warped";
  // The surface: u along the axis, v around it.
  const surface = (u, v) => {
    const turn = v + (warped ? (PANEL.twist * u) / PANEL.length : 0);
    const rho = PANEL.radius + (warped ? PANEL.bulge * Math.sin((Math.PI * u) / PANEL.length) : 0);
    return [u, rho * Math.sin(turn), rho * Math.cos(turn) - PANEL.radius];
  };
  const normal = (u, v) => {
    const h = 1e-6;
    return unit(cross(sub(surface(u + h, v), surface(u - h, v)), sub(surface(u, v + h), surface(u, v - h))));
  };
  const uOf = (i) => (PANEL.length * i) / nx;
  const vOf = (j) => -PANEL.arc / 2 + (PANEL.arc * j) / ny;
  const pointOf = (i, j) => surface(uOf(i), vOf(j));
  const normalOf = (i, j) => normal(uOf(i), vOf(j));

  // The displacement systems: a rectangular one along the clamped edge (parallel systems along a supported edge), a
  // cylindrical one and another rectangular one in turn on the rest of the panel and on the other grid points.
  const AXIS = [[0, 0, -PANEL.radius], [1, 0, -PANEL.radius], [0, 0, 0]];
  if (frames) {
    model.frame(11, "rectangular", [0.1, 0.2, 0.3], [0.4, 0.5, 1.2], [1.3, 0.1, 0.2]);
    model.frame(12, "rectangular", [0, 0, 0], [0.5, -0.3, 0.8], [0.9, 0.4, 0.1]);
    model.frame(13, "cylindrical", ...AXIS);
  }
  const panelFrame = (i, j) => (!frames ? "" : i === 0 ? 11 : (i + j) % 2 ? 13 : 12);
  const otherFrame = frames ? 12 : "";

  for (let j = 0; j <= ny; ++j)
    for (let i = 0; i <= nx; ++i) model.node(N(i, j), pointOf(i, j), panelFrame(i, j));

  // The material frame of the laminate: its x axis along the axis of the panel.
  model.frame(20, "rectangular", [0, 0, 0], [0, 0, 1], [1, 0, 0]);
  addMaterials(model);
  model.add("ShellProp", 1, 1, q(PANEL.skin, 1), 1, "", 1);
  model.add("ShellProp", 2, 1, q(PANEL.thick, 1), 1, "", 1, "", q(PANEL.thickNsm, -3));
  addLaminate(model, 3);
  const bandOf = (j) => Math.min(PANEL.ny - 1, Math.floor(j / refine));
  const bandProperty = [1, has("laminate") ? 3 : 1, has("thick") ? 2 : 1];
  const skinMass = ALUMINIUM.density * PANEL.skin;
  const bandMass = [skinMass, has("laminate") ? PLY.density * PANEL.ply * PANEL.plyAngles.length : skinMass,
    has("thick") ? ALUMINIUM.density * PANEL.thick + PANEL.thickNsm : skinMass];
  const shells = [];
  let shell = 0;
  const shiftOf = (count) => model.variant.cornerShift % count;
  for (let j = 0; j < ny; ++j)
    for (let i = 0; i < nx; ++i) {
      const band = bandOf(j);
      const corners = [N(i, j), N(i + 1, j), N(i + 1, j + 1), N(i, j + 1)];
      const tail = ["", band === 1 && has("laminate") ? 20 : "",
        band === 2 && has("thick") && has("offset") ? q(PANEL.thickOffset, 1) : ""];
      if (i < (PANEL.nx - 1) * refine || !has("triangles")) {
        model.add("ShellQuad4", element(++shell), bandProperty[band], ...turned(corners, shiftOf(4)).map(grid),
          ...tail);
        shells.push(shell);
      } else
        for (const corner of [[0, 1, 2], [0, 2, 3]]) {
          const three = turned(corner.map((c) => corners[c]), shiftOf(3));
          model.add("ShellTria3", element(++shell), bandProperty[band], ...three.map(grid), ...tail);
          shells.push(shell);
        }
    }
  // The mass of the bands: the area of each on the surface, by a midpoint rule fine enough for 1e-9 on the cylinder.
  const [stepsU, stepsV] = [240, 60];
  for (let band = 0; band < PANEL.ny; ++band) {
    const [v0, dv] = [-PANEL.arc / 2 + (PANEL.arc * band) / PANEL.ny, PANEL.arc / PANEL.ny / stepsV];
    let area = 0;
    for (let a = 0; a < stepsU; ++a)
      for (let b = 0; b < stepsV; ++b) {
        const [u, v, h] = [(PANEL.length * (a + 0.5)) / stepsU, v0 + (b + 0.5) * dv, 1e-5];
        area += norm(cross(scale(sub(surface(u + h, v), surface(u - h, v)), 0.5 / h),
          scale(sub(surface(u, v + h), surface(u, v - h)), 0.5 / h))) * (PANEL.length / stepsU) * dv;
      }
    model.mass += bandMass[band] * area;
  }

  // The stiffeners along the line of edges j = 1 (Bar2, offset inwards) and the ring across at i = 2 (Beam2).
  addBarProp(model, 4, 2, STIFFENER);
  model.add("BeamProp", 5, 2, q(RING.area, 2), q(RING.i1, 4), q(RING.i2, 4), "", q(RING.j, 4));
  const offset = [q(0, 1), q(has("offset") ? -PANEL.stiffenerOffset : 0, 1), q(0, 1)];
  for (let i = 0; i < nx && has("stiffeners"); ++i)
    model.add("Bar2", element(101001 + i), 4, grid(N(i, refine)), grid(N(i + 1, refine)),
      direction(normal(uOf(i + 0.5), vOf(refine)), { gap: true }), "OffsetsBasicOrientOrient", "", "", ...offset,
      ...offset);
  for (let j = 0; j < ny && has("ring"); ++j)
    model.add("Beam2", element(102001 + j), 5, grid(N(2 * refine, j)), grid(N(2 * refine, j + 1)),
      direction(normal(uOf(2 * refine), vOf(j + 0.5)), { gap: true }), BASIC_ORIENTATION);
  if (has("stiffeners"))
    model.mass += STEEL.density * STIFFENER.area * curveLength((u) => surface(u, vOf(refine)), 0, PANEL.length);
  if (has("ring"))
    model.mass += STEEL.density * RING.area *
      curveLength((v) => surface(uOf(2 * refine), v), -PANEL.arc / 2, PANEL.arc / 2);

  // The tripod: three Bar2 alone and a Rod2 from the apex T to the panel; T carries a mass with an offset.
  const [T, S, GA, GB] = [900001, 900002, 900021, 900022];
  const F = [900011, 900012, 900013, 900014];
  const apex = add(surface(0.7 * PANEL.length, 0.1 * PANEL.arc), scale(normal(0.7 * PANEL.length, 0.1 * PANEL.arc),
    PANEL.apexHeight));
  if (has("tripod")) model.node(T, apex, otherFrame);
  addBarProp(model, 6, 1, STRUT);
  model.add("RodProp", 7, 1, q(TIE_ROD.area, 2), q(TIE_ROD.j, 4));
  const lineMass = (a, b, density, area) => {
    model.mass += density * area * norm(sub(model.points.get(a), model.points.get(b)));
  };
  // Feet: on the stiffener line, on the ring, on a grid point of shells alone; the rod on the end of the ring.
  (has("tripod") ? [key(3, 1), key(2, 2), key(3, 2)] : []).forEach((foot, k) => {
    const axis = sub(model.points.get(foot), apex);
    model.add("Bar2", element(103001 + k), 6, grid(T), grid(foot),
      direction(unit(cross(axis, [0.3, 1, 0.2])), { gap: true }), BASIC_ORIENTATION);
    lineMass(T, foot, ALUMINIUM.density, STRUT.area);
  });
  if (has("tripod")) {
    model.add("Rod2", element(103004), 7, grid(T), grid(key(2, 3)));
    lineMass(T, key(2, 3), ALUMINIUM.density, TIE_ROD.area);
  }
  const APEX_MASS = { mass: 2.0, offset: [0.02, 0.0, 0.03], inertia: 0.01 };
  const massAt = (id, node, m) => {
    model.add("PointMassInertia", element(id), grid(node), "", q(m.mass, -1), direction(m.offset, { exponent: 1 }),
      q(m.inertia, 1), "", q(m.inertia, 1), "", "", q(m.inertia, 1));
    model.mass += m.mass;
  };
  if (has("tripod") && has("masses")) massAt(107001, T, APEX_MASS);
  // A mass with rotary inertia on a grid point of shells alone: it sees their mean rotation (an observer relation).
  if (has("masses"))
    massAt(107002, key(4, 2), { mass: 0.5, offset: [0, 0, 0], inertia: has("shellInertia") ? 0.002 : 0 });

  // The spider: the grid point S follows four grid points of the panel on average and carries a force and a couple.
  const centre = [(1.5 * PANEL.length) / PANEL.nx, 0];
  const legs = [key(1, 1), key(2, 1), key(1, 2), key(2, 2)];
  if (has("spider")) {
    model.node(S, add(surface(...centre), scale(normal(...centre), 0.1)), otherFrame);
    model.add("RigidInterpolation", element(108001), grid(S), 123456, 1, q(1), 123, legs.length, ...legs.map(grid));
  }

  // The grounded grid point GA under the free corner (4, 0), joined by springs in translation and rotation.
  const grounded = [];
  if (has("ground")) {
    model.node(GA, model.points.get(key(4, 0)), panelFrame(nx, 0));
    for (let c = 1; c <= 6; ++c)
      model.add("SpringDofInline", element(105000 + c), c <= 3 ? q(2.0e4, -1) : q(50.0, 1), grid(key(4, 0)), c,
        grid(GA), c);
    grounded.push(GA);
  }

  const framed = part === "all" && has("frame");
  if (framed) {
    // The frame under the panel: F1 and F3 below their grid points of the panel, F2 and F4 on theirs.
    const attach = [[1, 0], [3, 0], [3, 3], [1, 3]];
    const below = [true, false, true, false];
    attach.forEach(([i, j], k) => {
      const drop = below[k] ? scale(normalOf(i * refine, j * refine), -PANEL.frameDepth) : [0, 0, 0];
      model.node(F[k], add(model.points.get(key(i, j)), drop),
        below[k] ? otherFrame : panelFrame(i * refine, j * refine));
    });
    const attached = attach.map(([i, j]) => key(i, j));
    addBarProp(model, 8, 2, FRAME_BAR);
    [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([a, b], k) => {
      const axis = sub(model.points.get(F[b]), model.points.get(F[a]));
      model.add("Bar2", element(104001 + k), 8, grid(F[a]), grid(F[b]),
        direction(unit(cross(axis, [0.1, 0.2, 1])), { gap: true }), BASIC_ORIENTATION);
      lineMass(F[a], F[b], STEEL.density, FRAME_BAR.area);
    });
    model.add("Rod2", element(104005), 7, grid(F[0]), grid(F[2]));
    lineMass(F[0], F[2], ALUMINIUM.density, TIE_ROD.area);
    // A web in the frame (a warped quadrilateral): with it the frame is a component of its own on the condensed path.
    // It is left out of the assembly whose mass is computed by hand (the area of a warped quadrilateral is not).
    if (has("web")) model.add("ShellQuad4", element(104010), 1, ...turned(F, shiftOf(4)).map(grid));
    // F1: a bushing to the panel; F2: springs in translation; F3: the panel's grid point follows it rigidly in
    // translation; F4: a multipoint constraint on the three translations.
    model.add("BushProp", 9, "Stiffness", q(1.0e6, -1), q(1.0e6, -1), q(1.0e6, -1), q(2.0e3, 1), q(2.0e3, 1),
      q(2.0e3, 1));
    model.add("Bushing", element(106001), 9, grid(F[0]), grid(attached[0]), "", "", "", "", 0);
    for (let c = 1; c <= 3; ++c)
      model.add("SpringDofInline", element(105010 + c), q(5.0e5, -1), grid(F[1]), c, grid(attached[1]), c);
    if (joints === "mixed") {
      model.add("RigidBodyStar", element(108002), grid(F[2]), 123, grid(attached[2]));
      for (let c = 1; c <= 3; ++c) model.add("MultiPointConstr", 10, grid(F[3]), c, q(1), grid(attached[3]), c, q(-1));
    } else {
      model.add("Bushing", element(106003), 9, grid(F[2]), grid(attached[2]), "", "", "", "", 0);
      for (let c = 1; c <= 3; ++c)
        model.add("SpringDofInline", element(105020 + c), q(5.0e5, -1), grid(F[3]), c, grid(attached[3]), c);
    }
    // The grounded grid point GB at F1, joined by a bushing.
    model.node(GB, model.points.get(F[0]), otherFrame);
    model.add("Bushing", element(106002), 9, grid(F[0]), grid(GB), "", "", "", "", 0);
    grounded.push(GB);
  }

  // Supports. 1: the edge i = 0 clamped; 2: the same edge hinged and the far corner held across; 3: as 1 with an
  // enforced deflection of the far corner.
  const edge = Array.from({ length: ny + 1 }, (_, j) => N(0, j));
  for (const set of [1, 2, 3]) {
    model.add("SinglePointConstrList", set, set === 2 ? 123 : 123456, ...edge.map(grid));
    if (grounded.length) model.add("SinglePointConstrList", set, 123456, ...grounded.map(grid));
  }
  model.add("SinglePointConstrList", 2, 3, grid(key(4, 3)));
  model.add("SinglePointConstr", 3, grid(key(4, 3)), 3, q(0.002, 1));

  // Loads. 21: forces and couples at grid points; 22: a pressure on every shell of the panel; 23: gravity;
  // 24: temperatures.
  const force = (set, node, f) => model.force(set, node, f);
  const couple = (set, node, m) => model.couple(set, node, m);
  force(21, key(4, 0), [0, 0, -300]);
  force(21, key(4, 3), [100, 50, -200]);
  force(21, has("spider") ? S : key(2, 1), [0, 0, -500]);
  if (has("spider")) couple(21, S, [20, -30, 0]);
  force(21, has("tripod") ? T : key(3, 2), [50, 0, 0]);
  if (has("moments")) couple(21, key(4, 2), [3, 2, 0]);
  model.add("ShellPressureLoad", 22, q(2000, -2), ...shells.map(element));
  model.add("GravityLoad", 23, "", q(GRAVITY.magnitude, 1), direction(GRAVITY.direction));
  model.add("DefaultTemperatureLoad", 24, q(40));
  model.add("NodeTemperatureLoad", 24,
    ...Array.from({ length: PANEL.ny + 1 }, (_, j) => [grid(key(4, j)), q(80)]).flat());
  // 7: as 1 with an enforced shortening of the free edge along the axis, the preload of the buckling analysis: a
  // uniform shortening, which does not depend on how a load would be spread on the edge.
  model.add("SinglePointConstrList", 7, 123456, ...edge.map(grid), ...grounded.map(grid));
  for (let j = 0; j <= ny; ++j) model.add("SinglePointConstr", 7, grid(N(nx, j)), 1, q(-1.0e-5, 1));

  const global = framed && joints === "mixed" ? ["MultiPointConstrSelection;10"] : [];
  if (analysis === "statics")
    setAnalysis(model, "statics", { global, subcases: frames ? [[1, 1, 21, null], [3, 1, 23, null], [4, 1, null, 24]]
      : [[1, 1, 21, null], [2, 2, 22, null], [3, 1, 23, null], [4, 1, null, 24], [5, 3, 21, null]] });
  else if (analysis === "buckling") setAnalysis(model, "buckling", { global, constraints: 7, loads: 0, modes });
  else setAnalysis(model, "modes", { global, constraints: analysis === "modes" ? 1 : 0, modes });

  // The loads stated: the forces and couples of set 21; the pressure on the cylinder, whose resultant goes through
  // its axis at mid-length, and the weight (its moment is not computed by hand), on the exact geometry.
  model.state(1, 21);
  model.state(5, 21);
  if (!warped) {
    const lift = 2000 * PANEL.length * 2 * PANEL.radius * Math.sin(PANEL.arc / 2);
    model.stated.set(2, { force: [0, 0, lift], moment: cross([PANEL.length / 2, 0, -PANEL.radius], [0, 0, lift]) });
    model.stated.set(3, { force: scale(GRAVITY.direction, GRAVITY.magnitude * model.mass), moment: null });
  }
  model.internalForce.set(4, ALUMINIUM.young * ALUMINIUM.expansion * 40 * PANEL.skin * PANEL.radius * PANEL.arc);
  model.keys = { tipA: key(4, 0), tipB: key(4, 3), spider: has("spider") ? S : 0, apex: has("tripod") ? T : 0,
    frame: framed ? F[2] : 0, groundA: has("ground") ? GA : 0, groundB: framed ? GB : 0, shells, legs,
    exactGeometry: !warped };
  return model;
}

// ---------------------------------------------------------------------------------------------------------------------
// Assembly B: a bracket. A block of three cells along x: a SolidHexa, two SolidPenta, then a SolidPyramid on the
// interface and ten SolidTetra around its apex (some edges with a mid-side node: a partial set); shells on the top
// face of the first two cells (a quadrilateral, two triangles) continued into a flange that stands on the edge of
// the block; a Bar2 along another edge (on the shells' edge, then alone on the solids' edge); a second plate over
// the flange joined by two fasteners; a third plate glued under the block.

const BRACKET = { cell: [0.05, 0.04, 0.03], flange: 0.06, flangeThickness: 0.004, plateThickness: 0.004,
  gluedThickness: 0.002, bulge: 0.002, bar: { area: 2.0e-5, i: 4.0e-11, j: 6.0e-11 } };

// `midSide`: "none", "straight" (the mid-side nodes in the middle of their edges) or "curved" (two of them off);
// `refine` > 1: every cell cut into refine³ SolidHexa (the yardstick mesh of the classic formulation).
function bracketAssembly({ variant = {}, refine = 1, analysis = "statics", midSide = "curved", modes = 4,
  without = [], coverCells = 2 } = {}) {
  const model = new Assembly(variant);
  const has = (feature) => !without.includes(feature);
  model.name = `bracket${refine > 1 ? ` x${refine}` : ""}${midSide === "curved" ? "" : ` mid-side ${midSide}`}`;
  const [a, b, c] = BRACKET.cell;
  model.size = 3 * a;
  const r = refine;
  const [nx, ny, nz] = [3 * r, r, r];
  // Keep node and element families disjoint as the reference mesh is refined.
  const rows = 2 * r;
  const nodeStride = Math.max(1000, (nx + 1) * (ny + 1) * (nz + 1) + 1);
  const elementStride = Math.max(100, nx * rows);
  const B = (i, j, k) => 1 + i + (nx + 1) * (j + (ny + 1) * k);
  for (let k = 0; k <= nz; ++k)
    for (let j = 0; j <= ny; ++j)
      for (let i = 0; i <= nx; ++i) model.node(B(i, j, k), [(a * i) / r, (b * j) / r, (c * k) / r]);
  addMaterials(model);
  model.add("SolidProp", 21, 1);
  model.mass += ALUMINIUM.density * 3 * a * b * c;

  // The solids, numbered from FIRST_SOLID.
  const FIRST_SOLID = Math.max(5001, 7 * elementStride + 1);
  let solid = FIRST_SOLID - 1;
  const cellCorners = (i, j, k) => [B(i, j, k), B(i + 1, j, k), B(i + 1, j + 1, k), B(i, j + 1, k),
    B(i, j, k + 1), B(i + 1, j, k + 1), B(i + 1, j + 1, k + 1), B(i, j + 1, k + 1)];
  const mids = new Map();
  if (r > 1) {
    for (let k = 0; k < nz; ++k)
      for (let j = 0; j < ny; ++j)
        for (let i = 0; i < nx; ++i) addSolid(model, "SolidHexa", ++solid, 21, cellCorners(i, j, k));
  } else {
    const APEX = 90;
    const [c0, c1, c2] = [cellCorners(0, 0, 0), cellCorners(1, 0, 0), cellCorners(2, 0, 0)];
    model.node(APEX, [2.5 * a, b / 2, c / 2]);
    if (midSide !== "none") {
      const off = midSide === "curved" ? BRACKET.bulge : 0;
      // On a lower edge of the hexahedron (bulging sideways), on a diagonal of the lower face of the third cell and on
      // an edge of the end face (bulging out).
      const place = (id, g1, g2, shift) => {
        model.node(id, add(scale(add(model.points.get(g1), model.points.get(g2)), 0.5), shift));
        mids.set(edgeKey(g1, g2), id);
      };
      place(91, B(0, 0, 0), B(1, 0, 0), [0, -off, 0]);
      place(92, c2[0], c2[2], [0, 0, 0]);
      place(93, B(3, 0, 0), B(3, 1, 0), [off, 0, 0]);
    }
    addSolid(model, "SolidHexa", ++solid, 21, c0, mids);
    addSolid(model, "SolidPenta", ++solid, 21, [c1[0], c1[1], c1[2], c1[4], c1[5], c1[6]], mids);
    addSolid(model, "SolidPenta", ++solid, 21, [c1[0], c1[2], c1[3], c1[4], c1[6], c1[7]], mids);
    // The third cell: a pyramid on the interface, and each other face cut into two tetrahedra, around the apex.
    addSolid(model, "SolidPyramid", ++solid, 21, [c2[0], c2[3], c2[7], c2[4], APEX], mids);
    for (const face of [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3]]) {
      const [p, s, t, u] = face.map((n) => c2[n]);
      addSolid(model, "SolidTetra", ++solid, 21, [p, s, t, APEX], mids);
      addSolid(model, "SolidTetra", ++solid, 21, [p, t, u, APEX], mids);
    }
  }
  // The edges of the block are sharp.
  for (const j of [0, ny])
    for (const k of [0, nz])
      for (let i = 0; i < nx; ++i) model.sharpEdges.push([B(i, j, k), B(i + 1, j, k)]);
  for (const i of [0, nx])
    for (const t of [0, 1]) {
      for (let j = 0; j < ny; ++j) model.sharpEdges.push([B(i, j, t * nz), B(i, j + 1, t * nz)]);
      for (let k = 0; k < nz; ++k) model.sharpEdges.push([B(i, t * ny, k), B(i, t * ny, k + 1)]);
    }

  // The shells on the top face of the first two cells and the flange beyond the edge y = b, in the plane z = c.
  const FL = (i, m) => (m === 0 ? B(i, ny, nz) : nodeStride + i + (nx + 1) * (m - 1));
  model.add("ShellProp", 22, 1, q(BRACKET.flangeThickness, 1), 1, "", 1);
  const quad = (id, property, corners) =>
    model.add("ShellQuad4", element(id), property, ...turned(corners, model.variant.cornerShift % 4).map(grid));
  const tria = (id, property, corners) =>
    model.add("ShellTria3", element(id), property, ...turned(corners, model.variant.cornerShift % 3).map(grid));
  let cover = elementStride;
  const shellIds = [];
  if (has("cover")) {
    for (let j = 0; j < ny; ++j)
      for (let i = 0; i < coverCells * r; ++i) {
        const top = [B(i, j, nz), B(i + 1, j, nz), B(i + 1, j + 1, nz), B(i, j + 1, nz)];
        if (r === 1 && i >= 1) {
          tria(++cover, 22, [top[0], top[1], top[2]]);
          tria(++cover, 22, [top[0], top[2], top[3]]);
          shellIds.push(cover - 1, cover);
        } else { quad(++cover, 22, top); shellIds.push(cover); }
      }
    model.mass += ALUMINIUM.density * BRACKET.flangeThickness * coverCells * a * b;
  }
  const firstFlange = 2 * elementStride + 1;
  const flangeElement = (i, m) => firstFlange + i + nx * m;
  if (has("flange")) {
    for (let m = 1; m <= rows; ++m)
      for (let i = 0; i <= nx; ++i) model.node(FL(i, m), [(a * i) / r, b + (BRACKET.flange * m) / rows, c]);
    for (let m = 0; m < rows; ++m)
      for (let i = 0; i < nx; ++i) {
        quad(flangeElement(i, m), 22, [FL(i, m), FL(i + 1, m), FL(i + 1, m + 1), FL(i, m + 1)]);
        shellIds.push(flangeElement(i, m));
      }
    model.mass += ALUMINIUM.density * BRACKET.flangeThickness * 3 * a * BRACKET.flange;
  }
  // The cover stops before the end of the flange: the outline of the shells turns a re-entrant corner where the
  // edge of the cover meets the flange. The mesh alone does not say so beside triangles (the outline would be read
  // as one curve through the corner, and rounded): the lines of the two edges of the outline end there.
  const coverEnd = coverCells * r;
  if (has("cover") && has("flange") && coverEnd < nx)
    model.lineEnds.push([B(coverEnd, ny - 1, nz), B(coverEnd, ny, nz)], [B(coverEnd + 1, ny, nz), B(coverEnd, ny, nz)]);

  // The bar along the edge y = 0, z = c.
  if (has("bar")) {
    addBarProp(model, 25, 2, BRACKET.bar);
    for (let i = 0; i < nx; ++i)
      model.add("Bar2", element(5 * elementStride + 1 + i), 25, grid(B(i, 0, nz)), grid(B(i + 1, 0, nz)),
        direction([0, 0, 1], { gap: true }), BASIC_ORIENTATION);
    model.mass += STEEL.density * BRACKET.bar.area * 3 * a;
  }

  // The second plate over the flange (steel), joined by two fasteners.
  const gap = (BRACKET.flangeThickness + BRACKET.plateThickness) / 2;
  const plate = { x0: 0.02, x1: 0.13, y0: b + 0.005, y1: b + 0.055, nx: 2 * r, ny: r };
  const PL = (i, j) => 2 * nodeStride + i + (plate.nx + 1) * j;
  const plateElement = (i, j) => 3 * elementStride + 1 + i + plate.nx * j;
  if (has("plate") && has("flange")) {
    model.add("ShellProp", 23, 2, q(BRACKET.plateThickness, 1), 2, "", 2);
    for (let j = 0; j <= plate.ny; ++j)
      for (let i = 0; i <= plate.nx; ++i)
        model.node(PL(i, j), [plate.x0 + ((plate.x1 - plate.x0) * i) / plate.nx,
          plate.y0 + ((plate.y1 - plate.y0) * j) / plate.ny, c + gap]);
    for (let j = 0; j < plate.ny; ++j)
      for (let i = 0; i < plate.nx; ++i) {
        quad(plateElement(i, j), 23, [PL(i, j), PL(i + 1, j), PL(i + 1, j + 1), PL(i, j + 1)]);
        shellIds.push(plateElement(i, j));
      }
    model.mass += STEEL.density * BRACKET.plateThickness * (plate.x1 - plate.x0) * (plate.y1 - plate.y0);
    model.add("FastenerProp", 26, q(0.005, 1), 0, 1, q(1.0e7, -1), q(1.0e7, -1), q(1.0e7, -1), q(1.0e2, 1),
      q(1.0e2, 1), q(1.0e2, 1));
    [[0.04, b + 0.02], [0.11, b + 0.02]].forEach(([x, y], k) => {
      const below = flangeElement(Math.floor(x / (a / r)), Math.floor((y - b) / (BRACKET.flange / rows)));
      const above = plateElement(Math.floor((x - plate.x0) / ((plate.x1 - plate.x0) / plate.nx)),
        Math.floor((y - plate.y0) / ((plate.y1 - plate.y0) / plate.ny)));
      const point = 3 * nodeStride - 100 + k;
      model.node(point, [x, y, c + gap / 2]);
      model.add("Fastener", element(6 * elementStride + 1 + k), 26, "Elements", element(below), element(above), grid(point));
    });
  }

  // The third plate, glued under the first cell (the face z = 0 of the solids).
  const glued = { x0: 0.01, x1: 0.04, y0: 0.008, y1: 0.032, nx: 2 * r, ny: r };
  const GL = (i, j) => 3 * nodeStride + i + (glued.nx + 1) * j;
  if (has("glue")) {
    model.add("ShellProp", 24, 1, q(BRACKET.gluedThickness, 1), 1, "", 1);
    const gluedShells = [];
    for (let j = 0; j <= glued.ny; ++j)
      for (let i = 0; i <= glued.nx; ++i)
        model.node(GL(i, j), [glued.x0 + ((glued.x1 - glued.x0) * i) / glued.nx,
          glued.y0 + ((glued.y1 - glued.y0) * j) / glued.ny, 0]);
    for (let j = 0; j < glued.ny; ++j)
      for (let i = 0; i < glued.nx; ++i) {
        // Ordered so that the normal of the plate points out of the block.
        const id = 4 * elementStride + 1 + i + glued.nx * j;
        quad(id, 24, [GL(i, j), GL(i, j + 1), GL(i + 1, j + 1), GL(i + 1, j)]);
        gluedShells.push(id);
      }
    shellIds.push(...gluedShells);
    model.mass += ALUMINIUM.density * BRACKET.gluedThickness * (glued.x1 - glued.x0) * (glued.y1 - glued.y0);
    model.add("ContactPropertyRegion", 1, 21);
    model.add("ContactElemSurface", 2, ...gluedShells.map(element));
    model.add("ContactBody", 1, "ThreeDimensional", "Deformable", 1);
    model.add("ContactBody", 2, "ThreeDimensional", "Deformable", 2);
    model.add("ContactTable;1;;;;;;Secondary;2;;;;;1;;;;;;;;;;;;;;;;;;;;;;;;;Primary;1;1");
  }

  // Supports. 1: the face x = 0 held (the grid points of the shells there clamped); 2: the block held at three grid
  // points (3-2-1), free to expand; 3: as 1 with an enforced displacement of a corner of the end face.
  for (const set of [1, 3]) {
    for (let k = 0; k <= nz; ++k)
      for (let j = 0; j <= ny; ++j)
        model.add("SinglePointConstrList", set, k === nz && has("cover") ? 123456 : 123, grid(B(0, j, k)));
    if (has("flange"))
      model.add("SinglePointConstrList", set, 123456, ...Array.from({ length: rows }, (_, m) => grid(FL(0, m + 1))));
  }
  model.add("SinglePointConstrList", 2, 123, grid(B(0, 0, 0)));
  model.add("SinglePointConstrList", 2, 23, grid(B(nx, 0, 0)));
  model.add("SinglePointConstrList", 2, 3, grid(B(0, ny, 0)));
  model.add("SinglePointConstr", 3, grid(B(nx, 0, 0)), 2, q(1.0e-5, 1));

  // Loads. 31: forces and a couple; 32: pressures on a face of the hexahedron and on the flange; 33: gravity;
  // 34: a temperature rise.
  const force = (set, node, f) => model.force(set, node, f);
  const flangeTip = has("flange") ? FL(nx, rows) : B(nx, ny, nz);
  force(31, flangeTip, [0, 0, -200]);
  if (has("plate") && has("flange")) force(31, PL(plate.nx, plate.ny), [50, 0, 80]);
  if (has("glue")) force(31, GL(glued.nx, 0), [0, 30, -100]);
  if (has("flange") && has("moments"))
    model.couple(31, FL(2 * r, rows), [0.5, 0.8, 0]);
  // The face y = 0 of the first cell, by a corner and the corner across; the pressure pushes into the solid.
  for (let k = 0; k < nz; ++k)
    for (let i = 0; i < r; ++i)
      model.add("FacePressureLoad", 32, element(FIRST_SOLID + i + nx * ny * k), q(2.0e5, -2), "", "", "",
        grid(B(i, 0, k)), grid(B(i + 1, 0, k + 1)));
  if (has("flange"))
    model.add("ShellPressureLoad", 32, q(5.0e4, -2),
      ...Array.from({ length: nx * rows }, (_, k) => element(firstFlange + k)));
  model.add("GravityLoad", 33, "", q(GRAVITY.magnitude * 100, 1), direction(GRAVITY.direction));
  model.add("DefaultTemperatureLoad", 34, q(60));
  // 7: as 1 with an enforced shortening of the end of the flange along x, the preload of the buckling analysis.
  for (let k = 0; k <= nz; ++k)
    for (let j = 0; j <= ny; ++j)
      model.add("SinglePointConstrList", 7, k === nz && has("cover") ? 123456 : 123, grid(B(0, j, k)));
  for (let m = 1; m <= (has("flange") ? rows : 0); ++m) {
    model.add("SinglePointConstrList", 7, 123456, grid(FL(0, m)));
    model.add("SinglePointConstr", 7, grid(FL(nx, m)), 1, q(-1.0e-6, 1));
  }

  const global = has("glue") ? ["ContactSelection;1"] : [];
  if (analysis === "statics")
    setAnalysis(model, "statics", { global, subcases: [[1, 1, 31, null], [2, 2, 32, null], [3, 1, 33, null],
      [4, 2, null, 34], [5, 3, 31, null]] });
  else if (analysis === "buckling") setAnalysis(model, "buckling", { global, constraints: 7, loads: 0, modes });
  else setAnalysis(model, "modes", { global, constraints: analysis === "modes" ? 1 : 0, modes });

  // The loads stated: the forces and couples of set 31; the pressures on the flat face and on the flange (the face
  // is curved with the curved mid-side nodes: not stated then); the weight.
  model.state(1, 31);
  model.state(5, 31);
  const exactGeometry = midSide !== "curved" || r > 1;
  if (exactGeometry) {
    const onFace = [0, 2.0e5 * a * c, 0];
    const onFlange = has("flange") ? [0, 0, 5.0e4 * 3 * a * BRACKET.flange] : [0, 0, 0];
    model.stated.set(2, { force: add(onFace, onFlange), moment: add(cross([a / 2, 0, c / 2], onFace),
      cross([1.5 * a, b + BRACKET.flange / 2, c], onFlange)) });
    model.stated.set(3, { force: scale(GRAVITY.direction, GRAVITY.magnitude * 100 * model.mass), moment: null });
  }
  model.internalForce.set(4, ALUMINIUM.young * ALUMINIUM.expansion * 60 * b * c);
  model.keys = { flangeTip, blockTip: B(nx, 0, 0), blockTop: B(nx, ny, nz),
    plateCorner: has("plate") && has("flange") ? PL(plate.nx, plate.ny) : 0, glued: has("glue") ? GL(glued.nx, 0) : 0,
    shells: shellIds, exactGeometry };
  return model;
}

// ---------------------------------------------------------------------------------------------------------------------
// Assembly C: a wing box. A closed section (two skins, two spar webs) of three bays, tapered and twisted (every
// quadrilateral warped), the upper skin cambered; ribs closing the section at the end of each bay (three shells on
// each of their edges); the four corner lines declared sharp, and the outlines of the ribs sharp for the ribs (the
// skins and the webs go on across a rib, and its outline follows the camber of the skin); a Bar2 stringer under
// the upper skin and Rod2 caps on the upper corners; a flap behind the rear spar, a separate plate hung on two rigid
// elements and a bushing.

const WING = { span: 1.5, chord: 0.5, height: 0.12, camber: 0.02, taper: 0.3, twist: (5 * Math.PI) / 180, bays: 3,
  skin: 0.002, web: 0.003, rib: 0.0015, flapChord: 0.15, flapGap: 0.02, flapThickness: 0.003,
  stringer: { area: 6.0e-5, i1: 1.5e-9, i2: 5.0e-10, j: 3.0e-10 }, stringerOffset: 0.008,
  cap: { area: 1.0e-4, i: 8.0e-10, j: 2.0e-10 } };

// `caps`: "bar" (Bar2 on the upper corners) or "rod" (Rod2: no bending stiffness); `declared`: whether the folds are
// declared.
function wingAssembly({ variant = {}, refine = 1, analysis = "statics", modes = 4, without = [],
  declared = true, caps = "bar", joints = "mixed" } = {}) {
  const model = new Assembly(variant);
  const has = (feature) => !without.includes(feature);
  model.name = `wing box${refine > 1 ? ` x${refine}` : ""}`;
  model.size = WING.span;
  const r = refine;
  const [nc, nw, ns] = [2 * r, r, WING.bays * r];
  const W = (s, i, k) => 1 + i + (nc + 1) * k + (nc + 1) * (nw + 1) * s;
  const isRib = (s) => s > 0 && s % r === 0 && has("ribs");
  // The section at station s: tapered, twisted about the span axis; the upper skin a parabola over the chord.
  const pointOf = (s, i, k) => {
    const y = (WING.span * s) / ns;
    const shrink = 1 - (WING.taper * y) / WING.span;
    const xi = i / nc;
    const x = WING.chord * shrink * xi;
    const [low, high] = [(-WING.height * shrink) / 2, (WING.height * shrink) / 2 + 4 * WING.camber * xi * (1 - xi)];
    const z = low + ((high - low) * k) / nw;
    const angle = (WING.twist * y) / WING.span;
    return [x * Math.cos(angle) + z * Math.sin(angle), y, -x * Math.sin(angle) + z * Math.cos(angle)];
  };
  const onOutline = (i, k) => i === 0 || i === nc || k === 0 || k === nw;
  for (let s = 0; s <= ns; ++s)
    for (let k = 0; k <= nw; ++k)
      for (let i = 0; i <= nc; ++i)
        if (onOutline(i, k) || isRib(s)) model.node(W(s, i, k), pointOf(s, i, k));

  addMaterials(model);
  model.add("ShellProp", 41, 1, q(WING.skin, 1), 1, "", 1);
  model.add("ShellProp", 42, 1, q(WING.web, 1), 1, "", 1);
  model.add("ShellProp", 43, 1, q(WING.rib, 1), 1, "", 1);
  let shell = 0;
  const shells = [];
  const upper = [];
  const quad = (property, corners) => {
    model.add("ShellQuad4", element(++shell), property, ...turned(corners, model.variant.cornerShift % 4).map(grid));
    shells.push(shell);
    return shell;
  };
  for (let s = 0; s < ns; ++s) {
    for (let i = 0; i < nc; ++i) {
      quad(41, [W(s, i, 0), W(s + 1, i, 0), W(s + 1, i + 1, 0), W(s, i + 1, 0)]); // lower skin
      upper.push(quad(41, [W(s, i, nw), W(s, i + 1, nw), W(s + 1, i + 1, nw), W(s + 1, i, nw)])); // upper skin
    }
    for (let k = 0; k < nw; ++k) {
      quad(42, [W(s, 0, k), W(s, 0, k + 1), W(s + 1, 0, k + 1), W(s + 1, 0, k)]); // front web
      quad(42, [W(s, nc, k), W(s + 1, nc, k), W(s + 1, nc, k + 1), W(s, nc, k + 1)]); // rear web
    }
  }
  // The ribs. The outline of a rib is a junction (three shells on an edge, two at the tip): the rib ends there, a
  // sharp edge for that shell alone, while the skins and the webs go on across it. The edge keeps its mesh line,
  // which follows the camber of the upper skin (a raw edge would be straight: a roof at each rib).
  const ribOutline = [];
  for (let s = 1; s <= ns; ++s) {
    if (!isRib(s)) continue;
    for (let k = 0; k < nw; ++k)
      for (let i = 0; i < nc; ++i) {
        const rib = quad(43, [W(s, i, k), W(s, i + 1, k), W(s, i + 1, k + 1), W(s, i, k + 1)]);
        if (k === 0) ribOutline.push([W(s, i, 0), W(s, i + 1, 0), rib]);
        if (k === nw - 1) ribOutline.push([W(s, i, nw), W(s, i + 1, nw), rib]);
        if (i === 0) ribOutline.push([W(s, 0, k), W(s, 0, k + 1), rib]);
        if (i === nc - 1) ribOutline.push([W(s, nc, k), W(s, nc, k + 1), rib]);
      }
  }
  // The folds: the four corner lines, sharp (each keeps its own line along the span), and the outlines of the ribs.
  if (declared) {
    for (let s = 0; s < ns; ++s)
      for (const [i, k] of [[0, 0], [nc, 0], [nc, nw], [0, nw]]) model.sharpEdges.push([W(s, i, k), W(s + 1, i, k)]);
    model.sharpEdges.push(...ribOutline);
  }

  // The stringer under the middle of the upper skin and the caps on its two corners.
  const middle = nc / 2;
  if (has("stringer")) {
    addBarProp(model, 44, 1, WING.stringer);
    const offset = [q(0, 1), q(has("offset") ? -WING.stringerOffset : 0, 1), q(0, 1)];
    for (let s = 0; s < ns; ++s)
      model.add("Bar2", element(101001 + s), 44, grid(W(s, middle, nw)), grid(W(s + 1, middle, nw)),
        direction([0, 0, 1], { gap: true }), "OffsetsBasicOrientOrient", "", "", ...offset, ...offset);
  }
  if (has("caps")) {
    if (caps === "rod") model.add("RodProp", 45, 2, q(WING.cap.area, 2), q(WING.cap.j, 4));
    else addBarProp(model, 45, 2, WING.cap);
    for (let s = 0; s < ns; ++s)
      for (const [n, i] of [[0, 0], [1, nc]]) {
        const ends = [grid(W(s, i, nw)), grid(W(s + 1, i, nw))];
        if (caps === "rod") model.add("Rod2", element(102001 + 2 * s + n), 45, ...ends);
        else
          model.add("Bar2", element(102001 + 2 * s + n), 45, ...ends, direction([0, 0, 1], { gap: true }),
            BASIC_ORIENTATION);
      }
  }

  // The flap: a plate behind the rear spar along the last two bays, hung on the upper rear corner.
  const [fs, fc] = [2 * r, r];
  const FP = (s, m) => 500000 + m + (fc + 1) * s;
  const hinges = [r, 2 * r, 3 * r];
  if (has("flap")) {
    model.add("ShellProp", 46, 1, q(WING.flapThickness, 1), 1, "", 1);
    for (let s = 0; s <= fs; ++s) {
      const [rear, before] = [pointOf(r + s, nc, nw), pointOf(r + s, nc - 1, nw)];
      const back = unit([rear[0] - before[0], 0, rear[2] - before[2]]);
      for (let m = 0; m <= fc; ++m)
        model.node(FP(s, m), add(rear, scale(back, WING.flapGap + (WING.flapChord * m) / fc)));
    }
    for (let s = 0; s < fs; ++s)
      for (let m = 0; m < fc; ++m) quad(46, [FP(s, m), FP(s, m + 1), FP(s + 1, m + 1), FP(s + 1, m)]);
    // The first and last hinges: the flap's grid point follows the corner of the box rigidly in translation; the
    // middle one: a bushing.
    // `joints` "elastic": three bushings.
    if (joints === "mixed") {
      model.add("RigidBodyStar", element(108001), grid(W(hinges[0], nc, nw)), 123, grid(FP(0, 0)));
      model.add("RigidBodyStar", element(108002), grid(W(hinges[2], nc, nw)), 123, grid(FP(fs, 0)));
    } else {
      model.add("Bushing", element(106002), 47, grid(W(hinges[0], nc, nw)), grid(FP(0, 0)), "", "", "", "", 0);
      model.add("Bushing", element(106003), 47, grid(W(hinges[2], nc, nw)), grid(FP(fs, 0)), "", "", "", "", 0);
    }
    model.add("BushProp", 47, "Stiffness", q(2.0e6, -1), q(2.0e6, -1), q(2.0e6, -1), q(5.0e2, 1), q(5.0e2, 1),
      q(5.0e2, 1));
    model.add("Bushing", element(106001), 47, grid(W(hinges[1], nc, nw)), grid(FP(fs / 2, 0)), "", "", "", "", 0);
  }

  // Supports. 1: the root section clamped; 2: the root section held in translation and a tip corner propped;
  // 3: as 1 with an enforced deflection of a tip corner.
  const root = [];
  for (let k = 0; k <= nw; ++k) for (let i = 0; i <= nc; ++i) if (onOutline(i, k)) root.push(W(0, i, k));
  for (const set of [1, 2, 3]) model.add("SinglePointConstrList", set, set === 2 ? 123 : 123456, ...root.map(grid));
  model.add("SinglePointConstrList", 2, 3, grid(W(ns, 0, 0)));
  model.add("SinglePointConstr", 3, grid(W(ns, nc, 0)), 3, q(0.003, 1));

  // Loads. 41: forces at the tip (bending and torsion) and on the flap; 42: a pressure on the upper skin; 43: gravity;
  // 44: the upper skin warmer than the rest; 47: a downward force at the tip (the preload of the buckling analysis).
  const force = (set, node, f) => model.force(set, node, f);
  force(41, W(ns, 0, nw), [0, 0, 400]);
  force(41, W(ns, nc, nw), [0, 30, -150]);
  if (has("flap")) force(41, FP(fs, fc), [0, 0, 60]);
  model.add("ShellPressureLoad", 42, q(1500, -2), ...upper.map(element));
  model.add("GravityLoad", 43, "", q(GRAVITY.magnitude, 1), direction(GRAVITY.direction));
  model.add("DefaultTemperatureLoad", 44, q(10));
  const warm = [];
  for (let s = 0; s <= ns; ++s) for (let i = 0; i <= nc; ++i) warm.push(grid(W(s, i, nw)), q(60));
  model.add("NodeTemperatureLoad", 44, ...warm);
  force(47, W(ns, 0, nw), [0, 0, -500]);
  force(47, W(ns, nc, nw), [0, 0, -500]);

  if (analysis === "statics")
    setAnalysis(model, "statics", { subcases: [[1, 1, 41, null], [2, 2, 42, null], [3, 1, 43, null], [4, 1, null, 44],
      [5, 3, 41, null]] });
  else if (analysis === "buckling") setAnalysis(model, "buckling", { constraints: 1, loads: 47, modes });
  else setAnalysis(model, "modes", { constraints: analysis === "modes" ? 1 : 0, modes });

  model.state(1, 41);
  model.state(5, 41);
  model.internalForce.set(4, ALUMINIUM.young * ALUMINIUM.expansion * 50 * WING.skin * WING.chord);
  model.keys = { tipFront: W(ns, 0, nw), tipRear: W(ns, nc, nw), tipLow: W(ns, 0, 0),
    flapCorner: has("flap") ? FP(fs, fc) : 0, flapHinge: has("flap") ? FP(fs, 0) : 0, shells, upper,
    exactGeometry: false };
  return model;
}

// ---------------------------------------------------------------------------------------------------------------------
// The assemblies of the cases. `base`: the assembly of the equilibrium, invariance and yardstick cases; `paths`: the
// one of the path comparisons (two units or more for the condensed path); `exact`: one whose geometry the formulation
// reads exactly, so that its loads and its mass are stated by hand; `yardstick`: what the classic formulation is run
// with (no offsets: it refuses them in buckling; no mid-side nodes: it needs all of them) at the refinements given.

// `streamed`: what the streamed path is run without (it refuses a glued contact, and says so); `triplets`: the
// subcases whose supports hold whole triplets of components, the same in any axes.
const ASSEMBLIES = {
  panel: { make: panelAssembly, base: {}, paths: { joints: "elastic", without: ["tripod"] },
    exact: { shape: "cylinder", without: ["web", "offset"] }, streamed: [], triplets: [1, 3, 4],
    yardstick: { without: ["offset", "shellInertia"] }, refinements: [3, 6], bucklingRefinements: [6, 10],
    keys: ["tipA", "tipB", "spider", "apex", "frame"], rotationKeys: ["spider", "apex", "frame"] },
  bracket: { make: bracketAssembly, base: {}, paths: {}, exact: { midSide: "straight" }, streamed: ["glue"],
    triplets: [1, 3], yardstick: { midSide: "none", coverCells: 3 }, refinements: [2, 4], staticRefinements: [2, 4, 8], bucklingRefinements: [2, 4],
    keys: ["flangeTip", "blockTip", "blockTop", "plateCorner", "glued"], rotationKeys: [] },
  wing: { make: wingAssembly, base: {}, paths: { joints: "elastic" }, exact: null, streamed: [], triplets: [1, 3, 4],
    yardstick: { without: ["offset"] }, refinements: [3, 6], bucklingRefinements: [6, 10],
    keys: ["tipFront", "tipRear", "tipLow", "flapCorner"], rotationKeys: ["flapHinge"] },
};
// The scale of the forces of a subcase without loads, for the comparisons of its tables of forces.
const floorOf = (assembly) => (subcase) => assembly.internalForce.get(subcase) ?? 0;
// The subcases of the path and invariance comparisons: two constraint sets, forces and a pressure.
const SUBSET = [1, 2];

// A run made once and shared by the cases that read it; a run that throws does so for each of them.
const sharedRuns = new Map();
function shared(key, make) {
  if (!sharedRuns.has(key)) {
    try { sharedRuns.set(key, { value: make() }); } catch (error) { sharedRuns.set(key, { error }); }
  }
  const entry = sharedRuns.get(key);
  if (entry.error) throw entry.error;
  return entry.value;
}

// Runs `body(check)` and fails the case when a check it made is over its tolerance: every check is recorded first.
function checking(model, body) {
  const failures = [];
  const check = (name, value, tolerance, what = "") => {
    if (!record(name, model, value, tolerance, what))
      failures.push(`${name}: ${Number(value).toExponential(2)} over ${tolerance}${what ? ` (${what})` : ""}`);
  };
  body(check);
  assert.equal(failures.length, 0, `${model}: ${failures.join("; ")}`);
}

const baseRun = (name) => shared(`${name} base`, () => {
  const spec = ASSEMBLIES[name];
  return runAssembly(spec.make(spec.base));
});

// Writes the model of a failing case where OOS_ASSEMBLIES_DUMP says, and throws the failure again.
function dumping(name, assembly, formulation, extra, body) {
  try { return body(); } catch (error) {
    dump(name, assembly.text(), assembly.options(formulation, extra), String(error.message ?? error).slice(0, 2000));
    throw error;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Global equilibrium, the loads stated, the strain energy.

for (const [name, spec] of Object.entries(ASSEMBLIES)) {
  test(`1. ${name}: the applied loads and the support forces balance in every subcase`, () => {
    const assembly = spec.make(spec.base);
    dumping(`equilibrium ${name}`, assembly, MIXED_NAME, {}, () => {
      const run = baseRun(name);
      assertFinite(run, name);
      checking(assembly.name, (check) => {
        for (const subcase of subcasesOf(run)) {
          const { force, moment } = equilibriumResiduals(assembly, run, subcase);
          check(`1 equilibrium of the tables in force, subcase ${subcase}`, force, SAME);
          check(`1 equilibrium of the tables in moment, subcase ${subcase}`, moment, SAME);
          // The forces of the multipoint constraints and rigid elements are internal: they sum to zero.
          const internal = resultant(assembly, nodalTable(run, "multiPointConstrForces", subcase));
          const scaleOf = Math.max(internal.largestForce, Number.MIN_VALUE);
          check(`1 multipoint constraint forces sum to zero, subcase ${subcase}`,
            Math.max(norm(internal.total.slice(0, 3)) / scaleOf,
              norm(internal.total.slice(3)) / Math.max(internal.largestMoment, scaleOf * assembly.size)), SAME);
        }
      });
    });
  });

  // The loads stated by hand: on the base assembly the forces and couples at grid points (subcases 1 and 5); on the
  // exact geometry the pressure and the weight too (the weight is the mass computed by hand: check 6).
  for (const [label, options, settings] of [["", spec.base, undefined], [" (exact geometry)", spec.exact, [1, 2, 3]]]) {
    if (!options) continue;
    test(`1. ${name}${label}: the support forces balance the loads stated, which the applied load table gives`, () => {
      const assembly = spec.make(options);
      dumping(`stated loads ${name}${label}`, assembly, MIXED_NAME, {}, () => {
        const run = settings ? shared(`${name} exact`, () => runAssembly(assembly, MIXED_NAME, "statics",
          { subcases: settings })) : baseRun(name);
        checking(assembly.name, (check) => {
          for (const subcase of subcasesOf(run)) {
            const stated = assembly.stated.get(subcase);
            if (!stated) continue;
            const loads = statedResultant(stated);
            const reactions = resultant(assembly, nodalTable(run, "singlePointConstrForces", subcase)).total;
            const applied = resultant(assembly, nodalTable(run, "appliedLoads", subcase)).total;
            const balance = residualsOf(assembly, loads, reactions, subcase);
            const table = residualsOf(assembly, loads, applied.map((value) => -value), subcase);
            // The weight stated is the mass computed by hand times the acceleration: check 6.
            check(stated.moment ? `1 support forces against the stated force, subcase ${subcase}`
              : `6 weight carried by the supports against the mass computed by hand, subcase ${subcase}`,
            balance.force, SAME);
            check(`1 applied load table against the stated force, subcase ${subcase}`, table.force, SAME);
            if (!stated.moment) continue;
            check(`1 support forces against the stated moment, subcase ${subcase}`, balance.moment, SAME);
            check(`1 applied load table against the stated moment, subcase ${subcase}`, table.moment, SAME);
          }
        });
      });
    });
  }

  test(`1. ${name}: the strain energy tables sum to half the work of the forces and couples`, () => {
    const assembly = spec.make(spec.base);
    const run = baseRun(name);
    checking(assembly.name, (check) => {
      const [energy, work] = [strainEnergy(run, 1), halfWork(run, 1)];
      check("1 strain energy against half the work, subcase 1", Math.abs(energy / work - 1), SAME,
        `energy ${energy}, half the work ${work}`);
      let percent = 0;
      for (const table of run.elementTables.filter((t) => t.kind === "strainEnergy" && t.subcase === 1))
        table.ids.forEach((id, row) => { if (id !== SUBTOTAL_ID) percent += table.values[row * 3 + 1]; });
      check("1 strain energy percentages sum to 100, subcase 1", Math.abs(percent / 100 - 1), 1e-6);
    });
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Every path gives the results of the full path.

const pathsAssembly = (name, more = {}) => ASSEMBLIES[name].make({ ...ASSEMBLIES[name].paths, ...more });
const fullSubset = (name) => shared(`${name} paths full`, () =>
  runAssembly(pathsAssembly(name), MIXED_NAME, "statics", { subcases: SUBSET }, { condensation: true }));

// The envelopes asked of the path comparisons: components of every kind of table, and the magnitudes.
const ENVELOPES = [
  ...["t1", "t2", "t3", "r1", "r2", "r3"].map((quantity) => ({ kind: "displacements", quantity, mode: "absMax" })),
  { kind: "displacements", quantity: "magnitude", mode: "max" },
  ...["t1", "t3", "r2"].map((quantity) => ({ kind: "singlePointConstrForces", quantity, mode: "absMax" })),
  { kind: "singlePointConstrForces", quantity: "magnitude", mode: "max" },
  { kind: "appliedLoads", quantity: "t3", mode: "min" },
  { kind: "multiPointConstrForces", quantity: "t1", mode: "max" },
  { kind: "stress", quantity: "vonMises", mode: "max" }, { kind: "stress", quantity: "oxx", mode: "min" },
  { kind: "stress", quantity: "o11", mode: "absMax" }, { kind: "stress", quantity: "axial", mode: "absMax" },
  { kind: "strain", quantity: "exx", mode: "absMax" }, { kind: "strain", quantity: "vonMises", mode: "max" },
  { kind: "forces", quantity: "nx", mode: "absMax" }, { kind: "forces", quantity: "mx", mode: "min" },
  { kind: "forces", quantity: "qx", mode: "max" }, { kind: "forces", quantity: "bendingMomentA1", mode: "absMax" },
  { kind: "forces", quantity: "axial", mode: "max" }, { kind: "strainEnergy", quantity: "strainEnergy", mode: "max" },
];

// The envelope of `spec` computed from the tables of `run`, as the rows of `envelope`: the largest relative
// difference of the values, and whether the cases agree (where two cases are not within round-off of each other).
function envelopeDifference(run, envelope) {
  const nodalKind = envelope.elementType === "";
  const tables = (nodalKind ? run.nodalTables : run.elementTables).filter((t) => t.kind === envelope.kind &&
    (nodalKind || t.elementType === envelope.elementType));
  const orders = { max: (a, b) => a > b, min: (a, b) => a < b, absMax: (a, b) => Math.abs(a) > Math.abs(b) };
  const better = orders[envelope.mode];
  const expected = new Map(); // row key → { value, subcase, runnerUp }
  let largest = 0;
  for (const table of tables) {
    const width = table.components.length;
    const column = table.components.indexOf(envelope.quantity);
    for (let row = 0; row < table.ids.length; ++row) {
      const values = table.values.slice(row * width, (row + 1) * width);
      const value = envelope.quantity === "magnitude" ? Math.hypot(values[0], values[1], values[2]) : values[column];
      const key = nodalKind ? String(table.ids[row]) : elementRowKey(table, row);
      const now = expected.get(key);
      if (!now || better(value, now.value)) expected.set(key, { value, subcase: table.subcase, other: now?.value });
      else if (now.other === undefined || better(value, now.other)) now.other = value;
      largest = Math.max(largest, Math.abs(value));
    }
  }
  let [difference, wrongCases] = [expected.size === envelope.ids.length ? 0 : Infinity, 0];
  for (let row = 0; row < envelope.ids.length; ++row) {
    const key = nodalKind ? String(envelope.ids[row]) : elementRowKey(envelope, row);
    const want = expected.get(key);
    if (!want) { difference = Infinity; continue; }
    difference = Math.max(difference,
      Math.abs(envelope.values[row] - want.value) / Math.max(largest, Number.MIN_VALUE));
    const tie = want.other !== undefined && Math.abs(want.other - want.value) <= 1e-9 * largest;
    if (envelope.cases[row] !== want.subcase && !tie) ++wrongCases;
  }
  return { difference, wrongCases };
}

for (const name of Object.keys(ASSEMBLIES)) {
  const title = `2. ${name}`;
  const compareWithFull = (label, settings, map = {}) => {
    const assembly = pathsAssembly(name);
    dumping(`path ${label} ${name}`, assembly, MIXED_NAME, { condensation: true }, () => {
      const full = fullSubset(name);
      const run = runAssembly(assembly, MIXED_NAME, "statics", { subcases: SUBSET, ...settings },
        { condensation: true });
      if (settings.condensation)
        assert.ok(run.condensation.condensedUnits.length >= 1, `units condensed: ${JSON.stringify(run.condensation)}`);
      const worst = compareRuns(full, run, { size: assembly.size, forceFloor: floorOf(assembly), ...map });
      expectWithin(`2 ${label} against the full path`, assembly.name, worst.difference, SAME, worst.what);
      return run;
    });
  };

  for (const memoryMode of ["auto", "speed", "low_memory"])
    test(`${title}: the condensed path (memory mode ${memoryMode}) gives every table of the full path`, () => {
      compareWithFull(`condensed path, ${memoryMode}`, { condensation: { units: "all", memoryMode } });
    });

  test(`${title}: the condensed path with a cache gives every table of the full path, first and second run`, () => {
    const cache = solver.condensationCache();
    try {
      compareWithFull("condensed path, cache, first run", { condensation: { units: "all", cache } });
      compareWithFull("condensed path, cache, second run", { condensation: { units: "all", cache } });
      const statistics = cache.statistics();
      assert.ok(statistics.memoryHits > 0, `the second run takes units from the cache: ${JSON.stringify(statistics)}`);
    } finally { cache.delete(); }
  });

  test(`${title}: the streamed path gives the tables of the full path that it makes`, () => {
    const left = ASSEMBLIES[name].streamed;
    const assembly = pathsAssembly(name, left.length ? { without: left } : {});
    dumping(`path streamed ${name}`, assembly, MIXED_NAME, {}, () => {
      const full = left.length ? runAssembly(assembly, MIXED_NAME, "statics", { subcases: SUBSET }) : fullSubset(name);
      const run = runAssembly(assembly, MIXED_NAME, "statics", { subcases: SUBSET, streamed: true });
      for (const kind of ["stress", "strain", "forces"])
        assert.ok(run.elementTables.some((t) => t.kind === kind), `the streamed path gives the ${kind}`);
      // The streamed path makes no product with the stiffness (support forces, applied loads, strain energy): it
      // says so. Its tables are compared with those of the full path.
      const worst = compareRuns(run, full, { size: assembly.size, forceFloor: floorOf(assembly) });
      expectWithin("2 streamed path against the full path", assembly.name, worst.difference, SAME, worst.what);
    });
  });

  test(`${title}: the envelopes of the full path are those of its tables, and those recovered unit by unit too`, () => {
    const assembly = pathsAssembly(name);
    dumping(`path envelopes ${name}`, assembly, MIXED_NAME, { condensation: true }, () => {
      const full = fullSubset(name);
      const present = ENVELOPES.filter(({ kind, quantity }) => quantity === "magnitude" ||
        [...full.nodalTables, ...full.elementTables].some((t) => t.kind === kind && t.components.includes(quantity)));
      const enveloped = withAssembly(assembly, MIXED_NAME, (model) =>
        solver.run(model, "statics", { subcases: SUBSET, envelopes: present }), { condensation: true });
      // The recovery unit by unit makes no table of a case: the deck asks only the kinds of tables it can envelope
      // unit by unit (no product with the stiffness of the whole model).
      const wholeModel = /^(appliedLoads|multiPointConstrForces|strainEnergy)$/;
      const reduced = pathsAssembly(name);
      reduced.caseLines = reduced.caseLines.filter((line) =>
        !/^(AppliedLoadOutput|MultiPointConstrForceOutput|ElemStrainEnergyOutput)/.test(line));
      const byUnit = withAssembly(reduced, MIXED_NAME, (model) => solver.run(model, "statics", { subcases: SUBSET,
        envelopes: present.filter(({ kind }) => !wholeModel.test(kind)), keepsCaseTables: false,
        condensation: { units: "all", memoryMode: "low_memory" } }), { condensation: true });
      assert.equal(errorsOf(enveloped).length + errorsOf(byUnit).length, 0, JSON.stringify(errorsOf(byUnit)));
      assert.ok(enveloped.envelopes.length >= present.length, `${enveloped.envelopes.length} envelope tables`);
      checking(assembly.name, (check) => {
        let [worst, wrong, what] = [0, 0, ""];
        for (const envelope of enveloped.envelopes) {
          const { difference, wrongCases } = envelopeDifference(full, envelope);
          const here = `${envelope.kind} ${envelope.elementType} ${envelope.quantity}`;
          if (difference > worst) [worst, what] = [difference, here];
          wrong += wrongCases;
        }
        check("2 envelopes of the full path against its tables", worst, SAME, what);
        check("2 envelopes of the full path: rows with another case", wrong, 0);
        check("2 unit by unit recovery was used", byUnit.condensation?.unitByUnit ? 0 : 1, 0);
        const expected = enveloped.envelopes.filter((envelope) => !wholeModel.test(envelope.kind));
        [worst, what] = [byUnit.envelopes.length === expected.length ? 0 : Infinity, "the count of tables"];
        for (const envelope of expected) {
          const other = byUnit.envelopes.find((e) => e.kind === envelope.kind &&
            e.elementType === envelope.elementType && e.quantity === envelope.quantity && e.mode === envelope.mode);
          const here = `${envelope.kind} ${envelope.elementType} ${envelope.quantity}`;
          if (!other || other.ids.length !== envelope.ids.length) {
            [worst, what] = [Infinity, `${here} is missing`];
            continue;
          }
          const rowOf = new Map();
          for (let row = 0; row < other.ids.length; ++row) rowOf.set(elementRowKey(other, row), row);
          const largest = Math.max(...Array.from(envelope.values, Math.abs), Number.MIN_VALUE);
          for (let row = 0; row < envelope.ids.length; ++row) {
            const match = rowOf.get(elementRowKey(envelope, row));
            const difference = match === undefined ? Infinity
              : Math.abs(other.values[match] - envelope.values[row]) / largest;
            if (difference > worst) [worst, what] = [difference, here];
          }
        }
        check("2 envelopes recovered unit by unit against the full path", worst, SAME, what);
      });
    });
  });

  test(`${title}: the condensed path takes the whole assembly (rigid elements, bars alone, fasteners, glue)`, () => {
    const spec = ASSEMBLIES[name];
    const assembly = spec.make(spec.base);
    dumping(`condensed whole ${name}`, assembly, MIXED_NAME, { condensation: true }, () => {
      const run = runAssembly(assembly, MIXED_NAME, "statics", { subcases: SUBSET, condensation: "all" },
        { condensation: true });
      assert.ok(run.condensation.condensedUnits.length >= 1, `units condensed: ${JSON.stringify(run.condensation)}`);
      const worst = compareRuns(run, baseRun(name), { size: assembly.size, forceFloor: floorOf(assembly) });
      expectWithin("2 condensed path of the whole assembly against the full path", assembly.name, worst.difference,
        SAME, worst.what);
    });
  });

  // Modes and buckling: the small units condensed by Craig-Bampton with every fixed-interface mode kept.
  for (const analysis of ["modes", "buckling"])
    test(`${title}: the ${analysis} of the condensed path are those of the full path`, () => {
      const assembly = pathsAssembly(name, { analysis });
      dumping(`condensed ${analysis} ${name}`, assembly, MIXED_NAME, { condensation: true }, () => {
        const [full, condensed] = withAssembly(assembly, MIXED_NAME, (model) => [solver.run(model, analysis),
          solver.run(model, analysis, { condensation: { units: [2], modeCount: 100000 } })], { condensation: true });
        assert.equal(errorsOf(full).length + errorsOf(condensed).length, 0, JSON.stringify(errorsOf(condensed)));
        assert.deepEqual(condensed.condensation.condensedUnits, [2], JSON.stringify(condensed.condensation));
        const roots = (run) => run.eigenvalues[run.eigenvalues.length - 1].rows.map((row) => row.eigenvalue);
        const [a, b] = [roots(full), roots(condensed)];
        assert.equal(b.length, a.length, "the same count of roots");
        const largest = Math.max(...a.map(Math.abs));
        expectWithin(`2 ${analysis} of the condensed path against the full path`, assembly.name,
          Math.max(...a.map((value, k) => Math.abs(value - b[k]))) / largest, SAME, `${a} and ${b}`);
      });
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Invariance: the same assembly written another way gives the same results.

const MOTION = { rotation: rotationAbout([1, 2, 3], 0.7), translation: [0.3, -0.2, 0.5] };
const MILLIMETRES = 1000;
// The elements whose tables are written along the axes of the basic system, or by component of a displacement
// system: a rigid motion of the model changes their components.
const BASIC_AXES = (table) => table.elementType === "SpringDofInline" ||
  (/^(Bushing|Fastener)$/.test(table.elementType) && table.kind !== "strainEnergy");
const VARIANTS = {
  "grid points renumbered": { variant: { gridId: (id) => 2000000 - id }, subcases: SUBSET },
  "elements renumbered": { variant: { elementId: (id) => 1000000 - id }, subcases: SUBSET },
  "cards in another order": { variant: { shuffleSeed: 20261117 }, subcases: SUBSET },
  // The shells of an isotropic property write their stresses along their first side: their invariants are compared.
  "elements started from another corner": { variant: { cornerShift: 1 }, subcases: SUBSET,
    map: { invariantsOnly: (table) => /^Shell(Quad4|Tria3)$/.test(table.elementType) &&
      table.kind !== "strainEnergy" } },
  // The stresses of the solids are in the basic system: their invariants are compared.
  // The subcases compared are those whose supports hold whole triplets of components (`triplets`).
  "moved rigidly": { variant: MOTION, subcases: null,
    map: { skip: BASIC_AXES,
      invariantsOnly: (table) => /^Solid/.test(table.elementType) && table.kind !== "strainEnergy",
      nodal: (id, values) => [...apply(MOTION.rotation, values.slice(0, 3)),
        ...apply(MOTION.rotation, values.slice(3))] } },
  // Lengths in millimetres, forces unchanged: moduli in MPa, densities in tonnes per cubic millimetre. The forces
  // of the springs mix forces and couples in one column: left out.
  "in millimetres": { variant: { lengthScale: MILLIMETRES }, subcases: [1, 2, 3, 4, 5],
    map: { lengthScale: MILLIMETRES,
      skip: (table) => table.elementType === "SpringDofInline" && table.kind === "forces",
      nodal: (id, values, kind) => (kind === "displacements"
        ? [...scale(values.slice(0, 3), MILLIMETRES), ...values.slice(3)]
        : [...values.slice(0, 3), ...scale(values.slice(3), MILLIMETRES)]) } },
};

for (const [name, spec] of Object.entries(ASSEMBLIES))
  for (const [label, { variant, subcases, map = {} }] of Object.entries(VARIANTS))
    test(`3. ${name} with its ${label} gives the same results`, () => {
      const assembly = spec.make({ ...spec.base, variant });
      const compared = subcases ?? spec.triplets;
      dumping(`invariance ${label} ${name}`, assembly, MIXED_NAME, {}, () => {
        const base = baseRun(name);
        const run = runAssembly(assembly, MIXED_NAME, "statics", { subcases: compared });
        const skip = (table) => !compared.includes(table.subcase) || (map.skip?.(table) ?? false);
        const worst = compareRuns(base, run, { size: assembly.size, gridId: assembly.variant.gridId,
          elementId: assembly.variant.elementId, forceFloor: floorOf(assembly), ...map, skip });
        // The applied load table holds the part of the loads at the grid points only (check 1) and the strain energy
        // density is not the energy over the volume (check 7): they are recorded apart.
        const apart = ["appliedLoads", "strainEnergyDensity"];
        checking(assembly.name, (check) => {
          let others = { difference: 0, what: "" };
          for (const [kind, entry] of worst.byKind)
            if (!apart.includes(kind) && entry.difference > others.difference) others = entry;
          check(`3 invariance, ${label}: every table but the two below`, others.difference, SAME, others.what);
          for (const kind of apart) {
            const entry = worst.byKind.get(kind) ?? { difference: 0, what: "" };
            check(`3 invariance, ${label}: ${kind}`, entry.difference, SAME, entry.what);
          }
        });
      });
    });

test("3. panel with a displacement system on every grid point gives the same results in those systems", () => {
  // Rectangular systems in turn along the clamped edge, a cylindrical one on the rest of the panel: the supports are
  // whole triplets of components, the springs between coincident grid points have one stiffness in every direction.
  const subcases = [1, 3, 4];
  const assembly = panelAssembly({ frames: true });
  dumping("invariance displacement systems panel", assembly, MIXED_NAME, {}, () => {
    const run = runAssembly(assembly);
    const warnings = run.diagnostics.filter((d) => d.severity === "warning").map((d) => d.code);
    const nodal = (id, values) => {
      const axes = assembly.localAxes(id);
      return [...apply(axes, values.slice(0, 3)), ...apply(axes, values.slice(3))];
    };
    const worst = compareRuns(baseRun("panel"), run, { size: assembly.size, nodal, forceFloor: floorOf(assembly),
      skip: (table) => !subcases.includes(table.subcase) || table.elementType === "SpringDofInline" });
    expectWithin("3 invariance, displacement systems on every grid point", assembly.name, worst.difference, SAME,
      `${worst.what}; warnings: ${[...new Set(warnings)].join(", ")}`);
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// 4. Determinism: the same model built and run twice gives the same bits.

function bitDifference(first, second) {
  for (const kind of ["nodalTables", "elementTables"]) {
    if (first[kind].length !== second[kind].length) return `${kind}: ${first[kind].length} and ${second[kind].length}`;
    for (let k = 0; k < first[kind].length; ++k) {
      const [a, b] = [first[kind][k], second[kind][k]];
      if (!sameTable(a, b) || a.values.length !== b.values.length) return `${tableName(a)}: another table`;
      for (let v = 0; v < a.values.length; ++v)
        if (!Object.is(a.values[v], b.values[v]) && !(Number.isNaN(a.values[v]) && Number.isNaN(b.values[v])))
          return `${tableName(a)}, value ${v}: ${a.values[v]} and ${b.values[v]}`;
    }
  }
  return "";
}

for (const name of Object.keys(ASSEMBLIES))
  test(`4. ${name} built and run twice gives bit-identical tables`, () => {
    const assembly = pathsAssembly(name);
    const again = runAssembly(assembly, MIXED_NAME, "statics", { subcases: SUBSET }, { condensation: true });
    const difference = bitDifference(fullSubset(name), again);
    expectWithin("4 determinism (0: bit-identical)", assembly.name, difference === "" ? 0 : 1, 0, difference);
  });

// ---------------------------------------------------------------------------------------------------------------------
// 5. The classic formulation on finer meshes as a yardstick, and 6. the rigid-body modes and the mass.

// The translation and the rotation of the named grid points of an assembly in a subcase.
function keyMotions(assembly, run, subcase, keys) {
  const rows = rowsById(nodalTable(run, "displacements", subcase));
  return Object.fromEntries(keys.filter((key) => assembly.keys[key]).map((key) => {
    const values = rows.get(assembly.keys[key]);
    return [key, { translation: norm(values.slice(0, 3)), rotation: norm(values.slice(3)) }];
  }));
}

const lastRoots = (run) => run.eigenvalues[run.eigenvalues.length - 1].rows;
// The eigenvalues of an analysis of an assembly; null when the analysis fails (the yardstick has its own limits).
function rootsOf(assembly, formulation, analysis) {
  try {
    return lastRoots(runAssembly(assembly, formulation, analysis)).map((row) => row.eigenvalue);
  } catch (error) {
    yardstickNotes.push(`${assembly.name} ${formulation} ${analysis}: ${String(error.message ?? error).slice(0, 200)}`);
    return null;
  }
}
const yardstickNotes = [];
const yardstickRows = [];

// One quantity of the yardstick: the mixed value against the classic one on its finest mesh. It is asserted when the
// changes between successive classic meshes are all within `settled`. Point supports can give a false plateau on
// the first two bracket meshes, so its statics use three levels.
function yardstick(check, model, what, mixed, classic, tolerance = YARDSTICK, settled = 0.02) {
  const [coarse, fine] = classic.slice(-2);
  const ratio = mixed / fine;
  const drift = coarse / fine;
  const maxChange = Math.max(...classic.slice(1).map((value, i) => Math.abs(classic[i] / value - 1)));
  const conclusive = maxChange <= settled;
  yardstickRows.push({ model, what, mixed, coarse, fine, classic, ratio, drift, maxChange, conclusive,
    pass: Math.abs(ratio - 1) <= tolerance });
  if (conclusive)
    check(`5 ${what}: mixed / classic - 1`, Math.abs(ratio - 1), tolerance, `mixed ${mixed}, classic ${fine}`);
  // Nothing may be off by a large factor, where the classic value is settled within 20 % at least.
  else if (Math.abs(drift - 1) <= 0.2)
    check(`5 ${what}: within a factor 1.5 of the classic value`, Math.max(ratio, 1 / ratio) <= 1.5 ? 0 : 1, 0,
      `mixed ${mixed}, classic ${coarse} then ${fine}`);
}

for (const [name, spec] of Object.entries(ASSEMBLIES))
  test(`5. ${name}: displacements, frequencies and buckling factor against the classic formulation, finer`, () => {
    const subcases = [1, 2, 3];
    const make = (analysis, refine, modes) => spec.make({ ...spec.yardstick, analysis, refine, modes });
    const mixed = make("statics", 1);
    const mixedRun = runAssembly(mixed, MIXED_NAME, "statics", { subcases });
    const classic = (spec.staticRefinements ?? spec.refinements).map((refine) => make("statics", refine));
    const classicRuns = classic.map((assembly) => runAssembly(assembly, "ClassicFormulation", "statics", { subcases }));
    checking(mixed.name, (check) => {
      for (const subcase of subcases) {
        const named = [...spec.keys, ...spec.rotationKeys];
        const mine = keyMotions(mixed, mixedRun, subcase, named);
        const reference = classic.map((assembly, k) => keyMotions(assembly, classicRuns[k], subcase, named));
        for (const key of spec.keys)
          yardstick(check, mixed.name, `displacement of ${key}, subcase ${subcase}`, mine[key].translation,
            reference.map((level) => level[key].translation));
        for (const key of spec.rotationKeys)
          yardstick(check, mixed.name, `rotation of ${key}, subcase ${subcase}`, mine[key].rotation,
            reference.map((level) => level[key].rotation));
      }
      for (const [analysis, count] of [["modes", 4], ["buckling", 6]]) {
        const refinements = analysis === "modes" ? spec.refinements : spec.bucklingRefinements;
        const [mine, coarse, fine] = [rootsOf(make(analysis, 1, count), MIXED_NAME, analysis),
          ...refinements.map((refine) => rootsOf(make(analysis, refine, count), "ClassicFormulation", analysis))];
        assert.ok(mine, `the ${analysis} of the mixed formulation`);
        if (!coarse || !fine) continue;
        if (analysis === "modes") {
          const hertz = (value) => Math.sqrt(Math.abs(value)) / (2 * Math.PI);
          for (let k = 0; k < 3; ++k)
            yardstick(check, mixed.name, `frequency ${k + 1}`, hertz(mine[k]), [hertz(coarse[k]), hertz(fine[k])]);
        } else {
          // The factor of the preload as it is applied: the smallest positive root.
          const first = (roots) => Math.min(...roots.filter((value) => value > 0));
          yardstick(check, mixed.name, "first buckling factor", first(mine), [first(coarse), first(fine)]);
        }
      }
    });
  });

for (const [name, spec] of Object.entries(ASSEMBLIES))
  test(`6. ${name} free has six rigid-body modes, no spurious one and no negative eigenvalue`, () => {
    const assembly = spec.make({ ...spec.base, analysis: "freeModes", modes: 10 });
    dumping(`free modes ${name}`, assembly, MIXED_NAME, {}, () => {
      const run = runAssembly(assembly, MIXED_NAME, "modes");
      const values = lastRoots(run).map((row) => row.eigenvalue);
      assert.ok(values.length >= 8, `${values.length} eigenvalues`);
      checking(assembly.name, (check) => {
        check("6 the six rigid-body eigenvalues against the first elastic one",
          Math.max(...values.slice(0, 6).map(Math.abs)) / values[6], 1e-6, values.slice(0, 7).join(" "));
        check("6 negative eigenvalues below round-off", values.filter((value) => value < -1e-6 * values[6]).length, 0);
        // A seventh mode far below the eighth would be a mechanism of the assembly.
        check("6 no mechanism beside the rigid-body modes (seventh root against the eighth)",
          values[6] > 1e-4 * values[7] ? 0 : 1, 0, `${values[6]} and ${values[7]}`);
      });
    });
  });

// ---------------------------------------------------------------------------------------------------------------------
// 7. Robustness of the input: the solver gives a sound result or refuses with a clear diagnostic. It never throws an
// opaque exception, never hangs, never gives a value that is not a number.

// The mixed formulation, chosen by the program: the declarations of the geometry are in the model files.
const MIXED = { elementFormulation: MIXED_NAME };
const robustness = [];

// A deck of statics with one subcase: constraint set 1, load set 2, every nodal output and the stresses. The model is
// prepared: with nothing to declare of its geometry unless its cards do (`prepared`).
function staticDeck(cards) {
  return ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All", "SinglePointConstrForceOutput;All",
    "AppliedLoadOutput;All", "StressOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "[model]", ...prepared(cards), ""].join("\n");
}

// The positions of the grid points of a deck (in the basic system: no position frame in these decks).
function positionsOf(text) {
  const positions = new Map();
  for (const line of text.split("\n")) {
    const fields = line.split(";");
    if (fields[0] === "Node") positions.set(Number(fields[1]), fields.slice(3, 6).map(Number));
  }
  return positions;
}

// The resultant about the origin of a nodal table of forces of a deck without displacement systems.
function plainResultant(text, table) {
  const positions = positionsOf(text);
  const total = [0, 0, 0, 0, 0, 0];
  table.ids.forEach((id, row) => {
    const force = Array.from(table.values.slice(row * 6, row * 6 + 3));
    const moment = add(cross(positions.get(id) ?? [0, 0, 0], force),
      Array.from(table.values.slice(row * 6 + 3, row * 6 + 6)));
    for (let c = 0; c < 3; ++c) { total[c] += force[c]; total[3 + c] += moment[c]; }
  });
  return total;
}

// Runs a model the solver must solve soundly or refuse clearly, and records what it did. `stated`: the six values of
// the resultant of its loads about the origin, which the support forces of a sound result balance (1e-6: these
// models are ill-conditioned on purpose). `lenient`: an unsound result is returned (outcome "solved, unsound")
// for the case to record, instead of failing the case here.
function probe(name, text, options = MIXED, { analysis = "statics", stated = null, tolerance = 1e-6,
  lenient = false } = {}) {
  const entry = { name, outcome: "", detail: "", warnings: [], run: null, residual: null };
  robustness.push(entry);
  const start = performance.now();
  try {
    entry.run = withModel(text, options, (model) => solver.run(model, analysis));
  } catch (error) {
    const message = String(error?.message ?? error);
    entry.outcome = "refused";
    entry.detail = message.slice(0, 700);
    // An opaque exception: not an Error with a text, a bare number (a pointer of the module), or a trap.
    const opaque = !(error instanceof Error) || /^\s*\d*\s*$/.test(message) ||
      /unreachable|out of bounds|RuntimeError|bad_alloc|Aborted|null function|table index/.test(message);
    if (opaque) dump(`robustness ${name}`, text, options, message);
    assert.ok(!opaque, `${name}: an opaque exception: ${message.slice(0, 300)}`);
    return entry;
  } finally { entry.seconds = (performance.now() - start) / 1000; }
  entry.warnings = entry.run.diagnostics.filter((d) => d.severity === "warning").map((d) => d.text.slice(0, 300));
  if (errorsOf(entry.run).length) {
    entry.outcome = "refused";
    entry.detail = errorsOf(entry.run).map((d) => d.text).join(" | ").slice(0, 700);
    return entry;
  }
  entry.outcome = "solved";
  try {
    assertFinite(entry.run, name);
    if (stated) {
      const reactions = plainResultant(text, nodalTable(entry.run, "singlePointConstrForces", 1));
      const positions = [...positionsOf(text).values()];
      const size = Math.max(...positions.map((p) => norm(sub(p, positions[0]))), Number.MIN_VALUE);
      const forceScale = Math.max(norm(stated.slice(0, 3)), Number.MIN_VALUE);
      const momentScale = Math.max(norm(stated.slice(3)), forceScale * size);
      entry.residual = Math.max(norm(add(reactions.slice(0, 3), stated.slice(0, 3))) / forceScale,
        norm(add(reactions.slice(3), stated.slice(3))) / momentScale);
      assert.ok(entry.residual <= tolerance, `${name}: the support forces miss the loads by ${entry.residual}`);
    }
  } catch (error) {
    entry.outcome = "solved, unsound";
    entry.detail = String(error.message).slice(0, 500);
    dump(`robustness ${name}`, text, options, entry.detail);
    if (!lenient) throw error;
  }
  return entry;
}

const translationOf = (run, node) => rowsById(nodalTable(run, "displacements", 1)).get(node).slice(0, 3);

// A plate of nx × ny cells over [0, lx] × [0, ly] lifted by z(x, y), of ShellQuad4 (or pairs of ShellTria3), clamped
// along x = 0 (components `held`), with the force `force` at its far corner. Returns the deck, the far corner and
// the resultant of the force about the origin.
function plateDeck({ nx = 2, ny = 2, lx = 1, ly = 1, thickness = 0.01, z = () => 0, triangles = false, young = 7.0e10,
  poisson = 0.3, density = 2700, force = [0, 0, 1], held = "123456", cards = [], property = null } = {}) {
  const id = (i, j) => 1 + i + (nx + 1) * j;
  const lines = [];
  for (let j = 0; j <= ny; ++j)
    for (let i = 0; i <= nx; ++i) {
      const [x, y] = [(lx * i) / nx, (ly * j) / ny];
      lines.push(`Node;${id(i, j)};;${real(x)};${real(y)};${real(z(x, y))}`);
    }
  let shell = 0;
  for (let j = 0; j < ny; ++j)
    for (let i = 0; i < nx; ++i) {
      const c = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
      if (triangles)
        lines.push(`ShellTria3;${++shell};1;${c[0]};${c[1]};${c[2]}`,
          `ShellTria3;${++shell};1;${c[0]};${c[2]};${c[3]}`);
      else lines.push(`ShellQuad4;${++shell};1;${c.join(";")}`);
    }
  lines.push(property ?? `ShellProp;1;1;${real(thickness)};1;;1`,
    `IsoMat;1;${real(young)};;${real(poisson)};${real(density)}`);
  lines.push(`SinglePointConstrList;1;${held};${Array.from({ length: ny + 1 }, (_, j) => id(0, j)).join(";")}`);
  const tip = id(nx, ny);
  lines.push(`PointForceLoad;2;${tip};;1.;${force.map(real).join(";")}`, ...cards);
  const at = [lx, ly, z(lx, ly)];
  return { text: staticDeck(lines), tip, stated: [...force, ...cross(at, force)], id };
}

test("7. very thin and very thick shells are solved soundly", () => {
  // Thickness over length 1e-5: the bending stiffness is 1e-10 of the membrane stiffness. The deflection of a
  // plate in bending goes as 1 / t³: twice the thickness, an eighth of the deflection.
  const [thin, thicker] = [1.0e-5, 2.0e-5].map((thickness) => {
    const plate = plateDeck({ thickness });
    const entry = probe(`thin plate, t / L = ${thickness}`, plate.text, MIXED, { stated: plate.stated });
    assert.equal(entry.outcome, "solved", entry.detail);
    return translationOf(entry.run, plate.tip)[2];
  });
  expectWithin("7 thin plate: deflection ratio of t and 2 t against 8", "plate 2 x 2",
    Math.abs(thin / thicker / 8 - 1), 1e-4);
  const thick = plateDeck({ thickness: 1.0 });
  assert.equal(probe("thick plate, t / L = 1", thick.text, MIXED, { stated: thick.stated }).outcome, "solved");
});

test("7. elements of aspect ratio up to 1:100 give the deflection of the beam theory", () => {
  // A strip 1 wide of 1 × 2 elements (each `aspect` times longer than wide), Poisson's ratio 0, a force at each far
  // corner. The mixed formulation must solve what the classic one solves.
  const [width, thickness, young] = [1, 0.1, 7.0e10];
  checking("strip 1 x 2", (check) => {
    for (const aspect of [10, 30, 100]) {
      const length = (aspect * width) / 2;
      const plate = plateDeck({ nx: 1, ny: 2, lx: length, ly: width, thickness, poisson: 0, force: [0, 0, 0.5] });
      const text = plate.text.replace("[model]\n", `[model]\nPointForceLoad;2;${plate.id(1, 0)};;1.;0.;0.;0.5\n`);
      const beam = length ** 3 / (3 * young * (width * thickness ** 3) / 12);
      const classic = probe(`aspect ratio 1:${aspect} (classic formulation)`, text, {});
      const entry = probe(`aspect ratio 1:${aspect}`, text, MIXED);
      if (entry.outcome !== "solved") dump(`aspect ratio 1 to ${aspect}`, text, MIXED, entry.detail);
      if (classic.outcome === "solved")
        check(`7 aspect ratio 1:${aspect}: solved by the mixed formulation as by the classic one`,
          entry.outcome === "solved" ? 0 : 1, 0, entry.detail.slice(0, 300));
      if (entry.outcome === "solved")
        check(`7 aspect ratio 1:${aspect}: tip deflection against the beam theory`,
          Math.abs(translationOf(entry.run, plate.tip)[2] / beam - 1), 1e-3);
    }
  });
});

test("7. strongly warped and nearly degenerate elements are solved soundly or refused clearly", () => {
  const warped = plateDeck({ z: (x, y) => 0.5 * x * y });
  probe("strongly warped quadrilaterals (a corner lifted by half the side)", warped.text, MIXED,
    { stated: warped.stated });
  const quad = (corners, name) => {
    const lines = corners.map(([x, y, z = 0], k) => `Node;${k + 1};;${real(x)};${real(y)};${real(z)}`);
    lines.push("ShellQuad4;1;1;1;2;3;4", "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.",
      "SinglePointConstrList;1;123456;1;4", "PointForceLoad;2;2;;1.;0.;0.;1.");
    const at = corners[1].length === 3 ? corners[1] : [...corners[1], 0];
    return probe(name, staticDeck(lines), MIXED, { stated: [0, 0, 1, ...cross(at, [0, 0, 1])] });
  };
  quad([[0, 0], [1, 0], [1, 1.0e-6], [0, 1]], "quadrilateral with an edge 1e-6 of the others");
  quad([[0, 0], [1, 0], [2, Math.tan(Math.PI / 180)], [1, 1]], "quadrilateral with a corner of 179 degrees");
  quad([[0, 0], [1, 0], [2, 0], [3, 0]], "quadrilateral of zero area (four grid points on a line)");
  quad([[0, 0], [1, 0], [1, 0], [0, 1]], "quadrilateral with two corners at the same place");
  // Solids: a nearly flat tetrahedron, a hexahedron folded onto itself (two corners of a face exchanged).
  const solid = (card, corners, name) => {
    const lines = corners.map(([x, y, z], k) => `Node;${k + 1};;${real(x)};${real(y)};${real(z)}`);
    lines.push(`${card};1;1;${corners.map((_, k) => k + 1).join(";")}`, "SolidProp;1;1", "IsoMat;1;7.e10;;0.3;2700.",
      "SinglePointConstrList;1;123;1;2;3", `PointForceLoad;2;${corners.length};;1.;0.;0.;1.`);
    return probe(name, staticDeck(lines), MIXED);
  };
  solid("SolidTetra", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, 1.0e-6]], "tetrahedron nearly flat (height 1e-6)");
  const cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const folded = [cube[0], cube[1], cube[2], cube[3], cube[5], cube[4], cube[6], cube[7]];
  const entry = solid("SolidHexa", folded, "hexahedron folded (two corners exchanged)");
  assert.equal(entry.outcome, "refused", "a folded solid is refused");
});

test("7. three and four shells on one edge, a declared fold and an undeclared one", () => {
  // Flanges of 2 × 1 ShellQuad4 around the line y = z = 0 (the x axis), along the directions given; clamped at x = 0,
  // a force at the far corner of the first flange.
  const junction = (directions) => {
    const lines = [];
    const spine = (i) => i + 1;
    const edge = (f, i) => 10 * (f + 1) + i + 1;
    for (let i = 0; i <= 2; ++i) lines.push(`Node;${spine(i)};;${real(i / 2)};0.;0.`);
    directions.forEach(([y, z], f) => {
      for (let i = 0; i <= 2; ++i) lines.push(`Node;${edge(f, i)};;${real(i / 2)};${real(0.4 * y)};${real(0.4 * z)}`);
      for (let i = 0; i < 2; ++i)
        lines.push(`ShellQuad4;${10 * f + i + 1};1;${spine(i)};${spine(i + 1)};${edge(f, i + 1)};${edge(f, i)}`);
    });
    lines.push("ShellProp;1;1;0.005;1;;1", "IsoMat;1;7.e10;;0.3;2700.");
    lines.push(`SinglePointConstrList;1;123456;${[spine(0), ...directions.map((_, f) => edge(f, 0))].join(";")}`);
    const force = [0, 30, 100];
    lines.push(`PointForceLoad;2;${edge(0, 2)};;1.;${force.map(real).join(";")}`);
    const at = [1, 0.4 * directions[0][0], 0.4 * directions[0][1]];
    return { text: staticDeck(lines), stated: [...force, ...cross(at, force)], tip: edge(0, 2),
      rawEdges: [[spine(0), spine(1)], [spine(1), spine(2)]] };
  };
  // The model file with its fold declared raw.
  const declared = (model) => withSurface(model.text, { rawEdges: model.rawEdges });
  const classicTip = (model) => translationOf(withModel(model.text, {}, (m) => solver.run(m, "statics")), model.tip);
  for (const [name, directions] of [["three shells on an edge (a T)", [[1, 0], [-1, 0], [0, 1]]],
    ["four shells on an edge (a cross)", [[1, 0], [-1, 0], [0, 1], [0, -1]]],
    ["two shells at a right angle", [[1, 0], [0, 1]]]]) {
    const model = junction(directions);
    const results = {};
    for (const [label, text] of [["declared raw", declared(model)], ["not declared", model.text]]) {
      const entry = probe(`${name}, ${label}`, text, MIXED, { stated: model.stated });
      if (entry.outcome === "solved") results[label] = translationOf(entry.run, model.tip);
    }
    // The declared junction against the classic formulation (coarse mesh: 20 %), and the effect of not declaring it.
    const classic = classicTip(model);
    if (results["declared raw"])
      record(`7 ${name}, declared raw: tip displacement against the classic one (coarse)`, "junction",
        Math.abs(norm(results["declared raw"]) / norm(classic) - 1), 0.2);
    if (results["not declared"])
      record(`7 ${name}, not declared: tip displacement against the declared one (information)`, "junction",
        Math.abs(norm(results["not declared"]) / norm(results["declared raw"] ?? classic) - 1), Infinity);
  }
});

test("7. grid points without elements, elements without property, loads and supports off the structure", () => {
  const plate = plateDeck();
  probe("an unconnected grid point", plate.text.replace("[model]\n", "[model]\nNode;900;;5.;5.;5.\n"), MIXED,
    { stated: plate.stated });
  const noProperty = probe("a shell whose property is missing",
    plateDeck({ property: "ShellProp;7;1;0.01;1;;1" }).text);
  assert.equal(noProperty.outcome, "refused", "a missing property is refused");
  const noMaterial = probe("a property whose material is missing",
    plateDeck({ property: "ShellProp;1;9;0.01;9;;9" }).text);
  assert.equal(noMaterial.outcome, "refused", "a missing material is refused");
  // A force on a grid point no element uses, held by a support there: the support takes it.
  const off = plate.text.replace("[model]\n",
    "[model]\nNode;900;;5.;5.;5.\nSinglePointConstrList;1;123456;900\nPointForceLoad;2;900;;1.;3.;4.;5.\n");
  probe("a load and a support on a grid point no element uses", off, MIXED,
    { stated: plate.stated.map((value, c) => value + [3, 4, 5, ...cross([5, 5, 5], [3, 4, 5])][c]) });
  probe("a load on a grid point no element uses, without support",
    plate.text.replace("[model]\n", "[model]\nNode;900;;5.;5.;5.\nPointForceLoad;2;900;;1.;3.;4.;5.\n"));
  // Declarations of the geometry that name what the model does not have are refused (specification, section 4.4):
  // two grid points that are not the ends of an edge, a grid point or an element that does not exist. A model file
  // that no preprocessor prepared (no declaration at all) is refused too.
  const declaring = (geometry) => withSurface(plate.text, geometry);
  const cases = [["a raw edge between two grid points that no edge joins", declaring({ rawEdges: [[1, 9]] })],
    ["a raw edge with a grid point that does not exist", declaring({ rawEdges: [[1, 99]] })],
    ["a sharp edge between two grid points that no edge joins", declaring({ sharpEdges: [[1, 9]] })],
    ["a smooth edge for an element that is no shell of it", declaring({ smoothEdges: [[1, 2, 77]] })],
    ["a line end on two grid points that no edge joins", declaring({ lineEnds: [[1, 9]] })],
    ["an edge declared both smooth and sharp", declaring({ smoothEdges: [[1, 2]], sharpEdges: [[1, 2]] })],
    ["a raw element that does not exist", declaring({ rawElements: [77] })],
    ["no declaration at all", unprepared(plate.text)]];
  checking("plate 2 x 2", (check) => {
    for (const [name, text] of cases) {
      const entry = probe(`geometry declarations: ${name}`, text, MIXED);
      check(`7 geometry declarations: ${name}: refused`, entry.outcome === "refused" ? 0 : 1, 0, entry.outcome);
    }
  });
});

test("7. a model in millimetres and tonnes gives the results of the model in metres", () => {
  const metres = plateDeck({ nx: 3, ny: 2, lx: 0.6, ly: 0.4, thickness: 0.002,
    cards: ["GravityLoad;2;;9.81;0.;0.;-1."] });
  const millimetres = plateDeck({ nx: 3, ny: 2, lx: 600, ly: 400, thickness: 2, young: 7.0e4, density: 2.7e-9,
    cards: ["GravityLoad;2;;9810.;0.;0.;-1."] });
  const [a, b] = [probe("plate in metres, kilograms, pascals", metres.text),
    probe("plate in millimetres, tonnes, megapascals", millimetres.text)];
  assert.equal(a.outcome, "solved", a.detail);
  assert.equal(b.outcome, "solved", b.detail);
  const worst = compareRuns(a.run, b.run, { lengthScale: 1000, size: 0.6,
    nodal: (id, values, kind) => (kind === "displacements" ? [...scale(values.slice(0, 3), 1000), ...values.slice(3)]
      : [...values.slice(0, 3), ...scale(values.slice(3), 1000)]) });
  expectWithin("7 millimetres and tonnes against metres", "plate 3 x 2", worst.difference, SAME, worst.what);
});

test("7. a mechanism is refused with a diagnostic that names grid points", () => {
  // The plate without any support, and the plate held along a line (a hinge): both are mechanisms in statics.
  const free = plateDeck().text.replace(/SinglePointConstrSelection;1\n/, "")
    .replace(/SinglePointConstrList;[^\n]*\n/, "");
  const hinged = plateDeck({ held: "123" }).text;
  checking("plate 2 x 2", (check) => {
    for (const [name, text] of [["a plate without supports", free], ["a plate hinged along an edge", hinged]]) {
      const entry = probe(name, text);
      check(`7 ${name}: refused`, entry.outcome === "refused" ? 0 : 1, 0, entry.outcome);
      // The unknowns of the formulation have no grid point: "grid 0 component 26" tells a user nothing.
      check(`7 ${name}: the diagnostic names grid points, not unknowns of the formulation`,
        /grid 0 component|grid -\d+ component/.test(entry.detail) ? 1 : 0, 0, entry.detail.slice(0, 300));
      const classic = probe(`${name} (classic formulation)`, text, {});
      check(`7 ${name}: the classic formulation refuses it too`, classic.outcome === "refused" ? 0 : 1, 0);
    }
  });
});

test("7. a model of line elements only and a model of solids only are solved soundly", () => {
  // A portal frame of three Bar2 and a Rod2 brace.
  const frame = staticDeck(["Node;1;;0.;0.;0.", "Node;2;;0.;0.;1.", "Node;3;;1.;0.;1.", "Node;4;;1.;0.;0.",
    "Bar2;1;1;1;2;0.;;1.;0.", "Bar2;2;1;2;3;0.;;1.;0.", "Bar2;3;1;3;4;0.;;1.;0.", "Rod2;4;2;1;3",
    "BarProp;1;1;1.e-4;1.e-9;2.e-9;1.5e-9", "RodProp;2;1;5.e-5;1.e-10", "IsoMat;1;7.e10;;0.3;2700.",
    "SinglePointConstrList;1;123456;1;4", "PointForceLoad;2;2;;1.;100.;50.;0.", "PointMomentLoad;2;3;;1.;0.;5.;2."]);
  const lines = probe("line elements only", frame, MIXED,
    { stated: [100, 50, 0, ...add(cross([0, 0, 1], [100, 50, 0]), [0, 5, 2])] });
  assert.equal(lines.outcome, "solved", lines.detail);
  const block = bracketAssembly({ midSide: "none", without: ["cover", "flange", "plate", "glue", "bar", "moments"] });
  const solids = withAssembly(block, MIXED_NAME, (model) => solver.run(model, "statics", { subcases: [1] }));
  assert.equal(errorsOf(solids).length, 0, JSON.stringify(errorsOf(solids)));
  assertFinite(solids, "solids only");
  const { force, moment } = equilibriumResiduals(block, solids, 1);
  expectWithin("7 solids only: equilibrium of the tables", block.name, Math.max(force, moment), SAME);
});

// The failures found by the assemblies, each on the smallest model that shows it.

test("7. reduced: the support forces of a curved clamped edge balance the load in moment", () => {
  // A strip of 2 × 2 ShellQuad4 whose clamped edge x = 0 is an arc (a cylinder along x), a force at the far corner.
  const plate = plateDeck({ ly: 0.5, z: (x, y) => 0.5 * (y - 0.25) ** 2 });
  const entry = probe("reduced: curved clamped edge", plate.text, MIXED,
    { stated: plate.stated, tolerance: SAME, lenient: true });
  expectWithin("7 reduced: support forces of a curved clamped edge against the load", "strip 2 x 2", entry.residual,
    SAME);
});

test("7. reduced: a flat plate with a re-entrant corner beside triangles keeps its area", () => {
  // An L of two ShellTria3 and two ShellQuad4 in a plane, under a unit pressure: the load is the area, 3. The mesh
  // alone does not say that the outline turns a corner at grid point 7, where a triangle and two quadrilaterals meet.
  // The model file says it, as a preprocessor that knows the outline writes it (spec section 4.4): the lines of the
  // outline edges 3-7 and 8-7 end there, which makes no edge raw; or the two edges are raw. The plate then carries
  // its area whatever the corner its triangles start from.
  const cards = (first, second) => staticDeck(["Node;2;;1.;0.;0.", "Node;3;;2.;0.;0.", "Node;6;;1.;1.;0.",
    "Node;7;;2.;1.;0.", "Node;8;;3.;1.;0.", "Node;10;;1.;2.;0.", "Node;11;;2.;2.;0.", "Node;12;;3.;2.;0.",
    `ShellTria3;1;1;${first}`, `ShellTria3;2;1;${second}`, "ShellQuad4;3;1;6;7;11;10", "ShellQuad4;4;1;7;8;12;11",
    "ShellProp;1;1;0.01;1;;1", "IsoMat;1;7.e10;;0.3;2700.", "SinglePointConstrList;1;123456;2;3",
    "ShellPressureLoad;2;1.;1;2;3;4"]);
  const starts = [["first", cards("2;3;7", "2;7;6")], ["second", cards("3;7;2", "7;6;2")],
    ["third", cards("7;2;3", "6;2;7")]];
  const declarations = [["its outline declared ended at the corner", { lineEnds: [[3, 7], [8, 7]] }],
    ["the edges of the corner declared raw", { rawEdges: [[3, 7], [7, 8]] }]];
  checking("flat L of 4 shells", (check) => {
    for (const [declared, geometry] of declarations)
      for (const [start, plain] of starts) {
        const label = `flat L with triangles started from their ${start} corner, ${declared}`;
        const text = withSurface(plain, geometry);
        const entry = probe(`reduced: ${label}`, text, MIXED);
        assert.ok(entry.run, entry.detail);
        const carried = -plainResultant(text, nodalTable(entry.run, "singlePointConstrForces", 1))[2];
        if (Math.abs(carried / 3 - 1) > SAME) dump(label, text, MIXED, entry.detail);
        check(`7 reduced: pressure carried by a ${label}, against its area`, Math.abs(carried / 3 - 1), SAME,
          `carried ${carried}`);
        const warnings = entry.run.diagnostics.filter((d) => d.severity === "warning");
        check(`7 reduced: ${label}: warnings`, warnings.length, 0, warnings.map((d) => d.text).join(" | "));
      }
    // A model file that declares nothing at the corner: the outline is read as a line that goes on through it. The
    // solver says so: it refuses the model (a triangle left without an angle) or warns that the corner is rounded.
    for (const [start, plain] of starts) {
      const entry = probe(`reduced: flat L with triangles started from their ${start} corner, nothing declared`,
        plain, MIXED, { lenient: true });
      const said = entry.outcome === "refused" ||
        (entry.run?.diagnostics ?? []).some((d) => /geometry-(kink|rounding)/.test(d.code));
      check(`7 reduced: flat L with triangles started from their ${start} corner, nothing declared: refused, or ` +
        "warned that the corner is rounded", said ? 0 : 1, 0, entry.outcome);
    }
  });
});

test("7. reduced: a couple at a grid point of a Rod2 on a shell edge bends the shell", () => {
  // A strip of 4 × 1 ShellQuad4 clamped at x = 0, a line element along its edge y = 0, a couple about y at the tip of
  // that edge: without the line, or with a Bar2, or in the classic formulation, the strip bends.
  const strip = (line) => {
    const plate = plateDeck({ nx: 4, ny: 1, ly: 0.25, thickness: 0.005, force: [0, 0, 0] });
    const cards = ["RodProp;2;1;1.e-4;2.e-10", "BarProp;3;1;1.e-4;1.e-9;1.e-9;2.e-10",
      "PointMomentLoad;2;5;;1.;0.;1.;0."];
    for (let i = 1; i <= 4; ++i) {
      if (line === "Rod2") cards.push(`Rod2;${100 + i};2;${i};${i + 1}`);
      if (line === "Bar2") cards.push(`Bar2;${100 + i};3;${i};${i + 1};0.;;0.;1.`);
    }
    return plate.text.replace("[model]\n", `[model]\n${cards.join("\n")}\n`);
  };
  const deflection = (line, options) =>
    translationOf(withModel(strip(line), options, (m) => solver.run(m, "statics")), 5)[2];
  const alone = deflection("", MIXED);
  checking("strip 4 x 1", (check) => {
    check("7 reduced: couple at a shell grid point, mixed against classic", Math.abs(alone / deflection("", {}) - 1),
      YARDSTICK);
    check("7 reduced: couple at a grid point of a Bar2 on a shell edge, mixed against classic",
      Math.abs(deflection("Bar2", MIXED) / deflection("Bar2", {}) - 1), YARDSTICK);
    const rod = deflection("Rod2", MIXED);
    if (Math.abs(rod / deflection("Rod2", {}) - 1) > YARDSTICK)
      dump("rod on a shell edge", strip("Rod2"), MIXED);
    check("7 reduced: couple at a grid point of a Rod2 on a shell edge, mixed against classic",
      Math.abs(rod / deflection("Rod2", {}) - 1), YARDSTICK, `deflection ${rod}, classic ${deflection("Rod2", {})}`);
  });
});

// Two plates of n × n ShellQuad4 side by side, the first clamped along x = 0, the second loaded on its far edge,
// joined along their common edge (two grid points at each place) by springs of stiffness `stiffness` on the
// components 1 to 5.
function sprungPlates({ n = 3, young = 7.0e10, thickness = 0.002, stiffness = 1.0e8, cards = [] } = {}) {
  const id = (plate, i, j) => plate * 1000 + i * (n + 1) + j + 1;
  const lines = [];
  for (const plate of [0, 1])
    for (let i = 0; i <= n; ++i)
      for (let j = 0; j <= n; ++j) lines.push(`Node;${id(plate, i, j)};;${real(plate + i / n)};${real(j / n)};0.`);
  let shell = 0;
  for (const plate of [0, 1])
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j)
        lines.push(`ShellQuad4;${++shell};1;${id(plate, i, j)};${id(plate, i + 1, j)};${id(plate, i + 1, j + 1)};` +
          `${id(plate, i, j + 1)}`);
  lines.push(`ShellProp;1;1;${real(thickness)};1;;1`, `IsoMat;1;${real(young)};;0.3;2700.`);
  let spring = 500;
  for (let j = 0; j <= n; ++j)
    for (let c = 1; c <= 5; ++c)
      lines.push(`SpringDofInline;${++spring};${real(stiffness)};${id(0, n, j)};${c};${id(1, 0, j)};${c}`);
  lines.push(`SinglePointConstrList;1;123456;${Array.from({ length: n + 1 }, (_, j) => id(0, 0, j)).join(";")}`);
  for (let j = 0; j <= n; ++j) lines.push(`PointForceLoad;2;${id(1, n, j)};;1.;0.;0.;1.`);
  return { text: staticDeck([...lines, ...cards]), stated: [0, 0, n + 1, (n + 1) / 2, -2 * (n + 1), 0] };
}

test("7. reduced: stiff springs on the rotations of shell grid points are solved, as in the classic one", () => {
  // Springs of 1e6 (stiff links, as users write them) between two aluminium plates 2 mm thick.
  const plates = sprungPlates({ stiffness: 1.0e6 });
  const classic = probe("reduced: stiff rotational springs (classic formulation)", plates.text, {},
    { stated: plates.stated });
  const mixed = probe("reduced: stiff rotational springs", plates.text, MIXED, { stated: plates.stated });
  if (mixed.outcome !== "solved") dump("stiff rotational springs", plates.text, MIXED, mixed.detail);
  checking("two plates 3 x 3", (check) => {
    check("7 reduced: stiff rotational springs solved by the classic formulation",
      classic.outcome === "solved" ? 0 : 1, 0);
    check("7 reduced: stiff rotational springs solved by the mixed formulation", mixed.outcome === "solved" ? 0 : 1, 0,
      mixed.detail.slice(0, 400));
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// 8. The interface with the formulation.

test("8. checkSupport answers for each analysis", () => {
  const cases = [["statics", panelAssembly({})], ["modes", panelAssembly({ analysis: "modes" })],
    ["buckling", panelAssembly({ analysis: "buckling" })], ["aeroelastic", panelAssembly({})]];
  for (const [analysis, assembly] of cases)
    withAssembly(assembly, MIXED_NAME, (model) => {
      const viaModel = model.checkSupport(analysis);
      const viaSolver = solver.checkSupport(model, analysis);
      assert.deepEqual(viaSolver.refusals.map((d) => d.code), viaModel.refusals.map((d) => d.code));
      assert.ok(viaModel.supported.length > 0, "the cards supported are listed");
      // The panel has nothing the formulation refuses; an aeroelastic analysis of a deck without a trim is refused
      // or complete, never an exception.
      if (analysis !== "aeroelastic") assert.equal(viaModel.isComplete, true, JSON.stringify(viaModel.refusals));
      robustness.push({ name: `checkSupport("${analysis}")`, outcome: viaModel.isComplete ? "complete" : "refusals",
        detail: viaModel.refusals.map((d) => d.text).join(" | ").slice(0, 400),
        warnings: viaModel.ignored.map((d) => d.text) });
    });
});

// A plate whose deck holds a static subcase, a normal modes subcase and a buckling subcase of the first.
function threeAnalysesDeck() {
  const plate = plateDeck({ nx: 3, ny: 2, ly: 0.5, thickness: 0.004, force: [-1000, 0, 5], z: (x, y) => 0.1 * x * y });
  return plate.text.replace("SolutionSequence;101", "SolutionSequence;105")
    .replace("SubcaseStart;1\n", "EigenMethodSelection;3\nSubcaseStart;1\n").replace("[model]\n",
      "SubcaseStart;2\nSinglePointConstrSelection;1\nStaticSubcaseSelection;1\n[model]\nLanczosEigenMethod;3;;;4\n");
}

// Every number a run gives, as one list: its tables and its eigenvalues.
const numbersOf = (run) => [...run.nodalTables.flatMap((t) => [...t.values]),
  ...run.elementTables.flatMap((t) => [...t.values]),
  ...run.eigenvalues.flatMap((e) => e.rows.map((row) => row.eigenvalue))];

test("8. statics, modes and buckling one after another on one model leave no state behind", () => {
  const text = threeAnalysesDeck();
  const sequence = (order) => withModel(text, MIXED, (model) => order.map((analysis) => {
    const run = solver.run(model, analysis);
    assert.equal(errorsOf(run).length, 0, `${analysis}: ${JSON.stringify(errorsOf(run))}`);
    return [analysis, numbersOf(run)];
  }));
  const first = sequence(["statics", "modes", "buckling", "statics", "modes", "buckling"]);
  const second = sequence(["buckling", "modes", "statics"]);
  checking("plate 3 x 2", (check) => {
    for (const analysis of ["statics", "modes", "buckling"]) {
      const runs = [...first, ...second].filter(([name]) => name === analysis).map(([, numbers]) => numbers);
      assert.ok(runs[0].length > 0, `${analysis} gives results`);
      const same = runs.every((numbers) => numbers.length === runs[0].length &&
        numbers.every((value, k) => Object.is(value, runs[0][k])));
      check(`8 ${analysis} repeated and in another order (0: bit-identical)`, same ? 0 : 1, 0);
    }
  });
});

test("8. building, running and deleting a model many times does not grow the memory", () => {
  const text = threeAnalysesDeck();
  const once = () => withModel(text, MIXED, (model) => solver.run(model, "statics").nodalTables.length);
  for (let k = 0; k < 5; ++k) once();
  const megabytes = () => { const m = process.memoryUsage(); return (m.rss + m.external) / 1048576; };
  const before = megabytes();
  for (let k = 0; k < 40; ++k) once();
  const growth = megabytes() - before;
  expectWithin("8 memory growth over 40 models built, run and deleted, megabytes", "plate 3 x 2", growth, 64);
});

test("8. outputs asked for a set of grid points and elements are the rows of the whole tables", () => {
  const whole = panelAssembly({});
  const subset = panelAssembly({});
  const [grids, elements] = [[1, 7, 14, whole.keys.apex], [2, 5, 13, 101001, 103004]];
  subset.caseLines = [`SetDefinition;71;${grids.join(";")}`, `SetDefinition;72;${elements.join(";")}`,
    ...subset.caseLines.map((line) => line.replace(/^(Displacement|SinglePointConstrForce)Output;All/, "$1Output;71")
      .replace(/^(ElemForce|Stress|Strain)Output;All/, "$1Output;72"))];
  const run = runAssembly(subset, MIXED_NAME, "statics", { subcases: [1] });
  checking(subset.name, (check) => {
    let [stray, tables] = [0, 0];
    const where = [];
    const count = (table, wanted) => {
      const others = [...table.ids].filter((id) => !wanted.includes(id)).length;
      if (others) where.push(`${tableName(table)}: ${others} of ${table.ids.length}`);
      stray += others;
      ++tables;
    };
    run.nodalTables.filter((t) => /displacements|singlePointConstrForces/.test(t.kind))
      .forEach((table) => count(table, grids));
    run.elementTables.filter((t) => /^(stress|strain|forces)$/.test(t.kind)).forEach((table) => count(table, elements));
    assert.ok(tables >= 5, `${tables} tables of the sets`);
    check("8 rows outside the sets asked", stray, 0, where.join("; "));
    // Every row given is the row of the run with every output.
    const worst = compareRuns(run, baseRun("panel"), { size: whole.size,
      skip: (table) => !/displacements|singlePointConstrForces|stress|strain|forces/.test(table.kind) });
    const rows = [...worst.byKind.values()].filter((entry) => !/rows for|no row/.test(entry.what));
    check("8 values of the rows of the sets against the whole tables",
      Math.max(0, ...rows.map((entry) => entry.difference)), SAME, rows[0]?.what ?? "");
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// The summary: every check with its largest difference, the outcome of every robustness case, the yardstick and the
// times. OOS_ASSEMBLIES_REPORT=<file> also writes it as JSON.

test("summary of the checks (printed)", () => {
  const lines = ["", "check | model | value | tolerance | result"];
  for (const row of summary)
    lines.push(`${row.check} | ${row.model} | ${Number(row.value).toExponential(2)} | ${row.tolerance} | ` +
      `${row.pass ? "pass" : `FAIL ${row.what.slice(0, 160)}`}`);
  lines.push("", "yardstick: quantity | model | mixed | classic (successive meshes) | mixed / fine | conclusive");
  for (const row of yardstickRows)
    lines.push(`${row.what} | ${row.model} | ${row.mixed.toExponential(4)} | ${row.classic.map((v) => v.toExponential(4)).join(" -> ")} | ${row.ratio.toFixed(3)} | ${row.conclusive ? "yes" : "no"}`);
  lines.push(...yardstickNotes, "", "robustness: case | outcome | residual | diagnostic");
  for (const row of robustness) {
    const residual = row.residual === null || row.residual === undefined ? "" : row.residual.toExponential(1);
    const warnings = row.warnings?.length ? ` [warnings: ${row.warnings.join(" / ").slice(0, 200)}]` : "";
    const detail = row.detail.replace(/\s+/g, " ").slice(0, 260);
    lines.push(`${row.name} | ${row.outcome} | ${residual} | ${detail}${warnings}`);
  }
  lines.push("", "times: model | formulation | analysis | path | elements | seconds");
  for (const row of timings)
    lines.push(`${row.model} | ${row.formulation} | ${row.analysis} | ${row.settings} | ${row.elements} | ` +
      `${row.seconds.toFixed(2)}`);
  console.log(lines.join("\n"));
  if (process.env.OOS_ASSEMBLIES_REPORT)
    writeFileSync(process.env.OOS_ASSEMBLIES_REPORT, JSON.stringify({ summary, yardstick: yardstickRows, yardstickNotes,
      robustness: robustness.map(({ run, ...rest }) => rest), timings }, null, 1));
});
