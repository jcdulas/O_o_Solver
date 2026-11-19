// Validation of the glued contact of shells in the two element formulations, through the whole path (model file,
// reader, modelFromReader, run). Every model is synthetic.
//
// A grid point glued to a face of a shell follows the normal fibre of the shell at the point it is tied to
// (assembly/constraints/GlueConstraints): the translation of the sheet, its rotation about the axes of its plane (the
// rotations of the grid points in the ClassicFormulation) and its rotation in its plane, carried over the gap between
// the point and the face. With GlueOption 3 or 4 the rotations of the point are tied to that rotation too. In the
// MixedHybridFormulation, whose elements have unknowns along their edges, the edges between glued grid points are
// tied along their length the same way (their displacement, and with GlueOption 3 or 4 their rotation about
// themselves).
//
// What is checked:
//   1. two plates glued face to face with a gap between their mid-surfaces follow a rigid motion (both formulations);
//   2. a strip made of two parts glued end to end by their rotations is the strip in one piece (both formulations);
//   3. a web glued on a skin by its rotations is the web meshed with the skin (both formulations);
//   4. what the glue says of the rotations it ties, and of those it cannot tie (the face of a solid);
//   5. small glued joints against their closed forms, in both formulations: a beam of two glued layers (with and
//      without a gap between them), a lap splice in bending and in tension, a web glued on a rigid skin, the first
//      frequency of a beam of two layers, a plate glued on a solid under a uniform strain.
//
// A case that a formulation does not hold within its criterion keeps the criterion and is marked "todo" with what is
// measured.
//
// Run: node --test wasm/test/validation/glue.test.mjs   (Node.js 24; needs ../O_o_Reader/build/wasm/dist)

import assert from "node:assert/strict";
import { CLASSIC, MIXED, nodal, real, runModel, test } from "./support.mjs";

const FORMULATIONS = [{ name: "ClassicFormulation", options: CLASSIC }, { name: "MixedHybridFormulation",
  options: MIXED }];
const ALUMINIUM = { young: 7.0e10, poisson: 0.3, density: 2700 };
const q = real;

const add = (a, b) => a.map((value, i) => value + b[i]);
const scale = (factor, a) => a.map((value) => factor * value);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(...a);

// The cards of a model and the positions of its grid points.
class Deck {
  constructor() {
    this.cards = [];
    this.positions = new Map();
  }

  node(id, position) {
    this.positions.set(id, position);
    this.cards.push(`Node;${id};;${position.map(real).join(";")}`);
    return id;
  }

  add(...fields) { this.cards.push(fields.join(";")); }

  // A plate of ShellQuad4 of nx × ny cells: the grid point (i, j) at origin + i·alongX + j·alongY (the vectors of one
  // cell), numbered after `first`, as its elements. Returns node(i, j), the elements and the grid points.
  plate({ first, nx, ny, origin, alongX, alongY, property }) {
    const node = (i, j) => first + 1 + i + (nx + 1) * j;
    for (let j = 0; j <= ny; ++j)
      for (let i = 0; i <= nx; ++i) this.node(node(i, j), add(origin, add(scale(i, alongX), scale(j, alongY))));
    const elements = [];
    for (let j = 0; j < ny; ++j)
      for (let i = 0; i < nx; ++i) {
        elements.push(first + 1 + i + nx * j);
        const corners = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
        this.add("ShellQuad4", elements.at(-1), property, ...corners);
      }
    const nodes = [];
    for (let j = 0; j <= ny; ++j) for (let i = 0; i <= nx; ++i) nodes.push(node(i, j));
    return { node, elements, nodes, nx, ny };
  }

  // The body `secondary` (its elements) glued on the body `primary` (its elements, or { property } for the skin of
  // the solids of a property), with the glue option `option` (1: the
  // translations, 3: the rotations too) and, when given, the distance within which a grid point is glued to a face
  // (the default: 2.5 % of the size of the faces).
  glue(primary, secondary, option, distance = "") {
    if (primary.property) this.add("ContactPropertyRegion", 1, primary.property);
    else this.add("ContactElemSurface", 1, ...primary);
    this.add("ContactElemSurface", 2, ...secondary);
    this.add("ContactBody", 1, "ThreeDimensional", "Deformable", 1);
    this.add("ContactBody", 2, "ThreeDimensional", "Deformable", 2);
    this.add(`ContactTable;1;;;;;;Secondary;2;${distance};;;;${option};;;;;;;;;;;;;;;;;;;;;;;;;Primary;1;1`);
    this.isGlued = true;
  }

  // The text of a modal model of `count` modes on the supports of the set 1.
  modes(count) {
    return ["[executive]", "SolutionSequence;103", "[case]", ...(this.isGlued ? ["ContactSelection;1"] : []),
      "SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3", "[model]", ...this.cards,
      `LanczosEigenMethod;3;;;${count}`, ""].join("\n");
  }

  // The text of a static model of one subcase: the supports of the set 1, the loads of the set 2 (when given).
  statics(hasLoads = true) {
    return ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
      ...(this.isGlued ? ["ContactSelection;1"] : []), "SubcaseStart;1", "SinglePointConstrSelection;1",
      ...(hasLoads ? ["StaticLoadSelection;2"] : []), "[model]", ...this.cards, ""].join("\n");
  }
}

const shellCards = (deck, properties, thickness = 0.002, poisson = ALUMINIUM.poisson) => {
  for (const property of properties) deck.add("ShellProp", property, 1, q(thickness), 1, "", 1);
  deck.add("IsoMat", 1, q(ALUMINIUM.young), "", q(poisson), q(ALUMINIUM.density));
};

// ---------------------------------------------------------------------------------------------------------------------
// 1. A rigid motion of two plates glued with a gap
//
// A plate of 0.3 × 0.2 m (3 × 2 cells) whose grid points are given the translations of a rigid motion
// u(x) = a + ω × x, and a smaller plate of 0.16 × 0.1 m (3 × 2 cells, its lines not those of the first) glued on it,
// its mid-surface 4 mm above (the sum of the half thicknesses of two sheets of 4 mm). The glued plate hangs on the
// glue alone: it must follow the motion, to the rounding of the solution (measured: 1e-16 and 2e-13). Tied by the
// translations of the face alone, a glued point 4 mm off its face missed the motion by ω times the gap (0.97 % of
// the largest translation here).

const RIGID = { translation: [1.0e-3, -2.0e-3, 3.0e-3], rotation: [0.01, -0.02, 0.015], gap: 0.004 };

for (const { name, options } of FORMULATIONS)
  test(`1. a plate glued 4 mm above another follows the rigid motion of that plate (${name})`, (t) => {
    const deck = new Deck();
    const lower = deck.plate({ first: 0, nx: 3, ny: 2, origin: [0, 0, 0], alongX: [0.1, 0, 0], alongY: [0, 0.1, 0],
      property: 1 });
    const upper = deck.plate({ first: 100, nx: 3, ny: 2, origin: [0.07, 0.04, RIGID.gap],
      alongX: [0.16 / 3, 0, 0], alongY: [0, 0.05, 0], property: 2 });
    shellCards(deck, [1, 2], 0.004);
    deck.glue(lower.elements, upper.elements, 1, q(1.5 * RIGID.gap));
    const motion = (x) => add(RIGID.translation, cross(RIGID.rotation, x));
    for (const id of lower.nodes)
      motion(deck.positions.get(id)).forEach((value, c) => deck.add("SinglePointConstr", 1, id, c + 1, q(value)));
    const run = runModel(deck.statics(false), options);
    const largest = Math.max(...upper.nodes.map((id) => norm(motion(deck.positions.get(id)))));
    let worst = 0;
    for (const id of upper.nodes) {
      const u = nodal(run, "displacements", id).slice(0, 3);
      worst = Math.max(worst, norm(u.map((value, c) => value - motion(deck.positions.get(id))[c])));
    }
    t.diagnostic(`the glued plate less the rigid motion: ${(worst / largest).toExponential(2)} of its largest ` +
      "translation");
    assert.ok(worst <= 1e-8 * largest, `${worst} for ${largest}`);
    assert.ok(run.diagnostics.some((d) => d.code === "glued-pair" && /12 grid points tied/.test(d.text)),
      JSON.stringify(run.diagnostics.map((d) => d.text)));
    // The gap is followed: nothing to warn of.
    assert.ok(!run.diagnostics.some((d) => d.code === "glue-gap-kept"), "no gap is kept");
  });

// ---------------------------------------------------------------------------------------------------------------------
// 2. A strip glued end to end
//
// A strip of 0.2 × 0.04 m, 2 mm thick, clamped at X = 0, in two parts that meet at X = 0.1: the first of 4 × 2 cells,
// the second of 4 × 3 (its grid points of the joint are not those of the first part). The second part is glued on
// the first by its rotations too (GlueOption 3): only its grid points of the joint lie on the first part, a single
// line of points. Three loads at the free end: a force across the strip (bending), two opposite forces at the
// corners (torsion), a force in its plane along Y. Against the strip in one piece of 8 × 2 cells.
//
// Tied by the translations alone, the single line of glued points is a hinge (a mechanism: no solution).

const STRIP = { length: 0.2, width: 0.04 };
const STRIP_LOADS = [{ name: "bending", forces: [[0, 0, 0.5], [0, 0, 0.5]] },
  { name: "torsion", forces: [[0, 0, 0.5], [0, 0, -0.5]] }, { name: "in its plane", forces: [[0, 5, 0], [0, 5, 0]] }];

// The displacement of the two corners of the free end under `load`, the strip in one piece or glued with `option`.
function stripCase(load, option, options) {
  const deck = new Deck();
  const half = STRIP.length / 2;
  let end;
  if (option === 0) {
    const whole = deck.plate({ first: 0, nx: 8, ny: 2, origin: [0, 0, 0], alongX: [STRIP.length / 8, 0, 0],
      alongY: [0, STRIP.width / 2, 0], property: 1 });
    deck.add("SinglePointConstrList", 1, 123456, ...[0, 1, 2].map((j) => whole.node(0, j)));
    end = [whole.node(8, 0), whole.node(8, 2)];
  } else {
    const first = deck.plate({ first: 0, nx: 4, ny: 2, origin: [0, 0, 0], alongX: [half / 4, 0, 0],
      alongY: [0, STRIP.width / 2, 0], property: 1 });
    const second = deck.plate({ first: 100, nx: 4, ny: 3, origin: [half, 0, 0], alongX: [half / 4, 0, 0],
      alongY: [0, STRIP.width / 3, 0], property: 1 });
    deck.add("SinglePointConstrList", 1, 123456, ...[0, 1, 2].map((j) => first.node(0, j)));
    deck.glue(first.elements, second.elements, option);
    end = [second.node(4, 0), second.node(4, 3)];
  }
  shellCards(deck, [1]);
  load.forces.forEach((force, k) => deck.add("PointForceLoad", 2, end[k], "", q(1), ...force.map(q)));
  const run = runModel(deck.statics(), options);
  return { run, corners: end.map((id) => nodal(run, "displacements", id).slice(0, 3)) };
}

for (const { name, options } of FORMULATIONS)
  test(`2. a strip in two parts glued end to end by their rotations is the strip in one piece (${name})`, (t) => {
    for (const load of STRIP_LOADS) {
      const [whole, glued] = [stripCase(load, 0, options), stripCase(load, 3, options)];
      const size = Math.max(...whole.corners.map(norm));
      const worst = Math.max(...whole.corners.map((u, k) => norm(u.map((value, c) => value - glued.corners[k][c]))));
      t.diagnostic(`${load.name}: the corners of the free end move by ${size.toExponential(4)}, the glued strip ` +
        `differs by ${(worst / size).toExponential(2)} of it`);
      // Measured: 9e-4 (bending), 5e-3 (torsion), 2e-3 (in its plane); 2e-5, 7e-3 and 2e-4 in the
      // MixedHybridFormulation.
      assert.ok(worst <= 0.03 * size, `${load.name}: ${worst} for ${size}`);
      assert.ok(glued.run.diagnostics.some((d) => d.code === "glued-pair" && /4 by their rotations too/.test(d.text)),
        JSON.stringify(glued.run.diagnostics.map((d) => d.text)));
      assert.ok(!glued.run.diagnostics.some((d) => d.code === "glue-rotations-not-tied"), "the rotations are tied");
    }
  });

// ---------------------------------------------------------------------------------------------------------------------
// 3. A web glued on a skin
//
// A skin of 0.2 × 0.2 m (4 × 4 cells, 2 mm), clamped along X = 0 and X = 0.2, and a web of 0.2 × 0.06 m standing on
// its middle line X = 0.1, pushed along X at its top: the web bends as a cantilever out of the skin, which it turns
// about the line of the joint. The web is meshed with the skin (its grid points of the joint are those of the skin),
// or it has its own grid points there (5 cells along the joint for the 4 of the skin) and is glued by its rotations.

function webCase(isGlued, options) {
  const deck = new Deck();
  const skin = deck.plate({ first: 0, nx: 4, ny: 4, origin: [0, 0, 0], alongX: [0.05, 0, 0], alongY: [0, 0.05, 0],
    property: 1 });
  const cells = isGlued ? 5 : 4;
  const height = 0.06;
  // The web: the grid point (i, k) at Y = 0.2·i/cells, Z = height·k/3; its first row is the joint.
  const web = (i, k) => (k === 0 && !isGlued ? skin.node(2, i) : 200 + 1 + i + (cells + 1) * k);
  const webElements = [];
  for (let k = isGlued ? 0 : 1; k <= 3; ++k)
    for (let i = 0; i <= cells; ++i) deck.node(web(i, k), [0.1, 0.2 * i / cells, height * k / 3]);
  for (let k = 0; k < 3; ++k)
    for (let i = 0; i < cells; ++i) {
      webElements.push(300 + 1 + i + cells * k);
      deck.add("ShellQuad4", webElements.at(-1), 2, web(i, k), web(i + 1, k), web(i + 1, k + 1), web(i, k + 1));
    }
  shellCards(deck, [1, 2]);
  const held = [];
  for (let j = 0; j <= 4; ++j) held.push(skin.node(0, j), skin.node(4, j));
  deck.add("SinglePointConstrList", 1, 123456, ...held);
  if (isGlued) deck.glue(skin.elements, webElements, 3);
  const top = Array.from({ length: cells + 1 }, (_, i) => web(i, 3));
  top.forEach((id, i) => deck.add("PointForceLoad", 2, id, "", q(1), q(10 / cells * (i === 0 || i === cells ? 0.5 : 1)),
    q(0), q(0)));
  const run = runModel(deck.statics(), options);
  return { run, push: top.reduce((sum, id) => sum + nodal(run, "displacements", id)[0], 0) / top.length };
}

for (const { name, options } of FORMULATIONS)
  test(`3. a web glued on a skin by its rotations is the web meshed with the skin (${name})`, (t) => {
    const [meshed, glued] = [webCase(false, options), webCase(true, options)];
    t.diagnostic(`the top of the web moves by ${meshed.push.toExponential(4)} meshed with the skin, ` +
      `${glued.push.toExponential(4)} glued (ratio ${(glued.push / meshed.push).toFixed(4)})`);
    // Measured: 1.0004; 0.9924 in the MixedHybridFormulation.
    assert.ok(Math.abs(glued.push / meshed.push - 1) <= 0.05, `${glued.push} for ${meshed.push}`);
    assert.ok(!glued.run.diagnostics.some((d) => d.code === "glue-rotations-not-tied"), "the rotations are tied");
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. What is said of the rotations
//
// A plate glued by its rotations too on the face of a solid, which has no rotation to give: the translations are
// tied (nine points on one face hold the plate) and the glue says that the rotations are not.

test("4. a plate glued by its rotations on the face of a solid has its translations tied, and is told so", () => {
  const deck = new Deck();
  const corners = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]];
  [0, 0.05].forEach((z, layer) => corners.forEach(([x, y], k) => deck.node(1 + k + 4 * layer, [x, y, z])));
  deck.add("SolidHexa", 1, 3, 1, 2, 3, 4, 5, 6, 7, 8);
  deck.add("SolidProp", 3, 1);
  deck.add("SinglePointConstrList", 1, 123, 1, 2, 3, 4);
  const plate = deck.plate({ first: 100, nx: 2, ny: 2, origin: [0.02, 0.02, 0.05], alongX: [0.03, 0, 0],
    alongY: [0, 0.03, 0], property: 1 });
  shellCards(deck, [1]);
  deck.glue({ property: 3 }, plate.elements, 3);
  deck.add("PointForceLoad", 2, plate.node(1, 1), "", q(1), q(0), q(0), q(-10));
  const run = runModel(deck.statics(), CLASSIC);
  const texts = JSON.stringify(run.diagnostics.map((d) => `${d.code}: ${d.text}`));
  assert.ok(run.diagnostics.some((d) => d.code === "glued-pair" && /9 grid points tied by their translations \(0 by/
    .test(d.text)), texts);
  assert.ok(run.diagnostics.some((d) => d.code === "glue-rotations-not-tied" && /9 of the 9 tied grid points/
    .test(d.text)), texts);
});

// ---------------------------------------------------------------------------------------------------------------------
// 5. Small glued joints against their closed forms
//
// Strips of 200 × 40 mm, 2 mm thick, of a material without Poisson's ratio (ν = 0: the strips are beams), glued face
// to face on meshes whose grid points do not face each other (cells of 20 mm under cells of 16.7 × 13.3 mm). A glue
// that carries the shear between two layers one thickness apart makes one beam of them: the bending stiffness of the
// two layers together is E·b·(2t)³/12, eight times that of one layer and four times that of the two layers bending
// side by side.
//
// Each case gives the ratio of what a formulation finds to the closed form, as a diagnostic, and holds it within a
// criterion: 3 % for the deflections and the frequency on these meshes of ten cells (what the elements give on
// them, glued or not), tighter where the closed form is in the space of the elements. The "Measured" line of a case
// gives the ratios of the ClassicFormulation, then of the MixedHybridFormulation.
//
// The MixedHybridFormulation ties the grid points of the glued sheet and its edges between them, each point to its
// point of the face, with the rotation of the face read there (on an edge of the face, the rotation of that edge).
// Tied at their grid points alone, the layers of these cases slipped between them (1.28 for the cantilever of case
// 2, 1.09 for the splice of case 5); with their edges carried along their chords, as between the grid points of a
// rigid element, the glued layer stayed straight on a layer that bends (0.72 for case 1, 0.40 for case 3).

const THEORY = { width: 0.04, thickness: 0.002, length: 0.2, young: ALUMINIUM.young };
THEORY.shear = THEORY.young / 2; // G with ν = 0
// The bending stiffness E·I of `layers` layers that bend as one, and the shear stiffness κ·G·A (κ = 5/6) of a
// section of `layers` layers.
const bendingStiffness = (layers) => THEORY.young * THEORY.width * (layers * THEORY.thickness) ** 3 / 12;
const shearStiffness = (layers) => (5 / 6) * THEORY.shear * THEORY.width * layers * THEORY.thickness;
const axialStiffness = THEORY.young * THEORY.width * THEORY.thickness; // E·b·t of one layer
const strip = (deck, first, { from, to, nx, ny, z, property = 1 }) => deck.plate({ first, nx, ny, origin: [from, 0, z],
  alongX: [(to - from) / nx, 0, 0], alongY: [0, THEORY.width / ny, 0], property });
const column = (plate, i) => Array.from({ length: plate.ny + 1 }, (_, j) => plate.node(i, j));
// A force `force` shared by the grid points of a line across a strip (half shares at its two ends).
const lineLoad = (deck, ids, force) => ids.forEach((id, k) => deck.add("PointForceLoad", 2, id, "", q(1),
  ...scale((k === 0 || k === ids.length - 1 ? 0.5 : 1) / (ids.length - 1), force).map(q)));
const meanOf = (run, ids, component) => ids.reduce((sum, id) => sum + nodal(run, "displacements", id)[component], 0) /
  ids.length;
const glueDistance = q(1.5 * THEORY.thickness);

// Two strips of the whole length glued one on the other, `gap` between their mid-surfaces: the lower one of 10 × 2
// cells, the upper one of 12 × 3.
function twoLayers(deck, gap) {
  const lower = strip(deck, 0, { from: 0, to: THEORY.length, nx: 10, ny: 2, z: 0 });
  const upper = strip(deck, 1000, { from: 0, to: THEORY.length, nx: 12, ny: 3, z: gap });
  shellCards(deck, [1], THEORY.thickness, 0);
  deck.glue(lower.elements, upper.elements, 1, glueDistance);
  return { lower, upper };
}

// The end deflection of the cantilever made of the two strips of twoLayers, under a unit force across at its end.
// `isUpperClamped`: the two layers are clamped; otherwise the lower one alone, and the upper one is held by the glue.
function cantileverDeflection(options, gap, isUpperClamped) {
  const deck = new Deck();
  const { lower, upper } = twoLayers(deck, gap);
  deck.add("SinglePointConstrList", 1, 123456, ...column(lower, 0), ...(isUpperClamped ? column(upper, 0) : []));
  lineLoad(deck, column(lower, 10), [0, 0, 1]);
  return meanOf(runModel(deck.statics(), options), column(lower, 10), 2);
}

// A lap splice: one strip from the clamp to x2 = 0.12 (6 × 2 cells), another from x1 = 0.08 to L = 0.2 (6 × 3
// cells), one thickness above, glued over their overlap; a unit force `force` on the end of the second.
// `isHeldInPlane`: every grid point is held across the strips and in rotation. Returns the mean translation along
// `component` of the line of grid points `readColumn` of the second strip.
function lapSplice(options, force, component, isHeldInPlane, readColumn) {
  const deck = new Deck();
  const lower = strip(deck, 0, { from: 0, to: 0.12, nx: 6, ny: 2, z: 0 });
  const upper = strip(deck, 1000, { from: 0.08, to: THEORY.length, nx: 6, ny: 3, z: THEORY.thickness });
  shellCards(deck, [1], THEORY.thickness, 0);
  deck.add("SinglePointConstrList", 1, 123456, ...column(lower, 0));
  if (isHeldInPlane)
    deck.add("SinglePointConstrList", 1, 345, ...lower.nodes.filter((id) => !column(lower, 0).includes(id)),
      ...upper.nodes);
  deck.glue(lower.elements, upper.elements, 1, glueDistance);
  lineLoad(deck, column(upper, 6), force);
  return meanOf(runModel(deck.statics(), options), column(upper, readColumn), component);
}

// Each case: its name, the closed form, the value of a formulation, and its criterion by formulation: the largest
// |value / theory − 1| (null: the formulation does not make this joint; with `todo`: the case keeps its criterion
// and is marked "todo" with what is measured).
const DEFLECTION = 0.03;
const THEORY_CASES = [
  {
    // A cantilever of two layers glued over their whole length, both clamped, loaded across at its end:
    //     δ = P·L³/(3·E·I) + P·L/(κ·G·A),   I and A those of the two layers together.
    // Measured: 1.0125, 1.0012.
    name: "1. a cantilever of two glued layers under an end force",
    theory: () => THEORY.length ** 3 / (3 * bendingStiffness(2)) + THEORY.length / shearStiffness(2),
    value: (options) => cantileverDeflection(options, THEORY.thickness, true),
    criterion: { ClassicFormulation: { limit: DEFLECTION }, MixedHybridFormulation: { limit: DEFLECTION } },
  },
  {
    // The same cantilever, the lower layer alone clamped: the upper one takes its moment from the glue over its
    // first cells (5 % allowed for that). Measured: 1.0392, 1.0065.
    name: "2. a cantilever of two glued layers, the upper one held by the glue alone",
    theory: () => THEORY.length ** 3 / (3 * bendingStiffness(2)) + THEORY.length / shearStiffness(2),
    value: (options) => cantileverDeflection(options, THEORY.thickness, false),
    criterion: { ClassicFormulation: { limit: 0.05 }, MixedHybridFormulation: { limit: 0.05 } },
  },
  {
    // The two layers glued on one mid-surface (no gap), both clamped: they bend side by side, each about its own
    // mid-surface, and the glue carries no shear:   δ = P·L³/(3·E·2·I₁) + P·L/(κ·G·A).
    // Measured: 0.9961, 1.0001.
    name: "3. a cantilever of two layers glued without a gap bends as two layers side by side",
    theory: () => THEORY.length ** 3 / (3 * 2 * bendingStiffness(1)) + THEORY.length / shearStiffness(2),
    value: (options) => cantileverDeflection(options, 0, true),
    criterion: { ClassicFormulation: { limit: 0.01 }, MixedHybridFormulation: { limit: 0.01 } },
  },
  {
    // The lap splice under a force across its end. The beam of three segments, I₂ = 8·I₁ over the overlap,
    // M = P·(L − x):
    //     δ = P/(3·E)·[(L³ − (L − x1)³)/I₁ + ((L − x1)³ − (L − x2)³)/I₂ + (L − x2)³/I₁]   (plus the shear terms).
    // Measured: 1.0222, 1.0061 (5 % allowed: the overlap is two cells long).
    name: "4. a lap splice of two strips under an end force",
    theory: () => {
      const [x1, x2, length] = [0.08, 0.12, THEORY.length];
      const [a, b, c] = [length ** 3 - (length - x1) ** 3, (length - x1) ** 3 - (length - x2) ** 3, (length - x2) ** 3];
      return (a + c) / (3 * bendingStiffness(1)) + b / (3 * bendingStiffness(2)) +
        (x1 + length - x2) / shearStiffness(1) + (x2 - x1) / shearStiffness(2);
    },
    value: (options) => lapSplice(options, [0, 0, 1], 2, false, 6),
    criterion: { ClassicFormulation: { limit: 0.05 }, MixedHybridFormulation: { limit: 0.05 } },
  },
  {
    // The same splice pulled along its length, every grid point kept in its plane: two bars in series, the overlap
    // of twice the section. The translation is read at x = 0.18, one cell before the loaded end (a force on a grid
    // point dents a sheet in its plane around the point, and the MixedHybridFormulation resolves that dent: the
    // loaded grid points move 0.045 where the bar gives 0.020 over the last cell):
    //     u(0.18) = P/(E·b·t)·(x1 + (x2 − x1)/2 + 0.18 − x2).
    // Measured: 1.0004, 1.0081 (the elements of degree four resolve the sheets at the ends of the overlap, which
    // the two bars do not have: 1 % allowed).
    name: "5. a lap splice of two strips pulled along its length",
    theory: () => (0.08 + 0.02 + 0.06) / axialStiffness,
    value: (options) => lapSplice(options, [1, 0, 0], 0, true, 5),
    criterion: { ClassicFormulation: { limit: 0.005 }, MixedHybridFormulation: { limit: 0.01 } },
  },
  {
    // A web of 60 × 60 mm standing on a skin held at every grid point, glued by its rotations and pushed across at
    // its top: the cantilever clamped at the joint,   δ = P·h³/(3·E·I) + P·h/(κ·G·A).
    // Measured: 1.0000 (the web is a strip of three cells under a constant shear force), 1.0086 (the inside of
    // the elements of the skin is not held by its grid points: 2 % allowed).
    name: "6. a web glued by its rotations on a rigid skin, pushed at its top",
    theory: () => {
      const [height, width] = [0.06, 0.06];
      return height ** 3 / (3 * THEORY.young * width * THEORY.thickness ** 3 / 12) +
        height / ((5 / 6) * THEORY.shear * width * THEORY.thickness);
    },
    value: (options) => {
      const deck = new Deck();
      const skin = deck.plate({ first: 0, nx: 2, ny: 2, origin: [0, 0, 0], alongX: [0.05, 0, 0], alongY: [0, 0.05, 0],
        property: 2 });
      const web = deck.plate({ first: 100, nx: 2, ny: 3, origin: [0.04, 0.02, 0], alongX: [0, 0.03, 0],
        alongY: [0, 0, 0.02], property: 1 });
      shellCards(deck, [1], THEORY.thickness, 0);
      // the skin 10 mm thick: holding its grid points does not hold the inside of an element of degree four
      deck.add("ShellProp", 2, 1, q(0.01), 1, "", 1);
      deck.add("SinglePointConstrList", 1, 123456, ...skin.nodes);
      deck.glue(skin.elements, web.elements, 3);
      const top = [0, 1, 2].map((i) => web.node(i, 3));
      lineLoad(deck, top, [1, 0, 0]);
      return meanOf(runModel(deck.statics(), options), top, 0);
    },
    criterion: { ClassicFormulation: { limit: 0.001 }, MixedHybridFormulation: { limit: 0.02 } },
  },
  {
    // The first frequency of a beam of two glued layers on two simple supports:
    //     f = (π/(2·L²))·√(E·I/(ρ·A)) = (π/(2·L²))·(2t)·√(E/(12·ρ)).
    // Measured: 0.9792, 0.9973.
    name: "7. the first frequency of a simply supported beam of two glued layers",
    theory: () => Math.PI / (2 * THEORY.length ** 2) * 2 * THEORY.thickness *
      Math.sqrt(THEORY.young / (12 * ALUMINIUM.density)),
    value: (options) => {
      const deck = new Deck();
      const { lower } = twoLayers(deck, THEORY.thickness);
      deck.add("SinglePointConstrList", 1, 123, ...column(lower, 0));
      deck.add("SinglePointConstrList", 1, 23, ...column(lower, 10));
      return runModel(deck.modes(3), options, "modes").eigenvalues[0].rows[0].cycles;
    },
    criterion: { ClassicFormulation: { limit: DEFLECTION }, MixedHybridFormulation: { limit: DEFLECTION } },
  },
];

for (const theoryCase of THEORY_CASES)
  for (const { name, options } of FORMULATIONS) {
    const criterion = theoryCase.criterion[name];
    if (criterion === null) continue;
    test(`5.${theoryCase.name} (${name})`, criterion.todo ? { todo: criterion.todo } : {}, (t) => {
      const [value, theory] = [theoryCase.value(options), theoryCase.theory()];
      t.diagnostic(`${value.toExponential(5)} for ${theory.toExponential(5)}: ratio ${(value / theory).toFixed(4)}`);
      assert.ok(Math.abs(value / theory - 1) <= criterion.limit, `${value} for ${theory}`);
    });
  }

// A plate glued on a face of a solid whose grid points are given a uniform strain, u_x = ε·x: the shape functions of
// the face hold a linear field, so the plate has it too, to the rounding of the solution (measured: 2.7e-16). The
// solid of the MixedHybridFormulation does not take that strain from its grid points (the forces of its supports are
// 2.6 times those of the uniform strain; a rigid motion given the same way leaves it without a force): the plate
// reads what the face does, 3.8 % of the largest translation off the strain.
const SOLID_STRAIN_TODO = { ClassicFormulation: null, MixedHybridFormulation: "a solid of this formulation whose " +
  "grid points are given u_x = ε·x is not in a uniform strain" };
for (const { name, options } of FORMULATIONS)
  test(`5.8. a plate glued on a solid under a uniform strain has that strain (${name})`,
    SOLID_STRAIN_TODO[name] ? { todo: SOLID_STRAIN_TODO[name] } : {}, (t) => {
      const deck = new Deck();
      const strain = 1.0e-3;
      const corners = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]];
      [0, 0.05].forEach((z, layer) => corners.forEach(([x, y], k) => deck.node(1 + k + 4 * layer, [x, y, z])));
      deck.add("SolidHexa", 1, 3, 1, 2, 3, 4, 5, 6, 7, 8);
      deck.add("SolidProp", 3, 1);
      for (let id = 1; id <= 8; ++id)
        [strain * deck.positions.get(id)[0], 0, 0].forEach((value, c) => deck.add("SinglePointConstr", 1, id, c + 1,
          q(value)));
      const plate = deck.plate({ first: 100, nx: 2, ny: 2, origin: [0.02, 0.03, 0.05], alongX: [0.03, 0, 0],
        alongY: [0, 0.025, 0], property: 1 });
      shellCards(deck, [1], THEORY.thickness, 0);
      deck.glue({ property: 3 }, plate.elements, 1);
      const run = runModel(deck.statics(false), options);
      const worst = Math.max(...plate.nodes.map((id) => {
        const u = nodal(run, "displacements", id).slice(0, 3);
        return norm([u[0] - strain * deck.positions.get(id)[0], u[1], u[2]]);
      }));
      t.diagnostic(`the glued plate less the uniform strain: ${(worst / (strain * 0.1)).toExponential(2)} of the ` +
        "largest translation");
      assert.ok(worst <= 1e-8 * strain * 0.1, `${worst}`);
    });
