// Validation of the line elements of the ClassicFormulation on shells that are not plane: a twisted stiffened panel
// whose skin is of ShellQuad4 and ShellTria3, whose stringers are Bar2 (or Beam2), whose frames are Beam2 (or Bar2)
// and whose edge caps are Rod2, against the same panel whose stringers and frames are webs of ShellQuad4 on a fine
// mesh. The line elements carry their sections at half their depth under the skin (offsets along the normal of the
// skin at each grid point). Every quadrilateral of the skin is warped.
//
// Two skins: a thin one (the cells are many thicknesses wide) and a thick one (the cells come to the size of the
// thickness, where flat triangles need the bending term of their drilling stiffness: docs/shell-tria3-formulation.md,
// section 3).
//
// Last, the axis of a stiffener whose section is given by its type and dimensions: the neutral axis of a Bar2 on a
// BarShapeProp, the shear centre of a Beam2 on a BeamShapeProp (its neutral axis is then off its axis).
//
// Run: node --test wasm/test/validation/classic-stiffened-panel.test.mjs   (Node.js 24; needs the reader's dist)

import assert from "node:assert/strict";
import { CLASSIC, real, runModel, nodal, test } from "./support.mjs";

// The panel: a strip of width 0.6 and length 1.2 twisted by 60 degrees over its length (a helicoid), five stringers
// along its length and two frames across it, of rectangular sections, an edge cap along each long edge.
const PANEL = { halfWidth: 0.3, length: 1.2, twist: Math.PI / 3, young: 70e9, poisson: 0.3, stringers: 5,
  stringerDepth: 0.025, stringerThickness: 0.002, frames: [0.4, 0.8], frameDepth: 0.04, frameThickness: 0.003,
  capArea: 1.0e-4, row: 0.005 };
const PITCHES = PANEL.stringers + 1;
const BAYS = PANEL.frames.length + 1;

// The unit normal of the skin at the distance s from its axis and the station x, and the point at `depth` under it.
function normalAt(s, x) {
  const [rate, angle] = [PANEL.twist / PANEL.length, (PANEL.twist * x) / PANEL.length];
  const n = [-s * rate, -Math.sin(angle), Math.cos(angle)];
  const length = Math.hypot(...n);
  return n.map((c) => c / length);
}
function pointAt(s, x, depth = 0) {
  const angle = (PANEL.twist * x) / PANEL.length;
  const n = normalAt(s, x);
  return [x - depth * n[0], s * Math.cos(angle) - depth * n[1], s * Math.sin(angle) - depth * n[2]];
}
const fields = (v) => v.map(real).join(";");

// The skin: perPitch cells per stringer pitch, perBay per frame bay; `triangles` "none", "half" (every other cell cut
// in two, on alternating diagonals) or "all". Returns the cards and the grid.
function skin({ perPitch, perBay, triangles, thickness }) {
  const [ni, nj] = [PITCHES * perPitch, BAYS * perBay];
  const node = (i, j) => 1 + j * (ni + 1) + i;
  const across = (i) => -PANEL.halfWidth + (2 * PANEL.halfWidth * i) / ni;
  const along = (j) => (PANEL.length * j) / nj;
  const lines = [`ShellProp;1;1;${real(thickness)};1;;1`, `IsoMat;1;${real(PANEL.young)};;${real(PANEL.poisson)};0.`,
    `RodProp;9;1;${real(PANEL.capArea)};1.e-12`];
  for (let j = 0; j <= nj; ++j)
    for (let i = 0; i <= ni; ++i) lines.push(`Node;${node(i, j)};;${fields(pointAt(across(i), along(j)))}`);
  let id = 0;
  for (let j = 0; j < nj; ++j)
    for (let i = 0; i < ni; ++i) {
      const q = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
      const cut = triangles === "all" || (triangles === "half" && (i + j) % 2 === 0);
      if (!cut) lines.push(`ShellQuad4;${++id};1;${q.join(";")}`);
      else if ((i + 2 * j) % 4 < 2)
        lines.push(`ShellTria3;${++id};1;${q[0]};${q[1]};${q[2]}`, `ShellTria3;${++id};1;${q[0]};${q[2]};${q[3]}`);
      else lines.push(`ShellTria3;${++id};1;${q[0]};${q[1]};${q[3]}`, `ShellTria3;${++id};1;${q[1]};${q[2]};${q[3]}`);
    }
  const count = id;
  let cap = 900000;
  for (const i of [0, ni])
    for (let j = 0; j < nj; ++j) lines.push(`Rod2;${++cap};9;${node(i, j)};${node(i, j + 1)}`);
  return { lines, node, across, along, ni, nj, perPitch, count, held: [] };
}

// The stringers and the frames as line elements: a blade of depth d and thickness t has A = d t, I1 = t d³ / 12 in
// the plane of the normal, I2 = d t³ / 12, J = d t³ / 3, shear factors 5/6; its axis is at d / 2 under the skin.
function withLines(options) {
  const { stringerCard = "Bar2", frameCard = "Beam2" } = options;
  const m = skin(options);
  const property = (card, id, depth, t) => {
    const [area, i1, i2, j] = [depth * t, (t * depth ** 3) / 12, (depth * t ** 3) / 12, (depth * t ** 3) / 3];
    return card === "Bar2"
      ? `BarProp;${id};1;${real(area)};${real(i1)};${real(i2)};${real(j)};;${Array(8).fill("0.").join(";")};${
        real(5 / 6)};${real(5 / 6)}`
      : `BeamProp;${id};1;${real(area)};${real(i1)};${real(i2)};;${real(j)}`;
  };
  m.lines.push(property(stringerCard, 2, PANEL.stringerDepth, PANEL.stringerThickness),
    property(frameCard, 3, PANEL.frameDepth, PANEL.frameThickness));
  let id = 100000;
  const element = (card, pid, [ia, ja], [ib, jb], depth) => {
    const ends = [[m.across(ia), m.along(ja)], [m.across(ib), m.along(jb)]];
    const vector = normalAt(0.5 * (ends[0][0] + ends[1][0]), 0.5 * (ends[0][1] + ends[1][1]));
    const offsets = ends.map(([s, x]) => fields(normalAt(s, x).map((c) => -0.5 * depth * c)));
    m.lines.push(`${card};${++id};${pid};${m.node(ia, ja)};${m.node(ib, jb)};${real(vector[0])};;${real(vector[1])};${
      real(vector[2])};;;;${offsets[0]};${offsets[1]}`);
  };
  for (let s = 1; s <= PANEL.stringers; ++s)
    for (let j = 0; j < m.nj; ++j) element(stringerCard, 2, [s * m.perPitch, j], [s * m.perPitch, j + 1],
      PANEL.stringerDepth);
  for (const x of PANEL.frames) {
    const j = Math.round((x / PANEL.length) * m.nj);
    for (let i = 0; i < m.ni; ++i) element(frameCard, 3, [i, j], [i + 1, j], PANEL.frameDepth);
  }
  return m;
}

// The stringers and the frames as webs of ShellQuad4 under the skin, in rows of PANEL.row.
function withWebs(options) {
  const m = skin({ ...options, triangles: "none" });
  m.lines.push(`ShellProp;2;1;${real(PANEL.stringerThickness)};1;;1`,
    `ShellProp;3;1;${real(PANEL.frameThickness)};1;;1`);
  const web = new Map();
  let next = 500000;
  const webNode = (i, j, k) => {
    if (k === 0) return m.node(i, j);
    const key = `${i},${j},${k}`;
    if (!web.has(key)) {
      web.set(key, ++next);
      m.lines.push(`Node;${next};;${fields(pointAt(m.across(i), m.along(j), k * PANEL.row))}`);
      if (j === 0) m.held.push(next);
    }
    return web.get(key);
  };
  let id = 200000;
  const strip = (pid, rows, corners) => {
    for (let k = 0; k < rows; ++k)
      m.lines.push(`ShellQuad4;${++id};${pid};${webNode(...corners[0], k)};${webNode(...corners[1], k)};${
        webNode(...corners[1], k + 1)};${webNode(...corners[0], k + 1)}`);
  };
  for (let s = 1; s <= PANEL.stringers; ++s)
    for (let j = 0; j < m.nj; ++j)
      strip(2, Math.round(PANEL.stringerDepth / PANEL.row), [[s * m.perPitch, j], [s * m.perPitch, j + 1]]);
  for (const x of PANEL.frames) {
    const j = Math.round((x / PANEL.length) * m.nj);
    for (let i = 0; i < m.ni; ++i) strip(3, Math.round(PANEL.frameDepth / PANEL.row), [[i, j], [i + 1, j]]);
  }
  return m;
}

// The loads of the panel clamped along its end X = 0.
const LOADS = [
  { name: "a pressure", cards: (m) => [`ShellPressureLoad;2;1000.;1;Through;${m.count}`] },
  { name: "a force along the normal at the middle of the free end", cards: (m) =>
    [`PointForceLoad;2;${m.node(m.ni / 2, m.nj)};;100.;${fields(normalAt(0, PANEL.length))}`] },
  { name: "a force along the middle stringer at the free end", cards: (m) =>
    [`PointForceLoad;2;${m.node(m.ni / 2, m.nj)};;1000.;1.;0.;0.`] },
];

// The displacements of the grid points on the stringers and at the middles of the bays, at the middle of the panel,
// at five sixths of its length and at its free end.
function displacements(m, load) {
  const held = [...Array.from({ length: m.ni + 1 }, (_, i) => m.node(i, 0)), ...m.held];
  const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", ...m.lines,
    ...held.map((id) => `SinglePointConstrList;1;123456;${id}`), ...load.cards(m)].join("\n") + "\n";
  const run = runModel(text, CLASSIC);
  const values = [];
  for (const fraction of [0.5, 5 / 6, 1])
    for (let half = 0; half <= 2 * PITCHES; ++half)
      values.push(nodal(run, "displacements", m.node((half * m.perPitch) / 2, Math.round(fraction * m.nj)))
        .slice(0, 3));
  return values;
}

// The largest difference of the displacement vectors over the largest displacement of the reference.
function difference(values, reference) {
  const norm = (v) => Math.hypot(...v);
  const scale = Math.max(...reference.map(norm));
  return Math.max(...values.map((v, k) => norm(v.map((c, d) => c - reference[k][d])))) / scale;
}

// The models with line stiffeners, and what each is allowed to differ from the webs of shells by: [bending loads,
// the force along a stringer] (the measured values are in the output of the test).
const MODELS = [
  { name: "ShellQuad4, 2 cells per stringer pitch", perPitch: 2, perBay: 4, triangles: "none",
    thin: [0.03, 0.05], thick: [0.005, 0.04] },
  { name: "ShellQuad4, 4 cells per stringer pitch", perPitch: 4, perBay: 8, triangles: "none",
    thin: [0.012, 0.015], thick: [0.003, 0.025] },
  { name: "ShellQuad4 and ShellTria3, 4 cells per stringer pitch", perPitch: 4, perBay: 8, triangles: "half",
    thin: [0.012, 0.02], thick: [0.003, 0.03] },
  { name: "ShellQuad4 and ShellTria3, 8 cells per stringer pitch", perPitch: 8, perBay: 16, triangles: "half",
    thin: [0.006, 0.006], thick: [0.003, 0.02] },
  { name: "ShellTria3, 8 cells per stringer pitch", perPitch: 8, perBay: 16, triangles: "all",
    thin: [0.006, 0.008], thick: [0.003, 0.025] },
  { name: "ShellQuad4, 4 cells per stringer pitch, Beam2 stringers and Bar2 frames", perPitch: 4, perBay: 8,
    triangles: "none", stringerCard: "Beam2", frameCard: "Bar2", thin: [0.012, 0.015], thick: [0.003, 0.025] },
];

const SKINS = [{ skinName: "thin", thickness: 0.002 }, { skinName: "thick", thickness: 0.02 }];
for (const { skinName, thickness } of SKINS) {
  // The reference: 8 cells per stringer pitch and 16 per frame bay. At half that mesh the webs of shells give the
  // same displacements within 0.7 % (thin skin) and 1.4 % (thick skin, the force along a stringer).
  let reference;
  const referenceOf = () => {
    reference ??= LOADS.map((load) => displacements(withWebs({ perPitch: 8, perBay: 16, thickness }), load));
    return reference;
  };
  for (const model of MODELS)
    test(`a twisted stiffened panel with a ${skinName} skin of ${model.name}: Bar2, Beam2 and Rod2 against webs of ` +
      "shells", (t) => {
      const lines = withLines({ ...model, thickness });
      LOADS.forEach((load, k) => {
        const measured = difference(displacements(lines, load), referenceOf()[k]);
        const bound = model[skinName][k === 2 ? 1 : 0];
        t.diagnostic(`${load.name}: differs by ${(100 * measured).toFixed(2)} % (criterion ${100 * bound} %)`);
        assert.ok(measured <= bound, `${load.name}: differs by ${(100 * measured).toFixed(2)} %, criterion ${
          100 * bound} %`);
      });
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// The axis of a section given by its type and dimensions

// A T section: a flange of 0.05 by 0.004 at its top, a web of 0.003 down to a depth of 0.06. Its centroid is
// 0.015696 under its top, and its inertia in the plane of its web is 1.2634e-7 (the two rectangles). Its shear
// centre is where the web meets the middle of the flange, 0.002 under the top (the thin-walled value; the section
// solved as a solid has it 0.16 mm lower).
const TEE = { dimensions: [0.05, 0.06, 0.004, 0.003], top: 0.002, centroid: 0.015696, inertia: 1.2634e-7 };

// A cantilever of four elements along X whose section has its web along Y, pulled along its axis at its free grid
// point: the rotation of that grid point about Z.
function pulledCantilever(card, property) {
  const [length, count, force] = [1.0, 4, 1000.0];
  const lines = [`IsoMat;1;${real(PANEL.young)};;${real(PANEL.poisson)};0.`,
    `${property};1;1;;TSection;${TEE.dimensions.map(real).join(";")}`, "SinglePointConstrList;1;123456;1",
    `PointForceLoad;2;${count + 1};;${real(force)};1.;0.;0.`];
  for (let k = 0; k <= count; ++k) lines.push(`Node;${k + 1};;${real((length * k) / count)};0.;0.`);
  for (let k = 0; k < count; ++k) lines.push(`${card};${k + 1};1;${k + 1};${k + 2};0.;;1.;0.`);
  const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", ...lines].join("\n") + "\n";
  const rotation = nodal(runModel(text, CLASSIC), "displacements", count + 1)[5];
  return { rotation, expected: (force * (TEE.centroid - TEE.top) * length) / (PANEL.young * TEE.inertia) };
}

test("a Bar2 on a BarShapeProp pulled at its grid point stays straight: its axis is its neutral axis", () => {
  const { rotation, expected } = pulledCantilever("Bar2", "BarShapeProp");
  assert.ok(Math.abs(rotation) <= 1e-9 * expected, `rotation ${rotation}, against ${expected} of an axis at the ` +
    "shear centre");
});

test("a Beam2 on a BeamShapeProp pulled at its grid point bends: its axis is the shear centre of its section", (t) => {
  // The force acts on the shear centre, 13.5 mm from the centroid: the moment F e bends the beam in the plane of
  // its web, θ = F e L / (E I). The thin-walled e is 1.2 % above that of the solid section.
  const { rotation, expected } = pulledCantilever("Beam2", "BeamShapeProp");
  const measured = Math.abs(rotation) / expected;
  t.diagnostic(`rotation over F e L / (E I): ${measured.toFixed(4)} (criterion 0.03)`);
  assert.ok(Math.abs(measured - 1) <= 0.03, `rotation over F e L / (E I): ${measured.toFixed(4)}`);
});
