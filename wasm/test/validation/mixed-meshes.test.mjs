// Validation of the mixed hybrid formulation where its shells, line elements and solids meet
// (docs/mixed-hybrid-formulation-spec.md, section 3.4, ties 1 to 6): a shell on a face of solids, a shell standing on
// solids, a shell continued by solids, a line element along an edge of solids or of shells on solids, a line element
// that ends on a solid or on a shell. For each: no mechanism (six rigid-body modes when free), the uniform states
// exact, and the stiffness against a closed form or a model of solids alone, with the criteria of section 13.6
// (support.mjs). A case that does not meet its criterion is left failing; its comment says what was measured and why.
//
// Run: node --test wasm/test/validation/mixed-meshes.test.mjs

import assert from "node:assert/strict";
import { CRITERIA, real, runModel, nodal, assertRatio, test } from "./support.mjs";

// ---------------------------------------------------------------------------------------------------------------------
// Meshes of solids. A block is a lattice of nx × ny × nz cells mapped by `map(u, v, w)` from the unit cube; each cell
// is one SolidHexa, two SolidPenta (triangles in the u-v plane), six SolidTetra around its diagonal c0-c6, six
// SolidPyramid with their apex at its centre, or a "Transition" cell (pyramids on its two u faces, tetrahedra on the
// others). The splits are conforming from cell to cell: hexahedra, pyramid cells, transition cells and pentahedra meet
// on quadrilateral faces along u (pentahedra along v too); pentahedra, tetrahedra and transition cells meet on
// triangular faces along w (tetrahedra and transition cells along v too).

// The corners of a cell in the order of a SolidHexa card, and the centre of the cell as corner 8.
const CELL_CORNERS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const CELL_SPLITS = {
  SolidHexa: [["SolidHexa", [0, 1, 2, 3, 4, 5, 6, 7]]],
  SolidPenta: [["SolidPenta", [0, 1, 2, 4, 5, 6]], ["SolidPenta", [0, 2, 3, 4, 6, 7]]],
  SolidTetra: [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    .map((corners) => ["SolidTetra", corners]),
  SolidPyramid: [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    .map((base) => ["SolidPyramid", [...base, 8]]),
  Transition: [["SolidPyramid", [1, 5, 6, 2, 8]], ["SolidPyramid", [3, 7, 4, 0, 8]],
    ...[[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [2, 6, 3], [3, 6, 7]]
      .map((face) => ["SolidTetra", [...face, 8]])],
};

// The edges of each card in the order of its mid-side nodes (SolidTetra G5-G10, SolidPyramid G6-G13, SolidPenta
// G7-G15, SolidHexa G9-G20).
const CARD_EDGES = {
  SolidTetra: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
  SolidPyramid: [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
  SolidPenta: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5], [5, 3]],
  SolidHexa: [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]],
};
// The faces of each card, their corners turning about the outward normal of a right-handed element.
const CARD_FACES = {
  SolidTetra: [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
  SolidPyramid: [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
  SolidPenta: [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
  SolidHexa: [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
};
// The same element given from its next corner.
const CARD_ROTATIONS = { SolidTetra: [1, 2, 0, 3], SolidPyramid: [1, 2, 3, 0, 4], SolidPenta: [1, 2, 0, 4, 5, 3],
  SolidHexa: [1, 2, 3, 0, 5, 6, 7, 4] };

const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const near = (a, b, tolerance = 1e-9) => Math.abs(a - b) <= tolerance;

// A number in [−1, 1] that depends only on its four integers (the distortion of a lattice point).
function scatter(i, j, k, c) {
  const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719 + c * 4.581) * 43758.5453;
  return 2 * (s - Math.floor(s)) - 1;
}

// A mesh of solids. `surfaces` are predicates (x, y, z) → boolean of the smooth parts of its skin: an element edge
// whose two grid points lie on two of them is declared sharp. `quadratic` gives every edge its mid-side node, at the
// map of the middle of the edge in the lattice (on the exact curve of a curved map); a function (a, b) → boolean of
// the positions of the two ends chooses the edges that get one. `reversed` numbers the grid points and the elements
// downwards, `rotated` gives every element from its next corner (the invariance checks).
function createMesh({ surfaces = [], quadratic = false, tolerance = 1e-7, reversed = false, rotated = false } = {}) {
  const mesh = { points: new Map(), elements: [], surfaces };
  const byPosition = new Map();
  const key = (p) => p.map((x) => Math.round(x / tolerance)).join(",");
  let nextNode = reversed ? 90000 : 1;
  let nextElement = reversed ? 90000 : 1;

  // The grid point at `p`, created when it does not exist yet.
  mesh.point = (p) => {
    const k = key(p);
    if (!byPosition.has(k)) {
      byPosition.set(k, nextNode);
      mesh.points.set(nextNode, p);
      nextNode += reversed ? -1 : 1;
    }
    return byPosition.get(k);
  };
  // The existing grid point at (x, y, z).
  mesh.at = (x, y, z) => {
    const id = byPosition.get(key([x, y, z]));
    assert.ok(id, `a grid point at ${x}, ${y}, ${z}`);
    return id;
  };
  // The grid points whose position satisfies `predicate(x, y, z)`.
  mesh.where = (predicate) => [...mesh.points].filter(([, p]) => predicate(...p)).map(([id]) => id);

  // A block of n = [nx, ny, nz] cells: `shape` a cell kind or a function (i, j, k) → cell kind; `distortion` moves
  // every lattice point by up to that share of a cell (one share, or one per direction), along the directions in which
  // it is not on the boundary of the block (`distortsSkin` false: only the points inside the block move).
  mesh.addBlock = ({ n, map = (u, v, w) => [u, v, w], shape = "SolidHexa", distortion = 0, distortsSkin = true,
    propId = 1 }) => {
    const amount = Array.isArray(distortion) ? distortion : [distortion, distortion, distortion];
    const lattice = (i, j, k) => {
      const index = [i, j, k];
      const inside = index.map((value, c) => value > 0 && value < n[c]);
      const moves = distortsSkin || inside.every(Boolean);
      return index.map((value, c) => (value + (moves && inside[c] ? amount[c] * scatter(i, j, k, c) : 0)) / n[c]);
    };
    for (let k = 0; k < n[2]; ++k)
      for (let j = 0; j < n[1]; ++j)
        for (let i = 0; i < n[0]; ++i) {
          const corners = CELL_CORNERS.map(([a, b, c]) => lattice(i + a, j + b, k + c));
          corners.push([0, 1, 2].map((c) => corners.reduce((sum, p) => sum + p[c], 0) / 8));
          const kind = typeof shape === "function" ? shape(i, j, k) : shape;
          for (const [card, local] of CELL_SPLITS[kind]) {
            let params = local.map((l) => corners[l]);
            let x = params.map((p) => map(...p));
            // Tetrahedra and pyramids are given right-handed (the base turning towards the apex).
            const last = x.length - 1;
            const isCone = card === "SolidTetra" || card === "SolidPyramid";
            if (isCone && dot(cross(subtract(x[1], x[0]), subtract(x[last - 1], x[0])), subtract(x[last], x[0])) < 0) {
              const swap = card === "SolidTetra" ? [1, 0, 2, 3] : [0, 3, 2, 1, 4];
              params = swap.map((s) => params[s]);
              x = swap.map((s) => x[s]);
            }
            if (rotated) {
              params = CARD_ROTATIONS[card].map((s) => params[s]);
              x = CARD_ROTATIONS[card].map((s) => x[s]);
            }
            const element = { id: nextElement, card, propId, corners: x.map((p) => mesh.point(p)), mids: [] };
            nextElement += reversed ? -1 : 1;
            if (quadratic)
              for (const [a, b] of CARD_EDGES[card]) {
                const wanted = quadratic === true || quadratic(x[a], x[b]);
                const middle = map(...[0, 1, 2].map((c) => (params[a][c] + params[b][c]) / 2));
                element.mids.push(wanted ? mesh.point(middle) : 0);
              }
            mesh.elements.push(element);
          }
        }
    return mesh;
  };

  // One element of card `card` at the corner positions `x`.
  mesh.addElement = (card, x, propId = 1) => {
    const element = { id: nextElement, card, propId, corners: x.map((p) => mesh.point(p)), mids: [] };
    nextElement += reversed ? -1 : 1;
    if (quadratic)
      for (const [a, b] of CARD_EDGES[card])
        element.mids.push(mesh.point([0, 1, 2].map((c) => (x[a][c] + x[b][c]) / 2)));
    mesh.elements.push(element);
    return mesh;
  };

  // The Node and element records.
  mesh.records = () => [
    ...[...mesh.points].map(([id, p]) => `Node;${id};;${p.map(real).join(";")}`),
    ...mesh.elements.map((e) => `${e.card};${e.id};${e.propId};${[...e.corners, ...e.mids].join(";")}`)];
  // The cards of the elements.
  mesh.cards = () => [...new Set(mesh.elements.map((e) => e.card))];
  // The grid points that are corners of elements (the others are mid-side nodes).
  mesh.cornerIds = () => new Set(mesh.elements.flatMap((e) => e.corners));

  // The element edges (corner pairs, each once) for which `predicate(pa, pb)` holds.
  mesh.edges = (predicate = () => true) => {
    const seen = new Set();
    const edges = [];
    for (const e of mesh.elements)
      for (const [a, b] of CARD_EDGES[e.card]) {
        const pair = [e.corners[a], e.corners[b]].sort((p, q) => p - q);
        if (seen.has(pair.join("-"))) continue;
        seen.add(pair.join("-"));
        if (predicate(mesh.points.get(pair[0]), mesh.points.get(pair[1]))) edges.push(pair);
      }
    return edges;
  };
  // The sharp edges of the skin: those whose two grid points lie on the same two surfaces.
  mesh.sharpEdges = () => mesh.edges((a, b) =>
    surfaces.filter((surface) => surface(...a) && surface(...b)).length >= 2);
  // The options of modelFromReader of the mixed hybrid formulation with these declarations.
  mesh.options = () => ({ elementFormulation: "MixedHybridFormulation",
    formulationGeometry: { sharpEdges: mesh.sharpEdges() } });
  return mesh;
}

// The six faces of the box [0, a] × [0, b] × [0, c] as surfaces.
function boxSurfaces(a, b, c) {
  return [(x) => near(x, 0), (x) => near(x, a), (x, y) => near(y, 0), (x, y) => near(y, b),
    (x, y, z) => near(z, 0), (x, y, z) => near(z, c)];
}

// The faces of the skin of `mesh` (those one element only has): { element, card, nodes, x, centre, normal, area }
// with `nodes` turning about the outward `normal` (unit, of the plane of the corners) and `area` that of the corners.
function skinFaces(mesh) {
  const count = new Map();
  const faces = [];
  for (const element of mesh.elements)
    for (const local of CARD_FACES[element.card]) {
      const nodes = local.map((l) => element.corners[l]);
      const key = [...nodes].sort((a, b) => a - b).join("-");
      count.set(key, (count.get(key) ?? 0) + 1);
      faces.push({ key, element, card: element.card, nodes });
    }
  return faces.filter((face) => count.get(face.key) === 1).map((face) => {
    const x = face.nodes.map((id) => mesh.points.get(id));
    const centre = [0, 1, 2].map((c) => x.reduce((sum, p) => sum + p[c], 0) / x.length);
    // The vector area of a triangle, or of a quadrilateral: half the cross product of its diagonals.
    const vector = x.length === 3 ? cross(subtract(x[1], x[0]), subtract(x[2], x[0])).map((v) => v / 2)
      : cross(subtract(x[2], x[0]), subtract(x[3], x[1])).map((v) => v / 2);
    const area = Math.hypot(...vector);
    // The corners turn about the outward normal whatever the corner the element is given from.
    const inside = [0, 1, 2].map((c) =>
      face.element.corners.reduce((sum, id) => sum + mesh.points.get(id)[c], 0) / face.element.corners.length);
    const sign = dot(vector, subtract(centre, inside)) < 0 ? -1 : 1;
    if (sign < 0) { face.nodes.reverse(); x.reverse(); }
    return { ...face, x, centre, normal: vector.map((v) => (sign * v) / area), area };
  });
}

// The FacePressureLoad record of set `sid` on `face`: the pressures `p` (one, or one per corner in the order of
// face.nodes) pressing on the element, or, with `direction`, the traction p × direction. G1 is the first corner of the
// face and G3 the one across it (a quadrilateral), G4 the corner off the face (a tetrahedron), nothing (a triangle of
// a pentahedron). A triangle of a pyramid is named by the second corner of its base edge and the apex: the reader
// takes the first face of the element that holds G1 and G3, and this pair finds three of the four triangles (the
// fourth cannot be named: every pair of its corners belongs to a face listed before it).
function facePressure(sid, face, p, direction = null) {
  const values = Array.isArray(p) ? p : [p];
  const pressures = [0, 1, 2, 3].map((k) => (k < values.length ? real(values[k]) : "")).join(";");
  let first = face.nodes[0];
  let across = "";
  if (face.nodes.length === 4) across = face.nodes[2];
  else if (face.card === "SolidTetra") across = face.element.corners.find((id) => !face.nodes.includes(id));
  else if (face.card === "SolidPyramid") {
    const corners = face.element.corners;
    const base = face.nodes.map((id) => corners.indexOf(id)).filter((local) => local < 4).sort((a, b) => a - b);
    first = corners[base[0] === 0 && base[1] === 3 ? 0 : base[1]];
    across = corners[4];
  }
  const tail = direction ? `;0;${direction.map(real).join(";")}` : "";
  return `FacePressureLoad;${sid};${face.element.id};${pressures};${first};${across}${tail}`;
}

// ---------------------------------------------------------------------------------------------------------------------
// Decks and results.

// A deck: the solution number, the case records and the model records.
function deck(solution, caseRecords, modelRecords) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseRecords, "[model]", ...modelRecords, ""]
    .join("\n");
}

// The rows of an element table as objects of its columns, with their element and grid point (0 for the centre).
function rowsOf(run, elementType, kind = "stress", subcase = 1) {
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
const sixOf = (row) => [row.oxx, row.oyy, row.ozz, row.txy, row.tyz, row.txz];

// The sum of the forces (first three) and of their moments about the origin (last three) of a nodal table.
function resultant(run, mesh, kind, subcase = 1) {
  const total = [0, 0, 0, 0, 0, 0];
  for (const [id, p] of mesh.points) {
    const f = nodal(run, kind, id, subcase);
    const m = cross(p, f);
    for (let c = 0; c < 3; ++c) { total[c] += f[c]; total[3 + c] += m[c] + f[3 + c]; }
  }
  return total;
}

// The largest differences between the stresses of the solids of `mesh` and `exact(x, y, z)` (six components): at the
// corners of the elements, at their centres (`centres`: the elements have straight edges, the centre of a tetrahedron,
// a pentahedron or a hexahedron is then the mean of its corners) and in the table of the grid points.
function stressErrors(run, mesh, exact, { subcase = 1, centres = false } = {}) {
  const errors = { corner: 0, centre: 0, grid: 0 };
  const worstOf = (row, p) => {
    const e = exact(...p);
    return Math.max(...sixOf(row).map((v, c) => Math.abs(v - e[c])));
  };
  for (const card of mesh.cards())
    for (const row of rowsOf(run, card, "stress", subcase)) {
      if (row.nodeId !== 0) errors.corner = Math.max(errors.corner, worstOf(row, mesh.points.get(row.nodeId)));
      else if (centres && card !== "SolidPyramid") {
        const corners = mesh.elements.find((e) => e.id === row.id).corners.map((id) => mesh.points.get(id));
        const centre = [0, 1, 2].map((c) => corners.reduce((sum, p) => sum + p[c], 0) / corners.length);
        errors.centre = Math.max(errors.centre, worstOf(row, centre));
      }
    }
  for (const row of rowsOf(run, "SolidGridPoints", "stress", subcase))
    errors.grid = Math.max(errors.grid, worstOf(row, mesh.points.get(row.id)));
  return errors;
}

// The run of the mixed hybrid formulation on `mesh` with the model records `records`; a warning fails the case
// unless its code is in `allowed`.
function runMixed(mesh, caseRecords, records, { solution = 101, analysis = "statics", allowed = [] } = {}) {
  const run = runModel(deck(solution, caseRecords, [...mesh.records(), ...records]), mesh.options(), analysis);
  const warnings = run.diagnostics.filter((d) => d.severity === "warning" && !allowed.includes(d.code));
  assert.equal(warnings.length, 0, `warnings: ${JSON.stringify(warnings.map((d) => d.text))}`);
  return run;
}

// error ≤ tolerance × scale, with the measured ratio in the message.
function assertSmall(error, scale, tolerance, what) {
  assert.ok(error <= tolerance * scale, `${what}: error ${(error / scale).toExponential(2)} of the scale ` +
    `${scale.toExponential(3)}, tolerance ${tolerance}`);
}


// ---------------------------------------------------------------------------------------------------------------------
// The models: a block of solids (the core) of sides a, b, c along x, y, z, shells of another modulus and line
// elements on it. The solids and the shells have the same Poisson ratio: in a uniform strain each then takes its own
// uniaxial stress and they exchange no force.

const CORE = { a: 2.0, b: 1.0, c: 1.0, young: 2.0e11, poisson: 0.3, density: 7800.0 };
const SKIN = { thickness: 0.02, young: 0.7e11, poisson: 0.3, density: 2700.0 };
const LINE = { area: 0.01, i1: 2.0e-5, i2: 3.0e-5, j: 4.0e-5, young: 1.1e11, poisson: 0.3, density: 4500.0 };
const coreMap = (u, v, w) => [CORE.a * u, CORE.b * v, CORE.c * w];

// The core in n cells of `shape`, distorted inside (its skin keeps regular cells: the shells and the lines lie on
// its lattice).
function coreMesh({ shape = "SolidHexa", n = [2, 2, 2], quadratic = false, distortion = 0.2 } = {}) {
  return createMesh({ surfaces: boxSurfaces(CORE.a, CORE.b, CORE.c), quadratic })
    .addBlock({ n, map: coreMap, shape, distortion, distortsSkin: false });
}
// The properties and materials: solids (1), shells (2), bars (3).
function sections() {
  return ["SolidProp;1;1", `IsoMat;1;${real(CORE.young)};;${real(CORE.poisson)};${real(CORE.density)}`,
    `ShellProp;2;2;${real(SKIN.thickness)};2;;2`,
    `IsoMat;2;${real(SKIN.young)};;${real(SKIN.poisson)};${real(SKIN.density)}`,
    `BarProp;3;3;${real(LINE.area)};${real(LINE.i1)};${real(LINE.i2)};${real(LINE.j)}`,
    `IsoMat;3;${real(LINE.young)};;${real(LINE.poisson)};${real(LINE.density)}`];
}
// The case records: one subcase per [constraint set, load set or null], the corner rows of the shells asked.
function caseOf(subcases) {
  const records = ["DisplacementOutput;All", "SinglePointConstrForceOutput;All", "StressOutput;All;CornerResults",
    "ElemForceOutput;All;CornerResults"];
  subcases.forEach(([constraints, load], k) => records.push(`SubcaseStart;${k + 1}`,
    `SinglePointConstrSelection;${constraints}`, ...(load === null ? [] : [`StaticLoadSelection;${load}`])));
  return records;
}

// The shells of the top face z = c of the core, one cell of the lattice at a time, their normal along +z and their
// first side along ±x: "ShellQuad4", "ShellTria3 along the edges of the solids" (the diagonal of the top faces of the
// pentahedra and tetrahedra) or "ShellTria3 across" (the other diagonal). `offset` moves their section along +z.
function topShells(mesh, n, kind, offset = 0) {
  const at = (i, j) => mesh.at((CORE.a * i) / n[0], (CORE.b * j) / n[1], CORE.c);
  const records = [];
  let id = 1000;
  for (let j = 0; j < n[1]; ++j)
    for (let i = 0; i < n[0]; ++i) {
      const q = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
      const triangles = kind === "ShellTria3 across" ? [[q[0], q[1], q[3]], [q[2], q[3], q[1]]]
        : [[q[0], q[1], q[2]], [q[2], q[3], q[0]]];
      if (kind === "ShellQuad4") records.push(`ShellQuad4;${++id};2;${q.join(";")};;;${real(offset)}`);
      else for (const t of triangles) records.push(`ShellTria3;${++id};2;${t.join(";")};;;${real(offset)}`);
    }
  return records;
}

// The two uniform states of the core with what lies on it, prescribed on its two end faces (constraint sets 1 and 2):
// - a stretch ε along x: u = ε x, v = −ν ε y, w = −ν ε z;
// - a bending of curvature κ about y: u = −κ x z', v = ν κ y' z', w = κ (x² + ν (z'² − y'²)) / 2, with y' and z'
//   from the middle of the core: the strain of a fibre is −κ z' whatever its modulus.
// u is prescribed on the end faces, v and w at two grid points that no edge joins, and the rotation about y
// (−∂w/∂x = −κ x) at the grid points of the ends that have one (shells and lines): the moments of a shell or of a line
// in bending come in through its ends.
const STRETCH = 1.0e-3;
const CURVATURE = 1.0e-3;
const stretchField = (x, y, z) => [STRETCH * x, -CORE.poisson * STRETCH * y, -CORE.poisson * STRETCH * z];
function bendingField(x, y, z) {
  const yc = y - CORE.b / 2;
  const zc = z - CORE.c / 2;
  return [-CURVATURE * x * zc, CORE.poisson * CURVATURE * yc * zc,
    (CURVATURE * (x * x + CORE.poisson * (zc * zc - yc * yc))) / 2];
}
function endSupports(mesh, turning = []) {
  const records = [];
  [stretchField, bendingField].forEach((field, k) => {
    const set = k + 1;
    for (const id of mesh.where((x) => near(x, 0) || near(x, CORE.a)))
      records.push(`SinglePointConstr;${set};${id};1;${real(field(...mesh.points.get(id))[0])}`);
    const first = mesh.at(0, 0, 0);
    const second = mesh.at(0, CORE.b, 0);
    records.push(`SinglePointConstr;${set};${first};2;${real(field(0, 0, 0)[1])}`,
      `SinglePointConstr;${set};${first};3;${real(field(0, 0, 0)[2])}`,
      `SinglePointConstr;${set};${second};3;${real(field(0, CORE.b, 0)[2])}`);
    for (const id of turning)
      records.push(`SinglePointConstr;${set};${id};5;${real(k === 0 ? 0 : -CURVATURE * mesh.points.get(id)[0])}`);
  });
  return records;
}

// The largest difference between the translations of the grid points and `field`.
function translationError(run, mesh, field, subcase) {
  let worst = 0;
  for (const [id, p] of mesh.points)
    field(...p).forEach((value, c) => {
      worst = Math.max(worst, Math.abs(nodal(run, "displacements", id, subcase)[c] - value));
    });
  return worst;
}
// The force along x and the moment about y (taken at the middle of the core) of the supports of the face x = a.
function endResultant(run, mesh, subcase) {
  let force = 0;
  let moment = 0;
  for (const id of mesh.where((x) => near(x, CORE.a))) {
    const f = nodal(run, "singlePointConstrForces", id, subcase);
    force += f[0];
    moment += (mesh.points.get(id)[2] - CORE.c / 2) * f[0] + f[4];
  }
  return { force, moment };
}

// ---------------------------------------------------------------------------------------------------------------------
// A shell on a face of solids (ties 1 and 2): quadrilaterals on the quadrilateral faces of hexahedra and pyramids,
// triangles on the triangular faces of pentahedra and tetrahedra, quadrilaterals over two triangular faces (their
// diagonal is an edge of the solids), and two triangles on a quadrilateral face (tied by their grid points and their
// edges only: spec section 3.6). In the stretch and in the bending the shell takes E' ε at its fibres: its force
// N = E' t ε, its moment E' t³ κ / 12 about its own middle; the ends carry the sum of the core and of the skin.
const SKIN_CASES = [
  { shape: "SolidHexa", kind: "ShellQuad4" },
  { shape: "SolidHexa", kind: "ShellQuad4", offset: SKIN.thickness / 2 },
  { shape: "SolidHexa", kind: "ShellQuad4", quadratic: true },
  { shape: "SolidHexa", kind: "ShellTria3 across" },
  { shape: "SolidPyramid", kind: "ShellQuad4" },
  { shape: "SolidPenta", kind: "ShellTria3 along the edges of the solids" },
  { shape: "SolidPenta", kind: "ShellQuad4" },
  { shape: "SolidTetra", kind: "ShellTria3 along the edges of the solids" },
  { shape: "SolidTetra", kind: "ShellTria3 along the edges of the solids", offset: SKIN.thickness / 2 },
  { shape: "SolidTetra", kind: "ShellQuad4" },
];
for (const { shape, kind, offset = 0, quadratic = false } of SKIN_CASES)
  test(`skin: ${kind}${offset ? " with an offset" : ""} on the top face of a block of ${shape}` +
    `${quadratic ? " with mid-side nodes" : ""}, stretched and bent by its ends`, (t) => {
    const n = [2, 2, 2];
    const mesh = coreMesh({ shape, n, quadratic });
    const top = mesh.where((x, y, z) => (near(x, 0) || near(x, CORE.a)) && near(z, CORE.c))
      .filter((id) => mesh.cornerIds().has(id));
    const records = [...sections(), ...topShells(mesh, n, kind, offset), ...endSupports(mesh, top)];
    // The rotations held at the ends act through the shell edges of the ends (the note of the run says so).
    const run = runMixed(mesh, caseOf([[1, null], [2, null]]), records, { allowed: ["support-note"] });
    const card = kind.split(" ")[0];
    const { thickness } = SKIN;
    const lines = [];
    const checks = [];
    // The strain of the fibre at the height z above the middle of the core, in each state.
    const states = [{ name: "stretch", field: stretchField, strain: () => STRETCH, curvature: 0, scale: STRETCH },
      { name: "bending", field: bendingField, strain: (zc) => -CURVATURE * zc, curvature: CURVATURE,
        scale: CURVATURE * CORE.c }];
    states.forEach(({ name, field, strain, curvature, scale }, k) => {
      const subcase = k + 1;
      const moved = translationError(run, mesh, field, subcase) / (scale * CORE.a);
      const solid = stressErrors(run, mesh, (x, y, z) => [CORE.young * strain(z - CORE.c / 2), 0, 0, 0, 0, 0],
        { subcase });
      const solidError = Math.max(solid.corner, solid.grid) / (CORE.young * scale);
      // The fibres of the shell: layer 1 below its middle, layer 2 above, the middle at c/2 + offset from the core's.
      let shellError = 0;
      for (const row of rowsOf(run, card, "stress", subcase)) {
        const zc = CORE.c / 2 + offset + (row.layer === 1 ? -1 : 1) * (thickness / 2);
        shellError = Math.max(shellError, Math.abs(row.oxx - SKIN.young * strain(zc)), Math.abs(row.oyy),
          Math.abs(row.txy));
      }
      let forceError = 0;
      const membrane = SKIN.young * thickness * strain(CORE.c / 2 + offset);
      const plate = (SKIN.young * thickness ** 3 * curvature) / 12;
      for (const row of rowsOf(run, card, "forces", subcase))
        forceError = Math.max(forceError, Math.abs(row.nx - membrane), Math.abs(row.ny), Math.abs(row.nxy),
          Math.abs(row.mx - plate) / thickness, Math.abs(row.my) / thickness, Math.abs(row.mxy) / thickness);
      // The ends: E ε b c + E' ε t b, and −κ (E b c³ / 12 + E' t b ((c/2 + offset)² + t² / 12)).
      const end = endResultant(run, mesh, subcase);
      const skinArea = thickness * CORE.b;
      const expected = k === 0 ? STRETCH * (CORE.young * CORE.b * CORE.c + SKIN.young * skinArea)
        : -CURVATURE * ((CORE.young * CORE.b * CORE.c ** 3) / 12 +
          SKIN.young * skinArea * ((CORE.c / 2 + offset) ** 2 + thickness ** 2 / 12));
      const endRatio = (k === 0 ? end.force : end.moment) / expected;
      lines.push(`${name}: displacement ${moved.toExponential(1)}, solid stress ${solidError.toExponential(1)}, ` +
        `shell stress ${(shellError / (SKIN.young * scale)).toExponential(1)}, shell forces ` +
        `${(forceError / (SKIN.young * thickness * scale)).toExponential(1)}, end ratio ${endRatio.toFixed(9)}`);
      checks.push([moved, `${name}: displacements`], [solidError, `${name}: stresses of the solids`],
        [shellError / (SKIN.young * scale), `${name}: stresses of the shells`],
        [forceError / (SKIN.young * thickness * scale), `${name}: forces of the shells`],
        [Math.abs(endRatio - 1), `${name}: force or moment of the ends`]);
    });
    for (const line of lines) t.diagnostic(line);
    for (const [error, what] of checks) assertSmall(error, 1, CRITERIA.exact, what);
  });

// Free vibrations of the core with what lies on it: `count` modes of zero frequency (|λ| at most 1e-8 of the first
// elastic one) and the next one elastic, of the order of the following (no mechanism).
function freeModes(mesh, records, count = 9) {
  const run = runMixed(mesh, ["SubcaseStart;1", "EigenMethodSelection;3"],
    [...sections(), ...records, `LanczosEigenMethod;3;;;${count}`], { solution: 103, analysis: "modes" });
  return run.eigenvalues[0].rows.map((row) => row.eigenvalue);
}
function assertRigidModes(t, values, rigid = 6) {
  assert.ok(values.length > rigid + 1, `${values.length} eigenvalues`);
  const worst = Math.max(...values.slice(0, rigid).map(Math.abs)) / values[rigid];
  const hertz = values.slice(rigid, rigid + 2).map((v) => (Math.sqrt(Math.abs(v)) / (2 * Math.PI)).toFixed(1));
  t.diagnostic(`${rigid} rigid eigenvalues at most ${worst.toExponential(2)} of the next one; then ` +
    `${hertz.join(", ")} Hz`);
  assert.ok(worst <= CRITERIA.invariant, `${rigid} rigid modes: ${worst}`);
  assert.ok(values[rigid] > 1e-3 * values[rigid + 1],
    `mode ${rigid + 1} is a mechanism: ${values[rigid]} against ${values[rigid + 1]}`);
}
for (const { shape, kind } of [SKIN_CASES[0], SKIN_CASES[3], SKIN_CASES[6], SKIN_CASES[7]])
  test(`skin: a free block of ${shape} with ${kind} on its top face has six rigid-body modes`, (t) => {
    const mesh = coreMesh({ shape });
    assertRigidModes(t, freeModes(mesh, topShells(mesh, [2, 2, 2], kind)));
  });

// ---------------------------------------------------------------------------------------------------------------------
// A sandwich: a soft core of solids between two stiff faces, as shells on its two faces (their sections offset
// outwards by half their thickness, so that they lie on the core) or as two layers of solids. A cantilever of length
// 10 under a shear force on the end of its core, with ν = 0. The two models are the same structure: their tip
// deflections agree to the criterion of the displacements. Sandwich beam theory (Allen, Analysis and Design of
// Structural Sandwich Panels, 1969: w = P L³ / (3 D) + P L / S, D = E' b t d² / 2 + E' b t³ / 6 + E b c³ / 12,
// S = G b d² / c, d = c + t) is given for information: it leaves out the bending of the faces about their own
// middles near the root.
const SANDWICH = { length: 10.0, width: 1.0, core: 1.0, face: 0.05, coreYoung: 2.0e9, faceYoung: 2.0e11 };
function sandwichTip(faces) {
  const { length, width, core, face, coreYoung, faceYoung } = SANDWICH;
  const force = 1.0e4;
  const cells = [8, 1, 2];
  const layers = faces === "solids" ? [[-face, 0, 1, 2], [0, core, 2, 1], [core, core + face, 1, 2]]
    : [[0, core, 2, 1]];
  const [bottom, top] = [layers[0][0], layers[layers.length - 1][1]];
  const mesh = createMesh({ surfaces: [(x) => near(x, 0), (x) => near(x, length), (x, y) => near(y, 0),
    (x, y) => near(y, width), (x, y, z) => near(z, bottom), (x, y, z) => near(z, top)] });
  for (const [from, to, count, propId] of layers)
    mesh.addBlock({ n: [cells[0], cells[1], count], propId,
      map: (u, v, w) => [length * u, width * v, from + (to - from) * w] });
  const records = ["SolidProp;1;1", `IsoMat;1;${real(coreYoung)};;0.;0.`, `IsoMat;2;${real(faceYoung)};;0.;0.`,
    faces === "solids" ? "SolidProp;2;2" : `ShellProp;2;2;${real(face)};2;;2`];
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
  if (faces === "shells") {
    let id = 1000;
    for (let i = 0; i < cells[0]; ++i) {
      const at = (k, y, z) => mesh.at((length * k) / cells[0], y, z);
      // The top face turning about +z, the bottom face about −z: each offset along its own normal, outwards.
      const upper = [at(i, 0, core), at(i + 1, 0, core), at(i + 1, width, core), at(i, width, core)];
      const lower = [at(i, 0, 0), at(i, width, 0), at(i + 1, width, 0), at(i + 1, 0, 0)];
      for (const q of [upper, lower]) records.push(`ShellQuad4;${++id};2;${q.join(";")};;;${real(face / 2)}`);
    }
    // The faces are clamped too: the rotation about y of their grid points at the root.
    for (const id of mesh.where((x, y, z) => near(x, 0) && (near(z, 0) || near(z, core))))
      records.push(`SinglePointConstr;1;${id};5;0.`);
  }
  const tipFaces = skinFaces(mesh).filter((f) => f.x.every((p) => near(p[0], length) && p[2] > -1e-9 &&
    p[2] < core + 1e-9));
  for (const f of tipFaces) records.push(facePressure(2, f, force / (width * core), [0, 0, 1]));
  const run = runMixed(mesh, caseOf([[1, 2]]), records, { allowed: ["support-note"] });
  const tip = mesh.where((x, y, z) => near(x, length) && z > -1e-9 && z < core + 1e-9);
  const deflection = tip.reduce((sum, id) => sum + nodal(run, "displacements", id)[2], 0) / tip.length;
  const d = core + face;
  const bending = (faceYoung * width * face * d * d) / 2 + (faceYoung * width * face ** 3) / 6 +
    (coreYoung * width * core ** 3) / 12;
  const shear = ((coreYoung / 2) * width * d * d) / core;
  return { deflection, theory: (force * length ** 3) / (3 * bending) + (force * length) / shear };
}
test("sandwich: a cantilever of SolidHexa with ShellQuad4 on its two faces bends as with faces of solids", (t) => {
  const shells = sandwichTip("shells");
  const solids = sandwichTip("solids");
  t.diagnostic(`tip deflection: shells ${shells.deflection.toExponential(5)}, solids ` +
    `${solids.deflection.toExponential(5)}, ratio ${(shells.deflection / solids.deflection).toFixed(5)}; sandwich ` +
    `beam theory ${shells.theory.toExponential(5)} (ratio of the shells to it ` +
    `${(shells.deflection / shells.theory).toFixed(5)})`);
  assertRatio(shells.deflection, solids.deflection, CRITERIA.displacement, "tip deflection");
});

// ---------------------------------------------------------------------------------------------------------------------
// A fin: a shell standing on the top face of the core, along the line y = b / 2, of height H (tie 1 along its foot,
// tie 4 for its rotation about its foot).
const FIN = { height: 1.0, rows: 2 };
// The grid points and ShellQuad4 of the fin on `mesh` (the core in n cells); their normal is −y.
function finRecords(mesh, n, thicknessOffset = 0) {
  const at = (i, k) => mesh.point([(CORE.a * i) / n[0], CORE.b / 2, CORE.c + (FIN.height * k) / FIN.rows]);
  const records = [];
  let id = 2000;
  for (let k = 0; k < FIN.rows; ++k)
    for (let i = 0; i < n[0]; ++i)
      records.push(`ShellQuad4;${++id};2;${at(i, k)};${at(i + 1, k)};${at(i + 1, k + 1)};${at(i, k + 1)};;;` +
        `${real(thicknessOffset)}`);
  return records;
}

// The core and its fin stretched and bent by their ends: the fin takes E' ε in its plane (in the bending its strain
// −κ z' grows with its height); the ends carry E ε b c + E' ε t H and
// −κ (E b c³ / 12 + E' t ((c / 2 + H)³ − (c / 2)³) / 3).
for (const shape of ["SolidHexa", "SolidTetra", "SolidPyramid"])
  test(`fin: a ShellQuad4 fin standing on a block of ${shape}, stretched and bent by their ends`, (t) => {
    const n = [2, 2, 2];
    const mesh = coreMesh({ shape, n });
    const fin = finRecords(mesh, n);
    const records = [...sections(), ...fin, ...endSupports(mesh)];
    const run = runMixed(mesh, caseOf([[1, null], [2, null]]), records);
    const lines = [];
    const checks = [];
    const states = [{ name: "stretch", field: stretchField, strain: () => STRETCH, scale: STRETCH },
      { name: "bending", field: bendingField, strain: (zc) => -CURVATURE * zc, scale: CURVATURE * CORE.c }];
    states.forEach(({ name, field, strain, scale }, k) => {
      const subcase = k + 1;
      const moved = translationError(run, mesh, field, subcase) / (scale * CORE.a);
      // The grid points of the fin are not grid points of solids: the table of the grid point stresses leaves them.
      let solidError = 0;
      for (const card of mesh.cards())
        for (const row of rowsOf(run, card, "stress", subcase)) {
          if (row.nodeId === 0) continue;
          const zc = mesh.points.get(row.nodeId)[2] - CORE.c / 2;
          solidError = Math.max(solidError, ...sixOf(row).map((v, c) =>
            Math.abs(v - (c === 0 ? CORE.young * strain(zc) : 0))));
        }
      let shellError = 0;
      for (const row of rowsOf(run, "ShellQuad4", "stress", subcase)) {
        if (row.nodeId === 0) continue;
        const zc = mesh.points.get(row.nodeId)[2] - CORE.c / 2;
        shellError = Math.max(shellError, Math.abs(row.oxx - SKIN.young * strain(zc)), Math.abs(row.oyy),
          Math.abs(row.txy));
      }
      const end = endResultant(run, mesh, subcase);
      const top = CORE.c / 2 + FIN.height;
      const expected = k === 0 ? STRETCH * (CORE.young * CORE.b * CORE.c + SKIN.young * SKIN.thickness * FIN.height)
        : -CURVATURE * ((CORE.young * CORE.b * CORE.c ** 3) / 12 +
          (SKIN.young * SKIN.thickness * (top ** 3 - (CORE.c / 2) ** 3)) / 3);
      const endRatio = (k === 0 ? end.force : end.moment) / expected;
      lines.push(`${name}: displacement ${moved.toExponential(1)}, solid stress ` +
        `${(solidError / (CORE.young * scale)).toExponential(1)}, shell stress ` +
        `${(shellError / (SKIN.young * scale)).toExponential(1)}, end ratio ${endRatio.toFixed(9)}`);
      checks.push([moved, `${name}: displacements`], [solidError / (CORE.young * scale), `${name}: solids`],
        [shellError / (SKIN.young * scale), `${name}: shells`], [Math.abs(endRatio - 1), `${name}: ends`]);
    });
    for (const line of lines) t.diagnostic(line);
    for (const [error, what] of checks) assertSmall(error, 1, CRITERIA.exact, what);
  });

for (const shape of ["SolidHexa", "SolidTetra", "SolidPenta"])
  test(`fin: a free block of ${shape} with a ShellQuad4 fin standing on it has six rigid-body modes`, (t) => {
    const mesh = coreMesh({ shape });
    assertRigidModes(t, freeModes(mesh, finRecords(mesh, [2, 2, 2])));
  });

// A soft fin on a stiff block (its modulus 1e6 times that of the fin, clamped under), pushed across by a uniform
// pressure, with ν = 0: the fin is a cantilever plate clamped by the block (its rotation about its foot follows the
// solids): its top moves by p H⁴ / (8 D) + p H² / (2 κ G t), D = E' t³ / 12, κ = 5/6 (Timoshenko). On a block of
// the modulus of the fin the foot turns, and by more as the block is refined (the couple of the fin acts on a line of
// the face): 1.0015 times this deflection on 2 × 2 × 2 hexahedra, 1.005 on 2 × 4 × 4, 1.020 on 2 × 8 × 8, 1.037 on
// 2 × 16 × 8 (measured, not a case of this file).
for (const shape of ["SolidHexa", "SolidTetra"])
  test(`fin: a soft ShellQuad4 fin on a stiff block of ${shape} pushed across bends as a cantilever`, (t) => {
    const n = [2, 2, 2];
    const mesh = coreMesh({ shape, n });
    const pressure = 1000.0;
    const { thickness, young } = SKIN;
    const fin = finRecords(mesh, n);
    const records = ["SolidProp;1;1", `IsoMat;1;${real(young * 1e6)};;0.;0.`, `ShellProp;2;2;${real(thickness)};2;;2`,
      `IsoMat;2;${real(young)};;0.;0.`, ...fin];
    for (const id of mesh.where((x, y, z) => near(z, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
    for (const record of fin) records.push(`FacePressureLoad;2;${record.split(";")[1]};${real(pressure)}`);
    const run = runMixed(mesh, caseOf([[1, 2]]), records);
    const stiffness = (young * thickness ** 3) / 12;
    const expected = (pressure * FIN.height ** 4) / (8 * stiffness) +
      (pressure * FIN.height ** 2) / (2 * (5 / 6) * (young / 2) * thickness);
    const top = mesh.where((x, y, z) => near(z, CORE.c + FIN.height));
    const ratios = top.map((id) => Math.abs(nodal(run, "displacements", id)[1]) / expected);
    t.diagnostic(`top deflection ratios ${ratios.map((r) => r.toFixed(6)).join(" ")}`);
    for (const ratio of ratios) assertRatio(ratio, 1, CRITERIA.displacement, "deflection of the top of the fin");
  });

// ---------------------------------------------------------------------------------------------------------------------
// Line elements on solids.

// The Bar2 along the top edge y = 0, z = c of the core (n cells), their plane 1 the plane x-y: the bending of the
// core about y bends them in their plane 2 (inertia I2).
function edgeBars(mesh, n) {
  const records = [];
  for (let i = 0; i < n[0]; ++i)
    records.push(`Bar2;${3001 + i};3;${mesh.at((CORE.a * i) / n[0], 0, CORE.c)};` +
      `${mesh.at((CORE.a * (i + 1)) / n[0], 0, CORE.c)};0.;;1.;0.`);
  return records;
}

// A Bar2 along an edge of the solids (tie 3), and along an edge of shells that lie on the solids (the line then
// follows the shells, the solids follow them): stretched and bent by the ends, the bar takes E" ε on its axis and
// bends with the curvature of the core: the ends carry E ε b c + E" A ε (+ E' ε t b) and
// −κ (E b c³ / 12 + E" (A (c / 2)² + I2) (+ E' t b ((c / 2)² + t² / 12))).
for (const { shape, skin } of [{ shape: "SolidHexa", skin: false }, { shape: "SolidTetra", skin: false },
  { shape: "SolidPenta", skin: false }, { shape: "SolidPyramid", skin: false }, { shape: "SolidHexa", skin: true },
  { shape: "SolidTetra", skin: true }])
  test(`lines: Bar2 along an edge of a block of ${shape}${skin ? " under a skin of ShellQuad4" : ""}, stretched and ` +
    "bent by its ends", (t) => {
    const n = [2, 2, 2];
    const mesh = coreMesh({ shape, n });
    // The grid points of the ends that turn: those of the bar, and those of the skin.
    const turning = skin ? mesh.where((x, y, z) => (near(x, 0) || near(x, CORE.a)) && near(z, CORE.c))
      : [mesh.at(0, 0, CORE.c), mesh.at(CORE.a, 0, CORE.c)];
    const records = [...sections(), ...edgeBars(mesh, n), ...(skin ? topShells(mesh, n, "ShellQuad4") : []),
      ...endSupports(mesh, turning)];
    const run = runMixed(mesh, caseOf([[1, null], [2, null]]), records, { allowed: ["support-note"] });
    const lines = [];
    const checks = [];
    const skinArea = skin ? SKIN.thickness * CORE.b : 0;
    [{ name: "stretch", field: stretchField, strain: () => STRETCH, scale: STRETCH },
      { name: "bending", field: bendingField, strain: (zc) => -CURVATURE * zc, scale: CURVATURE * CORE.c }]
      .forEach(({ name, field, strain, scale }, k) => {
        const subcase = k + 1;
        const moved = translationError(run, mesh, field, subcase) / (scale * CORE.a);
        const solid = stressErrors(run, mesh, (x, y, z) => [CORE.young * strain(z - CORE.c / 2), 0, 0, 0, 0, 0],
          { subcase });
        const solidError = Math.max(solid.corner, solid.grid) / (CORE.young * scale);
        const end = endResultant(run, mesh, subcase);
        const expected = k === 0
          ? STRETCH * (CORE.young * CORE.b * CORE.c + LINE.young * LINE.area + SKIN.young * skinArea)
          : -CURVATURE * ((CORE.young * CORE.b * CORE.c ** 3) / 12 +
            LINE.young * (LINE.area * (CORE.c / 2) ** 2 + LINE.i2) +
            SKIN.young * skinArea * ((CORE.c / 2) ** 2 + SKIN.thickness ** 2 / 12));
        const endRatio = (k === 0 ? end.force : end.moment) / expected;
        lines.push(`${name}: displacement ${moved.toExponential(1)}, solid stress ${solidError.toExponential(1)}, ` +
          `end ratio ${endRatio.toFixed(9)}`);
        checks.push([moved, `${name}: displacements`], [solidError, `${name}: stresses of the solids`],
          [Math.abs(endRatio - 1), `${name}: force or moment of the ends`]);
      });
    for (const line of lines) t.diagnostic(line);
    for (const [error, what] of checks) assertSmall(error, 1, CRITERIA.exact, what);
  });

for (const shape of ["SolidHexa", "SolidTetra"])
  test(`lines: a free block of ${shape} with Bar2 along one of its edges has six rigid-body modes`, (t) => {
    const mesh = coreMesh({ shape });
    assertRigidModes(t, freeModes(mesh, edgeBars(mesh, [2, 2, 2])));
  });

// A Bar2 standing on the core, ending at one grid point of its top face: spec section 3.4: "a line alone that ends
// at a single grid point of a solid is pinned there". The free model then has nine modes of zero frequency: the six
// rigid motions, the two swings of the bar about its foot and its spin about its axis.
test("lines: a Bar2 that ends at one grid point of a block of SolidHexa is pinned there (nine modes of zero frequency)",
  (t) => {
    const mesh = coreMesh({});
    const foot = mesh.at(CORE.a / 2, CORE.b / 2, CORE.c);
    const head = mesh.point([CORE.a / 2, CORE.b / 2, CORE.c + 1.0]);
    const values = freeModes(mesh, [`Bar2;3001;3;${foot};${head};1.;;0.;0.`], 12);
    assertRigidModes(t, values, 9);
  });
// The same bar pulled along its axis and pushed across at its head, the core held under: pinned, it cannot be pushed
// across (the factorisation finds the swing), and it passes its axial force: its head moves by F L / (E" A) more than
// its foot.
test("lines: a Bar2 that ends at one grid point of a block of SolidHexa passes its axial force", (t) => {
  const mesh = coreMesh({});
  const foot = mesh.at(CORE.a / 2, CORE.b / 2, CORE.c);
  const head = mesh.point([CORE.a / 2, CORE.b / 2, CORE.c + 1.0]);
  const force = 1.0e5;
  // The swings and the spin of the pinned bar are held at its head (two translations) and at its foot (the twist).
  const records = [...sections(), `Bar2;3001;3;${foot};${head};1.;;0.;0.`,
    `PointForceLoad;2;${head};;${real(force)};0.;0.;1.`, `SinglePointConstr;1;${head};12;0.`,
    `SinglePointConstr;1;${head};6;0.`];
  for (const id of mesh.where((x, y, z) => near(z, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
  const run = runMixed(mesh, caseOf([[1, 2]]), records, { allowed: ["support-note"] });
  const stretch = nodal(run, "displacements", head)[2] - nodal(run, "displacements", foot)[2];
  const carried = -resultant(run, mesh, "singlePointConstrForces")[2] / force;
  t.diagnostic(`stretch ratio ${(stretch / (force / (LINE.young * LINE.area))).toFixed(8)}, force carried by the ` +
    `supports ${carried.toFixed(10)}`);
  assertRatio(stretch, force / (LINE.young * LINE.area), CRITERIA.exact, "stretch of the bar");
  assertRatio(carried, 1, CRITERIA.invariant, "force carried by the supports");
});

// A Bar2 standing on a plate of ShellQuad4 (no solid): where no line lies on a shell edge the rotation of the bar is
// the mean rotation of the shells at their corners (tie 6): the bar is clamped in the plate. On a stiff plate
// (clamped along its edges, its modulus 1e6 times that of the bar) the head pushed across moves by P L³ / (3 E" I),
// and free the model has six rigid-body modes and no swing.
function standingBar() {
  const mesh = createMesh({});
  const size = 1.0;
  const at = (i, j) => mesh.point([(size * i) / 2, (size * j) / 2, 0]);
  const records = [];
  for (let j = 0; j < 2; ++j)
    for (let i = 0; i < 2; ++i)
      records.push(`ShellQuad4;${1001 + 2 * j + i};2;${at(i, j)};${at(i + 1, j)};${at(i + 1, j + 1)};${at(i, j + 1)}`);
  const foot = at(1, 1);
  const head = mesh.point([size / 2, size / 2, 1.0]);
  records.push(`Bar2;3001;3;${foot};${head};1.;;0.;0.`);
  const options = { elementFormulation: "MixedHybridFormulation", formulationGeometry: {} };
  return { mesh, records, foot, head, options, edge: mesh.where((x, y, z) => near(z, 0) &&
    (near(x, 0) || near(x, size) || near(y, 0) || near(y, size))) };
}
test("lines: a Bar2 standing on a stiff plate of ShellQuad4 is clamped in it", (t) => {
  const { mesh, records, head, options, edge } = standingBar();
  const force = 100.0;
  const model = [...mesh.records(), ...records, `ShellProp;2;2;0.05;2;;2`, `IsoMat;2;${real(LINE.young * 1e6)};;0.3;0.`,
    `BarProp;3;3;${real(LINE.area)};${real(LINE.i1)};${real(LINE.i2)};${real(LINE.j)}`,
    `IsoMat;3;${real(LINE.young)};;${real(LINE.poisson)};0.`, `PointForceLoad;2;${head};;${real(force)};1.;0.;0.`];
  for (const id of edge) model.push(`SinglePointConstr;1;${id};123456;0.`);
  const run = runModel(deck(101, caseOf([[1, 2]]), model), options);
  // The orientation vector of the bar is x: the force along x bends it in its plane 1 (inertia I1).
  const expected = force / (3 * LINE.young * LINE.i1);
  const moved = nodal(run, "displacements", head)[0];
  t.diagnostic(`head deflection ratio ${(moved / expected).toFixed(6)}`);
  assertRatio(moved, expected, CRITERIA.displacement, "deflection of the head of the bar");
});
test("lines: a free plate of ShellQuad4 with a Bar2 standing on it has six rigid-body modes", (t) => {
  const { mesh, records, options } = standingBar();
  const model = [...mesh.records(), ...records, ...sections(), "LanczosEigenMethod;3;;;9"];
  const run = runModel(deck(103, ["SubcaseStart;1", "EigenMethodSelection;3"], model), options, "modes");
  assertRigidModes(t, run.eigenvalues[0].rows.map((row) => row.eigenvalue));
});

// ---------------------------------------------------------------------------------------------------------------------
// A plate continued by solids: a strip of thickness h, of solids over its first half (two cells through its
// thickness) and of ShellQuad4 over its second half, the shells joined to the solids along the middle line of their
// end face (tie 1 for the displacement of that edge, tie 4 for the rotation about it). With ν = 0 the strip is a beam.
const STRIP = { solids: 2.0, shells: 2.0, width: 0.5, thickness: 0.2, young: 7.0e10 };
function transitionStrip({ through = 2, along = 4, shellCells = 4 } = {}) {
  const { solids, shells, width, thickness, young } = STRIP;
  const mesh = createMesh({ surfaces: boxSurfaces(solids, width, thickness) })
    .addBlock({ n: [along, 1, through], map: (u, v, w) => [solids * u, width * v, thickness * w] });
  const at = (i, j) => mesh.point([solids + (shells * i) / shellCells, width * j, thickness / 2]);
  const records = ["SolidProp;1;1", `IsoMat;1;${real(young)};;0.;2700.`, `ShellProp;2;1;${real(thickness)};1;;1`];
  const shellIds = [];
  for (let i = 0; i < shellCells; ++i) {
    shellIds.push(1001 + i);
    records.push(`ShellQuad4;${1001 + i};2;${at(i, 0)};${at(i + 1, 0)};${at(i + 1, 1)};${at(i, 1)}`);
  }
  return { mesh, records, shellIds, tip: [at(shellCells, 0), at(shellCells, 1)] };
}

test("transition: a free strip of SolidHexa continued by ShellQuad4 has six rigid-body modes and no hinge", (t) => {
  const { mesh, records } = transitionStrip();
  const run = runMixed(mesh, ["SubcaseStart;1", "EigenMethodSelection;3"], [...records, "LanczosEigenMethod;3;;;9"],
    { solution: 103, analysis: "modes" });
  assertRigidModes(t, run.eigenvalues[0].rows.map((row) => row.eigenvalue));
});

// The strip clamped at the end of its solids, a uniform pressure p on its shells: the beam (Timoshenko, κ = 5/6) of
// the same section all along gives the tip deflection, with M and V the moment and shear force of the load and
// m = L − x that of a unit force at the tip:
// w = ∫ M m dx / (E I) + ∫ V dx / (κ G A)
//   = p b (L2 (L1³ / 3 + 3 L1² L2 / 4 + L1 L2² / 2) + L2⁴ / 8) / (E I) + p b L2 (L1 + L2 / 2) / (κ G A).
// LIMIT OF THE FORMULATION, not in the list of spec section 3.6 (kept failing): the moment of the shells enters the
// solids by the rotation of the line of their end face across the joint (spec section 3.4, tie 4; document 28 of the
// reference, section 6: "the solid is not told the thickness of the shell"): a couple on a line of the face, not a
// moment on the whole section. The joint is a hinge whose stiffness depends on the mesh and softens without bound as
// the solids are refined: the tip deflection is 1.26 times the beam's with 4 × 1 × 2 solids (this case), 1.41 with
// 8 × 1 × 2, 2.08 with 8 × 1 × 4, 2.66 with 16 × 1 × 4, 3.82 with 16 × 1 × 6.
test("transition: a cantilever strip of SolidHexa continued by ShellQuad4 bends as the beam", (t) => {
  const { solids: l1, shells: l2, width, thickness, young } = STRIP;
  const pressure = 1.0e4;
  const { mesh, records, shellIds, tip } = transitionStrip();
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
  for (const id of shellIds) records.push(`FacePressureLoad;2;${id};${real(pressure)}`);
  const run = runMixed(mesh, caseOf([[1, 2]]), records);
  const inertia = (width * thickness ** 3) / 12;
  const load = pressure * width;
  const bending = (load * (l2 * (l1 ** 3 / 3 + (3 * l1 * l1 * l2) / 4 + (l1 * l2 * l2) / 2) + l2 ** 4 / 8)) /
    (young * inertia);
  const shear = (load * l2 * (l1 + l2 / 2)) / ((5 / 6) * (young / 2) * width * thickness);
  const ratios = tip.map((id) => nodal(run, "displacements", id)[2] / (bending + shear));
  t.diagnostic(`tip deflection ratios ${ratios.map((r) => r.toFixed(5)).join(" ")}`);
  for (const ratio of ratios) assertRatio(ratio, 1, CRITERIA.displacement, "tip deflection");
});

// The same strip pulled along x by its tip: the force of the shells enters the solids along the middle line of their
// end face. Far from the joint the solids are in the uniform stress F / (b h); the tip moves by F L / (E A) and by
// what the joint gives. The same limit (kept failing): a force along a line; the stiffness is 0.932 of the bar's
// with 4 × 1 × 2 solids, 0.916 with 8 × 1 × 2, 0.913 with 8 × 1 × 4, 0.898 with 16 × 1 × 4, 0.896 with 16 × 1 × 6.
test("transition: a strip of SolidHexa continued by ShellQuad4 pulled by its tip stretches as the bar", (t) => {
  const { solids: l1, shells: l2, width, thickness, young } = STRIP;
  const { mesh, records, tip } = transitionStrip();
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
  records.push(`SinglePointConstr;1;${mesh.at(0, 0, 0)};23;0.`, `SinglePointConstr;1;${mesh.at(0, width, 0)};3;0.`);
  for (const id of tip) records.push(`SinglePointConstr;1;${id};1;1.e-3`);
  const run = runMixed(mesh, caseOf([[1, null]]), records);
  const pull = tip.reduce((sum, id) => sum + nodal(run, "singlePointConstrForces", id)[0], 0);
  const expected = (young * width * thickness * 1.0e-3) / (l1 + l2);
  const stress = rowsOf(run, "SolidGridPoints").filter((row) => near(mesh.points.get(row.id)[0], l1 / 2))
    .map((row) => row.oxx / (pull / (width * thickness)));
  t.diagnostic(`stiffness ratio ${(pull / expected).toFixed(5)}, stress of the solids at their middle ` +
    `${Math.min(...stress).toFixed(5)} to ${Math.max(...stress).toFixed(5)} of F / A`);
  assertRatio(pull, expected, CRITERIA.displacement, "stiffness");
});
