// Validation of the solids of the mixed hybrid formulation (docs/mixed-hybrid-formulation-spec.md, sections 3.3,
// 7 to 10 and 13.3): tetrahedra, pyramids, pentahedra and hexahedra, with straight edges and with mid-side nodes, on
// regular and distorted meshes, against closed-form solutions of elasticity and of beam theory, with the criteria of
// section 13.6 (support.mjs). A case that does not meet its criterion is left failing; its comment says what was
// measured and why.
//
// Run: node --test wasm/test/validation/solids.test.mjs

import assert from "node:assert/strict";
import { CRITERIA, MIXED, real, runModel, nodal, assertRatio, prepared, surfaceRecords, test }
  from "./support.mjs";

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
const SHAPES = ["SolidTetra", "SolidPyramid", "SolidPenta", "SolidHexa"];

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
// whose two grid points lie on two of them is declared sharp, on one of them smooth. `straightSkin`: the skin is
// made of planes meshed irregularly, whose mesh lines are no geometry; every line of the skin is declared ended at
// every grid point, so that each edge is straight. `quadratic` gives every edge its mid-side node, at the
// map of the middle of the edge in the lattice (on the exact curve of a curved map); a function (a, b) → boolean of
// the positions of the two ends chooses the edges that get one. `reversed` numbers the grid points and the elements
// downwards, `rotated` gives every element from its next corner (the invariance checks).
function createMesh({ surfaces = [], quadratic = false, tolerance = 1e-7, reversed = false, rotated = false,
  straightSkin = false } = {}) {
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

  // The Node and element records, and those of the declarations `geometry` of the skin (by default those of
  // `surfaces`), as a preprocessor writes them in the model file.
  mesh.records = (geometry = mesh.declarations()) => [
    ...[...mesh.points].map(([id, p]) => `Node;${id};;${p.map(real).join(";")}`),
    ...mesh.elements.map((e) => `${e.card};${e.id};${e.propId};${[...e.corners, ...e.mids].join(";")}`),
    ...surfaceRecords(geometry)];
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
  // The smooth edges of the skin: those whose two grid points lie on one surface only. On a curved surface meshed
  // with cells that turn by more than the fold angle of the model they would be read sharp without the declaration.
  mesh.smoothEdges = () => mesh.edges((a, b) =>
    surfaces.filter((surface) => surface(...a) && surface(...b)).length === 1);
  // The ends of the mesh lines of a skin of planes meshed irregularly: each edge of the skin, at both its grid points.
  mesh.lineEnds = () => mesh.edges((a, b) => surfaces.some((surface) => surface(...a) && surface(...b)))
    .flatMap(([a, b]) => [[a, b], [b, a]]);
  // The declarations of the skin: a complete declaration of its edges, which leaves nothing to the fold angle of the
  // model (spec section 4.4).
  mesh.declarations = () => ({ sharpEdges: mesh.sharpEdges(), smoothEdges: mesh.smoothEdges(),
    lineEnds: straightSkin ? mesh.lineEnds() : [] });
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
// a pentahedron). A triangle of a pyramid is named by the two corners of its edge on the base.
function facePressure(sid, face, p, direction = null) {
  const values = Array.isArray(p) ? p : [p];
  const pressures = [0, 1, 2, 3].map((k) => (k < values.length ? real(values[k]) : "")).join(";");
  let first = face.nodes[0];
  let across = "";
  if (face.nodes.length === 4) across = face.nodes[2];
  else if (face.card === "SolidTetra") across = face.element.corners.find((id) => !face.nodes.includes(id));
  else if (face.card === "SolidPyramid") {
    const apex = face.element.corners[4];
    [first, across] = face.nodes.filter((id) => id !== apex);
  }
  const tail = direction ? `;0;${direction.map(real).join(";")}` : "";
  return `FacePressureLoad;${sid};${face.element.id};${pressures};${first};${across}${tail}`;
}

// The PointsPressureLoad record of set `sid` pressing with `p` on `face`: its grid points turn about the outward
// normal, along which a positive pressure of this card acts.
function pointsPressure(sid, face, p) {
  return `PointsPressureLoad;${sid};${real(-p)};${face.nodes.join(";")}`;
}

// The records of the traction σ n of the uniform stress `stress` (xx, yy, zz, xy, yz, zx) on the flat `faces`.
function tractionRecords(sid, faces, stress) {
  const [xx, yy, zz, xy, yz, zx] = stress;
  const records = [];
  for (const face of faces) {
    const n = face.normal;
    const t = [xx * n[0] + xy * n[1] + zx * n[2], xy * n[0] + yy * n[1] + yz * n[2], zx * n[0] + yz * n[1] + zz * n[2]];
    const size = Math.hypot(...t);
    if (size > 0) records.push(facePressure(sid, face, size, t.map((v) => v / size)));
  }
  return records;
}

// ---------------------------------------------------------------------------------------------------------------------
// Decks and results.

const STEEL = { young: 2.0e11, poisson: 0.3, density: 7800.0, expansion: 1.2e-5 };

// A deck: the solution number, the case records and the model records, with the declarations of their geometry
// (`prepared`).
function deck(solution, caseRecords, modelRecords) {
  return ["[executive]", `SolutionSequence;${solution}`, "[case]", ...caseRecords, "[model]",
    ...prepared(modelRecords), ""].join("\n");
}

// The case records of a static analysis of load sets `loads` (one subcase each) on the constraint set 1.
function staticCase(loads = [2]) {
  const records = ["DisplacementOutput;All", "SinglePointConstrForceOutput;All", "StressOutput;All"];
  loads.forEach((load, k) => records.push(`SubcaseStart;${k + 1}`, "SinglePointConstrSelection;1",
    ...(load === null ? [] : [`StaticLoadSelection;${load}`])));
  return records;
}

// The property and isotropic material records of the solids.
function isotropic({ young, poisson, density = 0, expansion = 0 }) {
  return ["SolidProp;1;1", `IsoMat;1;${real(young)};;${real(poisson)};${real(density)};${real(expansion)};0.`];
}

// Supports that hold the six rigid motions and nothing else: three grid points that no element edge joins (a support
// is read along the edges, spec section 5.6: two held grid points of one edge would hold the edge between them).
function isostatic(a, b, c) {
  return [`SinglePointConstr;1;${a};123;0.`, `SinglePointConstr;1;${b};23;0.`, `SinglePointConstr;1;${c};3;0.`];
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

// Solves the augmented system [A | b] by Gauss elimination with partial pivoting.
function solveLinear(augmented) {
  const n = augmented.length;
  const a = augmented.map((row) => [...row]);
  for (let c = 0; c < n; ++c) {
    let pivot = c;
    for (let r = c + 1; r < n; ++r) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    for (let r = 0; r < n; ++r) {
      if (r === c) continue;
      const factor = a[r][c] / a[c][c];
      for (let k = c; k <= n; ++k) a[r][k] -= factor * a[c][k];
    }
  }
  return a.map((row, r) => row[n] / row[r]);
}

// The largest difference between the displacements of the grid points of `mesh` and `field(x, y, z)`, once the rigid
// motion a + b × x that fits it best (least squares) is taken off: the supports may then be any isostatic set.
function fieldError(run, mesh, field, subcase = 1) {
  const rows = [];
  const rhs = [];
  for (const [id, p] of mesh.points) {
    const u = nodal(run, "displacements", id, subcase);
    const e = field(...p);
    rows.push([1, 0, 0, 0, p[2], -p[1]], [0, 1, 0, -p[2], 0, p[0]], [0, 0, 1, p[1], -p[0], 0]);
    rhs.push(u[0] - e[0], u[1] - e[1], u[2] - e[2]);
  }
  const normal = Array.from({ length: 6 }, () => new Array(7).fill(0));
  rows.forEach((row, r) => {
    for (let i = 0; i < 6; ++i) {
      for (let j = 0; j < 6; ++j) normal[i][j] += row[i] * row[j];
      normal[i][6] += row[i] * rhs[r];
    }
  });
  const fit = solveLinear(normal);
  let worst = 0;
  rows.forEach((row, r) => {
    worst = Math.max(worst, Math.abs(rhs[r] - row.reduce((sum, v, i) => sum + v * fit[i], 0)));
  });
  return worst;
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
  const run = runModel(deck(solution, caseRecords, [...mesh.records(), ...records]), MIXED, analysis);
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
// 1. Patch tests.

const BLOCK = { a: 2.0, b: 1.0, c: 1.5 };
const blockMap = (u, v, w) => [BLOCK.a * u, BLOCK.b * v, BLOCK.c * w];
// The block with its inside pushed around: the skin keeps its planes and its straight edges, the mid-side nodes of the
// edges inside leave their chords (curved edges inside the block).
const curvedInsideMap = (u, v, w) => {
  const bubble = 0.25 * 64 * u * (1 - u) * v * (1 - v) * w * (1 - w);
  return [BLOCK.a * u + 0.1 * bubble, BLOCK.b * v + 0.05 * bubble, BLOCK.c * w - 0.07 * bubble];
};
// The meshes of the patch tests: one shape, or several that share triangular and quadrilateral faces.
const PATCH_MESHES = [
  { name: "SolidHexa", n: [2, 2, 2], shape: "SolidHexa" },
  { name: "SolidPenta", n: [2, 2, 2], shape: "SolidPenta" },
  { name: "SolidTetra", n: [2, 2, 2], shape: "SolidTetra" },
  { name: "SolidPyramid", n: [2, 2, 2], shape: "SolidPyramid" },
  { name: "SolidPyramid and SolidTetra on triangles", n: [2, 2, 2], shape: "Transition" },
  { name: "SolidHexa, SolidPyramid, SolidTetra and SolidPenta", n: [3, 2, 2],
    shape: (i) => ["SolidHexa", "Transition", "SolidPenta"][i] },
  { name: "SolidPenta under SolidTetra", n: [2, 2, 2], shape: (i, j, k) => (k === 0 ? "SolidPenta" : "SolidTetra") },
];
const GEOMETRIES = [
  { name: "straight edges", quadratic: false, map: blockMap },
  { name: "mid-side nodes and curved edges inside", quadratic: true, map: curvedInsideMap },
];

function patchMesh({ n, shape }, { quadratic, map }, more = {}) {
  return createMesh({ surfaces: boxSurfaces(BLOCK.a, BLOCK.b, BLOCK.c), quadratic, ...more.mesh })
    .addBlock({ n, map, shape, distortion: 0.2, ...more.block });
}
// The isostatic supports of the block at three of its corners.
const blockSupports = (mesh) => isostatic(mesh.at(0, 0, 0), mesh.at(BLOCK.a, 0, 0), mesh.at(0, BLOCK.b, 0));

// The strains (xx, yy, zz and the engineering shears xy, yz, zx) of an isotropic material under `stress`.
function isotropicStrain({ young, poisson }, stress) {
  const [xx, yy, zz, xy, yz, zx] = stress;
  const shear = young / (2 * (1 + poisson));
  return [(xx - poisson * (yy + zz)) / young, (yy - poisson * (xx + zz)) / young, (zz - poisson * (xx + yy)) / young,
    xy / shear, yz / shear, zx / shear];
}
// The displacement field of the uniform strain `e`.
const strainField = (e) => (x, y, z) =>
  [e[0] * x + (e[3] * y) / 2 + (e[5] * z) / 2, (e[3] * x) / 2 + e[1] * y + (e[4] * z) / 2,
    (e[5] * x) / 2 + (e[4] * y) / 2 + e[2] * z];

// The six uniform stress states (the patch test of Irons): each under its tractions σ n on the skin of a distorted
// block held isostatically. The displacements are those of the uniform strain S σ and every stress row is σ.
const UNIT_STRESS = 1.0e8;
for (const meshCase of PATCH_MESHES)
  for (const geometry of GEOMETRIES)
    test(`patch: the six uniform stress states on a distorted block of ${meshCase.name}, ${geometry.name}`, () => {
      const mesh = patchMesh(meshCase, geometry);
      const faces = skinFaces(mesh);
      const records = [...isotropic(STEEL), ...blockSupports(mesh)];
      const states = [0, 1, 2, 3, 4, 5].map((c) => [0, 1, 2, 3, 4, 5].map((k) => (k === c ? UNIT_STRESS : 0)));
      states.forEach((stress, c) => records.push(...tractionRecords(11 + c, faces, stress)));
      const run = runMixed(mesh, staticCase(states.map((_, c) => 11 + c)), records);
      states.forEach((stress, c) => {
        const strain = isotropicStrain(STEEL, stress);
        const scale = (UNIT_STRESS / STEEL.young) * BLOCK.a;
        assertSmall(fieldError(run, mesh, strainField(strain), c + 1), scale, CRITERIA.exact,
          `displacements of state ${c + 1}`);
        const errors = stressErrors(run, mesh, () => stress, { subcase: c + 1 });
        for (const card of mesh.cards())
          for (const row of rowsOf(run, card, "stress", c + 1))
            if (row.nodeId === 0)
              errors.centre = Math.max(errors.centre, ...sixOf(row).map((v, k) => Math.abs(v - stress[k])));
        for (const where of ["corner", "centre", "grid"])
          assertSmall(errors[where], UNIT_STRESS, CRITERIA.exact, `${where} stresses of state ${c + 1}`);
      });
    });

// A pure bending (Timoshenko and Goodier, Theory of Elasticity, section 102): the stress along the axis `a` is
// c (x_b − middle_b), linear along `b`, under that normal traction on the two faces across `a`. With t the third axis:
// u_a = c x_a x_b' / E, u_b = −c (x_a² + ν (x_b'² − x_t'²)) / (2 E), u_t = −ν c x_b' x_t' / E (x' from the middle).
const BLOCK_SIZE = [BLOCK.a, BLOCK.b, BLOCK.c];
function bendingState(a, b, c) {
  const t = 3 - a - b;
  const { young, poisson } = STEEL;
  const centred = (p, axis) => p[axis] - BLOCK_SIZE[axis] / 2;
  return {
    stress: (...p) => { const s = [0, 0, 0, 0, 0, 0]; s[a] = c * centred(p, b); return s; },
    field: (...p) => {
      const u = [0, 0, 0];
      u[a] = (c * p[a] * centred(p, b)) / young;
      u[b] = (-c * (p[a] * p[a] + poisson * (centred(p, b) ** 2 - centred(p, t) ** 2))) / (2 * young);
      u[t] = (-poisson * c * centred(p, b) * centred(p, t)) / young;
      return u;
    },
    // The pressure −σ_aa at each corner of the faces across `a`.
    records: (sid, faces) => faces.filter((face) => Math.abs(face.normal[a]) > 0.99)
      .map((face) => facePressure(sid, face, face.x.map((p) => -c * centred(p, b)))),
  };
}
const BENDING_STATES = [bendingState(0, 1, UNIT_STRESS), bendingState(1, 2, UNIT_STRESS),
  bendingState(2, 0, UNIT_STRESS)];

// The three bendings of the block and their checks: displacements and stresses (corners, centres of the straight
// elements, grid points) to the criterion of the exact states.
function bendingPatch(t, mesh, { centres }) {
  const faces = skinFaces(mesh);
  const records = [...isotropic(STEEL), ...blockSupports(mesh)];
  BENDING_STATES.forEach((state, c) => records.push(...state.records(11 + c, faces)));
  const run = runMixed(mesh, staticCase(BENDING_STATES.map((_, c) => 11 + c)), records);
  const worst = { displacement: 0, corner: 0, centre: 0, grid: 0 };
  BENDING_STATES.forEach((state, c) => {
    const scale = (UNIT_STRESS / STEEL.young) * BLOCK.a * BLOCK.a;
    worst.displacement = Math.max(worst.displacement, fieldError(run, mesh, state.field, c + 1) / scale);
    const errors = stressErrors(run, mesh, state.stress, { subcase: c + 1, centres });
    for (const where of ["corner", "centre", "grid"])
      worst[where] = Math.max(worst[where], errors[where] / UNIT_STRESS);
  });
  t.diagnostic(`errors: ${Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(2)}`).join(", ")}`);
  for (const [where, error] of Object.entries(worst)) assertSmall(error, 1, CRITERIA.exact, where);
}

// Linear stress fields, the skin of the block regular (its planes cut into equal cells) and its inside distorted.
for (const meshCase of PATCH_MESHES)
  test(`patch: three pure bendings on a block of ${meshCase.name} distorted inside, straight edges`, (t) => {
    bendingPatch(t, patchMesh(meshCase, GEOMETRIES[0], { block: { distortsSkin: false } }), { centres: true });
  });

// The same with mid-side nodes: the skin distorted too, the edges inside curved.
for (const meshCase of PATCH_MESHES)
  test(`patch: three pure bendings on a distorted block of ${meshCase.name}, mid-side nodes and curved edges inside`,
    (t) => { bendingPatch(t, patchMesh(meshCase, GEOMETRIES[1]), { centres: false }); });

// The same with straight edges and the grid points of the skin moved in their planes, three cells along each side.
// The faces of the block are planes meshed irregularly: their mesh lines are no geometry, and the model file declares
// every line of the skin ended at every grid point, as a preprocessor that knows the planes declares them (spec
// section 4.4). Without the declaration a mesh line of three grid points whose middle one is off-centre is read as
// one curve through them: it stays straight but is not run through at a uniform speed, and the linear stress states
// lose 1e-6 to 4e-6 of exactness with two or three cells along a side.
const STRAIGHT_SKIN = { mesh: { straightSkin: true } };
for (const meshCase of [PATCH_MESHES[0], PATCH_MESHES[4], PATCH_MESHES[5], PATCH_MESHES[6]])
  test(`patch: three pure bendings on a block of ${meshCase.name} whose skin is distorted too, straight edges`,
    (t) => {
      bendingPatch(t, patchMesh({ ...meshCase, n: [3, 3, 3] }, GEOMETRIES[0], STRAIGHT_SKIN), { centres: true });
    });

// The same block of hexahedra with two cells along each side. (The corner pressures of a FacePressureLoad were
// interpolated in the parameters of the face, which this case found: they are now interpolated by position.)
test("patch: three pure bendings on a block of 2 x 2 x 2 SolidHexa whose skin is distorted too, straight edges",
  (t) => { bendingPatch(t, patchMesh(PATCH_MESHES[0], GEOMETRIES[0], STRAIGHT_SKIN), { centres: true }); });

// A displacement prescribed on the whole skin (SinglePointConstr values on every grid point of the skin): the linear
// field of a uniform strain with its six components. Every grid point then follows the field and every stress row is
// the stress of that strain.
for (const meshCase of [PATCH_MESHES[0], PATCH_MESHES[5]])
  for (const geometry of GEOMETRIES)
    test(`patch: a uniform strain prescribed on the skin of a distorted block of ${meshCase.name}, ${geometry.name}`,
      (t) => {
        const mesh = patchMesh(meshCase, geometry);
        const strain = [1.0e-3, -0.4e-3, 0.7e-3, 0.5e-3, -0.3e-3, 0.8e-3];
        const field = strainField(strain);
        const onSkin = (x, y, z) => boxSurfaces(BLOCK.a, BLOCK.b, BLOCK.c).some((surface) => surface(x, y, z));
        const records = [...isotropic(STEEL)];
        for (const id of mesh.where(onSkin))
          field(...mesh.points.get(id)).forEach((value, c) =>
            records.push(`SinglePointConstr;1;${id};${c + 1};${real(value)}`));
        const run = runMixed(mesh, staticCase([null]), records);
        let worst = 0;
        for (const [id, p] of mesh.points)
          field(...p).forEach((value, c) => {
            worst = Math.max(worst, Math.abs(nodal(run, "displacements", id)[c] - value));
          });
        // σ = λ tr(ε) I + 2 G ε (the law of Hooke of an isotropic material).
        const { young, poisson } = STEEL;
        const shear = young / (2 * (1 + poisson));
        const lambda = (young * poisson) / ((1 + poisson) * (1 - 2 * poisson));
        const trace = strain[0] + strain[1] + strain[2];
        const stress = [0, 1, 2].map((c) => lambda * trace + 2 * shear * strain[c])
          .concat([3, 4, 5].map((c) => shear * strain[c]));
        const errors = stressErrors(run, mesh, () => stress);
        t.diagnostic(`errors: displacement ${(worst / 1e-3).toExponential(2)}, corner stress ` +
          `${(errors.corner / stress[0]).toExponential(2)}, grid stress ${(errors.grid / stress[0]).toExponential(2)}`);
        assertSmall(worst, 1.0e-3 * BLOCK.a, CRITERIA.exact, "displacements");
        assertSmall(errors.corner, Math.abs(stress[0]), CRITERIA.exact, "corner stresses");
        assertSmall(errors.grid, Math.abs(stress[0]), CRITERIA.exact, "grid point stresses");
      });

// Invariance (spec section 13.4): the same distorted block of mixed shapes under a pure bending, its grid points and
// elements numbered downwards, or every element given from its next corner. Displacements and grid point stresses at
// the same positions to 1e-8.
test("invariance: the numbering of the grid points and elements and the first corner of the elements", (t) => {
  const solve = (variant) => {
    const mesh = patchMesh(PATCH_MESHES[5], GEOMETRIES[1], { mesh: variant });
    const records = [...isotropic(STEEL), ...blockSupports(mesh), ...BENDING_STATES[0].records(11, skinFaces(mesh))];
    const run = runMixed(mesh, staticCase([11]), records);
    const stresses = new Map(rowsOf(run, "SolidGridPoints").map((row) => [row.id, sixOf(row)]));
    return { mesh, run, stresses };
  };
  const base = solve({});
  const scale = (UNIT_STRESS / STEEL.young) * BLOCK.a * BLOCK.a;
  for (const [name, variant] of [["numbered downwards", { reversed: true }], ["rotated", { rotated: true }]]) {
    const other = solve(variant);
    assert.equal(other.mesh.points.size, base.mesh.points.size);
    let displacement = 0;
    let stress = 0;
    for (const [id, p] of base.mesh.points) {
      const twin = other.mesh.at(...p);
      const u = nodal(base.run, "displacements", id);
      const v = nodal(other.run, "displacements", twin);
      for (let c = 0; c < 3; ++c) displacement = Math.max(displacement, Math.abs(u[c] - v[c]));
      if (!base.stresses.has(id)) continue;
      base.stresses.get(id).forEach((value, c) => {
        stress = Math.max(stress, Math.abs(value - other.stresses.get(twin)[c]));
      });
    }
    t.diagnostic(`${name}: displacement ${(displacement / scale).toExponential(2)}, stress ` +
      `${(stress / UNIT_STRESS).toExponential(2)}`);
    assertSmall(displacement, scale, CRITERIA.invariant, `displacements, ${name}`);
    assertSmall(stress, UNIT_STRESS, CRITERIA.invariant, `grid point stresses, ${name}`);
  }
});

// A sector of a thick ring between the radii a and b, of height h, in a lattice (radius, angle, height); with
// `quadratic` its mid-side nodes lie on the arcs. The planes at the angles 0 and `angle` cut it.
function ringSurfaces(a, b, h, angle = Math.PI / 2) {
  const azimuth = (x, y) => Math.atan2(y, x);
  return [(x, y) => near(Math.hypot(x, y), a), (x, y) => near(Math.hypot(x, y), b),
    (x, y) => near(azimuth(x, y), 0), (x, y) => near(azimuth(x, y), angle),
    (x, y, z) => near(z, 0), (x, y, z) => near(z, h)];
}
function ringMesh({ a, b, h, n, shape = "SolidHexa", quadratic = false, angle = Math.PI / 2, grading = (u) => u,
  distortion = 0 }) {
  const map = (u, v, w) => {
    const r = a + (b - a) * grading(u);
    return [r * Math.cos(angle * v), r * Math.sin(angle * v), h * w];
  };
  return createMesh({ surfaces: ringSurfaces(a, b, h, angle), quadratic }).addBlock({ n, map, shape, distortion });
}

// Free vibrations: exactly six modes of zero frequency (the rigid motions: |λ| at most 1e-8 of the first elastic λ)
// and no spurious mode: the first elastic frequency is that of the block, 646.8 Hz (4 × 4 × 4 SolidHexa of this
// formulation: 646.80 Hz; 3 × 3 × 3: 646.90 Hz).
const BLOCK_FIRST_FREQUENCY = 646.8;
function freeModes(mesh, count = 8) {
  const records = [...isotropic(STEEL), `LanczosEigenMethod;3;;;${count}`];
  const run = runMixed(mesh, ["SubcaseStart;1", "EigenMethodSelection;3"], records,
    { solution: 103, analysis: "modes" });
  return run.eigenvalues[0].rows.map((row) => row.eigenvalue);
}
function assertSixRigidModes(t, values) {
  assert.ok(values.length >= 7, `${values.length} eigenvalues`);
  const worst = Math.max(...values.slice(0, 6).map(Math.abs)) / values[6];
  t.diagnostic(`rigid eigenvalues at most ${worst.toExponential(2)} of the first elastic one, ` +
    `${(Math.sqrt(values[6]) / (2 * Math.PI)).toFixed(2)} Hz`);
  assert.ok(worst <= CRITERIA.invariant, `six rigid modes: ${worst}`);
}
const FREE_BLOCKS = [...PATCH_MESHES.slice(0, 5).map((meshCase) => [meshCase, GEOMETRIES[0]]),
  [PATCH_MESHES[0], GEOMETRIES[1]], [PATCH_MESHES[2], GEOMETRIES[1]]];
for (const [meshCase, geometry] of FREE_BLOCKS)
    test(`rigid modes: a free distorted block of ${meshCase.name}, ${geometry.name}`, (t) => {
      const values = freeModes(patchMesh(meshCase, geometry));
      assertSixRigidModes(t, values);
      assertRatio(Math.sqrt(values[6]) / (2 * Math.PI), BLOCK_FIRST_FREQUENCY, CRITERIA.frequency,
        "first elastic frequency");
    });

// The same on a quarter of a thick ring whose mid-side nodes lie on the arcs (curved edges on the skin); the seventh
// mode is elastic (its eigenvalue is of the order of the following one, not a mechanism).
for (const shape of SHAPES)
  test(`rigid modes: a free quarter ring of ${shape} with mid-side nodes on the arcs`, (t) => {
    const mesh = ringMesh({ a: 1, b: 2, h: 1, n: [1, 2, 1], shape, quadratic: true, distortion: 0.15 });
    const values = freeModes(mesh);
    assertSixRigidModes(t, values);
    assert.ok(values[6] > 0.05 * values[7], `the seventh eigenvalue ${values[6]} is a mechanism (${values[7]})`);
  });

// ---------------------------------------------------------------------------------------------------------------------
// 2. Beams of solids.

// The mean of the displacements of the grid points `ids`.
function meanDisplacement(run, ids, subcase = 1) {
  const sum = [0, 0, 0];
  for (const id of ids) nodal(run, "displacements", id, subcase).forEach((v, c) => { if (c < 3) sum[c] += v; });
  return sum.map((v) => v / ids.length);
}

// The torsion constant J of a rectangle of sides a ≥ b and the largest shear stress of a twist rate θ' (at the middle
// of the long sides), by the series of Saint-Venant (Timoshenko and Goodier, Theory of Elasticity, section 109):
// J = a b³ (1/3 − (64 b / (π⁵ a)) Σ tanh(n π a / (2 b)) / n⁵),
// τmax = G θ' b (1 − (8 / π²) Σ 1 / (n² cosh(n π a / (2 b)))), n odd.
// A square of side a: J = 0.1406 a⁴, τmax = 0.6753 G θ' a.
function rectangleTorsion(a, b) {
  let constant = 0;
  let stress = 0;
  for (let n = 1; n < 60; n += 2) {
    constant += Math.tanh((n * Math.PI * a) / (2 * b)) / n ** 5;
    stress += 1 / (n * n * Math.cosh((n * Math.PI * a) / (2 * b)));
  }
  return { constant: a * b ** 3 * (1 / 3 - ((64 * b) / (Math.PI ** 5 * a)) * constant),
    stressFactor: b * (1 - (8 / Math.PI ** 2) * stress) };
}

// A beam along x of length `length` and section `depth` (y) × `width` (z) in n cells, its stations moved by
// `shift(i)` × (y / depth − 1/2) (the trapezoids and parallelograms of MacNeal and Harder).
function beamMesh({ length, depth, width, n, shape = "SolidHexa", quadratic = false, distortion = 0,
  distortsSkin = true, shift = () => 0 }) {
  const map = (u, v, w) => {
    const station = Math.min(Math.floor(u * n[0] + 1e-9), n[0] - 1);
    const t = u * n[0] - station;
    const moved = (1 - t) * shift(station) + t * shift(station + 1);
    return [length * u + moved * (v - 0.5), depth * v, width * w];
  };
  return createMesh({ surfaces: boxSurfaces(length, depth, width), quadratic })
    .addBlock({ n, map, shape, distortion, distortsSkin });
}

// A root that holds the resultants of the beam and lets its section contract (the support of beam theory): the root
// face on the plane x = 0, v along its edge y = 0 and w along its edge z = 0.
function softRoot(mesh) {
  const records = [];
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
  for (const id of mesh.where((x, y) => near(x, 0) && near(y, 0))) records.push(`SinglePointConstr;1;${id};2;0.`);
  for (const id of mesh.where((x, y, z) => near(x, 0) && near(z, 0))) records.push(`SinglePointConstr;1;${id};3;0.`);
  return records;
}


// A cantilever of square section a, of length 10 a, with ν = 0 (the plane solution of Timoshenko and Goodier,
// section 21, is then the solution in space, and a clamped root does not hold any contraction of the section):
// - under an axial force P: u = P L / (E A), σxx = P / A everywhere;
// - under an end moment M: v = M L² / (2 E I), σxx = −M y' / I everywhere (y' from the axis);
// - under a tip shear P: v = P L³ / (3 E I) + 6 P L / (5 G A); away from the ends σxx = −P (L − x) y' / I and
//   τxy = P ((a / 2)² − y'²) / (2 I), read at the grid points of the middle of the span (0.3 L to 0.7 L: the root,
//   whose section cannot warp, and the tip, loaded by a uniform traction, disturb the stresses over one side of
//   the section and, through cells longer than it, further: with four cells along the span the shear stress at the
//   middle is 0.7 % (hexahedra) to 1.4 % (tetrahedra) off, with six 0.6 % to 1.0 %, with the eight of this case
//   below 0.1 %).
// Distorted meshes: the lattice moved by 0.2 of the side of the section; inside only with straight edges (the grid
// points of the skin stay where the reading of the skin leaves the edges straight), the skin too with mid-side nodes.
const CANTILEVER = { length: 10.0, side: 1.0, young: 2.0e11, poisson: 0.0, density: 7800.0 };
const CANTILEVER_VARIANTS = [{ shape: "SolidHexa", quadratic: false, distorted: true }];
for (const shape of SHAPES)
  CANTILEVER_VARIANTS.push({ shape, quadratic: false, distorted: false }, { shape, quadratic: true, distorted: true });
for (const variant of CANTILEVER_VARIANTS)
  variant.name = `${variant.distorted ? "distorted" : "regular"} ${variant.shape}` +
    `${variant.quadratic ? " with mid-side nodes" : ""}`;

for (const variant of CANTILEVER_VARIANTS)
  test(`beam: a cantilever of ${variant.name} under an axial force, an end moment and a tip shear`, (t) => {
    const { length, side, young } = CANTILEVER;
    const area = side * side;
    const inertia = side ** 4 / 12;
    const shear = young / 2;
    const force = 1.0e6;
    const moment = 1.0e6;
    const mesh = beamMesh({ length, depth: side, width: side, n: [8, 2, 2], shape: variant.shape,
      quadratic: variant.quadratic, distortion: variant.distorted ? [0.04, 0.2, 0.2] : 0,
      distortsSkin: variant.quadratic });
    const tipFaces = skinFaces(mesh).filter((face) => face.normal[0] > 0.99);
    const records = [...isotropic(CANTILEVER)];
    for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
    for (const face of tipFaces)
      records.push(facePressure(11, face, force / area, [1, 0, 0]),
        // σxx = −M (y − a/2) / I on the tip face: the moment M about z bends the beam towards +y.
        facePressure(12, face, face.x.map((p) => (moment * (p[1] - side / 2)) / inertia)),
        facePressure(13, face, force / area, [0, 1, 0]));
    const run = runMixed(mesh, staticCase([11, 12, 13]), records);
    const tip = mesh.where((x) => near(x, length));

    const extension = meanDisplacement(run, tip, 1)[0] / ((force * length) / (young * area));
    const axial = stressErrors(run, mesh, () => [force / area, 0, 0, 0, 0, 0], { subcase: 1 });
    const bent = meanDisplacement(run, tip, 2)[1] / ((moment * length ** 2) / (2 * young * inertia));
    const largest = (moment * side) / (2 * inertia);
    const bending = stressErrors(run, mesh, (x, y) => [(-moment * (y - side / 2)) / inertia, 0, 0, 0, 0, 0],
      { subcase: 2 });
    const sheared = meanDisplacement(run, tip, 3)[1] /
      ((force * length ** 3) / (3 * young * inertia) + (6 * force * length) / (5 * shear * area));
    const bendingAtMiddle = (force * (length / 2) * (side / 2)) / inertia;
    const neutral = (3 * force) / (2 * area);
    let bendingError = 0;
    let shearError = 0;
    for (const row of rowsOf(run, "SolidGridPoints", "stress", 3)) {
      const [x, y] = mesh.points.get(row.id);
      if (x < 0.3 * length || x > 0.7 * length) continue;
      bendingError = Math.max(bendingError, Math.abs(row.oxx + (force * (length - x) * (y - side / 2)) / inertia));
      shearError = Math.max(shearError,
        Math.abs(row.txy - (force * ((side / 2) ** 2 - (y - side / 2) ** 2)) / (2 * inertia)));
    }
    t.diagnostic(`axial: u ratio ${extension.toFixed(6)}, stress error corner ` +
      `${(axial.corner / (force / area)).toExponential(1)}, grid ${(axial.grid / (force / area)).toExponential(1)}`);
    t.diagnostic(`moment: v ratio ${bent.toFixed(6)}, stress error corner ` +
      `${(bending.corner / largest).toExponential(1)}, grid ${(bending.grid / largest).toExponential(1)}`);
    t.diagnostic(`shear: v ratio ${sheared.toFixed(6)}, grid stress error bending ` +
      `${(bendingError / bendingAtMiddle).toExponential(1)}, shear ${(shearError / neutral).toExponential(1)}`);
    assertRatio(extension, 1, CRITERIA.displacement, "tip extension under the axial force");
    assertSmall(axial.grid, force / area, CRITERIA.stress, "grid point stresses under the axial force");
    assertSmall(axial.corner, force / area, CRITERIA.stress, "corner stresses under the axial force");
    assertRatio(bent, 1, CRITERIA.displacement, "tip deflection under the end moment");
    assertSmall(bending.grid, largest, CRITERIA.stress, "grid point stresses under the end moment");
    assertSmall(bending.corner, largest, CRITERIA.stress, "corner stresses under the end moment");
    assertRatio(sheared, 1, CRITERIA.displacement, "tip deflection under the tip shear");
    assertSmall(bendingError, bendingAtMiddle, CRITERIA.stress, "bending stress under the tip shear");
    assertSmall(shearError, neutral, CRITERIA.stress, "shear stress under the tip shear");
  });

// The torsion of a bar of square section a (Saint-Venant; Timoshenko and Goodier, section 109): the sections turn in
// their planes and warp freely: v = w = 0 on the root face, the rotation φ of the tip face prescribed
// (v = −φ z', w = φ y'), u held at the middle of the root. T = G J φ / L with J = 0.1406 a⁴, and the largest shear
// stress 0.6753 G (φ / L) a at the middle of the sides. Four cells along each side of the section (with two the
// torque is within 5e-5 and the largest stress 0.8 % (pyramids) to 2.6 % (pentahedra) off), two cells along the
// length: with one, the diagonal edges of the tetrahedra and pentahedra would join a grid point of the root to one of
// the tip, and a support is read along the edge between two held grid points (spec section 5.6): v and w would be
// held linear along those diagonals, where the torsion makes them quadratic (torque 1.33 and 1.17 times too large).
const TORSION_BAR = { length: 1.0, side: 1.0, young: 2.0e11, poisson: 0.3, density: 7800.0 };
for (const shape of SHAPES)
  for (const quadratic of [false, true])
    test(`beam: a bar of ${shape}${quadratic ? " with mid-side nodes" : ""} twisted by its ends`, (t) => {
      const { length, side, young, poisson } = TORSION_BAR;
      const shear = young / (2 * (1 + poisson));
      const angle = 1.0e-3;
      const mesh = beamMesh({ length, depth: side, width: side, n: [2, 4, 4], shape, quadratic });
      const tip = mesh.where((x) => near(x, length));
      const records = [...isotropic(TORSION_BAR), `SinglePointConstr;1;${mesh.at(0, side / 2, side / 2)};1;0.`];
      for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};23;0.`);
      for (const id of tip) {
        const [, y, z] = mesh.points.get(id);
        records.push(`SinglePointConstr;1;${id};2;${real(-angle * (z - side / 2))}`,
          `SinglePointConstr;1;${id};3;${real(angle * (y - side / 2))}`);
      }
      const run = runMixed(mesh, staticCase([null]), records);
      const { constant, stressFactor } = rectangleTorsion(side, side);
      let torque = 0;
      for (const id of tip) {
        const [, y, z] = mesh.points.get(id);
        const force = nodal(run, "singlePointConstrForces", id);
        torque += (y - side / 2) * force[2] - (z - side / 2) * force[1];
      }
      const expected = (shear * constant * angle) / length;
      const largest = shear * (angle / length) * stressFactor;
      const grid = new Map(rowsOf(run, "SolidGridPoints").map((row) => [row.id, row]));
      // At the middle of the sides z = 0 and z = a: τxy = ±τmax; of the sides y = 0 and y = a: τxz = ∓τmax.
      const middle = length / 2;
      const stressError = Math.max(Math.abs(grid.get(mesh.at(middle, side / 2, side)).txy + largest),
        Math.abs(grid.get(mesh.at(middle, side / 2, 0)).txy - largest),
        Math.abs(grid.get(mesh.at(middle, side, side / 2)).txz - largest),
        Math.abs(grid.get(mesh.at(middle, 0, side / 2)).txz + largest));
      t.diagnostic(`torque ratio ${(torque / expected).toFixed(6)}, largest shear stress error ` +
        `${(stressError / largest).toExponential(2)}`);
      assertRatio(torque, expected, CRITERIA.displacement, "torque");
      assertSmall(stressError, largest, CRITERIA.stress, "largest shear stress");
    });

// The straight beam of MacNeal and Harder (A proposed standard set of problems to test finite element accuracy,
// Finite Elements in Analysis and Design 1 (1985) 3-20): length 6, depth 0.2, width 0.1, E = 1e7, ν = 0.3, six
// elements along its length, regular, trapezoids and parallelograms at 45 degrees. Unit loads at the tip: extension
// 3.0e-5, in-plane shear 0.1081, out-of-plane shear 0.4321 (beam theory with the shear deflection: P L / (E A) and
// P L³ / (3 E I) + 6 P L / (5 G A)). The root is that of beam theory (its section is free to contract): with every
// grid point of the root clamped, as the publication draws it, the solid is stiffer than the beam (measured with the
// regular hexahedra: 0.9981, 0.9949 and 0.9957 of these values, 0.9989, 0.9971 and 0.9964 with twice the cells).
// The twist is that of Saint-Venant, the sections free to warp: T = G J φ / L with J = 0.2287 a b³ of the series
// (the unit torque then turns the tip by 0.03406; the publication gives 0.03208).
const STRAIGHT_BEAM = { length: 6.0, depth: 0.2, width: 0.1, young: 1.0e7, poisson: 0.3 };
const STRAIGHT_BEAM_MESHES = [
  { name: "regular", shift: () => 0 },
  { name: "trapezoid", shift: (i) => (i === 0 || i === 6 ? 0 : (i % 2 ? 0.2 : -0.2)) },
  { name: "parallelogram", shift: (i) => (i === 0 || i === 6 ? 0 : 0.2) },
];
for (const shape of SHAPES)
  for (const { name, shift } of STRAIGHT_BEAM_MESHES)
    test(`beam: the straight beam of MacNeal and Harder in six ${name} cells of ${shape}`, (t) => {
      const { length, depth, width, young, poisson } = STRAIGHT_BEAM;
      const area = depth * width;
      const shear = young / (2 * (1 + poisson));
      const mesh = beamMesh({ length, depth, width, n: [6, 1, 1], shape, shift });
      const tip = mesh.where((x) => near(x, length));
      const angle = 1.0e-3;
      const records = [...isotropic({ young, poisson }), ...softRoot(mesh)];
      for (const face of skinFaces(mesh).filter((f) => f.normal[0] > 0.99))
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((direction, c) =>
          records.push(facePressure(11 + c, face, 1 / area, direction)));
      // The twist (constraint set 2): v = w = 0 on the root, the tip turned by φ, u held at one grid point.
      records.push(`SinglePointConstr;2;${mesh.at(0, 0, 0)};1;0.`);
      for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;2;${id};23;0.`);
      for (const id of tip) {
        const [, y, z] = mesh.points.get(id);
        records.push(`SinglePointConstr;2;${id};2;${real(-angle * (z - width / 2))}`,
          `SinglePointConstr;2;${id};3;${real(angle * (y - depth / 2))}`);
      }
      const caseRecords = [...staticCase([11, 12, 13]), "SubcaseStart;4", "SinglePointConstrSelection;2"];
      const run = runMixed(mesh, caseRecords, records);
      const bending = (inertia) => length ** 3 / (3 * young * inertia) + (6 * length) / (5 * shear * area);
      const ratios = {
        extension: meanDisplacement(run, tip, 1)[0] / (length / (young * area)),
        inPlane: meanDisplacement(run, tip, 2)[1] / bending((width * depth ** 3) / 12),
        outOfPlane: meanDisplacement(run, tip, 3)[2] / bending((depth * width ** 3) / 12),
      };
      let torque = 0;
      for (const id of tip) {
        const [, y, z] = mesh.points.get(id);
        const force = nodal(run, "singlePointConstrForces", id, 4);
        torque += (y - depth / 2) * force[2] - (z - width / 2) * force[1];
      }
      ratios.twist = torque / ((shear * rectangleTorsion(depth, width).constant * angle) / length);
      t.diagnostic(Object.entries(ratios).map(([k, v]) => `${k} ${v.toFixed(5)}`).join(", "));
      for (const [what, ratio] of Object.entries(ratios)) assertRatio(ratio, 1, CRITERIA.displacement, what);
    });

// A curved beam (the quarter ring of MacNeal and Harder: radii 4.12 and 4.32, thickness 0.1, E = 1e7, six elements
// along the arc) clamped at one end and loaded at the other by a radial force P, with ν = 0: the plane solution of
// Timoshenko and Goodier (Theory of Elasticity, section 33) is then the solution in space. With
// N = a² − b² + (a² + b²) ln(b / a) and P the force per unit thickness, the loaded end moves radially by
// π P (a² + b²) / (E N) and the hoop stress at the angle θ from it is ∓(P / N) (3 r − a² b² / r³ − (a² + b²) / r) sin θ
// (a force towards the outside stretches the inner fibre). Read at θ = 45 degrees.
const CURVED_BEAM = { a: 4.12, b: 4.32, h: 0.1, young: 1.0e7, poisson: 0.0 };
for (const shape of SHAPES)
  for (const quadratic of [true, false])
    test(`beam: a quarter ring of ${shape} ${quadratic ? "with mid-side nodes on the arcs" :
      "with straight edges, its skin read from the grid points"} under a radial end force`, (t) => {
      const { a, b, h, young } = CURVED_BEAM;
      const force = 1.0;
      const mesh = ringMesh({ a, b, h, n: [1, 6, 1], shape, quadratic });
      const records = [...isotropic(CURVED_BEAM)];
      for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};123;0.`);
      // The end on the plane y = 0, pulled along +x (the radial direction there).
      for (const face of skinFaces(mesh).filter((f) => f.x.every((p) => near(p[1], 0))))
        records.push(facePressure(2, face, force / ((b - a) * h), [1, 0, 0]));
      const run = runMixed(mesh, staticCase([2]), records);
      const n = a * a - b * b + (a * a + b * b) * Math.log(b / a);
      const perThickness = force / h;
      const expected = (Math.PI * perThickness * (a * a + b * b)) / (young * n);
      const moved = meanDisplacement(run, mesh.where((x, y) => near(y, 0)))[0];
      const hoop = (r) => (-perThickness / n) * (3 * r - (a * a * b * b) / r ** 3 - (a * a + b * b) / r) *
        Math.SQRT1_2;
      let stressError = 0;
      for (const row of rowsOf(run, "SolidGridPoints")) {
        const [x, y] = mesh.points.get(row.id);
        if (!near(x, y, 1e-6)) continue;
        // σθ = σxx sin² θ + σyy cos² θ − 2 τxy sin θ cos θ at θ = 45 degrees.
        const value = row.oxx / 2 + row.oyy / 2 - row.txy;
        stressError = Math.max(stressError, Math.abs(value - hoop(Math.hypot(x, y))));
      }
      t.diagnostic(`end displacement ratio ${(moved / expected).toFixed(5)} (expected ${expected.toExponential(5)}), ` +
        `hoop stress error ${(stressError / Math.abs(hoop(a))).toExponential(2)}`);
      assertRatio(moved, expected, CRITERIA.displacement, "radial displacement of the loaded end");
      assertSmall(stressError, Math.abs(hoop(a)), CRITERIA.stress, "hoop stress at 45 degrees");
    });

// ---------------------------------------------------------------------------------------------------------------------
// 3. Thick cylinder and thick sphere under internal pressure (Lamé; Timoshenko and Goodier, Theory of Elasticity,
// sections 28 and 136).

// The six components (xx, yy, zz, xy, yz, zx) of Σ s_k d_k ⊗ d_k for the values s and the unit directions d.
function tensorOf(values, directions) {
  const pairs = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]];
  return pairs.map(([i, j]) => values.reduce((sum, s, k) => sum + s * directions[k][i] * directions[k][j], 0));
}

// The thick cylinder of radii a and b under the internal pressure p, with A = p a² / (b² − a²):
// σr = A (1 − b² / r²), σθ = A (1 + b² / r²);
// - free ends (σz = 0): u_r = (r / E) ((1 − ν) A + (1 + ν) A b² / r²), u_z = −2 ν A z / E;
// - plane strain (u_z = 0): u_r = ((1 + ν) / E) ((1 − 2 ν) A r + A b² / r), σz = 2 ν A.
function lameCylinder({ a, b, pressure, young, poisson, planeStrain }) {
  const factor = (pressure * a * a) / (b * b - a * a);
  const radial = (r) => factor * (1 - (b * b) / (r * r));
  const hoop = (r) => factor * (1 + (b * b) / (r * r));
  const axial = planeStrain ? 2 * poisson * factor : 0;
  const displacement = planeStrain
    ? (r) => ((1 + poisson) / young) * ((1 - 2 * poisson) * factor * r + (factor * b * b) / r)
    : (r) => (r / young) * ((1 - poisson) * factor + ((1 + poisson) * factor * b * b) / (r * r));
  return {
    hoopAtBore: hoop(a),
    displacement,
    field: (x, y, z) => {
      const r = Math.hypot(x, y);
      const w = planeStrain ? 0 : (-2 * poisson * factor * z) / young;
      return [(displacement(r) * x) / r, (displacement(r) * y) / r, w];
    },
    stress: (x, y) => {
      const r = Math.hypot(x, y);
      return tensorOf([radial(r), hoop(r), axial], [[x / r, y / r, 0], [-y / r, x / r, 0], [0, 0, 1]]);
    },
  };
}

// The largest difference between the translations of the corner grid points of `mesh` and `field`, in the basic
// system (the rows of the table are in the displacement systems `frameOf(x, y, z)`: three unit axes, or nothing).
function displacementError(run, mesh, field, frameOf = () => null) {
  let worst = 0;
  for (const id of mesh.cornerIds()) {
    const p = mesh.points.get(id);
    const local = nodal(run, "displacements", id);
    const axes = frameOf(...p);
    const u = axes ? [0, 1, 2].map((c) => local[0] * axes[0][c] + local[1] * axes[1][c] + local[2] * axes[2][c])
      : local;
    field(...p).forEach((value, c) => { worst = Math.max(worst, Math.abs(u[c] - value)); });
  }
  return worst;
}

const TUBE = { a: 1.0, b: 2.0, h: 0.5, pressure: 1.0e6, young: 2.0e11, poisson: 0.3 };

// A quarter of the tube between its planes of symmetry x = 0 and y = 0, on the plane z = 0, its other end free;
// two cells through the wall, four of 22.5 degrees along the arc.
for (const shape of SHAPES)
  for (const quadratic of [true, false])
    test(`Lamé: a quarter of a thick cylinder of ${shape} ${quadratic ? "with mid-side nodes on the circles" :
      "with straight edges, its skin read from the grid points"} under internal pressure, free ends`, (t) => {
      const { a, b, h, pressure } = TUBE;
      const mesh = ringMesh({ a, b, h, n: [2, 4, 1], shape, quadratic });
      const records = [...isotropic(TUBE)];
      for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
      for (const id of mesh.where((x, y) => near(y, 0))) records.push(`SinglePointConstr;1;${id};2;0.`);
      for (const id of mesh.where((x, y, z) => near(z, 0))) records.push(`SinglePointConstr;1;${id};3;0.`);
      const bore = skinFaces(mesh).filter((face) => face.x.every((p) => near(Math.hypot(p[0], p[1]), a)));
      for (const face of bore) records.push(facePressure(2, face, pressure));
      const run = runMixed(mesh, staticCase([2]), records);
      const exact = lameCylinder({ ...TUBE, planeStrain: false });
      const moved = displacementError(run, mesh, exact.field) / exact.displacement(a);
      const errors = stressErrors(run, mesh, exact.stress);
      t.diagnostic(`displacement error ${moved.toExponential(2)}, stress error at the grid points ` +
        `${(errors.grid / exact.hoopAtBore).toExponential(2)}, at the corners of the elements ` +
        `${(errors.corner / exact.hoopAtBore).toExponential(2)}`);
      assertSmall(moved, 1, CRITERIA.displacement, "displacements");
      assertSmall(errors.grid, exact.hoopAtBore, CRITERIA.stress, "grid point stresses");
    });

// A sector of 30 degrees of the tube in plane strain, its grid points in a cylindrical displacement system: the
// hoop component held on the two planes that cut it, the axial one on both ends. The translations of the table are
// then radial, hoop and axial.
test("Lamé: a sector of a thick cylinder of SolidHexa in a cylindrical displacement system, plane strain", (t) => {
  const { a, b, h, pressure } = TUBE;
  const angle = Math.PI / 6;
  const mesh = ringMesh({ a, b, h, n: [2, 2, 1], quadratic: true, angle });
  const records = [...mesh.records().map((record) => (record.startsWith("Node;") ? `${record};1` : record)),
    "CylindricalFrame2;1;;0.;0.;0.;0.;0.;1.;1.;0.;0.", ...isotropic(TUBE)];
  const azimuth = (x, y) => Math.atan2(y, x);
  for (const id of mesh.where((x, y) => near(azimuth(x, y), 0) || near(azimuth(x, y), angle)))
    records.push(`SinglePointConstr;1;${id};2;0.`);
  for (const id of mesh.where((x, y, z) => near(z, 0) || near(z, h))) records.push(`SinglePointConstr;1;${id};3;0.`);
  const bore = skinFaces(mesh).filter((face) => face.x.every((p) => near(Math.hypot(p[0], p[1]), a)));
  for (const face of bore) records.push(facePressure(2, face, pressure));
  const run = runModel(deck(101, staticCase([2]), records), MIXED);
  // The hoop direction, the same at every grid point of each plane, is held along the edges and faces of the planes
  // (spec section 5.6: the directions common to the grid points). This case found that only the directions of the
  // basic system were (displacements 8 times the exact ones).
  const warnings = run.diagnostics.filter((d) => d.severity === "warning");
  t.diagnostic(`warnings: ${warnings.map((d) => d.code).join(", ") || "none"}`);
  const exact = lameCylinder({ ...TUBE, planeStrain: true });
  const cylindrical = (x, y) => {
    const r = Math.hypot(x, y);
    return [[x / r, y / r, 0], [-y / r, x / r, 0], [0, 0, 1]];
  };
  const moved = displacementError(run, mesh, exact.field, cylindrical) / exact.displacement(a);
  const errors = stressErrors(run, mesh, exact.stress);
  t.diagnostic(`displacement error ${moved.toExponential(2)}, stress error at the grid points ` +
    `${(errors.grid / exact.hoopAtBore).toExponential(2)}, at the corners ` +
    `${(errors.corner / exact.hoopAtBore).toExponential(2)}`);
  assertSmall(moved, 1, CRITERIA.displacement, "displacements");
  assertSmall(errors.grid, exact.hoopAtBore, CRITERIA.stress, "grid point stresses");
});

// The thick sphere of radii a and b under the internal pressure p, with A = p a³ / (b³ − a³):
// σr = A (1 − b³ / r³), σθ = A (1 + b³ / (2 r³)), u_r = (A / E) ((1 − 2 ν) r + (1 + ν) b³ / (2 r²)).
// An octant between the planes x = 0, y = 0 and z = 0, in three blocks (the faces of a cube thrown on the sphere,
// at equal angles), three cells through the wall and two of 22.5 degrees along each side of a block.
const BALL = { a: 1.0, b: 2.0, pressure: 1.0e6, young: 2.0e11, poisson: 0.3 };
function octantMesh({ a, b, n, shape, quadratic }) {
  const radius = (x, y, z) => Math.hypot(x, y, z);
  const surfaces = [(x, y, z) => near(radius(x, y, z), a), (x, y, z) => near(radius(x, y, z), b),
    (x) => near(x, 0), (x, y) => near(y, 0), (x, y, z) => near(z, 0)];
  const mesh = createMesh({ surfaces, quadratic });
  for (let block = 0; block < 3; ++block)
    mesh.addBlock({ n, shape, map: (u, v, w) => {
      const s = Math.tan((Math.PI / 4) * u);
      const t = Math.tan((Math.PI / 4) * v);
      const r = (a + (b - a) * w) / Math.hypot(1, s, t);
      // (1, s, t), (t, 1, s), (s, t, 1): the three faces of the cube, each turned the same way.
      const d = [[1, s, t], [t, 1, s], [s, t, 1]][block];
      return d.map((value) => value * r);
    } });
  return mesh;
}
// The stress criterion of each case is that of section 13.6 but for the two meshes whose skin is made of triangles:
// KNOWN LIMIT (spec section 3.6: triangles read on a sphere keep kinks between them): the stresses at the grid points
// of the bore are then 1.6 % (tetrahedra; 0.8 % with three and with four cells of the same wall along each side) and
// 4.5 % (pentahedra) off, the displacements within the criterion.
// With mid-side nodes the faces of the hexahedra are the surfaces of their eight grid points and not the sphere:
// the stress is 4.7 % off where the three blocks meet on the bore, whatever the cells through the wall (1.9 % with
// three cells of 15 degrees along each side, 1.0 % with four of 11.25 degrees); the skin read from the corner grid
// points alone gives 0.3 % on the same mesh. This case is left failing: the limit is not in section 3.6.
const BALL_CASES = [
  { shape: "SolidHexa", quadratic: false, stress: CRITERIA.stress },
  { shape: "SolidHexa", quadratic: true, stress: CRITERIA.stress },
  { shape: "SolidTetra", quadratic: false, stress: 2.0e-2 },
  { shape: "SolidPenta", quadratic: false, stress: 5.0e-2 },
];
for (const { shape, quadratic, stress: stressCriterion } of BALL_CASES)
  test(`Lamé: an octant of a thick sphere of ${shape} ${quadratic ? "with mid-side nodes on the spheres" :
    "with straight edges, its skin read from the grid points"} under internal pressure`, (t) => {
    const { a, b, pressure, young, poisson } = BALL;
    const mesh = octantMesh({ a, b, n: [2, 2, 3], shape, quadratic });
    const records = [...isotropic(BALL)];
    for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
    for (const id of mesh.where((x, y) => near(y, 0))) records.push(`SinglePointConstr;1;${id};2;0.`);
    for (const id of mesh.where((x, y, z) => near(z, 0))) records.push(`SinglePointConstr;1;${id};3;0.`);
    const bore = skinFaces(mesh).filter((face) => face.x.every((p) => near(Math.hypot(...p), a)));
    for (const face of bore) records.push(facePressure(2, face, pressure));
    const run = runMixed(mesh, staticCase([2]), records);
    const factor = (pressure * a ** 3) / (b ** 3 - a ** 3);
    const radial = (r) => (factor / young) * ((1 - 2 * poisson) * r + ((1 + poisson) * b ** 3) / (2 * r * r));
    const field = (x, y, z) => [x, y, z].map((value) => (radial(Math.hypot(x, y, z)) * value) / Math.hypot(x, y, z));
    const stress = (x, y, z) => {
      const r = Math.hypot(x, y, z);
      const hoop = factor * (1 + b ** 3 / (2 * r ** 3));
      const n = [x / r, y / r, z / r];
      // σ = σθ I + (σr − σθ) n ⊗ n
      const along = tensorOf([factor * (1 - b ** 3 / r ** 3) - hoop], [n]);
      return along.map((value, c) => value + (c < 3 ? hoop : 0));
    };
    const hoopAtBore = factor * (1 + b ** 3 / (2 * a ** 3));
    const moved = displacementError(run, mesh, field) / radial(a);
    const errors = stressErrors(run, mesh, stress);
    t.diagnostic(`displacement error ${moved.toExponential(2)}, stress error at the grid points ` +
      `${(errors.grid / hoopAtBore).toExponential(2)}, at the corners of the elements ` +
      `${(errors.corner / hoopAtBore).toExponential(2)}`);
    assertSmall(moved, 1, CRITERIA.displacement, "displacements");
    assertSmall(errors.grid, hoopAtBore, stressCriterion, "grid point stresses");
  });

// ---------------------------------------------------------------------------------------------------------------------
// 4. Loads.

// One distorted element of each shape (the corners of a unit cell moved).
const SINGLE_ELEMENTS = {
  SolidTetra: [[0, 0, 0], [1.1, 0.1, 0], [0.2, 0.9, 0.1], [0.3, 0.2, 1.0]],
  SolidPyramid: [[0, 0, 0], [1.1, 0.1, 0], [1.0, 1.2, 0.1], [-0.1, 0.9, 0], [0.4, 0.6, 0.9]],
  SolidPenta: [[0, 0, 0], [1.1, 0.1, 0], [0.2, 0.9, 0.1], [0.1, 0, 1.0], [1.0, 0.2, 1.1], [0.3, 1.0, 0.9]],
  SolidHexa: [[0, 0, 0], [1.1, 0.1, 0], [1.0, 1.2, 0.1], [-0.1, 0.9, 0], [0.1, 0, 1.0], [1.0, 0.2, 1.1],
    [1.2, 1.1, 0.9], [0, 1.0, 1.0]],
};

// A pressure on every face of one element, of every face type (the triangles and quadrilaterals of the four shapes):
// the state is hydrostatic, σ = −p I, u = −p (1 − 2 ν) x / E. A face loaded on the wrong side, or a wrong face,
// breaks it. A triangle of a pyramid is named by the two corners of its edge on the base (this case found that
// O_o_Reader took the first face holding the two grid points, which named no triangle of a pyramid for sure).
for (const shape of SHAPES)
  for (const quadratic of [false, true])
    for (const card of ["FacePressureLoad", "PointsPressureLoad"])
      test(`loads: a ${card} on every face of one distorted ${shape}${quadratic ? " with mid-side nodes" : ""}`,
        (t) => {
          const pressure = 1.0e6;
          const mesh = createMesh({ quadratic }).addElement(shape, SINGLE_ELEMENTS[shape]);
          const [g1, g2, g3] = mesh.elements[0].corners;
          const records = [...isotropic(STEEL), ...isostatic(g1, g2, g3)];
          for (const face of skinFaces(mesh))
            records.push(card === "FacePressureLoad" ? facePressure(2, face, pressure)
              : pointsPressure(2, face, pressure));
          // One element: every edge of it is a sharp edge of its skin.
          const model = [...mesh.records({ sharpEdges: mesh.edges() }), ...records];
          const run = runModel(deck(101, staticCase([2]), model), MIXED);
          const strain = (-pressure * (1 - 2 * STEEL.poisson)) / STEEL.young;
          const moved = fieldError(run, mesh, (x, y, z) => [strain * x, strain * y, strain * z]) / Math.abs(strain);
          let stressError = 0;
          for (const type of [shape, "SolidGridPoints"])
            for (const row of rowsOf(run, type))
              stressError = Math.max(stressError,
                ...sixOf(row).map((v, c) => Math.abs(v - (c < 3 ? -pressure : 0)) / pressure));
          const forces = resultant(run, mesh, "singlePointConstrForces").map((v) => Math.abs(v) / pressure);
          t.diagnostic(`displacement error ${moved.toExponential(2)}, stress error ${stressError.toExponential(2)}, ` +
            `support forces ${Math.max(...forces).toExponential(2)}`);
          assertSmall(Math.max(...forces), 1, CRITERIA.exact, "support forces (the load is in equilibrium)");
          assertSmall(moved, 1, CRITERIA.exact, "displacements");
          assertSmall(stressError, 1, CRITERIA.exact, "stresses");
        });

// A column hanging under its weight (Timoshenko and Goodier, Theory of Elasticity, section 86): held by a uniform
// traction ρ g L on its top, σz = ρ g z and nothing else; u = −ν ρ g x z / E, v = −ν ρ g y z / E,
// w = ρ g (z² + ν (x² + y²)) / (2 E). The supports carry no force, the traction carries the weight ρ g V.
const COLUMN = { a: 1.0, b: 1.5, length: 2.0 };
function hangingColumn(shape) {
  const { a, b, length } = COLUMN;
  const { young, poisson, density } = STEEL;
  const gravity = 9.81;
  const mesh = createMesh({ surfaces: boxSurfaces(a, b, length) }).addBlock({ n: [2, 2, 2], shape, distortion: 0.2,
    distortsSkin: false, map: (u, v, w) => [a * u, b * v, length * w] });
  const records = [...isotropic(STEEL), ...isostatic(mesh.at(0, 0, 0), mesh.at(a, 0, 0), mesh.at(0, b, 0))];
  const top = skinFaces(mesh).filter((face) => face.x.every((p) => near(p[2], length)));
  for (const face of top) records.push(facePressure(2, face, -density * gravity * length));
  records.push(`GravityLoad;3;;${real(gravity)};0.;0.;-1.`, "LoadCombination;10;1.;1.;2;1.;3");
  // Subcase 1: the weight and the traction; subcase 2: the weight alone on the supports.
  const run = runMixed(mesh, staticCase([10, 3]), records);
  const k = (density * gravity) / young;
  return { mesh, run, scale: density * gravity * length,
    field: (x, y, z) => [-poisson * k * x * z, -poisson * k * y * z, (k * (z * z + poisson * (x * x + y * y))) / 2],
    stress: (x, y, z) => [0, 0, density * gravity * z, 0, 0, 0],
    weight: density * gravity * a * b * length, deflection: (k * length * length) / 2 };
}
for (const shape of SHAPES)
  test(`loads: a column of ${shape} hanging under its weight (GravityLoad): displacements and weight`, (t) => {
    const column = hangingColumn(shape);
    const moved = fieldError(column.run, column.mesh, column.field) / column.deflection;
    const forces = resultant(column.run, column.mesh, "singlePointConstrForces", 1).map((v) => v / column.weight);
    const carried = resultant(column.run, column.mesh, "singlePointConstrForces", 2)[2] / column.weight;
    t.diagnostic(`displacement error ${moved.toExponential(2)}, support forces under the traction ` +
      `${Math.max(...forces.map(Math.abs)).toExponential(2)}, weight ratio ${carried.toFixed(10)}`);
    assertSmall(moved, 1, CRITERIA.exact, "displacements");
    assertSmall(Math.max(...forces.map(Math.abs)), 1, CRITERIA.invariant, "support forces under the traction");
    assertRatio(carried, 1, CRITERIA.invariant, "the weight carried by the supports");
  });

// The stresses of the same column: the recovery adds the particular stresses of the body force to those of the
// unknowns (spec section 10.3). Without them (what this case found) the stresses were those of the same displacements
// without the weight inside each element: 0.2 % (centres of the hexahedra) to 3 % (corners) of ρ g L on this mesh.
for (const shape of ["SolidTetra", "SolidHexa"])
  test(`loads: a column of ${shape} hanging under its weight (GravityLoad): stresses`, (t) => {
    const column = hangingColumn(shape);
    const errors = stressErrors(column.run, column.mesh, column.stress, { centres: true });
    const shares = ["corner", "centre", "grid"].map((where) => (errors[where] / column.scale).toExponential(2));
    t.diagnostic(`stress errors: corners ${shares[0]}, centres ${shares[1]}, grid points ${shares[2]}`);
    for (const where of ["centre", "grid", "corner"])
      assertSmall(errors[where], column.scale, CRITERIA.exact, `${where} stresses`);
  });

// A bar spinning about an axis across its end (RotationalForceLoad), with ν = 0: the body force ρ ω² (x, y', 0)
// gives σxx = ρ ω² (L² − x²) / 2 and σyy = ρ ω² (h² / 4 − y'²) / 2, u = ρ ω² (L² x / 2 − x³ / 6) / E,
// v = ρ ω² (h² y' / 8 − y'³ / 6) / E (x along the bar from the axis, y' across it from the axis; the axis is z).
// The stresses are quadratic: within the degree of the element, with the particular stresses of the body force.
test("loads: a bar of SolidHexa spinning about an axis across its end (RotationalForceLoad)", (t) => {
  const length = 4.0;
  const depth = 1.0;
  const width = 0.5;
  const material = { young: 2.0e11, poisson: 0.0, density: 7800.0 };
  const turns = 10.0; // revolutions per unit of time: ω = 2 π turns
  const omega2 = (2 * Math.PI * turns) ** 2;
  const mesh = beamMesh({ length, depth, width, n: [4, 2, 1] });
  const axis = mesh.at(0, depth / 2, 0);
  const records = [...isotropic(material), `RotationalForceLoad;2;${axis};;${real(turns)};0.;0.;1.`,
    `SinglePointConstr;1;${axis};23;0.`, `SinglePointConstr;1;${mesh.at(0, 0, 0)};3;0.`];
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
  const run = runMixed(mesh, staticCase([2]), records);
  const k = material.density * omega2;
  const field = (x, y) => {
    const yc = y - depth / 2;
    return [(k * ((length * length * x) / 2 - x ** 3 / 6)) / material.young,
      (k * ((depth * depth * yc) / 8 - yc ** 3 / 6)) / material.young, 0];
  };
  const stress = (x, y) => [(k * (length * length - x * x)) / 2,
    (k * (depth * depth / 4 - (y - depth / 2) ** 2)) / 2, 0, 0, 0, 0];
  const largest = (k * length * length) / 2;
  const moved = fieldError(run, mesh, field) / field(length, depth / 2)[0];
  const errors = stressErrors(run, mesh, stress, { centres: true });
  // The root carries the centrifugal force of the bar: ρ ω² (L² / 2) × section.
  const pull = -resultant(run, mesh, "singlePointConstrForces")[0] / (largest * depth * width);
  t.diagnostic(`displacement error ${moved.toExponential(2)}, root force ratio ${pull.toFixed(8)}, stress errors: ` +
    `corners ${(errors.corner / largest).toExponential(2)}, centres ${(errors.centre / largest).toExponential(2)}, ` +
    `grid points ${(errors.grid / largest).toExponential(2)}`);
  assertRatio(pull, 1, CRITERIA.exact, "force at the root");
  assertSmall(moved, 1, CRITERIA.exact, "displacements");
  for (const where of ["centre", "grid", "corner"])
    assertSmall(errors[where], largest, CRITERIA.exact, `${where} stresses`);
});

// Temperature loads on a distorted block of two cells (expansion α, reference temperature 0):
// - a uniform rise ΔT of the free block: u = α ΔT x, no stress;
// - the same rise between two sliding planes x = 0 and x = a: σxx = −E α ΔT, the block free across
//   (v = (1 + ν) α ΔT y, w = (1 + ν) α ΔT z);
// - a temperature c y linear across the free block: no stress, u = α c x y, v = α c (y² − x² − z²) / 2, w = α c y z.
const THERMAL_CASE = ["DisplacementOutput;All", "SinglePointConstrForceOutput;All", "StressOutput;All",
  "SubcaseStart;1", "SinglePointConstrSelection;1", "TemperatureSelection;7;TemperatureLoadPart",
  "SubcaseStart;2", "SinglePointConstrSelection;2", "TemperatureSelection;7;TemperatureLoadPart",
  "SubcaseStart;3", "SinglePointConstrSelection;1", "TemperatureSelection;8;TemperatureLoadPart"];
for (const meshCase of PATCH_MESHES.slice(0, 4))
  for (const geometry of GEOMETRIES)
    test(`loads: temperatures on a distorted block of ${meshCase.name}, ${geometry.name}`, (t) => {
      const { young, poisson, expansion } = STEEL;
      const rise = 100.0;
      const slope = 80.0;
      const mesh = patchMesh({ ...meshCase, n: [2, 1, 1] }, geometry);
      const records = [...isotropic(STEEL), ...blockSupports(mesh), `DefaultTemperatureLoad;7;${real(rise)}`];
      for (const id of mesh.where((x) => near(x, 0) || near(x, BLOCK.a)))
        records.push(`SinglePointConstr;2;${id};1;0.`);
      records.push(`SinglePointConstr;2;${mesh.at(0, 0, 0)};23;0.`,
        `SinglePointConstr;2;${mesh.at(0, BLOCK.b, 0)};3;0.`);
      for (const [id, p] of mesh.points) records.push(`NodeTemperatureLoad;8;${id};${real(slope * p[1])}`);
      const run = runMixed(mesh, THERMAL_CASE, records);
      const free = expansion * rise;
      const held = -young * free;
      const cases = [
        { name: "free rise", field: (x, y, z) => [free * x, free * y, free * z], stress: () => [0, 0, 0, 0, 0, 0],
          scale: free * BLOCK.a },
        { name: "rise between two planes", field: (x, y, z) => [0, (1 + poisson) * free * y, (1 + poisson) * free * z],
          stress: () => [held, 0, 0, 0, 0, 0], scale: free * BLOCK.a },
        { name: "linear temperature", field: (x, y, z) => [expansion * slope * x * y,
          (expansion * slope * (y * y - x * x - z * z)) / 2, expansion * slope * y * z],
        stress: () => [0, 0, 0, 0, 0, 0], scale: expansion * slope * BLOCK.a * BLOCK.a },
      ];
      const lines = [];
      const checks = [];
      cases.forEach(({ name, field, stress, scale }, c) => {
        const moved = fieldError(run, mesh, field, c + 1) / scale;
        const errors = stressErrors(run, mesh, stress, { subcase: c + 1 });
        const worst = Math.max(errors.corner, errors.grid) / Math.abs(held);
        lines.push(`${name}: displacement ${moved.toExponential(1)}, stress ${worst.toExponential(1)}`);
        checks.push([moved, `displacements, ${name}`], [worst, `stresses, ${name}`]);
      });
      t.diagnostic(lines.join("; "));
      for (const [error, what] of checks) assertSmall(error, 1, CRITERIA.exact, what);
    });

// A block turned by 30 degrees about z, its grid points in a rectangular displacement system turned with it, stretched
// along its own axis between two planes held in that system (component 1 of the grid points of its two end faces):
// the uniaxial state, u' = ε x' in the turned axes. Spec section 5.6: a common rectangular displacement system is
// carried along the edges as the basic one. (This case found that only the directions of the basic system were:
// 14 % of the stretch.)
test("loads: supports in a rectangular displacement system turned with a block of SolidHexa", (t) => {
  const angle = Math.PI / 6;
  const [c, s] = [Math.cos(angle), Math.sin(angle)];
  const turn = ([x, y, z]) => [c * x - s * y, s * x + c * y, z];
  const strain = 1.0e-3;
  const { poisson } = STEEL;
  const mesh = createMesh({}).addBlock({ n: [2, 2, 2], map: (u, v, w) => turn(blockMap(u, v, w)) });
  const local = new Map([...mesh.points].map(([id, p]) => [id, [c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2]]]));
  const onLocal = (predicate) => [...local].filter(([, p]) => predicate(...p)).map(([id]) => id);
  // The edges of the block in the turned axes are its sharp edges.
  const surfaces = boxSurfaces(BLOCK.a, BLOCK.b, BLOCK.c);
  const sharpEdges = mesh.edges().filter(([p, q]) =>
    surfaces.filter((surface) => surface(...local.get(p)) && surface(...local.get(q))).length >= 2);
  const records = [
    ...mesh.records({ sharpEdges }).map((record) => (record.startsWith("Node;") ? `${record};1` : record)),
    `RectFrame2;1;;0.;0.;0.;0.;0.;1.;${real(c)};${real(s)};0.`, ...isotropic(STEEL)];
  for (const id of onLocal((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
  for (const id of onLocal((x) => near(x, BLOCK.a)))
    records.push(`SinglePointConstr;1;${id};1;${real(strain * BLOCK.a)}`);
  records.push(`SinglePointConstr;1;${onLocal((x, y, z) => near(x, 0) && near(y, 0) && near(z, 0))[0]};23;0.`,
    `SinglePointConstr;1;${onLocal((x, y, z) => near(x, 0) && near(y, BLOCK.b) && near(z, 0))[0]};3;0.`);
  const run = runModel(deck(101, staticCase([null]), records), MIXED);
  const warnings = run.diagnostics.filter((d) => d.severity === "warning").map((d) => d.code);
  let worst = 0;
  for (const [id, p] of local) {
    const u = nodal(run, "displacements", id); // in the turned system
    [strain * p[0], -poisson * strain * p[1], -poisson * strain * p[2]].forEach((value, k) => {
      worst = Math.max(worst, Math.abs(u[k] - value));
    });
  }
  t.diagnostic(`displacement error ${(worst / (strain * BLOCK.a)).toExponential(2)}, warnings: ` +
    `${warnings.join(", ") || "none"}`);
  assertSmall(worst, strain * BLOCK.a, CRITERIA.exact, "displacements");
  assert.equal(warnings.length, 0, `warnings: ${warnings.join(", ")}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// 5. Materials.

// An anisotropic law with its 21 constants (xx, yy, zz, xy, yz, zx; engineering shear strains), positive definite:
// an isotropic law (E = 2e11, ν = 0.3) made stiffer along x and softer along z, with couplings between the direct and
// the shear components.
const ANISOTROPIC = (() => {
  const lambda = 1.1538e11;
  const shear = 0.7692e11;
  const c = Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) =>
    (i < 3 && j < 3 ? lambda : 0) + (i === j ? (i < 3 ? 2 * shear : shear) : 0)));
  c[0][0] *= 1.8;
  c[2][2] *= 0.7;
  c[4][4] *= 0.6;
  const couplings = [[0, 3, 0.20], [0, 4, -0.10], [0, 5, 0.15], [1, 3, -0.12], [1, 4, 0.18], [1, 5, 0.05],
    [2, 3, 0.08], [2, 4, -0.15], [2, 5, 0.10], [3, 4, 0.06], [3, 5, -0.09], [4, 5, 0.07], [0, 1, 0.1], [1, 2, -0.1]];
  for (const [i, j, share] of couplings) {
    c[i][j] += share * shear;
    c[j][i] = c[i][j];
  }
  return c;
})();
// The AnisoMat3D record of the law `c`: its upper triangle row by row, then the density. An IsoMat that nothing uses
// goes with it: a deck whose only materials are AnisoMat3D is refused when the model is built, in either formulation
// (the expansion column of the table of the materials then arrives as a list of six: "table 'materialsTable',
// column 'a': expected the type float64").
function anisotropicRecords(c, density = 7800.0) {
  const constants = [];
  for (let i = 0; i < 6; ++i) for (let j = i; j < 6; ++j) constants.push(real(c[i][j]));
  return [`AnisoMat3D;1;${constants.join(";")};${real(density)}`, "IsoMat;2;2.e11;;0.3;7800."];
}
// The strains of the law `c` under `stress`: ε = C⁻¹ σ.
const anisotropicStrain = (c, stress) => solveLinear(c.map((row, i) => [...row, stress[i]]));

// The components of the symmetric tensor `s` (xx, yy, zz, xy, yz, zx) in the axes `axes` (three unit vectors):
// s'_ij = e_i · s e_j. `engineering` doubles the shear components before and halves them after (strains).
function inAxes(s, axes, engineering = false) {
  const h = engineering ? 0.5 : 1;
  const m = [[s[0], h * s[3], h * s[5]], [h * s[3], s[1], h * s[4]], [h * s[5], h * s[4], s[2]]];
  const component = (i, j) => {
    let sum = 0;
    for (let a = 0; a < 3; ++a) for (let b = 0; b < 3; ++b) sum += axes[i][a] * m[a][b] * axes[j][b];
    return sum;
  };
  return [component(0, 0), component(1, 1), component(2, 2), component(0, 1) / h, component(1, 2) / h,
    component(2, 0) / h];
}
const transposed = (m) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => m[j][i]));

// A general uniform stress on a distorted block of an anisotropic material (AnisoMat3D): the displacements are those
// of the strain C⁻¹ σ, the stresses σ. Then the same with the material axes turned (SolidProp CORDM): the law acts
// in the axes of the frame, σ' = R σ Rᵀ, ε' = C⁻¹ σ', ε = Rᵀ ε' R; the rows of the elements are σ' (in CORDM), those
// of the grid points σ (basic system).
const GENERAL_STRESS = [1.0e8, -0.4e8, 0.6e8, 0.5e8, -0.3e8, 0.7e8];
for (const turned of [false, true])
  for (const meshCase of [PATCH_MESHES[0], PATCH_MESHES[5]])
    test(`materials: a uniform stress on a distorted block of anisotropic ${meshCase.name}` +
      `${turned ? ", the material axes turned" : ""}`, (t) => {
      const mesh = patchMesh(meshCase, GEOMETRIES[0]);
      // The material axes: the basic ones turned by 40 degrees about (1, 1, 1) (Rodrigues' formula).
      const [c, s] = [Math.cos((40 * Math.PI) / 180), Math.sin((40 * Math.PI) / 180)];
      const k = (1 - c) / 3;
      const q = s / Math.sqrt(3);
      const axes = turned ? [[c + k, k + q, k - q], [k - q, c + k, k + q], [k + q, k - q, c + k]]
        : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const records = [...anisotropicRecords(ANISOTROPIC), ...blockSupports(mesh),
        ...tractionRecords(2, skinFaces(mesh), GENERAL_STRESS)];
      if (turned)
        // The frame: its origin, a point on its z axis, a point on its x axis.
        records.push("SolidProp;1;1;5", `RectFrame2;5;;0.;0.;0.;${axes[2].map(real).join(";")};` +
          `${axes[0].map(real).join(";")}`);
      else records.push("SolidProp;1;1");
      const run = runMixed(mesh, staticCase([2]), records);
      const inMaterial = inAxes(GENERAL_STRESS, axes);
      const strain = inAxes(anisotropicStrain(ANISOTROPIC, inMaterial), transposed(axes), true);
      const scale = Math.max(...strain.map(Math.abs)) * BLOCK.a;
      const moved = fieldError(run, mesh, strainField(strain)) / scale;
      let elementError = 0;
      for (const card of mesh.cards())
        for (const row of rowsOf(run, card))
          elementError = Math.max(elementError, ...sixOf(row).map((v, i) => Math.abs(v - inMaterial[i])));
      let gridError = 0;
      for (const row of rowsOf(run, "SolidGridPoints"))
        gridError = Math.max(gridError, ...sixOf(row).map((v, i) => Math.abs(v - GENERAL_STRESS[i])));
      t.diagnostic(`displacement error ${moved.toExponential(2)}, stress errors: elements (material axes) ` +
        `${(elementError / 1e8).toExponential(2)}, grid points (basic system) ${(gridError / 1e8).toExponential(2)}`);
      assertSmall(moved, 1, CRITERIA.exact, "displacements");
      assertSmall(elementError, 1.0e8, CRITERIA.exact, "stresses of the elements in the material axes");
      assertSmall(gridError, 1.0e8, CRITERIA.exact, "stresses of the grid points in the basic system");
    });

// A nearly incompressible material: the thick cylinder of MacNeal and Harder (1985): radii 3 and 9, plane strain,
// E = 1000, unit internal pressure, five elements through the wall at the radii 3, 3.5, 4.2, 5.2, 6.75 and 9, with
// ν = 0.49, 0.499 and 0.4999. The radial displacement of the bore is (1 + ν) p a² (b² / a + (1 − 2 ν) a) /
// (E (b² − a²)): 5.0399e-3, 5.0602e-3, 5.0623e-3. A hybrid stress element does not lock: the same accuracy is
// expected at the three ratios. A quarter of the cylinder, four cells of 22.5 degrees, mid-side nodes on the circles.
for (const poisson of [0.49, 0.499, 0.4999])
  for (const shape of ["SolidHexa", "SolidTetra"])
    test(`materials: the thick cylinder of MacNeal and Harder of ${shape} with ν = ${poisson}`, (t) => {
      const tube = { a: 3.0, b: 9.0, h: 1.0, pressure: 1.0, young: 1000.0, poisson };
      const radii = [3.0, 3.5, 4.2, 5.2, 6.75, 9.0];
      // The share of the wall at the lattice coordinate u (piecewise linear through the radii).
      const grading = (u) => {
        const cell = Math.min(Math.floor(u * 5 + 1e-9), 4);
        const r = radii[cell] + (u * 5 - cell) * (radii[cell + 1] - radii[cell]);
        return (r - tube.a) / (tube.b - tube.a);
      };
      const mesh = ringMesh({ ...tube, n: [5, 4, 1], shape, quadratic: true, grading });
      const records = [...isotropic(tube)];
      for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
      for (const id of mesh.where((x, y) => near(y, 0))) records.push(`SinglePointConstr;1;${id};2;0.`);
      for (const id of mesh.where((x, y, z) => near(z, 0) || near(z, tube.h)))
        records.push(`SinglePointConstr;1;${id};3;0.`);
      const bore = skinFaces(mesh).filter((face) => face.x.every((p) => near(Math.hypot(p[0], p[1]), tube.a)));
      for (const face of bore) records.push(facePressure(2, face, tube.pressure));
      const run = runMixed(mesh, staticCase([2]), records);
      const exact = lameCylinder({ ...tube, planeStrain: true });
      const atBore = nodal(run, "displacements", mesh.at(tube.a, 0, 0))[0];
      const moved = displacementError(run, mesh, exact.field) / exact.displacement(tube.a);
      const errors = stressErrors(run, mesh, exact.stress);
      t.diagnostic(`bore displacement ${atBore.toExponential(5)}, ratio ` +
        `${(atBore / exact.displacement(tube.a)).toFixed(5)}, displacement error ${moved.toExponential(2)}, stress ` +
        `errors: grid points ${(errors.grid / exact.hoopAtBore).toExponential(2)}, corners ` +
        `${(errors.corner / exact.hoopAtBore).toExponential(2)}`);
      assertRatio(atBore, exact.displacement(tube.a), CRITERIA.displacement, "radial displacement of the bore");
      assertSmall(moved, 1, CRITERIA.displacement, "displacements");
      assertSmall(errors.grid, exact.hoopAtBore, CRITERIA.stress, "grid point stresses");
    });

// ---------------------------------------------------------------------------------------------------------------------
// 6. Modes, buckling and mass.

// A cube of side a whose six faces slide on their planes (no normal displacement, no shear traction): its modes are
// standing shear and dilatational waves, ω² = c² π² (l² + m² + n²) / a² with c² = G / ρ (shear) or (λ + 2 G) / ρ
// (dilatation), l, m, n the half-waves along the sides. The eight lowest, in units of π² G / (ρ a²): three shear
// modes at 2, two at 3, three dilatational ones at 2 (1 − ν) / (1 − 2 ν) = 3.5 for ν = 0.3. Three cells per side of
// hexahedra and of pentahedra (the largest error of the eight frequencies of regular hexahedra is 1.0 % with two
// cells per side, 0.23 % with three, 0.08 % with four), two of tetrahedra and of pyramids (six elements per cell).
const SLIDING_CUBE = { side: 1.0, young: 2.0e11, poisson: 0.3, density: 7800.0 };
for (const shape of SHAPES)
  test(`modes: a cube of ${shape} whose faces slide on their planes`, (t) => {
    const { side, young, poisson, density } = SLIDING_CUBE;
    const cells = shape === "SolidHexa" || shape === "SolidPenta" ? 3 : 2;
    const mesh = createMesh({ surfaces: boxSurfaces(side, side, side) }).addBlock({ n: [cells, cells, cells], shape,
      distortion: 0.15, distortsSkin: false, map: (u, v, w) => [side * u, side * v, side * w] });
    const records = [...isotropic(SLIDING_CUBE), "LanczosEigenMethod;3;;;8"];
    [0, 1, 2].forEach((axis) => {
      for (const id of mesh.where((...p) => near(p[axis], 0) || near(p[axis], side)))
        records.push(`SinglePointConstr;1;${id};${axis + 1};0.`);
    });
    const run = runMixed(mesh, ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"], records,
      { solution: 103, analysis: "modes" });
    const unit = (Math.PI ** 2 * young) / (2 * (1 + poisson) * density * side * side);
    const dilatation = (2 * (1 - poisson)) / (1 - 2 * poisson);
    const expected = [2, 2, 2, 3, 3, dilatation, dilatation, dilatation];
    const ratios = run.eigenvalues[0].rows.slice(0, 8)
      .map((row, k) => Math.sqrt(row.eigenvalue / (expected[k] * unit)));
    t.diagnostic(`frequency ratios ${ratios.map((r) => r.toFixed(5)).join(" ")}`);
    ratios.forEach((ratio, k) => assertRatio(ratio, 1, CRITERIA.frequency, `frequency ${k + 1}`));
  });

// A beam of square section a and length L = 10 a between two sliding planes (its end sections stay plane and are free
// to move across): it bends in half-waves cos(n π x / L), an exact solution of the beam of Timoshenko (Vibration
// Problems in Engineering): with k = n π / L and r² = I / A the frequency is the lower root of
// (ρ r² / (κ G)) ω⁴ − (1 + k² r² (1 + E / (κ G))) ω² + E r² k⁴ / ρ = 0, κ = 5/6 for ν = 0. The same beam shortened by
// its ends buckles at Euler's load Pe = π² E I / L² lowered by the shear (Engesser) and by the rotation of the
// stressed sections: the initial stress stiffness ∫ σ : ∇uᵀ ∇v of a solid (spec section 3.3) has the term
// P r² θ'² of the turning sections, which the beam of Euler leaves out: P = Pe / (1 + Pe / (κ G A) + Pe / (E A)).
// At L = 10 a the two corrections are 1.97 % and 0.82 % (against Euler with the shear alone the solids give 0.9922).
const SLIDING_BEAM = { length: 10.0, side: 1.0, young: 2.0e11, poisson: 0.0, density: 7800.0 };
function slidingBeam(shape, more = []) {
  const { length, side } = SLIDING_BEAM;
  const mesh = beamMesh({ length, depth: side, width: side, n: [4, 1, 1], shape });
  const records = [...isotropic(SLIDING_BEAM), ...more];
  for (const id of mesh.where((x) => near(x, 0))) records.push(`SinglePointConstr;1;${id};1;0.`);
  return { mesh, records };
}
for (const shape of SHAPES)
  test(`modes: a beam of ${shape} between two sliding planes bends at the frequency of Timoshenko`, (t) => {
    const { length, side, young, density } = SLIDING_BEAM;
    const { mesh, records } = slidingBeam(shape, ["LanczosEigenMethod;3;;;6"]);
    for (const id of mesh.where((x) => near(x, length))) records.push(`SinglePointConstr;1;${id};1;0.`);
    const run = runMixed(mesh, ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3"], records,
      { solution: 103, analysis: "modes" });
    const values = run.eigenvalues[0].rows.map((row) => row.eigenvalue);
    const k = Math.PI / length;
    const r2 = (side * side) / 12;
    const shear = (5 / 6) * (young / 2);
    const a = (density * r2) / shear;
    const b = 1 + k * k * r2 * (1 + young / shear);
    const c = (young * r2 * k ** 4) / density;
    const expected = (b - Math.sqrt(b * b - 4 * a * c)) / (2 * a);
    // Three rigid motions are left (across the planes, and about the axis), then the two planes of bending.
    const rigid = Math.max(...values.slice(0, 3).map(Math.abs)) / values[3];
    const ratios = [values[3], values[4]].map((value) => Math.sqrt(value / expected));
    t.diagnostic(`rigid eigenvalues ${rigid.toExponential(1)} of the first elastic one, frequency ratios ` +
      `${ratios.map((ratio) => ratio.toFixed(5)).join(" ")}`);
    assert.ok(rigid <= CRITERIA.invariant, `three rigid modes: ${rigid}`);
    ratios.forEach((ratio, i) => assertRatio(ratio, 1, CRITERIA.frequency, `bending frequency ${i + 1}`));
  });

for (const shape of SHAPES)
  test(`buckling: a column of ${shape} between two sliding planes buckles at Euler's load`, (t) => {
    const { length, side, young } = SLIDING_BEAM;
    const shortening = 1.0e-3;
    const { mesh, records } = slidingBeam(shape, ["LanczosEigenMethod;3;;;2"]);
    for (const id of mesh.where((x) => near(x, length)))
      records.push(`SinglePointConstr;1;${id};1;${real(-shortening)}`);
    // The column is held across at one corner of its foot (and about its axis at another corner of the same end).
    records.push(`SinglePointConstr;1;${mesh.at(0, 0, 0)};23;0.`, `SinglePointConstr;1;${mesh.at(0, side, 0)};3;0.`);
    const caseRecords = ["SubcaseStart;1", "SinglePointConstrSelection;1", "SubcaseStart;2",
      "SinglePointConstrSelection;1", "EigenMethodSelection;3", "StaticSubcaseSelection;1"];
    const run = runMixed(mesh, caseRecords, records, { solution: 105, analysis: "buckling" });
    const area = side * side;
    const euler = (Math.PI ** 2 * young * side ** 4) / (12 * length ** 2);
    const critical = euler / (1 + euler / ((5 / 6) * (young / 2) * area) + euler / (young * area));
    const preload = (young * area * shortening) / length;
    const ratios = run.eigenvalues[0].rows.slice(0, 2).map((row) => (row.eigenvalue * preload) / critical);
    t.diagnostic(`critical load ratios ${ratios.map((ratio) => ratio.toFixed(5)).join(" ")}`);
    ratios.forEach((ratio, i) => assertRatio(ratio, 1, CRITERIA.criticalLoad, `critical load ${i + 1}`));
  });

// The mass and its centre, read from the forces of the supports under a unit gravity along each axis: the resultant
// is the weight ρ V g, its moment about the origin x_c × weight. A distorted block of every shape with curved edges
// inside (its skin is the box: the mass is exact), and a quarter ring with mid-side nodes on the circles (the mass of
// its parabolas: within 1e-4 of ρ π (b² − a²) h / 4, its centre at 4 (b³ − a³) / (3 π (b² − a²)) from the axis).
function massProperties(mesh, supports) {
  const records = [...isotropic(STEEL), ...isostatic(...supports)];
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((direction, c) =>
    records.push(`GravityLoad;${11 + c};;1.;${direction.map(real).join(";")}`));
  const run = runMixed(mesh, staticCase([11, 12, 13]), records);
  // Supports: −weight along the gravity; moments: −x_c × weight.
  const [fx, , , , my, mz] = resultant(run, mesh, "singlePointConstrForces", 1);
  const moments = resultant(run, mesh, "singlePointConstrForces", 2);
  const mass = -fx;
  // Gravity along x: M = −x_c × (m, 0, 0) = (0, −z_c m, y_c m); along y: M = −x_c × (0, m, 0) = (z_c m, 0, −x_c m).
  return { mass, centre: [-moments[5] / mass, mz / mass, -my / mass],
    others: [-moments[1], -resultant(run, mesh, "singlePointConstrForces", 3)[2]] };
}
test("mass: a distorted block of mixed shapes with mid-side nodes weighs ρ V at its centre", (t) => {
  const mesh = patchMesh(PATCH_MESHES[5], GEOMETRIES[1]);
  const { mass, centre, others } = massProperties(mesh, blockSupports(mesh).map((record) => record.split(";")[2]));
  const expected = STEEL.density * BLOCK.a * BLOCK.b * BLOCK.c;
  t.diagnostic(`mass ratio ${(mass / expected).toFixed(12)}, centre ${centre.map((v) => v.toFixed(10)).join(", ")}`);
  for (const value of [mass, ...others]) assertRatio(value, expected, CRITERIA.invariant, "mass");
  [BLOCK.a / 2, BLOCK.b / 2, BLOCK.c / 2].forEach((value, c) =>
    assertRatio(centre[c], value, CRITERIA.invariant, `centre of gravity, component ${c + 1}`));
});
for (const shape of SHAPES)
  test(`mass: a quarter ring of ${shape} with mid-side nodes on the circles`, (t) => {
    const [a, b, h] = [1.0, 2.0, 1.0];
    // Three grid points that no edge joins hold the ring.
    const mesh = ringMesh({ a, b, h, n: [2, 4, 1], shape, quadratic: true });
    const { mass, centre } = massProperties(mesh, [mesh.at(a, 0, 0), mesh.at(0, b, 0), mesh.at(b, 0, h)]);
    const expected = (STEEL.density * Math.PI * (b * b - a * a) * h) / 4;
    const offset = (4 * (b ** 3 - a ** 3)) / (3 * Math.PI * (b * b - a * a));
    t.diagnostic(`mass ratio ${(mass / expected).toFixed(8)}, centre ${centre.map((v) => v.toFixed(6)).join(", ")} ` +
      `(expected ${offset.toFixed(6)}, ${offset.toFixed(6)}, ${(h / 2).toFixed(6)})`);
    assertRatio(mass, expected, CRITERIA.displacement, "mass");
    [offset, offset, h / 2].forEach((value, c) =>
      assertRatio(centre[c], value, CRITERIA.displacement, `centre of gravity, component ${c + 1}`));
  });

// The same ring of hexahedra held at three corners of one element, two of them the ends of an arc: the supports are
// read along the edges (spec section 5.6) and their forces are carried back to the grid points; the table of the
// support forces must still balance the weight and its moment. (This case found that the forces carried back from
// the held arc did not keep their moment: the centre of gravity read from the table was 0.6 % off along z.)
test("mass: the support forces of a quarter ring of SolidHexa held along an arc balance the moment of its weight",
  (t) => {
    const [a, b, h] = [1.0, 2.0, 1.0];
    const mesh = ringMesh({ a, b, h, n: [2, 4, 1], shape: "SolidHexa", quadratic: true });
    const { mass, centre } = massProperties(mesh, mesh.elements[0].corners.slice(0, 3));
    const offset = (4 * (b ** 3 - a ** 3)) / (3 * Math.PI * (b * b - a * a));
    t.diagnostic(`mass ratio ${(mass / ((STEEL.density * Math.PI * (b * b - a * a) * h) / 4)).toFixed(8)}, centre ` +
      `${centre.map((v) => v.toFixed(6)).join(", ")} (expected ${offset.toFixed(6)}, ${offset.toFixed(6)}, 0.5)`);
    [offset, offset, h / 2].forEach((value, c) =>
      assertRatio(centre[c], value, 1.0e-4, `centre of gravity, component ${c + 1}`));
  });

// ---------------------------------------------------------------------------------------------------------------------
// 7. Stress output: the rows of the elements (centre and corners) and the table of the grid points.

// The eigenvalues of the symmetric tensor s (xx, yy, zz, xy, yz, zx), largest first, by the trigonometric solution of
// its characteristic cubic (O. K. Smith, Eigenvalues of a symmetric 3 × 3 matrix, Communications of the ACM 4
// (1961) 168).
function principalValues(s) {
  const mean = (s[0] + s[1] + s[2]) / 3;
  const d = [s[0] - mean, s[1] - mean, s[2] - mean];
  const p = Math.sqrt((d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 2 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)) / 6);
  if (p === 0) return [mean, mean, mean];
  const determinant = d[0] * (d[1] * d[2] - s[4] ** 2) - s[3] * (s[3] * d[2] - s[4] * s[5]) +
    s[5] * (s[3] * s[4] - d[1] * s[5]);
  const phi = Math.acos(Math.max(-1, Math.min(1, determinant / (2 * p ** 3)))) / 3;
  return [0, 2, 1].map((k) => mean + 2 * p * Math.cos(phi + (2 * Math.PI * k) / 3)).sort((a, b) => b - a);
}

// A block of mixed shapes under a general uniform stress and a pure bending together: at every row the principal
// values are the eigenvalues of the six components (in the order of the rows of the solids: o1 the largest, o2 the
// smallest, o3 the middle one), the von Mises stress
// sqrt(((o1 − o2)² + (o2 − o3)² + (o3 − o1)²) / 2), the pressure −(oxx + oyy + ozz) / 3, the principal directions unit
// eigenvectors (σ d = o d); the row of a grid point is the mean of the corner rows of its elements.
test("output: principal values, von Mises stress, pressure and directions follow from the components", (t) => {
  const mesh = patchMesh(PATCH_MESHES[5], GEOMETRIES[1]);
  const faces = skinFaces(mesh);
  const records = [...isotropic(STEEL), ...blockSupports(mesh), ...tractionRecords(2, faces, GENERAL_STRESS),
    ...BENDING_STATES[0].records(2, faces), ...BENDING_STATES[1].records(2, faces)];
  const run = runMixed(mesh, staticCase([2]), records);
  const worst = { principal: 0, vonMises: 0, pressure: 0, direction: 0, mean: 0 };
  let unordered = 0;
  const sums = new Map();
  const scale = 1.0e8;
  const check = (row) => {
    const s = sixOf(row);
    const principal = principalValues(s);
    [row.o1, row.o2, row.o3].sort((a, b) => b - a).forEach((value, k) => {
      worst.principal = Math.max(worst.principal, Math.abs(value - principal[k]));
    });
    if (row.o1 < row.o3 || row.o3 < row.o2) ++unordered;
    const mises = Math.sqrt(((row.o1 - row.o2) ** 2 + (row.o2 - row.o3) ** 2 + (row.o3 - row.o1) ** 2) / 2);
    worst.vonMises = Math.max(worst.vonMises, Math.abs(row.vonMises - mises));
    worst.pressure = Math.max(worst.pressure, Math.abs(row.pressure + (s[0] + s[1] + s[2]) / 3));
    [["o1x", "o1y", "o1z", row.o1], ["o2x", "o2y", "o2z", row.o2], ["o3x", "o3y", "o3z", row.o3]]
      .forEach(([x, y, z, value]) => {
        const d = [row[x], row[y], row[z]];
        const image = [s[0] * d[0] + s[3] * d[1] + s[5] * d[2], s[3] * d[0] + s[1] * d[1] + s[4] * d[2],
          s[5] * d[0] + s[4] * d[1] + s[2] * d[2]];
        worst.direction = Math.max(worst.direction, Math.abs(Math.hypot(...d) - 1) * scale,
          ...image.map((v, c) => Math.abs(v - value * d[c])));
      });
  };
  for (const card of mesh.cards())
    for (const row of rowsOf(run, card)) {
      check(row);
      if (row.nodeId === 0) continue;
      const sum = sums.get(row.nodeId) ?? { count: 0, total: [0, 0, 0, 0, 0, 0] };
      sixOf(row).forEach((v, c) => { sum.total[c] += v; });
      ++sum.count;
      sums.set(row.nodeId, sum);
    }
  const grid = rowsOf(run, "SolidGridPoints");
  assert.deepEqual(grid.map((row) => row.id).sort((a, b) => a - b), [...sums.keys()].sort((a, b) => a - b),
    "one row per corner grid point of the solids");
  for (const row of grid) {
    check(row);
    const { count, total } = sums.get(row.id);
    sixOf(row).forEach((v, c) => { worst.mean = Math.max(worst.mean, Math.abs(v - total[c] / count)); });
  }
  t.diagnostic(`${Object.entries(worst).map(([k, v]) => `${k} ${(v / scale).toExponential(2)}`).join(", ")}, ` +
    `rows whose principal values are not largest, smallest, middle: ${unordered}`);
  for (const [what, error] of Object.entries(worst)) assertSmall(error, scale, CRITERIA.invariant, what);
  assert.equal(unordered, 0, "principal values in the order largest, smallest, middle");
});
