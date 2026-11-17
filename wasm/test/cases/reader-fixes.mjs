// Three defects found outside the element formulations, each in ClassicFormulation and in MixedHybridFormulation: the
// face a FacePressureLoad names on a SolidPyramid, a model whose only material is an AnisoMat3D, and the temperature
// gradients across the section of a bar or beam (LineElemTemperatureLoad).

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

// A real as a model file writes it: with a decimal point or an exponent.
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

function withModel(text, name, options, action) {
  const file = `/work/${name}.oos`;
  oor.writeFile(file, text);
  const deck = oor.readModel(file);
  try {
    const model = solver.modelFromReader(oor, deck, options);
    try { return action(model); } finally { model.delete(); }
  } finally { deck.delete(); }
}

function runStatics(text, name, options) {
  return withModel(text, name, options, (model) => {
    const run = solver.run(model, "statics");
    assert.ok(!run.diagnostics.some((d) => d.severity === "error"), JSON.stringify(run.diagnostics));
    return run;
  });
}

// The six values of a grid point in a table of the subcase (zeros for a grid point the table leaves out).
function nodal(run, kind, node) {
  const table = run.nodalTables.find((t) => t.kind === kind && t.subcase === 1);
  assert.ok(table, `a ${kind} table`);
  const row = table.ids.indexOf(node);
  return row < 0 ? [0, 0, 0, 0, 0, 0] : Array.from(table.values.slice(row * 6, row * 6 + 6));
}

// The rows of an element table of the subcase, each with its components by name.
function rowsOf(run, elementType, kind) {
  const table = run.elementTables.find((t) => t.elementType === elementType && t.kind === kind && t.subcase === 1);
  assert.ok(table, `a ${elementType} ${kind} table`);
  const width = table.components.length;
  return Array.from(table.ids, (id, r) => {
    const row = { id };
    table.components.forEach((name, c) => { row[name] = table.values[r * width + c]; });
    return row;
  });
}

// Each component within `tolerance` times the largest expected magnitude (at least `floor`).
function assertClose(actual, expected, tolerance, what, floor = 1e-30) {
  const scale = Math.max(...expected.map(Math.abs), floor);
  for (let k = 0; k < expected.length; ++k)
    assert.ok(Math.abs(actual[k] - expected[k]) <= tolerance * scale,
      `${what}: component ${k + 1} is ${actual[k]}, expected ${expected[k]}`);
}

const FORMULATIONS = ["ClassicFormulation", "MixedHybridFormulation"];

// ---------------------------------------------------------------------------------------------------------------------
// The faces of a pyramid. The base is named by two opposite corners, a triangle by the two corners of its base edge.

const PYRAMID = { pressure: 1.0e6, young: 2.0e11, poisson: 0.3 };
// Grid point 2 is on the x axis and grid point 4 in the plane z = 0: the supports below hold the rigid motions and
// leave the hydrostatic displacement u = ε x free.
const PYRAMID_POINTS = [[0, 0, 0], [1.1, 0, 0], [1.0, 1.2, 0.1], [-0.1, 0.9, 0], [0.4, 0.6, 0.9]];

// A distorted pyramid held against the rigid motions (grid point 1 in 1, 2, 3; grid point 2 in 2, 3; grid point 4 in
// 3), with the FacePressureLoad records of the pairs `faces` = [[G1, G3], ...].
function pyramid(faces) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "[model]"];
  PYRAMID_POINTS.forEach(([x, y, z], k) => lines.push(`Node;${k + 1};;${real(x)};${real(y)};${real(z)}`));
  lines.push("SolidPyramid;1;1;1;2;3;4;5", "SolidProp;1;1",
    `IsoMat;1;${real(PYRAMID.young)};;${real(PYRAMID.poisson)};7800.`);
  lines.push("SinglePointConstr;1;1;123;0.", "SinglePointConstr;1;2;3;0.", "SinglePointConstr;1;4;3;0.",
    "SinglePointConstr;1;2;2;0.");
  for (const [first, across] of faces)
    lines.push(`FacePressureLoad;2;1;${real(PYRAMID.pressure)};;;;${first};${across}`);
  return lines.join("\n") + "\n";
}

// Every edge of the pyramid is sharp: its skin is read as five faces.
const PYRAMID_EDGES = [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5]];
function pyramidOptions(formulation) {
  return formulation === "MixedHybridFormulation"
    ? { elementFormulation: formulation, formulationGeometry: { sharpEdges: PYRAMID_EDGES } } : {};
}

for (const formulation of FORMULATIONS)
  test(`a FacePressureLoad on each of the five faces of a SolidPyramid is hydrostatic in ${formulation}`, () => {
    const { pressure, young, poisson } = PYRAMID;
    // The triangles by their base edge, in either order of its two corners.
    const run = runStatics(pyramid([[1, 3], [1, 2], [3, 2], [3, 4], [1, 4]]), "fix-pyramid",
      pyramidOptions(formulation));
    for (let node = 1; node <= 5; ++node)
      assertClose(nodal(run, "singlePointConstrForces", node).slice(0, 3), [0, 0, 0], 1e-8, `support of ${node}`,
        pressure);
    // σ = −p I: u = −p (1 − 2 ν) x / E from the grid point held.
    const strain = (-pressure * (1 - 2 * poisson)) / young;
    for (let node = 2; node <= 5; ++node)
      assertClose(nodal(run, "displacements", node).slice(0, 3), PYRAMID_POINTS[node - 1].map((x) => strain * x), 1e-7,
        `displacement of grid point ${node}`);
  });

for (const formulation of FORMULATIONS)
  test(`a corner and the apex of a SolidPyramid name no face to a FacePressureLoad in ${formulation}`, () => {
    // Their edge belongs to two triangles: the load is refused, not put on one of them.
    withModel(pyramid([[1, 5]]), "fix-pyramid-apex", pyramidOptions(formulation), (model) => {
      assert.throws(() => solver.run(model, "statics"), /unsupported-load.*FacePressureLoad 2.*SolidPyramid 1/s);
    });
  });

// ---------------------------------------------------------------------------------------------------------------------
// A model whose only material is an AnisoMat3D.

const BLOCK = { young: 2.0e11, poisson: 0.3, stretch: 1.0e-3, length: 2.0, height: 1.0, width: 1.5 };

// The stiffness of the isotropic material of BLOCK written as an AnisoMat3D (the upper triangle by rows):
// C11 = λ + 2 G, C12 = λ, C44 = G.
const SHEAR = BLOCK.young / (2 * (1 + BLOCK.poisson));
const LAME = (2 * SHEAR * BLOCK.poisson) / (1 - 2 * BLOCK.poisson);
const ANISOTROPIC = (() => {
  const [c11, c12, c44] = [LAME + 2 * SHEAR, LAME, SHEAR].map(real);
  return `AnisoMat3D;1;${c11};${c12};${c12};0.;0.;0.;${c11};${c12};0.;0.;0.;${c11};0.;0.;0.;${c44};0.;0.;${c44};0.;${
    c44};7800.`;
})();

// One SolidHexa of material 1 (`materials`: the material records of the model) stretched along x by its supports:
// the face x = 0 held along x, the face x = length moved by `stretch`, the rigid motions held at three grid points.
function stretchedBlock(materials) {
  const { length, height, width, stretch } = BLOCK;
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "SubcaseStart;1", "SinglePointConstrSelection;1", "[model]"];
  const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  corners.forEach(([i, j, k], n) =>
    lines.push(`Node;${n + 1};;${real(length * i)};${real(height * j)};${real(width * k)}`));
  lines.push("SolidHexa;1;1;1;2;3;4;5;6;7;8", "SolidProp;1;1", ...materials);
  lines.push("SinglePointConstrList;1;1;1;4;5;8", "SinglePointConstrList;1;23;1", "SinglePointConstrList;1;3;4",
    "SinglePointConstrList;1;2;5");
  for (const node of [2, 3, 6, 7]) lines.push(`SinglePointConstr;1;${node};1;${real(stretch)}`);
  return lines.join("\n") + "\n";
}

// The twelve edges of the block are sharp: its skin is read as six faces.
const BLOCK_EDGES = [[1, 2], [2, 3], [3, 4], [1, 4], [5, 6], [6, 7], [7, 8], [5, 8], [1, 5], [2, 6], [3, 7], [4, 8]];

const blockOptions = (formulation) => (formulation === "MixedHybridFormulation"
  ? { elementFormulation: formulation, formulationGeometry: { sharpEdges: BLOCK_EDGES } } : {});

for (const formulation of FORMULATIONS)
  test(`a model whose only material is an AnisoMat3D runs in ${formulation}`, () => {
    const { young, poisson, stretch, length, height, width } = BLOCK;
    const run = runStatics(stretchedBlock([ANISOTROPIC]), "fix-anisotropic", blockOptions(formulation));
    // The uniaxial stress of the isotropic material it stands for: E ε along x, the section free to shrink.
    const strain = stretch / length;
    assertClose(nodal(run, "displacements", 7).slice(0, 3),
      [stretch, -poisson * strain * height, -poisson * strain * width], 1e-8, "displacement of grid point 7");
    let pull = 0;
    for (const node of [2, 3, 6, 7]) pull += nodal(run, "singlePointConstrForces", node)[0];
    const expected = young * height * width * strain;
    assert.ok(Math.abs(pull / expected - 1) < 1e-8, `pull ${pull}, expected ${expected}`);
  });

test("an isotropic material that no property names changes nothing to a model of an AnisoMat3D", () => {
  const alone = runStatics(stretchedBlock([ANISOTROPIC]), "fix-anisotropic", {});
  const withOther = runStatics(stretchedBlock([ANISOTROPIC, "IsoMat;9;7.e10;;0.3;2700."]), "fix-anisotropic-two", {});
  for (let node = 1; node <= 8; ++node)
    assert.deepEqual(nodal(withOther, "displacements", node), nodal(alone, "displacements", node));
});

// ---------------------------------------------------------------------------------------------------------------------
// The temperature gradients of LineElemTemperatureLoad: a hotter side is longer, so a free line bends away from it,
// v'' = −α T'y and w'' = −α T'z.

const LINE = { length: 1.2, area: 6.0e-4, i1: 4.5e-8, i2: 2.0e-8, j: 4.7e-8, young: 2.1e11, alpha: 1.2e-5 };

// The stress points C and D at y = ±0.01, E and F at z = ±0.02.
const STRESS_POINTS = "0.01;0.;-0.01;0.;0.;0.02;0.;-0.02";

// `count` Bar2 or Beam2 along x with their y axis along the basic y; `records` the supports and temperature cards.
function line({ card, count, records, caseRecords = [] }) {
  const lines = ["[executive]", "SolutionSequence;101", "[case]", "TemperatureSelection;7;TemperatureLoadPart",
    ...caseRecords, "SubcaseStart;1", "SinglePointConstrSelection;1", "DisplacementOutput;All", "ElemForceOutput;All",
    "StressOutput;All", "SinglePointConstrForceOutput;All", "[model]"];
  for (let i = 0; i <= count; ++i) lines.push(`Node;${i + 1};;${real((LINE.length * i) / count)};0.;0.`);
  for (let i = 0; i < count; ++i) lines.push(`${card};${i + 1};1;${i + 1};${i + 2};0.;;1.;0.`);
  lines.push(card === "Bar2"
    ? `BarProp;1;1;${real(LINE.area)};${real(LINE.i1)};${real(LINE.i2)};${real(LINE.j)};;${STRESS_POINTS}`
    : `BeamProp;1;1;${real(LINE.area)};${real(LINE.i1)};${real(LINE.i2)};;${real(LINE.j)};;${STRESS_POINTS}`);
  lines.push(`IsoMat;1;${real(LINE.young)};;0.3;7850.;${real(LINE.alpha)};0.`, ...records);
  return lines.join("\n") + "\n";
}

// The LineElemTemperatureLoad of set `set` on element `element`: T̄ at ends A and B, then T'y and T'z at A and B.
function lineTemperature(set, element, { mean = [0, 0], gradientY = [0, 0], gradientZ = [0, 0] }) {
  return `LineElemTemperatureLoad;${set};${element};${[...mean, ...gradientY, ...gradientZ].map(real).join(";")}`;
}

const optionsOf = (formulation) =>
  (formulation === "MixedHybridFormulation" ? { elementFormulation: formulation } : {});
// The mixed bar without shear flexibility has a stiff shear (k G A = 1e4 E A): exact to 1e-7.
const toleranceOf = (formulation) => (formulation === "MixedHybridFormulation" ? 1e-6 : 1e-9);

for (const formulation of FORMULATIONS)
  for (const card of ["Bar2", "Beam2"])
    test(`a free cantilever of two ${card} bends under the gradients of LineElemTemperatureLoad in ${formulation}`,
      () => {
        const { length, alpha, young, i1 } = LINE;
        const gradientY = 300.0;
        const gradientZ = -200.0;
        const cards = [1, 2].map((element) => lineTemperature(7, element,
          { gradientY: [gradientY, gradientY], gradientZ: [gradientZ, gradientZ] }));
        const run = runStatics(line({ card, count: 2, records: ["SinglePointConstrList;1;123456;1", ...cards] }),
          "fix-line-free", optionsOf(formulation));
        // v = −α T'y x² / 2, w = −α T'z x² / 2, θz = v', θy = −w'.
        for (const [node, x] of [[2, length / 2], [3, length]])
          assertClose(nodal(run, "displacements", node),
            [0, (-alpha * gradientY * x * x) / 2, (-alpha * gradientZ * x * x) / 2, 0, alpha * gradientZ * x,
              -alpha * gradientY * x], toleranceOf(formulation), `displacement of grid point ${node}`);
        // Free: no force in the elements nor at the support (against the moment E I α T' of the same line held).
        const held = young * i1 * alpha * gradientY;
        for (const row of rowsOf(run, card, "forces"))
          for (const [name, value] of Object.entries(row))
            if (name !== "id") assert.ok(Math.abs(value) < 1e-6 * held, `${name} of element ${row.id} is ${value}`);
        assertClose(nodal(run, "singlePointConstrForces", 1), [0, 0, 0, 0, 0, 0], 1e-6, "support", held);
      });

for (const formulation of FORMULATIONS)
  test(`a Bar2 held at both ends takes the moments and stresses of its temperature gradients in ${formulation}`, () => {
    const { alpha, young, i1, i2 } = LINE;
    const gradientY = 300.0;
    const gradientZ = -200.0;
    const run = runStatics(line({ card: "Bar2", count: 1, records: ["SinglePointConstrList;1;123456;1;2",
      lineTemperature(7, 1, { gradientY: [gradientY, gradientY], gradientZ: [gradientZ, gradientZ] })] }),
    "fix-line-held", optionsOf(formulation));
    // The elastic curvatures are those of the free bending, reversed: M1 = E I1 α T'y, M2 = E I2 α T'z, and the
    // stress of a fibre is −E α times its temperature rise, T'y y + T'z z.
    const [forces] = rowsOf(run, "Bar2", "forces");
    const moment1 = young * i1 * alpha * gradientY;
    const moment2 = young * i2 * alpha * gradientZ;
    const tolerance = toleranceOf(formulation);
    assertClose([forces.bendingMomentA1, forces.bendingMomentA2, forces.bendingMomentB1, forces.bendingMomentB2],
      [moment1, moment2, moment1, moment2], tolerance, "moments");
    assertClose([forces.shear1, forces.shear2, forces.axial, forces.torque], [0, 0, 0, 0], tolerance,
      "other forces", Math.abs(moment1));
    const [stresses] = rowsOf(run, "Bar2", "stress");
    const atY = -young * alpha * gradientY * 0.01;
    const atZ = -young * alpha * gradientZ * 0.02;
    assertClose([stresses.s1a, stresses.s2a, stresses.s3a, stresses.s4a], [atY, -atY, atZ, -atZ], tolerance,
      "stresses at end A");
    assertClose([stresses.s1b, stresses.s2b, stresses.s3b, stresses.s4b], [atY, -atY, atZ, -atZ], tolerance,
      "stresses at end B");
  });

for (const formulation of FORMULATIONS)
  test(`a gradient that varies along a Beam2 and its mean temperatures act together in ${formulation}`, () => {
    const { length, alpha } = LINE;
    // T'y from 100 at end A to 400 at end B, T̄ from 20 to 60 (the reference temperature of the material is 0).
    const run = runStatics(line({ card: "Beam2", count: 1, records: ["SinglePointConstrList;1;123456;1",
      lineTemperature(7, 1, { mean: [20, 60], gradientY: [100, 400] })] }), "fix-line-varying",
    optionsOf(formulation));
    // v'' = −α (a + (b − a) x / L): v(L) = −α L² (2 a + b) / 6, θz(L) = −α L (a + b) / 2; u(L) = α L (20 + 60) / 2.
    assertClose(nodal(run, "displacements", 2),
      [alpha * length * 40, (-alpha * length * length * (2 * 100 + 400)) / 6, 0, 0, 0, (-alpha * length * 500) / 2],
      toleranceOf(formulation), "tip");
  });

for (const formulation of FORMULATIONS)
  test(`the gradients of the initial temperatures of a Bar2 are taken off those of its load in ${formulation}`, () => {
    const { length, alpha } = LINE;
    const run = runStatics(line({ card: "Bar2", count: 1, caseRecords: ["TemperatureSelection;8;InitialTemperature"],
      records: ["SinglePointConstrList;1;123456;1", lineTemperature(7, 1, { gradientY: [300, 300] }),
        lineTemperature(8, 1, { gradientY: [100, 100] })] }), "fix-line-initial", optionsOf(formulation));
    assertClose(nodal(run, "displacements", 2),
      [0, (-alpha * 200 * length * length) / 2, 0, 0, 0, -alpha * 200 * length], toleranceOf(formulation), "tip");
  });
