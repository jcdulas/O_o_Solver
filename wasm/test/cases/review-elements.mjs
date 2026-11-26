// Cases of the review of the classic element kernels, each against a value stated by hand:
// - the thermal loads of an offset shell follow ModelParams OffsetDefinition, as those of every other family;
// - a laminate is rigid in transverse shear only where every ply is: a sandwich whose faces leave G1Z and G2Z blank
//   shears in its core;
// - a ShellQuad8 without shear material takes the shear modulus of its bending material in the material axes;
// - the transverse shear strain of a ply is its stress over the ply's modulus in the direction of that stress.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver } from "../helpers/runtime.mjs";
import { assertFinite, displacement, real, withModel } from "../helpers/static-models.mjs";

// The displacement component along Z.
const ALONG_Z = 2;

// The text of a static deck of one subcase with the supports of set 1: `loads` selects the load set 2, `thermal` the
// temperature set 7; `outputs` are the output requests beside the displacements.
function deck({ model, loads = false, thermal = false, outputs = [] }) {
  return ["[executive]", "SolutionSequence;101", "[case]",
    ...(thermal ? ["TemperatureSelection;7;TemperatureLoadPart"] : []), "SubcaseStart;1",
    "SinglePointConstrSelection;1", ...(loads ? ["StaticLoadSelection;2"] : []), "DisplacementOutput;All",
    ...outputs, "[model]", ...model, ""].join("\n");
}

// The static results of the model of `text` (the classic formulation: the default of every card).
function solve(text) {
  return withModel(text, (model) => {
    const run = solver.run(model, "statics");
    assertFinite(run);
    return run;
  });
}

// |actual / expected − 1| ≤ tolerance.
function assertRatio(actual, expected, tolerance, what) {
  assert.ok(Math.abs(actual / expected - 1) <= tolerance,
    `${what}: ${actual}, expected ${expected} (ratio ${(actual / expected).toFixed(8)}, tolerance ${tolerance})`);
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. The thermal loads of an offset shell.
//
// A free square plate of side 2 (2 × 2 cells, grid points 1 to 9 row by row, the middle one 5 at the origin), its
// section at ZOFFS = d above the grid points, under a uniform temperature rise. The thermal forces of the elements
// act in the plane of the section. Taken at the grid points (ElemOffsetsOnly, the default) they act d under it and
// bend the plate with the moment N_T d on its edges, N_T = E t α ΔT / (1 − ν): with D = E t³ / (12 (1 − ν²)) the
// curvature is N_T d / (D (1 + ν)) = 12 α ΔT d / t² in every direction, towards the side of the offset. Taken on the
// offset (ElemOffsetZ) they are the loads of the stiffness, and the plate expands without bending.
const HOT_PLATE = { thickness: 0.05, offset: 0.02, young: 1000, poisson: 0.3, expansion: 1.0e-3, rise: 10 };

// The plate of `shape` (ShellQuad4, or ShellTria3: each cell cut in two) under ModelParams OffsetDefinition
// `definition` (null: none given). Its supports hold the six rigid motions and nothing else: the three translations
// of grid point 5, the translations along Y and Z of 6 (at x = 1) and along Z of 8 (at y = 1).
function hotOffsetPlate(shape, definition) {
  const { thickness, offset, young, poisson, expansion, rise } = HOT_PLATE;
  const node = (i, j) => 1 + i + 3 * j;
  const model = definition ? [`ModelParams;OffsetDefinition;${definition}`] : [];
  for (let j = 0; j < 3; ++j)
    for (let i = 0; i < 3; ++i) model.push(`Node;${node(i, j)};;${real(i - 1)};${real(j - 1)};0.`);
  let element = 0;
  for (let j = 0; j < 2; ++j)
    for (let i = 0; i < 2; ++i) {
      const [a, b, c, d] = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)];
      // The fields after the grid points: the material angle, the material system, the offset.
      if (shape === "ShellQuad4") model.push(`ShellQuad4;${++element};1;${a};${b};${c};${d};;;${real(offset)}`);
      else
        model.push(`ShellTria3;${++element};1;${a};${b};${c};;;${real(offset)}`,
          `ShellTria3;${++element};1;${a};${c};${d};;;${real(offset)}`);
    }
  model.push(`ShellProp;1;1;${real(thickness)};1;;1`,
    `IsoMat;1;${real(young)};;${real(poisson)};0.;${real(expansion)};0.`,
    "SinglePointConstrList;1;123;5", "SinglePointConstrList;1;23;6", "SinglePointConstrList;1;3;8",
    `DefaultTemperatureLoad;7;${real(rise)}`);
  return deck({ model, thermal: true });
}

for (const shape of ["ShellQuad4", "ShellTria3"])
  test(`the thermal loads of an offset ${shape} follow ModelParams OffsetDefinition`, () => {
    const { thickness, offset, expansion, rise } = HOT_PLATE;
    const curvature = (12 * expansion * rise * offset) / thickness ** 2;
    // With the deflection held at 0 at (0, 0), (1, 0) and (0, 1), w = κ (x² + y² − x − y) / 2: κ at the grid points
    // 4 (−1, 0) and 2 (0, −1).
    const deflections = (definition) => {
      const run = solve(hotOffsetPlate(shape, definition));
      return [displacement(run, 4, ALONG_Z), displacement(run, 2, ALONG_Z)];
    };
    for (const definition of [null, "ElemOffsetsOnly"])
      for (const [k, w] of deflections(definition).entries())
        assertRatio(w, curvature, 1e-6, `deflection ${k} with the loads at the grid points (${definition})`);
    for (const [k, w] of deflections("ElemOffsetZ").entries())
      assert.ok(Math.abs(w) <= 1e-7 * curvature, `deflection ${k} with the loads on the offset: ${w}`);
  });

// ---------------------------------------------------------------------------------------------------------------------
// 2. The transverse shear of a sandwich.
//
// A cantilever strip of one ShellQuad4 (length L along X, width b, ν = 0) of a laminate of two faces and a core,
// under a tip force P. The faces leave G1Z and G2Z blank (rigid in transverse shear), the core gives them: the
// laminate shears in its core. With D the bending stiffness of the stack and H its transverse shear stiffness, both
// per unit width, the tip deflects by P L³ / (3 b D) + P L / (b H).
//
// H is that of the shear stresses of cylindrical bending (Whitney, Structural Analysis of Laminated Anisotropic
// Plates, section 4.6): per unit shear force τ(z) = S(z) / D, S the first moment of the moduli beyond z, and
// 1 / H = ∫ τ² / G dz, over the core alone since the faces have no shear strain. In the core, with c its thickness, h
// the total one and E_f, E_c the moduli, S(z) = a − e z², a = E_f (h² − c²) / 8 + E_c c² / 8, e = E_c / 2, so that
// ∫ S² dz = a² c − a e c³ / 6 + e² c⁵ / 80.
const SANDWICH = { length: 200, width: 50, face: 1, core: 20, faceYoung: 70000, coreYoung: 100, coreShear: 40,
  force: 10 };

function sandwichStiffnesses() {
  const { face, core, faceYoung, coreYoung, coreShear } = SANDWICH;
  const total = core + 2 * face;
  const bending = (coreYoung * core ** 3 + faceYoung * (total ** 3 - core ** 3)) / 12;
  const a = (faceYoung * (total ** 2 - core ** 2)) / 8 + (coreYoung * core ** 2) / 8;
  const e = coreYoung / 2;
  const integral = a * a * core - (a * e * core ** 3) / 6 + (e * e * core ** 5) / 80;
  return { bending, shear: (coreShear * bending ** 2) / integral };
}

// The strip with the cards of its property and materials `section`, clamped at X = 0, the force shared by its two
// tip grid points 2 and 3.
function cantileverStrip(section, outputs = []) {
  const { length, width, force } = SANDWICH;
  const model = ["Node;1;;0.;0.;0.", `Node;2;;${real(length)};0.;0.`, `Node;3;;${real(length)};${real(width)};0.`,
    `Node;4;;0.;${real(width)};0.`, "ShellQuad4;1;1;1;2;3;4", ...section, "SinglePointConstrList;1;123456;1;4",
    `PointForceLoad;2;2;;${real(force / 2)};0.;0.;1.`, `PointForceLoad;2;3;;${real(force / 2)};0.;0.;1.`];
  return deck({ model, loads: true, outputs });
}

test("a sandwich whose faces leave G1Z and G2Z blank shears in its core", () => {
  const { length, width, face, core, faceYoung, coreYoung, coreShear, force } = SANDWICH;
  // The plies from the bottom: material, thickness, angle, output.
  const section = [`CompProp;1;;;;;;;;1;${real(face)};0.;Yes;2;${real(core)};0.;Yes;1;${real(face)};0.;Yes`,
    `OrthoMat2D;1;${real(faceYoung)};${real(faceYoung)};0.;${real(faceYoung / 2)}`,
    `OrthoMat2D;2;${real(coreYoung)};${real(coreYoung)};0.;${real(coreShear)};${real(coreShear)};${real(coreShear)}`];
  const run = solve(cantileverStrip(section));
  const { bending, shear } = sandwichStiffnesses();
  const bent = (force * length ** 3) / (3 * width * bending);
  const sheared = (force * length) / (width * shear);
  // The shear is more than half of the deflection: a laminate taken as rigid would give `bent` alone.
  assert.ok(sheared > bent, `the case is one where the shear matters: ${sheared} against ${bent}`);
  for (const node of [2, 3]) assertRatio(displacement(run, node, ALONG_Z), bent + sheared, 2e-4, `tip ${node}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// 3. A ShellQuad8 without shear material.
//
// A thick cantilever strip of two ShellQuad8 along X of one orthotropic layer at 45° from the element's x axis,
// without shear material (ShellProp MID3 blank). Its transverse shear is that of the bending material, TS/T · G12 · t:
// the same strip with an isotropic shear material of modulus G12 has the same stiffness to round-off. The term (3, 3)
// of the layer's matrix in the element's axes, (Q11 + Q22 − 2 Q12) / 4 at 45°, is seven times G12 here: a strip with a
// shear material of that modulus deflects measurably less.
const THICK_STRIP = { length: 1.0, width: 0.5, thickness: 0.2, young1: 140000, young2: 10000, poisson: 0.3,
  shear: 5000, angle: 45, force: 1 };

// The strip with the shear material of modulus `shearModulus` (null: MID3 blank), and its tip deflection at the
// middle of its tip (grid point 10).
function thickStripDeflection(shearModulus) {
  const { length, width, thickness, young1, young2, poisson, shear, angle, force } = THICK_STRIP;
  // 5 × 3 grid points (the middles of the two cells left out), numbered row by row.
  const node = (i, j) => 1 + i + 5 * j;
  const model = [];
  for (let j = 0; j < 3; ++j)
    for (let i = 0; i < 5; ++i) {
      if (i % 2 === 1 && j === 1) continue; // the middle of a cell
      model.push(`Node;${node(i, j)};;${real((length * i) / 4)};${real((width * j) / 2)};0.`);
    }
  for (let cell = 0; cell < 2; ++cell) {
    const i = 2 * cell;
    const nodes = [node(i, 0), node(i + 2, 0), node(i + 2, 2), node(i, 2), node(i + 1, 0), node(i + 2, 1),
      node(i + 1, 2), node(i, 1)];
    // After the grid points: the four corner thicknesses (blank), then the material angle.
    model.push(`ShellQuad8;${cell + 1};1;${nodes.join(";")};;;;;${real(angle)}`);
  }
  model.push(`ShellProp;1;1;${real(thickness)};1${shearModulus === null ? "" : ";;2"}`,
    `OrthoMat2D;1;${real(young1)};${real(young2)};${real(poisson)};${real(shear)}`);
  // An isotropic material of shear modulus G: E = 2 G (1 + ν) with ν = 0.25.
  if (shearModulus !== null) model.push(`IsoMat;2;${real(2.5 * shearModulus)};${real(shearModulus)};0.25`);
  model.push(`SinglePointConstrList;1;123456;${node(0, 0)};${node(0, 1)};${node(0, 2)}`);
  // The tip force on the three tip grid points, as a uniform line load on a quadratic edge: 1/6, 4/6, 1/6.
  for (const [j, share] of [[0, 1 / 6], [1, 4 / 6], [2, 1 / 6]])
    model.push(`PointForceLoad;2;${node(4, j)};;${real(force * share)};0.;0.;1.`);
  return displacement(solve(deck({ model, loads: true })), node(4, 1), ALONG_Z);
}

test("a ShellQuad8 without shear material takes the shear modulus of its bending material in the material axes", () => {
  const { young1, young2, poisson, shear } = THICK_STRIP;
  const blank = thickStripDeflection(null);
  assertRatio(blank, thickStripDeflection(shear), 1e-9, "against a shear material of modulus G12");
  // The plane-stress terms of the layer (Jones, Mechanics of Composite Materials, eq. 2.61) and its term (3, 3) at 45°.
  const denominator = 1 - (poisson * poisson * young2) / young1;
  const [q11, q22, q12] = [young1 / denominator, young2 / denominator, (poisson * young2) / denominator];
  const turned = (q11 + q22 - 2 * q12) / 4;
  assert.ok(turned > 7 * shear, `the term (3, 3) at 45 degrees is far from G12: ${turned}`);
  const stiffer = thickStripDeflection(turned);
  assert.ok(blank / stiffer - 1 > 1e-3,
    `a shear material of the modulus of the element's axes is told apart: ${blank} against ${stiffer}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// 4. The transverse shear strains of the plies.
//
// The strip of case 2 with a laminate [0, 90, 0] of one material whose G1Z and G2Z differ, under its tip force: the
// shear force is along the material axis x of the element. The ply tables give the transverse shear stress of each
// ply along x (t1z) and its strain (e1z), the stress over the modulus of the ply in that direction: G1Z for the plies
// at 0°, G2Z for the ply at 90°, whose axis 2 is along x. The top ply has no stress at its top (a free surface).
const CROSS_PLY = { ply: 2, young1: 60000, young2: 8000, shear: 3000, shear1z: 4000, shear2z: 2000 };

// The rows of the ply table `kind` ("stress", "strain") of the run, by ply (layer), as objects of its columns.
function plyRows(run, kind, column) {
  const table = run.elementTables.find((t) => t.kind === kind && t.components.includes(column));
  assert.ok(table, `a ply ${kind} table with the column ${column}; the run has ${run.elementTables.map((t) =>
    `${t.elementType}/${t.kind}`).join(", ")}`);
  const width = table.components.length;
  const at = table.components.indexOf(column);
  const rows = new Map();
  Array.from(table.ids).forEach((_, r) => rows.set(table.layers[r], table.values[r * width + at]));
  return rows;
}

test("the transverse shear strain of a ply is its stress over the modulus of the ply in that direction", () => {
  const { ply, young1, young2, shear, shear1z, shear2z } = CROSS_PLY;
  const section = [`CompProp;1;;;;;;;;${[0, 90, 0].map((angle) => `1;${real(ply)};${real(angle)};Yes`).join(";")}`,
    `OrthoMat2D;1;${real(young1)};${real(young2)};0.;${real(shear)};${real(shear1z)};${real(shear2z)}`];
  const run = solve(cantileverStrip(section, ["StressOutput;All", "StrainOutput;All"]));
  const [stresses, strains] = [plyRows(run, "stress", "t1z"), plyRows(run, "strain", "e1z")];
  for (const [layer, modulus] of [[1, shear1z], [2, shear2z]]) {
    assert.ok(Math.abs(stresses.get(layer)) > 0, `ply ${layer} carries a transverse shear: ${stresses.get(layer)}`);
    assertRatio(stresses.get(layer) / strains.get(layer), modulus, 1e-9, `the shear modulus of ply ${layer} along x`);
  }
});
