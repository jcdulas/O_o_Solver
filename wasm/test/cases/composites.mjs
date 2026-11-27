// Cases of the laminated shells, each against a value stated by hand:
// - a symmetric stack whose ply heights are not exact in binary has no membrane-bending coupling: the drilling shells
//   of the classic formulation take it, and refuse a stack that does couple, naming the shells that take it;
// - the transverse shear of the plies follows the equilibrium distribution whatever the stack: the parabola of a
//   homogeneous plate in a stack of equal plies;
// - the failure indices follow their criteria and the strength ratios bring each criterion to one;
// - a failure theory without the strengths of a ply warns once per property;
// - an AnisoMat2D ply is rigid in transverse shear in the classic formulation and takes its in-plane shear modulus in
//   the mixed hybrid one;
// - a Hashin failure theory is refused by its name in the model;
// - the transverse shear stiffness of a stack follows the equilibrium of its plies, or 5/6 of their sheared sum under
//   LaminateTransverseShear = FiveSixths, and the report of a run gives it with its factor.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver } from "../helpers/runtime.mjs";
import { assertFinite, displacement, real, withModel } from "../helpers/static-models.mjs";

// The displacement component along Z.
const ALONG_Z = 2;

// A cantilever strip along X (units: N, mm, MPa), clamped at X = 0, a tip force shared by its two tip grid points.
const STRIP = { length: 200, width: 50, force: 1 };

// A carbon/epoxy ply: moduli, transverse shear moduli, strengths (Xt, Xc, Yt, Yc, S) and the Tsai-Wu term F12.
const PLY = { young1: 140000, young2: 10000, poisson: 0.3, shear: 5000, shear1Z: 5000, shear2Z: 3500, xt: 1500,
  xc: 1200, yt: 50, yc: 200, s: 70, f12: -3.7e-6 };
const BOND_ALLOWABLE = 60;

function plyMaterial(id, { strengths = true } = {}) {
  const elastic = `OrthoMat2D;${id};${real(PLY.young1)};${real(PLY.young2)};${real(PLY.poisson)};${real(PLY.shear)};` +
    `${real(PLY.shear1Z)};${real(PLY.shear2Z)};1.6e-9`;
  if (!strengths) return elastic;
  return `${elastic};;;;${real(PLY.xt)};${real(PLY.xc)};${real(PLY.yt)};${real(PLY.yc)};${real(PLY.s)};;${PLY.f12}`;
}

// A CompProp of the plies `angles` (degrees, from the bottom) of thickness `thickness` and material 10.
function laminate({ angles, thickness, theory = "", option = "" }) {
  const plies = angles.map((angle) => `10;${real(thickness)};${real(angle)};Yes`).join(";");
  return `CompProp;1;;;${real(BOND_ALLOWABLE)};${theory};;;${option};${plies}`;
}

// The strip of `elements` (the element records) with its property and material records `section`.
function strip({ elements = ["ShellQuad4;1;1;1;2;3;4"], section, parameters = [], outputs = ["StressOutput;All"] }) {
  const { length, width, force } = STRIP;
  return ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", "SinglePointConstrSelection;1",
    "StaticLoadSelection;2", "DisplacementOutput;All", ...outputs, "[model]", ...parameters, "Node;1;;0.;0.;0.",
    `Node;2;;${real(length)};0.;0.`, `Node;3;;${real(length)};${real(width)};0.`, `Node;4;;0.;${real(width)};0.`,
    ...elements, ...section, "SinglePointConstrList;1;123456;1;4", `PointForceLoad;2;2;;${real(force / 2)};0.;0.;1.`,
    `PointForceLoad;2;3;;${real(force / 2)};0.;0.;1.`, ""].join("\n");
}

function solve(text) {
  return withModel(text, (model) => {
    const run = solver.run(model, "statics");
    assertFinite(run);
    return run;
  });
}

// The rows of the ply table `kind` of the run: one object per row, its ply number and its columns by name.
function plyRows(run, kind) {
  const table = run.elementTables.find((t) => t.kind === kind && t.elementType.endsWith("Composite"));
  if (!table) return null;
  const width = table.components.length;
  return Array.from(table.ids, (id, row) => {
    const values = { element: id, ply: table.layers[row] };
    table.components.forEach((name, column) => { values[name] = table.values[row * width + column]; });
    return values;
  });
}

function assertRatio(actual, expected, tolerance, what) {
  assert.ok(Math.abs(actual / expected - 1) <= tolerance,
    `${what}: ${actual}, expected ${expected} (ratio ${(actual / expected).toFixed(10)}, tolerance ${tolerance})`);
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. The coupling of a symmetric stack.
//
// Eight plies of 0.13 mm, [0/45/−45/90]s: the heights of the plies are not exact in binary, so the coupling terms of
// the stack are sums of some 1e-17 of its membrane terms instead of zeros. They are not a coupling.
const SYMMETRIC = [0, 45, -45, 90, 90, -45, 45, 0];
const DRILLING_SHELLS = {
  ShellQuad4Drilling: ["ShellQuad4Drilling;1;1;1;2;3;4"],
  ShellTria3Drilling: ["ShellTria3Drilling;1;1;1;2;3", "ShellTria3Drilling;2;1;1;3;4"],
};

for (const [card, elements] of Object.entries(DRILLING_SHELLS)) {
  test(`a symmetric stack of plies of 0.13 has no coupling for the ${card}`, () => {
    for (const thickness of [0.13, 0.127, 0.1]) {
      const section = [laminate({ angles: SYMMETRIC, thickness }), plyMaterial(10)];
      const run = solve(strip({ elements, section }));
      assert.ok(displacement(run, 2, ALONG_Z) > 0, `plies of ${thickness}: the tip deflects along the force`);
    }
    // The stack given as its half and the Symmetry option is the same laminate.
    const section = [laminate({ angles: SYMMETRIC.slice(0, 4), thickness: 0.13, option: "Symmetry" }), plyMaterial(10)];
    const whole = solve(strip({ elements, section: [laminate({ angles: SYMMETRIC, thickness: 0.13 }), plyMaterial(10)] }));
    const mirrored = solve(strip({ elements, section }));
    assertRatio(displacement(mirrored, 2, ALONG_Z), displacement(whole, 2, ALONG_Z), 1e-12, "the mirrored stack");
  });

  test(`a stack that couples membrane and bending is refused by the ${card} with the shells that take it`, () => {
    const section = [laminate({ angles: [0, 90], thickness: 0.25 }), plyMaterial(10)];
    assert.throws(() => solve(strip({ elements, section })), (error) => {
      const message = String(error.message ?? error);
      assert.match(message, /unsupported-field/);
      assert.match(message, /ShellQuad4, ShellTria3, ShellQuad8 and ShellTria6 take the coupling/);
      assert.match(message, /symmetric stack with a blank bottomOffset/);
      assert.doesNotMatch(message, /couplingMatId/, "a laminate has no coupling material field");
      return true;
    });
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. The transverse shear of a stack of equal plies.
//
// Four plies of the same material at the same angle are a homogeneous plate of thickness t: under the shear force Q
// per unit width its transverse shear stress is the parabola 1.5 Q/t (1 − (2z/t)²) (Timoshenko and Goodier, Theory of
// Elasticity, article 21). The ply rows give it at the top face of each ply, z = −t/4, 0, t/4, t/2: 1.125, 1.5,
// 1.125 and 0 times Q/t. The bond index of a ply is that stress over the bond allowable; the top ply has none.
test("the plies of a stack of one material carry the parabola of a homogeneous plate", () => {
  const thickness = 0.25;
  const section = [laminate({ angles: [0, 0, 0, 0], thickness, theory: "Hill" }), plyMaterial(10)];
  const run = solve(strip({ section }));
  const mean = STRIP.force / STRIP.width / (4 * thickness);
  const expected = [1.125 * mean, 1.5 * mean, 1.125 * mean, 0];
  const stresses = plyRows(run, "stress");
  const indices = plyRows(run, "failureIndices");
  assert.equal(stresses.length, 4);
  stresses.forEach((row, k) => {
    if (expected[k] === 0) assert.ok(Math.abs(row.t1z) < 1e-12 * mean, `ply ${k + 1}: no shear at the free surface`);
    else assertRatio(Math.abs(row.t1z), expected[k], 1e-6, `ply ${k + 1}: shear at its top face`);
  });
  indices.slice(0, 3).forEach((row, k) =>
    assertRatio(row.bondFailureIndex, expected[k] / BOND_ALLOWABLE, 1e-6, `bond above ply ${k + 1}`));
  assert.ok(Number.isNaN(indices[3].bondFailureIndex), "the top ply has no bond above it");
});

// ---------------------------------------------------------------------------------------------------------------------
// 3. The failure indices and the strength ratios.
//
// The index of each ply is its criterion at the ply stresses of the stress table (the strains of the strain table for
// StrainTheory), written here from the formulas of docs/composites.md. The strength ratio λ brings the criterion to
// one: F(λσ) = 1, checked on the criterion itself.
function criterion(theory, [s1, s2, t12], [e1, e2, g12]) {
  const { xt, xc, yt, yc, s, f12, young1, young2, shear } = PLY;
  switch (theory) {
    case "Hill": {
      const x = s1 >= 0 ? xt : xc;
      const y = s2 >= 0 ? yt : yc;
      const product = s1 * s2 >= 0 ? xt : xc;
      return (s1 / x) ** 2 - (s1 * s2) / product ** 2 + (s2 / y) ** 2 + (t12 / s) ** 2;
    }
    case "Hoffman":
      return (1 / xt - 1 / xc) * s1 + (1 / yt - 1 / yc) * s2 + s1 ** 2 / (xt * xc) + s2 ** 2 / (yt * yc) -
        (s1 * s2) / (xt * xc) + (t12 / s) ** 2;
    case "TsaiWu":
      return (1 / xt - 1 / xc) * s1 + (1 / yt - 1 / yc) * s2 + s1 ** 2 / (xt * xc) + s2 ** 2 / (yt * yc) +
        (t12 / s) ** 2 + 2 * f12 * s1 * s2;
    case "StrainTheory":
      return Math.max(e1 >= 0 ? e1 / (xt / young1) : -e1 / (xc / young1),
        e2 >= 0 ? e2 / (yt / young2) : -e2 / (yc / young2), Math.abs(g12) / (s / shear));
    default:
      throw new Error(`no criterion ${theory}`);
  }
}

for (const theory of ["Hill", "Hoffman", "TsaiWu", "StrainTheory"])
  test(`the failure indices and strength ratios of ${theory} follow its criterion`, () => {
    const section = [laminate({ angles: [0, 45, 90, -45], thickness: 0.25, theory }), plyMaterial(10)];
    const run = solve(strip({ section, parameters: ["Parameter;StrengthRatioCompositeStress;Yes"],
      outputs: ["StressOutput;All", "StrainOutput;All"] }));
    const stresses = plyRows(run, "stress");
    const strains = plyRows(run, "strain");
    const indices = plyRows(run, "failureIndices");
    const ratios = plyRows(run, "strengthRatios");
    assert.equal(indices.length, 4);
    assert.equal(ratios.length, 4, "the strength ratio table has the rows of the failure index table");
    let largest = -Infinity;
    let smallest = Infinity;
    for (let k = 0; k < 4; ++k) {
      const stress = [stresses[k].o11, stresses[k].o22, stresses[k].t12];
      const strain = [strains[k].e11, strains[k].e22, strains[k].e12];
      const index = criterion(theory, stress, strain);
      assertRatio(indices[k].failureIndex, index, 1e-9, `${theory}, ply ${k + 1}: failure index`);
      // The criterion at λ times the stresses and strains of the ply is one.
      const ratio = ratios[k].strengthRatio;
      assert.ok(ratio > 0 && ratio < 1e19, `${theory}, ply ${k + 1}: a loaded ply has a ratio`);
      assertRatio(criterion(theory, stress.map((v) => ratio * v), strain.map((v) => ratio * v)), 1, 1e-9,
        `${theory}, ply ${k + 1}: the criterion at the strength ratio`);
      largest = Math.max(largest, indices[k].failureIndex);
      smallest = Math.min(smallest, ratio);
      if (k < 3) {
        assertRatio(ratios[k].bondStrengthRatio, 1 / indices[k].bondFailureIndex, 1e-12, `bond above ply ${k + 1}`);
        largest = Math.max(largest, indices[k].bondFailureIndex);
        smallest = Math.min(smallest, ratios[k].bondStrengthRatio);
        assert.ok(Number.isNaN(indices[k].elementFailureIndex) && Number.isNaN(ratios[k].elementStrengthRatio),
          "the element values are on the last row only");
      }
    }
    assert.ok(Number.isNaN(ratios[3].bondStrengthRatio), "the top ply has no bond above it");
    assertRatio(indices[3].elementFailureIndex, largest, 1e-12, "the element index is the largest index");
    assertRatio(ratios[3].elementStrengthRatio, smallest, 1e-12, "the element ratio is the smallest ratio");
  });

test("the strength ratio table is written only when the model asks for it", () => {
  const section = [laminate({ angles: [0, 90, 90, 0], thickness: 0.25, theory: "TsaiWu" }), plyMaterial(10)];
  assert.equal(plyRows(solve(strip({ section })), "strengthRatios"), null, "no table by default");
  const no = solve(strip({ section, parameters: ["Parameter;StrengthRatioCompositeStress;No"] }));
  assert.equal(plyRows(no, "strengthRatios"), null, "no table with No");
  const yes = solve(strip({ section, parameters: ["Parameter;StrengthRatioCompositeStress;Yes"] }));
  assert.equal(plyRows(yes, "strengthRatios").length, 4);
  // An unloaded laminate has the "no bound" ratio.
  const unloaded = strip({ section, parameters: ["Parameter;StrengthRatioCompositeStress;Yes"] })
    .replaceAll(`;${real(STRIP.force / 2)};0.;0.;1.`, ";0.;0.;0.;1.");
  for (const row of plyRows(solve(unloaded), "strengthRatios")) assert.equal(row.strengthRatio, 1e20);
});

// ---------------------------------------------------------------------------------------------------------------------
// 4. A failure theory without strengths.
test("a failure theory without the strengths of a ply warns once per property and writes no index", () => {
  const elements = ["ShellTria3;1;1;1;2;3", "ShellTria3;2;1;1;3;4"];
  const section = [laminate({ angles: [0, 90, 90, 0], thickness: 0.25, theory: "Hill" }),
    plyMaterial(10, { strengths: false })];
  const run = solve(strip({ elements, section }));
  const warnings = run.diagnostics.filter((d) => d.code === "failure-index-not-written");
  assert.equal(warnings.length, 1, JSON.stringify(run.diagnostics));
  assert.equal(warnings[0].severity, "warning");
  assert.equal(warnings[0].card, "CompProp");
  assert.equal(warnings[0].id, 1);
  assert.equal(warnings[0].field, "failureTheory");
  assert.match(warnings[0].message, /material 10 of ply 1/);
  assert.equal(plyRows(run, "failureIndices"), null, "no failure index table");
  assert.equal(plyRows(run, "stress").length, 8, "the ply stresses are written");
  // With the strengths, or without a theory, there is nothing to warn of.
  for (const quiet of [[laminate({ angles: [0, 90, 90, 0], thickness: 0.25, theory: "Hill" }), plyMaterial(10)],
    [laminate({ angles: [0, 90, 90, 0], thickness: 0.25 }), plyMaterial(10, { strengths: false })]])
    assert.ok(!solve(strip({ elements, section: quiet })).diagnostics.some((d) => d.code === "failure-index-not-written"));
});

// ---------------------------------------------------------------------------------------------------------------------
// 5. The transverse shear of an AnisoMat2D ply.
//
// A thick strip (thickness 20 for a length of 100) of one ply of an isotropic material, E = 70 000, ν = 0.3, written
// in three ways: an AnisoMat2D (the plane-stress matrix, which holds no transverse shear modulus), an OrthoMat2D whose
// transverse shear moduli are blank (rigid), an IsoMat (its shear modulus G is also the transverse one). The
// transverse shear is a visible part of the deflection: E h² / (4 κ G L²) of the bending part with κ = 5/6, 3 % here.
// - ClassicFormulation: the AnisoMat2D ply is rigid in transverse shear, as the OrthoMat2D one;
// - MixedHybridFormulation: it takes its in-plane shear modulus G33 = G, as the IsoMat one.
const THICK = { length: 100, width: 50, thickness: 20, young: 70000, poisson: 0.3 };

function thickStrip(material, parameters = []) {
  const { length, width, thickness } = THICK;
  return strip({ section: [`CompProp;1;;;;;;;;10;${real(thickness)};0.;Yes`, material], parameters, outputs: [] })
    .replace(`Node;2;;${real(STRIP.length)};0.;0.`, `Node;2;;${real(length)};0.;0.`)
    .replace(`Node;3;;${real(STRIP.length)};${real(STRIP.width)};0.`, `Node;3;;${real(length)};${real(width)};0.`);
}

function isotropicCards() {
  const { young, poisson } = THICK;
  const shear = young / (2 * (1 + poisson));
  const plane = young / (1 - poisson * poisson);
  return {
    aniso: `AnisoMat2D;10;${plane};${plane * poisson};0.;${plane};0.;${shear}`,
    orthoRigid: `OrthoMat2D;10;${real(young)};${real(young)};${real(poisson)};${shear}`,
    iso: `IsoMat;10;${real(young)};${shear};${real(poisson)}`,
  };
}

test("an AnisoMat2D ply is rigid in transverse shear in the classic formulation", () => {
  const cards = isotropicCards();
  const deflection = (material) => displacement(solve(thickStrip(material)), 2, ALONG_Z);
  const aniso = deflection(cards.aniso);
  assertRatio(aniso, deflection(cards.orthoRigid), 1e-10, "the AnisoMat2D ply against the rigid OrthoMat2D ply");
  const flexible = deflection(cards.iso);
  assert.ok(flexible / aniso - 1 > 5e-3, `the shear of the IsoMat ply shows: ${flexible} against ${aniso}`);
});

test("an AnisoMat2D ply takes its in-plane shear modulus in the mixed hybrid formulation", () => {
  const cards = isotropicCards();
  const parameters = ["Parameter;ElementFormulation;MixedHybridFormulation", "Parameter;GeometryDeclarations;Given"];
  const deflection = (material) => displacement(solve(thickStrip(material, parameters)), 2, ALONG_Z);
  const aniso = deflection(cards.aniso);
  assertRatio(aniso, deflection(cards.iso), 1e-10, "the AnisoMat2D ply against the IsoMat ply");
  const rigid = deflection(cards.orthoRigid);
  assert.ok(aniso / rigid - 1 > 5e-3, `the shear of the AnisoMat2D ply shows: ${aniso} against ${rigid}`);
});

// ---------------------------------------------------------------------------------------------------------------------
// 6. A failure theory the solver does not compute.
test("a Hashin failure theory is refused by its name in the model", () => {
  const section = [laminate({ angles: [0, 90, 90, 0], thickness: 0.25, theory: "HashinTape" }), plyMaterial(10)];
  assert.throws(() => solve(strip({ section })), (error) => {
    const message = String(error.message ?? error);
    assert.match(message, /failureTheory blank, Hill, Hoffman, TsaiWu or StrainTheory/);
    assert.match(message, /found HashinTape/);
    return true;
  });
});

// ---------------------------------------------------------------------------------------------------------------------
// 7. The transverse shear stiffness of a stack, its convention and the report of a run.
//
// A thick cross-ply strip, four plies of 5 for a length of 100. Along x the plies at 0 degrees are the stiff ones
// (E1) and shear with G1Z, those at 90 degrees are the soft ones (E2) and shear with G2Z; along y the roles swap.
// - Equilibrium (the default): the shear stress at a height z under a unit shear force is the first moment of the
//   ply moduli below z over the bending stiffness (the stack is symmetric: its neutral axis is its middle), and the
//   stiffness H follows from the shear energy, 1/H = ∫ τ²/G dz (Whitney, Structural Analysis of Laminated Anisotropic
//   Plates, 1987, section 4.6). Integrated here by the midpoint rule on thin slices, apart from the solver's own sums.
// - FiveSixths: H = 5/6 Σ G t, whatever the order of the plies.
// The report of a run gives H and the factor H / Σ G t of each laminate. The tip deflection of the cantilever differs
// between the two conventions by its shear part, P L / (H b), the bending part being the same.
const CROSS_PLY = { angles: [0, 90, 90, 0], ply: 5, length: 100 };
const FIVE_SIXTHS = 5 / 6;

// The plane-stress moduli of the ply along its fibres and across them: E / (1 − ν12 ν21), ν21 = ν12 E2 / E1.
function plyModuli() {
  const reduction = 1 - (PLY.poisson * PLY.poisson * PLY.young2) / PLY.young1;
  return { along: PLY.young1 / reduction, across: PLY.young2 / reduction };
}

// The plies of the cross-ply stack seen along the axis `axis` (0: x, 1: y): modulus, shear modulus and thickness.
function crossPlyAlong(axis) {
  const { along, across } = plyModuli();
  return CROSS_PLY.angles.map((angle) => {
    const isAlongFibres = (angle === 0) === (axis === 0);
    return { modulus: isAlongFibres ? along : across, shear: isAlongFibres ? PLY.shear1Z : PLY.shear2Z,
      thickness: CROSS_PLY.ply };
  });
}

// H of a symmetric stack by the midpoint rule: τ(z) = S(z) / D, S the first moment of the moduli below z.
function equilibriumShearStiffness(plies, slicesPerPly = 4000) {
  const total = plies.reduce((sum, ply) => sum + ply.thickness, 0);
  let [z, bending] = [-total / 2, 0];
  for (const ply of plies) {
    bending += (ply.modulus * ((z + ply.thickness) ** 3 - z ** 3)) / 3;
    z += ply.thickness;
  }
  let [moment, compliance] = [0, 0];
  z = -total / 2;
  for (const ply of plies) {
    const step = ply.thickness / slicesPerPly;
    for (let k = 0; k < slicesPerPly; ++k) {
      const middle = z + step / 2;
      const atMiddle = moment + ply.modulus * middle * (step / 2);
      compliance += (step * (atMiddle / bending) ** 2) / ply.shear;
      moment += ply.modulus * middle * step;
      z += step;
    }
  }
  return 1 / compliance;
}

const shearSum = (plies) => plies.reduce((sum, ply) => sum + ply.shear * ply.thickness, 0);

function crossPlyStrip({ angles = CROSS_PLY.angles, parameters = [] } = {}) {
  const section = [laminate({ angles, thickness: CROSS_PLY.ply }), plyMaterial(10)];
  return strip({ section, parameters, outputs: [] })
    .replace(`Node;2;;${real(STRIP.length)};0.;0.`, `Node;2;;${real(CROSS_PLY.length)};0.;0.`)
    .replace(`Node;3;;${real(STRIP.length)};${real(STRIP.width)};0.`,
      `Node;3;;${real(CROSS_PLY.length)};${real(STRIP.width)};0.`);
}

const FIVE_SIXTHS_PARAMETER = ["Parameter;LaminateTransverseShear;FiveSixths"];

test("the report of a run gives the transverse shear stiffness of each laminate and its factor", () => {
  const run = solve(crossPlyStrip());
  assert.equal(run.report.laminateShear.length, 1);
  const [row] = run.report.laminateShear;
  assert.equal(row.card, "CompProp");
  assert.equal(row.propId, 1);
  for (const axis of [0, 1]) {
    const plies = crossPlyAlong(axis);
    const expected = equilibriumShearStiffness(plies);
    assertRatio(row.stiffness[axis], expected, 1e-6, `H along axis ${axis}`);
    assertRatio(row.factor[axis], expected / shearSum(plies), 1e-6, `the factor along axis ${axis}`);
  }
  // Neither is the 5/6 of a homogeneous plate: 0.778 along x and 0.660 along y.
  assertRatio(row.factor[0], 0.778, 1e-3, "the factor along x");
  assertRatio(row.factor[1], 0.660, 1e-3, "the factor along y");

  // One material: the factor of a homogeneous plate along both axes.
  const [uniform] = solve(crossPlyStrip({ angles: [0, 0, 0, 0] })).report.laminateShear;
  const thickness = 4 * CROSS_PLY.ply;
  assertRatio(uniform.factor[0], FIVE_SIXTHS, 1e-12, "one material, along x");
  assertRatio(uniform.factor[1], FIVE_SIXTHS, 1e-12, "one material, along y");
  assertRatio(uniform.stiffness[0], FIVE_SIXTHS * PLY.shear1Z * thickness, 1e-12, "5/6 G1Z t");
  assertRatio(uniform.stiffness[1], FIVE_SIXTHS * PLY.shear2Z * thickness, 1e-12, "5/6 G2Z t");

  // A model without a laminate has no row.
  const homogeneous = solve(strip({ section: ["ShellProp;1;10;1.;10;;10", "IsoMat;10;70000.;;0.3"], outputs: [] }));
  assert.deepEqual(homogeneous.report.laminateShear, []);
});

test("LaminateTransverseShear = FiveSixths takes 5/6 of the plies' shear moduli times their thicknesses", () => {
  const equilibrium = solve(crossPlyStrip());
  const named = solve(crossPlyStrip({ parameters: ["Parameter;LaminateTransverseShear;Equilibrium"] }));
  const fiveSixths = solve(crossPlyStrip({ parameters: FIVE_SIXTHS_PARAMETER }));
  assert.equal(displacement(named, 2, ALONG_Z), displacement(equilibrium, 2, ALONG_Z), "Equilibrium is the default");

  const [row] = fiveSixths.report.laminateShear;
  for (const axis of [0, 1]) {
    assertRatio(row.stiffness[axis], FIVE_SIXTHS * shearSum(crossPlyAlong(axis)), 1e-12, `H along axis ${axis}`);
    assertRatio(row.factor[axis], FIVE_SIXTHS, 1e-12, `the factor along axis ${axis}`);
  }
  // The shear part of the tip deflection, P L / (H b), is all that changes.
  const [before] = equilibrium.report.laminateShear;
  const shearPart = (stiffness) => (STRIP.force * CROSS_PLY.length) / (stiffness * STRIP.width);
  const expected = shearPart(before.stiffness[0]) - shearPart(row.stiffness[0]);
  const found = displacement(equilibrium, 2, ALONG_Z) - displacement(fiveSixths, 2, ALONG_Z);
  assert.ok(expected > 1e-3 * displacement(equilibrium, 2, ALONG_Z), "the two conventions differ visibly here");
  assertRatio(found, expected, 1e-6, "the difference of the tip deflections");

  // The stresses of the plies are those of the equilibrium in both conventions.
  const withStress = (parameters) => plyRows(solve(crossPlyStrip({ parameters })
    .replace("DisplacementOutput;All", "DisplacementOutput;All\nStressOutput;All")), "stress");
  const [a, b] = [withStress([]), withStress(FIVE_SIXTHS_PARAMETER)];
  a.forEach((ply, k) => assertRatio(b[k].t1z || 1, ply.t1z || 1, 1e-9, `the shear stress of ply ${k + 1}`));
});

test("a LaminateTransverseShear that is neither Equilibrium nor FiveSixths is refused", () => {
  const text = crossPlyStrip({ parameters: ["Parameter;LaminateTransverseShear;Sometimes"] });
  withModel(text, (model) => {
    const refusals = model.checkSupport("statics").refusals.filter((d) => d.code === "invalid-parameter");
    assert.equal(refusals.length, 1, JSON.stringify(model.checkSupport("statics").refusals));
    assert.equal(refusals[0].field, "LaminateTransverseShear");
    assert.match(refusals[0].text, /Equilibrium/);
    assert.match(refusals[0].text, /FiveSixths/);
    assert.match(refusals[0].text, /found Sometimes/);
    assert.throws(() => solver.run(model, "statics"), /LaminateTransverseShear/);
  });
});

test("a run tells the laminates whose transverse shear stiffness is not 5/6 of their sheared sum", () => {
  const notes = (run) => run.diagnostics.filter((d) => d.code === "laminate-shear-factor");
  const told = notes(solve(crossPlyStrip()));
  assert.equal(told.length, 1, JSON.stringify(told));
  assert.equal(told[0].severity, "information");
  assert.equal(told[0].card, "CompProp");
  assert.equal(told[0].id, 1);
  assert.match(told[0].message, /^1 of the 1 laminate properties/);
  assert.match(told[0].message, /the furthest: CompProp 1, factor 0\.660\d* along y/);
  assert.match(told[0].message, /LaminateTransverseShear = FiveSixths/);
  // One material has the factor 5/6, and so has every stack under FiveSixths: nothing to tell.
  assert.deepEqual(notes(solve(crossPlyStrip({ angles: [0, 0, 0, 0] }))), []);
  assert.deepEqual(notes(solve(crossPlyStrip({ parameters: FIVE_SIXTHS_PARAMETER }))), []);
});
