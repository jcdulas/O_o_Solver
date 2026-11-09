// End-to-end test of the WebAssembly package: a model file is read by the O_o_Reader module, its tables are handed to the
// solver, and the results are compared with the analytic solutions of a cantilever plate.
//
//   node --test wasm/test/*.test.mjs
//
// The reader package is taken from OOR_READER_DIST, or from the sibling O_o_Reader build (../O_o_Reader/build/wasm/
// dist); the solver package from OOS_SOLVER_DIST, or build/wasm/dist.

import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";

// A cantilever plate in the XY plane, clamped along x = 0, meshed with nx × ny four-node shells (or two three-node
// shells per square), written as a model file (O_o_Reader docs/model-format.md). The model file and the interface name
// the cards and diagnostic fields by their exact neutral names.
const PLATE = { length: 2.0, width: 0.4, thickness: 0.02, young: 7.0e10, poisson: 0.3, density: 2700.0 };
const TIP_FORCE = -100.0; // the total vertical force at the free edge

// A real as a model file writes it: with a decimal point or an exponent (`1` would be an integer).
function real(value) {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.`;
}

// One record: the fields joined by ';' (a blank field is "").
function record(...fields) {
  return fields.join(";");
}

function plateModel({ nx, ny, elements, solution, caseRecords }) {
  const lines = ["[executive]", record("SolutionSequence", solution), "[case]", record("TitleText", "CANTILEVER PLATE"),
    ...caseRecords, "[model]"];
  const nodeId = (i, j) => j * (nx + 1) + i + 1;
  for (let j = 0; j <= ny; ++j) {
    for (let i = 0; i <= nx; ++i) {
      lines.push(record("Node", nodeId(i, j), "", real((PLATE.length * i) / nx), real((PLATE.width * j) / ny), "0."));
    }
  }
  let elementId = 1;
  for (let j = 0; j < ny; ++j) {
    for (let i = 0; i < nx; ++i) {
      const [a, b, c, d] = [nodeId(i, j), nodeId(i + 1, j), nodeId(i + 1, j + 1), nodeId(i, j + 1)];
      if (elements.startsWith("ShellQuad")) {
        lines.push(record(elements, elementId++, 1, a, b, c, d));
      } else {
        lines.push(record(elements, elementId++, 1, a, b, c));
        lines.push(record(elements, elementId++, 1, a, c, d));
      }
    }
  }
  lines.push(record("ShellProp", 1, 1, real(PLATE.thickness), 1, "", 1));
  lines.push(record("IsoMat", 1, real(PLATE.young), "", real(PLATE.poisson), real(PLATE.density)));
  for (let j = 0; j <= ny; ++j) lines.push(record("SinglePointConstrList", 1, 123456, nodeId(0, j)));
  for (let j = 0; j <= ny; ++j) {
    // The end nodes carry half the share of the inner ones: a uniform line load along the free edge.
    const share = (j === 0 || j === ny ? 0.5 : 1.0) / ny;
    lines.push(record("PointForceLoad", 2, nodeId(nx, j), "", real(TIP_FORCE * share), "0.", "0.", "1."));
  }
  lines.push(record("LanczosEigenMethod", 3, "", "", 3));
  return lines.join("\n") + "\n";
}

function modelOf(modelText, name) {
  const file = `/work/${name}.oos`;
  oor.writeFile(file, modelText);
  const deck = oor.readModel(file);
  try { return solver.modelFromReader(oor, deck); } finally { deck.delete(); }
}

// The vertical displacement of every point of the free edge, from the displacement table of a static run.
function tipDeflections(run, nx, ny) {
  const table = run.nodalTables.find((t) => t.kind === "displacements" && t.subcase === 1);
  assert.ok(table, "the run gives a displacement table");
  const tip = [];
  for (let j = 0; j <= ny; ++j) {
    const id = j * (nx + 1) + nx + 1;
    const row = table.ids.indexOf(id);
    tip.push(table.values[row * 6 + 2]);
  }
  return tip;
}

// Euler-Bernoulli: the deflection of a cantilever under an end force, δ = P L³ / (3 E I). A plate of width b bends
// with the stiffness E I between E b t³ / 12 (a narrow beam) and that divided by (1 − ν²) (a wide plate).
const inertia = (PLATE.width * PLATE.thickness ** 3) / 12;
const beamDeflection = (TIP_FORCE * PLATE.length ** 3) / (3 * PLATE.young * inertia);

test("the solver module loads and names itself", () => {
  assert.match(solver.version(), /^O_o_Solver \d+\.\d+\.\d+$/);
});

test("tables using the previous data contract are refused", () => {
  oor.writeFile("/work/contract.oos", "[model]\nNode;1;;0.;0.;0.\n");
  const deck = oor.readModel("/work/contract.oos");
  const table = oor.runQuery("nodesTable", { deck });
  const input = new solver.core.ModelInput();
  try {
    const bytes = table.toPackage().slice();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 16; // OOTP header: magic, format version and header byte length.
    const text = () => {
      const length = view.getUint32(offset, true);
      offset += 4;
      const start = offset;
      offset += length;
      return { start, length, value: new TextDecoder().decode(bytes.subarray(start, offset)) };
    };
    text(); // root name
    text(); // Arrow format
    offset += 8; // schema flags
    const count = view.getUint32(offset, true);
    offset += 4;
    let changed = false;
    for (let index = 0; index < count; ++index) {
      const key = text();
      const value = text();
      if (key.value === "oor_reader.data_contract_version") {
        assert.equal(value.value, "75");
        bytes.set(new TextEncoder().encode("74"), value.start);
        changed = true;
      }
    }
    assert.ok(changed, "the package identifies its data contract");
    assert.throws(() => input.setTable("nodes", bytes), /data_contract_version.*75.*74/);
  } finally {
    input.delete();
    table.delete();
    deck.delete();
  }
});

for (const elements of ["ShellQuad4", "ShellTria3", "ShellQuad4Drilling", "ShellTria3Drilling"]) {
  test(`a cantilever plate of ${elements} bends as a beam (statics)`, () => {
    const [nx, ny] = [20, 4];
    const model = modelOf(
      plateModel({ nx, ny, elements, solution: 101,
        caseRecords: ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All",
          "StressOutput;All"] }),
      `static-${elements}`);
    const support = model.checkSupport("statics");
    assert.equal(support.isComplete, true, support.refusals.map((d) => d.text).join("\n"));
    const run = solver.run(model, "statics");
    const tip = tipDeflections(run, nx, ny);
    const mean = tip.reduce((sum, value) => sum + value, 0) / tip.length;
    // Compared by magnitude: the plate is at most as flexible as the beam and at least as flexible as the wide plate.
    const plateLimit = beamDeflection * (1 - PLATE.poisson ** 2);
    const magnitude = Math.abs(mean);
    assert.ok(Math.sign(mean) === Math.sign(TIP_FORCE), `the plate bends the way it is pushed (${mean})`);
    assert.ok(magnitude >= 0.97 * Math.abs(plateLimit) && magnitude <= 1.03 * Math.abs(beamDeflection),
      `tip deflection ${mean} expected between ${plateLimit} (wide plate) and ${beamDeflection} (beam)`);
    assert.ok(run.elementTables.some((t) => t.kind === "stress" && t.elementType === elements),
      "the run gives the stresses asked for, under the neutral element type");
    model.delete();
  });
}

test("a cantilever plate of ShellQuad4 vibrates at the beam's first frequency (normal modes)", () => {
  const [nx, ny] = [20, 4];
  const model = modelOf(
    plateModel({ nx, ny, elements: "ShellQuad4", solution: 103,
      caseRecords: ["SubcaseStart;1", "SinglePointConstrSelection;1", "EigenMethodSelection;3", "DisplacementOutput;All"] }),
    "modes");
  const run = solver.run(model, "modes");
  const first = run.eigenvalues[0].rows[0];
  // f1 = (1.8751² / 2π) √(E I / (ρ A L⁴)), the first bending mode of a cantilever beam.
  const area = PLATE.width * PLATE.thickness;
  const beamFrequency =
    (1.8751 ** 2 / (2 * Math.PI)) * Math.sqrt((PLATE.young * inertia) / (PLATE.density * area * PLATE.length ** 4));
  const ratio = first.cycles / beamFrequency;
  assert.ok(ratio > 0.98 && ratio < 1.1, `first frequency ${first.cycles} Hz, beam ${beamFrequency} Hz`);
  model.delete();
});

const frameKinds = ["", "OffsetsGlobalGlobalGlobal", "OffsetsGlobalGlobalOrient", "OffsetsGlobalOrientGlobal",
  "OffsetsGlobalOrientOrient", "OffsetsBasicGlobalGlobal", "OffsetsBasicGlobalOrient", "OffsetsBasicOrientGlobal",
  "OffsetsBasicOrientOrient"];
for (const card of ["Bar2", "Beam2"]) {
  for (const kind of frameKinds) {
    test(`${card} reads offset frame ${kind || "default"}`, () => {
      const property = card === "Bar2" ? "BarProp;1;1;0.01;1.e-6;2.e-6;1.e-6"
        : "BeamProp;1;1;0.01;1.e-6;2.e-6;;1.e-6";
      const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
        "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All", "[model]",
        "Node;1;;0.;0.;0.", "Node;2;;2.;0.;0.", record(card, 1, 1, 1, 2, "0.", "", "1.", "0.", kind),
        property, "IsoMat;1;7.e10;;0.3", "SinglePointConstrList;1;123456;1",
        "PointForceLoad;2;2;;100.;1.;0.;0."].join("\n") + "\n";
      const model = modelOf(text, `${card}-${kind || "default"}`);
      try {
        const support = model.checkSupport("statics");
        assert.equal(support.isComplete, true, support.refusals.map((d) => d.text).join("\n"));
        const run = solver.run(model, "statics");
        const table = run.nodalTables.find((t) => t.kind === "displacements");
        const extension = table.values[table.ids.indexOf(2) * 6];
        // Uniform axial strain: u = F L / (E A), independent of the frame of zero offsets.
        const expected = 100 * 2 / (7.e10 * 0.01);
        assert.ok(Math.abs(extension / expected - 1) < 1.e-10, `${card} ${kind}: ${extension}`);
      } finally {
        model.delete();
      }
    });
  }
}

test("each element card has the ClassicFormulation formulation by default, and a formulation it lacks is refused", () => {
  const formulations = solver.elementFormulations();
  const quad = formulations.find((entry) => entry.card === "ShellQuad4");
  assert.ok(quad, "the CQUAD4 is listed under its neutral type");
  assert.ok(!formulations.some((entry) => entry.card === "CQUAD4"), "no card is listed by its deck keyword");
  assert.equal(quad.formulations[0], "ClassicFormulation");
  for (const entry of formulations) assert.ok(entry.formulations.length >= 1, `${entry.card} has a formulation`);

  const [nx, ny] = [4, 2];
  const subcase = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All"];
  const modelWith = (params) => plateModel({ nx, ny, elements: "ShellQuad4", solution: 101,
    caseRecords: [...params, ...subcase] });
  const refusalCodes = (model) => model.checkSupport("statics").refusals.map((d) => d.code);

  // In the model: the parameter of the formulation of the four-node shell with a known name, an unknown one, and in
  // one subcase.
  const known = modelOf(modelWith(["SubcaseParameter;ShellQuad4Formulation;ClassicFormulation"]), "formulation-known");
  assert.deepEqual(refusalCodes(known), []);
  known.delete();
  const unknown = modelOf(modelWith(["SubcaseParameter;ShellQuad4Formulation;NOSUCH"]), "formulation-unknown");
  assert.deepEqual(refusalCodes(unknown), ["unknown-formulation"]);
  // Diagnostics and errors use the neutral names of the model format.
  const [refusal] = unknown.checkSupport("statics").refusals;
  assert.match(refusal.text, /ShellQuad4/); // The formulation parameter and the element card
  assert.doesNotMatch(refusal.text, /CQUAD4/);
  assert.doesNotMatch(refusal.text, /[{}]/, "no marker of the solver's messages is left: {command:...} and the others");
  assert.throws(() => solver.run(unknown, "statics"),
    (error) => /NOSUCH/.test(error.message) && /ShellQuad4/.test(error.message) && !/CQUAD4/.test(error.message));
  unknown.delete();
  const perSubcase = plateModel({ nx, ny, elements: "ShellQuad4", solution: 101,
    caseRecords: ["SubcaseStart;1", "SubcaseParameter;ShellQuad4Formulation;ClassicFormulation", ...subcase.slice(1)] });
  const subcaseModel = modelOf(perSubcase, "formulation-subcase");
  assert.deepEqual(refusalCodes(subcaseModel), ["unknown-formulation"]);
  subcaseModel.delete();

  // From the program, by neutral card type: the choice wins over the parameter of the model.
  oor.writeFile("/work/formulation-api.oos", modelWith(["SubcaseParameter;ShellQuad4Formulation;NOSUCH"]));
  const deck = oor.readModel("/work/formulation-api.oos");
  const chosen = solver.modelFromReader(oor, deck, { elementFormulations: { ShellQuad4: "ClassicFormulation" } });
  assert.deepEqual(refusalCodes(chosen), []);
  chosen.delete();
  const wrong = solver.modelFromReader(oor, deck, { elementFormulations: { ShellQuad4: "NOSUCH" } });
  assert.deepEqual(refusalCodes(wrong), ["unknown-formulation"]);
  const [chosenRefusal] = wrong.checkSupport("statics").refusals;
  assert.equal(chosenRefusal.card, "ShellQuad4", "a refusal names the neutral card type");
  assert.doesNotMatch(chosenRefusal.text, /CQUAD4/);
  wrong.delete();
  // A deck keyword or an unknown card type is not a card type of the interface.
  for (const card of ["CQUAD4", "shellQuad4", "NoSuchElem"]) {
    assert.throws(() => solver.modelFromReader(oor, deck, { elementFormulations: { [card]: "ClassicFormulation" } }),
      /neutral name/, `${card} is refused`);
  }
  deck.delete();
});

for (const card of ["Rod2", "Rod2InlineSection"]) {
  test(`${card} takes an axial deformation and keeps its type in the results`, () => {
    const element = card === "Rod2" ? "Rod2;1;1;1;2" : "Rod2InlineSection;1;1;2;1;0.01;1.e-6";
    const text = ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
      "SinglePointConstrSelection;1", "DeformationLoadSelection;2", "DisplacementOutput;All", "StressOutput;All",
      "[model]", "Node;1;;0.;0.;0.", "Node;2;;2.;0.;0.", element, "RodProp;1;1;0.01;1.e-6",
      "IsoMat;1;7.e10;;0.3", "SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;2",
      "AxialDeformationLoad;2;1;0.001"].join("\n") + "\n";
    const model = modelOf(text, `${card}-deformation`);
    try {
      const support = model.checkSupport("statics");
      assert.equal(support.isComplete, true, support.refusals.map((item) => item.text).join("\n"));
      const run = solver.run(model, "statics");
      const displacements = run.nodalTables.find((table) => table.kind === "displacements");
      assert.ok(displacements, "the run provides displacements");
      const row = displacements.ids.indexOf(2);
      assert.ok(row >= 0, "node 2 has a result row");
      assert.ok(Math.abs(displacements.values[row * 6] - 0.001) < 1.e-12,
        "the free end extends by the imposed axial deformation");
      assert.ok(run.elementTables.some((table) => table.kind === "stress" && table.elementType === card),
        "the element result type is exactly the model record type");
    } finally {
      model.delete();
    }
  });
}
