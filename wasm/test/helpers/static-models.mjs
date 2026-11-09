import assert from "node:assert/strict";
import { oor, solver } from "./runtime.mjs";

export function* product(axes, values = {}, remaining = Object.entries(axes)) {
  if (!remaining.length) { yield values; return; }
  const [[key, choices], ...rest] = remaining;
  for (const value of choices) yield* product(axes, { ...values, [key]: value }, rest);
}
export function label(values) { return Object.entries(values).map(([key, value]) => `${key}=${value}`).join(" "); }
export function real(value) { return Number.isInteger(value) ? `${value}.` : String(value); }
export function near(actual, expected, message = "") {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 2e-8 * Math.max(1e-6, Math.abs(expected)),
    `${message}: ${actual} expected ${expected}`);
}
export function withModel(text, action) {
  oor.writeFile("/work/matrix.oos", text);
  const deck = oor.readModel("/work/matrix.oos", { strict: true });
  let model;
  try { model = solver.modelFromReader(oor, deck); return action(model, deck); }
  finally { model?.delete(); deck.delete(); }
}
export function displacement(run, node = 2, component = 0, subcase = 1) {
  const table = run.nodalTables.find(t => t.kind === "displacements" && t.subcase === subcase);
  assert.ok(table, `displacements of subcase ${subcase}`);
  const row = table.ids.indexOf(node);
  assert.ok(row >= 0, `point ${node} has a displacement`);
  return table.values[row * 6 + component];
}
export function assertFinite(run) {
  assert.ok(!run.diagnostics.some(d => d.severity === "error"), JSON.stringify(run.diagnostics));
  for (const table of run.nodalTables)
    assert.ok(table.values.every(Number.isFinite), `${table.kind} contains a nonfinite result`);
}
export const LINE_ELEMENTS = ["Rod2", "Rod2InlineSection", "Tube2", "Bar2", "Beam2"];
export const MASS_SETS = ["none", "unselected", "direct", "list", "union"];

export function massSet(kind, targetType, value, target = 1) {
  if (kind === "none") return { control: [], bulk: [], added: 0 };
  const bulk = kind === "union"
    ? [`NonStructMass;21;${targetType};${target};${real(value / 2)}`,
       `NonStructMass1;22;${targetType};${real(value / 2)};${target}`, "NonStructMassUnion;20;21;22"]
    : kind === "list" ? [`NonStructMass1;20;${targetType};${real(value)};${target}`]
    : [`NonStructMass;20;${targetType};${target};${real(value)}`];
  return { control: kind === "unselected" ? [] : ["NonStructMassSelection;20"], bulk,
    added: kind === "unselected" ? 0 : value };
}

export function lineModel({ element = "Rod2", load = "force", nsm = "none", coupled = 0, weight = 1,
  constraint = "list", scope = "local", propertyMass = 0, sign = 1, modulus = 1000, length = 3 } = {}) {
  let propertyType, elementRecord, property;
  let area = 2;
  if (element === "Rod2") {
    propertyType = "RodProp";
    elementRecord = "Rod2;1;1;1;2";
    property = `RodProp;1;1;2.;;;${real(propertyMass)}`;
  } else if (element === "Rod2InlineSection") {
    propertyType = "Element";
    elementRecord = `Rod2InlineSection;1;1;2;1;2.;;;${real(propertyMass)}`;
  } else if (element === "Tube2") {
    area = Math.PI;
    propertyType = "TubeProp";
    elementRecord = "Tube2;1;1;1;2";
    property = `TubeProp;1;1;2.;1.;${real(propertyMass)}`;
  } else {
    propertyType = element === "Bar2" ? "BarProp" : "BeamProp";
    elementRecord = `${element};1;1;1;2;0.;;1.;0.`;
    property = element === "Bar2" ? `BarProp;1;1;2.;1.;2.;1.;${real(propertyMass)}`
      : `BeamProp;1;1;2.;1.;2.;;1.;${real(propertyMass)}`;
  }
  const mass = massSet(nsm, propertyType, 2);
  const selection = ["SinglePointConstrSelection;1", ...mass.control];
  const bulk = ["Node;1;;0.;0.;0.", `Node;2;;${real(length)};0.;0.`, elementRecord,
    property, `IsoMat;1;${real(modulus)};;0.3;1.;1.e-5;0.`,
    `Parameter;CoupledMass;${coupled}`, `Parameter;WeightToMass;${real(weight)}`, ...mass.bulk].filter(Boolean);
  if (constraint === "list") bulk.push("SinglePointConstrList;1;123456;1", "SinglePointConstrList;1;23456;2");
  else if (constraint === "point") bulk.push("SinglePointConstr;1;1;123456", "SinglePointConstr;1;2;23456");
  else bulk.push("SinglePointConstrList;11;123456;1", "SinglePointConstrList;12;23456;2",
    "SinglePointConstrUnion;1;11;12");
  let expected = 0;
  if (load === "force" || load === "force1") {
    selection.push("StaticLoadSelection;2");
    bulk.push(load === "force" ? `PointForceLoad;2;2;;${real(sign * 10)};1.;0.;0.`
      : `PointForceLoad1;2;2;${real(sign * 10)};1;2`);
    expected = sign * 10 * length / (modulus * area);
  } else if (load === "gravity" || load === "acceleration") {
    selection.push("StaticLoadSelection;2");
    bulk.push(load === "gravity" ? `GravityLoad;2;;${real(sign * 10)};1.;0.;0.`
      : `AccelNodeLoad;2;;${real(sign * 10)};1.;0.;0.;;;1;2`);
    expected = (area + propertyMass + mass.added) * sign * 10 * length ** 2 / (2 * modulus * area) * weight;
  } else if (load === "thermal") {
    selection.push("TemperatureSelection;3;TemperatureLoadPart");
    bulk.push(`DefaultTemperatureLoad;3;${real(sign * 20)}`);
    expected = 1e-5 * sign * 20 * length;
  } else if (load === "deformation") {
    selection.push("DeformationLoadSelection;4");
    bulk.push(`AxialDeformationLoad;4;1;${real(sign * 0.006)}`);
    expected = sign * 0.006;
  }
  const control = scope === "global" ? [...selection, "SubcaseStart;1"] : ["SubcaseStart;1", ...selection];
  return { text: ["[executive]", "SolutionSequence;101", "[case]", ...control, "DisplacementOutput;All",
    "SinglePointConstrForceOutput;All", "[model]", ...bulk, ""].join("\n"), expected, area };
}
