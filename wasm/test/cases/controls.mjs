import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver } from "../helpers/runtime.mjs";
import { product, label, withModel, displacement, near, lineModel, LINE_ELEMENTS, MASS_SETS } from "../helpers/static-models.mjs";
import { SURFACE_ELEMENTS, SOLID_ELEMENTS } from "../helpers/surface-models.mjs";
import { CONNECTION_ELEMENTS, MASS_ELEMENTS } from "./connections.mjs";

test("every registered element family has an executable small-model fixture", () => {
  const covered = [...LINE_ELEMENTS, ...SURFACE_ELEMENTS, ...SOLID_ELEMENTS, ...CONNECTION_ELEMENTS, ...MASS_ELEMENTS];
  assert.equal(new Set(covered).size, covered.length);
  assert.deepEqual(covered.sort(), solver.elementFormulations().map(x => x.card).sort());
});

for (const values of product({ combination: ["direct", "one", "nested"], load: ["force", "gravity"],
  nsm: MASS_SETS, scope: ["local", "global"] }))
  test(`load combinations ${label(values)}`, () => {
    const built = lineModel(values);
    let text = built.text;
    const factor = values.combination === "direct" ? 1 : 3;
    if (factor !== 1) {
      text += "LoadCombination;30;2.;1.5;2\n";
      if (values.combination === "nested") text += "LoadCombination;40;0.5;2.;30\n";
      text = text.replace("StaticLoadSelection;2", `StaticLoadSelection;${values.combination === "one" ? 30 : 40}`);
    }
    withModel(text, model => near(displacement(solver.run(model, "statics")), factor * built.expected));
  });

for (const values of product({ constraint: ["mpc", "mpcUnion"], load: ["force", "gravity"],
  nsm: MASS_SETS, scope: ["local", "global"] }))
  test(`multipoint constraints ${label(values)}`, () => {
    const built = lineModel(values);
    let text = built.text.replace("SinglePointConstrSelection;1", `SinglePointConstrSelection;1\nMultiPointConstrSelection;${
      values.constraint === "mpc" ? 10 : 100}`);
    text += "Node;3;;3.;0.;0.\nSinglePointConstrList;1;23456;3\nMultiPointConstr;10;3;1;1.;2;1;-1.\n";
    if (values.constraint === "mpcUnion") text += "MultiPointConstrUnion;100;10\n";
    if (values.load === "force") text = text.replace("PointForceLoad;2;2;", "PointForceLoad;2;3;");
    withModel(text, model => {
      const run = solver.run(model, "statics");
      near(displacement(run), built.expected);
      near(displacement(run, 3), built.expected, "dependent point follows its independent point");
    });
  });

for (const values of product({ scope: ["local", "global"], firstMass: ["none", "direct", "union"],
  secondMass: ["inherit", "direct", "union"], selection: ["all", "combination"] }))
  test(`subcase combination ${label(values)}`, () => {
    const first = lineModel({ scope: values.scope, load: "gravity", nsm: values.firstMass });
    const second = lineModel({ load: "force", nsm: values.secondMass === "inherit" ? "none" : values.secondMass });
    const a = first.text.indexOf("[model]");
    const b = second.text.indexOf("[model]");
    const firstControl = first.text.slice(0, a);
    const secondControl = second.text.slice(second.text.indexOf("SubcaseStart;1"), b)
      .replace("SubcaseStart;1", "SubcaseStart;2").replace("NonStructMassSelection;20", "NonStructMassSelection;40");
    const secondMassRows = second.text.slice(b).split("\n").filter(line => line.startsWith("NonStructMass"))
      .map(line => {
        const fields = line.split(";");
        fields[1] = String(Number(fields[1]) + 20);
        if (fields[0] === "NonStructMassUnion")
          for (let i = 2; i < fields.length; ++i) fields[i] = String(Number(fields[i]) + 20);
        return fields.join(";");
      }).join("\n");
    const text = firstControl + secondControl +
      "SubcaseCombination;3\nDisplacementOutput;All\nSubcaseCombinationFactors;1.5,-0.5\n" +
      first.text.slice(a) + "PointForceLoad;3;2;;10.;1.;0.;0.\n" + secondMassRows + "\n";
    // Keep gravity and point-force set IDs distinct when concatenating the controls.
    const actualText = text.replace(secondControl, secondControl.replace("StaticLoadSelection;2", "StaticLoadSelection;3"));
    withModel(actualText, model => {
      const run = solver.run(model, "statics", values.selection === "all" ? {} : { subcases: [3] });
      near(displacement(run, 2, 0, 3), 1.5 * first.expected - 0.5 * second.expected, "linear combination");
    });
  });

const incompatibleSelections = {
  nonlinearStaticParamsSelection: "NonlinearStaticParamsSelection;100",
  dynamicLoadSelection: "DynamicLoadSelection;100",
  freq: "FreqSetSelection;100",
  timeStepSelection: "TimeStepSelection;100",
  temperatureMaterial: "TemperatureSelection;100;MaterialTemperature",
};
for (const values of product({ selection: Object.keys(incompatibleSelections), scope: ["local", "global"],
  load: ["force", "gravity"], nsm: ["none", "direct"] }))
  test(`unsupported selection ${label(values)}`, () => {
    const built = lineModel(values);
    const command = incompatibleSelections[values.selection];
    const text = built.text.replace("[case]", `[case]\n${values.scope === "global" ? command : ""}`)
      .replace("SubcaseStart;1", `SubcaseStart;1\n${values.scope === "local" ? command : ""}`);
    withModel(text, model => {
      const report = model.checkSupport("statics");
      assert.equal(report.isComplete, false);
      assert.ok(report.refusals.some(x => x.code === "unsupported-selection" && x.field === values.selection));
      assert.throws(() => solver.run(model, "statics"), new RegExp(values.selection));
    });
  });

for (const values of product({ element: LINE_ELEMENTS, load: ["force", "gravity"],
  invalid: ["missing-set", "missing-target", "negative", "lumped", "lumped-list"] }))
  test(`NSM refusal ${label(values)}`, () => {
    const built = lineModel(values);
    const target = values.element === "Rod2InlineSection" ? "Element"
      : values.element === "Tube2" ? "TubeProp" : values.element === "Bar2" ? "BarProp"
      : values.element === "Beam2" ? "BeamProp" : "RodProp";
    const invalidRows = {
      "missing-set": "",
      "missing-target": `NonStructMass;20;${target};999;2.\n`,
      negative: `NonStructMass;20;${target};1;-2.\n`,
      lumped: `NonStructMassLumped;20;${target};1;2.\n`,
      "lumped-list": `NonStructMassLumped1;20;${target};2.;1\n`,
    };
    const codes = { "missing-set": "nsm-unknown-set", "missing-target": "nsm-unknown-target",
      negative: "nsm-invalid-value", lumped: "nsm-lumped-unsupported", "lumped-list": "nsm-lumped-unsupported" };
    withModel(built.text.replace("SubcaseStart;1", "SubcaseStart;1\nNonStructMassSelection;20") + invalidRows[values.invalid],
      model => assert.throws(() => solver.run(model, "statics"), new RegExp(codes[values.invalid])));
  });
