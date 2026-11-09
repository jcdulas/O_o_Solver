import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { solver } from "../helpers/runtime.mjs";
import { product, label, withModel, displacement, near, assertFinite, lineModel, real } from "../helpers/static-models.mjs";

export const CONNECTION_ELEMENTS = ["SpringDofProperty", "SpringDofInline", "SpringScalarProperty", "SpringScalarInline",
  "Bushing", "Bushing1D", "Gap2", "Weld", "Fastener"];
export const MASS_ELEMENTS = ["PointMassMatrix", "PointMassInertia", "MassDofProperty", "MassDofInline",
  "MassScalarProperty", "MassScalarInline"];

function connectionModel(element, sign) {
  const scalar = element.includes("Scalar");
  const records = {
    SpringDofProperty: ["SpringDofProperty;1;1;1;1;2;1", "SpringProp;1;1000."],
    SpringDofInline: ["SpringDofInline;1;1000.;1;1;2;1"],
    SpringScalarProperty: ["SpringScalarProperty;1;1;101;0", "SpringProp;1;1000."],
    SpringScalarInline: ["SpringScalarInline;1;1000.;101;0"],
    Bushing: ["Bushing;1;1;1;2;;;;;0", "BushProp;1;Stiffness;1000.;1000.;1000.;1000.;1000.;1000."],
    Bushing1D: ["Bushing1D;1;1;1;2;0", "BushProp1D;1;1000."],
    Gap2: ["Gap2;1;1;1;2;;0.;1.;0.", "GapProp;1;0.;0.;1000.;1000."],
    Weld: ["Weld;1;1;;AlignedNodes;1;2", "WeldProp;1;1;2.", "IsoMat;1;1000.;;0.3;1."],
  };
  const control = scalar ? [] : ["SinglePointConstrSelection;1"];
  const bulk = scalar ? ["ScalarPoint;101", `ScalarPointLoad;2;101;${real(sign * 10)}`]
    : ["Node;1;;0.;0.;0.", "Node;2;;3.;0.;0.", "SinglePointConstrList;1;123456;1",
       "SinglePointConstrList;1;23456;2", `PointForceLoad;2;2;;${real(sign * 10)};1.;0.;0.`];
  return { text: ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1", ...control,
    "StaticLoadSelection;2", "DisplacementOutput;All", "[model]", ...bulk, ...records[element], ""].join("\n"),
    node: scalar ? 101 : 2, expected: element === "Weld" ? sign * 30 / (1000 * Math.PI) : sign * 0.01 };
}
for (const values of product({ element: CONNECTION_ELEMENTS.filter(x => x !== "Fastener"), sign: [-1, 1] }))
  test(`connection ${label(values)}`, () => {
    const built = connectionModel(values.element, values.sign);
    withModel(built.text, model => {
      const run = solver.run(model, "statics");
      assertFinite(run);
      near(displacement(run, built.node), built.expected, "spring or axial connector solution");
    });
  });

for (const sign of [-1, 1]) test(`Fastener joins two surface patches sign=${sign}`, () => {
  const points = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0.1],[1,0,0.1],[1,1,0.1],[0,1,0.1],
    [0.5,0.5,0.05],[0.5,0.5,0],[0.5,0.5,0.1]];
  const bulk = points.map(([x,y,z], i) => `Node;${i+1};;${real(x)};${real(y)};${real(z)}`);
  bulk.push("ShellQuad4;1;1;1;2;3;4", "ShellQuad4;2;2;5;6;7;8", "ShellProp;1;1;0.1;1;;1",
    "ShellProp;2;1;0.1;1;;1", "IsoMat;1;1000.;;0.3;1.", "Fastener;101;3;Elements;1;2;9;10;11",
    "FastenerProp;3;0.1;0;1;1000.;1000.;1000.;0.;0.;0.", "SinglePointConstrList;1;123456;1;2;3;4",
    "SinglePointConstrList;1;12456;5;6;7;8;9", `PointForceLoad;2;11;;${real(sign * 10)};0.;0.;1.`);
  withModel(["[case]", "SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "[model]", ...bulk, ""].join("\n"), model => {
    const run = solver.run(model, "statics");
    assertFinite(run);
    near(displacement(run, 11, 2) - displacement(run, 10, 2), sign * 0.01, "connector extension F/K");
  });
});

for (const values of product({ element: MASS_ELEMENTS, mass: [0.5, 3], weight: [0.5, 1] }))
  test(`concentrated mass ${label(values)}`, () => {
    const { element, mass, weight } = values;
    const scalar = element.includes("Scalar");
    if (scalar) {
      const records = element === "MassScalarProperty"
        ? [`MassScalarProperty;2;1;101;0`, `MassProp;1;${real(mass)}`]
        : [`MassScalarInline;2;${real(mass)};101;0`];
      withModel(["[case]", "SubcaseStart;1", "StaticLoadSelection;2", "EigenMethodSelection;3",
        "DisplacementOutput;All", "[model]", "ScalarPoint;101", "SpringScalarInline;1;1000.;101;0",
        "ScalarPointLoad;2;101;10.", "LanczosEigenMethod;3;;;1", `Parameter;WeightToMass;${real(weight)}`,
        ...records, ""].join("\n"), model => {
        near(displacement(solver.run(model, "statics"), 101), 0.01);
        const modes = solver.run(model, "modes");
        near(modes.eigenvalues[0].rows[0].eigenvalue, 1000 / (mass * weight), "scalar omega squared");
      });
    } else {
      const records = {
        PointMassMatrix: [`PointMassMatrix;2;2;;${real(mass)}`],
        PointMassInertia: [`PointMassInertia;2;2;;${real(mass)}`],
        MassDofProperty: ["MassDofProperty;2;2;2;1;0;0", `MassProp;2;${real(mass)}`],
        MassDofInline: [`MassDofInline;2;${real(mass)};2;1;0;0`],
      };
      const built = lineModel({ load: "gravity", weight });
      withModel(built.text + records[element].join("\n") + "\n", model => {
        const run = solver.run(model, "statics");
        assertFinite(run);
        near(displacement(run), built.expected + mass * weight * 10 * 3 / (1000 * 2), "point mass gravity");
      });
    }
  });
