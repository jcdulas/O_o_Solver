import { oor } from "./runtime.mjs";
export const AXIAL = `[executive]
SolutionSequence;101
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
[model]
Node;1;;0.;0.;0.
Node;2;;3.;0.;0.
Rod2;1;1;1;2
RodProp;1;1;2.
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
SinglePointConstrList;1;23456;2
PointForceLoad;2;2;;10.;1.;0.;0.
LanczosEigenMethod;3;;;1
`;
export function axial() {
  oor.writeFile("/work/study-axial.oos", AXIAL);
  return oor.readModel("/work/study-axial.oos", { strict: true });
}
export function wing({ link = false, load = false } = {}) {
  const deck = oor.createModel(); deck.setSolution(144);
  deck.caseControl.global.set("SinglePointConstrSelection", 1);
  deck.caseControl.global.set("TrimSelection", 5);
  deck.caseControl.global.set("DisplacementOutput", "All");
  deck.addCardBatch("Node", { id: [1, 2, 3, 4], position1: [0, 1, 1, 0], position2: [0, 0, 2, 2], position3: 0 });
  deck.addCards([
    { type: "ShellQuad4", fields: [1, 1, 1, 2, 3, 4] },
    { type: "ShellProp", fields: [1, 1, 0.1, 1, null, 1] },
    { type: "IsoMat", fields: [1, 1e6, null, 0.3, 1] },
    { type: "SinglePointConstrList", fields: [1, 123456, 1, 2] },
    { type: "AeroStaticParams", fields: [0, 0, 1, 2, 2] },
    { type: "AeroLiftingPanelProp1", fields: [10] },
    { type: "AeroLiftingPanel1", fields: [100, 10, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 1] },
    { type: "IdListSet1", fields: [50, 1, 2, 3, 4] },
    { type: "AeroSurfaceSpline1", fields: [60, 100, 100, 103, 50] },
    { type: "AeroRigidTrimVariable", fields: [70, "AngleOfAttack"] },
    { type: "AeroTrim", fields: [5, 0.1, 10, "AngleOfAttack", 0.05] },
  ]);
  if (link) deck.addCards([
    { type: "AeroRigidTrimVariable", fields: [71, "SideSlip"] },
    { type: "AeroTrimLink", fields: [0, "SideSlip", "AngleOfAttack", -0.2] },
  ]);
  if (load) {
    deck.addCard({ type: "PointForceLoad", fields: [20, 3, 0, 0.1, 0, 0, 1] });
    deck.caseControl.global.set("StaticLoadSelection", 20);
  }
  deck.applyDefaults(); return deck;
}
export function column() {
  oor.writeFile("/work/study-column.oos", `[executive]
SolutionSequence;105
[case]
SubcaseStart;1
SinglePointConstrSelection;1
StaticLoadSelection;2
SubcaseStart;2
SinglePointConstrSelection;1
EigenMethodSelection;3
StaticSubcaseSelection;1
[model]
Node;1;;0.;0.;0.
Node;2;;1.;0.;0.
Node;3;;2.;0.;0.
Bar2;1;1;1;2;;0.;1.;0.
Bar2;2;1;2;3;;0.;1.;0.
BarProp;1;1;0.1;0.01;0.02;0.01
IsoMat;1;1000.;;0.3;1.
SinglePointConstrList;1;123456;1
PointForceLoad;2;3;;1.;-1.;0.;0.
LanczosEigenMethod;3;;;2
`);
  return oor.readModel("/work/study-column.oos", { strict: true });
}
