import assert from "node:assert/strict";
import { test } from "../helpers/register.mjs";
import { oor, solver } from "../helpers/runtime.mjs";
import { wing } from "../helpers/optimization-models.mjs";

function near(actual, expected, context, tolerance = 2e-5) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${context}: ${actual} differs from ${expected}`);
}

function build(card, fields, { modal = false, buckling = false, coupled = true, released = false, tapered = false,
  thermal = false, initial = false, deformation = false, line = false, zeroExpansion = false,
  lineFields = false, mixedRows = false, edgeLoads = false } = {}) {
  const property = card === "Bar2" ? "BarProp" : "BeamProp";
  const preload = buckling ? "SubcaseStart;2\nSinglePointConstrSelection;1\nEigenMethodSelection;3\nStaticSubcaseSelection;1\n" : "";
  const text = `[executive]\nSolutionSequence;${modal ? 103 : buckling ? 105 : 101}\n[case]\nSubcaseStart;1\nSinglePointConstrSelection;1\n${modal ? "EigenMethodSelection;3" : "StaticLoadSelection;2"}\n${preload}[model]\nNode;1;;0.;0.;0.\nNode;2;;2.;0.;0.\nSinglePointConstrList;1;123456;1\nIsoMat;1;1000.;400.;0.25;1.3\nPointForceLoad;2;2;;10.;${buckling ? -1 : 1}.;2.;3.\nPointMomentLoad;2;2;;5.;1.;2.;3.\nParameter;CoupledMass;${coupled ? 1 : -1}\nLanczosEigenMethod;3;;;2\n`;
  let source = text;
  if (thermal) source = source.replace("[case]\n", `[case]\nTemperatureSelection;7;TemperatureLoadPart\n${
    initial ? "TemperatureSelection;8;InitialTemperature\n" : ""}`);
  if (deformation) source = source.replace("[case]\n", "[case]\nDeformationLoadSelection;9\n");
  source = source.replace("IsoMat;1;1000.;400.;0.25;1.3", `IsoMat;1;1000.;400.;0.25;1.3;${zeroExpansion ? 0 : 0.001};10.`);
  oor.writeFile("/work/beam-derivatives.oos", source);
  const deck = oor.readModel("/work/beam-derivatives.oos", { strict: true });
  if (thermal === "nodes") deck.addCard({ type: "NodeTemperatureLoad", fields: [7, 1, 40, 2, 80] });
  else if (thermal === "default") deck.addCard({ type: "DefaultTemperatureLoad", fields: [7, 60] });
  else if (thermal) deck.addCard({ type: "LineElemTemperatureLoad", fields: [7, 1, 40, 80] });
  if (initial) deck.addCard({ type: "DefaultTemperatureLoad", fields: [8, 0] });
  if (deformation) deck.addCard({ type: "AxialDeformationLoad", fields: [9, 1, 0.03] });
  if (line) deck.addCards([
    { type: "BeamDistributedLoad", fields: [2, 1, line, "ActualLength", 0.2, 3, 1.7, -2] },
    { type: "BeamDistributedLoad", fields: [2, 1, "MomentZAlongLength", "ActualLength", 0.7, 2, 0.7] },
    { type: "BeamDistributedLoad", fields: [2, 1, "ForceXAlongLength", "ActualLength", 2, 1, 2] },
  ]);
  if (edgeLoads) deck.addCards([
    { type: "BeamDistributedLoad", fields: [2, 1, "ForceZAlongLength", "ActualLength", 0, 2, 0] },
    { type: "BeamDistributedLoad", fields: [2, 1, "MomentYAlongLength", "ActualLength", 2, 3, 2] },
  ]);
  deck.addCard({ type: card, values: { id: 1, propId: 1, nodeA: 1, nodeB: 2,
    orientation1: 0, orientation2: 1, orientation3: 0, pinFlagsB: released ? 6 : 0,
    offset2A: buckling ? 0 : 0.03, offset3B: buckling ? 0 : -0.02 } });
  if (mixedRows) deck.addCard({ type: card, values: { id: 2, propId: 1, nodeA: 1, nodeB: 2,
    orientation1: 0, orientation2: 1, orientation3: 0 } });
  const section = { id: 1, matId: 1, area: 2, bendingInertia1: 0.3, bendingInertia2: 0.5,
    productInertia12: 0, torsionConstant: 0.4, nonStructMass: 0.2,
    stressPointCy: 0.2, stressPointCz: -0.1, stressPointDy: -0.2, stressPointDz: 0.3 };
  if (card === "Bar2") Object.assign(section, { shearFactor1: 0.8, shearFactor2: 0.7 });
  else section.stations = [
    ...(tapered ? [
      { key: "YesSameAsEndA", stationRatio: 1, stationArea: 3, stationBendingInertia1: 0.4,
        stationBendingInertia2: 0.6, stationProductInertia12: 0.02, stationTorsionConstant: 0.5,
        stationNonStructMass: 0.3 },
      { key: "Yes", stationRatio: 0.4, stationArea: 2.3, stationBendingInertia1: 0.35,
        stationBendingInertia2: 0.55, stationProductInertia12: 0.01, stationTorsionConstant: 0.45,
        stationNonStructMass: 0.25, stationStressPointCy: 0.3, stationStressPointCz: -0.2 },
    ] : []),
    { key: "", shearFactor1: 0.8, shearFactor2: 0.7, neutralAxisYA: 0.04, neutralAxisZA: -0.03,
      neutralAxisYB: 0.06, neutralAxisZB: -0.01, nsmCenterYA: 0.1, nsmCenterZA: -0.08,
      nsmCenterYB: 0.12, nsmCenterZB: -0.05 },
  ];
  deck.addCard({ type: property, values: section, ...(card === "Beam2" ? { form: "stressPointsAtEndA" } : {}) });
  // Independent springs keep released rotations supported and make end forces depend on beam stiffness.
  for (let dof = 1; dof <= 6; ++dof)
    deck.addCard({ type: "SpringDofInline", fields: [10 + dof, 5 + dof, 1, dof, 2, dof] });
  if (modal) deck.addCard({ type: "SinglePointConstrList", values: { id: 1, components: 4, nodes: [2] } });
  deck.applyDefaults();
  const model = solver.modelFromReader(oor, deck);
  try {
    const problem = solver.createOptimization(model, {
      variables: fields.map(({ value }, i) => ({ id: i + 1, initialValue: value,
        lower: value - Math.max(1, Math.abs(value)), upper: value + Math.max(1, Math.abs(value)) })),
      relations: fields.map(({ field, fieldItem, material }, i) => ({ id: i + 1,
        targetKind: material ? "material" : "property", targetCard: material ? "IsoMat" : property,
        targetId: 1, field, fieldItem, terms: [{ variableId: i + 1, coefficient: 1 }] })),
      responses: buckling ? [{ id: 10, type: "BucklingFactor", attributeA: 1, subcases: [2] }] : line ?
        [
          ...[1, 2, 3, 4, 5, 6].map(component => ({ id: 10 + component, type: "Displacement",
            attributeA: component, targets: [2] })),
          ...(lineFields ? ["Stress", "Strain", "Force"].flatMap((type, group) => {
            const codes = card === "Bar2" ? (type === "Force" ? [2, 3, 4, 5, 6, 7, 8, 9] :
              [2, 3, 4, 5, 6, 10, 11, 12, 13]) : (type === "Force" ?
              [4, 5, 6, 7, 8, 9, 10, 94, 95, 96, 97, 98, 99, 100] : [4, 5, 6, 7, 104, 105, 106, 107]);
            return codes.map((code, i) => ({ id: 100 + 100 * group + i, type, attributeA: code,
              propertyType: "Elements", targets: mixedRows ? [1, 2] : [1] }));
          }) : []),
        ] : [
        { id: 10, type: "Displacement", attributeA: 2, targets: [2], ...(modal ? { mode: 1 } : {}) },
        ...["Stress", "Strain", "Force"].map((type, i) => ({ id: 11 + i, type, attributeA: card === "Bar2" ? 2 : 4,
          propertyType: "Elements", targets: [1], ...(modal ? { mode: 1 } : {}) })),
        ...(modal ? [{ id: 14, type: "Eigenvalue", attributeA: 1 }] : []),
        { id: 15, type: "Stress", attributeA: card === "Bar2" ? 10 : 104,
          propertyType: "Elements", targets: [1], ...(modal ? { mode: 1 } : {}) },
        ...(thermal || deformation ? [
          { id: 16, type: "Displacement", attributeA: 1, targets: [2] },
          { id: 17, type: "Stress", attributeA: card === "Bar2" ? 6 : 4, propertyType: "Elements", targets: [1] },
        ] : []),
      ],
    }, modal ? { analyses: { modes: [1] } } : buckling ? { analyses: { buckling: [2] } } : {});
    return { problem, dispose() { problem.delete(); model.delete(); deck.delete(); } };
  } catch (error) { model.delete(); deck.delete(); throw error; }
}

function buildTrim(card, fields) {
  const deck = wing({ load: true });
  const property = card === "Bar2" ? "BarProp" : "BeamProp";
  deck.addCards([
    { type: "IsoMat", fields: [99, 1e6, 4e5, 0.25, 1.3] },
    { type: card, values: { id: 99, propId: 99, nodeA: 1, nodeB: 3,
      orientation1: 0, orientation2: 0, orientation3: 1 } },
    { type: property, fields: card === "Bar2" ? [99, 99, 0.2, 0.003, 0.005, 0.004, 0.02, 0.2, -0.1] :
      [99, 99, 0.2, 0.003, 0.005, 0, 0.004, 0.02, 0.2, -0.1] },
    { type: "GravityLoad", fields: [20, null, 10, 0, 0, 1] },
  ]);
  deck.applyDefaults();
  const model = solver.modelFromReader(oor, deck);
  try {
    const problem = solver.createOptimization(model, {
      variables: fields.map(({ value }, i) => ({ id: i + 1, initialValue: value, lower: value / 2, upper: value * 2 })),
      relations: fields.map(({ field, fieldItem }, i) => ({ id: i + 1, targetKind: "property",
        targetCard: property, targetId: 99, field, fieldItem, terms: [{ variableId: i + 1, coefficient: 1 }] })),
      responses: [
        { id: 10, type: "Displacement", attributeA: 3, targets: [3] },
        { id: 11, type: "Stress", attributeA: card === "Bar2" ? 2 : 4,
          propertyType: "Elements", targets: [99] },
      ],
    }, { analyses: { aeroelastic: [1] } });
    return { problem, dispose() { problem.delete(); model.delete(); deck.delete(); } };
  } catch (error) { model.delete(); deck.delete(); throw error; }
}

function check(card, fields, options) {
  const built = options?.trim ? buildTrim(card, fields) : build(card, fields, options);
  const x = fields.map(field => field.value), n = x.length;
  const point = built.problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.equal(gradients.report.analysisPerturbations, 0);
    const weights = Object.fromEntries(gradients.responses.map(response =>
      [response.id, Array.from({ length: response.itemCount }, (_, i) => 0.1 / (1 + i))]));
    const identity = Array.from({ length: n * n }, (_, i) => +(Math.floor(i / n) === i % n));
    const hessian = point.hessianVector({ responses: weights }, identity, n);
    const analyticFlag = options?.modal ? "usesModalTangents" : options?.buckling ? "usesBucklingTangents" :
      options?.trim ? "usesTrimTangents" : "hasHessianBlocks";
    assert.equal(hessian.report[analyticFlag], true,
      hessian.report.hessianBlocksReason);
    assert.equal(hessian.report.perturbedLinearisations, 0);
    for (let column = 0; column < n; ++column) {
      const step = 1e-4 * Math.max(0.1, Math.abs(x[column]));
      const plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
      try {
        const a = positive.values().responses, b = negative.values().responses;
        const ga = positive.gradients().responses, gb = negative.gradients().responses;
        for (let r = 0; r < a.length; ++r)
          for (let item = 0; item < a[r].values.length; ++item)
            near(gradients.responses[r].values[item * n + column],
              (a[r].values[item] - b[r].values[item]) / (2 * step), `gradient ${card} ${r}/${item}/${column}`);
        for (let row = 0; row < n; ++row) {
          let expected = 0;
          for (let r = 0; r < ga.length; ++r)
            for (let item = 0; item < ga[r].itemCount; ++item)
              expected += weights[ga[r].id][item] *
                (ga[r].values[item * n + row] - gb[r].values[item * n + row]) / (2 * step);
          near(hessian.values[row * n + column], expected, `Hessian ${card} ${row}/${column}`, 4e-5);
          near(hessian.values[row * n + column], hessian.values[column * n + row], "symmetry", 2e-8);
        }
      } finally { positive.delete(); negative.delete(); }
    }
  } finally { point.delete(); built.dispose(); }
}

const barGroups = [
  [{ field: "area", value: 2 }, { field: "youngsModulus", value: 1000, material: true }],
  [{ field: "bendingInertia1", value: 0.3 }, { field: "bendingInertia2", value: 0.5 },
    { field: "productInertia12", value: 0 }],
  [{ field: "torsionConstant", value: 0.4 }, { field: "shearModulus", value: 400, material: true }],
  [{ field: "shearFactor", fieldItem: 0, value: 0.8 }, { field: "shearFactor", fieldItem: 1, value: 0.7 },
    { field: "stressPointC", fieldItem: 0, value: 0.2 }],
];
for (const released of [false, true]) for (const [group, fields] of barGroups.entries())
  test(`analytic bar sizing group=${group} released=${released}`, () => check("Bar2", fields, { released }));

const beamGroups = [
  [{ field: "stationSection", fieldItem: 0, value: 2 }, { field: "youngsModulus", value: 1000, material: true }],
  [{ field: "stationSection", fieldItem: 1, value: 0.3 }, { field: "stationSection", fieldItem: 2, value: 0.5 },
    { field: "stationSection", fieldItem: 3, value: 0 }],
  [{ field: "stationSection", fieldItem: 4, value: 0.4 }, { field: "shearModulus", value: 400, material: true }],
  [{ field: "shearFactor", fieldItem: 0, value: 0.8 }, { field: "shearFactor", fieldItem: 1, value: 0.7 },
    { field: "stationStress", fieldItem: 0, value: 0.2 }],
  [{ field: "neutralAxis", fieldItem: 0, value: 0.04 }, { field: "neutralAxis", fieldItem: 3, value: -0.01 },
    { field: "stationSection", fieldItem: 0, value: 2 }],
];
for (const tapered of [false, true]) for (const [group, fields] of beamGroups.entries())
  test(`analytic beam sizing group=${group} tapered=${tapered}`, () => check("Beam2", fields, { tapered, released: true }));

for (const card of ["Bar2", "Beam2"]) for (const coupled of [false, true]) {
  const section = card === "Bar2" ? { field: "area", value: 2 } :
    { field: "stationSection", fieldItem: 0, value: 2 };
  const nsm = card === "Bar2" ? { field: "nonStructMass", value: 0.2 } :
    { field: "stationSection", fieldItem: 5, value: 0.2 };
  test(`analytic beam modal mass card=${card} coupled=${coupled}`, () => check(card,
    [section, nsm, { field: "density", value: 1.3, material: true }], { modal: true, coupled }));
}

for (const coupled of [false, true]) for (const kind of ["neutralAxis", "nsmCenter"])
  test(`analytic beam modal offset ${kind} coupled=${coupled}`, () => check("Beam2", [
    { field: kind, fieldItem: 0, value: kind === "neutralAxis" ? 0.04 : 0.1 },
    { field: kind, fieldItem: 3, value: kind === "neutralAxis" ? -0.01 : -0.05 },
    { field: "density", value: 1.3, material: true },
  ], { modal: true, coupled, tapered: true }));

test("analytic beam unsorted station sections and inherited stress points", () => check("Beam2", [
  { field: "stationSection", fieldItem: 7, value: 0.4 },
  { field: "stationSection", fieldItem: 14, value: 0.55 },
  { field: "stationStress", fieldItem: 0, value: 0.2 },
], { tapered: true }));

test("analytic beam intermediate recovery point", () => check("Beam2", [
  { field: "stationStress", fieldItem: 16, value: 0.3 },
  { field: "stationSection", fieldItem: 15, value: 0.01 },
], { tapered: true }));

for (const coupled of [false, true]) test(`analytic beam modal polar inertia coupled=${coupled}`, () => check("Beam2", [
  { field: "stationSection", fieldItem: 1, value: 0.3 },
  { field: "stationSection", fieldItem: 2, value: 0.5 },
  { field: "density", value: 1.3, material: true },
], { modal: true, coupled }));

for (const card of ["Bar2", "Beam2"]) for (const released of [false, true])
  test(`analytic beam buckling sizing card=${card} released=${released}`, () => check(card, [
    card === "Bar2" ? { field: "area", value: 2 } : { field: "stationSection", fieldItem: 0, value: 2 },
    card === "Bar2" ? { field: "productInertia12", value: 0 } :
      { field: "stationSection", fieldItem: 3, value: 0 },
  ], { buckling: true, released, tapered: card === "Beam2" }));

for (const card of ["Bar2", "Beam2"]) test(`analytic beam trim sizing card=${card}`, () => check(card,
  card === "Bar2" ? [{ field: "area", value: 0.2 }, { field: "nonStructMass", value: 0.02 }] :
    [{ field: "stationSection", fieldItem: 0, value: 0.2 },
      { field: "stationSection", fieldItem: 5, value: 0.02 }], { trim: true }));

for (const card of ["Bar2", "Beam2"]) for (const source of ["element", "nodes", "default"])
  for (const initial of [false, true])
    test(`analytic beam thermal sizing card=${card} source=${source} initial=${initial}`, () => check(card, [
      card === "Bar2" ? { field: "area", value: 2 } : { field: "stationSection", fieldItem: 0, value: 2 },
      { field: "thermalExpansion", value: 0.001, material: true },
      { field: "referenceTemperature", value: 10, material: true },
    ], { thermal: source, initial, released: true, tapered: card === "Beam2" }));

for (const card of ["Bar2", "Beam2"]) {
  test(`analytic beam expansion derivative at zero card=${card}`, () => check(card, [
    { field: "thermalExpansion", value: 0, material: true },
    { field: "youngsModulus", value: 1000, material: true },
  ], { thermal: "element", zeroExpansion: true }));
  test(`analytic beam imposed elongation sizing card=${card}`, () => check(card, [
    card === "Bar2" ? { field: "area", value: 2 } : { field: "neutralAxis", fieldItem: 0, value: 0.04 },
    { field: "youngsModulus", value: 1000, material: true },
  ], { deformation: true, released: true, tapered: card === "Beam2" }));
}

for (const card of ["Bar2", "Beam2"]) for (const released of [false, true])
  for (const line of ["ForceYAlongLength", "ForceZ", "MomentXAlongLength"])
    test(`analytic beam line load sizing card=${card} released=${released} kind=${line}`, () => check(card, [
      card === "Bar2" ? { field: "productInertia12", value: 0 } :
        { field: "stationSection", fieldItem: 7, value: 0.4 },
      { field: "shearFactor", fieldItem: 0, value: 0.8 },
      { field: "shearModulus", value: 400, material: true },
    ], { line, released, tapered: card === "Beam2" }));

for (const card of ["Bar2", "Beam2"])
  test(`analytic beam combined thermal and line curvature card=${card}`, () => check(card, [
    card === "Bar2" ? { field: "area", value: 2 } : { field: "stationSection", fieldItem: 0, value: 2 },
    { field: "thermalExpansion", value: 0.001, material: true },
    { field: "referenceTemperature", value: 10, material: true },
  ], { thermal: "element", line: "ForceYAlongLength", released: true, tapered: card === "Beam2" }));

for (const card of ["Bar2", "Beam2"]) test(`analytic beam thermal buckling gradient card=${card}`, () => {
  const fields = [
    card === "Bar2" ? { field: "area", value: 2 } : { field: "stationSection", fieldItem: 0, value: 2 },
    { field: "thermalExpansion", value: 0.001, material: true },
  ];
  const built = build(card, fields, { thermal: "element", buckling: true, released: true, tapered: card === "Beam2" });
  const x = fields.map(field => field.value), point = built.problem.evaluate();
  try {
    const gradients = point.gradients();
    assert.equal(gradients.report.fieldPerturbations, 0);
    assert.equal(gradients.report.analysisPerturbations, 0);
    for (let column = 0; column < x.length; ++column) {
      const step = 1e-4 * Math.max(0.1, Math.abs(x[column])), plus = [...x], minus = [...x];
      plus[column] += step;
      minus[column] -= step;
      const positive = built.problem.evaluate(plus), negative = built.problem.evaluate(minus);
      try {
        near(gradients.responses[0].values[column],
          (positive.values().responses[0].values[0] - negative.values().responses[0].values[0]) / (2 * step),
          `thermal buckling gradient ${column}`, 5e-5);
      } finally { positive.delete(); negative.delete(); }
    }
  } finally { point.delete(); built.dispose(); }
});

for (const card of ["Bar2", "Beam2"]) for (const released of [false, true])
  for (const line of ["ForceYAlongLength", "ForceZ", "MomentXAlongLength"])
    test(`analytic beam loaded recovery card=${card} released=${released} kind=${line}`, () => check(card, [
      card === "Bar2" ? { field: "productInertia12", value: 0 } :
        { field: "stationSection", fieldItem: 7, value: 0.4 },
      { field: "shearFactor", fieldItem: 0, value: 0.8 },
      { field: "shearModulus", value: 400, material: true },
    ], { line, lineFields: true, released, tapered: card === "Beam2", edgeLoads: true }));

for (const card of ["Bar2", "Beam2"])
  test(`analytic beam loaded recovery with thermal strain card=${card}`, () => check(card, [
    card === "Bar2" ? { field: "area", value: 2 } : { field: "neutralAxis", fieldItem: 0, value: 0.04 },
    { field: "thermalExpansion", value: 0.001, material: true },
    { field: "referenceTemperature", value: 10, material: true },
  ], { thermal: "element", line: "ForceYAlongLength", lineFields: true,
    released: true, tapered: card === "Beam2" }));

for (const card of ["Bar2", "Beam2"])
  test(`analytic beam loaded and unloaded rows share a batch card=${card}`, () => check(card, [
    card === "Bar2" ? { field: "bendingInertia1", value: 0.3 } :
      { field: "stationSection", fieldItem: 1, value: 0.3 },
    card === "Bar2" ? { field: "stressPointC", fieldItem: 0, value: 0.2 } :
      { field: "stationStress", fieldItem: 0, value: 0.2 },
  ], { line: "ForceZAlongLength", lineFields: true, mixedRows: true }));

for (const card of ["Bar2", "Beam2"]) test(`analytic beam loaded direct adjoint products card=${card}`, () => {
  const fields = [{ field: "youngsModulus", value: 1000, material: true }, card === "Bar2" ?
    { field: "stressPointC", fieldItem: 0, value: 0.2 } : { field: "stationStress", fieldItem: 0, value: 0.2 }];
  const built = build(card, fields, { line: "ForceZAlongLength", lineFields: true, released: true,
    tapered: card === "Beam2", thermal: "element" });
  const point = built.problem.evaluate();
  try {
    const direct = point.gradients({ method: "direct" }), adjoint = point.gradients({ method: "adjoint" });
    for (const result of [direct, adjoint]) {
      assert.equal(result.report.fieldPerturbations, 0);
      assert.equal(result.report.analysisPerturbations, 0);
    }
    direct.responses.forEach((response, r) => response.values.forEach((value, i) =>
      near(adjoint.responses[r].values[i], value, `direct/adjoint ${r}/${i}`, 1e-8)));
    const direction = [100, -0.03], jvp = point.jvp(direction);
    const expected = direct.responses.flatMap(response => Array.from({ length: response.itemCount }, (_, item) =>
      direction.reduce((sum, value, c) => sum + value * response.values[item * 2 + c], 0)));
    expected.forEach((value, i) => near(jvp.values[i], value, `JVP ${i}`, 1e-8));
    const weights = Object.fromEntries(direct.responses.map(response => [response.id,
      Array.from({ length: response.itemCount }, (_, i) => 0.1 / (i + 1))]));
    const vjp = point.vjp({ responses: weights });
    for (let column = 0; column < 2; ++column) {
      let expected = 0;
      for (const response of direct.responses)
        for (let item = 0; item < response.itemCount; ++item)
          expected += weights[response.id][item] * response.values[item * 2 + column];
      near(vjp.values[column], expected, `VJP ${column}`, 1e-8);
    }
  } finally { point.delete(); built.dispose(); }
});
