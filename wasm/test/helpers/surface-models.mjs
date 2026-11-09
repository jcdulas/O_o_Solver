import { massSet, real } from "./static-models.mjs";

export const SURFACE_ELEMENTS = ["ShellQuad4", "ShellTria3", "ShellQuad4Drilling", "ShellTria3Drilling",
  "ShellQuad8", "ShellTria6", "ShearPanel4"];

// Unit square or right triangle. Constant axial stress at nu=0 has the exact solution u_x = sigma*x/E.
export function surfaceModel({ element, load, nsm, coupled, weight, density = 1 }) {
  const triangular = element.includes("Tria");
  const higher = element.endsWith("8") || element.endsWith("6");
  const points = triangular ? [[0, 0], [1, 0], [0, 1]] : [[0, 0], [1, 0], [1, 1], [0, 1]];
  const corners = points.length;
  if (higher) for (let i = 0; i < corners; ++i) {
    const a = points[i], b = points[(i + 1) % corners];
    points.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
  }
  const property = element === "ShearPanel4" ? "ShearPanelProp" : "ShellProp";
  const mass = massSet(nsm, property, 0.1);
  const control = ["SubcaseStart;1", "SinglePointConstrSelection;1", "StaticLoadSelection;2",
    "DisplacementOutput;All", "SinglePointConstrForceOutput;All", ...mass.control];
  const bulk = points.map(([x, y], i) => `Node;${i + 1};;${real(x)};${real(y)};0.`);
  bulk.push(`${element};1;1;${points.map((_, i) => i + 1).join(";")}`,
    property === "ShellProp" ? "ShellProp;1;1;0.1;1;;1" : "ShearPanelProp;1;1;0.1;0.;1.;1.",
    `IsoMat;1;1000.;;0.;${real(density)}`, `Parameter;CoupledMass;${coupled}`,
    `Parameter;WeightToMass;${real(weight)}`, ...mass.bulk);
  points.forEach(([x], i) => bulk.push(`SinglePointConstrList;1;${x === 0 ? "123456" : "23456"};${i + 1}`));
  if (load === "gravity") bulk.push("GravityLoad;2;;10.;1.;0.;0.");
  else {
    const forces = triangular ? (higher ? { 2: 1 / 6, 5: 2 / 3 } : { 2: 1 / 2 })
      : (higher ? { 2: 1 / 6, 3: 1 / 6, 6: 2 / 3 } : { 2: 1 / 2, 3: 1 / 2 });
    for (const [node, force] of Object.entries(forces)) bulk.push(`PointForceLoad;2;${node};;${real(force)};1.;0.;0.`);
  }
  return { text: ["[executive]", "SolutionSequence;101", "[case]", ...control, "[model]", ...bulk, ""].join("\n"),
    points, densityEquivalent: density + mass.added / 0.1 };
}

const SOLIDS = {
  SolidHexa: {
    points: [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
    edges: [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]], axis: 0,
    linear: {2: 1/4, 3: 1/4, 6: 1/4, 7: 1/4},
    quadratic: {2: -1/12, 3: -1/12, 6: -1/12, 7: -1/12, 10: 1/3, 14: 1/3, 15: 1/3, 18: 1/3},
  },
  SolidPenta: {
    points: [[0,0,0],[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,0,1]],
    edges: [[0,1],[1,2],[2,0],[0,3],[1,4],[2,5],[3,4],[4,5],[5,3]], axis: 0,
    linear: {4: 1/6, 5: 1/6, 6: 1/6}, quadratic: {13: 1/6, 14: 1/6, 15: 1/6},
  },
  SolidTetra: {
    points: [[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
    edges: [[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]], axis: 0,
    linear: {2: 1/6}, quadratic: {6: 1/6, 9: 1/6},
  },
  SolidPyramid: {
    points: [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0.5,0.5,1]],
    edges: [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]], axis: 2,
    linear: {5: 1/3}, quadratic: {10: 1/6, 11: 1/6, 12: 1/6, 13: 1/6},
  },
};
export const SOLID_ELEMENTS = Object.keys(SOLIDS);

// Consistent nodal tractions are integrals of boundary shape functions for a constant uniaxial stress.
// They follow the unit-face areas and polynomial triangle/quadrilateral weights, independently of the solver.
export function solidModel({ element, order, sign, modulus, coupled }) {
  const shape = SOLIDS[element];
  const points = shape.points.map(p => [...p]);
  if (order === "quadratic") for (const [a, b] of shape.edges)
    points.push(shape.points[a].map((x, c) => (x + shape.points[b][c]) / 2));
  const bulk = points.map(([x, y, z], i) => `Node;${i + 1};;${real(x)};${real(y)};${real(z)}`);
  bulk.push(`${element};1;1;${points.map((_, i) => i + 1).join(";")}`, "SolidProp;1;1",
    `IsoMat;1;${real(modulus)};;0.;1.`, `Parameter;CoupledMass;${coupled}`);
  points.forEach((p, i) => {
    const components = p[shape.axis] === 0 ? "123456" : (shape.axis === 0 ? "23456" : "12456");
    bulk.push(`SinglePointConstrList;1;${components};${i + 1}`);
  });
  for (const [node, force] of Object.entries(shape[order]))
    bulk.push(`PointForceLoad;2;${node};;${real(sign * 10 * force)};${shape.axis === 0 ? "1.;0.;0." : "0.;0.;1."}`);
  return { text: ["[executive]", "SolutionSequence;101", "[case]", "SubcaseStart;1",
    "SinglePointConstrSelection;1", "StaticLoadSelection;2", "DisplacementOutput;All", "StressOutput;All",
    "[model]", ...bulk, ""].join("\n"), points, axis: shape.axis };
}
