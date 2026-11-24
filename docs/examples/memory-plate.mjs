// A square membrane under unit total edge force. Both left-hand nodes are clamped.
export function solvePlate(reader, solver) {
  const deck = reader.createModel();
  let model;
  try {
    deck.setSolution("LinearStatics");
    deck.caseControl.global.set("SinglePointConstrSelection", 7);
    deck.caseControl.global.set("DisplacementOutput", "All");
    deck.caseControl.addSubcase(1, { commands: [{ name: "StaticLoadSelection", value: 10 }] });
    deck.addCardBatch("Node", {
      id: new Int32Array([1, 2, 3, 4]),
      position1: new Float64Array([0, 1, 1, 0]),
      position2: new Float64Array([0, 0, 1, 1]),
      position3: 0,
    });
    deck.addCards([
      { type: "ShellQuad4", fields: [1, 1, 1, 2, 3, 4] },
      { type: "ShellProp", fields: [1, 1, 0.01] },
      { type: "IsoMat", fields: [1, 100000, null, 0] },
      { type: "SinglePointConstrList", values: { id: 7, components: 123456, nodes: [1, 4] } },
      ...[2, 3].map(node => ({ type: "PointForceLoad", fields: [10, node, 0, 0.5, 1, 0, 0] })),
    ]);
    deck.applyDefaults();
    const problems = deck.checkReferences();
    if (problems.length) throw new Error(problems[0].message);
    model = solver.modelFromReader(reader, deck);
    const result = solver.run(model, "statics");
    const u = result.nodalTables.find(table => table.kind === "displacements");
    return Object.fromEntries(Array.from(u.ids, (id, row) => [id, u.values[6 * row]]));
  } finally {
    model?.delete();
    deck.delete();
  }
}
