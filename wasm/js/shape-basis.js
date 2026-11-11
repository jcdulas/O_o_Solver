// Solve an auxiliary model once. Its translational displacement columns become a basic-frame shape basis, ready
// for shapeBatches; no callback or auxiliary factor is retained during subsequent optimization evaluations.
export function shapeBasisFromReader(solver, reader, auxiliary, { variableId, cases, gridIds } = {}) {
  if (!Number.isSafeInteger(variableId) || variableId < 1) throw new Error("a shape basis requires a positive variableId");
  const deck = auxiliary.copy();
  let model, nodes;
  try {
    deck.caseControl.global.set("DisplacementOutput", "All");
    for (const section of deck.caseControl.sections())
      if (section.kind === "subcase") deck.caseControl.section(section.key).set("DisplacementOutput", "All");
    model = solver.modelFromReader(reader, deck);
    const run = solver.run(model, "statics", cases ? { subcases: cases.map(entry => entry.subcase) } : {});
    nodes = reader.core.cardTable(deck, "nodes", ["id"]);
    const ids = nodes.column("id").values;
    const rotations = reader.core.displacementRotations(deck);
    const rows = new Map(Array.from(ids, (id, row) => [id, row]));
    const selected = gridIds ? new Set(gridIds) : null;
    const directions = new Float64Array(ids.length * 3);
    const weights = cases ? new Map(cases.map(entry => [entry.subcase, entry.coefficient ?? 1])) : null;
    if (weights && weights.size !== cases.length) throw new Error("shape basis cases must have distinct subcase IDs");
    if (weights && !Array.from(weights.values()).every(Number.isFinite)) throw new Error("shape basis coefficients must be finite");
    const found = new Set();
    for (const table of run.nodalTables) {
      if (table.kind !== "displacements" || weights && !weights.has(table.subcase)) continue;
      found.add(table.subcase);
      const weight = weights?.get(table.subcase) ?? 1;
      table.ids.forEach((id, row) => {
        if (selected && !selected.has(id) || table.pointTypes[row] !== 1) return;
        const target = rows.get(id);
        if (target === undefined) throw new Error(`auxiliary displacement names an unknown grid ${id}`);
        for (let basic = 0; basic < 3; ++basic)
          for (let local = 0; local < 3; ++local)
            directions[target * 3 + basic] += weight * rotations[target * 9 + basic * 3 + local] * table.values[row * 6 + local];
      });
    }
    if (weights && Array.from(weights.keys()).some(id => !found.has(id))) throw new Error("an auxiliary shape case has no displacement results");
    if (selected && Array.from(selected).some(id => !rows.has(id))) throw new Error("a selected shape grid is absent from the auxiliary model");
    const kept = Array.from(ids, (id, row) => row).filter(row => !selected || selected.has(ids[row]));
    return { variableId, gridIds: Int32Array.from(kept, row => ids[row]),
      directions: Float64Array.from(kept.flatMap(row => Array.from(directions.subarray(row * 3, row * 3 + 3)))) };
  } finally { nodes?.delete(); model?.delete(); deck.delete(); }
}
