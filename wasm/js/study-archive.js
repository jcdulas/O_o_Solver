const CODES = { static: 101, modes: 103, buckling: 105, aeroelastic: 144 };
const same = (a, b) => a?.length === b?.length && (!a || a.every((value, i) => Object.is(value, b[i])));
const array = (values, shape = [values.length], type = "float64") => ({ type, shape, values });

// Merge steps with identical row layouts. Different analyses retain separate names, including static preloads
// exported by a buckling run. The original subcase and entity identities remain unchanged.
export function analysisTables(runs) {
  const result = [];
  runs.forEach((run, analysisIndex) => {
    const groups = new Map();
    const append = table => {
      const key = JSON.stringify([table.kind, table.elementType, table.subcase, table.analysis, table.components]);
      const candidates = groups.get(key) ?? [];
      let group = candidates.find(group => ["ids", "nodeIds", "layers", "positions", "pointTypes"].every(name => same(group[name], table[name])));
      if (!group) {
        group = { ...table, analysisCode: CODES[table.analysis],
          tableName: `analysis_${analysisIndex}_${result.length}`, _parts: [], steps: [], eigenvalues: [], cycles: [] };
        if (table.analysis === "modes" || table.analysis === "buckling") group.stepKind = "mode";
        candidates.push(group); groups.set(key, candidates); result.push(group);
      }
      group._parts.push(table.values);
      group.steps.push(table.analysis === "modes" || table.analysis === "buckling" ? table.step : NaN);
      if (group.stepKind === "mode") {
        group.eigenvalues.push(table.eigenvalue);
        group.cycles.push(Math.sqrt(Math.abs(table.eigenvalue)) / (2 * Math.PI));
      }
    };
    run.nodalTables.forEach(append); run.elementTables.forEach(append);
    for (const table of run.gridPointForces) append({ ...table, kind: "gridPointForces", ids: table.nodeIds,
      texts: { elementName: Array.from(table.nameIndices, index => table.names[index]) },
      components: ["t1", "t2", "t3", "r1", "r2", "r3"] });
    for (const table of run.eigenvalues) {
      const components = ["extractionOrder", "eigenvalue", "radians", "cycles", "generalizedMass", "generalizedStiffness"];
      result.push({ kind: "eigenvalues", analysisCode: CODES[table.analysis], subcase: table.subcase,
        tableName: `analysis_${analysisIndex}_eigenvalues_${table.subcase}`, components,
        ids: Int32Array.from(table.rows, row => row.mode), values: Float64Array.from(table.rows.flatMap(row => components.map(name => row[name]))) });
    }
  });
  return result.map(({ _parts, ...group }) => {
    if (!_parts) return group;
    const values = new Float64Array(_parts.reduce((size, part) => size + part.length, 0));
    let offset = 0;
    for (const part of _parts) { values.set(part, offset); offset += part.length; }
    return { ...group, values };
  });
}

export function aeroNamedResults(aero, analysisIndex = 0) {
  if (!aero) return [];
  const results = [];
  for (const trim of aero.subcases) {
    const attributes = Object.fromEntries(["referenceChord", "referenceSpan", "referenceArea", "symmetryXz", "symmetryXy"]
      .map(name => [name, String(aero[name])]));
    Object.assign(attributes, { mach: String(trim.mach), dynamicPressure: String(trim.dynamicPressure),
      configuration: aero.configuration, basicToReference: JSON.stringify(Array.from(aero.basicToReference ?? [])),
      referenceOffset: JSON.stringify(Array.from(aero.referenceOffset ?? [])) });
    const add = (kind, arrays, texts = {}, name = kind) => results.push({ kind, arrays, texts,
      name: `analysis_${analysisIndex}_${name}`, subcase: trim.subcase, analysisCode: 144, attributes });
    const variables = trim.variables;
    add("aeroTrimVariables", {
      variableType: array(Int32Array.from(variables, v => v.kind === "controlSurface" ? 2 : 1), undefined, "int32"),
      statusCode: array(Int32Array.from(variables, v => ({ fixed: 3, free: v.kind === "controlSurface" ? 2 : 1, linked: 4, scheduled: 5 })[v.state]), undefined, "int32"),
      isFixed: array(Int32Array.from(variables, v => v.state === "fixed" ? 1 : 0), undefined, "int32"),
      variableId: array(Int32Array.from(variables, v => v.id), undefined, "int32"),
      value: array(Float64Array.from(variables, v => v.value)),
    }, { variable: variables.map(v => v.label) });
    const columns = ["rigidUnsplined", "rigidSplined", "elasticRestrained", "elasticUnrestrained", "inertialRestrained", "inertialUnrestrained"];
    const rows = [trim.reference, ...trim.derivatives];
    const coefficients = new Float64Array(rows.length * 36);
    rows.forEach((row, i) => {
      for (let component = 0; component < 6; ++component)
        columns.forEach((column, j) => { coefficients[i * 36 + component * 6 + j] = row[column][component]; });
    });
    add("stabilityDerivatives", { values: array(coefficients, [rows.length, 6, 6]) }, { variable: rows.map(row => row.label) });
    if (trim.pressures.length) {
      add("aeroPressures", { elementId: array(aero.boxIds, undefined, "int32"),
        pressureCoefficient: array(trim.pressures), pressure: array(Float64Array.from(trim.pressures, value => value * trim.dynamicPressure)) },
      { elementKind: Array(trim.pressures.length).fill("LS") });
      const forces = new Float64Array(trim.boxForces.length * 6);
      trim.boxForces.forEach((force, i) => { forces[i * 6 + 2] = force; forces[i * 6 + 4] = trim.boxMoments[i]; });
      add("aeroForces", { nodeId: array(aero.boxIds, undefined, "int32"), forces: array(forces, [aero.boxIds.length, 6]) },
        { elementKind: Array(trim.boxForces.length).fill("LS") });
    }
    if (aero.surfaces.length) {
      const definitions = aero.surfaceDefinitions ?? [];
      const field = (name, fallback) => array(Float64Array.from(aero.surfaces, (surface, i) => definitions[i]?.[name] ?? fallback));
      add("controlSurfaceResults", { position: array(Float64Array.from(aero.surfaces, label => variables.find(v => v.label === label)?.value ?? NaN)),
        positionLower: field("positionLower", NaN), positionUpper: field("positionUpper", NaN),
        hingeMoment: array(trim.hingeMoments), hingeMomentLower: field("hingeMomentLower", -1e10),
        hingeMomentUpper: field("hingeMomentUpper", 1e10) }, { surface: aero.surfaces });
      for (const hinge of trim.hingeMomentDerivatives) {
        const n = hinge.perVariable.length / 3 + 1, data = new Float64Array(n * 5).fill(NaN);
        data.set(hinge.atReference, 0);
        for (let i = 1; i < n; ++i) data.set(hinge.perVariable.subarray((i - 1) * 3, i * 3), i * 5);
        add("hingeMomentDerivatives", { values: array(data, [n, 5]) },
          { variable: ["Intercept", ...variables.map(v => v.label)] }, `hinge_${hinge.surface}`);
      }
    }
    const monitorRows = [];
    trim.monitorLoads?.forEach((monitor, index) => {
      const point = aero.monitorPoints?.[index];
      if (!point) return;
      const contributions = ["rigidAir", "elasticRestrained", "inertial", "rigidApplied", "restrainedApplied"];
      monitor.loads.forEach((value, i) => monitorRows.push({ ...point, side: point.isAerodynamic ? "aerodynamic" : "structural",
        aeroComponent: point.component, value, component: i % 6 + 1,
        contribution: contributions[Math.floor(i / 6)], card: "MonitorIntegratedLoad1" }));
    });
    trim.monitor3Loads?.forEach((monitor, index) => {
      const point = aero.monitorPoints3?.[index];
      if (!point) return;
      monitor.loads.forEach((value, i) => monitorRows.push({ ...point, side: "structural", value, component: i + 1,
        contribution: "resultant", card: "MonitorNodeForceSum3" }));
    });
    if (monitorRows.length) {
      const integers = name => array(Int32Array.from(monitorRows, row => row[name]), undefined, "int32");
      add("monitorPointLoads", { subcase: array(new Int32Array(monitorRows.length).fill(trim.subcase), undefined, "int32"),
        step: array(new Float64Array(monitorRows.length).fill(NaN)), component: integers("component"),
        positionFrameId: integers("positionFrameId"), displacementFrameId: integers("displacementFrameId"),
        point: array(Float64Array.from(monitorRows.flatMap(row => Array.from(row.point))), [monitorRows.length, 3]),
        value: array(Float64Array.from(monitorRows, row => row.value)) },
      Object.fromEntries(["name", "label", "card", "pointClass", "aeroComponent", "side", "contribution"].map(name =>
        [name, monitorRows.map(row => row[name] ?? "")] )));
    }
  }
  return results;
}

export function acceptedResponses(values, definition, designCycle = -1) {
  const constraints = new Map();
  for (const constraint of values.constraints) {
    const key = `${constraint.responseId}:${constraint.item}`;
    if (!constraints.has(key)) constraints.set(key, new Map());
    const sets = constraints.get(key);
    if (!sets.has(constraint.setId)) sets.set(constraint.setId, {});
    sets.get(constraint.setId)[constraint.side] = constraint;
  }
  const rows = values.responses.flatMap(response => Array.from(response.values, (value, item) => {
    const sets = constraints.get(`${response.id}:${item}`);
    return Array.from(sets?.values() ?? [null], bounds =>
      ({ response, value, item, identity: response.items[item], bounds }));
  }).flat());
  const definitions = new Map(definition.responses.map(response => [response.id, response]));
  const arrayOf = (type, read) => array(type === "int32" ? Int32Array.from(rows, read) : Float64Array.from(rows, read),
    [rows.length], type);
  return { kind: "responseValues", name: "accepted_responses", subcase: 0, analysisCode: 0,
    arrays: {
      designCycle: arrayOf("int32", () => designCycle),
      id: arrayOf("int32", row => row.response.id),
      subcase: arrayOf("int32", row => row.identity.subcase),
      value: arrayOf("float64", row => row.value),
      lower: arrayOf("float64", row => row.bounds?.lower?.bound ?? NaN),
      upper: arrayOf("float64", row => row.bounds?.upper?.bound ?? NaN),
      constraintSet: arrayOf("int32", row =>
        row.bounds?.lower?.setId ?? row.bounds?.upper?.setId ?? 0),
      isObjective: arrayOf("int32", row =>
        Number(definition.objective?.responseId === row.response.id &&
          (definition.objective.item ?? 0) === row.item)),
      itemId: arrayOf("int32", row => row.identity.entity),
      component: arrayOf("int32", row => row.identity.component),
      layer: arrayOf("int32", row => row.identity.layer),
    }, texts: {
      label: rows.map(row => row.response.label),
      responseType: rows.map(row => definitions.get(row.response.id)?.source ?? ""),
    } };
}

export function saveStudyPoint(reader, deck, runs, path, options = {}, accepted = null) {
  const results = reader.createResults(analysisTables(runs), "optimization");
  try {
    runs.forEach((run, index) => aeroNamedResults(run.aeroelastic, index).forEach(result => results.addNamed(result)));
    if (accepted) results.addNamed(acceptedResponses(accepted.values, accepted.definition, accepted.designCycle));
    results.attach(deck, false);
    return results.writeArchive(path, options);
  } finally { results.delete(); }
}
