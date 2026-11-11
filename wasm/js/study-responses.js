import { analysisTables, aeroNamedResults } from "./study-archive.js";

// Plain-array response selection shared by programmable studies and archive export. An application can also
// register a named response callback; callbacks receive the complete owned result arrays of every analysis.
function selected(value, selection) { return !selection?.length || selection.includes(value); }
function componentIndices(names, requested) {
  if (requested === undefined) return names.map((_, i) => i);
  return (Array.isArray(requested) ? requested : [requested]).map(name => {
    const index = typeof name === "number" ? name : names.indexOf(name);
    if (!Number.isInteger(index) || index < 0 || index >= names.length)
      throw new Error(`unknown result component '${name}'`);
    return index;
  });
}

function tracking(run, referenceRun, response, modeReferences) {
  const mode = response.mode ?? response.step;
  if (!response.tracksMode || mode === undefined) return { mode, sign: 1, tracked: null };
  const shapes = run.nodalTables.filter(t => t.kind === "eigenvectors" && t.subcase === (response.subcase ?? 1));
  const prior = modeReferences?.find(reference => reference.responseId === response.id);
  const referenceTable = referenceRun?.nodalTables.find(t => t.kind === "eigenvectors" &&
    t.subcase === (response.subcase ?? 1) && t.step === (prior?.mode ?? mode)) ?? shapes.find(t => t.step === mode);
  const reference = prior && referenceTable ? { ids: referenceTable.ids, values: prior.shape } : referenceTable;
  if (!reference) throw new Error(`no eigenvector reference for response ${response.id}, mode ${mode}`);
  const candidates = shapes.map(table => {
    if (table.values.length !== reference.values.length || table.ids.length !== reference.ids.length ||
        table.ids.some((id, i) => id !== reference.ids[i]))
      throw new Error("modal tracking requires corresponding node identities and ordering");
    let dot = 0, a = 0, b = 0;
    table.values.forEach((value, i) => { dot += value * reference.values[i]; a += value * value; b += reference.values[i] ** 2; });
    return { table, mac: a && b ? dot * dot / a / b : 0, sign: dot < 0 ? -1 : 1 };
  }).sort((a, b) => b.mac - a.mac || a.table.step - b.table.step);
  if (!candidates.length) throw new Error("modal tracking requires eigenvector outputs");
  const best = candidates[0];
  if (!referenceRun) {
    let anchor = 0;
    best.table.values.forEach((value, i) => { if (Math.abs(value) > Math.abs(best.table.values[anchor])) anchor = i; });
    best.sign = best.table.values[anchor] < 0 ? -1 : 1;
  }
  const roots = run.eigenvalues.find(table => table.subcase === best.table.subcase)?.rows ?? [];
  const root = roots.find(row => row.mode === best.table.step)?.eigenvalue;
  const repeatedRoot = roots.some(row => row.mode !== best.table.step && Math.abs(row.eigenvalue - root) <=
    1e-6 * Math.max(Math.abs(root), Math.abs(row.eigenvalue), Number.MIN_VALUE));
  return { mode: best.table.step, sign: best.sign,
    tracked: { responseId: response.id, subcase: response.subcase ?? 1, mode: best.table.step, mac: best.mac,
      isUncertain: repeatedRoot || best.mac < (response.minimumMac ?? 0.8) || best.mac - (candidates[1]?.mac ?? 0) < 0.05,
      repeatedRoot,
      shape: Float64Array.from(best.table.values, value => value * best.sign) } };
}

function reduce(values, items, method) {
  if (!method || method === "none") return { values, items };
  if (!values.length) throw new Error("a response selection is empty");
  let value;
  if (method === "sum" || method === "mean") value = values.reduce((a, b) => a + b, 0) / (method === "mean" ? values.length : 1);
  else if (method === "max") value = values.reduce((a, b) => Math.max(a, b), -Infinity);
  else if (method === "min") value = values.reduce((a, b) => Math.min(a, b), Infinity);
  else if (method === "maxAbs") value = values.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
  else if (method === "norm") value = values.reduce((norm, item) => Math.hypot(norm, item), 0);
  else throw new Error(`unknown response reduction '${method}'`);
  return { values: [value], items: [{ subcase: items[0]?.subcase ?? 0, entity: 0, component: 0, layer: 0 }] };
}

export function studyResponses(definitions, runs, context, hooks = {}, referenceRuns = []) {
  const trackedModes = [];
  let queryResults;
  try {
  const responses = definitions.map(response => {
    const index = response.analysis ?? 0, run = runs[index];
    if (!run) throw new Error(`response ${response.id} names an absent analysis`);
    let values = [], items = [];
    const add = (value, subcase = 0, entity = 0, component = 0, layer = 0, details = {}) => {
      values.push(value); items.push({ subcase, entity, component, layer, ...details });
    };
    const track = tracking(run, referenceRuns[index], response, context.modeReferences);
    if (track.tracked) trackedModes.push(track.tracked);
    if (response.source === "query") {
      const needsResults = context.reader.queryCatalogue().find(query => query.name === response.query)?.resultQuery;
      if (needsResults && !queryResults) {
        queryResults = context.reader.createResults(analysisTables(runs), "optimization response");
        runs.forEach((analysis, index) => aeroNamedResults(analysis.aeroelastic, index)
          .forEach(result => queryResults.addNamed(result)));
        queryResults.attach(context.deck, false);
      }
      const table = context.reader.runQuery(response.query, { deck: context.deck,
        ...(needsResults ? { results: queryResults } : {}), args: response.args ?? {} });
      try {
        const column = table.column(response.column);
        if (column.kind !== "number") throw new Error("a study query response requires a scalar numeric column");
        const ids = response.idColumn ? table.column(response.idColumn) : null;
        if (ids && ids.kind !== "number") throw new Error("a study query identity requires a scalar numeric column");
        column.values.forEach((value, row) => {
          if ((!column.valid || column.valid[row]) && (!ids?.valid || ids.valid[row]))
            add(value, response.subcase ?? 0, ids ? ids.values[row] : row + 1);
        });
      } finally { table.delete(); }
    } else if (response.source === "callback") {
      const callback = hooks.responses?.[response.callback];
      if (!callback) throw new Error(`missing response callback '${response.callback}'`);
      const result = callback({ ...context, runs, response, mode: track.mode, modeSign: track.sign });
      if (result && typeof result.then === "function") throw new TypeError("a synchronous response callback cannot return a Promise");
      const data = typeof result === "number" ? [result] : Array.isArray(result) || ArrayBuffer.isView(result) ? result : result?.values;
      values = Array.from(data);
      items = result?.items ?? values.map((_, i) => ({ subcase: 0, entity: i + 1, component: 0, layer: 0 }));
    } else if (["nodal", "element", "gridPointForce"].includes(response.source)) {
      const tables = response.source === "nodal" ? run.nodalTables : response.source === "element" ? run.elementTables : run.gridPointForces;
      for (const table of tables) {
        if (response.kind && table.kind !== response.kind || response.subcase !== undefined && table.subcase !== response.subcase ||
            response.elementType && table.elementType !== response.elementType || track.mode !== undefined && table.step !== track.mode) continue;
        const names = table.components ?? ["t1", "t2", "t3", "r1", "r2", "r3"];
        const columns = componentIndices(names, response.components);
        const ids = table.ids ?? table.nodeIds;
        for (let row = 0; row < ids.length; ++row) {
          if (!selected(ids[row], response.ids) || !selected(table.nodeIds?.[row] ?? 0, response.nodeIds) ||
              !selected(table.layers?.[row] ?? 0, response.layers)) continue;
          const rowName = table.nameIndices ? table.names[table.nameIndices[row]] : undefined;
          if (!selected(rowName, response.rowNames)) continue;
          for (const column of columns) {
            const sign = table.analysis === "modes" || table.analysis === "buckling" ? track.sign : 1;
            add(sign * table.values[row * names.length + column], table.subcase, ids[row], column + 1, table.layers?.[row] ?? 0,
              { nodeId: table.nodeIds?.[row], position: table.positions?.[row],
                step: response.tracksMode ? response.mode ?? response.step : table.step,
                rowName, elementId: table.elementIds?.[row] });
          }
        }
      }
    } else if (response.source === "eigenvalue") {
      const fields = ["eigenvalue", "radians", "cycles", "generalizedMass", "generalizedStiffness"];
      const columns = componentIndices(fields, response.components ?? "eigenvalue");
      for (const table of run.eigenvalues) {
        if (response.subcase !== undefined && table.subcase !== response.subcase) continue;
        for (const row of table.rows) {
          if (track.mode !== undefined && row.mode !== track.mode) continue;
          for (const column of columns) add(row[fields[column]], table.subcase,
            response.tracksMode ? response.mode ?? response.step : row.mode, column + 1);
        }
      }
    } else if (["trimVariable", "stabilityDerivative", "pressure", "boxForce", "boxMoment", "hingeMoment"].includes(response.source)) {
      const aero = run.aeroelastic;
      if (!aero) throw new Error(`response ${response.id} requires aerodynamic results`);
      for (const trim of aero.subcases) {
        if (response.subcase !== undefined && trim.subcase !== response.subcase) continue;
        if (response.source === "trimVariable") {
          for (const variable of trim.variables)
            if (selected(variable.id, response.ids) && (!response.variable || variable.label === response.variable))
              add(variable.value, trim.subcase, variable.id);
        } else if (response.source === "stabilityDerivative") {
          const columns = componentIndices(["cx", "cy", "cz", "cmx", "cmy", "cmz"], response.components);
          for (const row of trim.derivatives) {
            if (response.variable && row.label !== response.variable) continue;
            const variable = trim.variables.find(variable => variable.label === row.label);
            if (!selected(variable?.id ?? 0, response.ids)) continue;
            const data = row[response.derivativeColumn ?? "elasticRestrained"];
            if (!data) throw new Error("unknown stability derivative column");
            for (const column of columns) add(data[column], trim.subcase, variable?.id ?? 0, column + 1);
          }
        } else {
          const field = { pressure: "pressures", boxForce: "boxForces", boxMoment: "boxMoments", hingeMoment: "hingeMoments" }[response.source];
          trim[field].forEach((value, i) => {
            const id = response.source === "hingeMoment" ? i + 1 : aero.boxIds[i];
            if (selected(id, response.ids)) add(value, trim.subcase, id);
          });
        }
      }
    } else throw new Error(`unknown study response source '${response.source}'`);
    if (!values.length) throw new Error(`response ${response.id} selected no result`);
    if (items.length !== values.length) throw new Error(`response ${response.id} requires one item identity per value`);
    ({ values, items } = reduce(values, items, response.reduce));
    const data = Float64Array.from(values, value => value * (response.scale ?? 1));
    if (!data.every(Number.isFinite)) throw new Error(`response ${response.id} contains a non-finite value`);
    return { id: response.id, label: response.label ?? "", values: data, items,
      statuses: Array(data.length).fill("ok") };
  });
  return { responses, trackedModes };
  } finally { queryResults?.delete(); }
}
