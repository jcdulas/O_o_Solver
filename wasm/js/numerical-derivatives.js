// Full-analysis differences for responses whose native linearisation is unavailable. Steps respect bounds and
// use a second-order one-sided stencil at a bound. Layout checks prevent differentiating unrelated modal/mesh rows.
export function finiteVector(values, name) {
  const result = Float64Array.from(values);
  if (!result.every(Number.isFinite)) throw new TypeError(`${name} must contain finite numbers`);
  return result;
}

export function checkPoint(x, variables) {
  const result = finiteVector(x, "x");
  if (result.length !== variables.length) throw new RangeError("one value is required per independent variable");
  variables.forEach((v, i) => {
    if (result[i] < v.lower || result[i] > v.upper) throw new RangeError(`variable ${v.id} is outside its bounds`);
    if (v.kind === "integer" && !Number.isInteger(result[i])) throw new RangeError(`variable ${v.id} must be an integer`);
    if (v.choices && !v.choices.includes(result[i])) throw new RangeError(`variable ${v.id} must be one of its choices`);
  });
  return result;
}

export function differenceStencil(x, variables, direction, settings = {}) {
  const relative = settings.relativeStep ?? Math.cbrt(Number.EPSILON);
  const absolute = settings.absoluteStep ?? 0;
  if (!(relative > 0) || !Number.isFinite(relative) || !(absolute >= 0) || !Number.isFinite(absolute))
    throw new RangeError("difference steps must be finite, relativeStep positive and absoluteStep non-negative");
  const v = finiteVector(direction, "direction");
  if (v.length !== x.length) throw new RangeError("one direction component is required per independent variable");
  let step = Infinity, forward = Infinity, backward = Infinity, moving = false;
  variables.forEach((variable, i) => {
    if (v[i] === 0) return;
    if (variable.kind && variable.kind !== "continuous" || variable.choices)
      throw new Error(`variable ${variable.id} is discrete; derivatives require continuous variables`);
    if (variable.lower === variable.upper) throw new Error(`direction moves fixed variable ${variable.id}`);
    moving = true;
    step = Math.min(step, Math.max(absolute, relative * Math.max(Math.abs(x[i]), variable.scale ?? 1)) / Math.abs(v[i]));
    const upper = (variable.upper - x[i]) / v[i], lower = (variable.lower - x[i]) / v[i];
    forward = Math.min(forward, Math.max(upper, lower));
    backward = Math.min(backward, -Math.min(upper, lower));
  });
  if (!moving) return [];
  const side = settings.side ?? "central";
  if (!["central", "forward", "backward"].includes(side)) throw new Error("difference side must be central, forward or backward");
  let offsets, weights;
  if (side === "central" && forward >= step / 2 && backward >= step / 2) {
    step = Math.min(step, forward, backward);
    offsets = [-step, step]; weights = [-0.5 / step, 0.5 / step];
  } else {
    const sign = side === "forward" ? 1 : side === "backward" ? -1 : forward >= backward ? 1 : -1;
    step = Math.min(step, (sign > 0 ? forward : backward) / 2);
    offsets = [0, sign * step, sign * step * 2];
    weights = [-1.5 / (sign * step), 2 / (sign * step), -0.5 / (sign * step)];
  }
  if (!(step > 0) || !Number.isFinite(step)) throw new RangeError("no finite difference step fits the bounds");
  return offsets.map((offset, i) => {
    const point = Float64Array.from(x, (value, j) => value + offset * v[j]);
    if (offset && point.every((value, j) => value === x[j]))
      throw new RangeError("the difference step is too small to change x");
    return { x: point, weight: weights[i], offset };
  });
}

function sameLayout(base, point) {
  if (base.values.length !== point.values.length || base.keys.length !== point.keys.length ||
      base.keys.some((key, i) => key !== point.keys[i]))
    throw new Error("response layout changed during differentiation; use stable response identities and mode references");
  if (!point.values.every(Number.isFinite)) throw new Error("a perturbed response is not finite");
}

export function directionalDifference(evaluate, x, variables, direction, settings = {}, base = evaluate(x)) {
  const stencil = differenceStencil(x, variables, direction, settings);
  const values = new Float64Array(base.values.length);
  let evaluations = 0;
  for (const entry of stencil) {
    const point = entry.offset === 0 ? base : evaluate(entry.x);
    evaluations += entry.offset === 0 ? 0 : 1;
    sameLayout(base, point);
    // Subtract the base before multiplying: translation of a large response must not spoil small derivatives.
    for (let row = 0; row < values.length; ++row)
      values[row] += entry.weight * (point.values[row] - base.values[row]);
  }
  return { values, evaluations, stencil: stencil.map(({ offset }) => offset) };
}

export function numericalJacobian(evaluate, x, variables, settings = {}, base = evaluate(x)) {
  const n = x.length, m = base.values.length;
  const maximum = settings.maximumOutputBytes ?? 256 * 1024 * 1024;
  if (!Number.isSafeInteger(maximum) || maximum <= 0) throw new RangeError("maximumOutputBytes must be a positive safe integer");
  if (!settings.onBlock && m > maximum / 8 / Math.max(1, n))
    throw new RangeError("derivative output exceeds maximumOutputBytes; provide onBlock");
  const values = settings.onBlock ? undefined : new Float64Array(m * n);
  let evaluations = 0;
  for (let column = 0; column < n; ++column) {
    settings.signal?.throwIfAborted();
    let derivative = new Float64Array(m);
    if (variables[column].lower !== variables[column].upper) {
      const direction = new Float64Array(n); direction[column] = 1;
      const result = directionalDifference(evaluate, x, variables, direction, settings, base);
      derivative = result.values; evaluations += result.evaluations;
    }
    if (settings.onBlock) settings.onBlock({ rowBegin: 0, rowCount: m, columnBegin: column, columnCount: 1, values: derivative });
    else for (let row = 0; row < m; ++row) values[row * n + column] = derivative[row];
  }
  return { rows: m, columns: n, values, report: { method: "finiteDifference", evaluations, order: 2 } };
}

export function optimizationVector(point) {
  const values = [], keys = [];
  for (const response of point.responses) {
    response.values.forEach((value, i) => {
      values.push(value); keys.push(`response:${response.id}:${JSON.stringify(response.items[i])}`);
    });
  }
  for (const constraint of point.constraints) {
    values.push(constraint.normalised);
    keys.push(`constraint:${constraint.responseId}:${constraint.item}:${constraint.side}`);
  }
  if (point.objective !== null) { values.push(point.objective); keys.push("objective"); }
  return { values: Float64Array.from(values), keys };
}

export function numericalOptimizationGradients(problem, point, settings = {}) {
  const info = problem.info;
  const variables = Array.from(info.independentIndices, index => info.variables[index]);
  const evaluate = x => {
    const trial = problem.evaluate(x, point.trackedModes);
    try { return optimizationVector(trial.values()); } finally { trial.delete(); }
  };
  const matrix = numericalJacobian(evaluate, point.x, variables,
    { maximumOutputBytes: info.maximumOutputBytes, ...settings }, optimizationVector(point));
  const n = variables.length;
  let offset = 0;
  const responses = point.responses.map(response => {
    const values = matrix.values.slice(offset, offset + response.values.length * n);
    offset += values.length;
    settings.itemMasks?.filter(mask => mask.responseId === response.id).forEach(mask => {
      if (mask.isRetained.length !== response.values.length) throw new RangeError("item mask length differs from response");
      mask.isRetained.forEach((keep, row) => { if (!keep) values.fill(NaN, row * n, (row + 1) * n); });
    });
    return { id: response.id, itemCount: response.values.length, values };
  }).filter(response => !settings.responseIds?.length || settings.responseIds.includes(response.id));
  const constraints = point.constraints.map(() => {
    const values = matrix.values.slice(offset, offset + n); offset += n; return values;
  });
  return { variableCount: n, responses, constraints,
    objective: point.objective === null ? null : matrix.values.slice(offset, offset + n),
    report: { ...matrix.report, solves: 0, factorisations: 0, analyticDerivatives: 0,
      fieldPerturbations: matrix.report.evaluations, shapePerturbations: 0 } };
}
