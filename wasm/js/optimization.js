import { numericalOptimizationGradients } from "./numerical-derivatives.js";

export class OptimizationProblem {
  constructor(handle) { this.handle = handle; this.info = handle.metadata(); }
  requireAlive() { if (!this.handle) throw new Error("this optimization problem has been deleted"); }
  metadata() { this.requireAlive(); return this.handle.metadata(); }
  designPoint(x) { this.requireAlive(); return this.handle.designPoint(x); }
  evaluate(x = this.info.initialValues, modeReferences = []) {
    this.requireAlive();
    return new OptimizationEvaluation(this.handle.evaluate(x, modeReferences), this.info, this);
  }
  delete() { if (this.handle) { this.handle.delete(); this.handle = null; } }
}

export class OptimizationEvaluation {
  constructor(handle, info, owner) { this.handle = handle; this.info = info; this.owner = owner; this.busy = false; }
  operation(work) {
    if (!this.handle) throw new Error("this optimization evaluation has been deleted");
    if (this.busy) throw new Error("an optimization evaluation cannot be re-entered from a callback");
    this.busy = true;
    try { return work(); } finally { this.busy = false; }
  }
  values() { return this.operation(() => this.handle.values()); }
  optimizerValues() { return this.operation(() => this.handle.optimizerValues()); }
  objective() { return this.operation(() => this.handle.objective()); }
  constraintJacobian(rowColumns) {
    const csr = this.operation(() => this.handle.constraintJacobian(rowColumns));
    return {
      structure: () => csr.structure(),
      values: point => point.operation(() => csr.values(point.handle)),
      constraints: point => point.operation(() => csr.constraints(point.handle)),
      delete: () => csr.delete(),
    };
  }
  analysisResults() { return this.operation(() => this.handle.analysisResults()); }
  nodalCoordinateGradient(responseId, nodeIds = [], item = 0) {
    return this.operation(() => this.handle.nodalCoordinateGradient(responseId, nodeIds, item));
  }
  gradients(settings = {}) {
    return this.operation(() => settings.method === "finiteDifference"
      ? numericalOptimizationGradients({ info: this.info, evaluate: (x, refs) =>
          new OptimizationEvaluation(this.handle.evaluateAt(x, refs), this.info) }, this.handle.values(), settings)
      : this.handle.gradients(settings));
  }
  streamGradients(onBlock, block = 256) {
    return this.operation(() => this.handle.streamGradients(onBlock, block));
  }
  product(kind, input, { count = 1, block = 32, onBlock, ...selection } = {}) {
    return this.operation(() => {
      let rows = input.rows ?? 1;
      if (kind === "jvp") {
        const sizes = this.responseSizes ??= this.handle.responseSizes();
        const ids = selection.responseIds?.length ? new Set(selection.responseIds) : null;
        rows = 0;
        for (let i = 0; i < sizes.length; i += 2)
          if (!ids || ids.has(sizes[i])) rows += sizes[i + 1];
      }
      const columns = kind === "jvp" ? count : this.info.initialValues.length;
      if (![rows, columns, block].every(n => Number.isSafeInteger(n) && n >= 0) || block === 0 || columns === 0)
        throw new RangeError("product dimensions and block must be positive safe integers");
      if (!onBlock && rows * columns * 8 > this.info.maximumOutputBytes)
        throw new RangeError("derivative output exceeds maximumOutputBytes; provide onBlock");
      let values;
      const consume = onBlock ?? (part => {
        // The binding transfers an owned typed array. A complete block is already the requested result.
        if (part.first === 0 && part.rows === rows && part.columns === columns) {
          values = part.values;
          return;
        }
        values ??= new Float64Array(rows * columns);
        if (kind === "vjp") values.set(part.values, part.first * columns);
        else for (let r = 0; r < part.rows; ++r)
          values.set(part.values.subarray(r * part.columns, (r + 1) * part.columns), r * columns + part.first);
      });
      const report = kind === "jvp" ? this.handle.jvp(input, count, consume, { block, ...selection }) :
        this.handle.vjp(input, consume, { block });
      if (!onBlock && !values) values = new Float64Array(rows * columns);
      return { rows, columns, values, report };
    });
  }
  jvp(directions, settings = {}) { return this.product("jvp", directions, settings); }
  vjp(weights, settings = {}) { return this.product("vjp", weights, settings); }
  hessianVector(weights, directions, count = 1) {
    return this.operation(() => this.handle.hessianVector(weights, directions, count));
  }
  hessian(weights) { return this.operation(() => this.handle.hessian(weights)); }
  blocks(weights = null) { return this.operation(() => this.handle.blocks(weights)); }
  clearLinearisation() { return this.operation(() => this.handle.clearLinearisation()); }
  // Weights of objectiveFactor * signed objective + sum multipliers[c] * normalisedConstraint[c].
  // Feed these to vjp, hessianVector, hessian or blocks.
  lagrangianWeights(objectiveFactor = 1, multipliers = null) {
    const point = this.values();
    if (!Number.isFinite(objectiveFactor)) throw new TypeError("objectiveFactor must be finite");
    if (multipliers && multipliers.length !== point.constraints.length)
      throw new RangeError("one multiplier is required per constraint");
    const responses = Object.fromEntries(point.responses.map(r => [r.id, new Float64Array(r.values.length)]));
    if (this.info.objectiveResponseId !== null) {
      const response = point.responses.find(r => r.id === this.info.objectiveResponseId);
      const item = response.items.findIndex(i => i.subcase === 0 || i.subcase === this.info.objectiveSubcase);
      responses[response.id][item] += objectiveFactor * (this.info.objectiveSense === "max" ? -1 : 1);
    }
    point.constraints.forEach((c, i) => {
      const weight = multipliers?.[i] ?? 0;
      if (!Number.isFinite(weight)) throw new TypeError("multipliers must be finite");
      responses[c.responseId][c.item] += weight * c.derivativeScale;
    });
    return { rows: 1, responses };
  }
  delete() {
    if (this.busy) throw new Error("an active evaluation cannot be deleted from a callback");
    if (this.handle) { this.handle.delete(); this.handle = null; }
  }
}
