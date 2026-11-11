import { checkPoint, finiteVector, numericalJacobian, directionalDifference, optimizationVector } from "./numerical-derivatives.js";
import { studyResponses } from "./study-responses.js";
import { saveStudyPoint } from "./study-archive.js";

export const STUDY_DOCUMENT = "oos.optimization.study";
const clone = value => structuredClone(value);
const yieldToHost = () => new Promise(resolve => setTimeout(resolve, 0));

function serializable(value) {
  return JSON.parse(JSON.stringify(value, (_, item) => {
    if (typeof item === "function" || typeof item === "bigint" || typeof item === "number" && !Number.isFinite(item))
      throw new TypeError("study definitions must contain serializable data and finite numbers; register named callbacks separately");
    return ArrayBuffer.isView(item) ? Array.from(item) : item;
  }));
}

function uniqueIds(values, name) {
  const ids = new Set();
  for (const value of values) {
    if (!Number.isSafeInteger(value.id) || value.id <= 0 || ids.has(value.id))
      throw new TypeError(`${name} require distinct positive integer IDs`);
    ids.add(value.id);
  }
}

function requireDefinition(definition) {
  if (definition.version !== 1) throw new Error("unsupported study definition version");
  if (!Array.isArray(definition.variables) || !definition.variables.length) throw new Error("a study requires variables");
  if (!Array.isArray(definition.responses) || !definition.responses.length) throw new Error("a study requires responses");
  if (!Array.isArray(definition.analyses) || !definition.analyses.length) throw new Error("a study requires analyses");
  uniqueIds(definition.variables, "variables"); uniqueIds(definition.responses, "responses");
  for (const variable of definition.variables) {
    if (![variable.initialValue, variable.lower, variable.upper].every(Number.isFinite) || variable.lower > variable.upper)
      throw new TypeError(`variable ${variable.id} requires finite ordered bounds and an initial value`);
    if (![undefined, "continuous", "integer", "choice"].includes(variable.kind)) throw new Error("unknown variable kind");
    if (variable.kind === "choice" && (!variable.choices?.length || !variable.choices.every(Number.isFinite)))
      throw new Error("choice variables require finite choices");
    if (variable.scale !== undefined && (!(variable.scale > 0) || !Number.isFinite(variable.scale)))
      throw new Error("variable scales must be finite and positive");
  }
  if (definition.objective?.sense !== undefined && !["min", "max"].includes(definition.objective.sense))
    throw new Error("objective sense must be min or max");
  for (const response of definition.responses) {
    const index = response.analysis ?? 0;
    if (!Number.isSafeInteger(index) || index < 0 || index >= definition.analyses.length)
      throw new Error(`response ${response.id} requires a valid analysis index`);
  }
  checkPoint(definition.variables.map(v => v.initialValue), definition.variables);
  for (const entry of definition.analyses)
    if (!["statics", "modes", "buckling", "aeroelastic"].includes(entry.analysis)) throw new Error("unknown study analysis");
  const responses = new Set(definition.responses.map(response => response.id));
  for (const selection of [...(definition.constraints ?? []), ...(definition.objective ? [definition.objective] : [])])
    if (!responses.has(selection.responseId)) throw new Error(`unknown response ${selection.responseId}`);
}

function caseSection(deck, target) {
  if (target.sectionKey !== undefined) return deck.caseControl.section(target.sectionKey);
  if (target.subcase === undefined) return deck.caseControl.global;
  const sections = deck.caseControl.sections();
  const section = sections.find(section => section.id === target.subcase && section.kind === "subcase");
  if (!section && target.subcase === 1 && sections.every(section => ["global", "auxiliaryCase"].includes(section.kind)))
    return deck.caseControl.global;
  if (!section) throw new Error(`missing subcase ${target.subcase}`);
  return deck.caseControl.section(section.key);
}

function parameterValue(parameter, x, indices, hooks, context) {
  if (parameter.callback) {
    const callback = hooks.parameters?.[parameter.callback];
    if (!callback) throw new Error(`missing parameter callback '${parameter.callback}'`);
    const value = callback(context);
    if (value && typeof value.then === "function") throw new Error("parameter callbacks must be synchronous");
    if (typeof value === "number" && !Number.isFinite(value)) throw new Error("a parameter callback returned a non-finite value");
    return value;
  }
  let value = parameter.constant ?? 0;
  const terms = parameter.terms ?? (parameter.variableId === undefined ? [] : [{ variableId: parameter.variableId, coefficient: parameter.coefficient ?? 1 }]);
  for (const term of terms) {
    if (!indices.has(term.variableId)) throw new Error(`unknown parameter variable ${term.variableId}`);
    value += (term.coefficient ?? 1) * x[indices.get(term.variableId)];
  }
  if (!Number.isFinite(value)) throw new Error("a parameter evaluated to a non-finite value");
  return value;
}

const REQUESTS = { displacements: "DisplacementOutput", eigenvectors: "DisplacementOutput",
  singlePointConstrForces: "SinglePointConstrForceOutput", multiPointConstrForces: "MultiPointConstrForceOutput",
  appliedLoads: "AppliedLoadOutput", stress: "StressOutput", strain: "StrainOutput", forces: "ElemForceOutput",
  strainEnergy: "ElemStrainEnergyOutput" };

function constraintsOf(responses, definitions) {
  const result = [];
  definitions.forEach((definition, index) => {
    const response = responses.find(response => response.id === definition.responseId);
    response.values.forEach((value, item) => {
      if (definition.items && !definition.items.includes(item)) return;
      for (const side of ["lower", "upper"]) {
        const bound = definition[side];
        if (bound === undefined) continue;
        const scale = definition.scale ?? Math.max(Math.abs(bound), 1);
        if (!Number.isFinite(bound) || !(scale > 0) || !Number.isFinite(scale)) throw new Error("constraint bounds and scales must be finite");
        const sign = side === "lower" ? -1 : 1;
        result.push({ setId: index + 1, responseId: response.id, item, label: response.items[item], side, bound,
          value, normalised: sign * (value - bound) / scale, derivativeScale: sign / scale });
      }
    });
  });
  return result;
}

export class OptimizationStudy {
  constructor(solver, reader, deck, definition, settings = {}, hooks = {}) {
    this.definition = serializable(definition); requireDefinition(this.definition);
    this.solver = solver; this.reader = reader; this.settings = { maximumOutputBytes: 256 * 1024 * 1024,
      maximumCachedPoints: 0, ...settings }; this.hooks = hooks;
    if (!Number.isSafeInteger(this.settings.maximumOutputBytes) || this.settings.maximumOutputBytes < 1 ||
        !Number.isSafeInteger(this.settings.maximumCachedPoints) || this.settings.maximumCachedPoints < 0)
      throw new Error("study output and cache limits must be non-negative safe integers, with a positive output limit");
    this.indices = new Map(this.definition.variables.map((variable, index) => [variable.id, index]));
    this.base = deck.copy(); this.busy = false; this.cache = new Map(); this.revision = 1;
    this.initialX = Float64Array.from(this.definition.variables, variable => variable.initialValue);
    try { this.bind(); } catch (error) { this.base.delete(); this.base = null; throw error; }
  }
  bind() {
    if (this.definition.update && !this.hooks.updates?.[this.definition.update])
      throw new Error(`missing update callback '${this.definition.update}'`);
    for (const response of this.definition.responses)
      if (response.source === "callback" && !this.hooks.responses?.[response.callback])
        throw new Error(`missing response callback '${response.callback}'`);
    this.parameters = (this.definition.parameters ?? []).map(parameter => {
      if (parameter.callback && !this.hooks.parameters?.[parameter.callback])
        throw new Error(`missing parameter callback '${parameter.callback}'`);
      const target = parameter.target;
      if (!target || !["card", "case", "solution"].includes(target.kind)) throw new Error("unknown parameter target kind");
      if (target.kind !== "card") return clone(parameter);
      const part = target.part ? this.base.part(target.part) : this.base;
      try {
        const card = target.key === undefined ? part?.cardById(target.type, target.id) : part?.cardByKey(target);
        if (!card) throw new Error(`missing parameter card ${target.type}`);
        if (target.slot === undefined && !target.field) throw new Error("a card target requires a field name or a one-based slot");
        if (target.slot !== undefined && (!Number.isSafeInteger(target.slot) || target.slot < 1)) throw new Error("card slots start at one");
        return { ...clone(parameter), target: { ...clone(target), key: card.key } };
      } finally { if (part && part !== this.base) part.delete(); }
    });
  }
  operation(work) {
    if (!this.base) throw new Error("this optimization study has been deleted");
    if (this.busy) throw new Error("an optimization study cannot be re-entered from a callback");
    this.busy = true;
    try { return work(); } finally { this.busy = false; }
  }
  metadata() {
    return { version: 1, revision: this.revision, variables: clone(this.definition.variables),
      initialValues: this.initialX.slice(), maximumOutputBytes: this.settings.maximumOutputBytes,
      derivativeMethod: "finiteDifference", analyses: clone(this.definition.analyses) };
  }
  buildPoint(x) {
    const deck = this.base.copy();
    const context = { deck, reader: this.reader, solver: this.solver, x: x.slice(),
      variables: Object.fromEntries(this.definition.variables.map((variable, index) => [variable.id, x[index]])) };
    try {
      for (const parameter of this.parameters) {
        const target = parameter.target;
        const value = parameterValue(parameter, x, this.indices, this.hooks, context);
        if (target.kind === "solution") deck.setSolution(value);
        else if (target.kind === "case") caseSection(deck, target).set(target.command, value,
          { variant: target.variant, options: target.options });
        else {
          const part = target.part ? deck.part(target.part) : deck;
          try {
            if (!part) throw new Error("missing parameter part");
            if (target.slot === undefined) part.setCardValues(target, { [target.field]: value });
            else {
              const fields = [...part.cardByKey(target).fields];
              while (fields.length < target.slot) fields.push("");
              fields[target.slot - 1] = target.forceReal && typeof value === "number" ? { real: value } : value;
              part.updateCard(target, { fields });
            }
          } finally { if (part && part !== deck) part.delete(); }
        }
      }
      if (this.definition.update) {
        const callback = this.hooks.updates?.[this.definition.update];
        if (!callback) throw new Error(`missing update callback '${this.definition.update}'`);
        const value = callback(context);
        if (value && typeof value.then === "function") throw new Error("update callbacks must be synchronous");
      }
      for (const response of this.definition.responses) {
        const request = response.tracksMode ? "DisplacementOutput" : REQUESTS[response.kind];
        if (request) caseSection(deck, response).set(request, "All");
        if (response.tracksMode && REQUESTS[response.kind] && REQUESTS[response.kind] !== request)
          caseSection(deck, response).set(REQUESTS[response.kind], "All");
      }
      if (this.settings.applyDefaults !== false) deck.applyDefaults();
      if (this.settings.checkReferences) {
        const problems = deck.checkReferences();
        if (problems.length) throw new Error(problems.map(problem => problem.message).join("\n"));
      }
      return { deck, context };
    } catch (error) { deck.delete(); throw error; }
  }
  evaluate(x = this.initialX, references = null) {
    return this.operation(() => {
      x = checkPoint(x, this.definition.variables);
      const { deck, context } = this.buildPoint(x);
      let model;
      try {
        const key = Array.from(x).join(",");
        const cached = !references && this.cache.get(key);
        let runs, values;
        if (cached) ({ runs, values } = cached);
        else {
          model = this.solver.modelFromReader(this.reader, deck, this.settings.model);
          runs = this.definition.analyses.map(entry => this.solver.run(model, entry.analysis,
            { memoryBytes: this.settings.memoryBytes, threads: this.settings.threads, subcases: entry.subcases }));
          const selected = studyResponses(this.definition.responses, runs,
            { ...context, modeReferences: references?._values.trackedModes }, this.hooks, references?._runs);
          const constraints = constraintsOf(selected.responses, this.definition.constraints ?? []);
          let objective = null;
          if (this.definition.objective) {
            const selection = this.definition.objective;
            const response = selected.responses.find(response => response.id === selection.responseId);
            if (selection.item === undefined && response.values.length !== 1) throw new Error("an objective requires a scalar response or an explicit item");
            objective = response.values[selection.item ?? 0];
            if (!Number.isFinite(objective)) throw new Error("objective item is out of range");
          }
          values = { x, subcases: Int32Array.from(new Set(selected.responses.flatMap(response => response.items.map(item => item.subcase)))),
            objective, responses: selected.responses, constraints, trackedModes: selected.trackedModes };
          if (!references && this.settings.maximumCachedPoints > 0) {
            this.cache.set(key, { runs, values });
            while (this.cache.size > this.settings.maximumCachedPoints) this.cache.delete(this.cache.keys().next().value);
          }
        }
        return new StudyEvaluation(this, deck, runs, values);
      } catch (error) { deck.delete(); throw error; }
      finally { model?.delete(); }
    });
  }
  async evaluateMany(points, { signal, onPoint, includeResults = false } = {}) {
    const results = [];
    for (let index = 0; index < points.length; ++index) {
      signal?.throwIfAborted();
      const point = this.evaluate(points[index]);
      try {
        const result = { values: point.values(), ...(includeResults ? { analyses: point.analysisResults() } : {}) };
        if (onPoint) await onPoint(result, index); else results.push(result);
      } finally { point.delete(); }
      await yieldToHost();
    }
    return results;
  }
  saveDefinition(deck, x = this.initialX, state = null, diagnostics = [], trackedModes = []) {
    deck.setDocument(STUDY_DOCUMENT, JSON.stringify(serializable({ schema: "oos.optimization.study", version: 1,
      definition: this.definition, settings: this.settings, x: Array.from(checkPoint(x, this.definition.variables)),
      state, diagnostics, trackedModes })));
  }
  rebuild(deck, definition = this.definition, settings = this.settings, hooks = this.hooks) {
    const study = new OptimizationStudy(this.solver, this.reader, deck, definition, settings, hooks);
    study.revision = this.revision + 1; return study;
  }
  clearCache() { this.cache.clear(); }
  delete() {
    if (this.busy) throw new Error("an active study cannot be deleted from a callback");
    this.base?.delete(); this.base = null; this.cache.clear();
  }
}

export class StudyEvaluation {
  constructor(owner, deck, runs, values) {
    this.owner = owner; this._deck = deck; this._runs = runs; this._values = values; this.busy = false;
  }
  requireAlive() { if (!this._deck) throw new Error("this study evaluation has been deleted"); }
  values() { this.requireAlive(); return clone(this._values); }
  vector() { this.requireAlive(); return optimizationVector(this._values); }
  analysisResults() { this.requireAlive(); return clone(this._runs); }
  model() { this.requireAlive(); return this._deck.copy(); }
  derivative(work) {
    this.requireAlive();
    if (this.busy) throw new Error("a study evaluation cannot be re-entered");
    this.busy = true;
    try { return work(); } finally { this.busy = false; }
  }
  requireSmooth(settings) {
    if (!settings.allowNonsmooth && this._values.trackedModes.some(mode => mode.isUncertain))
      throw new Error("a modal response has uncertain tracking or a repeated root; use a forward directional jvp or explicitly allowNonsmooth");
  }
  vectorAt(x, allowNonsmooth = false) {
    const trial = this.owner.evaluate(x, this);
    try {
      if (!allowNonsmooth && trial._values.trackedModes.some(mode => mode.isUncertain))
        throw new Error("a perturbed modal response has uncertain tracking or a repeated root");
      return optimizationVector(trial._values);
    } finally { trial.delete(); }
  }
  gradients(settings = {}) {
    return this.derivative(() => {
      this.requireSmooth(settings);
      const base = optimizationVector(this._values);
      return { ...numericalJacobian(x => this.vectorAt(x, settings.allowNonsmooth), this._values.x,
        this.owner.definition.variables, { maximumOutputBytes: this.owner.settings.maximumOutputBytes, ...settings }, base), keys: base.keys };
    });
  }
  jvp(directions, { count = 1, ...settings } = {}) {
    return this.derivative(() => {
      const data = finiteVector(directions, "directions"), n = this._values.x.length;
      if (!Number.isSafeInteger(count) || count < 1 || data.length !== n * count) throw new RangeError("directions must have shape variables by count");
      const base = optimizationVector(this._values);
      if (base.values.length > this.owner.settings.maximumOutputBytes / 8 / count) throw new RangeError("derivative output exceeds maximumOutputBytes");
      const values = new Float64Array(base.values.length * count);
      let evaluations = 0;
      for (let column = 0; column < count; ++column) {
        const direction = Float64Array.from({ length: n }, (_, row) => data[row * count + column]);
        const uncertain = this._values.trackedModes.some(mode => mode.isUncertain);
        if (uncertain && settings.side && settings.side !== "forward" && !settings.allowNonsmooth)
          throw new Error("an uncertain modal direction requires a forward difference");
        const side = uncertain ? "forward" : "central";
        const result = directionalDifference(x => this.vectorAt(x, settings.allowNonsmooth || side === "forward"),
          this._values.x, this.owner.definition.variables, direction, { ...settings, side: uncertain ? "forward" : settings.side ?? side }, base);
        evaluations += result.evaluations;
        result.values.forEach((value, row) => { values[row * count + column] = value; });
      }
      return { rows: base.values.length, columns: count, values, keys: base.keys, report: { method: "finiteDifference", evaluations } };
    });
  }
  vjp(weights, { rows = 1, ...settings } = {}) {
    return this.derivative(() => {
      this.requireSmooth(settings);
      const base = this.vector(), data = finiteVector(weights, "weights"), n = this._values.x.length;
      if (!Number.isSafeInteger(rows) || rows < 1 || data.length !== base.values.length * rows)
        throw new RangeError("weights must have shape rows by vector items");
      if (rows > this.owner.settings.maximumOutputBytes / 8 / n) throw new RangeError("derivative output exceeds maximumOutputBytes");
      const values = new Float64Array(rows * n);
      const report = numericalJacobian(x => this.vectorAt(x, settings.allowNonsmooth), this._values.x, this.owner.definition.variables,
        { ...settings, onBlock: block => {
          for (let row = 0; row < rows; ++row) {
            let value = 0;
            for (let item = 0; item < base.values.length; ++item) value += data[row * base.values.length + item] * block.values[item];
            values[row * n + block.columnBegin] = value;
          }
        } }, base).report;
      return { rows, columns: n, values, report };
    });
  }
  lagrangianWeights(objectiveFactor = 1, multipliers = null) {
    this.requireAlive();
    if (!Number.isFinite(objectiveFactor)) throw new Error("objectiveFactor must be finite");
    if (multipliers && multipliers.length !== this._values.constraints.length) throw new RangeError("one multiplier is required per constraint");
    const result = new Float64Array(this.vector().values.length);
    const offset = this._values.responses.reduce((size, response) => size + response.values.length, 0);
    this._values.constraints.forEach((_, i) => {
      const multiplier = multipliers?.[i] ?? 0;
      if (!Number.isFinite(multiplier)) throw new Error("multipliers must be finite");
      result[offset + i] = multiplier;
    });
    if (this._values.objective !== null) result[result.length - 1] = objectiveFactor * (this.owner.definition.objective.sense === "max" ? -1 : 1);
    return result;
  }
  hessianVector(weights, directions, { count = 1, outerRelativeStep = 1e-3, ...settings } = {}) {
    return this.derivative(() => {
      this.requireSmooth(settings);
      const data = finiteVector(directions, "directions"), n = this._values.x.length;
      if (!Number.isSafeInteger(count) || count < 1 || data.length !== n * count) throw new RangeError("directions must have shape variables by count");
      if (n > this.owner.settings.maximumOutputBytes / 8 / count) throw new RangeError("derivative output exceeds maximumOutputBytes");
      let evaluations = 0;
      const gradientAt = x => {
        const trial = this.owner.evaluate(x, this);
        try {
          const gradient = trial.vjp(weights, settings); evaluations += gradient.report.evaluations + 1;
          return { values: gradient.values, keys: Array.from({ length: n }, (_, i) => `gradient:${i}`) };
        } finally { trial.delete(); }
      };
      const base = gradientAt(this._values.x), values = new Float64Array(n * count);
      for (let column = 0; column < count; ++column) {
        const direction = Float64Array.from({ length: n }, (_, row) => data[row * count + column]);
        const result = directionalDifference(gradientAt, this._values.x, this.owner.definition.variables, direction,
          { ...settings, relativeStep: outerRelativeStep }, base);
        result.values.forEach((value, row) => { values[row * count + column] = value; });
      }
      return { rows: n, columns: count, values, report: { method: "finiteDifference", evaluations, order: 2 } };
    });
  }
  hessian(weights, settings = {}) {
    this.requireAlive();
    const n = this._values.x.length;
    if (n > this.owner.settings.maximumOutputBytes / 8 / n) throw new RangeError("derivative output exceeds maximumOutputBytes");
    const directions = new Float64Array(n * n);
    for (let i = 0; i < n; ++i) directions[i * n + i] = 1;
    return this.hessianVector(weights, directions, { ...settings, count: n });
  }
  save(path, { state = null, ...options } = {}) {
    this.requireAlive();
    const deck = this._deck.copy();
    try {
      this.owner.saveDefinition(deck, this._values.x, state, this._runs.map(run => run.diagnostics),
        this._values.trackedModes.map(({ shape, ...summary }) => summary));
      const cycle = state?.iteration ?? state?.cycle;
      const designCycle = Number.isInteger(cycle) && cycle >= -2147483648 && cycle <= 2147483647 ? cycle : -1;
      return saveStudyPoint(this.owner.reader, deck, this._runs, path, options,
        { values: this._values, definition: this.owner.definition, designCycle });
    } finally { deck.delete(); }
  }
  delete() {
    if (this.busy) throw new Error("an active evaluation cannot be deleted");
    this._deck?.delete(); this._deck = null; this._runs = null; this._values = null;
  }
}

export function restoreStudy(solver, reader, deck, hooks = {}) {
  const text = deck.document(STUDY_DOCUMENT);
  if (text === null) throw new Error("the model contains no optimization study");
  const saved = JSON.parse(text);
  if (saved.schema !== "oos.optimization.study" || saved.version !== 1) throw new Error("unsupported optimization study document");
  const study = new OptimizationStudy(solver, reader, deck, saved.definition, saved.settings, hooks);
  try { study.initialX = checkPoint(saved.x, study.definition.variables); }
  catch (error) { study.delete(); throw error; }
  return { study, state: saved.state, diagnostics: saved.diagnostics ?? [], trackedModes: saved.trackedModes ?? [] };
}
