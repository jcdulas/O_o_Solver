// Each worker owns its Reader and Solver memories. Terminating a worker interrupts a synchronous WASM solve without
// leaving a half-mutated model in another consumer's module.
let send, listen;
if (typeof globalThis.postMessage === "function" && typeof globalThis.addEventListener === "function") {
  send = (message, transfer = []) => globalThis.postMessage(message, transfer);
  listen = callback => globalThis.addEventListener("message", event => callback(event.data));
} else {
  const { parentPort } = await import("node:worker_threads");
  send = (message, transfer = []) => parentPort.postMessage(message, transfer);
  listen = callback => parentPort.on("message", callback);
}

let reader, solver, deck, problem, mode;
function buffersOf(value, buffers = new Set(), seen = new Set()) {
  if (!value || typeof value !== "object" || seen.has(value)) return buffers;
  seen.add(value);
  if (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer) buffers.add(value.buffer);
  else if (value instanceof ArrayBuffer) buffers.add(value);
  else for (const child of Object.values(value)) buffersOf(child, buffers, seen);
  return buffers;
}

listen(async request => {
  try {
    if (request.type === "initialize") {
      const config = request.config;
      const [readerModule, solverModule] = await Promise.all([import(config.readerURL), import(config.solverURL)]);
      reader = await readerModule.loadOorReader();
      solver = await solverModule.loadOosSolver({ threads: 1, ...config.solverOptions });
      reader.writeFile("/work/study.ooh5", config.model);
      deck = reader.readModel("/work/study.ooh5", { strict: true });
      const imported = config.hooksURL ? await import(config.hooksURL) : {};
      const hooks = imported.default ?? imported;
      mode = config.mode ?? "study";
      if (mode === "native") problem = solver.optimizationFromReader(reader, deck,
        { ...config.definition, ...(hooks.callables ? { callables: hooks.callables } : {}),
          ...(hooks.suppliedAt ? { suppliedAt: hooks.suppliedAt } : {}) }, config.settings);
      else if (mode === "study") problem = config.definition
        ? solver.createStudy(reader, deck, config.definition, config.settings, hooks)
        : solver.restoreStudy(reader, deck, hooks).study;
      else throw new Error("worker mode must be native or study");
      send({ type: "ready", metadata: problem.metadata() });
    } else if (request.type === "evaluate") {
      if (!problem) throw new Error("worker is not initialized");
      const options = request.options;
      const reference = mode === "native" ? options.modeReferences ?? [] : options.referencePoint
        ? { _runs: options.referencePoint.analyses, _values: options.referencePoint.values } : null;
      const point = problem.evaluate(request.x, reference);
      try {
        const result = { values: point.values() };
        if (options.includeResults) result.analyses = mode === "native" ? [point.analysisResults()] : point.analysisResults();
        if (options.gradients) result.gradients = point.gradients(options.gradients === true ? {} : options.gradients);
        send({ type: "result", id: request.id, result }, Array.from(buffersOf(result)));
      } finally { point.delete(); }
    } else throw new Error("unknown optimization worker request");
  } catch (error) {
    send({ type: "error", id: request.id, message: error?.message ?? String(error), stack: error?.stack });
  }
});
