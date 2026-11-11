const abortError = reason => reason instanceof Error ? reason : new DOMException(String(reason ?? "The operation was aborted"), "AbortError");
const positiveInteger = (value, name) => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  return value;
};

export async function createOptimizationWorkerPool(config) {
  const { workers = 2, initializationTimeoutMs = 60000, maximumQueuedPoints = 1024, ...input } = config;
  positiveInteger(workers, "workers"); positiveInteger(initializationTimeoutMs, "initializationTimeoutMs");
  positiveInteger(maximumQueuedPoints, "maximumQueuedPoints");
  if (!input.readerURL) throw new Error("readerURL is required to load the Reader package inside workers");
  if (!(input.model instanceof Uint8Array)) throw new TypeError("model must contain the bytes of a model or archive");
  const options = { ...input, readerURL: String(input.readerURL),
    solverURL: String(input.solverURL ?? new URL("./index.js", import.meta.url)),
    ...(input.hooksURL ? { hooksURL: String(input.hooksURL) } : {}) };
  // Fail once, before starting workers, for callbacks or other values that cannot cross a worker boundary.
  const copied = structuredClone(options);
  let WorkerClass = globalThis.Worker;
  const node = !WorkerClass;
  if (node) ({ Worker: WorkerClass } = await import("node:worker_threads"));
  const pool = new OptimizationWorkerPool(WorkerClass, node, copied,
    { workers, initializationTimeoutMs, maximumQueuedPoints });
  try { await pool.initialize(); return pool; }
  catch (error) { pool.close(error); throw error; }
}

export class OptimizationWorkerPool {
  constructor(WorkerClass, node, config, settings) {
    this.WorkerClass = WorkerClass; this.node = node; this.config = config; this.settings = settings;
    this.slots = []; this.queue = []; this.nextId = 1; this.closed = false;
  }
  async initialize() {
    await Promise.all(Array.from({ length: this.settings.workers }, () => this.start()));
  }
  start() {
    if (this.closed) return Promise.reject(new Error("the optimization worker pool is closed"));
    const worker = new this.WorkerClass(new URL("./study-worker.js", import.meta.url), { type: "module" });
    const slot = { worker, ready: false, task: null, stopped: false };
    this.slots.push(slot);
    return new Promise((resolve, reject) => {
      let initializing = true;
      const timer = setTimeout(() => fail(new Error("optimization worker initialization timed out")), this.settings.initializationTimeoutMs);
      slot.abortInitialization = reason => {
        if (!initializing) return;
        initializing = false; clearTimeout(timer); reject(reason);
      };
      const fail = error => {
        if (slot.stopped) return;
        const wasInitializing = initializing;
        clearTimeout(timer); initializing = false;
        this.stop(slot);
        if (slot.task) this.finish(slot.task, error);
        slot.task = null;
        if (wasInitializing) reject(error);
        else if (!this.closed) this.start().then(() => this.dispatch(), failure => this.close(failure));
      };
      const message = data => {
        if (slot.stopped) return;
        if (data.type === "ready") {
          clearTimeout(timer); initializing = false; slot.ready = true;
          this.info ??= data.metadata; resolve(); this.dispatch();
        } else if (data.type === "error") {
          const error = new Error(data.message); if (data.stack) error.stack = data.stack;
          if (initializing) fail(error);
          else if (slot.task && data.id === slot.task.id) {
            this.finish(slot.task, error); slot.task = null; this.dispatch();
          }
        } else if (data.type === "result" && slot.task && data.id === slot.task.id) {
          this.finish(slot.task, null, data.result); slot.task = null; this.dispatch();
        }
      };
      if (this.node) {
        worker.on("message", message); worker.on("error", fail);
        worker.on("exit", code => { if (!slot.stopped) fail(new Error(`optimization worker exited (${code})`)); });
      } else {
        worker.addEventListener("message", event => message(event.data));
        worker.addEventListener("error", event => fail(new Error(event.message ?? "optimization worker failed")));
      }
      const model = this.config.model.slice();
      worker.postMessage({ type: "initialize", config: { ...this.config, model } }, [model.buffer]);
    });
  }
  metadata() { return structuredClone(this.info); }
  evaluate(x, { signal, timeoutMs, ...options } = {}) {
    if (this.closed) return Promise.reject(new Error("the optimization worker pool is closed"));
    if (this.queue.length >= this.settings.maximumQueuedPoints) return Promise.reject(new Error("maximumQueuedPoints exceeded"));
    if (signal?.aborted) return Promise.reject(abortError(signal.reason));
    if (timeoutMs !== undefined) positiveInteger(timeoutMs, "timeoutMs");
    const values = x === undefined ? undefined : Float64Array.from(x);
    return new Promise((resolve, reject) => {
      const task = { id: this.nextId++, x: values, options, timeoutMs, signal, resolve, reject };
      task.abort = () => this.cancel(task, abortError(signal.reason));
      signal?.addEventListener("abort", task.abort, { once: true });
      this.queue.push(task); this.dispatch();
    });
  }
  dispatch() {
    if (this.closed) return;
    for (const slot of this.slots) {
      if (!slot.ready || slot.task || slot.stopped || !this.queue.length) continue;
      const task = this.queue.shift(); slot.task = task;
      if (task.timeoutMs) task.timer = setTimeout(() => this.cancel(task, new Error("optimization evaluation timed out")), task.timeoutMs);
      try { slot.worker.postMessage({ type: "evaluate", id: task.id, x: task.x, options: task.options }); }
      catch (error) { this.finish(task, error); slot.task = null; queueMicrotask(() => this.dispatch()); }
    }
  }
  finish(task, error, result) {
    if (task.done) return;
    task.done = true; clearTimeout(task.timer); task.signal?.removeEventListener("abort", task.abort);
    if (error) task.reject(error); else task.resolve(result);
  }
  cancel(task, error) {
    if (task.done) return;
    const queued = this.queue.indexOf(task);
    if (queued >= 0) this.queue.splice(queued, 1);
    const slot = this.slots.find(slot => slot.task === task);
    this.finish(task, error);
    if (slot) {
      slot.task = null; this.stop(slot);
      this.start().then(() => this.dispatch(), failure => this.close(failure));
    }
  }
  stop(slot) {
    slot.stopped = true; slot.ready = false;
    slot.abortInitialization?.(new Error("optimization worker initialization was stopped"));
    const result = slot.worker.terminate();
    if (result?.catch) result.catch(() => {});
    const index = this.slots.indexOf(slot); if (index >= 0) this.slots.splice(index, 1);
  }
  async evaluateMany(points, { onPoint, ...options } = {}) {
    const results = new Array(points.length);
    await Promise.all(Array.from(points, async (point, index) => {
      const result = await this.evaluate(point, options);
      if (onPoint) await onPoint(result, index); else results[index] = result;
    }));
    return onPoint ? [] : results;
  }
  close(reason = new Error("the optimization worker pool was closed")) {
    if (this.closed) return;
    this.closed = true;
    for (const task of this.queue.splice(0)) this.finish(task, reason);
    for (const slot of [...this.slots]) { if (slot.task) this.finish(slot.task, reason); this.stop(slot); }
  }
}
