import { Worker } from "node:worker_threads";
import { CASE_BUDGET_MS } from "./budget.mjs";

export { CASE_BUDGET_MS, CASE_CEILING_MS } from "./budget.mjs";
export class CaseWorker {
  constructor(options = {}) {
    this.worker = new Worker(new URL("./worker.mjs", import.meta.url), { workerData: options });
    this.ready = new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error("test module initialization exceeded 60 seconds")), 60000);
      this.worker.on("message", message => {
        if (message.type === "ready") { clearTimeout(timer); resolve(message.cases); }
        else if (message.type === "result" && this.pending) {
          const pending = this.pending;
          this.pending = null;
          clearTimeout(pending.timer);
          if (message.error) pending.reject(new Error(message.error));
          else pending.resolve(message);
        }
      });
      this.worker.on("error", error => { clearTimeout(timer); reject(error); this.fail(error); });
      this.worker.on("exit", code => {
        clearTimeout(timer);
        const error = new Error(`test worker exited (${code})`);
        reject(error);
        if (!this.closed) this.fail(error);
      });
    });
  }
  fail(error) {
    this.failure = error;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(error);
      this.pending = null;
    }
    void this.close();
  }
  run(index, budget = CASE_BUDGET_MS) {
    if (this.failure) return Promise.reject(this.failure);
    if (this.pending) return Promise.reject(new Error("test worker already has a running case"));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error(`test exceeded ${budget} ms; its worker was terminated`)), budget);
      this.pending = { resolve, reject, timer };
      this.worker.postMessage({ index });
    });
  }
  close() {
    this.closed = true;
    return this.worker.terminate();
  }
}
