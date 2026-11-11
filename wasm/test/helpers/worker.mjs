import { parentPort, workerData } from "node:worker_threads";
import { cases } from "./register.mjs";

if (workerData?.probe) {
  // Original minimal module: an exported function containing an unconditional loop.
  const bytes = new Uint8Array([
    0,97,115,109,1,0,0,0, 1,4,1,96,0,0, 3,2,1,0,
    7,9,1,5,98,108,111,99,107,0,0, 10,9,1,7,0,3,64,12,0,11,11,
  ]);
  const block = workerData.probe === "wasm" ? new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports.block
    : () => { while (true) {} };
  parentPort.on("message", block); // Deliberately block the event loop; only the parent can interrupt it.
  parentPort.postMessage({ type: "ready", cases: [] });
} else {
  for (const suite of ["smoke", "optimization", "modal-derivatives", "trim-derivatives", "buckling-derivatives", "beam-derivatives", "studies", "static-inputs", "threading", "static-matrix", "connections", "controls"])
    await import(`../cases/${suite}.mjs`);
  parentPort.on("message", async ({ index }) => {
    const started = performance.now();
    try {
      await cases[index].body();
      parentPort.postMessage({ type: "result", elapsed: performance.now() - started });
    } catch (error) {
      parentPort.postMessage({ type: "result", elapsed: performance.now() - started,
        error: error?.stack ?? String(error) });
    }
  });
  parentPort.postMessage({ type: "ready", cases: cases.map(({ name, skip }) => ({ name, skip })) });
}
