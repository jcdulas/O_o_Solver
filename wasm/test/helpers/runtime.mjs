import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
export const dist = process.env.OOS_SOLVER_DIST ?? path.join(root, "build/wasm/dist");
export const readerDist = process.env.OOR_READER_DIST ?? path.resolve(root, "../O_o_Model/build/wasm/dist");
export const { loadOosSolver, supportsWasmThreads } = await import(pathToFileURL(path.join(dist, "index.js")));
const modelPackage = await import(pathToFileURL(path.join(readerDist, "index.js")));
// The cases read the diagnostics of their runs from the results: nothing is shown on the console.
export const solver = await loadOosSolver({ diagnostics: "silent" });
export const oor = await (modelPackage.loadOoModel ?? modelPackage.loadOorReader)();
export const info = solver.threading();
export const probePath = path.join(dist, "oos_threading_test.js");
export const checks = info.enabled && existsSync(probePath)
  ? await (await import(pathToFileURL(probePath))).default() : null;
