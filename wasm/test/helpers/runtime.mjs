import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
export const dist = process.env.OOS_SOLVER_DIST ?? path.join(root, "build/wasm/dist");
export const readerDist = process.env.OOR_READER_DIST ?? path.resolve(root, "../O_o_Reader/build/wasm/dist");
export const { loadOosSolver, supportsWasmThreads } = await import(pathToFileURL(path.join(dist, "index.js")));
const { loadOorReader } = await import(pathToFileURL(path.join(readerDist, "index.js")));
export const solver = await loadOosSolver();
export const oor = await loadOorReader();
export const info = solver.threading();
export const probePath = path.join(dist, "oos_threading_test.js");
export const checks = info.enabled && existsSync(probePath)
  ? await (await import(pathToFileURL(probePath))).default() : null;
