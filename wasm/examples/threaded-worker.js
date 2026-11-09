// Load this as a module Worker. Keep the Reader, Solver and their handles inside that Worker.
import { loadOorReader } from "../../../O_o_Reader/build/wasm/dist/index.js";
import { loadOosSolver } from "../../build/wasm-threads/dist/index.js";

const oor = await loadOorReader();
const solver = await loadOosSolver({ threads: 4 });

self.onmessage = ({ data }) => {
  let deck, model;
  try {
    oor.writeFile("/work/input.oos", data.modelText);
    deck = oor.readModel("/work/input.oos", { strict: true });
    model = solver.modelFromReader(oor, deck);
    const result = solver.run(model, data.analysis ?? "statics", data.settings ?? {});
    self.postMessage({ id: data.id, result, threading: solver.threading() });
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error.message ?? error) });
  } finally {
    model?.delete(); deck?.delete();
  }
};
self.postMessage({ ready: true, threading: solver.threading() });
