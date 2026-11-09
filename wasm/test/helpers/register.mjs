// Tests register in a reusable worker. The parent owns their deadline, including synchronous WASM calls.
export const cases = [];
export function test(name, options, body) {
  if (typeof options === "function") [options, body] = [{}, options];
  if (cases.some(entry => entry.name === name)) throw new Error(`duplicate test: ${name}`);
  cases.push({ name, skip: options?.skip ?? false, body });
}
