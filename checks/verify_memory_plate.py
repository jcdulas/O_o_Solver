"""Compare the documented in-memory plate in native Python and JavaScript/WASM."""
import argparse
import json
import math
from pathlib import Path
import runpy
import subprocess

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/examples"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reader_js", type=Path, help="Reader distribution index.js")
    parser.add_argument("solver_js", type=Path, help="Solver distribution index.js")
    parser.add_argument("--node", default="node", help="Node.js executable with Memory64 support")
    args = parser.parse_args()
    native = runpy.run_path(str(EXAMPLES / "memory_plate.py"))["solve_plate"]()
    runner = """
const [readerModule, solverModule, plateModule] = process.argv.slice(1);
const {loadOoModel} = await import(readerModule);
const {loadOosSolver} = await import(solverModule);
const {solvePlate} = await import(plateModule);
console.log(JSON.stringify(solvePlate(await loadOoModel(), await loadOosSolver())));
"""
    modules = [args.reader_js, args.solver_js, EXAMPLES / "memory-plate.mjs"]
    output = subprocess.check_output(
        [args.node, "--input-type=module", "-e", runner, *(p.resolve().as_uri() for p in modules)], text=True
    )
    wasm = {int(node): value for node, value in json.loads(output).items()}
    expected = {1: 0., 2: 0.001, 3: 0.001, 4: 0.}
    if native.keys() != expected.keys() or wasm.keys() != expected.keys():
        raise ValueError(f"Unexpected node IDs: Python {list(native)}, WASM {list(wasm)}")
    for node, target in expected.items():
        for value in (native[node], wasm[node]):
            if not math.isclose(value, target, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f"Node {node}: expected {target}, Python {native[node]}, WASM {wasm[node]}")
    print(f"In-memory plate passed in both interfaces: {native}")


if __name__ == "__main__":
    main()
