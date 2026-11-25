"""Check the in-memory plate in JavaScript, Python and optionally the compiled C++ example."""
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
    parser.add_argument("--cpp", type=Path, help="Compiled docs/examples/memory_plate.cpp executable")
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
    results = {"JavaScript": wasm, "Python": native}
    if args.cpp:
        output = subprocess.check_output([args.cpp.resolve()], text=True)
        results["C++"] = {int(node): float(value) for node, value in
                          (line.split() for line in output.splitlines())}
    for language, axial in results.items():
        if axial.keys() != expected.keys():
            raise ValueError(f"{language}: unexpected node IDs {list(axial)}")
        for node, target in expected.items():
            if not math.isclose(axial[node], target, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f"{language}, node {node}: expected {target}, got {axial[node]}")
    print(f"In-memory plate passed in {', '.join(results)}: {native}")


if __name__ == "__main__":
    main()
