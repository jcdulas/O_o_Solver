"""Turn the JSON file of run.py into Markdown tables: one table per analysis, one row per model and size.

Example: python report.py scaling-results.json --output scaling-results.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANALYSIS_TITLES = {
    "statics": "Linear statics",
    "modes": "Normal modes (first 10)",
    "buckling": "Linear buckling (first factor)",
}
STOPPED = {"timeout": "stopped (time)", "memory": "stopped (memory)", "failed": "failed"}
# The solver's message when a run needs more memory than its budget, and the budget it names.
OVER_BUDGET = "memory budget exceeded"
BUDGET = re.compile(r"the budget of ([0-9.]+) GB")
GIB = 1024.0**3
MIB = 1024.0**2


def seconds(value: float) -> str:
    """A time with three significant digits."""
    if value >= 100.0:
        return f"{value:.0f}"
    return f"{value:#.3g}"  # '#' keeps the trailing zeros: 0.300, 12.0


def memory(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    return f"{value / MIB:.0f} MiB"


def time_cell(entry: dict | None) -> str:
    if entry is None:
        return "–"
    if entry["status"] != "ok":
        if OVER_BUDGET in str(entry.get("error", "")):
            return "over budget"
        return STOPPED.get(entry["status"], entry["status"])
    return seconds(entry["run_s"])


def budget_note(runs: list[dict]) -> list[str]:
    """A sentence on the runs the solver stopped because their factor needed more memory than was free."""
    budgets = [float(match.group(1)) for entry in runs if OVER_BUDGET in str(entry.get("error", ""))
               for match in [BUDGET.search(str(entry["error"]))] if match]
    if not budgets:
        return []
    return ["", f"*Over budget*: the solver stopped the run with a message naming what it needed, because the "
                f"factorisation needed more than the memory budget of the run, which is the memory free when the run "
                f"starts ({min(budgets):.1f} to {max(budgets):.1f} GB on this machine)."]


def check_cell(entry: dict | None) -> str:
    if entry is None or entry.get("relative_error") is None:
        return "–"
    return f"{100.0 * entry['relative_error']:+.2f} %"


def table_rows(runs: list[dict], analysis: str, model_order: list[str], many: int) -> list[str]:
    """The rows of one analysis: by model, then by size."""
    by_size: dict[tuple[str, int], dict[int, dict]] = {}
    for entry in runs:
        if entry["analysis"] == analysis:
            by_size.setdefault((entry["model"], entry["size"]), {})[entry["threads"]] = entry
    rows = []
    for (model, size), by_threads in sorted(by_size.items(),
                                            key=lambda item: (model_order.index(item[0][0]), item[0][1])):
        one, all_threads = by_threads.get(1), by_threads.get(many)
        first = one or all_threads
        setup = "–"
        if first["status"] == "ok":
            setup = seconds(first["read_s"] + first["snapshot_s"])
        speed_up = "–"
        if one and all_threads and one["status"] == "ok" and all_threads["status"] == "ok":
            speed_up = f"{one['run_s'] / all_threads['run_s']:.2f}"
        peaks = [memory(entry["peak_memory_bytes"]) if entry and entry["status"] == "ok" else "–"
                 for entry in (one, all_threads)]
        rows.append(f"| {first['title']} | {first['dofs']:,} | {setup} | {time_cell(one)} | {time_cell(all_threads)} "
                    f"| {speed_up} | {peaks[0]} / {peaks[1]} | {check_cell(one or all_threads)} |")
    return rows


def report(document: dict) -> str:
    runs = document["runs"]
    machine, software = document["machine"], document["software"]
    many = max(entry["threads"] for entry in runs)
    model_order = list(document.get("references", {})) or sorted({entry["model"] for entry in runs})
    model_order += sorted({entry["model"] for entry in runs} - set(model_order))
    formulation = document.get("settings", {}).get("formulation") or "the solver's default"

    lines = [
        "# Solver scaling",
        "",
        f"- Machine: {machine['cpu']}, {machine['physical_cores']} cores, {machine['logical_cores']} logical "
        f"cores, {machine['memory_bytes'] / GIB:.0f} GiB of memory, {machine['os']}.",
        f"- Software: oo-solver {software['oo-solver']}, oo-model {software['oo-model']}, "
        f"Python {software['python']}; element formulation: {formulation}.",
        f"- Measured on {document['date']}.",
        "",
        f"Times in seconds. *Read + snapshot*: reading the deck and building the solver's model (one thread). "
        f"*1 thread*, *{many} threads*: the wall time of the analysis. *Peak memory*: the peak resident memory of "
        f"the process, 1 thread / {many} threads. *Check*: the relative difference from the model's reference value.",
        *budget_note(runs),
    ]
    for analysis in ANALYSIS_TITLES:
        rows = table_rows(runs, analysis, model_order, many)
        if not rows:
            continue
        lines += ["", f"## {ANALYSIS_TITLES[analysis]}", "",
                  f"| Model | DOF | Read + snapshot | 1 thread | {many} threads | Speed-up | Peak memory | Check |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *rows, ""]
        for model in model_order:
            reference = document.get("references", {}).get(model, {}).get(analysis)
            if reference and any(e["model"] == model and e["analysis"] == analysis for e in runs):
                title = next(e["title"] for e in runs if e["model"] == model)
                lines.append(f"- {title}: {reference['quantity']}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="the JSON file written by run.py")
    parser.add_argument("--output", type=Path, help="the Markdown file to write (default: print it)")
    args = parser.parse_args()
    text = report(json.loads(args.results.read_text(encoding="utf-8")))
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.reconfigure(errors="replace")
        print(text, end="")


if __name__ == "__main__":
    main()
