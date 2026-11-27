"""Measure how the solver scales with the size of a model and with the number of threads.

For each model, size, analysis and thread count, a fresh Python process reads the deck (oo_model.read_model), builds
the solver's snapshot (oo_solver.model_from_reader), releases the deck and runs the analysis; it reports the three
times, the peak memory of the process and the value compared with the model's reference. A fresh process per run
makes the peak memory that of one run. The results go to a JSON file (written after every run, so that an
interrupted benchmark keeps what it measured; --resume continues it) and a compact table is printed at the end.

Example: python run.py --models plate cylinder --dofs 1e4 3e4 --analyses statics modes
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

import models

RESULT_PREFIX = "SCALING-RESULT "
FORMAT_VERSION = 1
POLL_SECONDS = 0.2
MIB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------------------------------------------------
# One run, in its own process


def peak_memory_bytes() -> int:
    """The peak resident memory of this process so far."""
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    import resource  # not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # bytes on macOS, kibibytes on Linux


def diagnostic_summary(diagnostics: list[tuple[str, str, str]]) -> dict:
    """Counts by severity, and the first messages that are not information."""
    counts: dict[str, int] = {}
    for severity, _, _ in diagnostics:
        counts[severity] = counts.get(severity, 0) + 1
    notable = [f"{severity} {code}: {message}" for severity, code, message in diagnostics
               if severity.lower() not in ("information", "info")]
    return {"counts": counts, "messages": notable[:5]}


def json_ready(value):
    """A value of the result dictionary as JSON data (NumPy arrays and scalars as lists and numbers)."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def measure_in_this_process(spec: dict) -> dict:
    """Read, build and run one deck; the dictionary that the parent process reads."""
    import oo_model
    import oo_solver

    model = models.MODELS[spec["model"]]
    peaks = {"imports": peak_memory_bytes()}

    start = time.perf_counter()
    deck = oo_model.read_model(spec["path"])  # strict: a record with a wrong field stops the reading
    read_seconds = time.perf_counter() - start
    deck_diagnostics = [(str(d.severity), str(d.code), str(d.message)) for d in deck.diagnostics]
    peaks["read"] = peak_memory_bytes()

    start = time.perf_counter()
    if spec["formulation"]:
        snapshot = oo_solver.model_from_reader(deck, formulation=spec["formulation"])
    else:
        snapshot = oo_solver.model_from_reader(deck)
    snapshot_seconds = time.perf_counter() - start
    peaks["snapshot"] = peak_memory_bytes()
    del deck  # the snapshot owns the tables it took
    gc.collect()

    start = time.perf_counter()
    result = snapshot.run(spec["analysis"], threads=spec["threads"])
    run_seconds = time.perf_counter() - start
    peaks["run"] = peak_memory_bytes()

    measurement = {
        "status": "ok",
        "read_s": read_seconds,
        "snapshot_s": snapshot_seconds,
        "run_s": run_seconds,
        "peak_memory_bytes": peaks["run"],
        "stage_peaks_bytes": peaks,
        "measured": model.measured(spec["analysis"], result, spec["size"]),
        "deck_diagnostics": diagnostic_summary(deck_diagnostics),
        "solver_diagnostics": diagnostic_summary(
            [(str(d["severity"]), str(d["code"]), str(d["message"])) for d in result["diagnostics"]]),
    }
    if result["eigenvalues"]:
        rows = models.eigenvalue_rows(result)
        field = "cycles" if spec["analysis"] == "modes" else "eigenvalue"
        measurement["eigenvalues"] = [float(value) for value in rows[field]]
    if "report" in result:
        measurement["report"] = json_ready(result["report"])
    return measurement


def child_main(spec_text: str) -> None:
    spec = json.loads(spec_text)
    try:
        measurement = measure_in_this_process(spec)
    except Exception as error:  # reported to the parent as a failed run
        measurement = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    print(RESULT_PREFIX + json.dumps(measurement), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# The parent: one child process per run, watched for its time and memory


def tree_memory(process: psutil.Process) -> int:
    """The resident memory of a process and of its children (a virtual environment's python.exe on Windows is a
    launcher that runs the interpreter as its child)."""
    total = 0
    try:
        members = [process, *process.children(recursive=True)]
    except psutil.Error:
        return 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            pass
    return total


def stop_tree(process: psutil.Process) -> None:
    try:
        members = [*process.children(recursive=True), process]
    except psutil.Error:
        members = [process]
    for member in members:
        try:
            member.kill()
        except psutil.Error:
            pass


def run_child(spec: dict, kill_after: float, memory_limit: float) -> dict:
    """Run one measurement in a child process; stop it beyond kill_after seconds or memory_limit bytes."""
    with tempfile.TemporaryFile("w+", encoding="utf-8") as output, \
            tempfile.TemporaryFile("w+", encoding="utf-8") as errors:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)]
        process = subprocess.Popen(command, stdout=output, stderr=errors, cwd=Path(__file__).resolve().parent)
        watched = psutil.Process(process.pid)
        start = time.perf_counter()
        stopped = None
        largest = 0
        while process.poll() is None:
            largest = max(largest, tree_memory(watched))
            if largest > memory_limit:
                stopped = "memory"
            elif time.perf_counter() - start > kill_after:
                stopped = "timeout"
            if stopped:
                stop_tree(watched)
                process.wait()
                break
            time.sleep(POLL_SECONDS)
        elapsed = time.perf_counter() - start
        if stopped:
            return {"status": stopped, "elapsed_s": elapsed, "largest_seen_bytes": largest}
        output.seek(0)
        for line in output:
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX):])
        errors.seek(0)
        return {"status": "failed", "error": f"exit code {process.returncode}: {errors.read()[-2000:]}"}


def summarise(samples: list[dict]) -> dict:
    """Median times and the largest peak memory of the samples of one configuration."""
    ok = [sample for sample in samples if sample["status"] == "ok"]
    if len(ok) != len(samples):
        return dict(next(sample for sample in samples if sample["status"] != "ok"))
    summary = dict(ok[0])
    for key in ("read_s", "snapshot_s", "run_s"):
        summary[key] = statistics.median(sample[key] for sample in ok)
    summary["peak_memory_bytes"] = max(sample["peak_memory_bytes"] for sample in ok)
    summary["samples"] = [{key: sample[key] for key in ("read_s", "snapshot_s", "run_s", "peak_memory_bytes")}
                          for sample in ok]
    return summary


# ---------------------------------------------------------------------------------------------------------------------
# Machine and software


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    if sys.platform == "darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def package_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def machine() -> dict:
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "os": platform.platform(),
    }


def software() -> dict:
    return {
        "python": platform.python_version(),
        "oo-solver": package_version("oo-solver"),
        "oo-model": package_version("oo-model"),
        "numpy": package_version("numpy"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# The benchmark


def run_key(model: str, size: int, analysis: str, threads: int) -> tuple:
    return (model, size, analysis, threads)


class Benchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = Path(args.output)
        self.document = {
            "format_version": FORMAT_VERSION,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "machine": machine(),
            "software": software(),
            "settings": {key: value for key, value in vars(args).items() if key not in ("child", "output")},
            "references": {model.name: {analysis: {"value": model.reference(analysis),
                                                   "quantity": model.quantities[analysis]}
                                        for analysis in models.ANALYSES}
                           for model in models.MODELS.values()},
            "runs": [],
        }
        self.done: dict[tuple, dict] = {}
        if args.resume and self.output.exists():
            previous = json.loads(self.output.read_text(encoding="utf-8"))
            for run in previous.get("runs", []):
                if run.get("formulation") != args.formulation:
                    raise SystemExit(f"{self.output} holds runs of another formulation; choose another --output")
                self.done[run_key(run["model"], run["size"], run["analysis"], run["threads"])] = run
            self.document["runs"] = list(self.done.values())

    def save(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.document, indent=1)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8", newline="\n")
        temporary.replace(self.output)

    def record(self, entry: dict) -> None:
        self.document["runs"].append(entry)
        self.done[run_key(entry["model"], entry["size"], entry["analysis"], entry["threads"])] = entry
        self.save()
        print(format_row(entry), flush=True)

    def sizes_of(self, model: models.BenchmarkModel) -> list[int]:
        if not self.args.sizes:
            return sorted({model.size_for_dofs(target) for target in self.args.dofs})
        valid = model.sizes()
        for size in sorted(set(self.args.sizes) - set(valid)):
            print(f"skip {model.name} size {size}: its sizes are {valid.start}, {valid.start + valid.step}, ... "
                  f"up to {valid.stop - 1}")
        return sorted(set(self.args.sizes) & set(valid))

    def measure_size(self, model: models.BenchmarkModel, size: int, analysis: str, path: Path) -> list[dict]:
        """Every thread count of one model, size and analysis; repeated (alternating the order) while it is quick."""
        args = self.args
        threads = [count for count in args.threads
                   if run_key(model.name, size, analysis, count) not in self.done]
        samples: dict[int, list[dict]] = {count: [] for count in threads}
        for repeat in range(max(1, args.repeat)):
            for count in (threads if repeat % 2 == 0 else threads[::-1]):
                spec = {"model": model.name, "size": size, "analysis": analysis, "threads": count,
                        "path": str(path), "formulation": args.formulation}
                samples[count].append(run_child(spec, args.kill_after, args.memory_limit))
            slowest = max((sample.get("run_s", math.inf) for runs in samples.values() for sample in runs),
                          default=math.inf)
            if any(sample["status"] != "ok" for runs in samples.values() for sample in runs) or \
                    slowest > args.repeat_below:
                break
        entries = []
        reference = self.document["references"][model.name][analysis]["value"]
        for count in threads:
            entry = {"model": model.name, "title": model.title, "size": size, "grid_points": model.grid_points(size),
                     "dofs": model.dofs(size), "analysis": analysis, "threads": count,
                     "formulation": args.formulation}
            entry.update(summarise(samples[count]))
            entry["reference"] = reference
            measured = entry.get("measured")
            if measured is not None and reference:
                entry["relative_error"] = measured / reference - 1.0
            self.record(entry)
            entries.append(entry)
        return entries

    def run(self) -> None:
        args = self.args
        work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="oos-scaling-"))
        work.mkdir(parents=True, exist_ok=True)
        try:
            for name in args.models:
                model = models.MODELS[name]
                for analysis in args.analyses:
                    self.run_ladder(model, analysis, work)
        finally:
            if not args.keep_models and not args.work_dir:
                shutil.rmtree(work, ignore_errors=True)
        self.save()
        print()
        print_table(self.document["runs"])
        print(f"\nResults written to {self.output}")

    def run_ladder(self, model: models.BenchmarkModel, analysis: str, work: Path) -> None:
        """The sizes of one model and analysis, smallest first, until a run goes beyond the limits."""
        stop_reason = None
        for size in self.sizes_of(model):
            keys = [run_key(model.name, size, analysis, count) for count in self.args.threads]
            if stop_reason:
                print(f"skip {model.name} size {size} {analysis}: a smaller size {stop_reason}")
                continue
            previous = [self.done[key] for key in keys if key in self.done]
            if len(previous) < len(keys):
                path = work / f"{model.name}-{size}-{analysis}.oos"
                if not path.exists():
                    model.write(path, size, analysis)
                previous += self.measure_size(model, size, analysis, path)
                if not self.args.keep_models:
                    path.unlink(missing_ok=True)
            stop_reason = self.limit_reached(previous)

    def limit_reached(self, entries: list[dict]) -> str | None:
        for entry in entries:
            if entry["status"] == "timeout":
                return f"was stopped after {self.args.kill_after:g} s"
            if entry["status"] == "memory":
                return "was stopped at the memory limit"
            if entry["status"] == "failed":
                return "failed"
            total = entry["read_s"] + entry["snapshot_s"] + entry["run_s"]
            if total > self.args.time_limit:
                return f"took {total:.3g} s, beyond the time limit"
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Printing


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"


def format_row(entry: dict) -> str:
    head = f"{entry['model']:9} {entry['analysis']:9} {entry['dofs']:>9} {entry['threads']:>3}"
    if entry["status"] != "ok":
        return f"{head}  {entry['status']}: {entry.get('error', '')}"[:300]
    error = entry.get("relative_error")
    check = "-" if error is None else f"{100.0 * error:+.3f}%"
    return (f"{head} {format_seconds(entry['read_s']):>8} {format_seconds(entry['snapshot_s']):>8} "
            f"{format_seconds(entry['run_s']):>9} {entry['peak_memory_bytes'] / MIB:>9.0f} {check:>9}")


def print_table(runs: list[dict]) -> None:
    print(f"{'model':9} {'analysis':9} {'DOF':>9} {'thr':>3} {'read s':>8} {'snap. s':>8} {'run s':>9} "
          f"{'peak MiB':>9} {'check':>9}")
    for entry in sorted(runs, key=lambda e: (e["model"], e["analysis"], e["dofs"], e["threads"])):
        print(format_row(entry))


# ---------------------------------------------------------------------------------------------------------------------
# Command line


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return value


def parse_arguments() -> argparse.Namespace:
    logical = psutil.cpu_count(logical=True) or 1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=sorted(models.MODELS), default=list(models.MODELS),
                        help="the models to run (default: all)")
    parser.add_argument("--dofs", nargs="+", type=positive_float, default=list(models.BENCHMARK_DOFS),
                        help="target DOF counts; each model takes its nearest size (default: 1e4 3e4 1e5 3e5 1e6 2e6)")
    parser.add_argument("--sizes", nargs="+", type=int,
                        help="size parameters n instead of DOF counts (see models.py --ladder)")
    parser.add_argument("--analyses", nargs="+", choices=models.ANALYSES, default=list(models.ANALYSES))
    parser.add_argument("--threads", nargs="+", type=int, default=sorted({1, logical}),
                        help=f"thread counts (default: 1 and the {logical} logical cores)")
    parser.add_argument("--formulation", help="element formulation passed to model_from_reader (default: the "
                                              "solver's own, ClassicFormulation)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="samples of a quick configuration; the median is kept (default: 3)")
    parser.add_argument("--repeat-below", type=float, default=10.0, metavar="SECONDS",
                        help="repeat only while every run takes less than this (default: 10)")
    parser.add_argument("--time-limit", type=float, default=1800.0, metavar="SECONDS",
                        help="a run whose read + snapshot + run takes longer is the last size of its model and "
                             "analysis (default: 1800)")
    parser.add_argument("--kill-after", type=float, metavar="SECONDS",
                        help="stop a run after this long (default: 3 × the time limit)")
    parser.add_argument("--memory-limit", type=float, default=0.9, metavar="FRACTION",
                        help="stop a run whose memory exceeds this fraction of the RAM (default: 0.9)")
    parser.add_argument("--output", default="scaling-results.json", help="the JSON file (default: %(default)s)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the runs already in the output file and measure only the others")
    parser.add_argument("--work-dir", help="directory for the generated decks (default: a temporary one)")
    parser.add_argument("--keep-models", action="store_true", help="keep the generated decks")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.kill_after is None:
        args.kill_after = 3.0 * args.time_limit
    args.memory_limit = args.memory_limit * psutil.virtual_memory().total
    args.threads = sorted(set(args.threads))
    return args


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child_main(sys.argv[2])
        return
    sys.stdout.reconfigure(errors="replace")
    args = parse_arguments()
    print(f"{'model':9} {'analysis':9} {'DOF':>9} {'thr':>3} {'read s':>8} {'snap. s':>8} {'run s':>9} "
          f"{'peak MiB':>9} {'check':>9}", flush=True)
    Benchmark(args).run()


if __name__ == "__main__":
    main()
