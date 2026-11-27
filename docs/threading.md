# Threads

The native libraries (Python wheels, C++) run an analysis on every core of the machine unless the caller asks for
another count. The WebAssembly solver has two Memory64 build variants. `wasm` runs on its calling thread.
`wasm-threads` compiles the same C++ algorithms with shared-memory workers and uses `base::ThreadPool` for assembly,
factorisation, solves, recovery and derivatives. Model remains a separate module; it does not need a threaded build.

The thread count changes the time of a run, never its results: the assembly sums every term in the order of a serial
loop, and the dense kernels keep one order of operations for each result element whatever the tiles and threads
([linear algebra guide](../src/linalg/README.md)). The kernel family does change the last bits: the AVX2 kernels of a
recent x86-64 processor use fused multiply-adds, the reference kernels do not.

## Download

Each [release](https://github.com/jcdulas/O_o_Solver/releases) holds both WebAssembly packages:
`O_o_Solver-wasm.zip` (single-threaded) and `O_o_Solver-wasm-threads.zip` (shared memory). The `WebAssembly` workflow
builds both and runs the test suites on each before it publishes them.

## Build and test

With the toolchain environment loaded, build the threaded variant:

```sh
cmake --preset wasm-threads -DOOS_WARNINGS_AS_ERRORS=OFF
cmake --build --preset wasm-threads -j
```

The preset enables `OOS_BUILD_WASM`, `OOS_WASM_THREADS` and the optional runtime threading checks. Its package is in
`build/wasm-threads/dist/`. `OOS_WASM_MAX_THREADS` defaults to 8 and accepts 1 to 64; it includes the calling thread.
The runtime preloads `maximumThreads - 1` workers before the module factory resolves, so C++ may synchronously create
and join its workers. Every library is compiled with the same threading settings. The build without threads stays
available in `build/wasm/dist/`.

Run the existing analysis and optimization tests together with the threading tests. For a shell:

```sh
OOS_SOLVER_DIST=build/wasm-threads/dist node --test --test-concurrency=1 wasm/test/*.test.mjs
```

For a command prompt that sets environment variables separately, set `OOS_SOLVER_DIST` to the threaded distribution
directory before running the test command. Model defaults to `../O_o_Model/build/wasm/dist`.

The checks run real tasks on distinct threads, nested batches, task exceptions and worker reuse. Numerical checks
compare statics, modes, gradients and a Hessian-vector product with a one-thread reference. Build the optional C++
checks with `OOS_BUILD_WASM_TESTS=ON` when configuring without the preset.

## Select a thread count

**JavaScript / WebAssembly**

```javascript
import { loadOosSolver, supportsWasmThreads } from "./build/wasm-threads/dist/index.js";

const solver = await loadOosSolver({ threads: 4 });
console.log(solver.threading());
// { enabled: true, maximumThreads: 8, activeWorkers: 0, defaultThreads: 4 }

const results = solver.run(model, "statics", { threads: 4 });
const problem = solver.createOptimization(model, declarations, { threads: 4 });
```

**Python**

```python
from oo_solver import model_from_reader, optimization_from_reader

result = model_from_reader(deck).run("statics", threads=4)
problem = optimization_from_reader(deck, threads=4)
```

**C++20**

```cpp
const auto model = oos::api::modelFromDeck(deck);
const auto result = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {.threads = 4});
oos::api::OptimizationSettings settings;
settings.threads = 4;
const auto problem = oos::api::prepareOptimization(deck, settings);
```

`threads` counts the caller and its C++ workers: 4 means the calling worker plus 3 C++ workers.

- Python and C++: an analysis or a design problem that leaves `threads` out runs on every logical core the machine
  reports (`threads=None` in Python, `oos::api::AUTOMATIC_THREADS`, which is 0, in C++); `threads=1` keeps the work on
  the calling thread. The `report` of a run gives the count it used.
- JavaScript: without an explicit load setting, a threaded build defaults to at most 4 threads, capped by the reported
  hardware concurrency and the build's limit; a build without threads defaults to 1 and refuses larger counts.
  Per-analysis and per-problem settings override the load default. Counts must be positive integers within
  `maximumThreads`.

Problems with the same count share a pool. Its owner is retained by optimization evaluations and their derivative
factors, even when the JavaScript problem handle is deleted. Delete the evaluations and problem after use.
`activeWorkers` counts live C++ workers, excluding the caller; it does not count idle preloaded runtime workers.
Pools of different sizes share the instance's fixed worker capacity. A request exceeding the remaining capacity
is refused before creating any thread. Release unused problems/evaluations, reuse a live pool's count, or rebuild
with a larger `OOS_WASM_MAX_THREADS`.

The JS interface remains synchronous on the thread that owns the module. User equation callbacks, their partials and
their Hessian-vector callbacks execute on that same thread; C++ numerical workers do not access its JS objects.
Different build variants are separate packages. `supportsWasmThreads()` checks runtime support and browser isolation;
an application can select a package before loading it. There is no automatic downgrade inside a threaded binary.

## Browser integration

Load the threaded package **inside a dedicated module Worker**. The loader refuses the display thread, where waiting
for numerical work could block the interface. The calling Worker runs the synchronous facade; its C++ workers share
one linear memory. Calls made from the display thread should use messages and asynchronous responses.

**JavaScript / WebAssembly**

```javascript
const worker = new Worker(new URL("./wasm/examples/threaded-worker.js", import.meta.url), { type: "module" });
worker.onmessage = ({ data }) => {
  if (data.ready) worker.postMessage({ id: 1, modelText, analysis: "statics", settings: { threads: 4 } });
  else if (data.error) console.error(data.error);
  else console.log(data.result);
};
```

**Python**

In CPython the native solve releases the GIL. A host thread can submit the solve and collect the result later.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=1) as executor:
    pending = executor.submit(model.run, "statics", threads=4)
    # The calling thread can do other work here.
    result = pending.result()
```

**C++20**

```cpp
auto pending = std::async(std::launch::async, [model] {
    return oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {.threads = 4});
});
// The calling thread can do other work here.
const auto result = pending.get();
```

The [worker example](../wasm/examples/threaded-worker.js) reads and solves a model, then deletes its handles. Its
relative imports expect the two repositories beside each other. For an optimization, create the problem and register
its JS functions inside the calling Worker too. A function or bound model handle cannot be transferred with a message.

The application host must provide cross-origin isolation, including these response headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Use a secure context (or the local development exception) and serve the module, its assets and workers with policies
compatible with that isolation. Check `crossOriginIsolated` and `supportsWasmThreads()` in the calling Worker.
The loader reports missing shared Memory64 support or isolation before starting the module. Server-side runtimes do
not use the browser response-header requirement.

## Memory and performance

Threads share the same Memory64 heap; they do not multiply its maximum size. The configured memory ceiling remains
16 GiB. Each active C++ worker has an 8 MiB stack, accounted for separately from solver buffer reservations. Model
data and factors are shared, while the native algorithms budget their temporary workspaces. Returned JS arrays still
own copies. Existing memory and output limits continue to apply.

Speedup depends on model size, factorisation structure and memory bandwidth. Compare phases and total time against
one thread on representative models; increasing the worker count can slow small computations. The numerical methods
and their supported analyses are unchanged.
