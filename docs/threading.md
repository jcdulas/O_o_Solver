# Shared-memory WebAssembly threads

The solver has two Memory64 build variants. `wasm` runs on its calling thread. `wasm-threads` compiles the same C++
algorithms with shared-memory workers and uses `base::ThreadPool` for assembly, factorisation, solves, recovery and
derivatives. Reader remains a separate module; it does not need a threaded build.

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
directory before running the test command. Reader defaults to `../O_o_Reader/build/wasm/dist`.

The checks run real tasks on distinct threads, nested batches, task exceptions and worker reuse. Numerical checks
compare statics, modes, gradients and a Hessian-vector product with a one-thread reference. Build the optional C++
checks with `OOS_BUILD_WASM_TESTS=ON` when configuring without the preset.

## JavaScript interface

```javascript
import { loadOosSolver, supportsWasmThreads } from "./build/wasm-threads/dist/index.js";

const solver = await loadOosSolver({ threads: 4 });
console.log(solver.threading());
// { enabled: true, maximumThreads: 8, activeWorkers: 0, defaultThreads: 4 }

const results = solver.run(model, "statics", { threads: 4 });
const problem = solver.createOptimization(model, declarations, { threads: 4 });
```

`threads` counts the caller and its C++ workers: 4 means the calling worker plus 3 C++ workers. Without an explicit
load setting, a threaded build defaults to at most 4 threads, capped by the reported hardware concurrency and the
build's limit. A build without threads defaults to 1 and refuses larger counts. Per-analysis and per-problem settings
override the load default. Counts must be positive integers within `maximumThreads`.

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

```javascript
const worker = new Worker(new URL("./wasm/examples/threaded-worker.js", import.meta.url), { type: "module" });
worker.onmessage = ({ data }) => {
  if (data.ready) worker.postMessage({ id: 1, modelText, analysis: "statics", settings: { threads: 4 } });
  else if (data.error) console.error(data.error);
  else console.log(data.result);
};
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
