# `base`

Small building blocks every module uses. It is the lowest layer of the project: it includes only the standard
library, and `platform/` is the only folder of the project allowed to include operating-system headers.

## Folders

| Folder | What it holds |
|---|---|
| `errors/` | `SolverError` and its subclasses (`ContractError`, `UnsupportedCardError`, `SingularityError` with the DOFs and mechanism shapes, `MemoryBudgetError` with the shortfall and largest consumers, `OutputVolumeError` with the remedies, `CacheError`, `OptimisationError`), plus `FileError`, `OperationCancelledError`, `InternalError` and `OOS_ASSERT`, and `concatenate` to build messages. One header per class. An invalid argument of a public function is `std::invalid_argument`. |
| `diagnostics/` | `Diagnostic` (card, ID, field, message), `DiagnosticSink` (thread-safe, sorted output so that reports do not depend on thread timing), `SupportReport` (what a module accepts and refuses, gathered by the support check). |
| `ids/` | `EntityId`, `DofIndex` (32-bit), `Offset` (64-bit), `MAX_DOF_COUNT`; `ComponentCode` (the "123456" of a field). |
| `math/` | `Vec3`, `MatN<R, C = R>` (stack matrices for element kernels, fixed summation order), `Mat3` (with `inverse`), `Mat6` (with `rotationBlock`), `congruence` (tᵀ·k·t), `arctangent2` (atan2 by fdlibm's arithmetic, inlined, so that it gives the same bits on every target). |
| `memory/` | `MemoryBudget` (thread-safe accounting, peak, refusal before allocation with the shortfall), `BudgetedBuffer` (reserves, then allocates), `AlignedBuffer` (64-byte aligned, zero-initialised or uninitialised with `FOR_OVERWRITE`, 64-bit indexed), `KeptObject` (an object kept between analyses in a budget of its own, lent to an analysis with its bytes reserved in the analysis' budget while it holds it). |
| `threading/` | `Stopwatch` (the laps of the report's timing parts), `ThreadPool` (batches of indexed tasks, nested batches allowed; a waiting caller helps only its own and newer batches, newest first; a task runs in the subnormal mode of the thread that pushed its batch), `parallelFor` / `parallelReduce` (pieces fixed by the work, never by the thread count; reductions combined piece by piece in piece order), `TaskGraph` (waves of independent tasks), `CancellationToken`, `ProgressCounter` (at most ten reports a second), `OperationControl` (token and counter handed to a long operation). |
| `arrow/` | The Arrow C Data Interface structures, `ImportedTable` (owns a table moved out of a table package or a stream, checks its layout), `ContractReader` (checks the names and types of a table contract, gives typed views without any copy; the checks that cost O(rows) — nulls, dictionary indices, offsets — run once per column of a table and are recorded in it, `ImportedTable::hasPassed`) and the views, one header each (`ValidityView`, `BooleanView`, `Utf8View`, `DictionaryView`, `ListOffsets`, `ListView`, `FixedListView`, `NestedListView`; `ColumnViews.hpp` gathers them). |
| `platform/` | `TemporaryFile` (written next to its target, renamed onto it on commit, removed otherwise), `MemoryInfo` (total and available physical memory), `CpuFeatures`, `SubnormalModeScope` (the subnormal mode of a thread, IEEE or flushed to zero, set for a scope and given back), `ProcessMemory` (the process's page faults and resident memory, now and at the peak). |

## Rules a user of `base` must know

- **Determinism.** `parallelFor(pool, count, pieceSize, body)` splits `[0, count)` into pieces of `pieceSize`; choose
  `pieceSize` from the work (a named constant or a function of `count`), never from `pool->threadCount()`. Write only
  the part of the result a piece owns; combine partial results with `parallelReduce`, which combines them in piece
  order. Then the bits of a result do not depend on the number of threads.
- **Cancellation.** Pass a `CancellationToken*` to `parallelFor`, `parallelReduce` or `TaskGraph::run`: pieces not
  started are skipped and `OperationCancelledError` is thrown. A piece must stay well under 0.1 s of work, or check
  the token itself. A long operation takes an `OperationControl` (token and progress counter) and says the unit it
  advances the counter by.
- **Errors in parallel work.** The exception of the lowest task or piece index is rethrown, so the error a user sees
  does not depend on the timing either.
- **Memory.** A buffer whose size grows with the model is a `BudgetedBuffer`; its size is reserved before anything is
  allocated. The budget must outlive its reservations and buffers.
- **Arrow.** A `ContractReader` owns its table; the views it returns are valid as long as the reader lives. Nulls are
  refused unless the caller passes `NullPolicy::Allow` and reads `validity()`.

## Where to start reading

`errors/SolverError.hpp`, then `threading/ParallelFor.hpp` and `memory/BudgetedBuffer.hpp`, then
`arrow/ContractReader.hpp`.
