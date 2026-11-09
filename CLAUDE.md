# CLAUDE.md: working on O_o_Solver

O_o_Solver is a linear finite element solver in C++20, compiled to WebAssembly, that takes its models from the
O_o_Reader module (table packages) and gives its results as plain arrays. Start with [README.md](README.md) and the
`README.md` of each module in `src/<module>/`.

## Rules

1. **Legal** ([LEGAL.md](LEGAL.md)): no vendor or product name in code, comments, messages or documents ("Nastran"
   only to describe the input format, "NASTRAN-95" only to cite NASA's open-source release as a source); never copy or
   quote a vendor's documentation, never add it, an example deck or a result file of a vendor to the repository; no
   vendor file format (no F06-style listing, no binary result format). NASTRAN-95 code is never translated line by line.
   Scientific publications and NASA reports may be cited as the source of a method, with the name the publication
   gives it (author, report number, year).
2. **English** for everything in the repository. The owner writes in French: answer in French in the conversation.
3. **Readable by an intermediate developer** (about five years of experience): plain C++, small functions, named
   constants, comments that say why and give each formula with its source.
4. **Layering**: `base` → `linalg` → `input` → `elements` → `assembly` → `reduction` → `aero` → `output` → `solutions`
   → `design` → `api`; a module uses only those before it. `wasm/src/Bindings.cpp` only converts between JavaScript and
   `oos::api`. Operating-system headers only in `src/base/platform`, SIMD intrinsics only in `src/linalg/kernels/`.
5. **Coupling by data**: no O_o_Reader header nor library; its tables arrive as table packages
   ([docs/table-package.md](docs/table-package.md)) and are read by the data contract (`src/input/contract/`).
6. **WebAssembly threading**: the default build is single-threaded; `OOS_WASM_THREADS` enables shared-memory workers.
   All numerical threading goes through `base::ThreadPool`. JS callbacks stay on the module's calling thread.
7. **C++20 within what GCC 12 has**, `-Wall -Wextra -Wpedantic -Werror` natively. Names: types `CamelCase`, functions
   and variables `camelCase`, members `m_name`, constants `UPPER_CASE`; `.clang-format` (4 spaces, 120 columns).
8. **Only ShellQuad4 and ShellTria3** follow NASTRAN-95; ShellQuad4Drilling, ShellTria3Drilling and the other elements keep their own formulation.

## Build and test

```bash
cmake --preset release && cmake --build build/release -j          # native (MSYS2 UCRT64 on the PATH on Windows)
emcmake cmake -S . -B build/wasm -G Ninja -DCMAKE_BUILD_TYPE=Release -DOOS_BUILD_WASM=ON -DOOS_WARNINGS_AS_ERRORS=OFF
cmake --build build/wasm -j
node --test wasm/test/*.test.mjs                               # needs ../O_o_Reader/build/wasm/dist
```

## Way of working

- Commits on the owner's request only.
- Commit messages must be short, factual and neutral. Describe the project change without references to software
  vendors, products, AI tools or model names, comparisons, or claims about legal compliance.
- Do not append automatic tool-attribution or co-author trailers.
- Line endings LF (`.gitattributes`).
