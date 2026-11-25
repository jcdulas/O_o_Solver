# O_o_Solver documentation

Run static, modal, buckling and aeroelastic analyses from JavaScript (WebAssembly), Python or native C++20. An application builds a deck with O_o_Model, passes it to the Solver and reads result arrays directly.

**[Published documentation](https://jcdulas.github.io/O_o_Solver/)** · [JavaScript / WASM releases](https://github.com/jcdulas/O_o_Solver/releases) · [Python wheels on PyPI](https://pypi.org/project/oo-solver/)

## Start here

- [Getting started](docs/getting-started.md): install either package and run the same example in JavaScript and Python.
- [JavaScript and Python analysis examples](docs/python.md) cover modal frequencies and optimization gradients.
- [Project overview](README.md): capabilities, repository layout and build commands.
- [API reference](docs/api.rst): JavaScript and Python operations, TypeScript declarations and C++ headers.
- [WebAssembly threads](docs/threading.md): shared-memory builds, runtime settings and browser Workers.
- [Couple an external aerodynamic solver](docs/external-aeroelastic-coupling.md) with SOL 101, 103 and 105, or use the built-in SOL 144 trim. The RBE3 wing example gives results for JavaScript, Python and C++.
- [External optimizer API for JavaScript, Python and C++](docs/optimization.md) covers retained points, batched derivatives and CSR callbacks.
- [External optimization](docs/external-optimization.md): SOL 101, 103, 105 and 144 studies, workers and archives.
- [Derivative coverage](docs/derivatives.md): analytic paths, numerical fallbacks and diagnostic reports.
- [Architecture with diagrams](docs/architecture.md): module dependencies, data ownership and execution flow.
- [Module guide](docs/modules.md): implementation structure and detailed technical guides.
- [Download or build the documentation](docs/documentation.md): published site, HTML archives and local builds.

## Detailed guides

- [Structural benchmark comparison](docs/structural-benchmark-comparison.md) compares both formulations across shells, solids, vibration and buckling with analytical targets and published commercial results.
- [Thin-plate bending with solid elements](docs/thin-plate-solid-bending.md) compares element degree, mesh
  distortion and the number of layers against a three-dimensional elasticity solution.
- [Element names](docs/naming.md): shared roots, implementation roles and compatibility.
- [Table packages](docs/table-package.md): model-to-solver data contract.
- [Bar and beam derivatives](docs/beam-derivatives.md): native sizing rates and their solution paths.
- [Four-node shell](docs/shell-quad4-formulation.md), [three-node shell](docs/shell-tria3-formulation.md) and
  [eight-node and six-node shells](docs/shell-quad8-tria6-formulation.md): the formulations of ShellQuad4,
  ShellTria3, ShellQuad8 and ShellTria6.
- [Native element derivative API](docs/element-derivative-api.rst): live C++ contexts and derivative facets.
- [Elements](src/elements/README.md), [assembly](src/assembly/README.md) and [reduction](src/reduction/README.md).
- [Analysis solutions](src/solutions/README.md), [output](src/output/README.md) and [design](src/design/README.md).

## Model and Solver

| Project | Responsibility |
| --- | --- |
| [O_o_Model](https://github.com/jcdulas/O_o_Model/blob/main/index.md) | Model records, model queries and result storage |
| [O_o_Solver](https://github.com/jcdulas/O_o_Solver/blob/main/index.md) | Model snapshots, finite element analyses and result arrays |

JavaScript transfers typed table packages between the two WASM memories. Python passes Arrow buffers within the same process. The Solver returns arrays of results to the caller.

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

docs/getting-started
docs/python
docs/threading
docs/static-inputs
docs/external-aeroelastic-coupling
docs/testing
docs/structural-benchmark-comparison
docs/thin-plate-solid-bending
README
docs/table-package
docs/naming
~~~

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: Optimization and derivatives

docs/optimization
docs/benchmarks/optimizer-api/README
docs/external-optimization
docs/derivatives
docs/beam-derivatives
docs/modal-derivatives
docs/coupled-derivatives
~~~

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: Reference and development

docs/api
docs/element-derivative-api
docs/architecture
docs/modules
docs/mixed-hybrid-formulation-spec
docs/shell-quad4-formulation
docs/shell-tria3-formulation
docs/shell-quad8-tria6-formulation
src/output/sections/README
docs/documentation
LEGAL
~~~
