# O_o_Solver documentation

Solve finite element models through typed tables and return numerical results.

**[Read the public HTML site](https://jcdulas.github.io/O_o_Solver/)**

## Start here

- [Getting started](docs/getting-started.md): build requirements and a first working example.
- [Project overview](README.md): capabilities, repository layout and build commands.
- [API reference](docs/api.rst): the current JavaScript types and C++ entry points, included directly from source.
- [WebAssembly threads](docs/threading.md): shared-memory builds, runtime settings and browser Workers.
- [JavaScript optimization](docs/optimization.md): design points, derivatives and model updates.
- [Architecture with diagrams](docs/architecture.md): module dependencies, data ownership and execution flow.
- [Module guide](docs/modules.md): implementation structure and detailed technical guides.
- [Download or build the documentation](docs/documentation.md): published site, HTML archives and local builds.

## Detailed guides

- [Element names](docs/naming.md): shared roots, implementation roles and compatibility.
- [Table packages](docs/table-package.md): reader-to-solver data contract.
- [Elements](src/elements/README.md), [assembly](src/assembly/README.md) and [reduction](src/reduction/README.md).
- [Analysis solutions](src/solutions/README.md), [output](src/output/README.md) and [design](src/design/README.md).

## The three projects

| Project | Responsibility |
| --- | --- |
| [O_o_Reader](https://github.com/jcdulas/O_o_Reader/blob/main/index.md) | Model records, model queries and result storage |
| [O_o_Solver](https://github.com/jcdulas/O_o_Solver/blob/main/index.md) | Finite element analyses from reader tables |
| O_o_Converter | External input text and .oos interchange |

The reader and solver have public documentation sites. The conversion component is maintained in a separate
repository with its own access policy.

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

docs/getting-started
docs/optimization
docs/threading
README
docs/table-package
docs/naming
~~~

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: Reference and development

docs/api
docs/architecture
docs/modules
src/output/sections/README
docs/documentation
LEGAL
~~~
