# O_o_Solver documentation

Solve finite element models through typed tables and return numerical results.

## Start here

- [Getting started](docs/getting-started.md): build requirements and a first working example.
- [Project overview](README.md): capabilities, repository layout and build commands.
- [API reference](docs/api.rst): the current JavaScript types and C++ entry points, included directly from source.
- [Module guide](docs/modules.md): implementation structure and detailed technical guides.
- [Download or build the documentation](docs/documentation.md): private HTML artifacts and local builds.

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
| [O_o_Converter](https://github.com/jcdulas/O_o_Converter/blob/main/index.md) | External input text and .oos interchange |

The source pages and the generated HTML are kept in the private repositories. An authenticated account with access
to the repository is required to download its documentation artifact. No public site is deployed.

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

docs/getting-started
README
docs/table-package
docs/naming
~~~

~~~{toctree}
:hidden:
:maxdepth: 2
:caption: Reference and development

docs/api
docs/modules
src/output/sections/README
docs/documentation
LEGAL
~~~
