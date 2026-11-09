# Element names

The [shared naming convention](https://github.com/jcdulas/O_o_Reader/blob/main/docs/element-naming.md) defines the
element vocabulary shared by Reader and Solver. Model records, result `elementType` values, diagnostics and formulation
choices use the exact same case-sensitive root.

For example, `ShellQuad4` has the element family `ShellQuad4Element`, numerical kernel `ShellQuad4`, geometry
`ShellQuad4Geometry` and source directory `shellQuad4/`. `ShellQuad4Drilling` has its own family, numerical operators
and directory. Its class implements the drilling element directly.

Other families follow the same convention: `SolidHexaElement`, `Bar2Element`, `SpringDofPropertyElement` and
`PointMassMatrixElement`. Reader tables use the corresponding root followed by `Table`.

Formulations describe how an element is computed. Select a formulation with `ShellQuad4Formulation` in a model record
or the `ShellQuad4` key of `elementFormulations`. Call `elementFormulations()` to discover the available choices.
The selected formulation does not change the element type or the result container name. Result layout suffixes such
as `Composite` and `LoadStations` describe the rows in a result table.

Version `0.2.0` uses reader data contract **75**. Reader writes native `.ooh5` archives in format **4**, with native
model schema **1**, and accepts version **3** result archives. Solver consumes the reader's table packages and returns
plain result arrays. Rebuild Reader and Solver together and update clients and input records. Earlier names have no
compatibility aliases.
