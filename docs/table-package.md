# Table packages

A **table package** is one block of bytes that holds a query table of O_o_Model with its exact Arrow layout. It is how a table
leaves the O_o_Model WebAssembly module for this one, which cannot read the other module's memory: in
JavaScript, `QueryTable.toPackage()` gives a `Uint8Array`, and the other module rebuilds the Arrow C Data Interface
structures over those bytes.

In native Python, `model_from_reader(deck)` consumes Arrow C Data Interface capsules in the same process. The binary package described here carries the equivalent table between WASM memories. See the [two examples](getting-started.md).

The plain JavaScript columns of `QueryTable.column()` are made for people and charts: numbers as `Float64Array`,
texts as arrays. A package keeps what a program needs: the Arrow format of every column (`int32`, `float64`,
dictionary-encoded texts, lists, fixed-size lists, booleans, nulls), its validity bitmap, and the schema metadata
(`oor_reader.data_contract_version`).

Writer: O_o_Model's `include/oor/query/table/TablePackage.hpp` (`packTable`). Reader: this project's
`include/oos/base/arrow/TablePackage.hpp` (`importTablePackage`).

## Layout

All integers are little-endian.

| Bytes | What |
| --- | --- |
| 4 | the magic `OOTP` |
| 4 | `u32` the version, 1 |
| 8 | `u64` the byte length of the header |
| header | the root node, written depth first |
| 0 to 7 | padding to a multiple of 8 bytes from the start of the package |
| buffers | the buffers, each starting at a multiple of 8 bytes from the start of this area |

The root node is the table: a struct array (`+s`) with one child per column.

A **node** is one Arrow schema and its array:

| Field | Type |
| --- | --- |
| name | string |
| format | string (the format of the Arrow C Data Interface: `i`, `g`, `u`, `+l`, `+w:3`, ...) |
| flags | `i64` (the schema flags: nullable, dictionary ordered, ...) |
| metadata | `u32` count, then count pairs of strings (key, value) |
| length, null count, offset | three `i64` |
| buffers | `u32` count, then count pairs of `u64` (start in the buffer area, byte length); a null buffer has the start 2^64 − 1 |
| children | `u32` count, then count nodes |
| dictionary | `u8` 1 and a node, or `u8` 0 |

A **string** is a `u32` byte length followed by its UTF-8 bytes.

The buffers are whole: the slots before the array's offset are kept, and the offset is written with the array. Their
byte lengths follow from the format and the length of each array, as the Arrow columnar format defines them (the
validity bitmap: one bit per slot; a 32-bit offsets buffer: 4 bytes per slot plus one; the data of a text column: up to
its last offset).

The consumer checks the contract version in the schema metadata and validates each column before using it.
Card types, fields, parameter values and diagnostics use the neutral names of the model.

## Columns the reader takes when the table has them

The column contracts (`src/input/contract/TableContracts.cpp`) list the columns the solver reads from each table. A
column added to a table of O_o_Model without a change of the contract version is optional there: a table without it
(an earlier O_o_Model, a table built by hand) is still read, and the contract says what its absence means.

| Table | Column | Type | What it gives | Without it |
| --- | --- | --- | --- | --- |
| `pressureTable`, `pressureFields` | `surfaceOrLine` | `dictionary<utf8>`, null for the other cards | `Surface` or `Line` for a FacePressureLoad: a pressure on a face, or a load per unit length along the edges of the element, which the solver refuses | every row is a pressure on a face |
| `rigidElementsTable` | `alpha` | `float64`, null for a card that has none | the thermal expansion coefficient of the rigid element, on each of its rows; a non-zero one is refused under a thermal load | no rigid element has an expansion |
| `nonStructMassTable` | `propertyType` | `dictionary<utf8>`, null for an element target | the property card a row on a property names, by the name the element tables give in `propType`: the card of that ID when it is of the type the non-structural mass card names or counts as it, else the type named | the mass reaches every element of the property ID |
| `lineLoadsTable` | `p2` | `float64` | P2 as the solver reads it: a blank P2 is P1 | a null is read as P1 |
