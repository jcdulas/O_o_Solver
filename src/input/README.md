# `input`

The model snapshot: everything the solver reads from a model of O_o_Reader, checked against the data contract and
read in place without copy, in one immutable `ModelSnapshot`. The tables come from the O_o_Reader WebAssembly module as
table packages (bytes that keep the Arrow layout), which `api::ModelInput` imports and hands to the `SnapshotBuilder`.

Dependencies: `base` and `linalg` only.

## Names

The tables and diagnostics use the neutral names of the model (data contract 75): card types (`ShellQuad4`,
`ShellProp`), table names (`singlePointConstrTable`), columns (`propId`, `nodesRows`) and word values. Parameters
have their full names in `paramsTable`, such as `WeightToMass` and `ShellQuad4Formulation`.

## Folders

- `contract/`: what the tables must look like. `ContractVersion` (the `data_contract_version` values of O_o_Reader
  this build accepts, checked on the metadata of every table), `TableRole` (the tables that exist once per model, named
  after the O_o_Reader query they come from, with a short description of the optional ones for error messages),
  `TableContracts` (the columns read from each table, with their contract types; the element batch contract accepts
  the grid points as a fixed list or a list, the two layouts O_o_Reader gives), `GroupContracts` (the tables of
  user-defined groups, read by the regrouping of the reduction module).
- `tables/`: one typed view per table, each owning its `base::ContractReader`: `NodeTable` (grid points, positions in
  the basic system, borrowed CD rotations, scalar points), `ElementBatch` (one card type, with the row-index columns),
  `CardTable` (property and method cards), `MaterialTable`, `ConstraintTable` (SinglePointConstr and MultiPointConstr), `RigidTable`, `LoadTable`,
  `SubcaseTable`, `ParamTable` (the typed Parameter table `paramsTable`), `UnitTable` (components, revision stamps,
  boundaries, load case groups), `MassTable` (non-structural mass), `SectionTable` (BarShapeProp and BeamShapeProp section
  properties), `FastenerPiercingTable` (the piercing points of Fastener and Weld: O_o_Reader's `fastenerPiercings`,
  with the grid points and shape functions of the shells pierced, in the layout the elements read), `ContactTable`
  (glued contact: the node pairs, read as the two surfaces near each other, the glue parameters, the pairs a BCONTACT
  selects), `CoordinateSystemTable`.
- `snapshot/`: `ModelSnapshot` (immutable, borrowed), `SnapshotBuilder` (takes the tables over, checks their contracts,
  builds), `SnapshotDiff` (what changed between two snapshots: versions, component and load group stamps),
  `SupportCatalog` (the cards and field values the solver supports, and the `SupportReport` of a snapshot),
  `ColumnOverlay` (new values of some cells of real columns, and the snapshot that reads them: every other column is
  shared in place with the base snapshot, which it keeps alive; this is how a design point of SOL 200 is applied
  without editing the model), `OffsetDefinition` (the offset convention selected by the model), `ElementFormulationChoice`
  (the element formulations the program that built the snapshot chose, by card: `elements::ElementRegistry::forModel`
  reads them before the model's formulation parameter).
- `requests/`: `OutputRequests` (the case section output requests and sets of each subcase, from
  `outputRequestsTable` and `caseSetsTable`).

Optional tables and columns. Some tables are read only when the snapshot was taken with them: the inertial loads
(`inertialLoadsTable`), the unexpanded LoadCombination combinations (`loadCombinationsTable`, with the load tables taken with
`expandCombinations=False`), the compact pressure rows (`pressureFields`), the FreeBodySupport points (`dofSetsTable`), the
connection system (`componentNodes`, `connectionList`, `spiders`, `connectionSharing`), the contact tables
(`contactNodePairs`, `contactPairsTable`, `contactTablesTable`) and the aeroelastic model. Several columns are
optional in their contract, such as the SUBCOM columns of `subcasesTable` (`combinedSubcases`, `coefficients`) and its
`bcontact` and `deform` columns, or the glue columns of the contact tables: without the glue columns, a model with
contact pairs is refused by the assembly (`assembly::requireGlueContract`) rather than guessed.

## Where to start reading

`snapshot/ModelSnapshot.hpp` for what the solver gets, `snapshot/SnapshotBuilder.cpp` for how it is assembled, and
`src/api/ModelInput.cpp` for how the caller's table packages reach the builder.

## Rules

- Nothing is copied that can be read in place: the tables are imported as Arrow C Data Interface structures whose
  buffers stay the producer's; the arrays that are not Arrow tables (the CD rotations) are `BorrowedArray`s that keep
  their owner alive.
- A table the snapshot was taken without is absent, and its accessor throws `ContractError` naming it (and what it
  holds, `prerequisiteOf`). A table built by hand (in a test, for instance) carries no contract version and is vouched
  for by its column contract alone, so it must use O_o_Reader's schemas: a test on such a table and a run on O_o_Reader
  read the same contract.
- Element batches are checked only for the columns every element has (ID, property, grid points, row indices); each
  element folder reads its own fields through the batch's reader with its own column contract.
