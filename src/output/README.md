# `output`

The results of an analysis in the layouts of the O_o_Reader data contract, so that the caller reads them with the same
code as the results O_o_Reader reads from files: nodal and element tables, grid point forces, strain energies,
eigenvalues. The module also holds what is computed from those results: envelopes over the load cases, section and cut
loads, monitor points of a trim, the checks of a subcase and the estimate of the output volume.

It may use `base`, `linalg`, `input`, `elements`, `assembly`, `reduction` and `aero`; the solutions use it.

## Folders

| Folder | What it holds |
|---|---|
| `arrays/` | The result tables and what makes them from the solution vectors. |
| `envelopes/` | `EnvelopeAccumulator`: per grid point or element, the extreme of a quantity over the load cases and the case that gives it, computed on the fly. |
| `sections/` | Section loads, cut loads and zones of local models; see `src/output/sections/README.md`. |
| `monitors/` | `MonitorPoint1` (the integrated aerodynamic, inertial and applied loads of a component in a trim) and `MonitorPoint3` (the integrated grid point forces of a set of elements at a set of grid points). |
| `diagnostics/` | `SubcaseChecks`: the OLOAD and SinglePointConstr force resultants by load component (FX ... MZ), EPSILON, the external work, the support accelerations. |
| `volume/` | `OutputVolumeEstimate`: the bytes of the requested tables, estimated before the analysis, so that a run whose results would not fit is refused before it starts (`OutputVolumeError` with the estimate and the remedies). |

Where to start reading: `arrays/OutputPlan` (what a subcase asks for), then `arrays/NodalResultBuilder` and
`arrays/ElementResultBuilder` (how the tables are made).

## The result tables (`arrays/`)

- `NodalResultTable`: one nodal table (`displacements`, `singlePointConstrForces`, `multiPointConstrForces`,
  `appliedLoads`): ids, point types (1 grid, 2 scalar point), values (rows × t1 t2 t3 r1 r2 r3) in the displacement
  system of each grid point (stored frame "nodeOutputSystem"). `NodalResultBuilder` makes one from a g-set vector: every
  grid point, then every scalar point, in the order of the node table, or the points of a SET.
- `ElementResultTable`: one element table (`stress`, `strain`, `forces`, `strainEnergy` of one element type such as
  `ShellQuad4` or `Bar2LoadStations`): element IDs, grid point of each place (0 for the centre), layer, positions
  along bars and beams, values (rows × the columns the element family names), in the element frames.
- `GridPointForceTable`: the grid point forces of a subcase (appliedLoad, singlePointConstrForce, multiPointConstrForce,
  one row per element, total), each grid point in equilibrium to round-off.
- `EigenvalueTable` and `ModalEffectiveMassTable`: the eigenvalues of a modal or buckling solution (extraction order,
  eigenvalue, radians, cycles, generalised mass and stiffness) and the modal effective masses (MEFFMASS). A nodal or
  element table of a mode carries its `step` (the mode number, 0 for a static table) and its `eigenvalue`.
- `ResultArrays`: every table of a run, in the order they were made; the WebAssembly bindings hand them to the
  JavaScript layer.

## How the tables are made

- `OutputPlan`: the case section requests of a subcase (`outputRequestsTable` and the members of their SETs, the
  requests and their describers by their neutral names: `DisplacementOutput`, `CornerResults`) mapped to the tables: DISPLACEMENT, SPCFORCES, MPCFORCES, OLOAD, STRESS, STRAIN, PointForceLoad, GPFORCE, ESE, MEFFMASS, and the
  describers that change a table: CENTER or CORNER (one option for all the element outputs of a subcase), FIBER or
  STRCUR, VONMISES or MAXS. A request the solver does not produce is a warning, never a refusal. Without a requests
  table, every table of every point and element is given. `elementSelectionOf(plan)` gives the
  `ElementOutputSelection` the static and modal recoveries share.
- `ElementResultBuilder`: the element tables of a subcase from the displacements, for every family with a Recovery
  facet. The rows are computed range by range on the thread pool (about four ranges per thread, at least
  `MIN_RANGE_ELEMENTS` elements each); each element's rows depend on its own data only, so the bits do not depend on
  the thread count. The elements under BeamDistributedLoad are recovered under their loads, in the station layout
  (`Bar2LoadStations`).
- `ElementResultBuilder::buildCases`: the tables of a block of cases. For the families that complete their rows
  (`vectorisedRecoveryMinCases`, `vectorisedLayerRecoveryMinCases`), the Linear columns are made from the recovery
  slopes of each element, computed once for the block: rest + Σ S·u, summed for every case at once with each case's
  terms in a fixed order; the other columns are completed by the family (a ply's failure indices from its ply states,
  a Weld's stresses from its forces). The rows at rest are made once per set of temperatures. The kinds whose
  selections take the same elements are recovered together, each element resolved and formulated once. Every other
  kind is recovered case by case, as `build()` does. The solutions use it for the cases of the full path, the batches
  of the condensed path and the modes.
- `ResultCombination`: the results linear in the loads (displacements, constraint forces, applied loads) of the load
  cases as one matrix product of the results of the load basis and the coefficients, batch by batch, through the dense
  kernels (deterministic). The non-linear quantities and the element tables are recovered case by case from the
  combined displacements, which keeps every column exact without a per-family rule.
- `StiffnessRecovery`: the outputs that come from the element stiffness matrices: GPFORCE (`−(K_e·u_e − f_e,T)` per
  element and grid point) and ESE (energy, percentage of the total, density, and a subtotal row of element ID
  100000000 per element type). The element matrices are computed a wave of ranges at a time on the pool and their
  forces used in element order; `strainEnergies` computes the energies of many vectors (the modes) with each element's
  stiffness computed once for a group of them.
- `QuantityMap`: a linear map from element table values to declared quantities, with `combineUnitValues` to combine
  the values of the units of a load basis into those of the cases.
