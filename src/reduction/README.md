# `reduction`

Condensation: units, spider absorption, load basis per unit, plans, static and Craig-Bampton condensation, the interface problem, recovery, the cache, the scheduler.

It may use `base`, `linalg`, `input`, `elements`, `assembly`.

Folders: `units/`, `absorption/`, `patterns/`, `plan/`, `condense/`, `external/`, `interface/`, `recovery/`, `cache/`, `scheduling/`.

Where to start reading: the folders in that order, which is the order in which the pieces depend on one another.

## Units, regrouping, boundaries, plans

The objects a caller uses to decide what is condensed. In the order a plan uses them:

- `units/CondensationUnit`, `units/UnitSet`: the units of O_o_Reader as they come, one per component by default
  (`UnitSet::fromComponents`: connectedComponents, its elements, componentBoundaries), with its revision stamp and
  O_o_Reader's boundary.
- `units/Regrouping`: the caller's merges (units made of whole components, merged into one unit of the smallest ID)
  and element lists (new units after the largest ID; the units they take elements from keep the rest). `groups()` is
  what O_o_Reader is asked for; `apply()` builds the units from its answer, O_o_Reader's group tables
  (`input/contract/GroupContracts.hpp`): boundaries and stamps are computed by O_o_Reader, never here. The
  elements the caller declares connections (`DeclaredConnection`) leave every unit and ride with the `UnitSet`;
  `units/ConnectionSystem` adds them as connectors.
- `units/DissectionUnits`: automatic units that follow the connectivity of the model rather than
  its components. The model's node graph (every connection as edges: the planner's element and relation couplings)
  ordered by the geometric nested dissection (the FreeBodySupport points last), its symbolic factor, per supernode its
  subtree's panels, flops and working storage (Liu's stack, the panel of each front beside its children's update
  matrices); the units the maximal subtrees whose recovery (8·(panels + stack) bytes) is at most the bound, consecutive
  siblings of one parent merged while the sum fits; each element of the components in the unit of its first
  eliminated node (an element of the interface alone in the unit with most of its nodes). The interface is the rest of
  the tree; its sizes and those of each depth are reported. The caller regroups the elements as element lists
  (through O_o_Reader's group tables, so the units are cached by their elements); the fasteners
  whose two sides fall in one unit are then interior to it (`assembly::ElementConnectivity` keeps the element of each
  group and the elements of one node).
- `units/ConnectionSystem`: from O_o_Reader's connection tables (`componentNodes`, `connectionList`,
  snapshot roles), the points of each unit, the connections a group does not hold, the 1D/0D-only units linked to two
  units or more moved to the connection system (until none moves), and the clusters (connections joined through points
  no unit holds), each with its stamp (`units/StampHasher`, FNV-1a in a fixed order).
- `units/ClusterAbsorption`: a 1D/0D-only unit linked to one unit is absorbed into it; a cluster touching one unit is
  absorbed into it (its elements and relations become the unit's, its connection-only points interior points, its
  stamp part of the unit's); the rest stays in the connection system. An MultiPointConstr set some subcases do not select keeps its
  cluster in the connection system.
- `units/UnitBoundaries`: the boundary points of each unit and why: interface (a connection of the connection system
  uses it), shared point (an element list), retained, constrained in some subcases only, FreeBodySupport. Point granularity:
  a boundary point brings all its DOFs. A point two units share stays on both boundaries whatever DOFs it keeps for
  them (`plan/BoundaryPromotion`: held in every subcase it brings none, its reaction the sum of the
  units'), and a relation of one unit that makes it dependent is eliminated at the interface (`CondensationPlan`).
- `plan/PlanGraph`: the node graph of the stiffness pattern with, per point, the DOFs not constrained in every
  subcase, the free DOFs (no relation makes them dependent: the interface problem's) and the dependent DOFs by relation
  (`unitDofCount`: a unit eliminates the relations absorbed into it, the interface those of the connection system, so
  the legs of an RigidBodyStar whose reference is a connection-only point are boundary DOFs of their unit).
- `plan/UnitCostEstimate`: the symbolic analysis of each unit with its boundary as trailing set (the ordering rule of
  the full path, `ConstrainedOrdering`; `SymbolicFactor`): boundary and interior DOFs, interior factor, K*, flops,
  memory peak, time at `ESTIMATE_FLOP_RATE`; recommends keeping full a unit whose K* holds more values than its
  interior factor.
- `plan/InterfaceEstimate`: the interface problem before any condensation: its points (condensed units' boundaries,
  full units, connection-only points), its graph (the model's couplings between them and the structural pattern of
  each K*: a clique on the boundary points each connected part of a unit's interior touches; a boundary DOF dependent
  through a relation of the connection system replaced by the points it depends on), ordered on the quotient graph in
  which each condensed unit's boundary is one vertex (`orderOnUnits`), analysed symbolically.
- `plan/CondensationPlan`, `plan/PlanReport`: per unit condensed or full (`PlanRequest`: all, none, a list, units kept
  full, retained points, reduction kind; `keepsFullWhereRecommended`, the automatic plan, keeps full the units whose K*
  would cost more than their interior), the one-unit rule (a model of one unit and no connection is the full path),
  the estimates of the interface problem and of the full path; the text report of the plan.

What the plan needs from a solution (the analysis model, the subcases' SinglePointConstr and MultiPointConstr sets, Gm, the node graph) is
gathered by `solutions/common/CondensationPlanner`, which the caller drives (list the units, regroup them, plan).

The group and connection tables come from O_o_Reader: the points of each component from `componentNodes`, the group
summary and boundaries from `groupRevisions` and `componentBoundaries(groups=)` (keyed by `group`, the decimal text
of the group ID), the clusters from `connectionClusters`, and the list of every connection with its stamp
(`connectionList`) from `connectionClusters(detail="connections")` (the direct links between components with a null
cluster). Without the `sourceSetId` column of multiPointConstrTable, a MultiPointConstrUnion's cards are matched to it by their equations
(`MpcSetAlias` of `units/ConnectionSystem`, given by the planner).

## Spider and fastener side absorption

After the cluster absorption, the spiders still in the connection system are absorbed into their unit when they may
be (fastener sides are absorbed like spiders), so that the boundary holds their reference points instead of their
legs. In the order a plan uses them (`absorption/`):

- `Spider`, `SpiderCatalog`: every RigidBodyStar and RigidInterpolation of O_o_Reader's `spiders` and every side of every Fastener and Weld of
  the piercing table (`fastenerPiercings`), with O_o_Reader's topological decision (`isSpider`, `isEligible`, what
  blocks it) and its stamp. `fastenerPiercings` has no eligibility columns: its sides are not absorbed
  (`noEligibility`) until the reader gives them.
- `SpiderWeights` (a RigidInterpolation's weighting matrix from its rows of Gm), `FastenerPiercing` (a fastener side's from the
  interpolation of its piercing point, `elements::interpolatePiercing`): `u_ref = W·u_legs`.
- `SpiderChangeOfBasis`: `u_legs = W⁺·u_ref + N·q` by a Householder QR of the scaled `Wᵀ` (rotations scaled by the
  characteristic length), applied to the rows of the legs (`transformRows`, `transformSymmetric`) and to the
  recovery (`expand`); never the flexibility form, never Lagrange multipliers.
- `AbsorptionSafeguard`: the condition number of each change of basis; rank-deficient or above
  `SPIDER_ABSORPTION_MAX_CONDITION` (50, calibrated on shell models) keeps the spider on its legs.
- `AbsorptionEligibility`: the order of the rules (O_o_Reader's topology, a leg on a contact between bodies, a leg
  retained, on a spline node, on a FreeBodySupport point, constrained in some subcases only or in every subcase, not requested,
  the safeguard); `SharedConnectionNodes` (O_o_Reader's `connectionSharing`) names what touches a leg.
- `SpiderAbsorption`: the decisions for a cluster absorption; per unit the references put on its boundary, the
  RigidBodyStar/RigidInterpolation that become its elements and its stamp with them; per connection the points the absorption takes off the
  interface; the RigidInterpolation references freed (the planner frees their DOFs in the plan graph). `AbsorptionReport`: the text
  of the plan report.

`plan/BoundaryPromotion` is the rule for a boundary point some of whose DOFs a relation makes dependent (every
kind of relation, the glue equations included): a relation eliminated at the
interface leaves the DOFs to the unit as boundary DOFs; a relation eliminated inside the unit promotes the points the
DOFs depend on to the boundary, so that no coupling is lost.

Glued contact: each glued grid point is a relation of the plan graph, `glueRelationKey` (−(1 + nodeId), never
an element or set ID). `CondensationPlan` takes, with the connection system's relations, the glue relations whose
points lie on several units (its last argument, from `solutions::GlueUnitCheck`): eliminated at the interface, their
points on the boundaries (`BoundaryPromotion::interfaceRelationPoints`); the others are eliminated inside their unit.

The change of basis is applied by the static condensation to the assembled rows of the legs of each absorbed
spider (stiffness, load patterns) before the partial factorisation, and the legs are recovered from `(u_ref, q)`
by `expand`; for a fastener side whose GA (GB) is a grid point of the model, W is that point's rows of Gm
(`assembly::FastenerConstraints`), which `FastenerPiercing` documents.

## Load basis per unit, static condensation, recovery

One unit reduced exactly, with work proportional to its load basis. In the order an analysis uses them:

- `patterns/UnitLoadBasis`: the load basis (`assembly::LoadBasis`) chosen on the compact rows that load the
  unit (`UnitFamilyLoads::isUnitRow`), the families stacked; a loaded-DOF basis in the order of its DOFs, with a stamp
  of its DOFs only (`patternStamp`: new load values on the same DOFs reuse the condensation). `BasisCoefficients`: α
  of every case and the one combination of basis results into case results (`combine`: each case summed over the basis
  in increasing order by the dense kernels: the same bits in any batch, with any threads).
- `condense/UnitDescription`: what a unit is made of (its points, elements, boundary, the spiders absorbed by change of
  basis with their relation; a fastener side whose point is generated, not given, gets a point of its own, six DOFs in
  the basic system at the piercing point). `condense/UnitElementIndex`: the elements of the model by ID, built once.
- `condense/UnitSystem`: the unit's equations (each element's stiffness computed once for its graph and its assembly,
  kept within an eighth of the memory left: `KeptStiffness`). Local numbering of its points (model order, then generated points), the
  role of each DOF (interior, boundary, held in every subcase, dependent inside the unit, an empty leg slot), its own
  element matrices assembled in the model's order, the relations eliminated inside it (the units' Gm:
  `assembly::RigidElimination` with a `ConstraintRowFilter` that leaves the connection system's relations out, so that
  chains resolve inside the unit) by the linear elimination in place, each absorbed spider by `K ← Tᵀ·K·T` on the
  assembled leg rows (the reference rows moved to the reference point, the deformation rows to the last leg slots), the
  ordering with the boundary last (the rule of `UnitCostEstimate`), the symbolic factor and the SymmetricCsc. Loads go
  `P → Tᵀ·(P̄_n + Gmᵀ·P_m) − K_fs·u_s` (`equationLoads`), solutions back by `expandSolution`. The unit's mass gives
  `Φ_rᵀ·M·Φ_r` and `−M·Φ_r` (the inertial patterns) about the basic origin.
- `condense/StaticCondensation`: `PartialFactor` on the unit's equations; `K*`, the patterns `f*_k` by the forward
  sweep of the interior factor, the six inertial patterns; the interior factor kept in memory, released and recomputed
  from the kept equations to the same bits. `condense/ReducedUnit`: what is kept (only light data, never the interior factor): `K*` with its labels,
  `f*_k` with the patterns' stamp, the inertial patterns, the rigid-body mass properties, and the boundary forces
  `K*·u_a − f*·α`.
- `recovery/UnitRecovery`: `u_o = K_oo⁻¹·(Σ_k α_k f_o,k − K_oa·u_a)` on the boundary basis of the interface problem
  (`UnitBoundaryBasis`: V, β and the columns that are the unit's own patterns), on the regime of fewest interior solves
  (the basis columns with the patterns riding on them, the load cases, or the boundary DOFs with the patterns apart),
  combined per case in a fixed order, by batches of cases (`recoverCases`); `basisDisplacements` and
  `basisCoefficients` for element results computed on the basis then combined. `RecoveryStatistics` counts the solves.

## The interface problem and the boundary basis

The pieces between the condensed units and the static solution. In the order the condensed SOL 101 (`solutions/sol101/CondensedStatics`) uses them:

- `condense/PlannedUnits`: the units a plan condenses as their condensation takes them: a `UnitDescription` per unit
  (the plan's points, elements and boundary; the given GA of an absorbed fastener side added as its reference; the
  relations of its absorbed spiders: an RigidInterpolation's rows of the units' Gm, a given GA's rows, a generated side's
  interpolation at its piercing point); `unitRowFilter()` (the relations eliminated inside the units: all but the
  connection system's) and `interfaceRowFilter()` (those eliminated at the interface: all but the rows of the
  units' interior dependent DOFs and spider references; a row a unit eliminates at one of its boundary points is
  eliminated at the interface too, the elimination being a congruence). A point an element of the interface reaches
  inside a unit (`reachedPoints`, e.g. the grid points of a fastener's pierced element missing from the connection
  tables) is put on the unit's boundary, as `BoundaryPromotion` does for relations.
- `condense/LocalRelations` (the linear elimination `Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm` on a local block
  pattern; `reduce` P_n += Gmᵀ·P_m, `expand` u_m = Gm·u_n) and `condense/SymmetricBlockAccess` (the 6 × 6 blocks of a
  local pattern): shared by `UnitSystem` and the interface problem. `UnitSystem` now also decides the AutoSingularityConstr of its
  interior points (3 × 3 blocks of its raw stiffness, `assembly::SingularBlock`; a singular leg of an absorbed spider
  is refused) and keeps its raw boundary blocks for the interface's check and its held couplings for its SinglePointConstr forces.
- `interface/InterfaceProblem`: one node-block system per group of subcases: its points (every model point not
  interior to a condensed unit, the generated fastener points), `K*` of each condensed unit added on its labels, the
  elements of no condensed unit assembled in the model's order (a fastener on absorbed legs carried to the reference by
  W⁺, exact: `FASTENER_FACTOR_TOLERANCE` checks that its rows lie in W's row space), the group's relations of the
  interface eliminated, AutoSingularityConstr decided on the summed raw blocks (the interface elements' raw matrices and the units'
  raw boundary blocks: the full path's sums), ordered on the quotient graph (`InterfaceEstimate::orderOnUnits`) and
  factorised in memory (MaxDiagonalRatio, SingularityBailOut). Its rigid-body modes and mass (`Φ_rᵀ·M·Φ_r`, `−M·Φ_r`) serve inertia
  relief. Without its factor (`InterfaceInputs::keepsFactor` false): built without its factorisation, each
  solve streamed by `linalg::StreamedSolve` (by regions of its tree under `regionValues`), the pivot diagnostics with
  the first solve, the solves' figures (`InterfaceSolveFigures`).
- `interface/BoundaryBasis`: the Q distinct right-hand sides (the units' `f*` patterns the cases use, the interface's
  own load vectors, the relief loads) solved by blocks of columns: V, the reduced loads, the right-hand sides and the
  residuals per column; the boundary displacements of every case are `Σ_q β_q·V_q`; when Q exceeds the cases, the
  cases themselves (β the identity). `unitRows(u)` gives a unit's rows of V for `UnitRecovery`.
  `BoundarySolveOptions`: the solve skipped (the right-hand sides handed to the caller), or replaced by given solutions of
  the equations (per column or per case) or by given local displacements: the same V for the same solutions.

## The condensation cache

Never condense twice what has not changed, and store only what is light. In `cache/`, in the order a store uses them:

- `CacheHasher`: FNV-1a on 64-bit words, the hash of the keys, the content checksums and the file checksum.
- `CacheKey`: its parts (version, kernel, the `MATRIX_PARAMETERS`, the unit's identity,
  revision, design, constraint, boundary and spider stamps, the reduction), the patterns' checksum and the input stamp
  (what the solver reads for the unit beside the stamps: `unitInputStamp`), all made from the unit's inputs by
  `makeUnitKey` (no equation built); the checksum of the equations (`equationStampOf`) stored with
  an entry as its verification, not matched; `differenceFrom` names why a stored entry does not serve.
- `CacheEntry`: `CachedReduction` (the `ReducedUnit`, the pattern work G, the pivot ratio; no factor) and
  `CachedRecoveryData` (the Craig-Bampton modes).
- `CacheFile`: one entry per file, little-endian words, header, key, sections, checksum; written through a temporary
  file; a read refuses another format, a foreign, truncated or corrupted file, naming why.
- `CacheLimits` (cache_max_size, cache_recovery_max_size, cache_max_age), `CacheCleanup` (superseded, expired, least
  recently used with recovery data first, never an entry in use), `CacheLock` (a lock folder shared by processes,
  broken when abandoned).
- `CondensationCache`: the memory layer (its own budget) and the folder (`entries/<identity>/<R|M>-<key>.nsvc`), its
  sessions (an analysis pins what it uses), `info`, `statistics`, `clear`, `verify`. A store decides its cleanup on a
  listing of the folder it keeps, made again every `CACHE_LISTING_REFRESH_SECONDS` or when the entry needs room
  (a listing per store cost 10 000 units the square of their count in directory reads). The stores of a wave
  of condensations are made together (`Session::storeReductions`): each decided as `store` decides
  it, in order, under one lock, the planned entries kept in a listing made again, the files written on the pool; and
  off the analysis' critical path by the session's writer (`storeReductionsLater`, `finishStores`).

`condense/StaticCondensation` takes a cached reduced unit (no factor: `recomputeInteriorFactor` makes the first one
and checks `K*` to the bit); `solutions/sol101/CondensedStatics` looks each unit up and stores the new ones.

## Memory budget, scheduler, factor retention

The condensed path within the budget the caller sets, interior factors kept in memory when they fit. In `scheduling/`, in the order an analysis
uses them:

- `UnitFootprint`: what one unit takes, estimated before its equations exist from the plan's `UnitCostEstimate` (which
  now also gives the node blocks, the matrix entries, the panel rows and supernodes, and the bytes of the partial
  factorisation on one thread) and its patterns: its kept data (equations, patterns, reduced unit, pattern work), the
  transient of the build of its equations, the transient of its factorisation, its factor once made, its recovery.
  `InterfaceFootprint`: the interface phase of a constraint set from the plan's `InterfaceEstimate` (the same new
  sizes): the set's g-set arrays, the interface problem at its largest stage, the boundary basis on min(Q, cases)
  columns. Upper bounds, with the margins their comments justify by measurement.
- `CondensedMemoryPlan`: the minimum peak of the analysis (the condensation one unit at a time in the scheduler's order,
  or the later phases beside what every unit keeps: the interface, the recoveries, one refactorisation, one case, the
  results), its largest consumers, and `requireFits`, the refusal before any work (`base::MemoryBudgetError`). Its
  strategies take the interface problem without its factor (`MemoryStrategy::isInterfaceWithoutFactor`,
  `interfaceRegionBytes`; `InterfaceFootprint::solveBytes` and `recomputedFlops` from `InterfaceEstimate::streamedSolve`
  and `regionCosts`): `"low_memory"` when it needs no more than keeping the factor, `"auto"` when nothing fits with
  its factor.
- `UnitScheduler`: the units by decreasing work, in waves: a large unit (`LARGE_UNIT_MINIMUM_FLOPS` and at least the
  remaining work over the threads) alone with every thread; otherwise first fit of the following units, one thread
  each, up to `unitsPerThread` per thread (4 on the condensed path), while their peaks fit the bytes
  free and leave the reserve of the rest (`reserve()`); a wave of one unit has every thread.
- `FactorRetention`: after each wave, in order, a unit's interior factor is kept for its recovery when the bytes free
  with it held cover the reserve of the rest, released otherwise (named, with the reason); a unit not recovered always
  releases it.
- `KeptInteriorFactors`: the factors a static analysis still holds at its end, kept for the
  later analyses of the process (memory only, never in the cache), each with a copy of the equations it was made from
  (`KeptInteriorFactor`, `keepInteriorFactor`); `take` gives a unit the factor of its equations compared in full (the
  boundary labels, the interior count, the matrix to the bit, the symbolic shape, `PartialFactor::isFactorisationOf`:
  the kernel family and pivot policy), one factor per unit, within a limit (an eighth of the physical memory by
  default), the earliest offered dropped first. `condense/StaticCondensation` condenses with a kept factor (its third
  constructor: no factorisation, its patterns and inertial patterns by the factor's forward sweep, the bits of a fresh
  condensation), takes one for a recovery (`adoptInteriorFactor`, its `K*` checked to the bit) and hands its own over
  (`giveInteriorFactor`); the kept factor's buffers stay in the budget of the analysis that made them (kept alive), its
  bytes lent to the budget of the analysis that holds it.

`condense/StaticCondensation::releaseInteriorFactor` frees the whole partial factor (its own copy of `K*` with the
panels) and `recomputeInteriorFactor` checks the new `K*` against the reduced unit's to the bit;
`interface/InterfaceProblem` factorises on one thread when its threaded working storage does not fit the budget left.

The memory mode: `MemoryMode` (`Auto`, `Speed`, `LowMemory`)
and `CondensedMemoryPlan`'s strategies (`RecoveryStrategy::AllAtOnce` or `Sequential`, with the units whose equations
are released): the plan computes the smallest budget of every recovery at once, of the recovery in sequence and of
every unit's equations released, and takes the mode's (Auto: the first that fits, then the fewest releases that fit,
those that free the most first). `UnitFootprint::lightBytes` (a unit without its equations), `passBytes` and
`passPeakBytes` (its pass in sequence, its equations built again when released); `InterfaceFootprint::heldBytes` (the
factorised interface beside the later phases) and the sets' measured relations. `condense/StaticCondensation::
rebindSystem` replaces a unit's equations by another build of it (the system without them after its condensation, its
equations built again without the mass for its recovery: `UnitSystem::WithoutMass`); `UnitSystem::makeRigidBodyModes`
makes the rigid-body modes from the points when they are read (no system keeps them).

## `external/` — external units: the stiffness of a local model given back

A plugin of the caller replaces the elements of a zone by the matrices of its own model condensed onto the contour
DOFs of the zone. The reduction layer holds the unit and its checks; the full path uses them through
`solutions/sol101/ExternalUnitPath`.

- `ExternalUnit`: what the plugin gives, its rows named by `UnitDofLabel` (grid point and component in the
  displacement system of the grid point): K*, the reduced mass M* (optional), the source of its condensed loads
  (`ExternalLoadSource`, optional) and the derivatives of K* and M* with respect to the variables it uses (columns of
  x). The constructor checks the structure only; `designStamp()` is FNV-1a (`cache/CacheHasher`) over the zone, the
  plugin's identity, the labels and the bits of every matrix: the matrices are never cached, their hash stands for them.
- `ExternalLoadSource`: the plug-in point of the condensed loads, called with the loads of the zone's interior DOFs for
  the vectors of a load basis (interior DOFs × vectors) and giving f* (labels × vectors), and their derivatives. It may
  be called from an analysis' worker thread.
- `ExternalZone`: the zone as an analysis resolved it (its replaced elements, contour DOFs and interior DOFs with their
  g-set DOFs, in increasing grid ID then component; whether it holds constraints).
- `ExternalUnitChecks`: labels and sizes (exactly the contour DOFs, any order), symmetry of K*, M* and the derivatives
  (`EXTERNAL_UNIT_SYMMETRY_TOLERANCE`), the six rigid-body motions of the contour about its centroid in the
  displacement systems of the points unless the zone holds constraints (`EXTERNAL_UNIT_RIGID_BODY_TOLERANCE`, the
  stiffness the motion meets against the largest term of K*), semi-definiteness of K* and M* (`testSemiDefinite`,
  `EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE`). Every failure an error diagnostic naming the check and the zone. The three
  tolerances (1e-6) are calibrated on the condensation of a 9³ lattice of bars (the header gives the figures).
- `testSemiDefinite` (`SemiDefiniteness`): LDLᵀ with complete (diagonal) pivoting, stopping on the largest remaining
  diagonal below the tolerance and judging the remainder (Higham §10.3); it gives the rank and the smallest pivot.
- `ExternalUnitAssembly`: the unit placed on the model: its matrices in the contour order, symmetrised; the scatter of
  K* and M* into a node-block matrix (the contour nodes must be one group of the node graph); the contour values of a
  g-set vector; the resultant of a contour vector about the basic origin.
- `ReplacedElementFamily`, `ReplacedElements`: element families that leave the zone's elements out (no stiffness,
  mass, differential stiffness, thermal load or result row for them; their pressure and line loads pass through: those
  at the interior points are the zone loads given to the plugin, those at the contour stay loads of the analysis). An
  element set on these families gives every assembler, recovery and sensitivity the model without the zones, with no
  change of theirs.

### Design choices

- **The zone's elements leave the analysis through their families** (`ReplacedElements`), not through a filter in each
  assembler: the batches stay those of the snapshot, so that the outputs, the sensitivities and the recovery need no
  change; the interior DOFs are held as permanent constraints (nothing acts on them any more) and left out of the
  outputs.
- **The applied loads of the contour points hold f*** (the interior loads carried to the contour by the plugin): the
  SinglePointConstr forces are then K·u − P of the system solved; the GPFORCE row of the unit (named `UNIT`, the zone's ID) is
  −K*·u_a, and the resultant of K*·u_a (zero for a free unit; the reactions of the constraints the zone holds
  otherwise) joins the SinglePointConstr force resultant, so that the OLOAD and SinglePointConstr resultants balance.
- **Refused, naming the zone**: a rigid element, an MultiPointConstr equation or a fastener inside the zone (their constraint rows
  would stay in the analysis while their stiffness is the plugin's), a FreeBodySupport point inside, two units sharing an element
  or an interior point, a thermal load in an analysed subcase (the zone's temperatures are not given to the plugin), a
  zone, section or cut request that holds a replaced element, M* missing with inertial loads or inertia relief (a
  warning for Parameter WeightReferenceNode), condensed loads missing when the zone is loaded inside.

## Craig-Bampton

A unit keeps its boundary DOFs and N fixed-interface modes, `u_o = Ψ·u_a + Φ·q`. In the order an eigenvalue
analysis uses them:

- `condense/UnitSystem` with `UnitModel::keepsMass`: the unit's mass on its equations (relations eliminated, spiders
  transformed: `transformSpiders`), `interiorSymbolic` (the interior's own symbolic factor), and
  `differentialStiffness` from a preload (SOL 105). `linalg::SymmetricCsc::leadingBlock` cuts `K_oo` and `M_oo`.
- `condense/CraigBampton`: `computeModes` (the block Lanczos solver on `(K_oo, M_oo)`, a count or a cut-off:
  `ModeRequest`), `projectOnBoundary` (`Ψ̃ᵀ·A·Ψ̃`: `M_aa`), `reduceModes` (`K_qq`, `M_aq`, `M_qq`: `ModalReduction`),
  `project` (`T_CBᵀ·A·T_CB`: the reduced differential stiffness), `recover`; `Ψ` is formed by blocks of
  `CONSTRAINT_MODE_BLOCK` constraint modes through `StaticCondensation::recover` and `condenseLoads`, never whole (it
  would not be light). `CB_CUTOFF_FACTOR` = 20, calibrated here.
- `cache/`: `CacheKey` with the mode request (`modeCount`, `MODE_CUTOFF_REQUEST` and `modeCutoff`); a Craig-Bampton
  unit's boundary part (`CachedReduction` with `boundaryMass`), its modal part (`CachedReduction::modal`) and its modes
  (`CachedRecoveryData`); `CacheFile` format 2.
- `interface/InterfaceProblem`: modal points (`UnitPointKind::ModalCoordinate`, six coordinates at most each) per unit
  with modes, `K_qq` in the stiffness, a second matrix on the same pattern (`InterfaceSecondMatrix`: the Craig-Bampton
  masses or the projected differential stiffness, with the interface elements'), `symbolic()`, `matrix()`,
  `secondMatrix()` for the pencils of `linalg`; `InterfaceInputs::factorises = false` skips its own factorisation.
- `scheduling/UnitFootprint`: with `keepsMass`, the unit's mass, `M_aa` and the constraint-mode block.
