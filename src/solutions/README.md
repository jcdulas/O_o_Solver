# `solutions`

The drivers of SOL 101, 103, 105 and 144, on the full path and (SOL 101, 103, 105) on the condensed path.

This is the top layer: it may use `base`, `linalg`, `input`, `elements`, `assembly`, `reduction`, `aero`, `output`.

Folders: `common/`, `sol101/`, `sol103/`, `sol105/`, `sol144/`, `superelements/`. Where to start reading:
`sol101/StaticSolution`, then the sections below in order.

## The static solution

- `sol101/StaticSolution`: SOL 101 on the full model, from a `ModelSnapshot` to `output::ResultArrays`. Its class
  comment lists the steps; `src/assembly/README.md` ("How a static solve uses them") the assembly pieces it calls.
  Subcases are grouped by constraint set (same MultiPointConstr set and same constrained DOFs, EnforcedMotionConstr included): one assembly and one
  factorisation per group, all the subcases of a group solved at once. A deck without SUBCASE is solved as subcase 1
  (O_o_Reader gives its selections as the global section, subcase 0). A subcase with TEMPERATURE(LoadCombination) or AxialDeformationLoad
  gathers its initial strains (`assembly::TemperatureField`, with TEMPERATURE(INITIAL) when given and the element
  deformations of its AxialDeformationLoad; `sol101/StaticSubcases`: `initialStrainsOf`, one field per `InitialStrainKey`), adds the
  elements' thermal and deformation loads to Pg (so that the SinglePointConstr forces balance them) and hands them to the recovery
  (σ = E (ε − α ΔT − D/L)). After the solve, each subcase gives its element stresses, strains and forces
  (`output::ElementResultBuilder`), and the ply tables of the laminates.
- `sol101/StaticSubcases` (SUBCASE and SUBCOM rows, the SUBSEQ coefficients from subcasesTable; SYM, SYMCOM, REPCASE
  refused), `sol101/StaticLoadCases` (the compact load matrices of the cases of one constraint set per family, the
  elementary load sets of the LoadCombination combinations from loadsTable's source_sid and scale, the basis of each family, the
  basis vectors made on demand). The PARAMs read in addition: InertiaRelief (inertia relief), WeightReferenceNode (grid point weight
  generator, and the reference point of the resultants). Element results are recovered case by case from the combined
  displacements (see output/arrays/ResultCombination.hpp): the non-linear columns stay exact without a per-family rule,
  and for more than a few tens of basis vectors a recovery costs no more than a combination.
- With Parameter DofSetPrint >= 0 the static solution keeps the UserDofSet table of each constraint set (`StaticResults::usetTables`: the sets G, M, S, F, R, L;
  DofSetPrintSelection is not read).
- `common/FsetSolver`: the ordering, symbolic analysis, l-set numbering (the f set without the r set of FreeBodySupport),
  conversion and positive definite factorisation of Knn (MaxDiagonalRatio, SingularityBailOut), and the solves; reusable by every sequence
  that factorises Kff.
- `common/RunCatalog`: the support catalog of a static run (the element registry, the constraint, load and rigid
  cards the assembly applies), and the cards a static run ignores with a warning (damping, frequency-dependent
  properties, thermal-only materials, plot elements, eigenvalue methods; `checkStaticSupport`). An unsupported card
  stops the run before any computation (`UnsupportedCardError`); the problems found by the element check, the
  constraint and load builders stop it with every diagnostic listed (`SolverError`).
- `common/ProgressMonitor`: the stages ("connectivity", "assembly", "factorisation", "solve", "recovery") polled by
  the calling thread, which may report the progress and cancel the analysis.

The PARAMs read (through the typed Parameter table, defaults when absent): AutoSingularityConstr, PivotZeroThreshold, PrintSingularities (singularities),
MaxDiagonalRatio, SingularityBailOut (factorisation), CoupledMass, WeightToMass (mass, for GravityLoad), DofSetPrint (the UserDofSet table). The NonStructMass set of a subcase
(case section NonStructMass) is not in O_o_Reader's subcasesTable: the NonStructMass cards are left out of the mass of a run with a
warning ("nsm-not-selected").

## Pre-flight check and mechanism shapes

- `sol101/StaticPreflight`: the steps of the static solution before any computation (support check, element check,
  subcases and output plans, output volume), every problem kept, and the estimates of the full path (ordering and
  symbolic analysis of the unconstrained stiffness pattern: factor entries, flops, memory). The caller may complete it
  with what the core does not see (cards outside the snapshot, units, parameters, element statuses).
- `common/MechanismShapes`: the zero-energy shapes of a stiffness that cannot be factorised (a failed ratio test or a
  zero pivot): shifted subspace inverse iteration on the factor of K + 1e-14·diag(K) kept despite its failed pivots,
  then a Rayleigh-Ritz projection that keeps the vectors of energy ratio below 1/MaxDiagonalRatio. The static solution attaches
  them to its SingularityError (`mechanisms()`), from which the caller can make a result table.

## The eigenvalue solutions

- `common/AnalysisModel` and `common/ConstrainedSystem`: the steps of every solution before its factorisation: the
  support check of the analysis (`RunCatalog::checkSupport` with `AnalysisKind`: SOL 103 reads LanczosEigenMethod and RealEigenMethod, SOL 105
  LanczosEigenMethod and BucklingEigenMethod), the element check, the DOF map, the connectivity; per constraint set Gm, the sets, the pattern, Kgg,
  Knn, AutoSingularityConstr, and `reduce()` for the other g-set matrices on the pattern (the mass, the differential stiffness).
  The static solution uses them too, with the same bits: it adds the checks of its subcases and output plans and its
  output volume estimate to the model (`AnalysisModelOptions`), the r set of FreeBodySupport or the supports of Parameter
  InertiaRelief = −2 and the element matrices kept by the connectivity pass to the constraint set
  (`ConstrainedSystemOptions`). `common/RunChecks` (`stopOnErrors`, `reportSupport`) is shared by all.
- `common/FactorisedSet`: a constraint set of a static run kept with its factor (`StaticOptions::keepsSystems`,
  `StaticResults::systems`): the DOF map, Gm, the solved sets, Kgg and the l-set factor, shared with the parts of the
  run, budgeted until the results are dropped. The static sensitivities of a design point solve on it
  (`design/sensitivity/SensitivitySystem`): `DesignProblem::evaluateStatics` asks for it when gradients are wanted,
  and `gradients()` then makes no factorisation.
- `common/FsetOrdering`: the ordering, the symbolic factor and the equation numbering of a DOF set, shared by
  `FsetSolver` (the static factorisation) and the pencils of the eigenvalue solutions (one symbolic factor for K and
  for every shift).
- `common/StructureCache`: what depends only on the node graph and the solved sets kept between the analyses of one
  model (the caller holds one, its `DesignProblem`s use it): the sparse pattern (keyed by the graph and the node DOF
  counts), the `FsetOrdering` and the `SolvePlan` of its first positive definite factorisation (keyed besides by the
  elimination flags of every DOF, the grid positions and the ordering method), all compared exactly, so that a new
  design point or a run again that changes no topology is factorised numerically only with the bits of a fresh
  analysis. Each kept object in a budget of its own (`base::KeptObject`), its bytes reserved in the budget of each
  analysis that takes it (`StaticOptions::structures`, `ModesOptions`, `BucklingOptions`; the counts and bytes in
  `StaticResults::structureReuse`). It also keeps the panels' memory of the factors freed (`FactorStorageSlot`): an
  `FsetSolver` takes the storage of its shape and gives its own back when it goes, so that a design loop's points
  factorise in memory already mapped. `common/AnalysisTimings`: the parts of the setup and the factorisation.
- `common/EigenMethod`: the method of a subcase from the method tables (LanczosEigenMethod; RealEigenMethod with METHOD = LAN, its other
  methods refused naming them; BucklingEigenMethod with its inverse power method replaced by the Lanczos solver over its range) and
  the `linalg::EigenRequest` from the LanczosEigenMethod's V1, V2 and ND (which of them are given decides between a range and a mode
  count; frequencies converted to λ = sign(f)·(2πf)²).
- `common/EigenvectorScaling`: NORM = MASS (sign: the largest component positive), MAX, GeomPoint (falling back to MAX
  with a warning).
- `common/ModeRecovery`: the tables of one mode (eigenvectors, SinglePointConstr and MultiPointConstr forces from the residual
  Kgg·φ − λ·Mgg·φ, or Kgg·φ + λ·K_d·φ for a buckling shape, element stresses, strains, forces, ESE), stamped with the
  analysis, the mode number (`step`) and the eigenvalue.
- `sol103/ModesSolution` (its class comment lists the steps): the subcases grouped by constraints and method, one
  block Lanczos solve per group (`linalg::SparsePencil`, `BlockLanczos`), the eigenvalue table (λ, radians, cycles,
  generalised mass s², generalised stiffness s²·λ), MEFFMASS (participation factors, effective masses and fractions
  about the Node= point), Parameter WeightReferenceNode, the output volume checked before the recovery. FreeBodySupport is not separated as an r
  set (a warning: the rigid-body modes come from the free stiffness).
- `sol105/BucklingSolution`: the static solution of the preload subcases (`StaticOptions::keptDisplacements` keeps
  their g-set displacements; `StaticOptions::analysis` makes its support check read the method cards), K of the
  buckling subcase's constraints, K_d from `assembly::DifferentialStiffnessAssembler` on K's pattern reduced to the n
  set, the solve on `linalg::BucklingPencil`, the eigenvalue table (generalised stiffness s², "generalised mass"
  s²/λ), the tables of the shapes appended to the static result arrays. The preload of a buckling subcase: the column
  `staticSubcaseSelection` of subcasesTable when O_o_Reader gives it; else the subcase itself when it has a LoadCombination or
  TEMPERATURE(LoadCombination); else the static subcase just before it.

The PARAMs read in addition: CoupledMass and WeightToMass for the mass of the modes, WeightReferenceNode for both.

## The condensation plan

- `common/CondensationPlanner`: what `reduction::CondensationPlan` needs from a static solution, without assembling:
  the analysis model, the subcases (a DOF constrained in every subcase leaves the units, a DOF constrained in some
  subcases only puts its point on its unit's boundary; an MultiPointConstr set some subcases do not select stays in the connection
  system), Gm of the rigid elements, the fasteners and the MultiPointConstr set of every subcase, the node graph with the m-set
  couplings, the FreeBodySupport points; then `plan()` runs the connection system, the cluster absorption, the boundaries and
  the plan (src/reduction/README.md).
- `common/GlueUnitCheck`: the glued points whose equations join units (their face's corners in another unit), handed
  by `plan()` to the plan as interface relations; the glue of the BCONTACT every subcase selects is in the planner's
  Gm, one relation per glued point (`reduction::glueRelationKey`).

## The condensed SOL 101

- `common/AnalysisPath`: `Full` or `Condensed`, in `StaticResults::path`.
- `common/CommonConstraints`: what every subcase shares, as the planner and the condensed path both need it: the MultiPointConstr
  cards common to all subcases, the glued contact of every subcase, the DOFs held in every subcase.
- `sol101/StaticLoadCases`: its family descriptions (`familyDescription`: cases, load sets, the DOF of each compact
  row, the element and face of a distributed row) and basis vectors (`familyVector`), kept for the condensed path
  (`Inputs::keepsFamilies`).
- `sol101/CondensedStatics` (its class comment lists the steps): with `StaticOptions::condensation`, the units of the
  plan condensed once for every subcase (`reduction::PlannedUnits`, `UnitSystem`, `UnitLoadBasis` on the rows that
  load each unit, `StaticCondensation`); per constraint set the interface problem, inertia relief from the units'
  rigid-body mass properties and inertial patterns, the boundary basis and the recovery of the units asked for
  (`StaticOptions::recoveredUnits`, default: those an output request reaches), each case handed to `StaticSolution`'s
  outputs and checks as g-set vectors. At scale: the load vectors of the patterns made from the compact load matrix's
  columns where the family allows it (no pass over the g set per vector), the points the outputs may take listed once,
  a unit no case of a batch loads left out of the case loop, and the seconds of each phase in the record
  (`CondensedRunRecord`, `CondensedSetRecord`). The outputs of a unit not recovered are left out and listed
  ("output-not-recovered"); the checks (OLOAD and SinglePointConstr force resultants, EPSILON, external work) and the grid point
  forces at its boundary points (rows "UNIT", `K*·u_a − f*·α`) need no recovery. Errors as on the full path (a cycle of
  relations, a DOF dependent twice); Parameter InertiaRelief = −2 is refused on the condensed path (`inertia-relief-condensed`).
- The condensation cache: with `StaticOptions::cache`, each unit's key (`reduction::makeUnitKey`: its stamps,
  `StaticOptions::designStamps`, the `MATRIX_PARAMETERS` of the model, the version and kernel, the checksum of its
  patterns and its input stamp, no equation built) is looked up after its local DOFs and patterns are made
  (`reduction::unitLocalModelDofs`); a hit takes `K*`, `f*`, the inertial patterns, the rigid-body mass and the pattern
  work from the cache and factorises nothing (a unit recovered factorises its interior for the recovery, `K*` checked
  to the bit), and a hit the analysis does not recover builds no equation (`UnitSystem` from the `UnitBoundaryData` the
  `StructureCache` keeps, `StaticOptions::structures`, or made from its stiffness alone); a hit whose equations are
  built compares them with its entry's checksum (`StaticOptions::verifiesCacheHits` builds every hit's); the stores are
  made off the critical path by the session's writer and finished before the report; a miss condenses and stores. Each
  unit record says what happened (`isCondensedNow`, `cacheOutcome`, `cacheCause`, `cacheNote`). The results are the
  same bits either way. With an MultiPointConstrUnion and no `source_sid` in multiPointConstrTable, `CondensationPlanner` matches the cards'
  equations to the selected set (`reduction::MpcSetAlias`).
- The memory budget: `CondensedStatics` chooses each unit's load basis before its equations, makes the memory plan
  (`reduction::CondensedMemoryPlan` from the units' and the interface's footprints, `Inputs::resultBytes` from
  `StaticSolution`: the result tables and the SUBCOM vectors) and refuses a budget that cannot hold it before any unit
  is condensed; then condenses the units in the waves of `reduction::UnitScheduler` (several on one thread each, a large
  one or the only one the budget admits on every thread its factorisation's memory allows) and keeps or releases each
  interior factor (`reduction::FactorRetention`); the recoveries run one unit after the other, a released factor
  computed again and released once the unit's basis solutions are made; the case batches are sized on the budget left
  beside the results to come. `StaticSolution` keeps no element stiffness on the condensed path (no Kgg is assembled).
  The report: `CondensedRunRecord::memory`, and per unit and constraint set their waves, threads, factor decisions and
  batches. The memory mode (`StaticOptions::memoryMode`, `CondensedStatics::Inputs::memoryMode`) decides whether the
  recovered units are recovered all at once or in sequence (waves of consecutive units per batch of cases, each unit's
  recovery released after its part of the batch) and which units release their equations after their condensation
  (built again from the snapshot at their recovery); the case vectors of a batch are filled by the interface and by
  each unit on its own DOFs in the order of the sums of a case (no fill of the g set per case). Every mode gives the
  same bits.
- Staged runs for the caller: `StaticOptions::condensedStage` stops a condensed run after the memory plan
  (`CondensedStage::MemoryPlan`: the estimate of the pre-flight check, nothing condensed, no budget or output limit
  enforced) or after the condensation of the units (`CondensedStage::Condensation`: each unit's `ReducedUnit` and the
  coefficients of the cases on its patterns in `StaticResults::reducedUnits`, the units stored in the cache, nothing
  recovered; its memory plan counts the condensation only; with `condensesWithMass` and `unitModes` their M* and
  Craig-Bampton matrices) or after the interfaces (`CondensedStage::Interface`: `StaticResults::interfaces`);
  `StaticOptions::givenBoundaries` replaces the interface's solves by the caller's values. `common/PathChoice`: the
  automatic choice of the path: a static model of at most `FULL_PATH_MAX_DOFS` (measured, provisional: 100 000) whose
  full factorisation is estimated at most half of the budget takes the full path, a larger one the condensed path
  (Parameter InertiaRelief = −2 excepted); the other sequences take the full path, the reason saying so.

## The condensed SOL 103 and SOL 105

Craig-Bampton (the reduction: src/reduction/README.md):

- `sol101/CondensedStatics` serves the eigenvalue solutions too: with `Inputs::isCraigBampton` the units keep their
  mass (`UnitModel::keepsMass`) and their interior factors after the condensation stage; `reduceModes` gives each unit
  its fixed-interface modes and modal part (`reduction::CraigBampton`, from the cache or computed, the modes computed
  again to the same bits when the cache lost them); `chooseModeRequest` the plan's count or the cut-off
  (`CB_CUTOFF_FACTOR` times the highest frequency requested: V2, or the ND-th root of the statically condensed model);
  `solveEigenGroup` the reduced eigenproblem of a constraint set on the interface (modal coordinates, the mass or the
  differential stiffness from the preload as second matrix); `recoverModes` the shapes on the g set (the interface and
  the recovered units' interiors, `u_o = Ψ·u_a + Φ·q`). The record: `CraigBamptonRecord` and per unit its modes and
  where they came from.
- `sol103/ModesSolution` with `ModesOptions::condensation`: one condensed solve per group, the modes normalised on the
  components of the full path's analysis set the condensed path knows (`common/EigenvectorScaling`:
  `scaleEigenvectorsOnSet`), `common/ModeRecovery` without a constrained system (no SinglePointConstr or MultiPointConstr forces); MEFFMASS,
  WeightReferenceNode and the SinglePointConstr and MultiPointConstr forces are not produced there (an information diagnostic).
- `sol105/BucklingSolution` with `BucklingOptions::condensation`: the preload on the condensed path with every unit
  recovered, each unit's differential stiffness from it projected on its Craig-Bampton basis (never cached), the cut-
  off on the 10th vibration frequency of the statically condensed model (`BUCKLING_REFERENCE_MODE_COUNT`), every
  condensed unit listed first in a warning ("local-buckling-not-checked", `BucklingResults::uncheckedUnits`).
- `common/PathChoice`: SOL 103 follows the static rule, the reason saying the modes are approximate with the cut-off;
  SOL 105 stays on the full path while its full factorisation fits half of the budget, whatever its size.
- The run report states the Craig-Bampton request.

## The static aeroelastic trim

`sol144/TrimSolution`: the structure of each constraint set (AnalysisModel, ConstrainedSystem, FsetSolver on the l set,
the mass) coupled with the doublet lattice through the splines in the space of the boxes: one solve per box gives the
slopes E = S·K_ll⁻¹·T, then per subcase the pressures of each trim variable on the rigid, restrained and free
structure (dense LU of D − AEQR·q·E), the derivative table (rigid unsplined and splined, elastic and inertial,
restrained and unrestrained), the trim (aero::TrimSystem on the equilibrium of the supported motions), the hinge
moments and the trimmed displacements (u_r = 0). The rigid-body modes of the supports come from the geometry, not
from the stiffness.

## External units on the full path

`StaticOptions::externalUnits` replaces the elements of zones by the matrices of plugins (`reduction/external`).
`sol101/ExternalUnitPath` drives it, with three additive hooks of the common parts:
`AnalysisModelOptions::replacedElements` (the zones resolved on the plain element set, then the element set made again
on `reduction::ReplacedElements`, before the connectivity pass), `ConstrainedSystemOptions::heldDofs`, `extraCouplings`
and `addStiffness` (the interior DOFs held as permanent constraints, the contour nodes of each unit one group of the
node graph, K* added to Kgg before the m set is eliminated). In `StaticSolution::run`: M* added to Mgg (a unit without
M* refused for inertial loads and inertia relief before any factorisation), the loads of each zone's interior DOFs per
basis vector given to the unit's load source and replaced by its f* on the contour (`condenseLoads`,
`applyToBasisVector`; the written-precision tolerance's exact cases too), the outputs of the interior points and of the replaced
elements left out of the plans (ESE dropped, with the warning `external-unit-strain-energy`), per case u_a, f* and
K*·u_a − f* recorded (`StaticResults::externalUnits`) with a GPFORCE row `UNIT` per contour point (−K*·u_a) and the resultant of
K*·u_a in the SinglePointConstr force resultant. The condensed path refuses external units; `checkExternalUnits` runs the
resolution and the checks without an analysis.

## Superelements written in the deck

`superelements/` runs SESTATIC (SOL 101) and SEMODES (SOL 103) decks with BEGIN SuperelementProcessing parts and external
superelements of the external matrix file type, one level upstream of the residual structure:

- `PartReduction`: a part reduced on its own model as one condensation unit (`reduction::UnitSystem`, its boundary the
  SuperelementConnection points, its q set left out of the unit nodes): `StaticCondensation` for K* and f* per subcase, and in a
  modal analysis the Guyan mass or the Craig-Bampton modal part (`CraigBampton`, the method of the part's subcases,
  the modes normalised by its LanczosEigenMethod); `recoverStatic` and `recoverModes` give its g set from the boundary values and
  generalised coordinates.
- `PlacedSuperelement`: the reduced matrices and loads of a superelement on the residual's DOFs (its boundary points
  are residual points; the same displacement system on both sides), added to Kgg, Mgg and the load vectors.
- `SuperelementSolution`: the residual structure solved as the interface problem through `common/ConstrainedSystem`
  (`extraCouplings`, `addStiffness`, `heldDofs`, and `uncheckedDofs` so that AutoSingularityConstr leaves the q set free), then each
  part recovered and its tables made on its own model in the subcases its SuperelementProcessing selects (`SuperelementOutput`).

## The trim on the condensed path

`TrimSolution` reads its structure through a `TrimStructure` (`FullStructure`: the l-set factor; `CondensedStructure`:
`CondensedStatics::prepareLoadGroup`, the interface problem of a constraint set with its supports held, and
`solveLoads`, loads at the interface points plus rigid-body accelerations by the units' condensed inertial patterns).
The plan keeps the spline nodes on the boundaries (`trimRetainedPoints`).

## Robustness

- `common/ProgressMonitor::setStageListener`: called on the analysis thread when a stage begins, so that an analysis
  can be stopped at a chosen stage without depending on the timing of a poll (every stage of statics, modes and
  buckling, full and condensed, gives the budget back, leaves the cache whole, and the next run gives the same bits).
- `common/RunChecks::reportKeptPivots`: a factor kept with failed pivots (Parameter SingularityBailOut -1) gives a warning
  `max-diagonal-ratio-exceeded` naming the DOFs (the full path's f set and the condensed path's interface).
- `common/EigenMethod`: a method range the Lanczos solver cannot take (a bound not finite, V1 not below V2, V2 not
  above 0 when the roots are sought from the lowest) is an error diagnostic `method-range` naming the card's field.
- `sol101/StaticSolution`: a SingularityError of the condensed path has no zero-energy shapes (they are computed on the
  f set of the full stiffness, never assembled there); its message says to run the full path to have them.
- Damaged tables: a damaged column of any input table (wrong type, nulls, out-of-range values) ends in an error of the
  SolverError hierarchy or in a result, never in a crash.
