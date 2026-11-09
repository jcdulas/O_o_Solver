# `assembly`

From elements to systems: DOF sets, constraints, rigid elements, singularities, assembly, loads.

It may use `base`, `linalg`, `input`, `elements`.

Folders: `dofs/`, `constraints/`, `singularities/`, `assemble/`, `loads/`, `relief/`. Where to start reading: the
folders in that order, which is the order in which the pieces depend on one another.

## The g set and its reduction

- `dofs/DofMap`: the g set, six DOFs per grid point in its CD system and one per scalar point, numbered node-major as
  linalg's node blocks (grid row r, component c → 6·r + c − 1; scalar points after the grids). The CD-to-basic
  rotations and the positions come from the snapshot.
- `constraints/SpcSet`: the single-point constraints of a subcase — sg from Node PS, sb from the SinglePointConstr/SinglePointConstrList rows of the
  selected SinglePointConstr set and the EnforcedMotionConstr rows of the selected LoadCombination set (singlePointConstrTable of O_o_Reader, SinglePointConstrUnion expanded), and the
  nonzero enforced values (EnforcedMotionConstr overrides the SinglePointConstr value of the same DOF).
- `constraints/MpcEquations`: the MultiPointConstr equations of the selected MultiPointConstr set (multiPointConstrTable, MultiPointConstrUnion expanded), first term
  dependent.
- `constraints/RigidElementEquations`: the dependent DOFs of one rigid element as combinations of its other DOFs:
  RigidBodyMulti, RigidBodyStar, RigidBar by the rigid transfer from six independent DOFs, RigidRod by the extension equation, RigidInterpolation by the
  weighted least-squares fit (rotational weights scaled by Lc², REFC or UM dependent). Other rigid cards are refused.
- `constraints/FastenerConstraints`: the six rows of each Fastener or Weld GA or GB that is a grid point of the model,
  dependent on the grid points of the element its side pierces by the interpolation of
  `elements/fasteners/PiercingInterpolation` (the fastener elements condense their auxiliary points themselves; these
  rows recover a given GA or GB and pass what is attached to it to the pierced element; a point named by two fasteners
  is constrained once). Handed to RigidElimination as its last argument, built
  from `snapshot.fastenerPiercings()` when the snapshot has the table.
- `constraints/GlueConstraints`: the glue equations of glued contact, the node-to-surface tie of the open-source
  solver CalculiX (G. Dhondt, CalculiX CrunchiX User's Manual 2.21, section 7.132 *TIE). The two surfaces come from
  O_o_Reader's contactNodePairs taken with the reader's own rule (rule="components"): the grid points of each
  secondary surface near its primary surface and the faces of that surface, nothing else of the rows.
  `constraints/GlueSearch` projects every candidate point on the primary faces near it; a point is tied to the face
  whose nearest point is nearest when its orthogonal distance is within the position tolerance (the pair's ErrorHandling, else
  the default) and its projection within the face or no farther outside, parallel to it, than the default tolerance,
  2.5 % of the typical element size (here the mean edge length of the primary faces). The tie is
  u_s = R_sᵀ·Σ N_i·R_i·u_i on the translations only, for solids and shells alike as in CalculiX (shells there are
  expanded into solids, whose connection to 1D and 2D elements is hinged): shell rotations stay free, IGLUE 3 and 4
  are warned. Grid points are never moved (CalculiX's ADJUST=NO; its default, moving them onto the face, is not done:
  the snapshot geometry is immutable), a gap kept is warned. Exact elimination only (no penalty); the weights are those
  at the tied point within the face (`constraints/FaceShapeFunctions`: linear or quadratic, quadrilateral or
  triangular faces, absent mid-side points; `clampToFace`). `constraints/GluedContactPairs` selects the pairs of the
  subcase's BCONTACT, refuses touching contact (IGLUE 0) and a model without the columns the glue reads
  (ContractError). A component an SinglePointConstr holds or another relation makes dependent keeps it (never a glue through an
  SinglePointConstr'd DOF); a point tied by an earlier pair keeps that tie, an equation closing a loop of glue equations is left
  out; each case is reported per pair. Handed to RigidElimination after the MPCs.
- `constraints/RigidElimination`: Gm, u_m = Gm·u_n, from all of them, chains resolved in dependency order (a cycle and a
  DOF dependent twice are diagnostics); `expandToG`, `reduceToN` (loads by virtual work) and `computeMpcForces`
  (MPCFORCES: q_m = (K·u − P)_m, q_n = −Gmᵀ·q_m).
- `dofs/DofSets`: the exclusive set of every DOF (m, sb, sg, AutoSingularityConstr, o, r, b) and the combined sets under their
  usual names (g, n, s, f, a, l...), with the overlap rules; `withAutoSpc` after the singularity processor;
  `eliminationFlags` for linalg::DofSelection. `dofs/Uset` prints the UserDofSet table.
- `singularities/GridPointSingularities`: the grid point singularity processor (AutoSingularityConstr, PivotZeroThreshold, PrintSingularities) on the 3 × 3
  diagonal blocks of Knn handed over through `NodeDiagonalBlocks`, and the table of the singularities it found.
  `singularities/SmallSymmetricEigen`: cyclic Jacobi for the 3 × 3 blocks and the 6 × 6 RigidInterpolation normal matrix.

## Assembly and loads

- `assemble/ElementSet`: the element batches with a family in the registry (the others are refused by the run's
  support catalog), the element check of every family (with a pool, a long batch of a family is checked row by row by
  ranges on its threads: `ElementFamily::checkRows`).
- `assemble/ElementConnectivity`: the nodes each element couples, from the DOFs of its stiffness (the mass for a mass
  element): the groups of the node graph. It takes one pass over the element matrices because the connectivity
  columns O_o_Reader gives do not cover the springs, Bushing and Bar2. The stiffness matrices it computes are kept for
  the assembly of Kgg in an `assemble/ElementMatrixStore` within a share of the budget (on a 1 M DOF plate model,
  setup and assembly went from 1.57 s to 1.2 s), dropped beyond it (the assembly then computes them again).
- `assemble/StiffnessAssembler`, `assemble/MassAssembler`: Kgg and Mgg through linalg's `SparseAssembler`
  (ElementDof → NodeDof); the mass lumped or coupled (Parameter CoupledMass), times Parameter WeightToMass once assembled, with the
  non-structural mass of an NonStructMass set (`assemble/NonStructuralMassShares`: NonStructMass and NonStructMass1 per unit length or area;
  NonStructMassLumped and NonStructMassLumped1 refused). `assemble/ParamLookup` reads the PARAMs with the bad-param rule.
- `constraints/MsetCouplings`: the node couplings the m-set elimination adds (each element group touching a
  dependent point, extended with the points of its Gm rows; each dependent point with its Gm points), so that Kgg
  and Knn share one pattern. `constraints/MsetElimination`: Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm on the
  block structure, serial.
- `singularities/NodeDiagonalBlockSet`: the 3 × 3 blocks of Knn for `GridPointSingularities`.
- `loads/LoadVectorBuilder`: Pg of a load set from loadsTable (PointForceLoad, PointForceLoad1, PointForceLoad2, PointMomentLoad, PointMomentLoad1, PointMomentLoad2 in
  basic turned into CD; ScalarPointLoad; GravityLoad as Mgg·r), pressureTable (ShellPressureLoad, FacePressureLoad through the pressure facet; PointsPressureLoad on the
  face of its grid points: a triangle's P·A·n in thirds, a quadrilateral as two sets of overlapping triangles at half
  the pressure each) and lineLoadsTable (BeamDistributedLoad through the line-load facet); everything else is a diagnostic.
  `lineLoadsOf` gives the BeamDistributedLoad of a load set per batch, grouped by element, for the recovery under line loads.
  `checkThermalLoads` and `addThermalLoads` add the thermal loads of every family under the temperatures of a subcase.
- `loads/TemperatureField`: the temperatures of a TEMPERATURE(LoadCombination) set and of the TEMPERATURE(INITIAL) set from
  temperatureTable (NodeTemperatureLoad per grid point, the DefaultTemperatureLoad default, PlateTemperatureLoad and LineElemTemperatureLoad per element), and the element
  deformations of a AxialDeformationLoad set from the deform table (the initial strains of a subcase, a AxialDeformationLoad with or without
  temperatures), as the element facets read them (`elements::ElementTemperatures`); `LoadVectorBuilder::
  checkThermalLoads` refuses a AxialDeformationLoad on an element that is not a Rod2, Rod2InlineSection, Tube2, Bar2 or Beam2.

## Loads by family, the load basis, inertia relief

- `loads/LoadFamily`: concentrated, distributed, inertial, thermal, enforced. `LoadVectorBuilder::addFamilyLoads`
  makes the vector of one family of a load set; `concentratedEntries` and `distributedValues` give the compact values
  of the rank test (the loaded DOFs; the card values of the pressures and line loads, with their written precision
  when asked). `loads/LoadSetIndex` finds the rows of a set by binary search and the text columns are read once: a
  deck of thousands of load sets was quadratic without them. `checkLoads` reports every problem of a set without
  assembling it (a case of a combination is never assembled on its own).
- `loads/InertialLoads`: GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad (METHOD 1) as Mgg times a rigid acceleration field, from
  inertialLoadsTable or the GravityLoad rows of loadsTable. `addAccelerations` (and
  `LoadVectorBuilder::addInertialAccelerations`) gives the field itself, Σ factor·a of the rows of a set: the design
  sensitivities form the inertial loads of a mass derivative element by element from it.
- `loads/CompactLoadMatrix`, `loads/LoadBasis`: the smallest exact basis of a family (load sets as written, loaded DOFs,
  rank at round-off by column-pivoted Householder QR, the cases), every case reconstructed and checked against
  LOAD_BASIS_RESIDUAL_LIMIT (the written-precision tolerance: within the written precision of each value, root mean square), a case
  that fails joining the basis; the constants LOAD_RANK_TOLERANCE, LOAD_BASIS_RESIDUAL_LIMIT,
  WRITTEN_PRECISION_CHECK_CASES are justified in LoadBasis.hpp. When the cases are the smallest candidate and number
  RANK_CERTIFICATE_MIN_CASES or more, a full-rank certificate (`linalg::FullRankCertificate`, the unpivoted blocked QR
  of the scaled cases, σ_min ≥ 1/‖R⁻¹‖_F) may prove that the pivoted QR would take every case, which then does not
  run (the proof and its margins in LoadBasis.hpp); the reconstruction check runs over the cases in parallel (each case
  alone, the same bits). `loads/FieldPrecision`: half a unit of the last written digit from the written numeric precision
  O_o_Reader gives for the load cards (the finest precision of the cards of a load set and type, a LoadCombination set taking those
  of the sets it combines).
- `relief/RigidBodyModes` (D of the grid point weight generator), `relief/GridPointWeight` (Parameter WeightReferenceNode: MO, S, the
  masses and centres of gravity, I(S) with the products of inertia printed as +Σ m·x·y, the opposite sign of the
  tensor terms of MO, I(Q), Q), `relief/InertiaRelief` (FreeBodySupport, FreeBodySupportSet from dofSetsTable; Parameter InertiaRelief −1 with the
  support modes of the stiffness, −2 with the supports `chooseAutomaticSupports` picks and the rigid motion removed in
  the mass metric).

## How a static solve uses them

```
DofMap dofMap(snapshot.nodes(), snapshotIndex.points());
SpcSet spc = SpcSet::forSubcase(&snapshot.constraints(), snapshot.subcases(), dofMap, subcase, budget, sink);
MpcEquations mpc(&snapshot.constraints(), dofMap, snapshot.subcases().selectedSet(subcase, "mpc"), budget, sink);
FastenerConstraints fasteners(snapshot.hasTable(TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr,
                              dofMap, budget, sink);
GlueConstraints glue(snapshot, subcaseContact, subcase, dofMap,
                     {&snapshot.rigidElements(), &mpc, &fasteners, spc.sbDofs(), spc.sgDofs()}, budget, sink);
RigidElimination gm(&snapshot.rigidElements(), &mpc, dofMap, budget, sink, &fasteners, &glue);
DofSets sets(dofMap, {gm.mDofs(), spc.sbDofs(), spc.sgDofs(), {}, {}}, budget, sink);
//  stop if sink.hasErrors()
//  NodeGraph from ElementConnectivity + MsetCouplings; Kgg (StiffnessAssembler); Knn (MsetElimination);
//  its diagonal node blocks (NodeDiagonalBlockSet)
GridPointSingularities singularities(dofMap, sets, blocks, SingularityOptions::fromParams(params, subcase, sink),
                                     budget, sink);
DofSets solveSets = sets.withAutoSpc(singularities.autoSpcDofs());
//  DofSelection from dofMap.nodeDofCounts() and solveSets.eliminationFlags(DofSet::F); factorise Kff
//  (solutions::FsetSolver does these two)
//  loads: Pg (LoadVectorBuilder) → gm.reduceToN(Pg); enforced values: Pf −= Kfs·us (spc.enforcedDisplacements())
//  solve, then u_s = enforced values, gm.expandToG(ug)
//  gm.computeMpcForces(Kgg·ug, Pg, qMpc); SinglePointConstr forces q_s = (Kgg·ug − Pg − qMpc)_s
```

`solutions/sol101/StaticSolution.cpp` is this sequence.

## Rules of the input format this module applies

These rules are stated in the code where they are applied and tested on hand-made and analytic cases:

- EnforcedMotionConstr DOFs join the s set, and an EnforcedMotionConstr value overrides the SinglePointConstr value of the same DOF. An EnforcedMotionConstr whose DOF is not in the
  subcase's SinglePointConstr set still makes that DOF a member of the s set; whether such an EnforcedMotionConstr should rather be ignored is an
  open question.
- An SinglePointConstr on a Node PS component is accepted (the DOF stays in sg); an SinglePointConstr on a dependent DOF is an error.
- AutoSingularityConstr tests the free components only (not m, not s), so the PS components are never listed as singular.
- The singular directions are removed by constraining the components maximising |det V_C|; "aligned" when that
  determinant is at least cos 1°.
- A scalar point is singular when its diagonal is at most PivotZeroThreshold times the largest diagonal term.
- AutoSingularityConstr DOFs are printed in sb, the NEW UserDofSet union as "S".
- The RigidInterpolation rotational weights are WTi·Lc², Lc the mean distance of the independent grid points to the reference point.
- The UserDofSet table layout is this project's own.
- GravityLoad is Mgg·r with r the rigid translation by the acceleration, WeightToMass applied to Mgg.
- A blank BeamDistributedLoad P2 is taken as P1, as O_o_Reader gives it, also when X2 is given.

## Differential stiffness

`assemble/DifferentialStiffnessAssembler`: K_d of the g set under a preload, from the facet
`ElementFamily::computeDifferentialStiffness` of every batch with stiffness (the inert types add nothing), assembled on
the pattern of K in the order of the stiffness assembly; `check()` reports the element types without the facet.
The mass of an eigenvalue solution is `MassAssembler` on the pattern of K (its structural zeros kept), reduced to the n
set by `MsetElimination::reduce` as K is.
