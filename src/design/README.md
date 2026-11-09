# `design`

Optimisation: design model, DesignEquation, responses, sensitivities, optimisers, SOL 200, DesignProblem.

It may use `base`, `linalg`, `input`, `elements`, `assembly`, `reduction`, `aero`, `output`, `solutions`.

Folders: `model/`, `expressions/`, `responses/`, `sensitivity/`, `approximation/`, `optimiser/`, `sol200/`, `aids/`,
`external/`, `problem/`.

Where to start reading: the folder list above follows the order in which the pieces build on each other.

## `expressions/` — equations with exact derivatives

- `IntrinsicFunction`: the intrinsic functions a DesignEquation equation of the input format may call, exactly that list (ABS,
  ACOS, ACOSH, ASIN, ASINH, ATAN, ATAN2, ATANH, ATANH2, AVG, COS, COSH, DB, DBA, DIM, EXP, INVDB, INVDBA, LOG, LOG10,
  LOGX, MAX, MIN, MOD, PI, RSS, SIN, SINH, SQRT, SSQ, SUM, TAN, TANH), each with its value, its partial derivatives
  and its domain.
- `ExpressionProgram`: one equation as a postfix program, its invariants checked at construction.
- `ProgramTables`: the contract tables `designEquationSymbols` and `designEquationInstructions` read into programs.
- `ExpressionEvaluator`: values and exact forward-mode derivatives at a batch of points; refusals as diagnostics.

`O_o_Solver` never parses DesignEquation text: `O_o_Reader` does. The tests build the tables with small hand-written parsers of
the DesignEquation syntax (test code only).

### The expression program contract

An equation `variable-1(x1, ..., xn) = expression-1; variable-2 = expression-2; ...; variable-m = expression-m` is
given by two tables.

`designEquationSymbols`, one row per name:

| column | type | meaning |
|---|---|---|
| `equationId` | int32 | the EQID of the DesignEquation entry (0 for a text compiled on its own) |
| `kind` | dictionary<utf8> | `argument` (x1..xn) or `variable` (variable-1..variable-m) |
| `slot` | int32 | from 0: the position in the header list for an argument, the statement index for a variable |
| `name` | utf8 | the name, upper case |

Every slot from 0 to the last appears once per kind; names are unique within an equation; at least one variable.

`designEquationInstructions`, one row per instruction; the rows of one equation are contiguous and in execution order:

| column | type | meaning |
|---|---|---|
| `equationId` | int32 | as above |
| `statement` | int32 | the statement (from 0) the instruction belongs to; non-decreasing, by steps of 1 |
| `opcode` | dictionary<utf8> | one of the opcodes below |
| `operand` | int32 | `argument`: the argument slot; `variable`: an earlier statement; `call`: the number of arguments; 0 otherwise |
| `value` | float64 | the constant of `realConstant` and `integerConstant`; 0 otherwise |
| `function` | dictionary<utf8> | the name of the intrinsic function of a `call` (upper case); null otherwise |
| `position` | int32 | the column (from 1) in the equation text of the token that produced the instruction; 0 if unknown |

Opcodes (a stack machine: each pops its operands and pushes its result):

| opcode | pops | pushes |
|---|---|---|
| `realConstant` | 0 | `value`, a constant written with a `.` or an exponent (`2.`, `1.5E3`) |
| `integerConstant` | 0 | `value`, a constant written without (`2`): integer arithmetic; \|value\| ≤ 2^53 |
| `argument` | 0 | the argument of slot `operand` |
| `variable` | 0 | the value of the earlier statement `operand` |
| `negate` | a | -a (a unary minus; a unary plus produces no instruction) |
| `add`, `subtract`, `multiply`, `divide`, `power` | a, b (b pushed last) | a + b, a - b, a·b, a/b, a\*\*b |
| `call` | `operand` values (the last argument pushed last) | `function(arguments)` |

The instructions of statement s leave exactly one value on the stack: the value of variable-s; the next statement
starts on an empty stack. The value of the equation is variable-m. The producer resolves the Fortran operator
precedence (`-X**Y` is `-(X**Y)`, `X**-Y` is `X**(-Y)`, `**` groups from the right, `-X-Y` is `(-X)-Y`), resolves
every name (an argument or an earlier statement; anything else is a parse error), checks the argument count of every
function (AVG, MAX, MIN, RSS, SSQ, SUM: 1 to 96), and refuses a text that does not parse with the column of the error.
Compiling one equation text gives the two tables of that equation, with equation ID 0; a text that does not parse is
refused with an error that gives the column.

### Semantics and decisions

- Arithmetic is in double precision. Integer constants combine in integer arithmetic: `1/2 = 0`, `7/2 = 3` (truncated
  toward zero), `2**-1 = 0`; any real operand makes an operation real; every function and every variable is real, so
  `MOD(5,2)/2` is 0.5.
- `a**b`: a negative base needs a whole exponent; `0**b` with `b < 0` is a division by zero; `0**0 = 1`.
- Function definitions: ATAN2(x, y) = tan⁻¹(x/y) in the quadrant of (y, x) (Fortran's ATAN2 with the numerator first);
  ATANH2(x, y) = tanh⁻¹(x/y); DIM(x, y) = x − MIN(x, y); LOGX(x, y) = ln y / ln x; MOD(x, y) = x − y·INT(x/y) with INT
  truncating; PI(x) = π·x; DB(P, PREF) = 20·log10(P/PREF); DBA(P, PREF, F) = DB + 10·log10(Ta1) + 10·log10(Ta2) with
  the A-weighting terms Ta1, Ta2 (they agree with IEC 61672 within 0.001 dB, which the tests check);
  INVDB(DB, PREF) = 10^(DB/20 + log10 PREF); INVDBA(DBA, PREF, F) = 10^((DBA − 10·log10 Ta1 − 10·log10 Ta2)/20 +
  log10 PREF), the inverse of DBA in P (the `+ log10 PREF` term as in INVDB).
- Derivatives where a function is not differentiable: ABS takes the right derivative (+1) at 0; MAX and MIN give the
  whole derivative to the first argument, in written order, that holds the extreme; DIM at x = y gives 0; MOD gives
  (1, −INT(x/y)) between its jumps.
- Refusals: a point outside a domain, a division by zero, an overflow, an integer overflow beyond 2^53 and (when
  derivatives are asked) an infinite or undefined partial derivative with respect to a free argument (SQRT at 0)
  give NaN at that point and a `Diagnostic` (card `DesignEquation`, the EQID, field = the statement name) naming the statement,
  the column, the function or operator, the arguments and the points; the other points are computed. A partial that
  is infinite with respect to a constant (an argument bound by `constants`) is not refused: it multiplies a zero
  tangent and is never formed.
- Batches run in blocks of 256 points, instruction by instruction across a block; every point follows the same
  operations in the same order, so a batch gives the same bits as single evaluations.

## `model/` — the design model, design points and design stamps

- `DesignVariables` (DesignVar: XINIT, XLB, XUB, DELXV; DesignDiscreteValues refused, naming it), `DesignLinks` (DesignVarLink: the split into
  independent variables, the columns of x, and dependent ones, `DDVID = C0 + CMULT·Σ Ci·IDVi`), `DesignRelations`
  (DesignPropRelation1/2, DesignMatRelation1/2, DesignElemRelation1/2 and the relations given by the caller's functions: field values and their exact
  derivatives with respect to x through the chain rule), `DesignModel` (the three, the DesignEquation programs and the DesignConstants
  constants, checked at construction; `checkDesignModel` lists every problem as a diagnostic).
- `DesignTables`: the design tables of O_o_Reader read into `DesignModelParts`.
- `FieldTargets`: the designed fields bound to the columns of one snapshot ("properties/<TYPE>", "elements/<TYPE>",
  materialsTable and "materials/<TYPE>"), and the units (components) each changes.
- `DesignPoint`: x → variable values, field values and derivatives, the `input::ColumnOverlay` of the snapshot, the
  design stamps; `computeDesignStamps` (FNV-1a of the field identities and values of each unit).

### How a design point reaches the elements

The snapshot is immutable and the element families read their fields straight from the columns of the snapshot's
tables. A design point is therefore a new snapshot, made by `input::ColumnOverlay::applyTo`: each table is re-wrapped
as a new Arrow struct whose children are the base table's columns, borrowed, except the overlaid columns, replaced by
copies holding the design values (and their validity: a blank field that gets a value becomes valid). The base
snapshot stays alive through the shared tables. No element, assembly or solution code changes: the static solution
runs on the overlaid snapshot as on any other (`DesignProblem::evaluateStatics`). The cost of a point is a copy of the
overlaid columns (8 bytes per row of each designed column) and the contract checks of the re-wrapped tables, never a
copy of the model.

What the overlay does not change: the summaries O_o_Reader computes from the whole model (the `mass` of
materialsTable and connectedComponents) and the revision stamps (the design stamp stands for them in the cache
key); the tables O_o_Reader derives from cards, but for the ply stacks of a designed CompProp or CompPropGlobal
(laminatePliesTable, laminateSummary) and the section properties of a designed BarShapeProp or BeamShapeProp (sectionsTable),
which the design point overlays too (model/DerivedProperties: the ply stack by O_o_Reader's arithmetic, the sections by
its own section library, so that the point gives the same bits as the model with the design written into it); the
defaults O_o_Reader fills in when a card is read (G of a IsoMat from E and NU, Z1 and Z2 of a ShellProp from T): like a card
edit, the overlay keeps them as they were. Deriving a blank G or Z1 again from the designed E or T would need
O_o_Reader to keep the blank fields known.

### The design tables contract (as O_o_Solver reads it)

The design tables of O_o_Reader, with the columns marked "proposed", which O_o_Solver asks for:

`designVariablesTable`: `id` int32 · `label` utf8 or dictionary<utf8> · `value` float64 (XINIT) · `lower` float64 ·
`upper` float64 · `delta` float64 (DELXV, null when blank) · `ddval` int32 (proposed, optional; null when blank).

`designLinksTable`, one row per (DesignVarLink, independent variable): `dependentVariable` int32 · `constant` float64 ·
`multiplier` float64 · `independentVariable` int32 · `coefficient` float64.

`designRelationsTable`, one row per (relation, design variable or argument): `relationCard` text (DesignPropRelation1 ...
DesignElemRelation2) · `relationId` int32 · `targetType` text (`property`, `material`, `element`) · `targetCard` text ·
`targetId` int32 · `field` text (the column of the data contract that holds the field) · `designVariable` int32 (null
for a DesignConstants argument) · `coefficient` float64 (DVxREL1) · `constant` float64 (C0) · `equationId` int32 (DVxREL2) ·
proposed, optional: `fieldItem` int32 (the item of a fixed-list column such as BushProp `k`, null for a scalar),
`argumentIndex` int32 (the position of the argument in the DesignEquation header), `constantLabel` utf8 (the DesignConstants label of
a constant argument), `fieldMin`, `fieldMax` float64 (PMIN, PMAX).

`designConstantsTable` (proposed, DesignConstants): `label` utf8 · `value` float64.

The DesignEquation programs come as in `expressions/` (designEquationSymbols, designEquationInstructions). The tests build these tables by hand
from the design cards of their decks.

## `problem/` — the DesignProblem core

`DesignProblem`: a `DesignModel` bound to a shared snapshot (`FieldTargets`), giving design points, their overlaid
snapshots and the static analysis on them. The caller gathers the deck's design model and its own declarations
(variables, relations, links) into `DesignModelParts` and builds this object again after each declaration, so that a
declaration the model cannot take is refused at once. The responses, sensitivities, approximations and optimisers
below build on it.

## `responses/` and `constraints/` — the responses, constraints and objective

- `ResponseDefinitions` (DesignResp1 types, DesignResp2/DesignResp3 and the caller's equation responses, their arguments),
  `ResponseTables` (the contract tables read into `ResponseParts`), `ElementCatalog` (elements by ID and by property),
  `ResultLookup` (the result tables of an analysis indexed), `stress_strain_force/ItemCodes` (item codes → table
  columns), `weight_volume/MassProperties` (rigid-body mass and volumes element by element), `composite/StrengthRatio`
  (CSTRAT), `DesignResp1Response` (one DesignResp1 resolved and evaluated), `equation/EquationResponse` (DesignResp2 through DesignEquation,
  DesignResp3 and the caller's callables), `BatchCallable` (the batch convention of every callable: `EQUATION_BATCH_ITEMS`,
  `BatchCaller`), `DesignResponses` (the set, in dependency order; the outputs the analysis must give).
- `ConstraintDefinitions`, `ConstraintTables`, `DesignConstraints` (DesignConstr, DesignConstrUnion, DESSUB, DESGLB, DESOBJ, DesignConstrScreen
  stored for the screening; which constraint acts in which subcase; the normalised values),
  `ConstraintNormalisation` (GNORM = |bound|, or GSCAL near zero).
- `DesignProblem::evaluateStatics` adds the outputs the responses read to the requests of the analysis
  (`StaticOptions::requiredOutputs`), whatever the case section asks; `DesignProblem::evaluateResponses` evaluates the
  responses on the calling thread (the caller's callables are called from it), then the constraints and the objective.

### Rules and decisions

- A response is global (one item, subcase 0) when its type is WEIGHT, VOLUME or FRMASS, or when it is an equation by
  the card rules whose response arguments are all global. The others are evaluated in the subcases where a
  constraint on them acts (DESSUB of the subcase, or the global DESSUB it inherits, DesignConstrUnion expanded), in the subcase of
  the objective, in their own subcases when the caller gives some, and in the subcases of the equations that read
  them; a response used by none of these (a monitored response) is evaluated in every analysed subcase.
- A DesignResp1 argument of a DesignResp2 gives one argument per item, in its item order (a DesignResp1 that identifies several
  responses contributes them all); the arguments fill the DesignEquation header in the order of the card (DesignVar, DesignConstants,
  DesignResp1, DNODE, DVxRELi, DesignResp2). A deck DesignResp3 callable gets one keyword per argument kind (`desvar`, `dtable`,
  `designResp1`, `dnode`, `dvprel1` ... `dresp2`), each `(n, c)` or `(n,)` for one value, one item per subcase.
- WEIGHT is an entry of the rigid-body mass matrix MO about Parameter WeightReferenceNode (the origin otherwise), in the units of the
  deck (before the WeightToMass scaling), without the NonStructMass cards (as the analysis); VOLUME is the mass of the elements with
  every density 1 and every NonStructMass 0, the concentrated masses left out; FRMASS is the mass of the selected elements over
  their mass at the initial design (the topometry definition of the format, applied to sizing).
- The items the format lists as "von Mises or maximum shear" give the von Mises value the result tables hold; the
  maximum shear form (STRESS(MAXS)) is not offered.
- ESE item 4 (the energy density) is the element's strain energy over its volume (the density column of the strain
  energy table is empty).
- CSTRAT: the strength ratio λ with F(λσ) = 1 (`composite/StrengthRatio`), from the ply stresses and the allowables of
  the laminate on the overlaid snapshot; the bond ratio is 1/FB.
- A constraint value is given for every item and every bound in every subcase where it acts, in the normalised form
  g = (L − r)/max(|L|, GSCAL) or (r − U)/max(|U|, GSCAL), GSCAL = 0.001 by default (DesignOptimParams); the screening of DesignConstrScreen
  is applied by the approximation. A constraint of set 0 (declared by the caller without a set) acts wherever its
  response is evaluated.
- Every callable of the caller is called by batch (`BatchCaller`): blocks of `EQUATION_BATCH_ITEMS` (8 192, measured)
  in item order; a result of the wrong shape or not finite is an `OptimisationError` naming the response, the item and
  its argument values. The relations given by the caller's functions are called together, one call per callable,
  `function(values (n, k)) → (n,)`, `gradient → (n, k)`.

### Item codes

`stress_strain_force/ItemCodes.cpp` holds, for each element family, a table of item code -> column name of
O_o_Solver's own result tables, the single source of the mapping; a value is looked up by that name, so the order of
the columns of a result table does not matter. In short: rods give the axial and torsional stresses or strains, their
margins, the axial force and the torque; Bar2 the stresses at the recovery points of each end, the axial stress, the
extremes, the margins and its end forces; Beam2 the same at end A and end B (the intermediate stations are refused);
the shells the fibre distance, the in-plane components, the angle, the principal and von Mises values at each fibre (or
the membrane strains and the curvatures) and the centre forces (the corner format has the grid point first and its
corners refused); ShearPanel4 the shears, the margin and its corner and kick forces; the springs and Bushing their
components; the plies of the laminated shells their stresses or strains (the lamina is ATTB), the failure index of the
direct stresses and that of the interlaminar shear; the strain energy, its percentage and its density.

### The response and constraint tables (as O_o_Solver reads them)

`designResponsesTable` (with the proposed columns): `id` int32 · `label` text · `responseType` text (RTYPE, or
the card name of a DesignResp2/DesignResp3) · `propertyType` text (PTYPE) · `region` int32 · `attributeA` int32 (ATTA) ·
`attributeB` float64 (ATTB) · `targets` list<int32> (ATTi; empty for blank or ALL) · `isObjective` bool ·
`objectiveSense` text · proposed: `responseCard` text · `equationId` int32 · `function` text (DesignResp2 FUNC, refused)
· `group` text, `userType` text (DesignResp3).

`designResponseArgumentsTable` (proposed): `responseId` int32 · `kind` text (DesignVar ... DesignResp2) · `position` int32
· `id` int32 · `component` int32 (DNODE) · `label` utf8 (DesignConstants).

`designConstraintsTable` (with the proposed columns): `constraintSet` int32 · `responseId` int32 · `lower`
float64 · `upper` float64 (null when blank) · `subcases` list<int32> (optional restriction) · proposed:
`lowFrequency`, `highFrequency` float64 (refused when given).

`designConstraintSetsTable` (proposed, DesignConstrUnion): `setId` int32 · `memberSet` int32.
`designSubcasesTable` (proposed): `subcase` int32 (0 the global section) · `subcaseDesignConstrSelection` · `globalDesignConstrSelection` · `designObjectiveSelection` int32 ·
`objectiveSense` text.
`designScreeningTable` (proposed, DesignConstrScreen): `responseType` text · `threshold` float64 · `maxRetained` int32.

The tests build these tables by hand from the cards of their decks (responses, constraints and case section
included).

## `sensitivity/` — static sensitivities

`DesignProblem::gradients`: the gradient of every item of the responses asked for with respect to x, and of the
normalised constraints and the objective, by `StaticSensitivity`:

1. **Linearisation of the items** (`ResponseLinearisation`): each item gets its displacement weights c = ∂r/∂u_g and
   its explicit part ∂r/∂x at fixed u. DISP: the unit vector of its DOF. SPCFORCE: r = wᵀ·(K·u − P) with w the
   constrained DOF and the m-set rows of Gm that read it (`RigidElimination::computeMpcForces`), so c = K·w and
   ∂r/∂v = −wᵀ·q. Element items (STRESS, STRAIN, PointForceLoad, ESE, CSTRESS, CSTRAIN, CFAILURE, CSTRAT): `ElementItemFunction`
   recomputes the item as the analysis does (the family's recovery, the row of the place and layer, the strength
   ratio, ½·uᵀ·K_e·u), checked against the analysis' value (a derivative of another quantity is never returned), and
   differences it with respect to the element's displacements (extrapolated central differences, a large step for the
   items linear in u, a small one for those that are not: von Mises, principal stresses, margins, strength ratios) and
   to the fields. An item whose column its family declares Linear or PlaneVonMises takes its weights from the slopes
   of the recovery (`ElementItemFunction::slopes`: the family's recovery matrix, or recoveries at unit displacements;
   a von Mises chained; a strain energy K_e·u_e), exact, and the differences remain for the other items and as the
   reference (`GradientOptions::elementDerivatives = Differences`). ESE item 3 is 100·E/T, T the total energy. WEIGHT,
   VOLUME, FRMASS: `MassProperties` on the perturbed snapshots restricted to the elements the field reaches (linear in
   the fields: exact); the masses of those elements alone are computed (`rigidBodyMassOf`, `volumeOf`: the terms of
   the filtered sum, the same bits). Equations (DesignResp2, DesignResp3, the caller's): `EquationResponse::evaluateWithPartials`
   gives ∂h/∂ of every argument (DesignEquation forward mode; the callable's gradient, required); `ChainedAdjoint::chain`
   forms c_e = Σ ∂h/∂r_k·c_k and the explicit part, level by level.
2. **Pseudo-loads** (`PseudoLoads`): for every designed field (relation) moved by a variable, `FieldPerturbation`
   builds the four overlays v ± h, v ± h/2 (each an overlay of the design point's own snapshot with the relation's
   cells only, its index sharing the point's lookups: O(the relation) per variable instead of O(model)); on the
   elements the field reaches (`AffectedElements`: the property's elements, the elements whose property or card names
   the material, the element itself) `ElementDerivatives` differences the stiffness and the thermal loads (the
   family's `FieldUse` spares what cannot change; the stiffness, the mass and the explicit parts of the items come in
   closed form from the families that give them for a field of one plain cell, a scalar column or one item of a
   fixed-list column, the AnisoMat2D `stiffness` included: `stiffnessFieldDerivative`, `massFieldDerivative`,
   `recoveryFieldRates`, and for the buckling factors `differentialStiffnessFieldDerivative`; the four overlays are
   made at their first use (`LazyFieldPerturbation`: none for a field whose computations are all in closed form) and a
   wave's tasks of one element are computed together, its family asked once for all their fields' stiffness rates and
   its jobs' rates (`FieldWave::computeGroup`)); the inertial loads of a field that changes the mass (GravityLoad, AccelFieldLoad,
   RotationalForceLoad) are WeightToMass·(∂M_e/∂v)·a element by element, a the subcase's acceleration field (P = Mgg·a is linear in the
   mass and a field moves no grid point); the distributed loads of an element whose field changes them, and every load
   a shape moves, are differenced on the whole load vector. q_rs = ∂P_s/∂v_r − ∂K/∂v_r·u_s, and
   b_js = Σ_r ∂v_r/∂x_j·q_rs. The element terms of the fields are computed on the pool, a wave of fields and their
   (field, element) tasks a chunk at a time (`FieldWave`, `elementTermsOf`), and added to the pseudo-loads in field and
   element order (`PseudoLoads::stiffnessTerms`, `addStiffnessTerms`): the serial loop's chains, the same bits; the
   serial loop stays for one thread, the shapes, and the eigenvalue and trim items.
3. **Solves** per subcase on the factor of its constraint set (`SensitivitySystem`: the factor the analysis kept,
   `StaticResults::systems`, or else the reduction of the static solution made again, one factorisation per
   constraint set; inertia relief and SUBCOM refused): `MethodChoice` counts the solves of the three orders (direct:
   the variables with a pseudo-load; adjoint: the items read with the displacements, an argument of several equations
   once; chained: the plain items asked for and the equation items) and takes the fewest (the direct order, then the
   adjoint, on a tie); `DirectSensitivity`, `AdjointSensitivity` (with `combineArguments`, the argument-first order for
   the equations) or `ChainedAdjoint`. Forced methods agree to round-off (tests). On the full path the adjoint and
   chained subcases of one factor are gathered and solved together (`AdjointSensitivity::solveTogether`):
   `AdjointColumns` makes their right-hand sides from the items' entries (the values of `toEquations`) and their
   products by tiles of the g set (`PseudoLoadDots`, the bits of `PseudoLoads::dotAll`); the `vjp`'s routes use the
   same two classes (`external/VectorJacobianProducts`).

### Decisions

- **The step of the semi-analytic differences**: relative 1e-2 of the field (1e-3 absolute at zero), central
  differences at h and h/2 extrapolated (Richardson, truncation O(h⁴), none for the polynomial dependences). The
  rounding of ∂K/∂v is amplified by the cancellation of the pseudo-load (K·u ≫ P): measured, a plain central
  difference at 1e-5 lost 1.3e-5 on d(von Mises)/dE of a ShellTria3 plate, and 1e-3 extrapolated still lost 3e-3 on the
  tip displacement of a Weld joint with respect to D, which 1e-2 brings to the limit of the reference (1e-5).
  `FieldPerturbation.hpp` gives the figures.
- **The steps of the differences with respect to the displacements**: 1e-3 of the largest |u| of the element's DOFs
  of the kind, extrapolated, for the linear items; 1e-5 for the items whose two differences disagree beyond 1e-8
  (`StaticSensitivity.cpp`).
- **The factor of the analysis is reused**: `DesignProblem::evaluateStatics` with `needsDisplacements` sets
  `StaticOptions::keepsSystems`, the results keep each constraint set with its factor (`solutions::FactorisedSet`, in
  the memory budget until the evaluation is dropped), and `SensitivitySystem` solves on it: the gradients make no
  factorisation (the report counts 0; the same bits as a factor made again). A result without the kept set (an
  analysis made without the option) still gets one factorisation per constraint set.
- The items are recomputed, never read from the tables, so that each item's function is differentiated; a STRAIN
  request with FIBER (the response then reads another form than the recovery's default) is refused by the value
  check.

## `approximation/` and `optimiser/` — the approximate problem and the optimisers

The inner loop of a SOL 200 design cycle, every piece public so that an external optimiser can use any of them:

1. **Screening** (`RetainedResponses`, `ConstraintScreening`, `ExplicitConstraintGroup`, `ScreeningReport`): every
   constraint value of an evaluation is renormalised with the GSCAL of the DesignOptimParams, given its DesignConstrScreen type (the RTYPE,
   EQUA for a DesignResp2, DesignResp3) and region (the REGION of its response; without one, the response for a grid response or
   an equation, the property of the item's element for an element response; none for WEIGHT, VOLUME, FRMASS: NSTR not
   used), then retained when above TRS and among the NSTR largest of its region and subcase (defaults TRS = −0.5,
   NSTR = 20; RTYPE = ALL overrides). The explicit group (by DesignConstr set, response, REGION, element or grid) is kept
   whatever its value and takes no NSTR place. The retained items (and the objective's) become item masks of
   `GradientOptions`: `StaticSensitivity` linearises, solves and counts (MethodChoice) the retained items only; an item
   an equation of a retained item reads is linearised for it. The masked-out rows are NaN.
2. **Approximate problem** (`ApproximateProblem`, `ResponseApproximation`, `MoveLimits`): built from the values and
   gradients of the retained items at x0 (`fromDesign`), no analysis after. Bounds: [XLB, XUB] ∩ x0 ± max(DELX·|x0|,
   DXMIN) (DELXV of the DesignVar first), cut by the limits p0 ± max(DELP·|p0|, DPMIN) ∩ [PMIN, PMAX] of a property one
   variable drives linearly; the other properties (several variables, DVxREL2) give constraints on the linearised
   property; the dependent variables (DesignVarLink, linear) give exact constraints on their move-limited bounds. Items in
   the intermediate variables of APRCOD: 1 direct, 2 mixed (1/x for DISP, SPCFORCE, STRESS, STRAIN, PointForceLoad, ESE and the
   composite types; direct for WEIGHT, VOLUME, FRMASS and the equations), 3 convex (1/x where the derivative of the
   approximated function — the normalised constraint, the minimised objective — is negative); 1/x only for a
   variable with x0 > 0 and a positive lower bound. Constraint order: retained response constraints, dependent
   variables (lower, upper), properties. Default move limits (ADS's RMVLMZ): DELX = DELP = 0.4 under APRCOD 2 and 3
   (the convex linearisation strategy of ADS), 0.2 under APRCOD 1 (its sequential linear programming); DXMIN = DPMIN
   = 0.001 (ADS's smallest scale factor SCLMIN, a choice of this project).
3. **Optimisers** (`Optimiser`, `FeasibleDirections`, `SequentialLinear`, `SequentialQuadratic`, on
   `OptimisationProblem`; `ScaledProblem` scales the variables by |x0| as ISCAL ≥ 0): MFD with the projected direction
   (QP, push-off 0) and Rosen's restoration on the binding constraints, Zoutendijk's push-off LP when violated (the
   push-off factors and participation coefficient of CONMIN); SLP with move limits (0.2 at the start, ADS's RMVLMZ;
   Fletcher's trust region rule) and an elastic LP, the L1 merit; SQP (Han-Powell, damped BFGS, elastic QP, L1 merit).
   The LP is `LinearProgram` (two-phase simplex, Bland's rule), the QP `QuadraticProgram` (Goldfarb-Idnani dual active
   set), both written here. Stops (the optimizer level of ADS): DABOBJ (|F0|/1000) or DELOBJ (0.001) for ITRMOP = 3
   iterations, Kuhn-Tucker (STOL = 0.001), no progress, ITMAX = 40.
4. **Convergence** (`Convergence`), on the exact analysis of each cycle (the strategy level of ADS with CONMIN's
   feasibility condition): the objective is steady when |ΔF| ≤ CONV2 (default |F0|/1000, ADS DABSTR) or |ΔF|/|F| ≤
   CONV1 (0.001, ADS DELSTR); the run converges to a feasible design after 2 consecutive steady cycles with max g ≤
   GMAX (0.01, the role of ADS CTMIN; 2 is ADS ITRMST), ends at an infeasible design after 5 consecutive steady
   cycles with max g > GMAX (CONMIN: "a feasible design may not exist"), and after the analysis of cycle DESMAX (20,
   ADS JTMAX) otherwise.
5. **DesignOptimParams** (`OptimisationParameters`, `ParameterTables`): every keyword of the DesignOptimParams entry with the default and
   range O_o_Solver applies, each from the ADS or CONMIN parameter whose role it plays
   (`optimiser/OptimiserDefaults.hpp` gives the sources: NASA CR-177985 and TM X-62282) or a documented choice of
   this project, and a use: optimiser (used here), driver (the property checks PTOL, PLVIOL of the SOL 200 driver),
   no effect (accepted with any value of its type and listed by `ignoredNames()` for the driver to report: the print
   controls, OPTCOD, CONVDV, CONVPR, DELB, STPSCL, and the controls of capabilities a linear sizing run never enters),
   default only (refused unless at the value that turns it off: FSDMAX, IGMAX, OBJMOD, PENAL, TREGION; METHOD above
   3).

`DesignProblem` gives the approximate problem of a point, the screening and its report, the explicit group, the
screened gradients and the DesignOptimParams parameters; `ApproximateProblem`, `Optimiser`, `Convergence` and
`OptimisationParameters` can be used on their own.

### Decisions

- **METHOD = 0** (automatic) runs the modified method of feasible directions; the format leaves the automatic choice
  open.
- **The mixed method** takes the structural responses in 1/x: the reciprocal approximation is exact for a statically
  determinate structure. Intermediate responses (a stress approximated through the force of its element) are not
  used.
- **Property move limits**: DELP/DPMIN and PMIN/PMAX of a property driven linearly by one variable cut that variable's
  bounds; other properties become constraints on the linearised property.
- **Public defaults only**: every default, range and constant of the optimisation comes from Vanderplaats' ADS
  (NASA CR-177985, 1985) and CONMIN (NASA TM X-62282, 1973), from the approximation concepts literature (Schmit and
  Farshi 1974; Schmit and Miura, NASA CR-2552, 1976; Fleury and Braibant 1986) or from a textbook (Fletcher's trust
  region), or it is a choice of this project and says so; `optimiser/OptimiserDefaults.hpp` gathers them.
- **One convergence test**, on the exact analysis: a run is never ended on the approximate optimisation's prediction
  alone. CONVDV and CONVPR have no effect (ADS and CONMIN converge on the objective and the feasibility); the changes
  of the variables and properties are reported in each `ConvergenceCheck`.
- **DABOBJ and CONV2** default to |F0|/1000 (ADS DABOBJ, DABSTR), at least 1.0E-10 (CONMIN's floor of DABFUN).
- `Convergence` takes CONV1, CONV2, GMAX and DESMAX; DELP and DELX are move limits, not convergence criteria, and are
  not taken.

### The DesignOptimParams table (as O_o_Solver reads it)

`designParametersTable` (proposed), one row per PARAMi/VALi pair: `name` text (PARAMi by its neutral
name: `MaxDesignCycles` for DESMAX) · `integerValue` int64
(VALi when an integer, else null) · `realValue` float64 (when a real, else null) · `textValue` text (OPTCOD, else
null); exactly one value per row. The tests build it by hand.

## `sol200/` — the SOL 200 driver, its outputs and checkpoints

The design cycles of a SOL 200 deck. One cycle k is a sequence of public calls: the approximate problem of x_k (one
analysis of x_k on the overlay and the screened gradients of its retained items, with the cycle number and the
approximate objective f̃_k the previous optimisation predicted), then the convergence tests on the recorded cycles,
then, when not converged, the optimisation of the approximate problem (MFD, SLP or SQP, no analysis), whose design is
x_{k+1} and whose approximate objective is f̃_{k+1}. Cycle 0 is the initial analysis; the run ends when `Convergence`
decides (a feasible design, an infeasible one whose objective no longer changes) or after the analysis of cycle
DESMAX.

- **The driver is a client of the public API**: it lives with the caller, which owns the declarations, the callables
  and the design tables of the problem it drives, and it touches nothing private of the problem, so that a plain loop
  on the public API replays its cycles to the bit. Its cost is nothing beside the analysis of each cycle.
- **No optimiser state crosses a cycle**: each approximate optimisation starts again from the new design, with the
  move limits of the DesignOptimParams (DELP, DELX, DPMIN, DXMIN apply to each cycle), so the state of a run is its history. The
  SQP quasi-Newton matrix is internal to one approximate optimisation.
- **The final model and analysis**: the final design is written into a copy of the model, whose static analysis gives
  the results of the final cycle (with `designCycle`, the same bits as the cycle's analysis on the overlay). This costs
  one analysis more than taking the last cycle's analysis as the final one, which would need the run handle of an
  overlay analysis from the static solution entry point, outside this module; reversible by adding it.
- **The optimisation tables**: O_o_Reader's named results `designVariables`, `designHistory`,
  `designVariableHistory`, `responseValues` (one row per response item and cycle: the bounds of the first
  constraint set on the item) and `sensitivities` (the retained items and the objective of each cycle, by independent
  variable), each with `designCycle`.
- **Checkpoints**: a file written through a temporary file at the end of every cycle: the versions (checkpoint
  format, O_o_Solver, O_o_Reader, data contract), the stamps of the model (a hash of the table of every card type, of
  the case section and of the SOL), of the design model (variables, relations, responses, constraints, DesignOptimParams,
  explicit group) and the names of the caller's callables, the recorded cycles (every float in its exact hexadecimal
  form) and the driver's state. A resumed run analyses its last cycle again (the same
  bits) and goes on as the uninterrupted run; a checkpoint of another model, design model or version is refused,
  naming each difference by card type or part. A progress callback is called at the start of every cycle (stage
  "design cycle"): a False answer stops the run there, the checkpoint of the previous cycle intact.
- A SOL 200 run accepts the subcases with ANALYSIS = STATICS, MODES, BUCK and SAERO (below; the others refused as out
  of scope) and DesignNodeMove; the design tables come from O_o_Reader or from the caller.

## The `DesignProblem` for external optimisers

What an external optimiser or a partner solver drives through the caller.

- **`sensitivity/GradientStream`** — the full gradient mode: every item of every response (screening ignored), G
  formed by blocks of `block` columns (the direct method by blocks of variables, `ByColumns`) or of `block` rows (the
  adjoint method by blocks of items, an equation item with its chained weights, each row block by column blocks,
  `ByRows`), whichever needs fewer solves (the direct order on a tie), each block handed to a `GradientSink` on the
  calling thread and discarded: the derivatives of the displacement field are never stored. A block holds ByColumns
  rows × B + equations × B doubles, ByRows (gSize + equations) × B + B × B: never more with more columns of G (the
  test compares 3 and 12 variables). `GradientStream::requireDestination` refuses, before any computation, a
  destination that cannot hold G (OutputVolumeError with the size, the destination and three remedies);
  `StaticSensitivity::stream` linearises every item as `compute` does and hands over to the stream;
  `DesignProblem::streamGradients` and `gradientSize`.
- **Explicit parts as sparse rows** (`ResponseLinearisation`: `VariableRow`): the explicit part ∂r/∂x of an item is a
  sparse row over the variables (a dense one would be items × variables doubles, the size of G itself), summed per
  column in the order the fields add to it, which gives the bits of the dense accumulation (`compressRow` sums from
  0.0 in the same order). `PseudoLoads::dotRange` gives λᵀ·b_j for a block of variables (the same sums as `dotAll`).
- **`problem/EvaluationStatus`** — the statuses of a point (`ok`, `singular`, `nonlinear`, `budget`, `failed`,
  `partial`) and `classifyFailure`: a SingularityError is `singular`, a MemoryBudgetError, OutputVolumeError or
  std::bad_alloc `budget`, any other SolverError (an UnsupportedCardError of a field the point makes invalid, an
  OptimisationError of an equation outside its domain or of a callable's result) `failed`; a cancellation, a broken
  contract (ContractError, FileError, CacheError), a misuse (std::invalid_argument) or an exception of a caller's
  callable always propagates. `nonlinear` comes from the linearity indicators, `partial` from the external responses
  (below).
- **`problem/DesignBatch`** — the analyses of several points in order, `parallel` at a time (a group finished before
  the next starts), each on its share of the threads (`threads / parallel`, at least one) and of the memory
  (`memory / parallel`, its own budget), sharing the base snapshot; the responses after, point by point on the
  calling thread (the callables are called from it). The same bits as the serial run (a static analysis is the same
  whatever its threads); a share too small for a point gives it `budget` where the serial run would not fail: the
  documented price of the parallel points.
- **BeamProp station fields as design targets** (`model/FieldTargets`, `input::ColumnOverlay`): a column
  `list<fixed_list<float64>[n]>` (`station_section`: A, I1, I2, I12, J, NonStructMass of each station; `station_stress`: C1 to
  SensitivityPerturbation) is a target, its item `station · n + component` as O_o_Reader's `designRelationsTable` flattens it, checked
  against the stations of the row; the overlay copies the items and values of the column (the base's list offsets
  borrowed), so that the Beam2 reads the designed station as any other.

## Shape variables

Shape optimisation by node positions along basis vectors. The deck's DesignNodeMove cards come from O_o_Reader's
`shapeVariablesTable` (`designVariable`, `nodeId`, `frameId`, `coefficient`, `directionX/Y/Z`, the vector N in the basic
system along the directions of CID at the grid point); the caller may declare shape terms of its own, N in the basic
system.

- **`model/ShapeRelations`** — the DesignNodeMove terms (`ShapeTerm`) and, per shaped variable, its basis vector
  (`ShapeBasis`: per grid point Σ COEFF·N); the shape value of a design point s_v = X_v − XINIT_v with ds/dx (the
  format defines a grid move from the change of a design variable from its initial value), in `DesignValues::shapes`
  beside the fields. Checks: a DesignVar for every DVID, a resolved and non-zero N, a finite COEFF
  (`invalid-design-shape`).
- **`model/FieldTargets`** — the grid points of each basis bound to the node table (`ShapeTarget`, `movedRows`,
  `basePositions`), the units each moves (componentNodes; every unit without it), and the refusals
  (`unsupported-design-target`) of the moves the overlay cannot make right: a grid point of an element a fastener
  pierces or the GA, GB, GS of a Fastener or Weld (fastenerPiercings), of a glued contact (contactNodePairs), or
  whose CD is cylindrical or spherical (displacementRotations): tables O_o_Reader derives from the positions at read time.
  `overlayOf` writes the field cells and the moved positions g = g0 + Σ_v s_v·T_v(g) into `xyzGlobal` of "nodes";
  `DesignPoint` and `FieldPerturbation` both build their overlays with it, so a sizing perturbation at a shaped point
  keeps the shape. `computeDesignStamps` hashes the moved positions into the stamps of their units.
- **`sensitivity/ShapeElements`** — the elements a basis reaches (the grid points of the DOFs of their stiffness, or
  mass, matrices, and the orientation point G0/GO of the line elements and connectors) and the step of its shape value,
  h = SHAPE_STEP_RELATIVE·L/|T|max (L the smallest element reached); `deformedRigidElements` lists the rigid elements
  whose grid points the basis moves relative to each other.
- **`sensitivity/StaticSensitivity`** — the designed quantities are the fields, then the shapes (`PseudoLoads` indexes
  them so): each basis is one `FieldPerturbation` (every grid point of the basis moved together: the derivative along
  the basis vector, one perturbation whatever the number of grid points), the element stiffness and thermal loads of
  the reached elements differenced (`ElementDerivatives`, FieldUse::all()), the distributed loads (PointsPressureLoad on grid
  points, BeamDistributedLoad, ShellPressureLoad, FacePressureLoad) and the inertial loads (GravityLoad, AccelFieldLoad, RotationalForceLoad: the mass and the positions) of every
  subcase that has some differenced on the whole load vector, and the explicit parts of the element items, the SinglePointConstr
  forces and the mass items (WEIGHT about the reference point of each side: a moved WeightReferenceNode moves the moments of every
  element). A DesignResp2/DesignResp3 DNODE argument moves with the shapes (`EquationResponse::coordinateShapes`, an
  `ArgumentSource::Kind::Coordinate`): its explicit part ∂h/∂c·Σ T·ds/dx.
- **`problem/ShapeDistortion`** — the mesh distortion checks: the element checks of the families (re-entrant or
  collapsed shells, a Jacobian of changing sign, coincident end points, an orientation vector along the axis, a warp
  above 0.05...) at the design point, against those of the base model; each new one is a `shape-distortion`
  diagnostic of the element. `DesignProblem::evaluateStatics` refuses a point with a distortion error before its
  analysis (SolverError: status "failed" in a loop) and adds the warnings to its results; `distortions(point)`.

Writing a design back into the model moves the Node cards and makes x the XINIT of the shaped variables. A moved grid
point that defines a RectFrame1/C/S or the direction of a PointForceLoad1/2 or PointMomentLoad1/2, which O_o_Reader resolves at read time,
is handled by `model/DerivedGeometry` (below).

### Decisions

- **N as written**. The format gives g − g0 = Σ COEFF·(X − X0)·N, N determined from CID and the Ni, and does not ask
  for a unit vector; O_o_Reader takes N as written in `shapeVariablesTable` and when it writes a design into the
  model. O_o_Solver does the same, so that the overlay of a point and the model with the design written into it are
  the same geometry. To reverse: normalise N in `ShapeRelations` (one line) and in O_o_Reader.
- **The step of the shapes**: SHAPE_STEP_RELATIVE = 3e-4 of the smallest element reached, measured
  (`ShapeElements.hpp` gives the table): at the field step 1e-2 the truncation dominates (1.7e-5 on rods, 2.7e-4 on
  higher-order shells, a warped ShellQuad4's stress lost); below 3e-4 the rounding of thin higher-order shells grows.
  Every displacement within 3e-7 of the finite differences of the analysis, every element item within 1e-5.
- **Rigid elements**: a rigid element whose grid points a shape moves relative to each other changes Gm, whose
  derivative the sensitivities do not form; they refuse it (OptimisationError naming it), its analysis being right.
  One moved whole (the same vector on all its grid points) is accepted. MultiPointConstr equations have constant coefficients:
  their sensitivities are correct.
- **Derived tables** (refusals): fastener piercing points, glued contact and curvilinear CD stay as O_o_Reader read
  them; a shape that moves such a grid point is refused at its declaration.
- **Writing a design** makes x the XINIT of the shaped variables in the problem too, as O_o_Reader does for the deck's:
  the positions and XINIT stay a pair.
- **The ShellTria3 transverse shear frame** (the median of the shortest edge) jumps when two edges are equally short; a
  shape that moves a grid point of such a triangle makes the analysis itself discontinuous (measured: the deflection
  of a 0.01 thick plate of right isosceles triangles jumps by 3 % between two positions 1e-4 apart). The gradient is
  that of the branch of the point. The tests use triangles with distinct edges.

## `aids/` — optimisation aids beyond gradients

What an optimiser or a partner solver uses besides the gradients. None changes the analysis results.

- **`LinearityIndicator`** — from the displacements, no solve: per shell grid point |n·u|/t̄ (n the mean normal of the
  shells around it, t̄ their mean thickness: volume over area), per grid point |θ|; a warning above
  `LINEARITY_DEFLECTION_RATIO` (0.336) or `LINEARITY_ROTATION` (0.150 rad). `DesignProblem::linearityReport`; the
  caller chooses whether a point beyond the limits only warns (the default), gets the status "nonlinear", or is not
  checked. The same quantities are the DesignResp1 kinds `DEFLTHK` and `ROTATION` (`ResponseType::DeflectionToThickness`,
  `Rotation`) with their gradients (`SensitivityComputation::linearisePoint`: the weights of |n·u| over t̄, and the
  explicit part of t̄ through the element volumes).
- **`SelfAdjointResponses`** — `COMPLIANCE` (c = Pᵀu = 2·the external work of the equilibrium checks) and `MEANDISP`
  (c/W, W = Σ_g |F_g| of the applied forces, OLOAD): their adjoint is the displacement, c's λ = u, so their gradient
  is uᵀ·(2·∂P/∂v − ∂K/∂v·u) (`foldSelfAdjoint`, after the explicit load parts PseudoLoads keeps with
  `keepsLoadParts`), with no solve; `GradientOptions::usesSelfAdjoint = false` solves for it (a check).
- **`SensitivityMap`** (with `ColumnPerturbation`) — dr/dv_e for every element e "as if each element had its own
  value": the whole column of the field moved per row (each element reads its own value), each element's stiffness and
  thermal loads differenced alone, one adjoint per item (none for a self-adjoint item), summed by element, property or
  unit.
- **Weight and volume** — `MassProperties` on the perturbed snapshots, linear in the fields (exact): checked against
  O_o_Reader's `designMassTable` (`massSensitivity`, `isExact`).
- **`ApproximationHints`** — the shares of the powers 0 to 3 of each field in the stiffness energy and the mass (the
  energies at four scales of the field fitted by a cubic, no solve), the exponents of each variable, and the
  intermediate variable y = x^p of each response (displacements −round(stiffness exponent), stresses and forces −1,
  masses +round(mass exponent), equations 1). `ApproximationKind::Hints`: r ≈ r0 + g·(x^p − x0^p)/(p·x0^(p−1)).
- **`SecondDerivatives`** — the adjoint second-order sensitivities on the element-level Lagrangian at fixed u, λ,
  du (Haftka and Gürdal, section 7.3), differenced at x ± h (`SECOND_DERIVATIVE_STEP_RELATIVE`); Hessian-vector
  products with two solves per product.
- **`LinearityGeometry`** — the shells' corners, normals, areas and the elements around each grid point that
  `LinearityIndicator` reads, kept by a `DesignProblem` without shape variables (built once, lent to each
  evaluation's budget); the thicknesses of each point from the elements' volumes on the pool.
- **`DisplacementDerivatives`** — du/dx_j on the g-set, one direct solve per variable and subcase, by blocks handed to a
  sink; the size estimated and a destination too small refused before any solve (OutputVolumeError).
- **`FastReanalysis`** — Kirsch's combined approximations on the kept factor: the preconditioned residuals as basis,
  the Ritz solution, η = (ρᵀK⁻¹ρ / f'ᵀu')^½ against `FAST_REANALYSIS_TOLERANCE` (1e-6) within
  `FAST_REANALYSIS_MAX_BASIS` (8) vectors; the tables recovered as the static solution does; a full analysis otherwise,
  its reason named.
- **`MonitoredResponses`** — the item count, extremes and where of responses evaluated without being constraints,
  the frequencies, eigenvalues and buckling factors and the AeroTrim and STABDER values included.
  **`NumericalDiagnostics`** — per factorisation the largest factor diagonal ratio and where
  (`FactorisedSet::pivotDiagnostics`), MaxDiagonalRatio, the pivot issues, the AutoSingularityConstr DOFs; per subcase EPSILON and the
  external work; the warnings, which the gradients carry.

`src/design/sensitivity/SensitivityComputation.hpp` is the internal header of the linearisation, pseudo-loads and
solves that `StaticSensitivity`, `SensitivityMap` and `SecondDerivatives` share (`oos::design::detail`).

### Decisions

- **The linearity thresholds**: set where the linear answer is 5 % off the geometrically non-linear one
  (`LINEARITY_RELATIVE_ERROR`): 0.336 from the clamped circular plate of Timoshenko and Woinowsky-Krieger (Theory of
  Plates and Shells, section 97), 0.150 rad from the elastica of a strip under an end moment (`LinearityIndicator.hpp`
  gives the derivations; the tests check both on the models: a clamped circular plate of 16 rings within 0.6 % of
  q·a⁴/(64·D), its two-mesh extrapolation within 1e-3; a Bar2 cantilever whose tip rotates by M·L/(E·I)). The clamped
  plate example's force went from 50 N to 25 N: at 50 N its deflection is 0.54 of the thickness, beyond the limit.
- **The kinds of the aids are extension types**: `COMPLIANCE`, `MEANDISP`, `DEFLTHK`, `ROTATION` have no RTYPE in
  the input format and are refused in a deck; only the caller declares them.
- **The compliance with enforced displacements** is refused (Pᵀu is then not twice the energy, the fold not exact):
  OptimisationError naming the subcase; a SUBCOM too.
- **The sensitivity map refuses** a field that moves the inertial or distributed loads of a subcase the response reads
  (each element's share would be needed apart) and an element reading two values of the field (a ShellProp whose MID1 and
  MID2 are two IsoMat of the column), rather than give a sum of shares under one element.
- **Second derivatives** are refused for items not linear in u (von Mises, strain energies, rotations), equations and
  relations given by the caller's functions: their second-order terms are not formed.
- **The fast reanalysis tolerance**: η ≤ 1e-6 keeps the responses within 1e-6 of the full analysis on steps of
  1 %, 10 % and 30 % of the thickness of the clamped plate, in 3 to 6 solves; enforced displacements, inertia relief
  and SUBCOM go to a full analysis, named.
- **The hints do not replace APRCOD by default**: the caller asks for them. On the clamped plate without transverse
  shear (w ∝ t⁻³ exactly) the internal optimiser takes 3 cycles with the hints and 4 with APRCOD 2, to the same
  optimum.

## Section and cut loads as responses

What an analytical sizing tool optimises with: the section and cut loads of `output/sections` as responses of a
`DesignProblem`, with their whole gradients.

- **`ResponseType::SectionLoad`, `CutLoad`** (`SECTLOAD`, `CUTLOAD`): extension types (refused in a deck, as the aids'
  kinds); `DesignResp1Definition::sections` and `cuts` hold the definitions, `componentName` selects one component.
  `DesignResp1Response` checks them at the declaration as the analysis checks them (`output::SectionLoads`, `CutLoads` on
  the snapshot: an unknown element, mixed families, a zero total weight ... are design problems, OptimisationError at
  once); its items are one per section (cut) and component, entity the section ID, component its number from 1.
- **Values**: `DesignResponses::sectionLoadRequest` gives the analysis of a design point the definitions of every
  such response tagged with its ID (`SectionDefinition::group`), `DesignProblem::evaluateStatics` adds them to
  `StaticOptions::sectionLoads` (values kept), and the items read `StaticResults::sectionLoads`/`cutLoads`
  (`ResponseContext`): computed on the load basis as any section load.
- **Gradients** (`SensitivityComputation`: `lineariseSection`, `differenceSectionItems`): the weights ∂r/∂u are the
  displacement rows of the sections and cuts on the design point's element set (exact: the loads are affine in u); the
  explicit part of a section, at fixed u, is the sum of its rows at the rates of its reached members' forces where
  their families give them in closed form (`SectionLoads::sectionFieldRates`: the shells' T and AnisoMat2D terms), a cut's
  the grid point forces of its reached elements' stiffness rates (`CutLoads::cutFieldRates`), and elsewhere the
  section or cut evaluated again on the four perturbed element sets of each designed quantity that reaches one of its
  elements (FieldPerturbation's differences, the element forces and stiffness matrices of the perturbed design; a
  shape moves the frames, the areas of the default weights and the moment arms). An equation over them (a reserve
  factor by a batch callable, each section's components an argument (n, c)) takes the chained adjoint: one solve per
  equation item (tests: 2 reserve factors, 2 solves, against 16 for the argument-first order on the 16 section loads
  they read).
- **As deck cards**: a section load can be written as one DesignResp2 per section and component, a DesignEquation of the constant
  coefficients of `SectionLoads::memberCoefficients` (weights, reduction, rotation at the model's geometry) over
  DesignResp1 PointForceLoad items (one per force column and item code, the section's elements as ATTi, the item code found by
  `locateItem`): read back, the deck gives the same values and gradients. An objective on a section load of several
  items, an equation over one, a problem with shape variables (the coefficients move with the shape) and every cut
  load (no DesignResp1 type reads grid point forces) cannot be written so.
- **Refused by the aids**: the sensitivity map (its explicit part per element is not formed) and the second
  derivatives (no role for the kind); the fast reanalysis evaluates no section loads (the response then says so).

## `external/` — external local models, their responses and products

Plugins coupled by data through the zones of the model: the contour data of `output/sections` (`ZoneContour`,
`ZoneRun`) as responses and as the arguments of external responses.

- **`ResponseType::ContourDisplacement`, `ContourForce`** (`CONTDISP`, `CONTFORC`): extension types (refused in a deck);
  `DesignResp1Definition::zones` names the zones, `ResponseParts::zones` holds their definitions and `zoneSubcases` the
  constraints of the subcases that decided their contours. Items per zone, subcase and contour DOF: entity the zone ID,
  component the component (1-6, in the zone's system), layer the grid point. `DesignResponses::zoneRequest` gives the
  analysis the zones, `DesignProblem::evaluateStatics` adds them to `StaticOptions::zones`, the items read
  `StaticResults::zones` (`ResponseContext::zones`).
- **`ExternalResponseDefinition`, `ExternalResponse`**: the third response kind beside DesignResp1 and equations. Its
  labels are zones, its arguments contour displacements, contour forces, zone loads or design variables (one per
  label: `variableByLabel`); `isLinear` (on the load basis), `hasPost`, `hasSelection`, the coupling (`Callback`, or
  `Supplied`: two-phase) and the name of its `ExternalEvaluator` (the plug-in point, which the caller implements). Per
  zone a callback that is not linear (and a supplied response) is called per batch of at most `blockItems` of its
  selected cases, its arguments the combined data of those cases (rows) × the zone's DOFs (unpadded: each zone its own
  size); a linear one once per constraint set with the data of the basis vectors (`ZoneBasisSet`), its outputs
  combined with α, then `post` per batch of cases. Items: zones, then subcases, then outputs (component the output
  number from 1).
- **Statuses** (`ItemStatus`: ok, failed, not evaluated; `ResponseValues::statuses`): a value that is not finite or a
  status the evaluator gives fails the item (NaN); the cases the selection leaves out are not evaluated. Equations over
  a response propagate the statuses (an item reading a failed or unevaluated one is so too); `DesignConstraints` leaves
  out every item that is not ok, as do the gradients (`SensitivityComputation::initialiseItems`), the screening and the
  products. A point with such items is "partial".
- **Gradients** (`SensitivityComputation`: `lineariseContour`, `lineariseExternals`, `differenceZoneItems`): a contour
  displacement's weight is its row of the zone (exact); a contour force's weights are its row of Σ K_e·u_e on the
  members (with the MultiPointConstr part w = Mᵀ·c of the rigid rows, Kgg·w its weights and −wᵀ·q its explicit pseudo-load term),
  its explicit part the members' forces at the perturbed element sets (fixed u). An external item composes the
  plugin's partials with the zone rows of its arguments: ∂h/∂u_g = Σ_j (∂h/∂y_j)·(∂y_j/∂u_g); a linear plugin's
  operator L (outputs × zone DOFs) once per zone, with post: ∂h/∂u = ∂post/∂Y·L, ∂h/∂t = ∂post/∂Y·Σ α ∂Y_j/∂t; a design
  variable argument gives its partial directly. One adjoint solve per retained item, as any response.
- **`VectorJacobianProducts`** (`vjp`, `jvp`): W·dr/dx with one adjoint solve per row of W and subcase (the weights of
  every item of every response summed into one load per row), and dr/dx·V with one direct solve per direction and
  subcase, by blocks of rows with a sink (`ProductSink`), on the full path; the items that are not ok are left out of
  W and NaN in dr/dx·V. `hessianVector` of Σ yᵢ·rᵢ: from the element blocks of the Lagrangian of (x, y)
  (`HessianBlocks`, internal: A_uu, A_up, A_pp per element and subcase from the families' closed second derivatives
  and `ElementItemFunction::secondOrder`, the SinglePointConstr force items folded into the adjoint as λ − Σ y·w), kept with the
  linearisation for the next products at the same weights; where the closed forms stop, the differences of the
  linearisation at the moved states, the reason in `ProductReport::hessianBlocksReason`
  (`ProductContext::usesHessianBlocks` false forces them).
  The products also cover the section and cut loads (differenced); inertia relief (the linearisation at a moved state
  takes the relief of the moved point, `SensitivityComputation::movedReliefOf` and `movedAccelerationsOf`, from a
  `detail::ReliefPoint`; a relative step of 1e-3 there); the condensed path (δu read where the weighted items are too:
  `ProductComputation::Reads::Direction`); the relations given by functions (`ProductContext::suppliedAt`) and the
  derived sections on the point's meshes (`StaticSensitivityInputs::sectionReference`, `DesignPoint`'s `reference`);
  the equation responses of the second order (`EquationResponse::secondOrderTerms`, `BatchCallable::hessianVector`:
  their weights moved onto their arguments, `ProductComputation::equationWeights`, their callable's terms after δu,
  `equationTerms`). Only the subcases a weight reads are solved (`weightedSubcases`), and a one-row `vjp` keeps its
  adjoint vectors for a product at the same weights.
  The section loads' blocks are in closed form (`ProductComputation::sectionMemberWeights`: a section's rows are
  Σ_m C_m·F_m of its members' force items, C_m of the geometry, so their member blocks come from
  `ElementItemFunction::secondOrder` of PointForceLoad items weighted by Σ_k y_k·C_m(k, l); `sectionForceItemCode` maps a
  section's force column to its item code); a call on responses the kept linearisation does not hold linearises every
  response (`VectorJacobianProducts::products`); a section's loads and the products of its rows are recovered once for
  its components (`SensitivityComputation::lineariseSection`, `m_sectionValues`), not at a perturbed state.
- **SUBCOM subcases**: `SensitivityComputation::prepareCombinations` gives a SUBCOM its terms (`SubcaseState::terms`),
  its terms' constraint set and `u_c = Σ a_k·u_k`; the element loop adds its stiffness terms and no load rate of its
  own; `PseudoLoads::addCombination` adds its terms' load parts after each field's `finish` (before the relief's
  projection); a term no item reads (`isTermOnly`) is never solved. `FastReanalysis` combines the terms' approximated
  vectors; `SensitivityMap` combines the terms' load parts; `SecondDerivatives` (the aid) refuses a SUBCOM.

### Decisions

- **The caller holds the plugin-facing protocol** (the external model, its identity, the selection, the two-phase
  values tied to x, to the model versions and to the declarations' generation); the core holds what is computed
  (`ExternalEvaluator` is called with arrays and gives arrays). The identity of a plugin is a stamp of the checkpoint
  ("plugin <name>"), recorded in the run record and in each cycle.
- **A linear plugin is refused on a constraint set with a thermal case** (its contour forces are not a combination of
  the basis there); it is then called case by case, declared not linear.
- **The products** (`vjp`, `jvp`) run on the full path and on the condensed path (below). A section load as an
  argument of a plugin is not offered (the plugin reads zone data); external units (a plugin's stiffness replacing its
  zone) are below.
- **No deck card holds the contour responses and the external responses**: they cannot be written as a deck.

## Stiffness from local models: multi-field relations, external units

A local model gives its stiffness back to the global model, in two forms.

- **Multi-field relations**: one relation of form "supplied" per field (DesignPropRelation1, DesignMatRelation1 or DesignElemRelation1 by the target's
  kind), all given by one callable of the caller, called once per design point for every relation that shares it,
  `values` (n, k) → (n, fields) and (n, fields, k); each field's value and derivatives reach the core as any relation
  given by a function (`SuppliedField`). The core is unchanged: a field is a field. No deck card holds such a
  relation.
- **External units** (`reduction/external`, `solutions/sol101/ExternalUnitPath`): a zone and the plugin's unit give
  at each design point K*, M*, the condensed loads and their derivatives; the analysis of the point
  (`evaluateStatics` with `StaticOptions::externalUnits`) replaces the zone's elements by them.
  `StaticResults::externalUnits` keeps, per unit, its placement (the symmetrised matrices on the contour order), the
  zone loads given to the plugin per basis vector, its f* and the coefficients α.
- **Gradients** (`SensitivityComputation::addExternalUnitLoads`): per unit derivative j and subcase s, the pseudo-load
  b_js = ∂f*_s/∂x_j − ∂K*/∂x_j·u_a,s on the contour DOFs, ∂f*/∂x_j from the unit's load source on the zone loads of
  the basis (`condensedLoadDerivatives`, one call per constraint set) combined with α, added to `PseudoLoads` directly
  for x_j (`addVariableLoad`: not through a field; a variable may drive no field of the model); the SinglePointConstr force items
  get −wᵀ·b_js in their explicit part. The direct, adjoint and chained orders, the full mode, vjp and jvp read the
  pseudo-loads as for the fields. The element set of the sensitivities is built on `ReplacedElements`, as the
  analysis', so that a field of a replaced element has no pseudo-load.

### Decisions

- **The plugin's matrices are given at each design point** (a callable of the variables, called on the calling thread
  before the analysis); its condensed loads are a callable of the zone loads per basis vector, called during the
  analysis (on the calling thread) and again for their derivatives during the gradients. The analysis kept for the
  gradients is keyed by x and the declarations, as before: a plugin is taken as a function of x (its identity is
  recorded).
- **Refused with an external unit**: the gradients of compliance and mean displacement responses (their explicit part
  would need ∂f*/∂x·u), second derivatives and the sensitivity map (no element of the zone to differentiate), the fast
  reanalysis (it falls back to a full analysis, naming the unit), a sensitivity analysis without the kept factor (its K*
  is in the factor). A batch of points is analysed one after the other (the plugins are called on the calling
  thread). The weight and volume responses read the model's elements, the zone's included (the plugin's mass is not a
  weight response).
- **No deck card holds a multi-field relation or an external unit.**

## SOL 200 and the DesignProblem on the condensed path

The analyses and the gradients of a design may take the condensed path, reduced by the boundaries of the units:

- **The path**: the caller gives a plan, "auto" or none (the full path). "auto" is the static rule of an optimisation
  (size and memory, the core's `PathChoice`), its plan condensing every unit but those the design drives
  (`DesignProblem::drivenUnits`: the units of the relations' fields and of the shape variables' grid points), those
  holding the zone of an external unit and those the planner recommends full. A SOL 200 deck runs "auto" by default;
  its final analysis takes the same path.
- **Each analysis** (`DesignProblem::evaluateStatics` → `StaticSolution` with the plan): the design point's stamps per
  unit (`DesignPoint::designStamps` by `FieldTargets::unitIds`) are part of the condensation cache key, so that a unit
  the design does not drive is condensed in the first cycle and taken from the cache in every later one; a driven unit
  given to the condensed path is condensed again at each point (its stamp changes). The interface problem is assembled
  and factorised at each analysis. The units recovered are those the design drives, those the responses read (their
  points and elements as outputs) and those the caller asks for; the others get no recovery. The displacements the
  responses and the gradients read come from the kept g-set displacements (`keptDisplacements`), never from an output
  request on every point, which would recover every unit.
- **The kept set** (`solutions::CondensedFactorisedSet`, `StaticResults::condensedSystems`): per constraint set, the
  condensed units (their UnitSystem and StaticCondensation, shared with the analysis, interior factor kept or released
  by FactorRetention), the interface factor and the m-set elimination: the gradients' K⁻¹. Nothing more is cached: the
  cache keeps K* and the condensed loads only.
- **The sensitivity solves** (`sensitivity/BoundaryReducedSensitivity`): condensation of the loaded units' interior rows
  onto their boundary, one interface solve per column, or a boundary basis Y_u (n_a solves, kept for the later blocks
  within `BOUNDARY_BASIS_RETENTION_SHARE` of the budget, the least recently used released and made again) for a unit
  more columns of a block load (its interior or its boundary) than it has boundary DOFs, each such column then
  Y_u·(f* + its rows on the boundary); then the recovery of the units read only (`SolutionReads`: the DOFs the
  linearised weights read in the direct method, the support of the pseudo-loads in the adjoint one). Interface solves
  per block: at most Σ_u min(columns of u, n_a) plus the columns loading interface rows on no condensed unit's boundary
  (a full unit's interior, an external unit's contour). `SensitivityReport::condensed` gives per constraint set and
  unit the columns condensed and recovered, the basis solves and the interior factors made again.
- **The aids** read only what they need: the section and cut loads, their reserve factors and their gradients recover
  the units holding a section's or a cut's elements; a unit that is the zone of a cut (a unit-zone) gives its contour
  forces from its K* and f* (`K*·u_a − f*`, `CondensedStatics::UnitBoundaryForces`, the zone's interior equilibrium
  rows by the unit's equilibrium) with no recovery, and its gradient weights from the rows of K*.
- **External units** on the condensed path: an external unit is a block of the interface problem
  (`reduction::InterfaceExternalBlock`, its K* on its contour DOFs), the zone's interior DOFs held; its condensed loads
  come from the plugin per interface basis vector as on the full path. The zone lies in units kept full.

### Decisions

- **An external zone lies in full units**: a condensed unit holding a replaced element is refused
  (`external-unit-condensed`); "auto" keeps such units full. The other way, a zone across condensed units, would need
  the zone's elements taken out of a K* already cached.
- **What needs the whole field refuses or falls back on the condensed path**: a total-energy response (compliance)
  and a sensitivity map need every unit recovered: they throw an OptimisationError naming the remedy (recover every
  unit, or the units needed); the fast reanalysis falls back to a full analysis, naming the reason.
- **The linearity check** reads the kept outputs (`StaticOptions::knownOutputs`): it does not force a recovery.
- **A batch analyses one point after the other on the condensed path** (the condensation cache and the kept sets are
  not shared across threads).
- **The full path stays the default of a `DesignProblem`**; a SOL 200 deck takes "auto".

## SOL 200 MODES and BUCK: eigenvalue responses and their sensitivities

The analyses of a design point are those of its subcases' ANALYSIS case section command: STATICS by the static
solution, MODES by the real eigenvalue solution (SOL 103), BUCK by the buckling solution (SOL 105), on the full or the
condensed path (the same rules as for the statics).

- **`problem/DesignAnalyses`** — the subcases of each analysis (`statics`, `modes`, `buckling`; an absent kind is not
  run, an empty list is the solution's default). `DesignProblem::evaluateAnalyses` runs the statics (as
  `evaluateStatics`), then the buckling (`BucklingSolution` with `preloadOptions` the static options of the point: the
  overlay, the design stamps and the plan; the preload results are the point's static results, so that the preload
  subcases are static subcases of the sensitivities), then the modes (`ModesSolution`). On the condensed path both take
  the point's plan (Craig-Bampton), the driven units recovered (`recoversDrivenUnits`), and for a load factor
  response every condensed unit recovered too (its derivative reads the shape and the preload on every element).
- **`responses/`** — the DesignResp1 EIGN (λ of a normal mode), FreqList (√λ/2π, the CYCLES column) and LAMA (the load factor):
  `ATTA` the mode number (1 or more), `ATTB` blank, 1 or 2 (the approximation), `PTYPE` blank (or STRUC for EIGN and
  FreqList), no targets. One item per subcase of their analysis, `(subcase, ATTA, 0, mode taken)`; each response is
  evaluated only in the subcases of its analysis (`DesignResponses::isEvaluatedIn`, `SubcaseAnalysis`): a static
  response in STATICS subcases, EIGN and FreqList in MODES ones, LAMA in BUCK ones, an equation where every response it
  reads is. A constrained response or the objective left in no subcase is refused, naming it.
- **Mode tracking** (`responses/ModeTracking`, an extension of O_o_Solver, asked by the caller) — at each design point
  the item takes the mode whose shape matches the reference best by the modal assurance criterion (Allemang 2003,
  MAC = (φ_refᵀφ)² / (φ_refᵀφ_ref · φᵀφ), on the g-set, blind to scale and sign); the reference is the shape of the
  last recorded design cycle (the first evaluation takes `ATTA`). A best MAC below `MODE_TRACKING_MIN_MAC` (0.9) or a
  second within `MODE_TRACKING_AMBIGUITY` (0.1) is an uncertain match: a warning (`mode-tracking-uncertain`), the
  evaluation's `tracking` entry. The references travel in the checkpoint (the last cycle's). A tracked response cannot
  be written as a deck card.
- **`sensitivity/EigenItems`** (internal) — the derivatives (the formulas in its class comment): vibration
  dλ/dv = φᵀ(∂K/∂v − λ∂M/∂v)φ (Fox & Kapoor 1968; φ M-normalised: the stored shape over its scale); buckling
  dλ/dv = λφᵀ(∂K/∂v + λ dK_d/dv)φ (φ K-normalised), dK_d/dv = ∂K_d/∂v|_u + K_d(du/dv), the preload part a static item
  of the preload subcase with the weights c = λ²∇_u(φᵀK_d(u)φ) (exact: K_d is linear in u), solved by the direct,
  adjoint or chained order as any static item; repeated roots (within `REPEATED_ROOT_TOLERANCE` = 1e-6 relative) by
  the eigenvalues of the subspace matrix D_j = Φᵀ(∂K/∂x_j − λ∂M/∂x_j)Φ (buckling: λΦᵀ(∂K/∂x_j + λ dK_d/dx_j)Φ, the
  preload part by the direct method), the one-sided derivatives along +x_j sorted to the cluster's modes in the order
  of their roots. `ElementDerivatives::mass` and `differentialStiffness` give the semi-analytic differences of the
  element mass and differential stiffness, as `stiffness` does for K. Per single root of EIGN and FreqList the gradients
  of the Rayleigh quotient's parts at fixed φ, ∇(φᵀKφ) and ∇(φᵀMφ) (`DesignGradients::rayleigh`).
- **`approximation/`** — the Rayleigh quotient approximation of a root (Canfield, "High-quality approximation of
  eigenvalues in structural optimization", AIAA J. 28(6), 1990): λ̃ = (λ0 + ∇U·Δx) / (1 + ∇T·Δx), FreqList from λ̃; under
  APRCOD 1 with ATTB blank and under APRCOD 2 for EIGN and FreqList of a single root; ATTB 1 the direct form, 2 the
  reciprocal; LAMA and a repeated root the direct form.
- **`aids/`** — the frequencies, eigenvalues and load factors are monitored responses and have sensitivity maps
  (`SensitivityMap`: the element shares of the same sums); the fast reanalysis stays a statics method
  (a point with MODES or BUCK subcases is analysed in full, naming why).
- **`sol200/`** — the rows of an eigenvalue response among the responses of the design model have the eigenvalue layout
  (the mode); a SOL 200 run accepts ANALYSIS = STATICS, MODES, BUCK and SAERO, honours DSAPRT(END=SENS) (the analysis
  and sensitivities of the initial design, then an information message that the run stops after the sensitivities), and
  runs the final analysis of each kind; the run's results are those of the buckling when the STATICS
  subcases are its preloads and no subcase is a MODES one, else of the statics, else of the modes (the eigenvalue
  tables of the other kinds are not in them).
- A batch of points of a problem with MODES or BUCK subcases is analysed one point after the other. "auto"
  condensation takes the rule of the analysis that asks the most (BUCK: SOL 105's, full while it fits; MODES:
  SOL 103's with its Craig-Bampton plan; else the static rule of an optimisation).

The references of the tests: the analytic roots of a spring-mass chain and their derivatives, central differences on
plates and a pinned column (direct = adjoint), the split of a repeated Euler root into 0 and λ/I1; SOL 200 decks in
memory reaching their analytic least-weight thickness, the replay loop on the public API equal to the driver's run to
the bit; the condensed path equal to the full path to round-off with every mode of every unit and within 1e-3 with two
modes per unit, design histories included.

### Decisions

- **A DesignConstr without bounds takes the defaults of the format** (LALLOW −1e20, UALLOW 1e20).
- **The eigenvector responses of a mode are not supported** (a displacement or stress "of a mode": a DesignResp1 of a
  MODES subcase on eigenvector results): mode selection and tracking belong to EIGN, FreqList and LAMA only; the others
  are refused naming it.
- **The references of the tracking are the shapes of the last recorded design cycle**, else of the first evaluation:
  a design cycle is the natural "previous" point, and an exploration between cycles (line searches, finite
  differences) does not move them.
- **The preload of a BUCK subcase is the static analysis of the point** (`preloadOptions`): its sensitivities are
  those of a static subcase, so that the preload part of dK_d needs no solve of its own beyond the order's.
- **The final eigenvalue tables carry no design cycle** (the statics' `designCycle` has no counterpart in the
  eigenvalue results of SOL 103 and 105 yet).

## SOL 200 SAERO: trim and stability derivative responses and their sensitivities

A subcase with ANALYSIS = SAERO is analysed by the static aeroelastic trim (SOL 144's `TrimSolution`, on the full or
the condensed path, the spline nodes retained) and its responses are those of the trimmed state, the trim values and
the stability derivatives differentiated through the trim system.

- **`problem/DesignAnalyses`** — `aeroelastic`: the SAERO subcases (an absent list: none; an empty one: every subcase
  of the solution's default), with `links` (the AeroTrimLink of the model). `DesignProblem::evaluateAnalyses` runs
  `TrimSolution` on the point's overlay, with the point's plan on the condensed path (the units the design drives
  condensed again at the point, their stamps in the cache key) and, when gradients are asked, `keepsLinearisation`;
  the trim's arrays join the point's results, its subcases are `SubcaseAnalysis::Aeroelastic`.
- **`solutions/sol144/TrimLinearisation`** — the trimmed state kept as a function of the structure: the trim equations
  are linear in the structure, so the derivative of the trimmed state along a design direction is the same trim
  system with the pseudo-load Q = −∂K·u − ∂M·D_r·ü_r on the structure and the constant q·D_rᵀ·T·δΔcp(Q) + D_rᵀ·(−∂M·D_r·
  ü_r) in the support equilibrium, δΔcp(Q) = (D − AEQR·q·E)⁻¹·AEQR·S·K⁻¹·Q (its class comment derives them); the
  fixed variables are held at zero. The restrained and unrestrained elastic coefficients of a variable have the same
  form, the unrestrained one on the free factor (D − AEQR·q·E_R) with the relief R·y = y − M·D_r·m_rr⁻¹·D_rᵀ·y, its
  inertia given to the structure as support accelerations (the condensed path forms it from the units' inertial
  patterns).
- **`responses/`** — the DesignResp1 AeroTrim (`ATTA` the AeroRigidTrimVariable or AeroControlSurface ID of a trim variable, no `ATTB`, no targets: its
  value, in radians for an angle) and STABDER (`ATTA` the variable, `ATTB` 0 unrestrained or 1 restrained, one target
  `ATT1` the component 1 to 6, CX ... CMZ of the elastic coefficients); `PTYPE` blank. One item per SAERO subcase,
  `(subcase, ATTA, component, restrained)`. The static responses (DISP, STRESS, PointForceLoad, ...) of a SAERO subcase are
  those of its trimmed state.
- **`sensitivity/TrimItems`** (internal) — per designed quantity, the pseudo-loads −∂K_e·u_d and −∂M_e·w_d of every
  direction d (the trimmed state of each SAERO subcase; the coefficient states of each variable a STABDER reads),
  element by element on the elements the quantity reaches (the semi-analytic differences of `ElementDerivatives`, the
  mass with the formulation and WeightToMass the trim used), then the derivatives of the trim values, of the coefficients
  and of the trimmed displacements (the static items of the subcase: dr = cᵀ·du); one structural solve per SAERO
  subcase (two with static items) and one per coefficient read, on the trim's factor. The derivatives join the
  explicit parts of the items, so the equations that read them see complete rows. On the condensed path the units the
  design drives must be full: the derivatives solve on the interface problem, which knows the interior of a condensed
  unit by its condensed load patterns only (an element of a condensed unit reached by the design is refused, naming
  it and the remedy; "auto" condensation keeps the driven units full).
- **`sol200/`** — ANALYSIS = SAERO runs; the final analyses include the trim of the SAERO subcases; the run's results
  and listing are the statics', else the modes', else the trim's, else the buckling's.
- "auto" condensation keeps the spline nodes on the boundaries of their units when a subcase is a SAERO one.

The references of the tests: the trim values and coefficients of the design problem equal SOL 144's; their gradients
and those of a displacement and a bar force of the trimmed state against central differences of the whole trim, along
the area, the inertia and the torsion constant of a bar and the thickness of a plate; the condensed path (the tail
condensed, the driven wing full) equal to the full path to round-off, gradients and design history cycle by cycle; a
SOL 200 deck with ANALYSIS = SAERO run to its bound, and its replay on the public API equal to the bit. Decks with
DirectMatrix aerodynamic correction matrices or a flutter analysis (FLUT) are out of scope.

### Decisions

- **The static responses of a SAERO subcase are those of its trimmed state** (the displacements and element results
  of the trim), differentiated through the trim system.
- **A fixed trim variable keeps its value along a design direction** (its derivative is zero): the AeroTrim card fixes it.
- **The SAERO gradients refuse a condensed unit the design drives** (the trim keeps no interior factor to condense
  the pseudo-loads; the static sensitivities' `BoundaryReducedSensitivity` is above `solutions`): keep the driven
  units full (the rule of "auto" condensation); the evaluation alone accepts them.
- **The SAERO gradients refuse a shape variable**: a shape moves the splines, the rigid-body modes and the support
  modes of the trim with its grid points, which TrimItems' ∂K and ∂M at fixed geometry do not hold; the evaluation of
  a shape design point with SAERO subcases is exact (the trim at the moved positions).

## Shape basis vectors from auxiliary models, and the directions and frames a shape moves

DesignNodeMove (shape: node positions along basis vectors), DesignBoundaryShape and DesignShapeBasis with auxiliary models.

- **The basis vectors of DesignBoundaryShape and DesignShapeBasis** — O_o_Reader gives the cards (their containers: DVID, AUXMOD, the
  (COLi, SFi) terms), the parts `BEGIN AUXMODEL = n` as models of their own and the subcases of the `AUXCASE` sections
  in `subcasesTable` (kind `auxiliary`, `auxmodel` the model they apply to). A **DesignBoundaryShape** solves its auxiliary model
  once, on its own snapshot: a copy of the part whose case section is the AUXCASE subcases of that model (their SinglePointConstr,
  MultiPointConstr, LoadCombination and TEMPERATURE selections, DISPLACEMENT = ALL; nothing of the primary case section is inherited), by the
  static solution with the problem's Solver; each subcase is a load sequence COLi, and the basis vector of the
  variable is Σ SFi·u_COLi, the translations of the auxiliary model's grid points in the basic system (u_basic =
  R_CD·u_CD) at the primary grid points of the same IDs (boundary shape vectors from an auxiliary model); a primary
  grid point the auxiliary model does not have does not move, an auxiliary point the primary does not have is left
  out. A **DesignShapeBasis** combines the columns of a displacement matrix of the primary model, which a deck takes from a
  database (DBLOCATE) and the caller gives here (the displacement tables of a previous analysis, in order, or
  `{column: {grid: move}}`). Several cards of one variable add, and add to its DesignNodeMove entries. Each vector becomes
  DesignNodeMove-like terms of the design model (N the move, COEFF 1), so that everything of the shape variables applies: the
  overlay of the positions, the element derivatives of the positions, the distortion checks.
- **`model/DerivedGeometry`** — the directions and frames O_o_Reader computes from positions when it reads the model,
  derived again at every design point from the moved positions (G1 to G4 of loadsTable, G1 to G3 of coordsTable)
  and written into the overlay by `FieldTargets::overlayOf`, so that the analysis, the perturbations of the
  sensitivities and the distortion checks all read them: PointForceLoad1, PointMomentLoad1 (G1 → G2), PointForceLoad2, PointMomentLoad2
  ((G2 − G1) × (G4 − G3)), RectFrame1, CylindricalFrame1, SphericalFrame1 (origin G1, z along G1 → G2, G3 in the x-z plane), the systems
  defined in a moved one (carried rigidly: their definition points are in RID), and the vectors given in a moved
  rectangular system (PointForceLoad, PointMomentLoad, GravityLoad of loadsTable; the direction of a GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad of
  inertialLoadsTable). Each is checked at the base positions against O_o_Reader's value (`DERIVED_CHECK_TOLERANCE`).
  `ShapeElements` adds to the elements a basis reaches those that read a system it moves (the `frameId`, `offsetFrameId`,
  `matFrameId` of the element, the `frameId`, `matFrameId` of its property) and bounds its step by the smallest distance between the
  grid points that define what it changes; `StaticSensitivity` differences the concentrated loads of a basis that turns
  one; a basis that moves a system changes every unit (its design stamp), one that turns a load the unit of its grid
  point.
- **The loads of a perturbation side read the side's positions** (a DOF map of the side: the radius of an RotationalForceLoad, the
  location of an AccelFieldLoad, besides the derived directions), in `StaticSensitivity` and `SecondDerivatives`.

The caller may also give a DesignBoundaryShape-like vector of an auxiliary model of the deck, or a vector given as values; writing
a design into the model moves the grid points of the DesignBoundaryShape and DesignShapeBasis vectors and writes x as their XINIT.

The references of the tests: the analytic deflection of an auxiliary cantilever as the basis vector, the columns of a
previous analysis as a DesignShapeBasis vector, the gradients with a turned PointForceLoad1, a moved RectFrame1 (two Bushing, a PointForceLoad, a PointMassInertia
offset and a GravityLoad in it) against central differences of the analysis, the overlay of a design point equal to the
model O_o_Reader reads after the design is written into it (to 1e-9), the two-bar truss of least weight with a DesignBoundaryShape
and with a DesignShapeBasis vector at its known optimum h = b, a SOL 200 deck with its auxiliary model; the derived directions,
frames and refusals on hand-built tables.

### Decisions

- **DesignShapeBasis has no AUXMOD**: its columns are those of a displacement matrix of a database; O_o_Solver takes the matrix
  from the caller, and a deck without it is refused, naming DesignShapeBasis and the remedy. The auxiliary models give the
  vectors of DesignBoundaryShape.
- **ShapeBoundaryNodeSet is refused** (in the check of a SOL 200 deck and in a DesignProblem): the interior motion of the boundary
  shapes it lists (analytic and geometric boundary shapes) is not computed; this version moves the grid points of the
  auxiliary model only. Write the auxiliary model over every grid point the shape moves.
- **An auxiliary grid point of a primary ID at another position is warned about**, not refused: a final design deck
  keeps its auxiliary models as they were while the primary grid points moved, and the vectors still apply by ID.
- **Refused** (UNSUPPORTED_TARGET, naming the grid point or the card): a grid point located (CP) in or displaced (CD)
  in a moved system, a DesignNodeMove whose CID is moved, a PointForceLoad or PointMomentLoad in a curvilinear system at a moved grid point or in
  a moved curvilinear system, a GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad in a moved curvilinear system, an aerodynamic model when a
  system moves; the fastener piercing points, glued contact and curvilinear CD rotations of moved grid points stay
  refused. With an O_o_Reader that does not give the derived positions, every move of such a direction or frame is
  refused.
