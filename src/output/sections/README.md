# `output/sections` — section loads, cut loads and zones of local models

The internal loads that analytical sizing tools read: the loads of sections (weighted combinations of element forces)
and of cuts (integrated grid point forces, the quantity of MonitorNodeForceSum3), computed on the load basis of a static analysis and
combined for every load case. Both are linear in the displacements, so each is a set of sparse rows applied to the
basis vectors once, then combined per load case.

- `SectionDefinition`, `CutDefinition`: what the caller defines: a section's elements with their weights, its frame
  (an axis in basic or a rectangular CID, for shells) and its reduction (average, sum); a cut's grid points, its
  elements on one side, its point (in CP) and the system CD of its axes. `SectionFamily` and `sectionComponents`: the
  components of each family (shell nx ... qy, Bar2 end forces, Beam2 ends A and B, rods axial and torsion).
- `SectionLoads`: the sections resolved against an element set (every problem a diagnostic: unknown element, element
  type no section takes, mixed families, an element named twice, a zero total weight, no frame or two, an unknown or
  curvilinear CID, an axis normal to a member); `evaluate`/`evaluateSection` recover the members with their
  families' Recovery facet on any element set with the same batches (an overlay: the sensitivities); the
  `displacementRows` (`SparseRows`) are Σ w·T·R_e/W made from unit displacements (exact: the recovery is linear);
  `memberCoefficients` the constant coefficients of the element forces in each row (used when the design problem
  writes its responses into a model); `sectionFieldRates` the rows' rates in a designed field at fixed displacements
  from the reached members' force rates (their families' `recoveryFieldRates`), false where a family gives none (the
  sensitivities then use finite differences).
- `CutLoads`: the cuts resolved (unknown grid point or element, empty sets, CP/CD missing or not rectangular); the
  six loads −Σ a_g·(K_e·u_e − f_e,T)|g along CD about the point; `displacementRows` from the element stiffness
  matrices. The class comment says how the MonitorNodeForceSum3 of a deck calls it.
- `SparseRows`: the compressed rows and their product with the k displacement vectors of a basis (rows shared by
  blocks between the threads, each value one sum in a fixed order).
- `SectionLoadAccumulator`, `SectionLoadTable`, `SectionLoadBatch`: the values batch by batch of load cases, the
  envelope (largest and smallest value of each row with the subcase of each, the first on a tie), the sink, the kept
  values (rows × cases, reserved from the budget before the analysis).
- `SectionLoadRun`: what `solutions::StaticSolution` drives (`StaticOptions::sectionLoads`): the definitions checked
  before the connectivity (SolverError naming every problem), the line-loaded bars refused per subcase, B = S·U per
  constraint set, Y = B·A(:, batch) per batch (`ResultCombination`), the thermal part of each thermal case added
  (the recovery at zero displacements), a case known by its displacements only (SUBCOM, condensed path) from the
  rows directly. On the condensed path the units of the section and cut elements are recovered.

## Design choices

- **Families**: shell (ShellQuad4, ShellQuad4Drilling, ShellQuad8, ShellTria3, ShellTria3Drilling, ShellTria6: the frame of the corners), bar (Bar2), beam
  (Beam2, its end stations), rod (Rod2, Rod2InlineSection, Tube2); other element types are refused. The section frame of a shell
  keeps the element normal: x_s is the projection of the axis on the element plane (as MCID).
- **The displacement rows from unit recoveries** rather than a recovery per basis vector: n_e recoveries per member
  (24 for a ShellQuad4) against k per member; the rows are also the exact weights ∂r/∂u of the sensitivities.
- **Bars under BeamDistributedLoad** are refused in a section (their forces are then given at the stations of the element, a layout
  the sections do not read); the cut loads exclude line and pressure loads as the grid point forces do (they are
  appliedLoad rows).
- **Named tables**: O_o_Reader's `cutLoads` named result takes the cut loads (turned to basic, about the point in
  basic); its `sectionLoads` holds a force and a moment in basic, not the components of a section, so the kept
  section loads are O_o_Reader's named result `sectionComponents` from its data contract 44 (the named matrix
  `SECTION_LOADS` before it: rows section ID and component number, columns subcases).
- **The convention of O_o_Reader's groups**: the sections give the numbers O_o_Reader's `readBlock` gives for element
  groups in a frame; an element whose normal opposes the section's has its y_s turned to the section's, as O_o_Reader
  does.

## Zones of local models

The contour of a zone and its contour data: what an external local model (a plugin) takes from the analysis of the
global model.

- `ZoneDefinition`: a zone as the caller gives it: its members, O_o_Reader's boundary of the group (`zoneBoundaries`:
  per point its components as text, whether it is a grid point of an element a fastener pierces, its causes) and its
  enclosed elements (`enclosedElements`), the policy for them (warn, include, exclude), the system of its data (0, a
  rectangular CID, or -1 the displacement systems) and the unit it was made from. `ZoneSubcaseConstraint`: the SinglePointConstr and
  load sets of the subcases that decide the contour. `ComponentMask`: the components of a point as bits.
- `ContourReport`: per contour point its components and causes, the interior points, the enclosed elements with what
  was done with each (added, held, included, kept outside, excluded), the constraints held in every subcase, the
  diagnostics; `text()` the report a person reads.
- `ZoneContour`: the zones resolved on an element set and the DOF map (every problem a diagnostic: empty zone, unknown
  element or grid point, a frame that is not rectangular, a contour point where rigid elements or MultiPointConstr equations of the
  zone and of the rest of the model both act). The rigid elements whose grid points are all in the zone are held, the
  0D elements on one of its points added, the other enclosed elements by the policy (kept outside: their grid points
  join the contour, with the components of their non-zero stiffness rows); the grid points constrained in some
  subcases only join the contour (cause "constraint"), those constrained in every subcase are interior (held). `rows`
  builds the rows of a zone (`ZoneRows`, sparse, from the element stiffness matrices of its members): the contour
  displacements in the zone's system (exact components when the frames agree, else the whole triples), the contour
  forces Σ_{e∈zone} K_e·u_e at each point less the MultiPointConstr forces where only the zone's rigid rows act, the interior loads,
  and the rows of the equilibrium check (resultants about the origin of the zone's system); `thermalPart` the element
  thermal loads of the members (−Σ f_e,T at the contour, a constant per thermal case).
- `ZoneTable`: the data of one zone in one run: labels, cases, values (contour DOFs or interior DOFs × cases), the
  equilibrium (12 × cases) and its residual per case (relative to the gross loads: Σ|F| at the contour, Σ|P| inside,
  and the moments scaled by the largest arm), and the data on the load basis per constraint set (`ZoneBasisSet`: the
  basis vectors' data and the coefficients α) when kept.
- `ZoneRun`: what `solutions::StaticSolution` drives (`StaticOptions::zones`, a `ZoneRequest`): the zones resolved
  before the connectivity (SolverError naming every problem), B = S·[u; q; P; R] per constraint set, Y = B·A per batch
  (`ResultCombination`), the thermal part of each thermal case added; a case known by its vectors only (SUBCOM, the
  condensed path) from the rows directly. On the condensed path the units of the zone's members and enclosed elements
  are recovered.

### Design choices

- **The contour comes from O_o_Reader**, never from the deck: the components of each point are its `dof` (O_o_Reader
  gives a Rod2 the six components, its torsion acting on the rotations). O_o_Solver adds the constraints of some
  subcases only and applies the enclosed policy.
- **The contour forces follow the grid point forces**: Σ_{e∈zone} (K_e·u_e − f_e,T) at the point (the element rows of
  the grid point forces with the opposite sign, thermal loads included, pressure and line loads not: they are applied
  loads), less the zone's MultiPointConstr forces where only rigid elements or MultiPointConstr equations of the zone act on the point; where
  rows of the zone and of the rest of the model both act, the zone is refused (`zone-shared-constraint`): the split of
  the MultiPointConstr force between them is not defined.
- **The loads of the zone** are the applied loads on its interior points (OLOAD); the pressures and temperatures of
  its elements are not given as loads (a plugin reads them from the model; the thermal part is in the contour forces).
- **A zone of a unit** takes the unit's elements and its absorbed spiders: its contour is the reference points. A unit
  with an absorbed fastener side is refused, naming it (the contour forces through the interpolation of its piercing
  point are not formed in this version).
