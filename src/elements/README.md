# `elements`

See [element naming](../../docs/naming.md) for the shared roots and implementation roles.

Element formulations: one folder per element type, with the same files in each.

It may use `base`, `linalg`, `input`.

Folders: `core/`, `springs/`, `masses/`, `rods/`, `bars_beams/`, `fasteners/`, `shells/`, `solids/`.

## How a batch is processed

The snapshot holds one `input::ElementBatch` per neutral card type (`ShellQuad4`), with its neutral field names
(`id`, `propId`, `offset`). Diagnostics use those same names.
For each batch the caller (the assemblers of `assembly`):

1. builds one `SnapshotIndex` per snapshot (grid and scalar points, materials and coordinate systems by ID: the lookups
   the row-index columns of `O_o_Reader` do not give; an overlay that changes no ID shares the lookups of the index of
   its base);
2. finds the `ElementFamily` of the card in `ElementRegistry::forModel(snapshot)` (the card in the formulation the
   model chose: see "Formulations" below) and makes an `ElementBatchView`;
3. calls `check()`, which reports every element it cannot process as a `Diagnostic` (card, ID, field), and stops on
   errors;
4. calls the facets on ranges of rows — `computeStiffness`, `computeMass` (`MassOptions`: lumped or coupled, the NonStructMass
   cards' share), `recover` (`ElementOutput::Forces`, `Stresses`, `Strains`, `StrainCurvatures`) — which fill an
   `ElementMatrixBlock` (per element its DOFs and its dense matrix, in the displacement systems of its grid points) or
   an `ElementResultBlock` (per element one or more rows in the columns of the `O_o_Reader` data contract, each with
   its place: the corner grid point, 0 for the centre, and the layer);
5. for the distributed loads, `checkPressureLoads` then `computePressureLoads` on a list of `ElementPressure` (the rows
   of `pressureTable` of a load set, with the batch rows of their elements), and `checkLineLoads` then
   `computeLineLoads` on a list of `ElementLineLoad` (the rows of `lineLoadsTable`: BeamDistributedLoad on bars and beams), which
   fill an `ElementVectorBlock` (per load its DOFs and the equivalent nodal loads);
6. for a subcase with TEMPERATURE(LoadCombination) or AxialDeformationLoad, `checkThermalLoads` then `computeThermalLoads` on the initial
   strains of the subcase (`ElementTemperatures`: the grid point and element temperatures of the load set and of the
   initial set, and the element deformations of its AxialDeformationLoad, the free strain D/L the rods, bars and beams add to their
   α ΔT; without TEMPERATURE(LoadCombination) no family takes a thermal strain);
   the families without thermal expansion (springs, bushes, fasteners, masses, shear panels: `isThermallyInert`) take
   none, a family with expansion that does not override the facet is refused; the recovery reads the same temperatures
   from `NodalDisplacements::temperatures` and subtracts the thermal strains;
7. for the laminated shells, `recoverLayers` (`LayerOutput`: the ply stresses, strains and failure indices) in the
   tables of `layerResultElementType()` ("ShellQuad4Composite");
8. for the elements under line loads, `recoverUnderLineLoads` (the loads of each element contiguous) instead of
   `recover`: the element forces then include the loads between the ends and the stations, in the columns of
   `outputColumnsUnderLineLoads` (a Bar2 under BeamDistributedLoad is written with the layout of its stations, in a
   result table of its own); `resultElementType` names the result table of each layout ("Bar2",
   "Bar2LoadStations").

The grid points and properties are taken by ID: a row of `nodesRows` or `propIdRow` is used only when the node or
property table holds the element's ID there (`core/ElementReferences::checkedGridRow`, `core/PropertyLookup`), since
`O_o_Reader` may give these columns in the reading order of the cards while its tables are sorted by ID.

The family is the only virtual interface, called once per range; inside a facet the batch's columns are read once and
the loop over rows is plain code. `WeightToMass` is not applied by the elements: the mass assembler multiplies the assembled
mass by it.

## Formulations

A card may be computed by several formulations, each its own `ElementFamily` with the same `card()` and its own
`formulation()` name (`core/ElementFormulation.hpp`). Every family written so far is `ClassicFormulation`, the default of every
card. A model chooses one formulation per card, for all the elements of that card, the first choice given winning:

1. the program building the snapshot: `SnapshotBuilder::setElementFormulation`, `api::ModelInput`, or the
   `elementFormulations` option of `modelFromReader`, with the exact neutral card type;
2. the model parameter `<card>Formulation`, such as `Parameter;ShellQuad4Formulation;ClassicFormulation`,
   in the model section or global case section;
3. the first formulation registered for the card in `ElementRegistry::standard()`.

`ElementRegistry::forModel` applies the choice. Unknown formulations and per-subcase choices are refused
before computation (`unknown-formulation`). The chosen formulation enters the condensation cache key.

To add a formulation of a card: write its family (in the card's folder, `formulation()` returning its name), and
register it in `ElementRegistry::standard()` after the card's default.

## Where to start reading

- `core/ElementFamily.hpp`: the facets, the options, the contract of check() and of the facets.
- `springs/springDofInline/`: the reference element without geometry (copy it for a scalar element).
- `rods/RodFormulation.hpp` and `rods/rod2/`: a line element with stiffness, lumped and coupled mass and stress
  recovery with margins.
- `springs/bushing/`: an element system from CID, GO or X, and rigid arms to an offset spring point.
- `masses/pointMassInertia/`: the rigid-body mass of an offset concentrated mass.
- `bars_beams/BeamFormulation.hpp` and `bars_beams/bar2/`: the reference line element with offsets, pin flags,
  sections and recovery points. `BeamGeometry` (end points, element system from X1-X3 or G0, OFFT, offsets, pin flags,
  the map from grid DOFs to end DOFs), `BeamSection` (stations), `BeamFlexibility` (stiffness and fixed-end loads by
  the flexibility method, with shear flexibility and taper), `PinRelease` (condensation of the pin flags), `BeamMass`,
  `BeamLineLoad` (BeamDistributedLoad), `BeamResponse` (end forces, stress resultants along the element, stresses at C, D, E, F).
  `bars_beams/beam2/` reads a BeamProp with its stations and recovers at them.
- `shells/ShellSection`, `ShellMaterial`, `Laminate`: the sections of every shell (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D;
  CompProp and CompPropGlobal as their equivalent ShellProp), in the material frame, turned into the element frame by each element;
  `ShellRecovery` the output rows (the force table with its moment sign convention, the ply tables and failure
  indices); `ShellTemperature` the temperature change of a shell, `ShellPressure` the direction of a FacePressureLoad.
- `shells/shellQuad8/`, `shells/shellTria6/`: the higher-order shells, flat or curved, `MindlinShell` on `HigherOrderShapes`,
  with the facets shared in `HigherOrderShell` and the geometry in `HigherOrderGeometry`: the degenerated
  (continuum-based) shell of Ahmad, Irons & Zienkiewicz (1970) with uniform reduced integration (2 × 2, three points:
  Zienkiewicz, Taylor & Too 1971; Hughes 2000; Bathe 1996; the S8R of CalculiX); one director per grid point shared by
  the ShellQuad8 and ShellTria6 of its smooth group there (`HigherOrderShellDirectors`, built once per snapshot index through
  `SnapshotIndex::computedOnce`: the normalised mean of the elements' own normals, Hughes & Liu 1981, Bathe 1996 §5.4;
  separate groups across edges sharper than 30°, folds, T-junctions, stiffeners; a group of flat elements only, or of
  one element, keeps the elements' own normals), five DOFs per node in the shell sense carried by the six of the grid
  point, the rotation about the director without stiffness (no drilling stiffness, as the ShellQuad4 and ShellTria3: no ShellDrillingStiffness;
  at a grid point of one smooth group it is exactly singular and AutoSingularityConstr removes it); the linear shell strains, linear
  through the thickness, with the bending correction of Sanders (1959: a uniform dilatation bends nothing), on the
  section's resultants (MID1-MID4, CompProp, CompPropGlobal: Büchter & Ramm 1992; Simo & Fox 1989); on a flat element exactly the
  isoparametric membrane and Reissner-Mindlin plate of the shape, bit for bit the former flat element. The mass and the
  pressure loads with the full rule (3 × 3, seven points) and HRZ lumping, FacePressureLoad normal to the surface at each point,
  ZOFFS along each node's director, the stresses and forces of the centre and corners in the tangent frame of each place
  (x the element x axis projected on it). An element turning by more than 45° from the plane of its corners is refused.
  Validated (throwaway decks, meshes genuinely curved, the ShellTria6 by splitting each quadrilateral; ratio to the
  reference; no Parameter SingularityBailOut): the Scordelis-Lo roof (0.3024, whole roof) ShellQuad8 0.989 / 1.003 / 0.999 / 0.999 on 4²,
  8², 12², 16², ShellTria6 0.33 / 0.82 / 0.95 / 0.98 (the membrane locking of the six-node triangle on curved shells); the
  pinched cylinder with rigid diaphragms (1.8248e-5, octant) ShellQuad8 0.896 / 0.981 / 0.996 / 1.003 / 1.009 on 4² to 24²,
  ShellTria6 0.39 / 0.73 / 0.89 / 0.95 / 0.99; the pinched hemisphere with an 18° hole (0.094, quadrant) ShellQuad8 0.811 /
  0.992 / 0.996 / 0.997 on 4² to 16², ShellTria6 0.05 / 0.38 / 0.73 / 0.88; each the same, to 1e-9, with the rotation about
  the radius constrained explicitly in a cylindrical or spherical system (AutoSingularityConstr removes the skewed singular direction
  exactly); a cylindrical panel with shear diaphragms within 1 % of Donnell's first six frequencies (lumped and coupled
  mass), the free panel's six rigid-body modes below 1e-5 of the first elastic one; the axisymmetric buckling of a
  cylinder in axial compression 0.996 of the classical value; a free curved panel under a uniform temperature the
  homothety to 1e-6, its stresses below 1e-5 of E α ΔT; the hoop force p R of a cylinder under pressure, with and
  without ZOFFS (with ZOFFS the director of a node at a free or symmetry edge, the element's own normal, is off by its
  interpolation error α³/4: a scatter of 0.4 % of the radial displacement on 15° elements, 0.01 % on 4°); a curved
  panel with a stiffener, a 90° flange or a ShellQuad4 stiffener solves with no singularity at the junction.
  `shells/shellQuad4Drilling/`, `shells/shellTria3Drilling/`: the ShellQuad4Drilling and ShellTria3Drilling,
  families implemented directly by `ShellQuad4DrillingElement` and `ShellTria3DrillingElement`, with the
  drilling membranes `ShellQuad4DrillingMembrane` and `ShellTria3DrillingMembrane` (below).
  `shells/shearPanel4/`: the shear panel (`ShearPanel4Panel`).
- `shells/shellQuad4/ShellQuad4*`, `shells/shellTria3/ShellTria3*`: the ShellQuad4 and ShellTria3, the formulation of NASTRAN-95 (the
  program of NASA, open-sourced in 2015), implemented from its mathematical description, not translated. `ShellQuad4`:
  a degenerated shell with nodal normals (2 × 2 × 2 Gauss points), reduced in-plane shear at the centre, transverse
  shear from the four tangential edge strains with the NASTRAN-95 shear-correction factors (C_b = 1.6 on the even
  part, C_t on the odd part, the anisotropic factors; `MID3` in series), `MID4` as G1 − 6ζ G4. `ShellTria3`: linear
  membrane and bending at the edge midpoints, a constant transverse shear plus one parameter fitted to the edge strains,
  its stiffness the inverse of the compliance Z (`MID3` plus a bending relief from L_x, L_y). Neither has a drilling
  stiffness (no ShellDrillingStiffness: θz of a flat mesh is removed by the automatic constraint of singular DOFs) nor rotary inertia.
  `PlateConstitutive` reads a section (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D; CompProp and CompPropGlobal through their equivalent section,
  this project's choice) into the matrices of the kernels; `ShellCentreRecovery` the centre recovery as an affine map
  of the local DOFs; `ShellScalar` the small matrices templated on the scalar. `EdgeShearShellFamily` (with
  `EdgeShearShellFamily.ipp`, `EdgeShearShellRecovery.ipp`, `EdgeShearShellDerivatives.ipp`, instantiated in
  `ShellQuad4Element.cpp` and `ShellTria3Element.cpp`) gives both their facets: the sections, the material axis
  (THETA from G1→G2, MCID projected), ZOFFS through `ShellNodeFrames`, the rows of `ShellRecovery` at the centre in
  the user frame (x along G1→G2; no corner output), the plies of a laminate. Its header lists the deliberate
  deviations from NASTRAN-95 (OrthoMat2D G1Z and G2Z on their physical shears, an MCID angle measured and applied in one
  frame, the offset in the thermal loads, the thickness of a ShellQuad4 without MID1 kept, the connectivity order of the
  TRIA3 pressures, the coupling in the recovered membrane forces, the thermal curvature and MID4 of the TRIA3
  recovery, the differential stiffness).
- `shells/shellQuad4Drilling/`, `shells/shellTria3Drilling/`: the ShellQuad4Drilling and ShellTria3Drilling families.
  `ShellQuad4DrillingElement.hpp` says which file holds which
  part: `ShellQuad4DrillingGeometry` (geometry and frames), `ShellQuad4DrillingBending` (bending and transverse shear), `ShellQuad4DrillingWarpOffsets`
  (the rigid offsets of a warped element formed flat), `ShellQuad4DrillingPlate` (the sum with the drilling membrane
  `ShellQuad4DrillingMembrane`, the mass, the pressure, the recovery points); `ShellTria3DrillingPlate` the edge-shear triangle with
  `ShellTria3DrillingMembrane`. Neither takes a membrane-bending coupling. The section (`shells/ShellSection`), the passage to the
  displacement systems with ZOFFS (`shells/ShellNodeFrames`) and the output rows (`shells/ShellRecovery`) are shared by
  every shell.
- `shells/Laminate`: the transverse shear of a laminate from the equilibrium of cylindrical bending (`H_x`, `H_y` and
  the shear at the top of each ply); `ShellRecovery` gives each ply that distribution, or the mean `Q / t` for a
  homogeneous stack or a rigid direction.

## Solids

`core/SolidShapeFunctions` and `core/SolidIntegrationRules` give the parent shapes (tetrahedron, pentahedron,
hexahedron and the pyramid as a collapsed hexahedron, with and without mid-side nodes) and their rules;
`core/MaterialMatrices` the 6 × 6 isotropic matrix and the strain transformation. The family folder `solids/` holds
what the four cards share: `SolidConnectivity` (the grid points, the mid-side nodes all or none), `SolidSection`
(SolidProp: CORDM, IN, ISOP, STRESS, FCTN; IsoMat from E and NU, AnisoMat3D from the snapshot table `materials/AnisoMat3D`),
`SolidIsoparametric` (the displacement-based element, consistent mass, Jacobian check, the extrapolation of Gauss
point values to the corners), `SolidPressure` (FacePressureLoad on a face from `ElementPressure::faceNodes`), `SolidRecovery`
(the 20 columns of the solids' tables, the principal values in the order and with the signs of those tables),
`SolidResponse` (stiffness and recovery operators of one element in its material system, and the thermal share of
condensed internal modes) and `SolidElementKernel` (the facets: resolve, check, stiffness, mass, recovery, pressure and
thermal loads, the passage between the material system and the displacement systems). Each card's folder gives its
formulation (`<Card>Stiffness.cpp`). The default of each shape (SolidProp IN and ISOP blank) is a textbook element:

| Shape | Default | Other SolidProp choices |
|---|---|---|
| SolidHexa, 8 nodes | `solidHexa/SolidHexaIncompatibleModes`: the trilinear element with the incompatible modes (1 − ξ²), (1 − η²), (1 − ζ²) of Wilson, Taylor, Doherty & Ghaboussi (1973), their derivatives through the centre Jacobian with Taylor, Beresford & Wilson's factor det J0 / det J (1976), condensed, 2 × 2 × 2 points (CalculiX C3D8I; Bathe §4.4.1); stresses at the corners extrapolated from the Gauss points | ISOP = FULL: the plain trilinear element, 2 × 2 × 2 (C3D8) |
| SolidHexa, 20 nodes | the serendipity element, 3 × 3 × 3 (C3D20) | ISOP = REDUCED or IN = TWO: 2 × 2 × 2 (C3D20R), stresses extrapolated from the Gauss points |
| SolidPenta, 6 and 15 nodes | the isoparametric wedge, 3 × 2 and degree-4 triangle × 3 points (C3D6, C3D15; Code_Aster PENTA6, PENTA15) | none (the 6-node wedge takes IN = BUBBLE, ISOP = REDUCED or FULL as the same element) |
| SolidTetra, 4 and 10 nodes | the constant strain tetrahedron (one point); the quadratic tetrahedron, Keast's five points | none (4 nodes: IN = TWO, ISOP = FULL or REDUCED, STRESS = GAUSS name the same element) |
| SolidPyramid, 5 and 13 nodes | the collapsed hexahedron, 2 × 2 × 2 and 3 × 3 × 3 (Zienkiewicz & Taylor vol. 1, degenerate elements) | none |

The mass of every solid is the consistent mass or its HRZ lumping (Hinton, Rock & Zienkiewicz 1976). Validated (a
throwaway check on the element operators): the 7-element distorted patch of MacNeal & Harder (1985) exact to
round-off for the 8-node defaults and ISOP = FULL and both 20-node rules; six zero eigenvalues for the 8-node elements
on a distorted shape (twelve for the reduced 20-node element alone, as the rule gives); a cantilever L/h = 10 of
10 × 1 × 1 incompatible-modes SolidHexa within 0.4 % of Euler-Bernoulli plus shear (ν = 0; the plain element 33 % stiff);
pure bending exact for ν = 0, 0.3, 0.49; a free linear temperature gradient bending a beam without stress; on
MacNeal-Harder's straight beam (six elements) 0.98 / 0.97 (in-plane / out-of-plane shear) for rectangles, 0.62 /
0.53 for parallelograms, 0.05 / 0.03 for trapezoids (the known limit of incompatible modes on tapered shapes).

## Differential stiffness

The facet of the buckling solution (SOL 105), built from the element's static internal forces:
`checkDifferentialStiffness` then `computeDifferentialStiffness(view, range, preload, block)`, `preload` the
displacements (and temperatures) of the static preload subcase; each family recovers its own internal forces from them,
so the SOL 105 driver calls it once per batch after the static solve and assembles K_d on the stiffness pattern. The
buckling problem is (K + λ K_d) φ = 0.

| Family | K_d | Files |
|---|---|---|
| `Rod2`, `Rod2InlineSection`, `Tube2` | the axial force P turning with the axis: (P/L)·[D −D; −D D], D = I − x xᵀ, on the translations (the thermal force subtracted) | `rods/RodFormulation` (`rodDifferentialStiffness`), `<card>Differential.cpp` |
| `Bar2`, `Beam2` | the consistent matrix of the axial force on the Hermite deflections, N/(30L)·[36 3L ...] in both planes; pin flags condensed with the elastic map; offsets and CD by T | `bars_beams/BeamDifferential`, `bar2/Bar2Differential.cpp`, `beam2/Beam2Differential.cpp` |
| `ShellQuad4`, `ShellTria3` | the membrane resultants (Nx, Ny, Nxy) the element recovers at its centre (the mean over the Gauss points of the ShellQuad4), on the in-plane gradients of u, v, w (∫ ∇Nᵀ·N·∇N dA), ZOFFS by ShellNodeFrames; NASTRAN-95 forms these elements' differential stiffness with the sub-triangles of its older QUAD1 and TRIA1, a deviation of this project | `shells/ShellDifferential.hpp`, `ShellQuad4`, `ShellTria3`, `EdgeShearShellFamily.ipp` |
| `ShellQuad4Drilling`, `ShellTria3Drilling`, `ShellQuad8`, `ShellTria6` | the membrane resultants (Nx, Ny, Nxy) at the centre on the in-plane gradients of u, v, w (∫ ∇Nᵀ·N·∇N dA), ZOFFS by ShellNodeFrames | `shells/ShellDifferential.hpp`, `ShellQuad4DrillingPlate`, `ShellTria3DrillingPlate`, `MindlinShell::differentialStiffness` (on a curved element the gradients along the tangent axes of each point), `<card>Differential.cpp` |
| `ShearPanel4` | the mean shear flow q̄ as Nxy on the bilinear functions of the panel (the rods of F1, SensitivityPerturbation and the varying flows left out) | `shearPanel4/ShearPanel4Panel::differentialStiffness`, `ShearPanel4Differential.cpp` |
| `SolidHexa`, `SolidPenta`, `SolidTetra`, `SolidPyramid` | the initial-stress stiffness ∫ ∇Nᵀ·σ·∇N dV of the centre stress, on the isoparametric functions with the mass rule (whatever the linear formulation) | `solids/SolidIsoparametric` (`buildInitialStressStiffness`), `SolidElementKernel`, `<card>Differential.cpp` |
| `CELASi`, `Bushing`, `Bushing1D`, `Gap2`, `Fastener`, `Weld`, masses | none (`isDifferentiallyInert`: scalar or local springs whose force does not turn; no stiffness) | `core/ElementFamily` |

A family with stiffness and without the facet is refused by the default `checkDifferentialStiffness`
(`unsupported-card`). The choices above (the centre values, no bending-moment or torque terms, no rotation terms) are
documented in each header; they are the standard initial-stress matrices (Bathe §6.3.2, Cook et al. §18,
Przemieniecki §15.3). They are tested so that each element's K_d turns its prestress under a rigid rotation
(K_d·u_rot = ω × K·u0) and gives the exact energy of a linear (cubic for the beams) field; the rods against the central
difference of their turning axial force; Euler columns of Bar2 and Beam2 (four end conditions, pin flags), a simply
supported square ShellQuad4 plate (k = 4), the ShellTria3 plate, a SolidHexa cantilever column, all through the block Lanczos
solver.

## Design derivatives

The Derivatives facet (`Facet::Derivatives`): every family of the registry offers it, in a `<Card>Derivatives.cpp` of
its folder, as `designFieldUse(DesignedField)`: what a designed real field of the element's property, materials or own
card changes in its computations (`FieldUse`: stiffness, mass, loads, recovery; `core/DesignFieldUse` holds the rules
every family shares: a density changes the mass only, an allowable the margins only, a structural damping nothing, NonStructMass
the mass only, any field not named everything). The derivatives themselves are computed by
`design/sensitivity/ElementDerivatives`, semi-analytically: the family's own facets run on the snapshots where the
field is moved by ±h and ±h/2 (a `ColumnOverlay` of the design point), and their differences are extrapolated
(Richardson). Nothing in a family is written twice: whatever an element computes from its fields (a section, a
laminate, an offset, a thermal strain, a fastener's interpolation), its derivative follows; `FieldUse` only spares the
computations a field cannot change (a zero difference, exactly). The fields O_o_Reader derives at read time stay as read
(the G of a IsoMat from E and NU, the Z1 and Z2 of a ShellProp from T, the OD2 of a TubeProp from OD: a designed OD makes a
tapered tube, refused), as for the analysis of a design point.

**Analytic derivatives**: a family may give what it knows in closed form, and the sensitivities difference the rest;
each is optional, the choice explicit per family and field. The `ShellQuad4` and `ShellTria3` (`EdgeShearShellDerivatives.ipp`)
take theirs by the complex step through their kernels (templates on the scalar: Im f(x + i·h ẋ)/h, h = 1e-30, exact
first derivatives), their second derivatives in T by the 4-point central rule on complex-step first derivatives
(δ = 1e-3 T); the rows below name the fields.

| What | Facet | Families |
|---|---|---|
| The form of each output column in u (`ColumnForm`: Linear, PlaneVonMises of three Linear columns, Other) | `outputColumnForms`, `layerOutputColumnForms` | the shells (`shellOutputColumnForms`, `laminateOutputColumnForms`, the ply states all Linear), the rods, `Bar2`, `Beam2`, `ShearPanel4`, the solids (`solidOutputColumnForms`: the six components Linear), the springs and fasteners (every column Linear but the `Weld`'s stresses, whose axial stress alone is: `core/LinearRowRecovery`) |
| The recovery matrix of an element (`RecoverySlopes`: the slope of every row and column with respect to its DOFs) | `recoverySlopes`, `layerRecoverySlopes` | `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling` (the plate built once, `shellRecoverySlopes`), `Rod2`, `Rod2InlineSection`, `Tube2` (closed form, `rodRecoverySlopes`), `ShellQuad8`, `ShellTria6` (`higherOrderShellSlopes`), `Beam2` (one end-force map), `ShearPanel4` (its mean flow), the solids (the operators of their formulation, `SolidElementKernel::recoverySlopesOfOutputs`), the springs (`SpringDofProperty` to `SpringScalarInline`, `Bushing`, `Bushing1D`, `Gap2`: their maps K·s, c·H) and fasteners (`Fastener`: diag(K)·H·M; `Weld`: its end forces' map K·T·M through the forces at A' and B'), from one resolution, the piercing points interpolated once; the others' slopes from recoveries at unit displacements |
| The slopes of several outputs at once (the vectorised recovery) | `recoverySlopesOfOutputs` (element and layer outputs, `RecoveryOutput`) | `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`, `ShellQuad8`, `ShellTria6` (one plate and one set of place states for every output and ply table, `shellRecoverySlopesOfOutputs`, the rows without their principal values, `ShellRowValues::LinearOnly`), the solids (one formulation), the springs and fasteners (one resolution, `linearRowSlopesOfOutputs`); the others each output by `recoverySlopes` (the `Bar2`'s from one end-force map, `BeamEndForces`). The shells form the place states' parts that do not depend on the displacements once for every unit displacement (`ShellQuad4DrillingPlate::PlaceOperators`, `ShellTria3DrillingPlate::CentreOperators`, `MindlinShell::PlaceOperators`: the bits of the states made one by one) |
| The rows of several outputs from one formulation (the rows at rest of a block of cases, the outputs of a case recovered alone) | `recoverOutputs` (by default `recover` or `recoverLayers` for each output) | the shells (one resolution, plate and set of place states per element: `resetShellRecoveryBlocks`, `appendShellOutputRows` with `ShellRowValues::All`; the `ShellQuad4` and `ShellTria3` one centre recovery operator), the solids (one formulation, `SolidElementKernel::recoverOutputs`); each output's rows those of its recovery alone, bit for bit |
| The columns that are not Linear completed from the Linear ones (the vectorised recovery), and from how many cases a block pays for the slopes | `completeRecoveredRows`, `vectorisedRecoveryMinCases`; the ply tables `completeRecoveredLayerRows`, `vectorisedLayerRecoveryMinCases`, `vectorisedLayerSource` | the shells above (`completeShellRows`, 8 cases; their ply tables `completeLaminateRows`, the failure indices from the ply states, `completeLaminateFailureRows`, 4), `Bar2` (its extremes and margins, 8), `Beam2` (its extremes and the margins at one station, 8), `ShearPanel4` (its largest shear and margin from its mean flow, 8), the solids (`completeSolidRows`, 4: their principal values by Jacobi sweeps of four rows side by side, `principalStatesOf`, the bits of one row's), `Rod2`, `Rod2InlineSection`, `Tube2` (their margins, `completeRodRows`, 4), the springs (nothing to complete, `SPRING_VECTORISED_RECOVERY_MIN_CASES`, 4) and fasteners (`FASTENER_VECTORISED_RECOVERY_MIN_CASES`, 2; the `Weld`'s stresses made from its forces, `vectorisedSource`, `completeRecoveredRowsFromSource`) |
| dK/dv, dM/dv of a designed field | `stiffnessFieldDerivative`, `massFieldDerivative`; several fields of one element at once `stiffnessFieldDerivatives` (by default one by one) | the `ShellProp` T of `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling` whose corners follow T; the A and J of `Rod2` (`RodProp`) and `Rod2InlineSection`; the AnisoMat2D stiffness terms (`stiffness`, named by their MID: `DesignedField::id`, `item`) of the `ShellProp`'s MID1 to MID4 for `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`, `ShellQuad8`, `ShellTria6` (`ShellSectionRate`; no mass: the terms change none), a MID4 at a coupling zero at the point too but on the drilling membranes; the ply thicknesses of a CompProp or CompPropGlobal (`ply_thickness`, the item the card's ply) for the six shells (`LaminateRate`, `laminateSectionRate`; the mass of the ShellQuad4, ShellTria3 and their drilling variants) |
| The ply rows' rates in a designed field | `layerRecoveryFieldRates`, several fields `layerRecoveryFieldRatesOfFields` | a ply thickness of the shells' laminates: the ply stresses, strains, failure and bond indices, ply states (`appendLaminateRowRates`) |
| dK_d/dv at a fixed preload (the buckling factors' explicit parts) | `differentialStiffnessFieldDerivative` | the shells above in their AnisoMat2D terms, and the T of `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`; none under a temperature change of the preload |
| The rows' rates with respect to a designed field at fixed u (the explicit parts) | `recoveryFieldRates`; several fields at once `recoveryFieldRatesOfFields` | the same fields and families (the AnisoMat2D terms not under a temperature change) |
| d²K/dv², d²M/dv² (the Hessian blocks) | `stiffnessFieldSecondDerivative`, `massFieldSecondDerivative` (true with nothing appended: zero) | the same fields and families (the rods' all zero) |
| The slopes of the rows' rates, ∂S/∂v (the mixed second derivatives), and the rows' second rates at fixed u | `recoverySlopeRates`, `recoveryFieldSecondRates` | the same fields and families, not under temperatures (the columns that are not Linear NaN: not formed; a von Mises from its components) |

**Checks by ranges**: a family whose check is row by row says so (`isCheckedByRow`) and checks a range of rows
(`checkRows`), so that `assembly::ElementSet::check` checks a long batch by ranges on the pool: the `ShellQuad4` and
`ShellTria3` families (and so `ShellQuad4Drilling`, `ShellTria3Drilling`); the others check their batch whole.

Shape variables (DesignNodeMove) need nothing more from a family either: `design/sensitivity/ShapeElements` finds the elements
whose grid points (those of the DOFs of their matrices, and the orientation point G0 or GO) a shape moves, and the same
facets run on snapshots whose positions (`xyzGlobal`) are moved along the basis vector by ±h, ±h/2; a family whose
computations read no position (the scalar springs and masses) gives exact zeros. What a family reads from a table
O_o_Reader derives from the positions (the piercing points of Fastener and Weld, `fastenerPiercings`) cannot follow a
move: such a shape is refused by `design/model/FieldTargets`.

## Element status

| Card | Facets written | Not yet supported |
|---|---|---|
| `SpringDofProperty`–`SpringScalarProperty` | stiffness, recovery (force, stress), derivatives | |
| `SpringScalarInline` | stiffness, recovery (force), derivatives | |
| `Bushing` | stiffness, recovery (forces) | stresses and strains (`RCV`), `BushProp` mass, curvilinear `CID`/`OCID` |
| `Rod2`, `Rod2InlineSection`, `Tube2` | stiffness, mass (lumped, coupled), differential stiffness (axial force), thermal loads, recovery (forces, stresses, strains, margins) | strain energy, tapered `Tube2` |
| `PointMassMatrix`, `PointMassInertia` | mass, derivatives | curvilinear `CID` |
| `MassDofProperty`–`MassScalarInline` | mass, derivatives | |
| `ShellQuad4` | the NASTRAN-95 QUAD4: stiffness (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D: MID1-MID4, MID2 = −1, 12I/T³, TS/T; CompProp, CompPropGlobal as their equivalent section; ZOFFS, T1-T4, THETA/MCID; no drilling stiffness, no ShellDrillingStiffness), mass (translational, lumped or coupled, NonStructMass), loads (FacePressureLoad and ShellPressureLoad with corner values and direction; thermal: the mean temperature change and the PlateTemperatureLoad gradient), recovery (the centre only, in the user frame: stresses, strains, curvatures, forces; plies and failure indices), differential stiffness (its recovered membrane resultants), derivatives (complex step) | strain energy, corner output |
| `ShellTria3` | the NASTRAN-95 TRIA3, as `ShellQuad4` | as `ShellQuad4` |
| `ShellQuad8`, `ShellTria6` | as `ShellQuad4` (flat or curved, every edge node; no drilling stiffness, no ShellDrillingStiffness; rotary inertia in the coupled mass), recovery at the centre and corners | as `ShellQuad4`; an element turning by more than 45° |
| `ShellQuad4Drilling`, `ShellTria3Drilling` | a membrane with a drilling DOF (`ShellQuad4DrillingMembrane`, `ShellTria3DrillingMembrane`) with the bending of `ShellQuad4DrillingBending` and of the edge-shear triangle (the ShellQuad4Drilling recovered at the centre and corners); a warped ShellQuad4Drilling formed flat in its mean plane and joined to its grid points by rigid offsets along the normal (`ShellQuad4DrillingWarpOffsets`, Taylor 1988), in every facet but the mass (on the grid points) | strain energy; coupling |
| `ShearPanel4` | stiffness (ShearPanelProp, IsoMat, F1, SensitivityPerturbation), mass, recovery (stresses, strains, forces) | warped panels |
| `Bar2` | stiffness (BarProp, IsoMat: A, I1, I2, I12, J, K1, K2; X1-X3 or G0, OFFT, WA, WB, PA, PB), mass (lumped, coupled, NonStructMass), loads (BeamDistributedLoad, every type and scale), recovery (forces at A and B; stresses and strains at C-F, axial, extremes, margins; the station layout under BeamDistributedLoad) | BarShapeProp, differential stiffness, thermal loads, strain energy, Bar2Stations stations |
| `Beam2` | as `Bar2` with BeamProp stations (taper, intermediate stations, SO), recovery at the stations, the torsional inertia ρ·(I1 + I2) in the mass; a BeamProp's CW is not used without SA, SB | BeamShapeProp, BeamLumpedAreaProp, warping (SA, SB), StressSign, S2, NSI, M1-M2, N1-N2 offsets, as `Bar2` |
| `SolidHexa` | stiffness (8 nodes: the default incompatible modes, or ISOP = FULL; 20 nodes: 3 × 3 × 3, or 2 × 2 × 2 with ISOP = REDUCED), mass (lumped HRZ, coupled), loads (FacePressureLoad on a face, corner pressures; thermal loads of the grid point temperatures, IsoMat A, AnisoMat3D A1-A6), recovery (centre and corners: stresses with the thermal stress subtracted, total strains, principal, von Mises, pressure, in CORDM); SolidProp, IsoMat, AnisoMat3D | partial mid-side nodes, CORDM −1/−2, other IN/ISOP, STRESS = GAUSS |
| `SolidPenta` | as `SolidHexa` (the isoparametric wedge, 6 and 15 nodes; the 15-node not validated) | as `SolidHexa` |
| `SolidTetra` | as `SolidHexa` (4 nodes constant strain, 10 nodes five points); with 4 nodes also IN = TWO, ISOP = FULL or REDUCED and STRESS = GAUSS (its one Gauss point the centroid) | as `SolidHexa`, but STRESS = GAUSS with 4 nodes |
| `SolidPyramid` | as `SolidHexa` (collapsed hexahedron, corner stresses extrapolated; not validated) | as `SolidHexa` |
