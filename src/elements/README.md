# `elements`

See [element naming](../../docs/naming.md) for the shared roots and implementation roles.

Element formulations: one folder per element type, with the same files in each.

It may use `base`, `linalg`, `input`.

Folders: `core/`, `springs/`, `masses/`, `rods/`, `bars_beams/`, `fasteners/`, `shells/`, `solids/`.

The strain energy of an element (ESE) and its grid point forces come from its stiffness matrix, for every family
with the Stiffness facet (`output/arrays/StiffnessRecovery`): no family writes them itself.

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

`MixedHybridFormulation` (`mixedHybrid/`, [docs/mixed-hybrid-formulation-spec.md](../../docs/mixed-hybrid-formulation-spec.md))
is chosen for the whole model (`ElementFormulation` parameter, `SnapshotBuilder::setModelFormulation`,
`api::ModelInput::setModelFormulation`, the `elementFormulation` option): every card of `MIXED_HYBRID_CARDS` takes
it together, a per-card choice of it is refused (`formulation-per-card`), and so is a per-card choice of another
formulation for one of those cards (`formulation-mix`); `ShearPanel4` has no counterpart (`unsupported-card`). Its
elements share unknowns on edges, faces and interiors: the entities of `core/EntityLayout`, numbered by the DofMap
after the scalar points, which the model of the formulation (`core/FormulationModel`, `mixedHybrid/model/
MixedHybridModel`) gives with its ties, the grid components it determines and the supports it derives.

The folder `mixedHybrid/`:

| Folder | Content |
|---|---|
| `core/` | dense matrices and decompositions, quadrature, polynomial bases and spaces, dual numbers |
| `geometry/` | the reading of the mesh: topology, surfaces of the shells, skin of the solids |
| `shell/`, `line/`, `solid/` | the kernels of one shell, one line, one solid (ports of the reference crate) |
| `model/` | `ShellPart`, `SolidPart` (the elements of one kind in a model, their unknowns, supports, ties and loads), `MixedHybridModel` (the `FormulationModel` of a snapshot: reads the cards as their other families do, numbers the entities, answers the ties, the determined grid components, the derived supports, the affine fields, the grid point stresses of the solids and the pressures given by grid points) |
| `families/` | the element families of the cards (`MixedHybridLineFamily`, `MixedHybridShellFamily`, `MixedHybridSolidFamily`), thin: they call the model and carry the element matrices to the displacement systems (`LocalMap`) |

The shell and solid parts keep the element matrices and kernels they build, within a memory budget (each facet of
an element then reuses them): a shell of degree 4 takes a few milliseconds to build, a hexahedron about 40 ms (native;
about twice that in the WebAssembly build). A kept kernel built without the mass gets it when an analysis asks for it
(`VolumeKernel::withMass`, a third of a build) instead of being built again. The functions of a solid depend only on
the relative order of its grid IDs and are kept per order.

The families offer `Facet::Derivatives`: the design module differences their facets on the snapshots of the design
point moved by one field, each building its own model of the formulation. Beside the ties and supports, the model
answers what the paths around the elements ask: the elements of an entity (`entityElements`, for the external units),
the grid points whose positions an element reads (`readGridRows`, two rings of elements: the shape derivatives), and
what the elements of a set of grid points read around them (`describeSurroundings`, for the keys of the condensation
cache) (docs/mixed-hybrid-formulation-spec.md sections 5.11 to 5.15).

Membrane-only and bending-only shell properties are taken (`SectionParts`): a membrane alone without offset has no
moments, no transverse shear and no rotation about its edges, a bending alone no membrane stiffness. The model holds
at zero the entity DOFs that no element stiffens (`ShellPart::unstiffenedUnknowns`: the shear and rotations of the
edges of membranes, the singular directions of the displacement functions, as the singularity processor does at the
grid points); two shells on the same corner grids are refused.

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
  `BarStations` reads the output stations the `Bar2Stations` cards give to bars (the snapshot's `barStations` table):
  a `Bar2` with a card is written in the station layout in every subcase (`ElementFamily::stationRows`,
  `recoverStationRows`: its end A, its stations as fractions of its length, its end B, in the table of the bars under
  distributed loads), a station beyond its end is refused and a card that names no `Bar2` is a warning.
  `bars_beams/beam2/` reads a BeamProp with its stations and recovers at them.
- `shells/ShellSection`, `ShellMaterial`, `Laminate`: the sections of every shell (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D;
  CompProp and CompPropGlobal as their equivalent ShellProp), in the material frame, turned into the element frame by each element;
  `ShellRecovery` the output rows (the force table with its moment sign convention, the ply tables and failure
  indices); `ShellTemperature` the temperature change of a shell, `ShellPressure` the direction of a FacePressureLoad.
- `shells/shellQuad8/`, `shells/shellTria6/`: the higher-order shells, flat or curved, `MindlinShell` on
  `HigherOrderShapes`, with the facets shared in `HigherOrderShell` and the geometry in `HigherOrderGeometry`: the
  elements of [docs/shell-quad8-tria6-formulation.md](../../docs/shell-quad8-tria6-formulation.md), the degenerated
  (continuum-based) shell of Ahmad, Irons & Zienkiewicz (1970) with the conventions of the `ShellQuad4` and of the
  `ShellTria3`: the director of each grid point is the normal it shares with the neighbouring elements of the four
  cards or is given by a ShellNormal card (`ShellDirectors`, below), the rotation about it is held by the drilling
  term of the `ShellQuad4` (c_d = 10⁻⁶ × PARAM `ShellDrillingStiffness`, with the full rule), and the dilatation of a
  free thermal strain is the homothety of the element, without stress (a section without shear material keeps the
  transverse shear of its bending material: section 6 of the document says why it does not take c_K). Against locking the `ShellQuad8` has uniform reduced integration (2 × 2: Zienkiewicz,
  Taylor & Too 1971; the S8R of CalculiX) and the `ShellTria6` tied covariant transverse shears and membrane strains
  with the seven-point rule (Lee & Bathe 2004): with the strains of its displacements and three points it locked (a
  clamped plate of 8 × 8 cells 0.85 to 0.90 of its deflection, the pinched hemisphere 0.38 to 0.44; 1.02 to 1.03 and
  0.93 to 0.95 now). The linear shell strains, linear through the thickness, with the bending correction of Sanders
  (1959: a uniform dilatation bends nothing), on the section's resultants (MID1-MID4, CompProp, CompPropGlobal:
  Büchter & Ramm 1992; Simo & Fox 1989); on a flat element the isoparametric membrane and Reissner-Mindlin plate of
  the shape. The mass and the pressure loads with the full rule (3 × 3, seven points) and HRZ lumping,
  FacePressureLoad normal to the surface at each point, ZOFFS along each node's director, the stresses and forces of
  the centre and corners in the tangent frame of each place (x the element x axis, along G1-G2, projected on it); the
  `ShellQuad8` recovers them from the strains at the four points of its reduced rule (Barlow 1976), the `ShellTria6`
  from its tied strains. An element turning by more than 45° from the plane of its corners is refused. Section 7 of
  the document gives the measured results (bending, torsion, plates, the barrel vault, the pinched hemisphere, the
  twisted beam, the free thermal expansion, frequencies) and section 8 the check values;
  `checks/shells/HigherOrderShellChecks.cpp` holds them.
  `shells/shellQuad4Drilling/`, `shells/shellTria3Drilling/`: the ShellQuad4Drilling and ShellTria3Drilling,
  families implemented directly by `ShellQuad4DrillingElement` and `ShellTria3DrillingElement`, with the
  drilling membranes `ShellQuad4DrillingMembrane` and `ShellTria3DrillingMembrane` (below).
  `shells/shearPanel4/`: the shear panel (`ShearPanel4Panel`).
- `shells/shellQuad4/ShellQuad4*`, `shells/shellTria3/ShellTria3*`: the ShellQuad4 and ShellTria3, the elements of
  [docs/shell-quad4-formulation.md](../../docs/shell-quad4-formulation.md) and
  [docs/shell-tria3-formulation.md](../../docs/shell-tria3-formulation.md), from the methods of the open literature
  those documents list. `ShellQuad4`: a shell with a director at each node (Ahmad, Irons & Zienkiewicz 1970), a local
  frame at each of its 2 × 2 points, four enhanced parameters on the membrane strains and four on the curvatures
  (Pian & Sumihara 1984, Simo & Rifai 1990), transverse shear tied at the mid-sides (Dvorkin & Bathe 1984) with the
  residual bending flexibility of MacNeal (1978), a stabilised rotation about the normal (Hughes & Brezzi 1989), the
  rotary inertia in the coupled mass; a cantilever strip gives the deflection of the beam with one element.
  `ShellTria3`: the same element on three nodes, so that two triangles on the grid points of a quadrilateral
  behave closely to it: the same kinematics with a director at each node (`DirectorShellKinematics.hpp`), no
  enhanced strains, the covariant transverse shears tied at the mid-sides (a mean shear and one rotational part: Lee
  & Bathe 2004), the factors of the `ShellQuad4` on the widths of the parallelogram the triangle is half of, the same
  drilling term and mass. Section 6 of its document gives the measured closeness (bending, torsion, plates, curved
  shells, frequencies) and its limit: the constant membrane strains of a triangle bend in their plane only with
  several elements across the depth. `EdgeShearShellSection.hpp` holds what the two kernels share (the section at a
  point, the shear factors, the constants).
  The directors of both elements are the normals of their grid points, shared between the neighbouring `ShellQuad4`,
  `ShellTria3`, `ShellQuad8` and `ShellTria6` whose normals there are, in direction, within PARAM `ShellNormalAngle`
  (20° by default) of each other
  (`ShellDirectors`: Hughes & Liu 1981, Bathe 1996; smooth groups, so that a fold, a junction or a stiffener
  on a skin keeps one normal per side, and a flat mesh the elements' own; a ShellNormal card gives a grid point
  its normal, or none with a zero vector). The rotation of a grid point about its
  normal then moves nothing in any of its elements, and a curved surface no longer softens with the refinement of
  its mesh (a twisted beam of `ShellTria3` gave 1.19, 1.38 and 1.89 times its tip displacement at 12 × 2, 24 × 4 and
  48 × 8 cells with each element's own normal; 0.978, 0.990 and 0.995 with shared normals). That rotation is held by
  the drilling term, c_d = 10⁻⁶ × PARAM `ShellDrillingStiffness` (100 by default; 0: no term, and the automatic
  constraint of singular DOFs removes the rotation where nothing else holds it). It is a stabilisation: an inertia or
  a moment about the normal on one grid point is held by it alone, and a modal analysis points out the
  PointMassInertia in that case (`inertia-on-shell-normal`), and a static or a modal analysis the solutions whose
  strain energy is held by the drilling term for more than 1 % (`stabilisation-carries-load`: a bar attached to a
  shell at one grid point, a moment about the normal on one grid point).
  The thermal loads take the dilatation of the free strain as K q, q the homothety of the element: a warped mesh,
  or one whose normals are shared, expands freely under a uniform temperature rise (Sanders 1959: a uniform
  dilatation bends nothing).
  A section without `MID3`, or a laminate whose plies give no G1Z and G2Z, has no transverse shear flexibility: the
  kernels take a shear stiffness 10⁴ times the bending one (section 4 of the ShellQuad4 document).
  On the roof of Scordelis and Lo (sag of the free edge over 0.3024) the `ShellQuad4` gives 1.012, 1.002 at 8², 16²,
  and two `ShellTria3` per cell 0.883, 0.965.
  `PlateConstitutive` reads a section (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D; CompProp and CompPropGlobal through their equivalent section,
  this project's choice) into the matrices of the kernels; `ShellCentreRecovery` the centre recovery as an affine map
  of the local DOFs; `ShellScalar` the small matrices templated on the scalar. `EdgeShearShellFamily` (with
  `EdgeShearShellFamily.ipp`, `EdgeShearShellRecovery.ipp`, `EdgeShearShellDerivatives.ipp`, instantiated in
  `ShellQuad4Element.cpp` and `ShellTria3Element.cpp`) gives both their facets: the sections, the material axis
  (THETA from G1→G2, MCID projected), ZOFFS through `ShellNodeFrames`, the rows of `ShellRecovery` at the centre in
  the reference frame (x along G1→G2), the rows of the four corners of a `ShellQuad4` after them when the output
  is asked with its corners (`RecoveryOutput::hasCorners`, the `CornerResults` describer of a request:
  `ShellQuad4::cornerRecovery`, the values of the 2 × 2 integration points with the enhanced strains read at the
  corners on the bilinear surface through them; the recovery by blocks of cases takes their slopes too), the plies
  of a laminate. Its header lists the choices of
  this project (an MCID angle measured and applied in the element frame, the offset in the thermal loads, the
  thickness of a section without MID1 kept, the coupling and the thermal strains in the recovery, the differential
  stiffness, the rotation about the normal).
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
| SolidPyramid, 5 and 13 nodes | collapsed coordinates, 2 × 2 × 2 and 3 × 3 × 3 integration. The 13-node functions match the 6-node triangle on each side and the 8-node quadrilateral on the base | none |

The default solid mass uses the HRZ diagonal (Hinton, Rock & Zienkiewicz 1976) adjusted to preserve mass and centre of gravity.
The consistent matrix is retained if the positive diagonal correction does not converge. CoupledMass > 0 selects the consistent matrix directly. Validated (a
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
| `ShellQuad4`, `ShellTria3` | the membrane resultants (Nx, Ny, Nxy) the element recovers at its centre, on the in-plane gradients of u, v, w of its integration points (∫ ∇Nᵀ·N·∇N dA), ZOFFS by ShellNodeFrames | `shells/ShellDifferential.hpp`, `ShellQuad4`, `ShellTria3`, `EdgeShearShellFamily.ipp` |
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
first derivatives). Their thickness curvature propagates the exact chain rule through the same kernels with
`SecondOrderScalar`: value, first derivative and second derivative at the same thickness, with no subtraction or
step size. This covers stiffness, mass, recovered rows and fixed-preload geometric stiffness. The rows below name
the fields.

`Bar2` and `Beam2` also provide first, second and mixed sizing rates of stiffness, mass, linear recovery and
fixed-preload geometric stiffness through `bars_beams/BeamFieldDerivatives`. `BeamSizing` differentiates the
integrated flexibility and end-release condensation; `BeamSizingRecovery` carries the same rates through station
resultants. `MixedScalar` holds two first partials and their mixed partial. Section and material directions,
supported fields and numerical fallbacks are documented in [bar and beam derivatives](../../docs/beam-derivatives.md).

`ElementLoadContext`, `loadFieldDerivative` and `loadFieldMixedDerivative` extend the derivative facets to applied
element loads. The bar/beam implementation differentiates thermal strain, imposed elongation and equivalent nodal
line loads, including end-release condensation and the load transform. Static gradient assembly uses these rates
before numerical differences; static Hessian blocks add the diagonal and mixed applied-load curvature.

The optional `NodalDisplacements::lineLoads` recovery context also carries those loads into the ordinary bar/beam
result layouts and their first, second and mixed field rates. The end-force map subtracts the condensed fixed-end
loads and section equilibrium includes the loads on the left span. Displacement slopes and their field rates stay
independent of load values, so their existing caches remain valid across subcases.

| What | Facet | Families |
|---|---|---|
| The form of each output column in u (`ColumnForm`: Linear, PlaneVonMises of three Linear columns, Other) | `outputColumnForms`, `layerOutputColumnForms` | the shells (`shellOutputColumnForms`, `laminateOutputColumnForms`, the ply states all Linear), the rods, `Bar2`, `Beam2`, `ShearPanel4`, the solids (`solidOutputColumnForms`: the six components Linear), the springs and fasteners (every column Linear but the `Weld`'s stresses, whose axial stress alone is: `core/LinearRowRecovery`) |
| The recovery matrix of an element (`RecoverySlopes`: the slope of every row and column with respect to its DOFs) | `recoverySlopes`, `layerRecoverySlopes` | `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling` (the plate built once, `shellRecoverySlopes`), `Rod2`, `Rod2InlineSection`, `Tube2` (closed form, `rodRecoverySlopes`), `ShellQuad8`, `ShellTria6` (`higherOrderShellSlopes`), `Beam2` (one end-force map), `ShearPanel4` (its mean flow), the solids (the operators of their formulation, `SolidElementKernel::recoverySlopesOfOutputs`), the springs (`SpringDofProperty` to `SpringScalarInline`, `Bushing`, `Bushing1D`, `Gap2`: their maps K·s, c·H) and fasteners (`Fastener`: diag(K)·H·M; `Weld`: its end forces' map K·T·M through the forces at A' and B'), from one resolution, the piercing points interpolated once; the others' slopes from recoveries at unit displacements |
| The slopes of several outputs at once (the vectorised recovery) | `recoverySlopesOfOutputs` (element and layer outputs, `RecoveryOutput`) | `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`, `ShellQuad8`, `ShellTria6` (one plate and one set of place states for every output and ply table, `shellRecoverySlopesOfOutputs`, the rows without their principal values, `ShellRowValues::LinearOnly`), the solids (one formulation), the springs and fasteners (one resolution, `linearRowSlopesOfOutputs`); the others each output by `recoverySlopes` (the `Bar2`'s from one end-force map, `BeamEndForces`). The shells form the place states' parts that do not depend on the displacements once for every unit displacement (`ShellQuad4DrillingPlate::PlaceOperators`, `ShellTria3DrillingPlate::CentreOperators`, `MindlinShell::PlaceOperators`: the bits of the states made one by one) |
| The rows of several outputs from one formulation (the rows at rest of a block of cases, the outputs of a case recovered alone) | `recoverOutputs` (by default `recover` or `recoverLayers` for each output) | the shells (one resolution, plate and set of place states per element: `resetShellRecoveryBlocks`, `appendShellOutputRows` with `ShellRowValues::All`; the `ShellQuad4` and `ShellTria3` one centre recovery operator), the solids (one formulation, `SolidElementKernel::recoverOutputs`); each output's rows those of its recovery alone, bit for bit |
| The columns that are not Linear completed from the Linear ones (the vectorised recovery), and from how many cases a block pays for the slopes | `completeRecoveredRows`, `vectorisedRecoveryMinCases`; the ply tables `completeRecoveredLayerRows`, `vectorisedLayerRecoveryMinCases`, `vectorisedLayerSource` | the shells above (`completeShellRows`, 8 cases; their ply tables `completeLaminateRows`, the failure indices from the ply states, `completeLaminateFailureRows`, 4), `Bar2` (its extremes and margins, 8), `Beam2` (its extremes and the margins at one station, 8), `ShearPanel4` (its largest shear and margin from its mean flow, 8), the solids (`completeSolidRows`, 4: their principal values by Jacobi sweeps of four rows side by side, `principalStatesOf`, the bits of one row's), `Rod2`, `Rod2InlineSection`, `Tube2` (their margins, `completeRodRows`, 4), the springs (nothing to complete, `SPRING_VECTORISED_RECOVERY_MIN_CASES`, 4) and fasteners (`FASTENER_VECTORISED_RECOVERY_MIN_CASES`, 2; the `Weld`'s stresses made from its forces, `vectorisedSource`, `completeRecoveredRowsFromSource`) |
| dK/dv, dM/dv of a designed field | `stiffnessFieldDerivative`, `massFieldDerivative`; several fields of one element at once `stiffnessFieldDerivatives` (by default one by one) | the `ShellProp` T of `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling` whose corners follow T; the A and J of `Rod2` (`RodProp`) and `Rod2InlineSection`; the AnisoMat2D stiffness terms (`stiffness`, named by their MID: `DesignedField::id`, `item`) of the `ShellProp`'s MID1 to MID4 for `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`, `ShellQuad8`, `ShellTria6` (`ShellSectionRate`; no mass: the terms change none), a MID4 at a coupling zero at the point too but on the drilling membranes; the ply thicknesses of a CompProp or CompPropGlobal (`ply_thickness`, the item the card's ply) for the six shells (`LaminateRate`, `laminateSectionRate`; the mass of the ShellQuad4, ShellTria3 and their drilling variants) |
| The ply rows' rates in a designed field | `layerRecoveryFieldRates`, several fields `layerRecoveryFieldRatesOfFields` | a ply thickness of the shells' laminates: the ply stresses, strains, failure and bond indices, ply states (`appendLaminateRowRates`) |
| dK_d/dv at a fixed preload (the buckling factors' explicit parts) | `differentialStiffnessFieldDerivative` | the shells above in their AnisoMat2D terms, and the T of `ShellQuad4`, `ShellQuad4Drilling`, `ShellTria3`, `ShellTria3Drilling`; none under a temperature change of the preload; `Rod2` area and torsion rates, including fixed initial strains |
| The gradient of φᵀ·K_d(u)·φ in the preload u, along a mode φ (the buckling factors' preload part) | `differentialStiffnessPreloadGradient` (false by default: the caller forms K_d of each unit displacement) | the shells and solids of `MixedHybridFormulation` (one pass per element: K_d linear in the stresses of the preload) |
| Mixed d2K_d/dp dq at fixed preload | `differentialStiffnessFieldMixedDerivative` | thickness of the quadrilateral and triangular shells above; `Rod2` area/torsion zero curvature; an inert or unaffected field gives a known zero |
| The rows' rates with respect to a designed field at fixed u (the explicit parts) | `recoveryFieldRates`; several fields at once `recoveryFieldRatesOfFields` | the same fields and families (the AnisoMat2D terms not under a temperature change) |
| d²K/dv², d²M/dv² (the Hessian blocks) | `stiffnessFieldSecondDerivative`, `massFieldSecondDerivative` (true with nothing appended: zero) | the same fields and families (the rods' all zero) |
| The slopes of the rows' rates, ∂S/∂v (the mixed second derivatives), and the rows' second rates at fixed u | `recoverySlopeRates`, `recoveryFieldSecondRates` | the same fields and families, not under temperatures (the columns that are not Linear NaN: not formed; a von Mises from its components) |

The four scalar spring families give exact field derivatives for their spring rate: `dK/dk = s s^T` and
`d(force)/dk = s^T u`. The three families with a stress coefficient also give
`d(stress)/dk = S s^T u` and `d(stress)/dS = k s^T u`. Their same-field second derivatives vanish.
These rates feed static, modal and buckling sensitivities through the common element facets.
`Bushing1D` also supplies exact rates for its axial stiffness, lumped mass, stress factor and strain factor.
Its seven stress and strain table components are available to optimization responses as item codes 2 through 8.
The four scalar mass families supply `dM/dm = s s^T` and a zero same-field second derivative.

`Bushing` supplies exact stiffness-component, mass and recovery-coefficient rates. `Fastener` differentiates its
six spring rates and mass before carrying the matrices through its piercing interpolation. `Gap2` differentiates
its closed, open and transverse spring rates within the initial branch; changing the opening through zero is a
branch change, outside those formulas. Optimization force items are mapped for `Fastener` (2 to 7) and `Gap2`
(2 to 9); `Fastener` stress items (2 to 7) hold the same forces and moments.

`PointMassMatrix` differentiates each of its 21 symmetric matrix terms before rotation to the displacement frame.
`PointMassInertia` differentiates the rigid-body matrix in mass, all three offsets and all six inertia terms,
including the parallel-axis terms and translation/rotation coupling. Its offset second derivatives and mixed
mass/offset and offset/offset derivatives retain those terms. Rectangular input and displacement frames, and the
absolute centre-of-gravity convention, follow the same transformations as the element's mass calculation.

The optional `stiffnessFieldMixedDerivative`, `massFieldMixedDerivative` and `recoveryFieldMixedRates` facets give
second derivatives in two fields. A known zero is returned explicitly; an unavailable formula returns `false`.
Scalar springs and axial bushings include the stiffness/stress-coefficient product. The six-component bushing
includes the corresponding stiffness/recovery-coefficient products; fastener and gap spring rates have zero mixed
field rates at fixed displacement. These formulas are implemented in C++ and are shared by native and WebAssembly builds.

**Checks by ranges**: a family whose check is row by row says so (`isCheckedByRow`) and checks a range of rows
(`checkRows`), so that `assembly::ElementSet::check` checks a long batch by ranges on the pool: the `ShellQuad4` and
`ShellTria3` families (and so `ShellQuad4Drilling`, `ShellTria3Drilling`); the others check their batch whole.

Shape variables (DesignNodeMove) need nothing more from a family either: `design/sensitivity/ShapeElements` finds the elements
whose grid points (those of the DOFs of their matrices, and the orientation point G0 or GO) a shape moves, and the same
facets run on snapshots whose positions (`xyzGlobal`) are moved along the basis vector by ±h, ±h/2; a family whose
computations read no position (the scalar springs and masses) gives exact zeros. What a family reads from a table
O_o_Reader derives from the positions (the piercing points of Fastener and Weld, `fastenerPiercings`), and the
footprints the snapshot builds on them, cannot follow a move: such a shape is refused by `design/model/FieldTargets`.
A designed FastenerProp or WeldProp D does not move the footprint either, which stays that of the D of the model
(for a Weld D still sizes the beam).

## Element status

| Card | Facets written | Not yet supported |
|---|---|---|
| `SpringDofProperty`–`SpringScalarProperty` | stiffness, recovery (force, stress), derivatives | |
| `SpringScalarInline` | stiffness, recovery (force), derivatives | |
| `Bushing` | stiffness, mass (`BushProp` M), recovery (forces; stresses and strains by the coefficients of `RCV`), derivatives | curvilinear `CID`/`OCID` |
| `Rod2`, `Rod2InlineSection`, `Tube2` | stiffness, mass (lumped, coupled), differential stiffness (axial force), thermal loads, recovery (forces, stresses, strains, margins) | tapered `Tube2` |
| `PointMassMatrix`, `PointMassInertia` | mass, derivatives | curvilinear `CID` |
| `MassDofProperty`–`MassScalarInline` | mass, derivatives | |
| `ShellQuad4` | the element of docs/shell-quad4-formulation.md: stiffness (ShellProp with IsoMat, AnisoMat2D, OrthoMat2D: MID1-MID4, MID2 = −1, 12I/T³, TS/T; CompProp, CompPropGlobal as their equivalent section; ZOFFS, T1-T4, THETA/MCID; the normals of the grid points shared within ShellNormalAngle, a drilling term scaled by ShellDrillingStiffness), mass (lumped on the translations, or coupled with the rotary inertia; NonStructMass), loads (FacePressureLoad and ShellPressureLoad with corner values and direction; thermal: the mean temperature change and the PlateTemperatureLoad gradient), recovery (in the user frame: stresses, strains, curvatures, forces at the centre, and at the four grid points with the `CornerResults` describer of the request, from the values of the 2 × 2 points; plies and failure indices at the centre), differential stiffness (its recovered membrane resultants), derivatives (complex step) | the moment of a corner along a direction of constant curvature is that of the centre (section 12 of the document); the derivatives of the corner rows |
| `ShellTria3` | the element of docs/shell-tria3-formulation.md, as `ShellQuad4`; recovery at the centre only (its strains are constant over the element) | |
| `ShellQuad8`, `ShellTria6` | the elements of docs/shell-quad8-tria6-formulation.md, as `ShellQuad4` (flat or curved, every edge node), recovery at the centre and corners | an element turning by more than 45° |
| `ShellQuad4Drilling`, `ShellTria3Drilling` | a membrane with a drilling DOF (`ShellQuad4DrillingMembrane`, `ShellTria3DrillingMembrane`) with the bending of `ShellQuad4DrillingBending` and of the edge-shear triangle (the ShellQuad4Drilling recovered at the centre and corners); a warped ShellQuad4Drilling formed flat in its mean plane and joined to its grid points by rigid offsets along the normal (`ShellQuad4DrillingWarpOffsets`, Taylor 1988), in every facet but the mass (on the grid points) | membrane-bending coupling |
| `ShearPanel4` | stiffness (ShearPanelProp, IsoMat, F1, SensitivityPerturbation), mass, recovery (stresses, strains, forces) | warped panels |
| `Bar2` | stiffness (BarProp, IsoMat: A, I1, I2, I12, J, K1, K2; X1-X3 or G0, OFFT, WA, WB, PA, PB), mass (lumped, coupled, NonStructMass), loads (BeamDistributedLoad, every type and scale), differential stiffness (the axial force), thermal loads, recovery (forces at A and B; stresses and strains at C-F, axial, extremes, margins; the station layout, at the ends and at the stations of a `Bar2Stations` card, for a bar with that card or under BeamDistributedLoad); `BarShapeProp` from the sections table | materials other than IsoMat; a differential stiffness with offsets under the default `OffsetDefinition` |
| `Beam2` | as `Bar2` with BeamProp stations (taper, intermediate stations, SO), recovery at the stations, the torsional inertia ρ·(I1 + I2) in the mass; a BeamProp's CW is not used without SA, SB | `BeamLumpedAreaProp`, warping (SA, SB: the warping DOFs), the shear relief of taper S1, S2, NSI; as `Bar2` (no `Bar2Stations`: its stations are those of its property) |
| `SolidHexa` | stiffness (8 nodes: the default incompatible modes, or ISOP = FULL; 20 nodes: 3 × 3 × 3, or 2 × 2 × 2 with ISOP = REDUCED), mass (moment-preserving lumping, coupled), loads (FacePressureLoad on a face, corner pressures; thermal loads of the grid point temperatures, IsoMat A, AnisoMat3D A1-A6), recovery (centre and corners: stresses with the thermal stress subtracted, total strains, principal, von Mises, pressure, in CORDM); SolidProp, IsoMat, AnisoMat3D | partial mid-side nodes, CORDM −1/−2, other IN/ISOP, STRESS = GAUSS |
| `SolidPenta` | as `SolidHexa` (the isoparametric wedge, 6 and 15 nodes; the 15-node not validated) | as `SolidHexa` |
| `SolidTetra` | as `SolidHexa` (4 nodes constant strain, 10 nodes five points); with 4 nodes also IN = TWO, ISOP = FULL or REDUCED and STRESS = GAUSS (its one Gauss point the centroid) | as `SolidHexa`, but STRESS = GAUSS with 4 nodes |
| `SolidPyramid` | as `SolidHexa` (collapsed hexahedron, corner stresses extrapolated; not validated) | as `SolidHexa` |
