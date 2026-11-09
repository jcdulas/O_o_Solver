# `aero`

Aerodynamics of the static aeroelastic solution (SOL 144): boxes, doublet-lattice at zero frequency, splines, trim.

It may use `base`, `linalg`, `input`, `elements`, `assembly`, `reduction`. The trim solution itself is
`solutions/sol144/TrimSolution`, one layer up, because it needs the structural solver.

Folders: `geometry/`, `dlm/`, `splines/`, `trim/`, `outputs/`. Where to start reading: the folders in that order,
which is the order in which the pieces depend on one another.

## `geometry/`

- `AeroReference`: the AeroStaticParams (ACSID, RCSID, REFC, REFB, REFS, SYMXZ, SYMXY) read from O_o_Reader's
  `aeroReferenceTable`, with the frames of ACSID and RCSID from `coordsTable`.
- `AeroBoxes`: the boxes of the AeroLiftingPanel1 panels (`aeroBoxesTable`) in the aerodynamic frame: the quarter-chord line
  (the doublet line, at k = 0 the bound vortex) oriented along n × x, the control point at three quarters of the
  chord at mid span, the force point at the middle of the quarter-chord line. Other panels (AeroBody2 to AeroPistonPanel5) are
  refused.

## `dlm/`

- `DoubletLatticeSteady`: the normalwash factors D at k = 0, the horseshoe vortex of the vortex-lattice method with
  the Prandtl-Glauert stretch and the images of the symmetry keys; `solvePressures` on `linalg::DenseLu`.

## `splines/`

- `SurfaceSpline`: the infinite plate spline (Harder & Desmarais) of AeroSurfaceSpline1 and AeroSurfaceSpline4 (METH IPS, TPS), with the
  attachment flexibility DZ.
- `BeamSpline`: the beam spline of AeroBeamSpline2 and AeroBeamSpline5 (deflection, slopes when DTHX ≥ 0, twist when DTHY ≥ 0, DTOR).
- `SplineMatrix`: the splines of a model from `splinesTable`, `splineNodes` and the AeroBeamSpline2, AeroBeamSpline5 and AeroBoxList
  card tables: the plunge and slope of each box at its center from the motions of the grid points (GKA), and the box
  forces given back to the grid points by the transpose; FPS, RIS, AeroSplineEquation3, AeroFiniteSurfaceSpline6, AeroFiniteBeamSpline7, AeroUserSpline refused.

## `trim/`

- `TrimVariables`: the normalwash of a unit value of AngleOfAttack, SideSlip, RollRate, PitchRate, YawRate and of each AeroControlSurface (linear
  downwash, one component).
- `RigidStabilityDerivatives`: the rigid, unsplined stability and control derivatives of the trim listing, in the
  reference frame.
- `TrimCase`: the trim case of a subcase from `aeroTrimCasesTable` (Mach, Q, AEQR, the variables and their states) and
  the AeroTrimLink of its linked variables (u_D + Σ C_i·u_i = 0, given by the caller with the tables).
- `TrimSystem`: the linear trim, the equilibrium of the supported motions solved for the free variables, the links
  substituted.
- `ReferenceCoefficients`: the six coefficients of a resultant about the RCSID origin, the images of a half model
  added (shared with the rigid derivatives).
- `TrimCorrections`: the direct matrix input of a trim: the DownwashMatrix and AeroPressureMatrix in effect of O_o_Reader's
  `directMatricesTable` applied to the boxes, a DirectMatrix by the positions of the j set and a DirectMatrixAeroJ by box ID and component
  3, one column for every subcase. DownwashMatrix is a normalwash added to the downwash of the trim variables; AeroPressureMatrix is a set of
  pressure coefficients used as they are. A AeroWeightingMatrix or AeroWeightingFactor is accepted as the identity only; the refusals are named.
  `solutions::TrimSolution` solves these corrections as a fixed trim variable (an intercept term).

## `outputs/`

- `HingeMoments`: the hinge moments of the AeroControlSurface about their hinge lines.

The monitor points are `output/monitors/MonitorPoint1` (MonitorIntegratedLoad1, plus the whole-vehicle and hinge moment points) and
`output/monitors/MonitorPoint3` (MonitorNodeForceSum3, on the cut loads of `output/sections/CutLoads`).
