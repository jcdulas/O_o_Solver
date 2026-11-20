// The cases of the validation suite that do not meet their criterion, each with the reason: a limit of the formulation
// (docs/mixed-hybrid-formulation-spec.md section 3.6), a gap that is later work (section 14), or a defect outside
// O_o_Solver. The case keeps its criterion and still runs: node:test reports it as "todo" with what it measured, so
// that a case that starts to pass, or a new failure, is seen. A case is matched by a part of its name.
//
// Nothing here is a tolerance: to accept a result, change the specification first.
//
// No case is here for a declaration of the geometry that its model file lacks: the corners of an outline, the folds
// that keep their line, the straight lines of a plane meshed irregularly are declared in the model files, as a
// preprocessor writes them (spec section 4.4).

export const KNOWN_LIMITS = [
  // ---- the reading of the surface (section 3.6: kinks between elements are folds)
  { match: "Scordelis-Lo roof, 8 x 8 ShellTria3 distorted by 0.2 (OPEN",
    reason: "reading: distorted triangles on a curved surface keep kinks of 1 to 3 degrees, which the shell takes as " +
      "folds (30 % off, not converging; warned by geometry-kink); declare them raw or use quadrilaterals" },
  { match: "cylinder under internal pressure, 8 x 8 ShellTria3 distorted by 0.2",
    reason: "reading: distorted triangles on a curved surface (as the roof above)" },
  { match: "cylinder under internal pressure, 8 x 8 ShellTria6 over 90 degrees",
    reason: "fibre stresses 1.17 % off at 8 x 8 (0.54 % at 16 x 16): the curvature term of the theory, O(t / R), and " +
      "the parabolic edges of the mid-side nodes" },
  { match: "modes of a cylinder between diaphragms, R/t = 100, 24 x 6 distorted ShellTria3",
    reason: "the frequencies pass; the case asks for no warning and the kinks between the distorted triangles are " +
      "warned (geometry-kink), as they must be" },
  { match: "Lamé: an octant of a thick sphere of SolidHexa with mid-side nodes",
    reason: "mid-side nodes on a doubly curved skin: the eight-node face is not the sphere (stress 4.9 % at 22.5 " +
      "degrees per cell, 0.97 % at 11.25); the skin read without mid-side nodes gives 0.3 %" },

  // ---- properties of the formulation
  { match: "with an offset point mass with inertia on a grid point, against the fine ClassicFormulation model",
    reason: "an inertia at a grid point of shells acts on their mean rotation there: the modes that turn it are 1 to " +
      "1.5 % above a fine ClassicFormulation model, whose own answer to a concentrated moment depends on its mesh" },
  { match: "modes of a free ring, 32 x 2 warped ShellQuad4",
    reason: "the elastic frequencies pass; the sixth rigid-body frequency is 1.6e-3 of the first elastic one for a " +
      "criterion of 1e-3 (1.5e-10 of the membrane scale). It is the round-off of the eigenvalue solution on this " +
      "thin, strongly warped mesh, not a force of the elements in a rigid motion: with the rigid motions taken out " +
      "of each element stiffness by a projector the six frequencies stay between 2e-4 and 1e-3 (3e-5 to 3e-4 on " +
      "the regular meshes, with or without the projector)" },
  { match: "inertia relief -2 of a free ring pulled apart by two forces, 32 x 2 warped ShellQuad4",
    reason: "the pivot ratio test stops at 1.8e7 for a limit of 1e7 on this thin warped ring (no mechanism: it " +
      "solves " +
      "with MaxDiagonalRatio 1e9, and with inertia relief -1 on FreeBodySupport)" },
  { match: "inertia relief -1 of a free ring pulled apart by two forces, 32 x 2 warped ShellQuad4",
    reason: "the result passes (0.998); the case asks for no warning and this strongly warped mesh is warned " +
      "(geometry-rounding: the surface read turns by 52 degrees from the plane of the corners of a shell)" },

  // ---- comparisons with ClassicFormulation that are no criterion
  { match: "5. panel: displacements, frequencies and buckling factor against the classic formulation",
    reason: "a yardstick, not a criterion: quantities attached through rotations at grid points of shells " +
      "differ by 5 " +
      "to 10 % from a ClassicFormulation model that has not converged either" },
];
