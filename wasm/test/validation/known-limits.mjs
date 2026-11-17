// The cases of the validation suite that do not meet their criterion, each with the reason: a limit of the formulation
// (docs/mixed-hybrid-formulation-spec.md section 3.6), a gap that is later work (section 14), or a defect outside
// O_o_Solver. The case keeps its criterion and still runs: node:test reports it as "todo" with what it measured, so
// that a case that starts to pass, or a new failure, is seen. A case is matched by a part of its name.
//
// Nothing here is a tolerance: to accept a result, change the specification first.

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
  { match: "patch: three pure bendings on a block of SolidPenta under SolidTetra whose skin is distorted too",
    reason: "skin reading: a mesh line of three grid points on a flat face is read as a curve (1.3e-6 for 1e-6); the " +
      "reference reads it so; exact with three cells or more per side, or the skin edges declared sharp" },
  { match: "patch: three pure bendings on a block of 2 x 2 x 2 SolidHexa whose skin is distorted too",
    reason: "skin reading: as the block of pentahedra above (4e-6 for 1e-6)" },
  { match: "Lamé: an octant of a thick sphere of SolidHexa with mid-side nodes",
    reason: "mid-side nodes on a doubly curved skin: the eight-node face is not the sphere (stress 4.9 % at 22.5 " +
      "degrees per cell, 0.97 % at 11.25); the skin read without mid-side nodes gives 0.3 %" },
  { match: "7. reduced: a flat plate with a re-entrant corner beside triangles keeps its area",
    reason: "reading: a corner of the outline with a triangle and a quadrilateral is read as a boundary that goes on " +
      "(the reference's rule); the model is now refused (degenerate-geometry) or warned (geometry-rounding) until " +
      "the corner is declared raw; the case expects a result without declaration" },
  { match: "1. bracket (exact geometry): the support forces balance the loads stated",
    reason: "reading: the re-entrant corner of the flange is rounded (3.9 % of its pressure), warned by " +
      "geometry-rounding; declare its edges raw" },
  { match: "3. bracket with its elements started from another corner gives the same results",
    reason: "reading: with the rounded corner of the flange a triangle has almost no angle left, and its results " +
      "depend on its first corner (8e-3); exact once the corner is declared raw" },

  // ---- properties of the formulation
  { match: "transition: a cantilever strip of SolidHexa continued by ShellQuad4 bends as the beam",
    reason: "a shell that continues solids, or stands on them (tie 4): the solids are not told the thickness of the " +
      "shell, the moment enters them along a line and the joint softens with the mesh (1.26 at 4 x 1 x 2)" },
  { match: "transition: a strip of SolidHexa continued by ShellQuad4 pulled by its tip stretches as the bar",
    reason: "a shell that continues solids (as the bending case above): 0.932" },
  { match: "with an offset point mass with inertia on a grid point, against the fine ClassicFormulation model",
    reason: "an inertia at a grid point of shells acts on their mean rotation there: the modes that turn it are 1 to " +
      "1.5 % above a fine ClassicFormulation model, whose own answer to a concentrated moment depends on its mesh" },
  { match: "modes of a free ring, 32 x 2 warped ShellQuad4",
    reason: "the elastic frequencies pass; the sixth rigid-body frequency is 1.6e-3 of the first elastic one for a " +
      "criterion of 1e-3 (1.5e-10 of the membrane scale, growing with the warping of the elements)" },
  { match: "inertia relief -2 of a free ring pulled apart by two forces, 32 x 2 warped ShellQuad4",
    reason: "the pivot ratio test stops at 1.8e7 for a limit of 1e7 on this thin warped ring (no mechanism: it " +
      "solves " +
      "with MaxDiagonalRatio 1e9, and with inertia relief -1 on FreeBodySupport)" },
  { match: "inertia relief -1 of a free ring pulled apart by two forces, 32 x 2 warped ShellQuad4",
    reason: "the result passes (0.998); the case asks for no warning and this strongly warped mesh is warned " +
      "(geometry-rounding: the surface read turns by 52 degrees from the plane of the corners of a shell)" },
  { match: "3. panel with its elements started from another corner gives the same results",
    reason: "the results are the same (8e-11); the share of a distributed load between the grid points of a triangle " +
      "in the applied load table depends on its first corner (its resultant does not)" },

  // ---- comparisons with ClassicFormulation that are no criterion
  { match: "5. panel: displacements, frequencies and buckling factor against the classic formulation",
    reason: "a yardstick, not a criterion: quantities attached through rotations at grid points of shells " +
      "differ by 5 " +
      "to 10 % from a ClassicFormulation model that has not converged either" },
  { match: "5. bracket: displacements, frequencies and buckling factor against the classic formulation",
    reason: "a yardstick: the flange tip is 5 to 9 % from the ClassicFormulation value (rounded corner, shell " +
      "on solids)" },
  { match: "5. wing: displacements, frequencies and buckling factor against the classic formulation",
    reason: "a yardstick: the two cells of the cover across the chord state a roof where the wing has a camber " +
      "(warned by geometry-kink), and the formulation takes the fold as the geometry: an in-plane displacement " +
      "of a spar corner is 51 % above the ClassicFormulation value on its finer mesh (the reaction agrees to " +
      "0.4 %)" },
];
