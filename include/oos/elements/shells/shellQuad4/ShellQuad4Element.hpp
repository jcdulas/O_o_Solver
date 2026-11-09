#pragma once

// ShellQuad4 — four-node shell (CQUAD4, ShellProp, CompProp, CompPropGlobal; IsoMat, AnisoMat2D, OrthoMat2D):
//
//   | ShellQuad4 | EID   | PID | G1 | G2 | G3 | G4 | THETA or MCID | ZOFFS |
//   |        | TFLAG | T1  | T2 | T3 | T4 |
//
// The formulation of NASTRAN-95's QUAD4 (ShellQuad4: degenerated shell with nodal normals, reduced in-plane shear,
// assumed edge transverse shear with its shear-correction factors, no drilling stiffness), with the facets of
// EdgeShearShellFamily (sections, offsets, mass, loads, centre recovery in the user frame, membrane-stress
// differential stiffness, complex-step design derivatives). Problems reported by check(): an unknown grid point, a
// grid point named twice, a property or material the shells cannot take (ShellSection), a degenerate geometry (parallel
// diagonals, a corner of 0° or 180°, a Jacobian not positive at an integration point), a thickness that is not
// positive, an MCID normal to the element, an offset without MID1 and MID2.
//
// ShellQuad4DrillingElement implements the separate shell family with a drilling rotation.

#include "oos/elements/shells/EdgeShearShellFamily.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

namespace oos::elements
{

extern template class EdgeShearShellFamily<ShellQuad4>;

// The element family of ShellQuad4. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShellQuad4Element final : public EdgeShearShellFamily<ShellQuad4>
{
};

} // namespace oos::elements
