#pragma once

// ShellTria3 — three-node shell (CTRIA3, ShellProp, CompProp, CompPropGlobal; IsoMat, AnisoMat2D, OrthoMat2D):
//
//   | ShellTria3 | EID   | PID | G1 | G2 | G3 | THETA or MCID | ZOFFS |
//   |        | TFLAG | T1  | T2 | T3 |
//
// The formulation of NASTRAN-95's TRIA3 (ShellTria3: linear membrane and bending, constant assumed transverse shear
// with one extra parameter and the shear compliance Z, no drilling stiffness), with the facets of EdgeShearShellFamily
// (sections, offsets, mass, loads, centre recovery in the user frame, membrane-stress differential stiffness,
// complex-step design derivatives). Problems reported by check(): as the ShellQuad4 (ShellQuad4Element.hpp), with
// collinear grid points the degenerate geometry of a triangle.
//
// ShellTria3DrillingElement implements the separate shell family with a drilling rotation.

#include "oos/elements/shells/EdgeShearShellFamily.hpp"
#include "oos/elements/shells/shellTria3/ShellTria3.hpp"

namespace oos::elements
{

extern template class EdgeShearShellFamily<ShellTria3>;

// The element family of ShellTria3. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class ShellTria3Element final : public EdgeShearShellFamily<ShellTria3>
{
};

} // namespace oos::elements
