#pragma once

#include <array>

namespace oos::elements::mixedHybrid
{

// The small fixed-size values of the shell kernel of the mixed hybrid formulation (document of the port:
// docs/mixed-hybrid-formulation-spec.md, section 3.1). They are plain arrays, indexed [i][j] as the formulas of the
// headers write them.
//
// Thread-safety: plain value types.

// A vector of the three-dimensional space, Cartesian components.
using SpaceVector = std::array<double, 3>;
// Two vectors tangent to the surface: a covariant or contravariant basis, or the local orthonormal frame (e1, e2).
using SurfaceFrame = std::array<SpaceVector, 2>;
// A 3 × 3 matrix by rows: a section stiffness (A, B, D) or a map between Voigt vectors.
using Matrix33 = std::array<std::array<double, 3>, 3>;
// A 2 × 2 matrix by rows: a transverse shear stiffness, a metric, components of a frame.
using Matrix22 = std::array<std::array<double, 2>, 2>;
// A Voigt vector of the membrane or bending part: (11, 22, 12), the meaning of the third term given where it is used
// (engineering shear 2 ε12 for a strain, the plain component for a force).
using Voigt3 = std::array<double, 3>;
// The value of a function of (ξ, η) and its derivatives (∂ξ, ∂η, ∂ξξ, ∂ξη, ∂ηη).
using ReferenceJet = std::array<double, 6>;
// The value of a function and its first derivatives (∂ξ, ∂η) at an edge point.
using EdgeJet = std::array<double, 3>;

} // namespace oos::elements::mixedHybrid
