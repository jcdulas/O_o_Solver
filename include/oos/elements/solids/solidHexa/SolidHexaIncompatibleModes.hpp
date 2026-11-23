#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/solids/SolidResponse.hpp"

#include <span>

namespace oos::elements
{

// Default HEX8 with the EAS-21 strain interpolation of Andelfinger & Ramm (IJNME, 1993,
// doi:10.1002/nme.1620360805). The 24 nodal translations give the compatible trilinear field.
// For each cyclic permutation (a,b,c) of the parent coordinates, enhance the aa strain by
// a, ab, ac and the ab shear by a, b, ac, bc. Map these tensors with the centre inverse Jacobian
// and scale them by det J0 / det J. Each mode then integrates to zero against a constant stress,
// including on distorted bricks. The 2 x 2 x 2 rule samples all modes independently.
//
// Eliminate the 21 local coefficients from the elastic energy including thermal eigenstrain.
// The same solution supplies the nodal thermal loads and the total strain at the Gauss points.
// At the centre all enhanced modes vanish. Corner strains extrapolate the Gauss values.
// Positions, elasticity, thermal expansion and the returned operators use the material frame.
void buildSolidHexaIncompatibleModesResponse(std::span<const base::Vec3> positions, const Mat6& elasticity,
                                            const Voigt6& expansion, SolidResponseParts parts, SolidResponse& response);

} // namespace oos::elements
