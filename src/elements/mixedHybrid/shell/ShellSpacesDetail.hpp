#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSpaces.hpp"

#include <cstddef>
#include <vector>

// The helpers that the spaces of the shell and its reference element share (private to this folder).
//
// Thread-safety: pure functions.
namespace oos::elements::mixedHybrid::shellspaces
{

// The points per direction of the rules used to build the spaces and their functionals (exact for every product met
// there: the rule of 10 points is exact to degree 19 per direction).
// Reference: tdnns/space.rs::build_rules
inline constexpr std::size_t BUILD_POINTS = 10;

// Σ_j c_j g_j, the zero coefficients skipped.
// Reference: tdnns/space.rs::combine
[[nodiscard]] Field combine(const std::vector<Field>& generators, const Vector& coefficients);

// An L2-orthonormal basis of the span of the fields on the reference domain of `cornerCount` corners: the eigenpairs
// (λ_k, v_k) of their Gram matrix above 1e-11 of the largest give the functions Σ_i v_ik g_i / √λ_k, in the order of
// the eigenvalues of jacobiEigen.
// Reference: tdnns/space.rs::orthonormal
[[nodiscard]] std::vector<Field> orthonormal(const std::vector<Field>& generators, std::size_t cornerCount);

} // namespace oos::elements::mixedHybrid::shellspaces
