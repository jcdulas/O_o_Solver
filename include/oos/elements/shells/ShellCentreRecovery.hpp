#pragma once

// The centre recovery of the NASTRAN-95 shells (ShellQuad4, ShellTria3): the values each kernel recovers at the
// element centre, as an affine map of the element's local DOFs.
//
// The recovery of a linear element is linear in its displacements, with an offset from the thermal strains. Each
// kernel therefore builds the map once per element, a matrix R of CENTRE_RECOVERY_ROWS rows and 6N + 1 columns (the 6N
// local DOFs, then the thermal column), and the values are R·[q; 1]. The same matrix serves a recovery, the recovery
// slopes (its first 6N columns) and, in complex numbers, the design rates.
//
// The rows, in the element frame, with the sign conventions of ShellRecovery.hpp (ε(z) = ε_m − z·κ):
//   0-2   membrane forces (Nx, Ny, Nxy) = ∫σ dz, the thermal stress removed
//   3-5   moments (Mx, My, Mxy) as the force table writes them, M = D·κ (= −∫σ z dz)
//   6-7   transverse shear forces (Qx, Qy) per unit length
//   8-10  stresses (σx, σy, τxy) at the fibre Z1;  11-13 at Z2
//   14-16 membrane strains (εx, εy, γxy), total (the thermal strain included)
//   17-19 curvatures (κx, κy, κxy), total

#include "oos/elements/shells/ShellScalar.hpp"

#include <array>
#include <cstddef>

namespace oos::elements
{

inline constexpr std::size_t CENTRE_RECOVERY_ROWS = 20;

namespace centre_row
{
inline constexpr std::size_t MEMBRANE_FORCE = 0;
inline constexpr std::size_t MOMENT = 3;
inline constexpr std::size_t SHEAR_FORCE = 6;
inline constexpr std::size_t STRESS_FIBER1 = 8;
inline constexpr std::size_t STRESS_FIBER2 = 11;
inline constexpr std::size_t MEMBRANE_STRAIN = 14;
inline constexpr std::size_t CURVATURE = 17;
} // namespace centre_row

template <typename T, std::size_t DOFS>
using CentreRecoveryOperator = ScalarMatrix<T, CENTRE_RECOVERY_ROWS, DOFS + 1>;

// One linear functional of the local DOFs plus a constant: the row of a CentreRecoveryOperator in the making.
template <typename T, std::size_t DOFS>
using AffineRow = std::array<T, DOFS + 1>;

// target += scale · source (affine rows of N = DOFS + 1 entries).
template <typename T, std::size_t N>
void addScaled(std::array<T, N>& target, const std::array<T, N>& source, const T& scale)
{
    for (std::size_t k = 0; k < N; ++k)
        target[k] += scale * source[k];
}

template <typename T, std::size_t R, std::size_t N>
void setRow(ScalarMatrix<T, R, N>& recovery, std::size_t row, const std::array<T, N>& values)
{
    for (std::size_t k = 0; k < N; ++k)
        recovery(row, k) = values[k];
}

} // namespace oos::elements
