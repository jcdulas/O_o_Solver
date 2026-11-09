#pragma once

#include "oos/aero/trim/TrimCase.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <span>
#include <vector>

namespace oos::aero
{

// The linear trim of a static aeroelastic subcase, without redundant control surfaces: the equilibrium of the supported
// rigid-body motions (one equation per r-set DOF of FreeBodySupport) is linear in the trim variables,
//     Σ_x A_kx·u_x + b_k = 0,    k = 1 ... r,
// each AeroTrimLink replaces its dependent variable by −Σ C_i·u_i of the independent ones (aero::AeroLink: the column
// of the dependent added, times −C_i, to those of the independent), and the free variables are the unknowns of the
// square system
//     Â_F·u_F = −(Â_X·u_X + b),
// with X the fixed variables. A trim whose free variables are not as many as the supported DOFs is refused (more would
// be a redundant-surface trim, an optimisation problem this version does not solve). The system is solved by
// linalg::DenseLu on its rows and columns scaled to a largest magnitude of 1 (forces and moments, angles and
// accelerations are not of one size).
//
// Thread-safety: stateless.
class TrimSystem
{
public:
    // The pivot ratio of the scaled system below which the free variables are taken as unable to balance the supported
    // motions: the scaled system has terms of order 1, and a ratio near the rounding of a double (1e-16) is singular;
    // 1e-12 leaves four digits of margin for the rounding of the equations themselves.
    static constexpr double SINGULAR_PIVOT_RATIO = 1.0e-12;

    // The value of every variable of `trimCase`, in its order: the fixed ones as given, the free ones solved, the
    // linked ones from their AeroTrimLink. `equations` (r × settings): column x the equilibrium for a unit value of
    // variable x; `constant` (r values, or empty for zeros): what does not depend on the variables. Throws SolverError
    // when the free variables are not as many as the equations (naming them) or cannot balance them (a singular
    // system).
    [[nodiscard]] static std::vector<double> solve(const TrimCase& trimCase, const linalg::DenseMatrix& equations,
                                                   std::span<const double> constant, base::MemoryBudget& budget);
};

} // namespace oos::aero
