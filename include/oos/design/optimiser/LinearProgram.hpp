#pragma once

#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

enum class LinearProgramStatus : std::uint8_t
{
    Optimal,
    Infeasible,
    Unbounded
};

// The solution of a linear program: x, cᵀx, and the multiplier λ_i ≥ 0 of each row of A (the Kuhn-Tucker condition
// c + Aᵀλ + ν = 0 holds with ν the multipliers of the bounds).
struct LinearProgramSolution
{
    LinearProgramStatus status = LinearProgramStatus::Optimal;
    std::vector<double> x;
    double objective = 0.0;
    std::vector<double> multipliers;
    std::size_t pivots = 0;
};

// A dense linear program solved by the two-phase simplex method with Bland's rule (Dantzig; Bland, "New finite
// pivoting rules for the simplex method", Math. Oper. Res. 2(2), 1977, which excludes cycling on degenerate vertices),
// written here (no dependency):
//
//     minimise cᵀx  subject to  A·x ≤ b  and  lower ≤ x ≤ upper  (finite bounds).
//
// The bounds are shifted (y = x − lower ≥ 0) and the upper ones become rows y ≤ upper − lower; a row with a negative
// right-hand side gets an artificial variable, and phase 1 minimises their sum before phase 2 minimises cᵀx. The
// tableau is dense ((m + n + 1) × (2n + m + artificials + 1) values), budgeted: the optimisers hand it their active
// constraints only.
//
// Thread-safety: stateless. Complexity: each pivot O(rows × columns); Bland's rule terminates.
class LinearProgram
{
public:
    // `a` is row-major, b.size() rows × c.size() columns. Throws OptimisationError for inconsistent sizes or infinite
    // bounds, MemoryBudgetError when the tableau does not fit.
    [[nodiscard]] static LinearProgramSolution solve(std::span<const double> c, std::span<const double> a,
                                                     std::span<const double> b, std::span<const double> lower,
                                                     std::span<const double> upper, base::MemoryBudget& budget);
};

} // namespace oos::design
