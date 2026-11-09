#pragma once

#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

enum class QuadraticProgramStatus : std::uint8_t
{
    Optimal,
    Infeasible
};

// The solution of a quadratic program: x, its objective, the multiplier u_i ≥ 0 of each row (zero off the active set)
// and the active rows in the order they were added.
struct QuadraticProgramSolution
{
    QuadraticProgramStatus status = QuadraticProgramStatus::Optimal;
    std::vector<double> x;
    double objective = 0.0;
    std::vector<double> multipliers;
    std::vector<std::size_t> active;
    std::size_t iterations = 0;
};

// A strictly convex quadratic program solved by the dual active-set method of Goldfarb and Idnani ("A numerically
// stable dual method for solving strictly convex quadratic programs", Math. Programming 27, 1983), written here (no
// dependency):
//
//     minimise ½·xᵀ·G·x + aᵀ·x  subject to  C·x ≥ d,   G symmetric positive definite.
//
// It starts at the unconstrained minimum x = −G⁻¹·a and adds, one at a time, the most violated row, dropping the
// active rows whose multiplier would become negative, so that every iterate is dual feasible; no feasible start is
// needed, and an infeasible set of rows is detected (a violated row that no step can satisfy). The factors are those
// of the paper: G = L·Lᵀ, J = L⁻ᵀ·Q and the triangular R of the active normals (N = Q·[R; 0] with N in L⁻¹ space),
// updated by Givens rotations when a row is added or dropped.
//
// Thread-safety: stateless. Complexity: O(n²) per added or dropped row, O(n³) for the factorisation; budgeted.
class QuadraticProgram
{
public:
    // `g` is n × n and `c` m × n, row-major (n = a.size(), m = d.size()). Throws OptimisationError for inconsistent
    // sizes or a G that is not positive definite, MemoryBudgetError when the factors do not fit.
    [[nodiscard]] static QuadraticProgramSolution solve(std::span<const double> g, std::span<const double> a,
                                                        std::span<const double> c, std::span<const double> d,
                                                        base::MemoryBudget& budget);
};

} // namespace oos::design
