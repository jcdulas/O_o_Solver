#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::linalg
{

// The pencil (K, M) of two sparse symmetric matrices assembled on one pattern and converted with one DOF selection
// (the order of the symbolic factor): K − σM is formed on that pattern at each shift (SymmetricCsc::combine), so the
// one symbolic factor serves every shift, and factorised by the supernodal driver in indefinite mode (Bunch-Kaufman
// pivots, delayed pivots). The mass products use the pool of the kernels.
//
// Invariants: K and M have the same structure; the diagonal of M is finite and not negative (a positive semi-definite
// M has a non-negative diagonal; the converse is not checked). The factor of the last shift is kept; the previous one
// and its shifted matrix are released before the next is made, so at most one factor is alive.
//
// Thread-safety: as ShiftedPencil. Ownership: references the symbolic factor, the matrices, the kernels and the budget,
// which must outlive it; owns the shifted matrix and the factor (budgeted).
class SparsePencil final : public ShiftedPencil
{
public:
    // Throws std::invalid_argument when the matrices differ in structure or are not of the dimension of the symbolic
    // factor; SolverError when a diagonal term of M is negative or not finite (the Lanczos method needs a positive
    // semi-definite mass). The pivoting of the options is replaced by PivotingMode::Indefinite.
    SparsePencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness, const SymmetricCsc& mass,
                 const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept override { return m_stiffness.dimension(); }
    // MemoryBudgetError or OperationCancelledError from the factorisation propagate.
    [[nodiscard]] ShiftFactorisation factorise(double shift) override;
    // Throws std::logic_error before the first factorisation; SingularityError on a singular factor.
    void solve(DenseMatrixView block) const override;
    void multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] double diagonalRayleighBound() const override { return m_diagonalRayleighBound; }

    // True when every off-diagonal term of M is zero (a lumped mass): the products with M then use its diagonal.
    [[nodiscard]] bool hasDiagonalMass() const noexcept { return !m_massDiagonal.empty(); }

    // The factorisations made so far (one per shift tried).
    [[nodiscard]] std::size_t factorisationCount() const noexcept { return m_factorisationCount; }
    // The factor of the last shift (for reports and tests); empty before the first factorisation.
    [[nodiscard]] const SupernodalFactor& factor() const noexcept { return m_factor; }

    // An upper bound of the bytes the pencil holds at once: the shifted matrix and one factor (before the growth of
    // delayed pivots).
    [[nodiscard]] static std::uint64_t estimateMemoryBytes(const SymbolicFactor& symbolic,
                                                           const SymmetricCsc& stiffness, std::size_t threadCount);

private:
    const SymbolicFactor& m_symbolic;
    const SymmetricCsc& m_stiffness;
    const SymmetricCsc& m_mass;
    const DenseKernels& m_kernels;
    FactorOptions m_options;
    base::MemoryBudget& m_budget;
    double m_diagonalRayleighBound = 0.0;
    std::size_t m_factorisationCount = 0;
    bool m_hasFactor = false;
    // The diagonal of a lumped M (empty for a coupled one).
    base::BudgetedBuffer<double> m_massDiagonal;
    SymmetricCsc m_shifted;
    SupernodalFactor m_factor;
};

} // namespace oos::linalg
