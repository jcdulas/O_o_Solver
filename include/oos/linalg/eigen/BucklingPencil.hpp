#pragma once

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

// The buckling pencil of SOL 105, (K + λ K_d) φ = 0, written K φ = λ B φ with B = −K_d (SpectralTransform::
// Buckling): K the stiffness (positive definite once the constraints are applied: the preload's static solve needs
// it), K_d the differential stiffness of the preload, assembled on the pattern of K and converted with the same DOF
// selection (the order of the symbolic factor). At each shift K − σB = K + σK_d is formed on that pattern
// (SymmetricCsc::combine) and factorised by the supernodal driver in indefinite mode (Bunch-Kaufman pivots): one
// symbolic factor serves every shift. The operator is Op = (K − σB)⁻¹·B in the K-inner product (ShiftedPencil).
//
// Invariants: K and K_d have the same structure. The factor of the last shift is kept; the previous one and its
// shifted matrix are released before the next is made, so at most one factor is alive. Whether K is positive definite
// is not checked here: BlockLanczos factorises at σ = 0 first and refuses a K with a negative or zero pivot.
//
// Thread-safety: as ShiftedPencil. Ownership: references the symbolic factor, the matrices, the kernels and the budget,
// which must outlive it; owns the shifted matrix and the factor (budgeted).
class BucklingPencil final : public ShiftedPencil
{
public:
    // Throws std::invalid_argument when the matrices differ in structure or are not of the dimension of the symbolic
    // factor. The pivoting of the options is replaced by PivotingMode::Indefinite.
    BucklingPencil(const SymbolicFactor& symbolic, const SymmetricCsc& stiffness, const SymmetricCsc& differential,
                   const DenseKernels& kernels, const FactorOptions& options, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept override { return m_stiffness.dimension(); }
    // K + σK_d. MemoryBudgetError or OperationCancelledError from the factorisation propagate.
    [[nodiscard]] ShiftFactorisation factorise(double shift) override;
    // Throws std::logic_error before the first factorisation; SingularityError on a singular factor.
    void solve(DenseMatrixView block) const override;
    // Y ← B·X = −K_d·X.
    void multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] SpectralTransform transform() const noexcept override { return SpectralTransform::Buckling; }
    // Y ← K·X.
    void multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const override;
    // min K_ii / |K_d,ii| over the equations with K_d,ii ≠ 0 and K_ii > 0 (+∞ when there is none).
    [[nodiscard]] double diagonalRayleighBound() const override { return m_diagonalRayleighBound; }

    [[nodiscard]] std::size_t factorisationCount() const noexcept { return m_factorisationCount; }

    // An upper bound of the bytes the pencil holds at once: the shifted matrix and one factor (before the growth of
    // delayed pivots).
    [[nodiscard]] static std::uint64_t estimateMemoryBytes(const SymbolicFactor& symbolic,
                                                           const SymmetricCsc& stiffness, std::size_t threadCount);

private:
    const SymbolicFactor& m_symbolic;
    const SymmetricCsc& m_stiffness;
    const SymmetricCsc& m_differential;
    const DenseKernels& m_kernels;
    FactorOptions m_options;
    base::MemoryBudget& m_budget;
    double m_diagonalRayleighBound = 0.0;
    std::size_t m_factorisationCount = 0;
    bool m_hasFactor = false;
    SymmetricCsc m_shifted;
    SupernodalFactor m_factor;
};

} // namespace oos::linalg
