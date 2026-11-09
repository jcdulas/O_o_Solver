#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::linalg
{

// A pencil (K, B) of two dense symmetric matrices, K given by its diagonal (the reduced pencil of multilevel
// substructuring: K̄ = diag(Λ) and B̄ = I + Z + Zᵀ for vibration, K̄ = I and B̄ = diag(1/λ) + Z + Zᵀ for buckling), so that
// the block Lanczos solver solves it with its Sturm checks and its request's semantics. K − σB is formed dense and
// factorised by the Bunch-Kaufman LDLᵀ of the dense kernels (DenseKernels::factorWithPivoting), whose inertia is the
// Sturm count (Sylvester); the solves are its permutation, its unit triangles and its block diagonal.
//
// Memory: B (n²), the factor (n²) and the diagonal of K, reserved in the budget (MemoryBudgetError).
//
// Thread-safety: as ShiftedPencil. Ownership: owns its matrices and factor; references the kernels, which must outlive
// it.
class DensePencil final : public ShiftedPencil
{
public:
    // K = diag(stiffnessDiagonal), B the symmetric `rightHand` (its lower triangle is read and mirrored). Throws
    // std::invalid_argument when the sizes differ, MemoryBudgetError.
    DensePencil(std::span<const double> stiffnessDiagonal, ConstDenseMatrixView rightHand, SpectralTransform transform,
                const DenseKernels& kernels, base::MemoryBudget& budget);
    // The same taking `rightHand`'s storage (its lower triangle mirrored in place): no copy of B beside it (the
    // multilevel substructuring's reduced pencil: n² values fewer at its peak). The same values as the copying
    // constructor.
    DensePencil(std::span<const double> stiffnessDiagonal, DenseMatrix&& rightHand, SpectralTransform transform,
                const DenseKernels& kernels, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept override
    {
        return static_cast<std::int32_t>(m_stiffness.size());
    }
    [[nodiscard]] ShiftFactorisation factorise(double shift) override;
    // Throws std::logic_error before the first factorisation, SingularityError on a singular factor.
    void solve(DenseMatrixView block) const override;
    void multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] SpectralTransform transform() const noexcept override { return m_transform; }
    void multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const override;
    [[nodiscard]] double diagonalRayleighBound() const override { return m_diagonalRayleighBound; }

    // The bytes the pencil reserves for a dimension (its B, its factor, the diagonal and the permutation).
    [[nodiscard]] static std::uint64_t estimateMemoryBytes(std::size_t dimension) noexcept;

private:
    void initialise(std::span<const double> stiffnessDiagonal, base::MemoryBudget& budget);
    const DenseKernels& m_kernels;
    SpectralTransform m_transform;
    base::BudgetedBuffer<double> m_stiffness;
    DenseMatrix m_rightHand;                // B, both triangles
    DenseMatrix m_factor;                   // the LDLᵀ of K − σB in place (its lower part)
    std::vector<std::size_t> m_permutation; // bounded by the dimension
    std::vector<double> m_subdiagonal;      // bounded by the dimension
    bool m_hasFactor = false;
    bool m_isSingular = false;
    double m_diagonalRayleighBound = 0.0;
};

} // namespace oos::linalg
