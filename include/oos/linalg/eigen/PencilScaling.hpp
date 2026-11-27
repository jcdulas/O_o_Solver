#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

// The scaling of the shifted matrices of the eigenvalue solvers before their factorisation.
//
// The factorisation of K − σB in indefinite mode accepts a pivot when it is not small beside the largest term of its
// column (the threshold test of the delayed pivots, DELAYED_PIVOT_THRESHOLD). That test compares terms of different
// physical dimensions: in a plate the rotation terms are of the order of D, the terms that couple a rotation to a
// deflection of the order of D/h (h the size of an element). With lengths in metres and elements of a few
// millimetres the rotations fail the test by their units alone, are delayed to their parents and grow every front on
// their way up: on a plate of two million DOFs the delayed pivots took as much memory as the factor itself.
//
// The remedy is the symmetric scaling of the matrix by its diagonal (Duff, Erisman and Reid, Direct Methods for
// Sparse Matrices, 2nd ed., 2017, section 4.15): S·(K − σB)·S with S = diag(s_i), s_i about 1/sqrt(K_ii), has a
// diagonal of the order of one whatever the units, and its off-diagonal terms are at most of the order of one where
// the matrix is positive definite. The scales are powers of two, so that scaling rounds nothing: the factor differs
// from the unscaled one by its choice of pivots only. A congruence keeps the inertia (the Sturm counts), and the
// solves are those of the matrix itself: (K − σB)⁻¹ = S·(S·(K − σB)·S)⁻¹·S.

// The scales of the equations of `stiffness`: a power of two near 1/sqrt(K_ii) (the scaled diagonal term is between
// 1/4 and 2), 1 where K_ii is not positive (an equation without stiffness of its own).
[[nodiscard]] base::BudgetedBuffer<double> pencilScaleOf(const SymmetricCsc& stiffness, base::MemoryBudget& budget);
// The same from the diagonal of the stiffness (a pencil given by its sources).
[[nodiscard]] base::BudgetedBuffer<double> pencilScaleOf(std::span<const double> stiffnessDiagonal,
                                                         base::MemoryBudget& budget);

// Each row i of the block multiplied by scale[i].
void scaleRows(DenseMatrixView block, std::span<const double> scale);

// The front source of S·A·S from the front source of A (a pencil given front by front, no matrix assembled): each term
// (i, j) of `inner` times scale[i]·scale[j], on the equations of `symbolic`. addTerms() scales the front once `inner`
// has filled it: the front then holds the source's terms alone (they come before the children's update matrices).
//
// Thread-safety: as `inner` (addTerms() concurrently for distinct supernodes). Ownership: borrows the source, the
// scales and the symbolic factor, which outlive it.
class ScaledFrontSource final : public FrontSource
{
public:
    ScaledFrontSource(FrontSource& inner, std::span<const double> scale, const SymbolicFactor& symbolic) noexcept
        : m_inner(inner), m_scale(scale), m_symbolic(symbolic)
    {
    }

    void addTerms(FrontTerms& front) override;
    void diagonal(std::int32_t supernode, std::span<double> values) const override;

private:
    FrontSource& m_inner;
    std::span<const double> m_scale;
    const SymbolicFactor& m_symbolic;
};

} // namespace oos::linalg
