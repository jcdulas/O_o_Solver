#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <span>

namespace oos::linalg
{

// The linearised equation of a simple symmetric eigenpair, with its mass-normalisation constraint:
// [K - lambda M, M phi; (M phi)^T, 0]. One sparse border removes the eigenvector's scale ambiguity.
// The interior keeps the caller's fill-reducing order; the border is eliminated last. No dense model matrix is built.
// Reference: Fox and Kapoor, "Rates of change of eigenvalues and eigenvectors", AIAA Journal 6(12), 1968.
class BorderedEigenSystem
{
public:
    BorderedEigenSystem(const SymmetricCsc& stiffness, const SymmetricCsc& mass, double root,
                        std::span<const double> shape, const DenseKernels& kernels, base::MemoryBudget& budget);

    // Solves [A b; b^T 0] [adjoint; multiplier] = [weights; normalisationRate] in the caller's equation order.
    // The returned multiplier uses the physical border b = M phi, independently of its internal balancing.
    // Throws SingularityError for a repeated root, whose individual eigenvector has no unique derivative.
    [[nodiscard]] double solve(std::span<const double> weights, std::span<double> adjoint,
                                double normalisationRate = 0.0) const;

private:
    const DenseKernels& m_kernels;
    base::MemoryBudget& m_budget;
    SymbolicFactor m_symbolic;
    SupernodalFactor m_factor;
    double m_borderScale = 1.0;
};

} // namespace oos::linalg
