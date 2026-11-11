#pragma once

#include "oos/design/external/VectorJacobianProducts.hpp"
#include "oos/design/problem/DesignProblem.hpp"

#include <optional>

namespace oos::design
{

// Exact second-order vibration products for simple eigenvalues, frequencies and modal fields, linear sizing
// relations and closed element derivatives. No perturbed analysis is run. An unavailable formulation returns no
// report so the caller can select its general derivative path; a repeated root has no unique Hessian and is refused.
[[nodiscard]] std::optional<ProductReport> modalHessianVector(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights,
    std::span<const double> directions, std::size_t count, std::vector<double>& values,
    base::MemoryBudget& budget, linalg::KernelKind kernel, base::ThreadPool* pool);

} // namespace oos::design
