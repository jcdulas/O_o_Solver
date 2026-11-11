#pragma once

#include "oos/design/external/VectorJacobianProducts.hpp"
#include "oos/design/problem/DesignProblem.hpp"

#include <optional>

namespace oos::design
{

// Closed mechanical-preload sizing Hessians of simple buckling factors on the full path. Both preload and
// buckling-shape tangents are differentiated. An unavailable element or load formulation selects the general path.
[[nodiscard]] std::optional<ProductReport> bucklingHessianVector(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights,
    std::span<const double> directions, std::size_t count, std::vector<double>& values,
    base::MemoryBudget& budget, linalg::KernelKind kernel, base::ThreadPool* pool);

} // namespace oos::design
