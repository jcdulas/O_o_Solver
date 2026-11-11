#pragma once

#include "oos/design/external/VectorJacobianProducts.hpp"
#include "oos/design/problem/DesignProblem.hpp"

#include <optional>

namespace oos::design
{

// Analytic sizing curvature of the coupled trim state at fixed aerodynamic and support geometry. Uses the kept
// trim factors and closed element rates. An unavailable response/formulation returns no report for the general path.
[[nodiscard]] std::optional<ProductReport> trimHessianVector(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights,
    std::span<const double> directions, std::size_t count, std::vector<double>& values,
    base::MemoryBudget& budget, base::ThreadPool* pool);

} // namespace oos::design
