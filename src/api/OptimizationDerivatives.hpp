#pragma once

#include "oos/api/Optimization.hpp"

namespace oos::api::detail
{

// A Hessian-vector product from native first-order gradients, used when second-order element blocks are unavailable.
[[nodiscard]] design::ProductReport nativeGradientHessian(OptimizationEvaluation& point,
                                                          const design::ProductWeights& weights,
                                                          std::span<const double> directions, std::size_t count,
                                                          std::vector<double>& values);

// A full-analysis column for shape changes that also move trim or inertia-relief support geometry.
struct FullShapeColumn
{
    std::vector<double> values; // one value per response item, in evaluation order
    std::int64_t analysisPerturbations = 0;
};
[[nodiscard]] FullShapeColumn fullShapeColumn(OptimizationEvaluation& point, std::size_t column);

void completeShapeGradients(OptimizationEvaluation& point, const design::GradientOptions& options,
                            design::DesignGradients& gradients, const std::vector<std::size_t>& shapeColumns);

} // namespace oos::api::detail
