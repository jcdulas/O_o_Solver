#pragma once

#include "oos/api/Optimization.hpp"

namespace oos::api::detail
{

// Shape columns that move support or aerodynamic geometry use the completed gradient from gradients().
[[nodiscard]] design::ProductReport fullShapeJvp(OptimizationEvaluation& point, std::span<const double> directions,
                                                  std::size_t count, std::size_t block, design::ProductSink& sink,
                                                  const design::ProductSelection& selection);
[[nodiscard]] design::ProductReport fullShapeVjp(OptimizationEvaluation& point,
                                                  const design::ProductWeights& weights, std::size_t block,
                                                  design::ProductSink& sink);
[[nodiscard]] design::GradientStreamReport fullShapeStream(OptimizationEvaluation& point,
                                                            base::MemoryBudget& budget, design::GradientSink& sink,
                                                            std::size_t block);

} // namespace oos::api::detail
