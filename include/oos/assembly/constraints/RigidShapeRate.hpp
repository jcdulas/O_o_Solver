#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"

namespace oos::assembly
{
// Imposed part of du_g when the independent displacements are held fixed.
[[nodiscard]] inline std::vector<double> imposedRate(const RigidElimination& gm,
                                                     std::span<const LocalConstraintRows> rates,
                                                     std::span<const double> displacements)
{
    std::vector<double> imposed(displacements.size(), 0.0);
    for (const auto& spider : rates)
        for (std::size_t i = 0; i < spider.size(); ++i)
            for (const auto& term : spider.row(i))
                imposed[spider.dependents[i]] += term.coefficient * displacements[term.dof];
    const auto order = gm.chainOrder();
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        for (const auto& term : gm.relationTerms(*it))
            imposed[gm.mDofs()[*it]] += term.coefficient * imposed[term.dof];
    return imposed;
}

} // namespace oos::assembly
