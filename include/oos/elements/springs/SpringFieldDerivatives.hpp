#pragma once

#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/RecoverySlopes.hpp"
#include "oos/elements/core/ScalarConnection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements
{

// A scalar spring has K = k s s^T, force = k delta and stress = S k delta, where delta = s^T u.
// These helpers share the exact rates of that formulation across the four spring card families.
enum class SpringField
{
    Other,
    Stiffness,
    StressCoefficient
};

[[nodiscard]] inline SpringField springFieldOf(const DesignedField& field, std::string_view card,
                                               bool hasStressCoefficient)
{
    if (field.card != card || field.item != -1)
        return SpringField::Other;
    if (field.column == "stiffness")
        return SpringField::Stiffness;
    if (hasStressCoefficient && field.column == "stressCoefficient")
        return SpringField::StressCoefficient;
    return SpringField::Other;
}

inline bool appendSpringStiffnessRate(std::int64_t row, const ScalarConnection& connection, SpringField field,
                                      ElementMatrixBlock& block)
{
    if (field != SpringField::Stiffness)
        return false;
    appendScalarMatrix(block, row, connection, 1.0);
    return true;
}

[[nodiscard]] inline double springRecoveryFactor(SpringField field, ElementOutput output, double stiffness,
                                                 double stressCoefficient)
{
    if (field == SpringField::Stiffness)
        return output == ElementOutput::Forces ? 1.0 : stressCoefficient;
    return output == ElementOutput::Forces ? 0.0 : stiffness;
}

inline bool appendSpringRecoveryRate(const ElementBatchView& view, std::int64_t row,
                                      const ScalarConnection& connection, SpringField field,
                                      ElementOutput output, const NodalDisplacements& displacements,
                                      double stiffness, double stressCoefficient, ElementResultBlock& rates)
{
    if (field == SpringField::Other)
        return false;
    rates.reset(output == ElementOutput::Forces ? std::vector<std::string>{"springForce"}
                                                 : std::vector<std::string>{"springStress"});
    const double factor = springRecoveryFactor(field, output, stiffness, stressCoefficient);
    const std::array<double, 1> value{factor * connection.relativeMotion(displacements)};
    rates.append(row, view.elementId(row), value);
    return true;
}

inline bool appendSpringSlopeRate(const ScalarConnection& connection, SpringField field, ElementOutput output,
                                   double stiffness, double stressCoefficient, RecoverySlopes& rates)
{
    if (field == SpringField::Other)
        return false;
    rates.reset(connection.presentDofs(), 1);
    const std::size_t row = rates.appendRow({});
    const double factor = springRecoveryFactor(field, output, stiffness, stressCoefficient);
    for (std::size_t dof = 0; dof < connection.count; ++dof)
        rates.slopes(row, 0)[dof] = factor * connection.signs[dof];
    return true;
}

inline bool appendSpringRecoveryMixedRate(const ElementBatchView& view, std::int64_t row,
                                          const ScalarConnection& connection, SpringField first, SpringField second,
                                          ElementOutput output, const NodalDisplacements& displacements,
                                          ElementResultBlock& rates)
{
    if (first == SpringField::Other || second == SpringField::Other)
        return false;
    const bool isCrossTerm = (first == SpringField::Stiffness && second == SpringField::StressCoefficient) ||
                             (second == SpringField::Stiffness && first == SpringField::StressCoefficient);
    rates.reset(output == ElementOutput::Forces ? std::vector<std::string>{"springForce"}
                                                : std::vector<std::string>{"springStress"});
    const std::array<double, 1> values{
        isCrossTerm && output == ElementOutput::Stresses ? connection.relativeMotion(displacements) : 0.0};
    rates.append(row, view.elementId(row), values);
    return true;
}

inline bool appendSpringRecoverySecondRate(const ElementBatchView& view, std::int64_t row, SpringField field,
                                            ElementOutput output, ElementResultBlock& rates)
{
    if (field == SpringField::Other)
        return false;
    rates.reset(output == ElementOutput::Forces ? std::vector<std::string>{"springForce"}
                                                 : std::vector<std::string>{"springStress"});
    const std::array<double, 1> zero{0.0};
    rates.append(row, view.elementId(row), zero);
    return true;
}

} // namespace oos::elements
