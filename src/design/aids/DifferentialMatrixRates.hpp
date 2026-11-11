#pragma once

#include "RecoveredFieldCurvature.hpp"

namespace oos::design::detail
{

// Actions of the geometric stiffness and its explicit field rates. At a fixed mechanical preload K_d is linear
// in that preload; mixed total rates also need the first and mixed preload displacements supplied by the caller.
class DifferentialMatrixRates
{
public:
    DifferentialMatrixRates(const solutions::AnalysisModel& model, std::span<const RecoveredFieldTarget> targets);
    [[nodiscard]] bool prepare(std::span<const double> preload) const;
    void addPreload(std::span<const double> preload, std::span<const double> vector, std::span<double> into) const;
    void addField(std::size_t quantity, std::span<const double> preload, std::span<const double> vector,
                   std::span<double> into, double factor = 1.0) const;
    void addDirection(std::span<const double> moves, std::span<const double> preload,
                       std::span<const double> vector, std::span<double> into) const;
    void addMixed(std::size_t quantity, std::span<const double> moves, std::span<const double> preload,
                   std::span<const double> vector, std::span<double> into) const;

private:
    [[nodiscard]] elements::NodalDisplacements nodal(std::span<const double> vector) const;
    void action(const elements::ElementMatrixBlock& matrices, double factor, std::span<const double> vector,
                 std::span<double> into) const;
    [[nodiscard]] bool isActive(const RecoveredFieldTarget& target) const;

    const solutions::AnalysisModel& m_model;
    std::span<const RecoveredFieldTarget> m_targets;
    std::map<ElementRef, std::vector<std::size_t>> m_byElement;
    std::map<std::size_t, std::vector<std::size_t>> m_byQuantity;
};

} // namespace oos::design::detail
