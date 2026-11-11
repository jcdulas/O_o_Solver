#pragma once

#include "RecoveredFieldCurvature.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/design/problem/DesignProblem.hpp"

#include <map>
#include <span>
#include <vector>

namespace oos::design::detail
{

// Closed first and mixed element matrix rates for linear sizing relations. Matrices stay on their elements;
// actions scatter directly to a g-set vector. There is no assembled matrix per design variable.
class SizingMatrixRates
{
public:
    SizingMatrixRates(const DesignProblem& problem, const DesignEvaluation& evaluation,
                      const solutions::AnalysisModel& model, base::MemoryBudget& budget);
    [[nodiscard]] bool prepare(const assembly::MassAssemblyOptions& options, bool needsMass = true);
    [[nodiscard]] std::span<const RecoveredFieldTarget> targets() const noexcept { return m_targets; }

    // Adds factor * K_p * u and factor * M_p * w for field p, or for a field direction.
    void addFirst(std::size_t quantity, std::span<const double> u, std::span<const double> w,
                   std::span<double> stiffness, std::span<double> mass, double factor = -1.0) const;
    void addDirection(std::span<const double> moves, std::span<const double> u, std::span<const double> w,
                       std::span<double> stiffness, std::span<double> mass, double factor = -1.0) const;
    void addMixed(std::size_t quantity, std::span<const double> moves, std::span<const double> u,
                   std::span<const double> w, std::span<double> stiffness, std::span<double> mass,
                   double factor = -1.0) const;

private:
    struct Matrices
    {
        elements::ElementMatrixBlock stiffness;
        elements::ElementMatrixBlock mass;
        base::MemoryReservation reservation;
    };
    struct First
    {
        RecoveredFieldTarget target;
        Matrices matrices;
    };
    struct Mixed
    {
        std::size_t first;
        std::size_t second;
        Matrices matrices;
    };
    [[nodiscard]] bool hasPlainRelations() const;
    void reserve(Matrices& matrices);
    void add(const Matrices& matrices, double factor, std::span<const double> u, std::span<const double> w,
              std::span<double> stiffness, std::span<double> mass) const;
    void action(const elements::ElementMatrixBlock& matrix, double factor, std::span<const double> vector,
                 std::span<double> into) const;

    const DesignProblem& m_problem;
    const DesignEvaluation& m_evaluation;
    const solutions::AnalysisModel& m_model;
    base::MemoryBudget& m_budget;
    double m_weightToMass = 1.0;
    std::vector<First> m_first;
    std::vector<Mixed> m_mixed;
    std::vector<RecoveredFieldTarget> m_targets;
    std::map<std::size_t, std::vector<std::size_t>> m_byQuantity;
    std::map<std::size_t, std::vector<std::size_t>> m_mixedByQuantity;
};

} // namespace oos::design::detail
