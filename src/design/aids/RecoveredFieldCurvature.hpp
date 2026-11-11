#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/design/responses/DesignResp1Response.hpp"
#include "oos/design/sensitivity/AffectedElements.hpp"
#include "oos/design/sensitivity/ElementItemFunction.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"

#include <map>
#include <span>
#include <vector>

namespace oos::design::detail
{

struct WeightedRecoveredField
{
    const DesignResp1Response* response;
    ResponseItem label;
    double weight;
    double value;
};

struct RecoveredFieldTarget
{
    ElementRef element;
    std::size_t quantity;
    elements::DesignedField field;
    elements::FieldUse use;
};

// Partial derivatives of weighted recovered fields R(u, p), before applying a solution's implicit state derivatives.
// The local displacement Hessians and displacement/field blocks remain on their elements; no dense model Hessian
// and no DOF-by-field matrix are formed. The modal and trim Hessian paths supply their own state derivatives.
class RecoveredFieldCurvature
{
public:
    RecoveredFieldCurvature(const solutions::AnalysisModel& model,
                        std::shared_ptr<const input::ModelSnapshot> snapshot, base::MemoryBudget& budget);
    bool prepare(std::span<const WeightedRecoveredField> fields, std::span<const RecoveredFieldTarget> targets,
                  std::span<const double> shape);
    [[nodiscard]] std::span<const double> weights() const noexcept { return m_weights.span(); }
    // d(R_psi) = R_psi,psi psi' + R_psi,p p'.
    void weightRate(std::span<const double> shapeRate, std::span<const double> fieldMoves,
                     std::span<double> into) const;
    // Adds R_p,psi psi' + R_p,p p' to the field curvature.
    void addExplicitRate(std::span<const double> shapeRate, std::span<const double> fieldMoves,
                          std::span<double> into) const;

private:
    struct DisplacementBlock
    {
        std::vector<elements::ElementDof> dofs;
        std::vector<double> values;
    };
    struct MixedBlock
    {
        std::size_t quantity;
        std::vector<elements::ElementDof> dofs;
        std::vector<double> values;
    };
    bool add(const WeightedRecoveredField& field, std::span<const RecoveredFieldTarget> targets,
              std::span<const double> shape);
    bool addRecovery(const WeightedRecoveredField& field, const ElementRef& element, const ElementItemSpec& spec,
                      std::span<const RecoveredFieldTarget> targets, std::span<const double> shape);

    const solutions::AnalysisModel& m_model;
    std::shared_ptr<const input::ModelSnapshot> m_snapshot;
    base::MemoryBudget& m_budget;
    base::BudgetedBuffer<double> m_weights;
    std::vector<DisplacementBlock> m_displacements;
    std::vector<MixedBlock> m_mixed;
    std::map<std::pair<std::size_t, std::size_t>, double> m_fieldPairs;
    base::MemoryReservation m_reservation;
};

} // namespace oos::design::detail
