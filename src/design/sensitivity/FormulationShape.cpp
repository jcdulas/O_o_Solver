#include "FormulationShape.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace oos::design::detail
{

std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT> sideFormulationModels(
    const FieldPerturbation& perturbation)
{
    std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT> models{};
    for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
        for (const assembly::AssemblyBatch& batch : perturbation.set(static_cast<FieldSide>(side)).batches())
            if (batch.view.formulation() != nullptr)
                models[side] = batch.view.formulation();
    if (std::any_of(models.begin(), models.end(), [](const auto* model) { return model == nullptr; }))
        throw std::logic_error("sideFormulationModels: a side of a shape perturbation without the element formulation "
                               "of the design point (a bug of O_o_Solver: report it)");
    return models;
}

std::array<assembly::FormulationRelations, FIELD_SIDE_COUNT> sideRelations(
    const assembly::FormulationConstraints& constraints,
    const std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT>& models, base::DiagnosticSink& sink)
{
    return {constraints.relationsOn(*models[0], sink), constraints.relationsOn(*models[1], sink),
            constraints.relationsOn(*models[2], sink), constraints.relationsOn(*models[3], sink)};
}

std::vector<double> dependentDerivative(const assembly::RigidElimination& gm,
                                        const assembly::FormulationRelations& relations,
                                        const std::array<assembly::FormulationRelations, FIELD_SIDE_COUNT>& sides,
                                        const FieldPerturbation& perturbation, std::span<const double> values)
{
    // ∂(Σ c_mj·values_j)/∂v of each formulation row, on its row of Gm.
    std::vector<double> source(gm.rowCount(), 0.0);
    for (std::size_t k = 0; k < relations.rowCount(); ++k)
    {
        const std::optional<std::size_t> row = gm.findRow(relations.dependentDof(k));
        if (!row)
            continue;
        std::array<double, FIELD_SIDE_COUNT> sums{};
        for (std::size_t side = 0; side < FIELD_SIDE_COUNT; ++side)
            for (const assembly::ConstraintTerm& term : sides[side].row(k))
                sums[side] += term.coefficient * values[static_cast<std::size_t>(term.dof)];
        source[*row] = perturbation.derivative(sums[0], sums[1], sums[2], sums[3]);
    }
    std::vector<double> derivative(values.size(), 0.0);
    for (std::size_t pass = 0; pass <= gm.rowCount(); ++pass)
    {
        bool isChanged = false;
        for (std::size_t i = 0; i < gm.rowCount(); ++i)
        {
            double value = source[i];
            for (const assembly::ConstraintTerm& term : gm.relationTerms(i))
                value += term.coefficient * derivative[static_cast<std::size_t>(term.dof)];
            const auto m = static_cast<std::size_t>(gm.mDofs()[i]);
            isChanged = isChanged || value != derivative[m];
            derivative[m] = value;
        }
        if (!isChanged)
            break;
    }
    return derivative;
}

} // namespace oos::design::detail
