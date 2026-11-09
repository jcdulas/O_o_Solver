#include "oos/solutions/common/ModeRecovery.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace oos::solutions
{

ModeRecovery::ModeRecovery(const AnalysisModel& model, const ConstrainedSystem& system, std::string analysis,
                           std::int64_t recoveryRangeElements, bool isRecoveredCaseByCase)
    : m_model(&model),
      m_system(&system),
      m_analysis(std::move(analysis)),
      m_nodalBuilder(model.dofMap()),
      m_elementBuilder(model.elementSet(), model.dofMap(), recoveryRangeElements),
      m_isRecoveredCaseByCase(isRecoveredCaseByCase)
{
}

ModeRecovery::ModeRecovery(const AnalysisModel& model, std::string analysis, std::int64_t recoveryRangeElements,
                           bool isRecoveredCaseByCase)
    : m_model(&model),
      m_system(nullptr),
      m_analysis(std::move(analysis)),
      m_nodalBuilder(model.dofMap()),
      m_elementBuilder(model.elementSet(), model.dofMap(), recoveryRangeElements),
      m_isRecoveredCaseByCase(isRecoveredCaseByCase)
{
}

std::vector<std::vector<output::ElementResultTable>> ModeRecovery::elementTables(
    std::int32_t subcase, const output::OutputPlan& plan, linalg::ConstDenseMatrixView modes,
    base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation) const
{
    std::vector<std::vector<output::ElementResultTable>> tables; // one per mode
    const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
    if (m_isRecoveredCaseByCase ||
        (!selection.stress.isRequested && !selection.strain.isRequested && !selection.forces.isRequested))
        return tables;
    tables.resize(modes.columns);
    // The tables of every mode carry the subcase; recover() stamps them with the mode's step and eigenvalue.
    for (std::size_t first = 0; first < modes.columns; first += MODE_BLOCK)
    {
        const std::size_t count = std::min(MODE_BLOCK, modes.columns - first);
        const std::vector<std::int32_t> labels(count, subcase);
        m_elementBuilder.buildCases(labels, modes.block(0, first, modes.rows, count), modes.columns, budget,
                                    std::span(tables).subspan(first, count), pool, cancellation, &selection);
    }
    return tables;
}

std::vector<std::vector<output::ElementResultTable>> ModeRecovery::strainEnergyTables(
    std::int32_t subcase, const output::OutputPlan& plan, linalg::ConstDenseMatrixView modes,
    base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation) const
{
    std::vector<std::vector<output::ElementResultTable>> tables; // one per mode
    if (!plan.of(output::OutputKind::StrainEnergy).isRequested)
        return tables;
    output::StiffnessRecovery(m_model->elementSet(), m_model->dofMap(), pool)
        .strainEnergies(subcase, modes, plan.of(output::OutputKind::StrainEnergy), budget, tables, cancellation);
    return tables;
}

base::BudgetedBuffer<double> ModeRecovery::expand(const linalg::DenseMatrix& vectors, std::size_t column,
                                                  const linalg::DofSelection& selection,
                                                  base::MemoryBudget& budget) const
{
    if (m_system == nullptr)
        throw std::logic_error(
            "ModeRecovery::expand: no constrained system (the condensed path expands its own "
            "vectors)");
    const auto gSize = static_cast<std::size_t>(m_model->dofMap().gSetSize());
    base::BudgetedBuffer<double> modeG(budget, gSize, "mode shape ug");
    std::fill(modeG.data(), modeG.data() + gSize, 0.0);
    for (std::size_t e = 0; e < vectors.rows(); ++e)
        modeG[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(e)))] = vectors(e, column);
    m_system->gm().expandToG(modeG.span());
    return modeG;
}

void ModeRecovery::recover(std::int32_t subcase, const output::OutputPlan& plan, std::span<const double> modeG,
                           std::int32_t step, double eigenvalue, const linalg::BlockSparseMatrix* second,
                           double secondFactor, base::MemoryBudget& budget, base::ThreadPool* pool,
                           const base::CancellationToken* cancellation, output::ResultArrays& arrays,
                           std::vector<output::ElementResultTable>* strainEnergies,
                           std::vector<output::ElementResultTable>* elementTables) const
{
    const auto gSize = modeG.size();
    const auto stamp = [&](auto& table)
    {
        table.analysis = m_analysis;
        table.step = step;
        table.eigenvalue = eigenvalue;
    };
    const auto addNodal = [&](output::OutputKind kind, const char* tableKind, std::span<const double> values)
    {
        const output::OutputRequestPlan& request = plan.of(kind);
        if (!request.isRequested)
            return;
        output::NodalResultTable table =
            m_nodalBuilder.build(tableKind, subcase, values, budget, request.isAll, request.members);
        stamp(table);
        arrays.nodalTables.push_back(std::move(table));
    };
    addNodal(output::OutputKind::Displacement, "eigenvectors", modeG);

    const bool hasMset = m_system != nullptr && m_system->gm().rowCount() > 0;
    const bool wantsSpc = m_system != nullptr && plan.of(output::OutputKind::SpcForces).isRequested;
    const bool wantsMpc = hasMset && plan.of(output::OutputKind::MpcForces).isRequested;
    if (wantsSpc || wantsMpc)
    {
        // r = Kgg·φ + c·Bgg·φ split into MPC forces (Gm) and SPC forces, with −c·Bgg·φ in the place of the loads.
        base::BudgetedBuffer<double> elementForces(budget, gSize, "mode Kgg phi");
        m_system->kgg().multiply(modeG, elementForces.span());
        base::BudgetedBuffer<double> loads(budget, gSize, "mode inertia forces");
        std::fill(loads.data(), loads.data() + gSize, 0.0);
        if (second != nullptr)
        {
            second->multiply(modeG, loads.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                loads[dof] *= -secondFactor;
        }
        base::BudgetedBuffer<double> mpcForces(budget, gSize, "mode MPC forces");
        std::fill(mpcForces.data(), mpcForces.data() + gSize, 0.0);
        if (hasMset)
            m_system->gm().computeMpcForces(elementForces.span(), loads.span(), mpcForces.span());
        base::BudgetedBuffer<double> spcForces(budget, gSize, "mode SPC forces");
        const std::span<const std::uint8_t> isConstrained = m_system->isConstrained();
        for (std::size_t dof = 0; dof < gSize; ++dof)
            spcForces[dof] = isConstrained[dof] != 0 ? elementForces[dof] - loads[dof] - mpcForces[dof] : 0.0;
        if (wantsSpc)
            addNodal(output::OutputKind::SpcForces, "singlePointConstrForces", spcForces.span());
        if (wantsMpc)
            addNodal(output::OutputKind::MpcForces, "multiPointConstrForces", mpcForces.span());
    }

    const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
    std::vector<output::ElementResultTable> tables; // bounded by the element types × kinds
    if (elementTables != nullptr)
        tables = std::move(*elementTables);
    else if (selection.stress.isRequested || selection.strain.isRequested || selection.forces.isRequested)
        m_elementBuilder.build(subcase, modeG, {}, budget, tables, pool, cancellation, nullptr, &selection);
    if (plan.of(output::OutputKind::StrainEnergy).isRequested)
    {
        if (strainEnergies != nullptr)
            for (output::ElementResultTable& table : *strainEnergies)
                tables.push_back(std::move(table));
        else
            output::StiffnessRecovery(m_model->elementSet(), m_model->dofMap(), pool)
                .strainEnergy(subcase, modeG, plan.of(output::OutputKind::StrainEnergy), budget, tables, cancellation);
    }
    for (output::ElementResultTable& table : tables)
    {
        stamp(table);
        arrays.elementTables.push_back(std::move(table));
    }
}

} // namespace oos::solutions
