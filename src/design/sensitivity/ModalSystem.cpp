#include "ModalSystem.hpp"

#include "oos/base/errors/OptimisationError.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <cmath>

namespace oos::design::detail
{

std::optional<std::size_t> modalNormalisationPivot(const solutions::EigenMethod& method,
                                                  std::span<const double> shape,
                                                  const linalg::DofSelection& selection,
                                                  const assembly::DofMap& dofMap)
{
    if (method.normalization == solutions::EigenNormalization::Mass)
        return std::nullopt;
    if (method.normalization == solutions::EigenNormalization::Point)
        if (const auto dof = dofMap.findDof(method.normalizationPoint, method.normalizationComponent))
            if (selection.equationOfDof(*dof) != linalg::DofSelection::ELIMINATED &&
                shape[static_cast<std::size_t>(*dof)] != 0.0)
                return static_cast<std::size_t>(*dof);
    std::size_t pivot = 0;
    double largest = 0.0, second = 0.0;
    for (std::int32_t equation = 0; equation < selection.keptDofCount(); ++equation)
    {
        const auto dof = static_cast<std::size_t>(selection.dofOfEquation(equation));
        const double magnitude = std::abs(shape[dof]);
        if (magnitude > largest)
        {
            second = largest;
            largest = magnitude;
            pivot = dof;
        }
        else
            second = std::max(second, magnitude);
    }
    if (largest - second <= 1e-10 * largest)
        throw base::OptimisationError(
            "modal field derivatives have tied MAX normalisation components; use Mass or an explicit Point");
    return pivot;
}

ModalSystem::ModalSystem(const solutions::AnalysisModel& model, std::int32_t subcase,
                         base::MemoryBudget& budget, base::ThreadPool* pool)
{
    base::DiagnosticSink sink;
    const auto subcases = solutions::staticSubcasesOf(model.snapshot(), {subcase}, sink);
    const auto& selected = subcases.front();
    const solutions::ConstraintSelection selection{selected.label, selected.spc, selected.mpc, std::nullopt,
                                                   selected.contact};
    m_constraints = std::make_unique<solutions::ConstrainedSystem>(model, selection, budget, pool, nullptr, sink);
    m_massOptions = assembly::MassAssemblyOptions::fromParams(model.params(), subcase, sink);
    linalg::BlockSparseMatrix mgg(m_constraints->pattern(), budget, "modal derivative mass Mgg");
    const assembly::NonStructuralMassShares shares(model.elementSet());
    assembly::MassAssembler(model.elementSet(), model.dofMap(), shares, m_massOptions).assemble(mgg, pool);
    const auto mnn = m_constraints->reduce(mgg, budget, "modal derivative mass Mnn");
    m_ordering = std::make_unique<solutions::FsetOrdering>(m_constraints->graph(), model.dofMap(),
        m_constraints->solveSets(), assembly::DofSet::F, linalg::OrderingMethod::Automatic, budget, pool);
    m_stiffness = linalg::SymmetricCsc::fromBlocks(m_constraints->knn(), m_ordering->selection(), budget, pool);
    m_mass = linalg::SymmetricCsc::fromBlocks(mnn, m_ordering->selection(), budget, pool);
}

} // namespace oos::design::detail
