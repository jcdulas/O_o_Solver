#include "oos/solutions/common/ElementEigenSystem.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/solutions/sol101/StreamedFront.hpp"

#include <optional>
#include <utility>

namespace oos::solutions
{

ElementEigenSystem::ElementEigenSystem(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_pool(inputs.pool)
{
    const AnalysisModel& model = *inputs.model;
    const assembly::DofMap& dofMap = model.dofMap();
    // The constraints, the sets and the singularities without Kgg nor Knn: the n-set diagonal blocks of the grid point
    // singularity processor summed element by element (as on the streamed full path).
    {
        std::optional<StreamedFront::NodeDiagonals> diagonals;
        ConstrainedSystemOptions systemOptions;
        systemOptions.diagonalBlocks = [&](const assembly::RigidElimination& gm, const assembly::DofSets&)
        {
            diagonals.emplace(StreamedFront::nodeDiagonals(model, gm, budget, inputs.pool, inputs.cancellation));
            return std::move(diagonals->blocks);
        };
        m_system = std::make_unique<ConstrainedSystem>(model, inputs.selection, budget, inputs.pool,
                                                       inputs.cancellation, sink, systemOptions);
    }
    m_ordering = std::make_unique<FsetOrdering>(
        m_system->graph(), dofMap, m_system->solveSets(), assembly::DofSet::F, linalg::OrderingMethod::Automatic,
        budget, inputs.pool, inputs.cancellation, std::span<const std::int32_t>(), inputs.thresholds);
    if (m_ordering->equationCount() == 0)
        return;
    m_isOutputNode = base::BudgetedBuffer<std::uint8_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                        "pencil from the elements: output points");
    StreamedElements::Inputs placement;
    placement.model = &model;
    placement.gm = &m_system->gm();
    placement.selection = &m_ordering->selection();
    placement.symbolic = &m_ordering->symbolic();
    placement.isOutputNode = m_isOutputNode.span();
    m_elements = std::make_unique<StreamedElements>(placement, budget);
    m_shares = std::make_unique<assembly::NonStructuralMassShares>(model.elementSet());
    ElementPencil::Inputs pencil;
    pencil.model = &model;
    pencil.elements = m_elements.get();
    pencil.massOptions = assembly::MassAssemblyOptions::fromParams(model.params(), inputs.selection.subcase, sink);
    pencil.shares = m_shares.get();
    pencil.transform = inputs.transform;
    pencil.preloadDisplacements = inputs.preloadDisplacements;
    pencil.preloadTemperatures = inputs.preloadTemperatures;
    pencil.hasMassSource = inputs.transform == linalg::SpectralTransform::Buckling;
    pencil.pool = inputs.pool;
    pencil.cancellation = inputs.cancellation;
    m_pencil = std::make_unique<ElementPencil>(pencil, budget);
}

ElementEigenSystem::~ElementEigenSystem() = default;

std::uint64_t ElementEigenSystem::productBytes(std::size_t columns) const noexcept
{
    return ElementPencil::workBytes(m_pool != nullptr ? m_pool->threadCount() : 1, columns);
}

} // namespace oos::solutions
