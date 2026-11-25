#include "oos/api/AeroCoupling.hpp"

#include "AdaptiveShellStatics.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"

#include <numeric>
#include <stdexcept>

namespace oos::api
{
namespace
{
AeroCoupledResult newRun()
{
    AeroCoupledResult result;
    const auto bytes = base::MemoryInfo::query().availableBytes;
    result.analysis.budget = std::make_shared<base::MemoryBudget>(bytes ? bytes : base::MemoryBudget::UNLIMITED);
    return result;
}
}

AeroCoupling::AeroCoupling(std::shared_ptr<const input::ModelSnapshot> snapshot) : m_snapshot(std::move(snapshot))
{
    if (!m_snapshot) throw std::invalid_argument("aero coupling needs a model");
    m_transfer = std::make_shared<AeroTransfer>(*m_snapshot);
    base::MemoryBudget budget(base::MemoryBudget::UNLIMITED);
    const input::PointIndex points(m_snapshot->nodes(), budget);
    const assembly::DofMap map(m_snapshot->nodes(), points);
    for (const auto id : nodeIds())
    {
        const auto row = points.findGrid(id).value();
        m_attachments.push_back({map.gridDof(row, 1), map.cdToBasic(row)});
    }
}

std::vector<double> AeroCoupling::attachmentMotion(std::span<const double> field) const
{
    std::vector<double> motion(6 * m_attachments.size());
    for (std::size_t i = 0; i < m_attachments.size(); ++i)
    {
        const auto& attachment = m_attachments[i];
        for (std::size_t offset : {0u, 3u})
        {
            const auto j = attachment.start + offset;
            const auto v = attachment.cdToBasic * base::Vec3{field[j], field[j + 1], field[j + 2]};
            for (std::size_t c = 0; c < 3; ++c) motion[6 * i + offset + c] = v[c];
        }
    }
    return motion;
}

AeroCoupledResult AeroCoupling::solveStatic(base::EntityId subcase, std::span<const double> boxForces) const
{
    auto result = newRun();
    if (m_snapshot->hasTable(input::TableRole::Subcases))
    {
        const auto& cases = m_snapshot->subcases();
        const auto row = cases.findRow(subcase);
        if (row >= 0 && cases.kinds()[row] == "subcaseCombination")
            throw std::invalid_argument("apply external forces to a static load case before combining its results");
    }
    const auto loads = structuralLoads(boxForces);
    std::vector<solutions::AppliedCaseLoads::Entry> entries;
    for (std::size_t i = 0; i < m_attachments.size(); ++i)
    {
        const auto& attachment = m_attachments[i];
        for (int offset : {0, 3})
        {
            const auto j = 6 * i + offset;
            const auto v = attachment.cdToBasic.transposed() * base::Vec3{loads[j], loads[j + 1], loads[j + 2]};
            for (int c = 0; c < 3; ++c)
                if (v[c] != 0) entries.push_back({attachment.start + offset + c, v[c]});
        }
    }
    solutions::StaticOptions options;
    options.usesAeroTransfer = true;
    options.appliedLoads = {subcase, entries};
    options.subcases = options.keptDisplacements = {subcase};
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    auto state = std::make_shared<AeroStaticPreload>();
    state->subcase = subcase;
    state->snapshot = m_snapshot;
    state->budget = result.analysis.budget;
    auto solved = runStaticSolution(*m_snapshot, options, *state->budget, result.analysis.selectedQuarticShells,
                                    &state->index);
    state->displacement = std::move(solved.keptDisplacements.at(0).values);
    result.boxMotions.push_back(boxMotion(attachmentMotion(state->displacement.span())));
    result.preload = std::move(state);
    result.analysis.tables = std::move(solved.arrays);
    result.analysis.diagnostics = std::move(solved.diagnostics);
    return result;
}

AeroCoupledResult AeroCoupling::solveModes(base::EntityId subcase) const
{
    auto result = newRun();
    solutions::ModesOptions options;
    options.usesAeroTransfer = true;
    options.subcases = {subcase};
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    auto solved = solutions::ModesSolution(*m_snapshot, options).run(*result.analysis.budget);
    const auto& sub = solved.subcases.at(0);
    const auto& vectors = solved.solves.at(sub.solve).eigenvectors;
    auto basis = std::shared_ptr<AeroModalBasis>(new AeroModalBasis(m_transfer));
    for (std::size_t k = 0; k < vectors.columns(); ++k)
    {
        auto motion = attachmentMotion({vectors.data() + k * vectors.leadingDimension(), vectors.rows()});
        result.boxMotions.push_back(boxMotion(motion));
        basis->attachmentModes.push_back(std::move(motion));
    }
    result.modalBasis = std::move(basis);
    result.analysis.eigenvalues.push_back(sub.eigenvalues);
    result.analysis.tables = std::move(solved.arrays);
    result.analysis.diagnostics = std::move(solved.diagnostics);
    return result;
}

std::vector<double> AeroModalBasis::generalizedLoads(std::span<const double> boxForces) const
{
    const auto loads = transfer->structuralLoads(boxForces);
    std::vector<double> generalized;
    for (const auto& mode : attachmentModes)
        generalized.push_back(std::inner_product(mode.begin(), mode.end(), loads.begin(), 0.0));
    return generalized;
}

AeroCoupledResult AeroCoupling::solveBuckling(base::EntityId subcase, const AeroStaticPreload& preload) const
{
    if (preload.snapshot != m_snapshot)
        throw std::invalid_argument("the static preload belongs to a different model snapshot");
    auto result = newRun();
    solutions::BucklingOptions options;
    options.subcases = {subcase};
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    options.snapshotIndex = preload.index;
    options.requiredPreload = preload.subcase;
    options.trimPreloads.emplace(preload.subcase, preload.displacement.span());
    auto solved = solutions::BucklingSolution(*m_snapshot, options).run(*result.analysis.budget);
    for (const auto& solve : solved.solves)
    {
        result.analysis.eigenvalues.push_back(solve.eigenvalues);
        const auto& vectors = solve.eigenvectors;
        for (std::size_t k = 0; k < vectors.columns(); ++k)
            result.boxMotions.push_back(boxMotion(attachmentMotion(
                {vectors.data() + k * vectors.leadingDimension(), vectors.rows()})));
    }
    result.analysis.tables = std::move(solved.preload.arrays);
    result.analysis.diagnostics = std::move(solved.preload.diagnostics);
    result.analysis.diagnostics.insert(result.analysis.diagnostics.end(), solved.diagnostics.begin(), solved.diagnostics.end());
    return result;
}
} // namespace oos::api
