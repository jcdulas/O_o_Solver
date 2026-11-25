#include "AdaptiveShellStatics.hpp"

#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/mixedHybrid/families/MixedHybridShellFamily.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace oos::api
{
namespace
{
namespace mh = elements::mixedHybrid;

// Keep half of the work and include ties, independently of element numbering.
double halfWorkCutoff(std::span<double> scores)
{
    std::sort(scores.begin(), scores.end(), std::greater<>());
    double total = 0.0;
    for (double score : scores)
        total += score;
    double sum = 0.0;
    for (double score : scores)
    {
        sum += score;
        if (sum >= 0.5 * total)
            return score * (1.0 - 1.e-9);
    }
    return std::numeric_limits<double>::infinity();
}

void markLoadedShells(const assembly::ElementSet& elements, const elements::NodalDisplacements& field,
                      bool testInterior, const mh::ShellPart& part, std::set<std::int32_t>& marked,
                      std::set<std::int32_t>& interiorMarks, base::MemoryBudget& budget)
{
    std::vector<mh::ShellEdgeDefect> sides;
    double scale = 0.0;
    struct EdgeIndicator
    {
        double peak = 0.0;
        unsigned sides = 0;
        mh::SpaceVector work{};
        bool hasLine = false;
    };
    base::BudgetedBuffer<EdgeIndicator> storage(budget, part.topology().edges.size(), "shell edge indicators");
    auto edges = storage.span();
    base::BudgetedBuffer<double> work(budget, part.input().shells.size(), "shell interior correction work");
    double referenceEnergy = 0.0;
    for (const auto& line : part.input().lines)
        edges[part.topology().indexOf(line.element.grids[0], line.element.grids[1])].hasLine = true;
    double forceScale = 0.0;
    for (const auto& batch : elements.batches())
    {
        if (!dynamic_cast<const mh::MixedHybridShellFamily*>(batch.family))
            continue;
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
        {
            sides.clear();
            const auto cell = mh::shellDefects(batch.view, row, field, testInterior, sides);
            work[cell.shell] = cell.work;
            referenceEnergy = std::max(referenceEnergy, cell.referenceEnergy);
            for (const auto& side : sides)
            {
                auto& edge = edges[side.edge];
                edge.peak = std::max(edge.peak, side.defect);
                scale = std::max(scale, side.scale);
                ++edge.sides;
                edge.work = mh::add3(edge.work, side.membraneWork);
                forceScale = std::max(forceScale, side.membraneScale);
            }
        }
    }
    // Compare angles and generalized forces only after each has been scaled by
    // its own field in this load case. A stiffener carries part of the edge force.
    base::BudgetedBuffer<double> weights(budget, std::max(work.size(), edges.size()), "shell marking weights");
    std::size_t count = 0;
    for (auto& edge : edges)
    {
        edge.peak = scale > 0.0 ? edge.peak / scale : 0.0;
        if (edge.sides == 2 && !edge.hasLine && forceScale > 0.0)
            edge.peak = std::max(edge.peak, mh::norm3(edge.work) / forceScale);
        if (edge.peak > 1.e-6)
            weights[count++] = edge.peak * edge.peak;
    }
    const double edgeCutoff = halfWorkCutoff(weights.span().first(count));
    for (std::size_t edge = 0; edge < edges.size(); ++edge)
        if (edges[edge].peak * edges[edge].peak >= edgeCutoff)
            for (const auto shell : part.topology().edgeElements[edge])
                marked.insert(part.input().shells[shell].elementId);
    count = 0;
    for (double value : work.span())
        if (value > 1.e-18 * referenceEnergy)
            weights[count++] = value;
    const double cellCutoff = halfWorkCutoff(weights.span().first(count));
    for (std::size_t shell = 0; shell < work.size(); ++shell)
        if (work[shell] >= cellCutoff)
            interiorMarks.insert(part.input().shells[shell].elementId);
}

// Construct the model only in a new index. Its DOF numbering and its matrices
// then belong to this solve, including when two analyses share the snapshot.
const mh::MixedHybridModel& installLayout(solutions::StaticOptions& options, const input::ModelSnapshot& snapshot,
                                          const std::set<std::int32_t>* quarticShells, base::MemoryBudget& budget)
{
    options.snapshotIndex = std::make_shared<elements::SnapshotIndex>(snapshot, budget);
    return options.snapshotIndex->computedOnce<mh::MixedHybridModel>(
        [&] { return mh::MixedHybridModel(*options.snapshotIndex, quarticShells); });
}
} // namespace

solutions::StaticResults runStaticSolution(const input::ModelSnapshot& snapshot, solutions::StaticOptions options,
                                           base::MemoryBudget& budget,
                                           std::optional<std::vector<std::int32_t>>& selected,
                                           std::shared_ptr<const elements::SnapshotIndex>* finalIndex)
{
    const auto degree = snapshot.hasTable(input::TableRole::Params)
                            ? snapshot.params().find(mh::MIXED_HYBRID_DEGREE_PARAMETER)
                            : std::nullopt;
    // Condensation and streamed recovery use the layout of their plan throughout.
    if (options.isStreamed || options.condensation || !options.externalUnits.empty() || !degree ||
        degree->text != mh::MIXED_HYBRID_AUTOMATIC_DEGREE ||
        !elements::ElementRegistry::forModel(snapshot).isMixedHybrid())
        return solutions::StaticSolution(snapshot, options).run(budget);
    const auto* part = installLayout(options, snapshot, nullptr, budget).shellPart();
    if (!part || !part->usable())
    {
        if (finalIndex) *finalIndex = options.snapshotIndex;
        return solutions::StaticSolution(snapshot, options).run(budget);
    }
    std::set<std::int32_t> pilot;
    const auto& shells = part->input().shells;
    for (std::size_t i = 0; i < shells.size(); ++i)
    {
        const auto id = shells[i].elementId;
        if (part->shellDegree(i) == 4)
            pilot.insert(id);
    }
    auto quartic = pilot;
    std::set<std::int32_t> interiorMarks;
    if (pilot.size() < shells.size())
        options.observeDisplacements = [&](const auto& set, const auto& field, bool testInterior)
        { markLoadedShells(set, field, testInterior, *part, quartic, interiorMarks, budget); };
    auto result = solutions::StaticSolution(snapshot, options).run(budget);
    const auto traceCount = quartic.size();
    quartic.insert(interiorMarks.begin(), interiorMarks.end());
    const auto pilotEquations = result.largestFsetSize;
    const auto added = quartic.size() - pilot.size();
    if (added > 0)
    {
        // Release pilot results, model and matrix caches before building the final layout.
        result = {};
        options.observeDisplacements = {};
        options.snapshotIndex.reset();
        installLayout(options, snapshot, &quartic, budget);
        result = solutions::StaticSolution(snapshot, options).run(budget);
    }
    if (finalIndex) *finalIndex = options.snapshotIndex;
    selected.emplace(quartic.begin(), quartic.end());
    base::Diagnostic info;
    info.code = "mixed-hybrid-shell-adaptation";
    info.severity = base::Severity::Information;
    info.message =
        "Loaded shell checks selected " + std::to_string(added) +
        " additional quartic shells; selected by interior equilibrium: " + std::to_string(interiorMarks.size()) +
        "; additional to trace selection: " + std::to_string(quartic.size() - traceCount) +
        "; pilot equations: " + std::to_string(pilotEquations) +
        "; final equations: " + std::to_string(result.largestFsetSize) + "; static solves: " + (added ? "2" : "1");
    result.diagnostics.push_back(std::move(info));
    return result;
}
} // namespace oos::api
