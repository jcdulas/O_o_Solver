#include "oos/api/Analyses.hpp"

#include "oos/api/Threading.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/platform/MemoryInfo.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/linalg/dense/KernelSelection.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"
#include "oos/solutions/sol103/ModesSolution.hpp"
#include "oos/solutions/sol105/BucklingSolution.hpp"

namespace oos::api
{

namespace
{

constexpr const char* PROGRAM_VERSION = "0.2.0";

solutions::AnalysisKind kindOf(Analysis analysis)
{
    switch (analysis)
    {
    case Analysis::Statics:
        return solutions::AnalysisKind::Statics;
    case Analysis::Modes:
        return solutions::AnalysisKind::Modes;
    case Analysis::Buckling:
        return solutions::AnalysisKind::Buckling;
    case Analysis::Aeroelastic:
        return solutions::AnalysisKind::StaticAeroelastic;
    }
    return solutions::AnalysisKind::Statics;
}

// The budget of the run: what the caller gives, or the memory left (in WebAssembly, what the module's memory may still
// grow to).
std::uint64_t budgetBytes(const AnalysisSettings& settings)
{
    if (settings.memoryBytes > 0)
        return settings.memoryBytes;
    const std::uint64_t available = base::MemoryInfo::query().availableBytes;
    return available > 0 ? available : base::MemoryBudget::UNLIMITED;
}

AnalysisOutput runStatics(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                          base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::StaticOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    solutions::StaticResults results = solutions::StaticSolution(snapshot, options).run(budget);
    AnalysisOutput output;
    output.diagnostics = std::move(results.diagnostics);
    output.tables = std::move(results.arrays);
    return output;
}

AnalysisOutput runModes(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                        base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::ModesOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    solutions::ModesResults results = solutions::ModesSolution(snapshot, options).run(budget);
    AnalysisOutput output;
    for (const solutions::ModesSubcaseResult& subcase : results.subcases)
        output.eigenvalues.push_back(subcase.eigenvalues);
    output.diagnostics = std::move(results.diagnostics);
    output.tables = std::move(results.arrays);
    return output;
}

AnalysisOutput runBuckling(const input::ModelSnapshot& snapshot, const AnalysisSettings& settings,
                           base::MemoryBudget& budget, base::ThreadPool* pool)
{
    solutions::BucklingOptions options;
    options.subcases = settings.subcases;
    options.pool = pool;
    options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
    solutions::BucklingResults results = solutions::BucklingSolution(snapshot, options).run(budget);
    AnalysisOutput output;
    for (const solutions::BucklingSolve& solve : results.solves)
        output.eigenvalues.push_back(solve.eigenvalues);
    // The preload's warnings first, then those of the buckling part: the order in which the run made them.
    output.diagnostics = std::move(results.preload.diagnostics);
    output.diagnostics.insert(output.diagnostics.end(), results.diagnostics.begin(), results.diagnostics.end());
    // The static tables of the preload and the buckling shapes are in the same arrays (BucklingSolution adds the shapes
    // to the preload's tables).
    output.tables = std::move(results.preload.arrays);
    return output;
}

} // namespace

std::string programName()
{
    return std::string("O_o_Solver ") + PROGRAM_VERSION;
}

std::vector<ElementFormulations> elementFormulations()
{
    const elements::ElementRegistry registry = elements::ElementRegistry::standard();
    std::vector<ElementFormulations> list;
    for (const elements::ElementFamily* family : registry.families())
    {
        ElementFormulations entry;
        entry.card = std::string(family->card());
        for (const std::string_view name : registry.formulationsOf(family->card()))
            entry.formulations.emplace_back(name);
        list.push_back(std::move(entry));
    }
    return list;
}

SupportCheck checkSupport(const input::ModelSnapshot& snapshot, Analysis analysis)
{
    const base::SupportReport report =
        solutions::checkSupport(snapshot, elements::ElementRegistry::forModel(snapshot), kindOf(analysis));
    // Diagnostics keep the neutral names used by the model and the tables.

    return {report.supported, report.refusals, report.ignored};
}

AnalysisOutput runAnalysis(const input::ModelSnapshot& snapshot, Analysis analysis, const AnalysisSettings& settings)
{
    const auto pool = acquireThreadPool(settings.threads);
    auto budget = std::make_shared<base::MemoryBudget>(budgetBytes(settings));
    AnalysisOutput output;
    switch (analysis)
    {
    case Analysis::Statics:
        output = runStatics(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Modes:
        output = runModes(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Buckling:
        output = runBuckling(snapshot, settings, *budget, pool.get());
        break;
    case Analysis::Aeroelastic:
    {
        solutions::TrimOptions options;
        options.subcases = settings.subcases;
        options.links = aero::trimLinks(snapshot);
        options.links.insert(options.links.end(), settings.links.begin(), settings.links.end());
        options.pool = pool.get();
        options.kernel = linalg::KernelSelection::selectForThisProcess().kind();
        output.aeroelastic = std::make_shared<solutions::TrimResults>(
            solutions::TrimSolution(snapshot, options).run(*budget));
        output.diagnostics = output.aeroelastic->diagnostics;
        output.tables = std::move(output.aeroelastic->arrays);
        break;
    }
    }
    output.budget = std::move(budget);
    return output;
}

} // namespace oos::api
