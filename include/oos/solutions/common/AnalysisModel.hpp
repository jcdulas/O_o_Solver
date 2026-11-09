#pragma once

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/ElementMatrixStore.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/reduction/external/ReplacedElements.hpp"
#include "oos/solutions/common/AnalysisTimings.hpp"
#include "oos/solutions/common/RunCatalog.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace oos::solutions
{

// What a solution adds to the building of its AnalysisModel (the static solution; the eigenvalue solutions take the
// defaults).
struct AnalysisModelOptions
{
    // Called after the element check, before the model stops on errors: the caller's checks of the same pass (the
    // static solution's subcases and output plans), so that one SolverError names every problem of the model.
    std::function<void(base::DiagnosticSink&)> checkCases;
    // Called once the element set and the DOF map exist, before the connectivity pass (the first pass that computes
    // element matrices): the static solution's output volume estimate, which refuses an output that would not fit
    // before any element matrix is computed.
    std::function<void(const assembly::ElementSet&, const assembly::DofMap&)> checkBeforeConnectivity;
    // The share of the memory budget left before the connectivity pass that the element stiffness matrices it
    // computes may take, kept for the first assembly of Kgg (assembly::ElementMatrixStore, keptStiffness()); 0: none
    // kept.
    double keptStiffnessBudgetFraction = 0.0;
    // The elements external units replace, chosen once the element set and the DOF map exist (the zones are resolved on
    // them), before checkBeforeConnectivity: the model's element set is then made again on reduction::ReplacedElements,
    // so that every pass leaves their stiffness, mass and results out. The element set the callable gets is dropped
    // after it: nothing may keep it. Empty result: none.
    std::function<std::vector<base::EntityId>(const assembly::ElementSet&, const assembly::DofMap&)> replacedElements;
};

// The model of an analysis, the part that does not depend on the subcase: the support check of the analysis
// (UnsupportedCardError listing every refusal; the ignored cards are warnings of the sink), the element set and its
// check (SolverError listing every problem, with those of AnalysisModelOptions::checkCases), the DOF map and the
// element connectivity. Shared by the static solution and the eigenvalue solutions (SOL 101, 103, 105).
//
// Thread-safety: immutable once built, except the store of the kept element matrices (keptStiffness(), used and
// released by the first assembly on the calling thread). Ownership: owns the index, the element set, the DOF map and
// the connectivity (budgeted); the index and the DOF map are shared with what outlives the model (a FactorisedSet kept
// by the static solution); borrows the snapshot, which must outlive it and them. Not movable (its parts refer to each
// other).
class AnalysisModel
{
public:
    AnalysisModel(const input::ModelSnapshot& snapshot, AnalysisKind analysis, base::MemoryBudget& budget,
                  base::ThreadPool* pool, const base::CancellationToken* cancellation, base::DiagnosticSink& sink,
                  const AnalysisModelOptions& options = {});
    AnalysisModel(const AnalysisModel&) = delete;
    AnalysisModel& operator=(const AnalysisModel&) = delete;
    AnalysisModel(AnalysisModel&&) = delete;
    AnalysisModel& operator=(AnalysisModel&&) = delete;
    ~AnalysisModel() = default;

    [[nodiscard]] const input::ModelSnapshot& snapshot() const noexcept { return *m_snapshot; }
    [[nodiscard]] const elements::SnapshotIndex& index() const noexcept { return *m_index; }
    [[nodiscard]] const assembly::ElementSet& elementSet() const noexcept { return *m_elementSet; }
    [[nodiscard]] const assembly::DofMap& dofMap() const noexcept { return *m_dofMap; }
    [[nodiscard]] const assembly::ElementConnectivity& connectivity() const noexcept { return *m_connectivity; }
    [[nodiscard]] std::span<const std::uint8_t> nodeDofCounts() const noexcept { return m_nodeDofCounts.span(); }
    [[nodiscard]] const input::ParamTable* params() const noexcept { return m_params; }
    // The index and the DOF map, for what keeps them beyond the model (the DOF map refers to the index's points).
    [[nodiscard]] std::shared_ptr<const elements::SnapshotIndex> sharedIndex() const noexcept { return m_index; }
    [[nodiscard]] std::shared_ptr<const assembly::DofMap> sharedDofMap() const noexcept { return m_dofMap; }
    // The element stiffness matrices the connectivity pass kept (empty without keptStiffnessBudgetFraction).
    [[nodiscard]] assembly::ElementMatrixStore& keptStiffness() noexcept { return m_keptStiffness; }
    // The elements external units replace (AnalysisModelOptions::replacedElements), or null.
    [[nodiscard]] const reduction::ReplacedElements* replacedElements() const noexcept { return m_replaced.get(); }
    // The time of its checks and of its connectivity pass (AnalysisPartTimings::checks, connectivity).
    [[nodiscard]] const AnalysisPartTimings& timings() const noexcept { return m_timings; }

private:
    const input::ModelSnapshot* m_snapshot;
    elements::ElementRegistry m_registry;
    // Declared before the element set, which borrows its families.
    std::unique_ptr<reduction::ReplacedElements> m_replaced;
    std::shared_ptr<elements::SnapshotIndex> m_index;
    std::unique_ptr<assembly::ElementSet> m_elementSet;
    std::shared_ptr<assembly::DofMap> m_dofMap;
    assembly::ElementMatrixStore m_keptStiffness;
    std::unique_ptr<assembly::ElementConnectivity> m_connectivity;
    base::BudgetedBuffer<std::uint8_t> m_nodeDofCounts;
    const input::ParamTable* m_params = nullptr;
    AnalysisPartTimings m_timings;
};

} // namespace oos::solutions
