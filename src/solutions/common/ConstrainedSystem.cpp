#include "oos/solutions/common/ConstrainedSystem.hpp"

#include "oos/assembly/assemble/StiffnessAssembler.hpp"
#include "oos/assembly/constraints/MsetCouplings.hpp"
#include "oos/assembly/constraints/MsetElimination.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/assembly/singularities/NodeDiagonalBlockSet.hpp"
#include "oos/base/threading/Stopwatch.hpp"
#include "oos/solutions/common/RunChecks.hpp"

#include <algorithm>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace oos::solutions
{

ConstrainedSystem::ConstrainedSystem(const AnalysisModel& model, const ConstraintSelection& selection,
                                     base::MemoryBudget& budget, base::ThreadPool* pool,
                                     const base::CancellationToken* cancellation, base::DiagnosticSink& sink,
                                     const ConstrainedSystemOptions& options)
    : m_model(&model), m_supportDofs(options.choosesSupports ? std::vector<base::DofIndex>{} : options.supportDofs)
{
    base::Stopwatch watch;
    const input::ModelSnapshot& snapshot = model.snapshot();
    const assembly::DofMap& dofMap = model.dofMap();
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* fastenerPiercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;

    m_mpc = std::make_unique<assembly::MpcEquations>(constraints, dofMap, selection.mpc, budget, sink);
    m_fasteners = std::make_unique<assembly::FastenerConstraints>(fastenerPiercings, dofMap, budget, sink);
    if (options.spc != nullptr)
    {
        m_spc = options.spc;
    }
    else
    {
        m_ownSpc = std::make_unique<assembly::SpcSet>(constraints, snapshot.findLoads(), dofMap, selection.spc,
                                                      selection.load, budget, sink);
        m_spc = m_ownSpc.get();
    }
    // The glue leaves out the grid points the other constraints hold or make dependent: built after them.
    const assembly::GlueContext glueContext{rigidElements, m_mpc.get(), m_fasteners.get(), m_spc->sbDofs(),
                                            m_spc->sgDofs()};
    m_glue = std::make_unique<assembly::GlueConstraints>(snapshot, selection.contact, selection.subcase, dofMap,
                                                         glueContext, budget, sink);
    m_gm = std::make_shared<assembly::RigidElimination>(rigidElements, m_mpc.get(), dofMap, budget, sink,
                                                        m_fasteners.get(), m_glue.get());
    // The interior DOFs of the external units join the permanent constraints (GRID PS): held in every subcase.
    if (!options.heldDofs.empty())
    {
        m_permanentDofs.assign(m_spc->sgDofs().begin(), m_spc->sgDofs().end());
        m_permanentDofs.insert(m_permanentDofs.end(), options.heldDofs.begin(), options.heldDofs.end());
    }
    const std::span<const base::DofIndex> permanentDofs =
        options.heldDofs.empty() ? m_spc->sgDofs() : std::span<const base::DofIndex>(m_permanentDofs);
    const auto makeSets = [&]
    {
        return std::make_unique<assembly::DofSets>(
            dofMap, assembly::DofSetInputs{m_gm->mDofs(), m_spc->sbDofs(), permanentDofs, {}, m_supportDofs}, budget,
            sink);
    };
    m_sets = makeSets();
    stopOnErrors(sink, "the assembly");
    m_timings.constraints = watch.lap();

    const assembly::MsetCouplings couplings(model.connectivity().groups(), *m_gm, dofMap, budget);
    std::vector<linalg::NodeGroupList> groupLists{model.connectivity().groups(), couplings.groups()};
    if (options.extraCouplings.groupCount() > 0)
        groupLists.push_back(options.extraCouplings);
    m_graph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), groupLists, budget, pool);
    std::optional<assembly::NodeDiagonalBlockSet> givenBlocks;
    if (options.diagonalBlocks)
    {
        m_timings.graph = watch.lap();
        givenBlocks.emplace(options.diagonalBlocks(*m_gm, *m_sets));
        m_timings.stiffness = watch.lap();
    }
    else
    {
        if (options.structures != nullptr)
        {
            const StructurePattern taken = options.structures->pattern(*m_graph, model.nodeDofCounts());
            m_pattern = taken.kept->lend(budget, "sparse pattern (kept)");
            m_isPatternReused = taken.isReused;
        }
        else
        {
            m_pattern = std::make_shared<const linalg::SparsePattern>(*m_graph, model.nodeDofCounts(), budget);
        }
        m_timings.graph = watch.lap();
        // PERFORMANCE: Kgg and Knn zeroed on the pool (their 2 × 345 MB at 1 M DOFs took 0.09 s of first touches on one
        // thread).
        m_kgg = std::make_shared<linalg::BlockSparseMatrix>(*m_pattern, budget, "stiffness Kgg", pool);
        assembly::StiffnessAssembler(model.elementSet(), dofMap)
            .assemble(*m_kgg, pool, cancellation, options.keptStiffness);
        if (options.keptStiffness != nullptr && options.releasesKeptStiffness)
            options.keptStiffness->release();
        if (options.addStiffness)
            options.addStiffness(*m_kgg);
        m_knn = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "stiffness Knn", pool);
        assembly::MsetElimination::reduce(*m_kgg, *m_graph, *m_gm, dofMap, *m_knn);
        m_timings.stiffness = watch.lap();
    }

    // The grid point singularity processor on Knn (always on the stiffness, whatever the analysis).
    const assembly::NodeDiagonalBlockSet diagonalBlocks =
        givenBlocks ? std::move(*givenBlocks) : assembly::NodeDiagonalBlockSet(*m_knn, dofMap, budget);
    const assembly::GridPointSingularities singularities(
        dofMap, *m_sets, diagonalBlocks.blocks(),
        assembly::SingularityOptions::fromParams(model.params(), selection.subcase, sink), budget, sink);
    // The DOFs the processor must leave free (ConstrainedSystemOptions::uncheckedDofs) taken out of its findings.
    const auto isChecked = [&](base::DofIndex dof)
    { return !std::binary_search(options.uncheckedDofs.begin(), options.uncheckedDofs.end(), dof); };
    std::vector<base::DofIndex> autoSpcDofs; // bounded by the singular DOFs
    for (const base::DofIndex dof : singularities.autoSpcDofs())
        if (isChecked(dof))
            autoSpcDofs.push_back(dof);
    m_solveSets = std::make_shared<assembly::DofSets>(m_sets->withAutoSpc(autoSpcDofs));
    if (options.choosesSupports)
    {
        // The supports of INREL = −2 among the free DOFs left by the singularity processor.
        const assembly::RigidBodyModes geometric(dofMap, options.referencePoint, budget);
        m_supportDofs = assembly::chooseAutomaticSupports(geometric, *m_solveSets);
        m_sets = makeSets();
        m_solveSets = std::make_shared<assembly::DofSets>(m_sets->withAutoSpc(autoSpcDofs));
    }
    std::ostringstream table;
    if (options.uncheckedDofs.empty())
        singularities.writeTable(table);
    else
    {
        std::vector<assembly::SingularityRow> rows; // bounded by the singular DOFs
        for (const assembly::SingularityRow& row : singularities.rows())
            if (isChecked(row.dof))
                rows.push_back(row);
        assembly::writeSingularityTable(table, rows, singularities.options());
    }
    m_singularityTable = table.str();

    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    m_isConstrained = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "constrained DOFs");
    for (std::size_t dof = 0; dof < gSize; ++dof)
    {
        const auto index = static_cast<base::DofIndex>(dof);
        m_isConstrained[dof] = m_solveSets->contains(assembly::DofSet::S, index) ||
                                       m_solveSets->exclusiveSetOf(index) == assembly::ExclusiveSet::R
                                   ? 1
                                   : 0;
    }
    m_timings.singularities = watch.lap();
}

linalg::BlockSparseMatrix ConstrainedSystem::reduce(const linalg::BlockSparseMatrix& gg, base::MemoryBudget& budget,
                                                    const char* label) const
{
    linalg::BlockSparseMatrix nn(*m_pattern, budget, label);
    assembly::MsetElimination::reduce(gg, *m_graph, *m_gm, m_model->dofMap(), nn);
    return nn;
}

} // namespace oos::solutions
