#include "oos/solutions/sol101/StaticSolution.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/dofs/Uset.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/errors/OperationCancelledError.hpp"
#include "oos/base/errors/SingularityError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/contract/ContractVersion.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/IterativeRefinement.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/ResultCombination.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/output/volume/OutputVolumeEstimate.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/KeptRelief.hpp"
#include "oos/solutions/common/MechanismShapes.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/BasisFixedRows.hpp"
#include "oos/solutions/sol101/CondensedLoadBasis.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::solutions
{

namespace
{

// The external work of the equilibrium check (with EPSILON) is half the work of the loads on the displacements of the
// solve, ½·uᵀ·P on the l set.
constexpr double EXTERNAL_WORK_FACTOR = 0.5;
// The share of the memory budget left at the start that the result tables may take when the caller gives no limit:
// the other half for the factorisation and the recovery.
constexpr double OUTPUT_BUDGET_FRACTION = 0.5;
// The subcases whose tables a streamed run holds at once: the one being recovered and those the destination has not
// taken yet (a caller queues up to four).
constexpr std::uint64_t STREAMED_SUBCASES_IN_FLIGHT = 5;
// The fewest cases of a constraint set that share a thermal case's temperature sets for its element tables to be
// recovered by blocks: each set of temperatures costs the rows at rest of the
// block once more (a recovery at zero displacements), which these cases share. Calibrated on a thermal plate
// (40 000 ShellQuad4, two temperature sets, six threads): a case costs
// 0.33 s recovered alone, 0.04 s by blocks once the slopes are made, and each set of temperatures its rows at rest,
// about one recovery alone: a set pays from two cases; 4 leaves a margin for the slopes the run shares.
constexpr std::size_t THERMAL_GROUP_MIN_CASES = 4;
// The vectors a subcase that a SUBCOM combines keeps (StoredCase: displacements, applied loads, SPC and MPC forces, as
// doubles; its constrained flags, one byte, counted as one more), and a kept displacement vector.
constexpr std::uint64_t STORED_CASE_VECTORS = 5;
// The share of the memory budget left at the start that the element stiffness matrices kept by the connectivity pass
// may take (assembly::ElementMatrixStore): a ShellQuad4 keeps 4.8 KB, so 166 464 of them (the 1 M DOF plate) 0.8 GB,
// released before the factorisation. Measured on the 1 M DOF plate (6 threads, three runs each): setup + assembly
// 1.56-1.59 s without the store, 1.15-1.25 s with it (the assembly 0.68-0.73 s -> 0.32-0.41 s); the budget peak
// unchanged (5.12 GB, in the factorisation).
constexpr double KEPT_STIFFNESS_BUDGET_FRACTION = 0.25;
// PARAM GRDPNT: −1 (the default) no grid point weight generator; 0 about the origin; a grid point about it.
constexpr std::int32_t NO_WEIGHT_REFERENCE_NODE = -1;
// The stiffness products of the basis vectors made together (basisProductGroup): at most this share of the memory left
// for their blocks (the products and the l-set displacements: two g-set vectors each), and at most this many vectors
// (the reads of the matrix are then a small part of the products; more only holds more memory).
constexpr std::uint64_t BASIS_PRODUCT_SHARE = 16;
constexpr std::uint64_t BASIS_PRODUCT_GROUP_MAX = 64;

// The quantities combined from the basis for each case (ResultCombination): displacements, applied loads, SPC
// forces, MPC forces, the reduced loads of the solve and its residual, and the displacements before the rigid motion
// of INREL = −2 is removed.
enum Quantity : std::size_t
{
    Displacements,
    Applied,
    SpcForces,
    MpcForces,
    ReducedLoads,
    Residuals,
    SolvedDisplacements
};
constexpr std::size_t QUANTITY_COUNT = 7;

class Stopwatch
{
public:
    // The seconds since the last call (or the construction).
    double lap()
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - m_last).count();
        m_last = now;
        return seconds;
    }

private:
    std::chrono::steady_clock::time_point m_last = std::chrono::steady_clock::now();
};

void checkCancelled(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

base::ProgressCounter* beginStage(ProgressMonitor* monitor, const char* stage, std::uint64_t total)
{
    return monitor == nullptr ? nullptr : &monitor->beginStage(stage, total);
}

// The pivot policy of a subcase: PARAM MAXRATIO (default 1e7) and BAILOUT (default 0).
linalg::PivotPolicy pivotPolicyOf(const input::ParamTable* params, std::optional<std::int32_t> subcase,
                                  base::DiagnosticSink& sink)
{
    linalg::PivotPolicy policy;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", subcase, sink))
    {
        if (*ratio > 0.0)
            policy.maxRatio = *ratio;
        else
            sink.report(assembly::badParamDiagnostic("MaxDiagonalRatio", "a positive real number"));
    }
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", subcase, sink))
        policy.bailout = *bailout;
    return policy;
}

// PARAM INREL: 0 (default), −1, −2.
assembly::InertiaReliefMode reliefModeOf(const input::ParamTable* params, std::optional<std::int32_t> subcase,
                                         base::DiagnosticSink& sink)
{
    const std::optional<std::int32_t> value = assembly::findIntegerParam(params, "InertiaRelief", subcase, sink);
    if (!value || *value == 0)
        return assembly::InertiaReliefMode::None;
    if (*value == -1)
        return assembly::InertiaReliefMode::OnSupports;
    if (*value == -2)
        return assembly::InertiaReliefMode::Automatic;
    sink.report(assembly::badParamDiagnostic("InertiaRelief", "0, -1 or -2"));
    return assembly::InertiaReliefMode::None;
}

base::Diagnostic subcaseProblem(base::EntityId subcase, base::Severity severity, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// y = A·x for the symmetric matrix of a BlockSparseMatrix, x and y of its DOF count (serial, fixed order).
base::BudgetedBuffer<double> multiply(const linalg::BlockSparseMatrix& matrix, std::span<const double> x,
                                      base::MemoryBudget& budget, const char* label)
{
    base::BudgetedBuffer<double> y(budget, x.size(), label);
    matrix.multiply(x, y.span());
    return y;
}

// The basis vectors made by one piece of the parallel loop that makes their loads: few enough for the
// pieces to keep six threads busy with 64 vectors, enough for their three g-set work vectors (24 MB at 1 M DOFs) to be
// made once for several vectors.
constexpr std::size_t BASIS_VECTORS_PER_PIECE = 4;
// The parallel loop of the basis vectors runs when its work vectors, one piece per thread, take at most this fraction
// of the memory left (1/8, as the kept element stiffness of the full path's first assembly).
constexpr std::uint64_t BASIS_WORK_MEMORY_DIVISOR = 8;

// The basis vectors whose stiffness products are made together (BlockSparseMatrix::multiply of a block): as many as
// their blocks (the products, and the l-set displacements for the residuals) fit in BASIS_PRODUCT_SHARE of the memory
// left, at least one and at most BASIS_PRODUCT_GROUP_MAX. The bits do not depend on the group (each column's products
// are its own), so the memory may decide it.
std::size_t basisProductGroup(std::size_t gSize, std::size_t basisSize, bool hasResiduals,
                              const base::MemoryBudget& budget)
{
    const std::uint64_t perVector =
        static_cast<std::uint64_t>(std::max<std::size_t>(gSize, 1)) * sizeof(double) * (hasResiduals ? 2U : 1U);
    const std::uint64_t fits = budget.available() / BASIS_PRODUCT_SHARE / perVector;
    return static_cast<std::size_t>(std::clamp<std::uint64_t>(
        fits, 1, std::min<std::uint64_t>(BASIS_PRODUCT_GROUP_MAX, std::max<std::size_t>(basisSize, 1))));
}

// The l-set factorisation; a stiffness that cannot be factorised (a mechanism, a free body: a failed ratio test or a
// zero pivot) stops the run with its SingularityError carrying the zero-energy shapes:
// MechanismShapes factorises the slightly shifted stiffness with the failed pivots kept (BAILOUT −1) and finds the
// shapes from that factor. When the shifted factorisation fails too, or the shapes do not fit the budget, the error is
// the first one, without shapes.
std::unique_ptr<FsetSolver> factoriseWithShapes(const linalg::BlockSparseMatrix& knn, const linalg::NodeGraph& graph,
                                                const assembly::DofMap& dofMap, const assembly::DofSets& sets,
                                                const linalg::DenseKernels& kernels, const FsetSolverOptions& options,
                                                const assembly::RigidElimination& gm, base::MemoryBudget& budget,
                                                base::ThreadPool* pool)
{
    try
    {
        return std::make_unique<FsetSolver>(knn, graph, dofMap, sets, kernels, options, budget, pool);
    }
    catch (const base::SingularityError& error)
    {
        if (error.dofs().empty() || !error.mechanisms().empty())
            throw;
        std::vector<base::MechanismShape> shapes;
        try
        {
            FsetSolverOptions kept = options;
            kept.pivotPolicy.bailout = -1;
            kept.progress = nullptr;
            const linalg::BlockSparseMatrix shifted = MechanismShapes::shiftedStiffness(knn, budget);
            const FsetSolver bailedOut(shifted, graph, dofMap, sets, kernels, kept, budget, pool);
            shapes = MechanismShapes::compute(bailedOut, knn, gm, dofMap,
                                              static_cast<std::size_t>(bailedOut.pivotDiagnostics().issueCount()),
                                              budget, options.cancellation);
        }
        catch (const base::OperationCancelledError&)
        {
            throw;
        }
        catch (const base::SolverError&)
        {
            throw error;
        }
        if (shapes.empty())
            throw;
        // The message first: the shapes are moved into the error.
        std::string message = base::concatenate(
            {error.what(), "\n", std::to_string(shapes.size()),
             shapes.size() == 1 ? " zero-energy shape (mechanism) is" : " zero-energy shapes (mechanisms) are",
             " attached to this error: the motions to constrain"});
        throw base::SingularityError(message, error.dofs(), std::move(shapes));
    }
}

// The g-set vectors of one case kept for the SUBCOMs that combine it.
struct StoredCase
{
    base::EntityId label = 0;
    base::BudgetedBuffer<double> displacements;
    base::BudgetedBuffer<double> applied;
    base::BudgetedBuffer<double> spcForces;
    base::BudgetedBuffer<double> mpcForces;
    base::BudgetedBuffer<assembly::RelationForce> mpcRelationForces; // with the split multiPointConstrForce rows
    base::BudgetedBuffer<std::uint8_t> isConstrained;
    // The condensed path: the resultant of the reactions of the units not recovered and their grid point force rows.
    std::array<double, 6> unrecoveredReactions{};
    std::vector<output::StiffnessRecovery::UnitRow> unitRows; // bounded by the boundaries of those units
    // Their boundary forces (the contour forces of a zone that is one of them).
    std::vector<CondensedStatics::UnitBoundaryForces> unitForces; // bounded by the units
};

// The result tables of one subcase, gathered in the order the Case Control gives the subcases.
struct SubcaseTables
{
    std::vector<output::NodalResultTable> nodal;     // bounded by the kinds
    std::vector<output::ElementResultTable> element; // bounded by the element types × kinds
    std::vector<output::GridPointForceTable> gridPointForces;
    std::optional<output::SubcaseCheck> check;
    std::optional<MonitorPoint3Loads> monitorPoints3; // with MonitorNodeForceSum3
};

// One case to recover: its g-set vectors (spans into the combined matrices or the SUBCOM sums) and what it reads.
struct CaseState
{
    const StaticSubcase* subcase = nullptr;
    const output::OutputPlan* plan = nullptr;
    std::span<const double> displacements;
    // The displacements the element results are made from, a gauge of the same deformation (a free body's as
    // solved on its conditioned supports); empty: `displacements`.
    std::span<const double> elementDisplacements;
    [[nodiscard]] std::span<const double> elementDisplacementsOrAll() const noexcept
    {
        return elementDisplacements.empty() ? displacements : elementDisplacements;
    }
    std::span<const double> applied;
    std::span<const double> spcForces;
    std::span<const double> mpcForces; // empty without an m set
    // The multiPointConstrForce rows split by relation (from an O_o_Reader of data contract 56): the relations' forces
    // (RigidElimination::relationForces; a SUBCOM the combination of its terms').
    bool isMpcSplit = false;
    base::BudgetedBuffer<assembly::RelationForce> mpcRelationForces;
    std::span<const std::uint8_t> isConstrained;
    std::vector<std::vector<elements::ElementLineLoad>> lineLoads; // bounded by the batches
    // The condensed path: the grid point forces of the units not recovered (null otherwise), and the resultant of their
    // reactions about the basic origin (by their equilibrium), added to the SPC force resultant.
    const output::StiffnessRecovery::CondensedUnits* gridPointForceUnits = nullptr;
    std::array<double, 6> unrecoveredReactions{};
    std::span<const CondensedStatics::UnitBoundaryForces> unitForces; // their boundary forces
};

// The cases of a constraint set whose element tables are recovered by blocks (ElementResultBuilder::buildCases): every
// case with an element table, but a thermal case whose temperature sets fewer than THERMAL_GROUP_MIN_CASES cases of the
// set share, and every case when the run recovers case by case (StaticOptions::isRecoveredCaseByCase). Their count
// decides the vectorised kinds for the whole set, so that the bits of a case depend on the cases asked, never on the
// batches or the memory.
// PERFORMANCE: the thermal and line-loaded cases recovered by blocks too, rather than case by case: 32 thermal cases of
// a 40 000 ShellQuad4 plate 10.6 → 2.4 s of recovery, 32 cases of a grillage of 45 300 bars and beams under
// BeamDistributedLoad 3.9 → 1.3 s.
struct BlockCases
{
    std::vector<std::uint8_t> isBlockCase;                  // per case of the group
    std::vector<output::ElementOutputSelection> selections; // per case of the group
    std::vector<output::CaseLoads> loads;                   // per case of the group (those of a block case)
    std::size_t count = 0;
    // What the loads borrow: the line loads of each block case that has some, the initial strains of each key of
    // temperature and deformation sets of the thermal block cases (InitialStrainKey).
    std::vector<std::vector<std::vector<elements::ElementLineLoad>>> lineLoads; // per case of the group
    std::vector<std::unique_ptr<assembly::TemperatureField>> temperatures;
};

BlockCases blockCasesOf(std::span<const std::size_t> group, const std::vector<StaticSubcase>& subcases,
                        const std::vector<output::OutputPlan>& plans, const assembly::LoadVectorBuilder& loadBuilder,
                        const input::ModelSnapshot& snapshot, const input::PointIndex& points, bool isCaseByCase,
                        base::MemoryBudget& budget)
{
    BlockCases blocks;
    const std::size_t caseCount = group.size();
    blocks.isBlockCase.assign(caseCount, 0);
    blocks.loads.resize(caseCount);
    blocks.lineLoads.resize(caseCount);
    // The thermal cases of each key of initial strains (TEMPERATURE(LOAD), TEMPERATURE(INITIAL), DEFORM).
    std::map<InitialStrainKey, std::size_t> sharing;
    for (const std::size_t s : group)
    {
        blocks.selections.push_back(output::elementSelectionOf(plans[s]));
        if (subcases[s].hasInitialStrains())
            ++sharing[initialStrainKeyOf(subcases[s])];
    }
    std::map<InitialStrainKey, const elements::ElementTemperatures*> fields;
    for (std::size_t g = 0; g < caseCount && !isCaseByCase; ++g)
    {
        const StaticSubcase& subcase = subcases[group[g]];
        const output::ElementOutputSelection& chosen = blocks.selections[g];
        if (!chosen.stress.isRequested && !chosen.strain.isRequested && !chosen.forces.isRequested)
            continue;
        if (subcase.hasInitialStrains())
        {
            const InitialStrainKey key = initialStrainKeyOf(subcase);
            if (sharing[key] < THERMAL_GROUP_MIN_CASES)
                continue;
            auto [found, isNew] = fields.try_emplace(key, nullptr);
            if (isNew)
            {
                // Its problems are reported by the recovery of each case (recoverCase), which reads the same sets.
                base::DiagnosticSink reportedByEachCase;
                blocks.temperatures.push_back(initialStrainsOf(snapshot, points, key, budget, reportedByEachCase));
                found->second = &blocks.temperatures.back()->temperatures();
            }
            blocks.loads[g].temperatures = found->second;
        }
        if (subcase.load)
        {
            std::vector<std::vector<elements::ElementLineLoad>> lineLoads = loadBuilder.lineLoadsOf(*subcase.load);
            if (std::any_of(lineLoads.begin(), lineLoads.end(), [](const auto& batch) { return !batch.empty(); }))
            {
                blocks.lineLoads[g] = std::move(lineLoads);
                blocks.loads[g].lineLoads = blocks.lineLoads[g];
            }
        }
        blocks.isBlockCase[g] = 1;
        ++blocks.count;
    }
    return blocks;
}

// The element tables of the block cases of a batch of cases (one column of `displacements` per case of the batch, the
// first `firstCase` of the group): consecutive block cases with one selection recovered together, at most `blockLimit`
// at a time; tables[c] receives those of case firstCase + c.
void recoverBlockCases(const output::ElementResultBuilder& builder, const BlockCases& blocks,
                       const std::vector<StaticSubcase>& subcases, std::span<const std::size_t> group,
                       std::size_t firstCase, linalg::ConstDenseMatrixView displacements, std::size_t blockLimit,
                       base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation,
                       std::vector<std::vector<output::ElementResultTable>>& tables)
{
    const std::size_t count = displacements.columns;
    tables.clear();
    tables.resize(count);
    std::size_t j = 0;
    while (j < count)
    {
        if (blocks.isBlockCase[firstCase + j] == 0)
        {
            ++j;
            continue;
        }
        std::size_t end = j + 1;
        while (end < count && end - j < blockLimit && blocks.isBlockCase[firstCase + end] != 0 &&
               blocks.selections[firstCase + end].selectsSameAs(blocks.selections[firstCase + j]))
            ++end;
        std::vector<std::int32_t> labels; // bounded by the block
        for (std::size_t i = j; i < end; ++i)
            labels.push_back(subcases[group[firstCase + i]].label);
        builder.buildCases(labels, displacements.block(0, j, displacements.rows, end - j), blocks.count, budget,
                           std::span(tables).subspan(j, end - j), pool, cancellation, &blocks.selections[firstCase + j],
                           std::span(blocks.loads).subspan(firstCase + j, end - j));
        j = end;
    }
}

// Adds a resultant given about the basic origin (forces F, moments M) to one about `referencePoint`: M_ref = M − r × F,
// on the diagonal of its rows (the forces in rows T1-T3, the moments in rows R1-R3) and in its totals. These are the
// condensed path's reactions of the units not recovered, known by their equilibrium as a total: their split by grid
// point and component needs the recovery.
void addResultant(output::LoadResultant& resultant, const std::array<double, 6>& aboutOrigin, base::Vec3 referencePoint)
{
    const base::Vec3 force{aboutOrigin[0], aboutOrigin[1], aboutOrigin[2]};
    const base::Vec3 moment =
        base::Vec3{aboutOrigin[3], aboutOrigin[4], aboutOrigin[5]} - base::cross(referencePoint, force);
    for (std::size_t k = 0; k < 3; ++k)
    {
        resultant.rows[k][k] += force[k];
        resultant.rows[3 + k][3 + k] += moment[k];
        resultant.totals[k] += force[k];
        resultant.totals[3 + k] += moment[k];
    }
}

// The zones, sections and cuts of the options that hold an element an external unit replaces (sorted): refused, naming
// them (their contour data, section and cut loads would read the elements the plugin's matrices stand for).
void refuseReplacedRequests(const std::vector<base::EntityId>& replaced, const StaticOptions& options,
                            base::DiagnosticSink& sink)
{
    const auto isReplaced = [&](std::int32_t elementId)
    { return std::binary_search(replaced.begin(), replaced.end(), elementId); };
    const auto refuse = [&](const std::string& what, std::int32_t elementId)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Error;
        diagnostic.code = "external-unit-request";
        diagnostic.card = "ZONE";
        diagnostic.message = base::concatenate(
            {what, " holds element ", std::to_string(elementId),
             ", which an external unit replaces: the plugin's matrices stand for it; take the contour data of the "
             "unit (StaticResults::externalUnits) or leave the element out of the request"});
        sink.report(diagnostic);
    };
    for (const output::ZoneDefinition& zone : options.zones.zones)
        for (const std::int32_t elementId : zone.elements)
            if (isReplaced(elementId))
            {
                refuse(std::string("the zone ").append(zone.name), elementId);
                break;
            }
    for (const output::SectionDefinition& section : options.sectionLoads.sections)
        for (const output::SectionMember& member : section.members)
            if (isReplaced(member.elementId))
            {
                refuse(std::string("section ").append(std::to_string(section.id)), member.elementId);
                break;
            }
    for (const output::CutDefinition& cut : options.sectionLoads.cuts)
        for (const std::int32_t elementId : cut.elements)
            if (isReplaced(elementId))
            {
                refuse(std::string("cut ").append(std::to_string(cut.id)), elementId);
                break;
            }
}

// The grid point force rows of a case on the condensed path with external units: those of the condensed
// units not recovered and those of the external units, their elements left out of the element rows.
struct MergedUnitRows
{
    std::vector<base::EntityId> excluded;                 // sorted; bounded by the units' and the zones' elements
    std::vector<output::StiffnessRecovery::UnitRow> rows; // sorted by node; bounded by the boundaries and contours
    output::StiffnessRecovery::CondensedUnits view;
};

MergedUnitRows mergeUnitRows(const output::StiffnessRecovery::CondensedUnits* condensed,
                             std::span<const base::EntityId> replaced,
                             std::span<const output::StiffnessRecovery::UnitRow> externalRows)
{
    MergedUnitRows merged;
    merged.excluded.assign(replaced.begin(), replaced.end());
    merged.rows.assign(externalRows.begin(), externalRows.end());
    if (condensed != nullptr)
    {
        merged.excluded.insert(merged.excluded.end(), condensed->excludedElements.begin(),
                               condensed->excludedElements.end());
        merged.rows.insert(merged.rows.end(), condensed->unitRows.begin(), condensed->unitRows.end());
    }
    std::sort(merged.excluded.begin(), merged.excluded.end());
    std::stable_sort(merged.rows.begin(), merged.rows.end(),
                     [](const output::StiffnessRecovery::UnitRow& a, const output::StiffnessRecovery::UnitRow& b)
                     { return a.node < b.node; });
    return merged;
}

// Whether a zone is a unit the plan condenses (ZoneDefinition::unit): its contour data then come from the unit's
// boundary.
bool isCondensedUnitZone(const output::ZoneDefinition& zone, const reduction::CondensationPlan& plan)
{
    if (zone.unit == 0)
        return false;
    return std::any_of(plan.units().begin(), plan.units().end(), [&](const reduction::UnitPlan& unit)
                       { return unit.unitId == zone.unit && unit.decision == reduction::UnitDecision::Condensed; });
}

// A mechanism met on the condensed path, without its shapes: MechanismShapes works on the f set of the full stiffness,
// which the condensed path never assembles (an interior factor of a unit or the interface factor fails instead). The
// error keeps its message and DOFs and says how to get the shapes.
base::SingularityError withShapesRemedy(const base::SingularityError& error)
{
    return base::SingularityError(
        base::concatenate({error.what(),
                           "\nthe condensed path does not compute the zero-energy shapes: run the analysis on the full "
                           "path (without condensation) to have them in the error's mechanism shapes"}),
        error.dofs());
}

// The rigid-body motions of a body free apart from six FreeBodySupport DOFs: Φ, the six motions about the centroid of
// the grid points (from the geometry; zero on the s set, u_m = Gm·u_n), and C = Φ_S⁻¹
// (assembly::rigidSupportCoefficients), so that D = Φ·C is the motion of a unit motion of each support with the others
// at rest. Taken when the body is free: the largest |Knn·D| of every DOF within assembly::RIGID_SUPPORT_TOLERANCE of
// the largest stiffness of the model (the round-off of the element sums, which scales with the stiffest element; not
// amplified by a solve, where D_l = −K_ll⁻¹·K_lr carries that round-off amplified by the condition number of the body
// held at its supports: 1.6e-4 of D on a free barrel of four frames at 1/32, 1.3 M DOFs). Nothing otherwise (fewer than
// six supports, or a constraint or a grounded element that resists a rigid motion): D is solved from the stiffness, D_l
// = −K_ll⁻¹·K_lr. The relief is made on Φ, not on D: the relieved loads P − M·Φ·(Φᵀ·M·Φ)⁻¹·Φᵀ·P are those of D in exact
// arithmetic and do not depend on the FreeBodySupport DOFs in floating point either, so that the FreeBodySupport DOFs
// are only the gauge of the displacements (u ← u − D·u_r) and the frame of the accelerations reported (ü_D = Φ_S·ü_Φ).
struct RigidSupports
{
    std::optional<linalg::DenseMatrix> modes; // Φ (g × 6), when the body is free
    std::array<double, assembly::MAX_SUPPORT_DOFS * assembly::RigidBodyModes::MODE_COUNT> coefficients{}; // C
    double reaction = -1.0; // max |Knn·D| over the largest stiffness (-1: not six independent supports)
};

RigidSupports rigidSupportModes(const assembly::DofMap& dofMap, const linalg::BlockSparseMatrix& knn,
                                const assembly::RigidElimination& gm, std::span<const std::uint8_t> isConstrained,
                                std::span<const base::DofIndex> supports, base::MemoryBudget& budget)
{
    constexpr std::size_t MODES = assembly::RigidBodyModes::MODE_COUNT;
    if (supports.size() != assembly::MAX_SUPPORT_DOFS)
        return {};
    // About the centroid of the grid points: Φᵀ·M·Φ is then as well conditioned as the body's inertia allows (about the
    // basic origin, a body far from it has its rotations dominated by translations).
    base::Vec3 centroid{};
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
        centroid = centroid + dofMap.positionInBasic(row);
    if (dofMap.gridCount() > 0)
        centroid = centroid * (1.0 / static_cast<double>(dofMap.gridCount()));
    const assembly::RigidBodyModes rigid(dofMap, centroid, budget);
    const linalg::DenseMatrix& phi = rigid.modes();
    std::array<double, assembly::MAX_SUPPORT_DOFS * MODES> atSupports{};
    for (std::size_t j = 0; j < supports.size(); ++j)
        for (std::size_t a = 0; a < MODES; ++a)
            atSupports[j * MODES + a] = phi(static_cast<std::size_t>(supports[j]), a);
    const auto coefficients = assembly::rigidSupportCoefficients(atSupports, supports.size());
    if (!coefficients)
        return {};
    const std::size_t g = phi.rows();
    linalg::DenseMatrix modes(budget, g, MODES, "inertia relief rigid-body modes");
    linalg::DenseMatrix forces(budget, g, MODES, "inertia relief K Phi");
    for (std::size_t a = 0; a < MODES; ++a)
    {
        const std::span<double> column(&modes(0, a), g);
        for (std::size_t dof = 0; dof < g; ++dof)
            column[dof] = isConstrained[dof] == 0 ? phi(dof, a) : 0.0;
        // The FreeBodySupport DOFs are held in the solve sets (isConstrained), not at rest in the rigid motions.
        for (const base::DofIndex support : supports)
            column[static_cast<std::size_t>(support)] = phi(static_cast<std::size_t>(support), a);
        // Knn·Φ_n on the n set: Knn keeps the rows and columns of the m DOFs as Kgg has them (MsetElimination), so the
        // product takes Φ on the n set alone and is read on the n set alone (with Φ_m in the column, K̄nm·Φ_m would be
        // added a second time: a model with an RigidBodyStar or an MPC equation on stiff points would fail the test).
        for (const base::DofIndex dependent : gm.mDofs())
            column[static_cast<std::size_t>(dependent)] = 0.0;
        knn.multiply(column, std::span<double>(&forces(0, a), g));
        for (const base::DofIndex dependent : gm.mDofs())
            forces(static_cast<std::size_t>(dependent), a) = 0.0;
        gm.expandToG(column);
    }
    // K·D = (K·Φ)·C.
    double reaction = 0.0;
    for (std::size_t j = 0; j < supports.size(); ++j)
        for (std::size_t dof = 0; dof < g; ++dof)
        {
            double force = 0.0;
            for (std::size_t a = 0; a < MODES; ++a)
                force += forces(dof, a) * (*coefficients)[j * MODES + a];
            reaction = std::max(reaction, std::abs(force));
        }
    // The round-off of K·Φ scales with the largest stiffness of the model, not with the supports' own (fasteners 1e8
    // times stiffer than the skins make it 1e8 times a skin support's): the largest |Knn| term, the
    // largest diagonal term of a positive semi-definite matrix (|K_ij| ≤ √(K_ii·K_jj)).
    double stiffness = 0.0;
    for (const double value : knn.values())
        stiffness = std::max(stiffness, std::abs(value));
    RigidSupports result;
    result.reaction = stiffness > 0.0 ? reaction / stiffness : 0.0;
    result.coefficients = *coefficients;
    if (reaction <= assembly::RIGID_SUPPORT_TOLERANCE * stiffness)
        result.modes = std::move(modes);
    return result;
}

// Back to the gauge of the FreeBodySupport DOFs of a free body solved on its conditioned supports:
// u ← u − D·u_r with D = Φ·C (rigidSupportModes; `relief` made on Φ), every DOF but the s set's (zero in Φ).
void toSupportGauge(const assembly::InertiaRelief& relief,
                    const std::array<double, assembly::MAX_SUPPORT_DOFS * assembly::RigidBodyModes::MODE_COUNT>& c,
                    std::span<const base::DofIndex> supports, std::span<double> ug)
{
    constexpr std::size_t MODES = assembly::RigidBodyModes::MODE_COUNT;
    std::array<double, MODES> amplitudes{}; // C·u_r
    for (std::size_t j = 0; j < supports.size(); ++j)
        for (std::size_t a = 0; a < MODES; ++a)
            amplitudes[a] += c[j * MODES + a] * ug[static_cast<std::size_t>(supports[j])];
    const linalg::DenseMatrix& phi = relief.supportModes();
    for (std::size_t dof = 0; dof < ug.size(); ++dof)
    {
        double motion = 0.0;
        for (std::size_t a = 0; a < MODES; ++a)
            motion += phi(dof, a) * amplitudes[a];
        ug[dof] -= motion;
    }
    // D_r = Φ_r·C = I: the FreeBodySupport DOFs at rest exactly (their round-off left out).
    for (const base::DofIndex support : supports)
        ug[static_cast<std::size_t>(support)] = 0.0;
}

// The accelerations of the FreeBodySupport DOFs of a relief made on Φ (rigidSupportModes): ü_D = Φ_S·ü_Φ.
std::array<double, assembly::MAX_SUPPORT_DOFS> supportAccelerationsOf(
    const assembly::InertiaRelief& relief, std::span<const base::DofIndex> supports,
    const std::array<double, assembly::MAX_SUPPORT_DOFS>& accelerationsOfModes)
{
    std::array<double, assembly::MAX_SUPPORT_DOFS> result{};
    const linalg::DenseMatrix& phi = relief.supportModes();
    for (std::size_t j = 0; j < supports.size(); ++j)
        for (std::size_t a = 0; a < phi.columns(); ++a)
            result[j] += phi(static_cast<std::size_t>(supports[j]), a) * accelerationsOfModes[a];
    return result;
}

// The relieved problem of a free body solved on its conditioned supports, for refineRelievedSolution.
struct RelievedSystem
{
    const FsetSolver* solver = nullptr;              // K_ll on the conditioned supports
    const linalg::BlockSparseMatrix* kgg = nullptr;  // Kgg, for the residuals
    const assembly::RigidElimination* gm = nullptr;  // u_m = Gm·u_n, and the products reduced to the n set
    const assembly::InertiaRelief* relief = nullptr; // Φ and M·Φ: the gauge of the residuals
    linalg::ConstDenseMatrixView relievedLoads;      // equations × k: the relieved loads P'_l
};

// The iterative refinement of the relieved solution of a free body (linalg::IterativeRefinement). Each step: the
// displacements u (zero at the conditioned supports) taken to the gauge M-orthogonal to the rigid motions,
// û = u − Φ·(Φᵀ·M·Φ)⁻¹·(M·Φ)ᵀ·u, so that the round-off of K on the rigid motions (|K·Φ| about 1e-14 of the largest
// stiffness) does not act on a rigid part of u; the residual r = P'_l − (K·û)_l with K·û = Gmᵀ·Kgg·(Gm·û) (the relative
// motion across a stiff element from the displacements of its own points: with fasteners 1e8 times the skins the
// refinement converged with it to 1e-9, stalled at 4e-5 with Knn·u_n); δu_l = K_ll⁻¹·r on the
// same factor; u_l ← u_l + δu_l (û and u are the same deformation: the gauge of the FreeBodySupport DOFs is made
// after). The residual's resultant, the imbalance the round-off leaves, is taken by the conditioned supports: relieving
// it by the inertia instead changed the refined solution by less than the residual's noise once û is M-orthogonal (by
// 5e-8 with the residuals of u itself). `solution` (equations × k) holds u_l on entry and the refined u_l on exit. Each
// step costs one product with Kgg and one solve of the k vectors; the same bits whatever the threads.
std::vector<double> refineRelievedSolution(const RelievedSystem& system, linalg::DenseMatrixView solution,
                                           base::MemoryBudget& budget, base::ThreadPool* pool,
                                           const base::CancellationToken* cancellation)
{
    const FsetSolver& solver = *system.solver;
    const auto gSize = static_cast<std::size_t>(system.kgg->pattern().dofCount());
    const std::size_t equations = solution.rows;
    const std::size_t k = solution.columns;
    linalg::IterativeRefinement refinement;
    if (k == 0 || equations == 0)
        return refinement.corrections();
    const std::size_t group = basisProductGroup(gSize, k, true, budget);
    linalg::DenseMatrix displacements(budget, gSize, group, "relief refinement displacements");
    linalg::DenseMatrix products(budget, gSize, group, "relief refinement products");
    linalg::DenseMatrix steps(budget, equations, k, "relief refinement corrections");
    const linalg::DenseMatrix& phi = system.relief->supportModes();
    const auto dofOf = [&solver](std::size_t e)
    { return static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e))); };
    while (true)
    {
        for (std::size_t first = 0; first < k; first += group)
        {
            checkCancelled(cancellation);
            const std::size_t count = std::min(group, k - first);
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::span<double> u(&displacements(0, j), gSize);
                std::fill(u.begin(), u.end(), 0.0);
                for (std::size_t e = 0; e < equations; ++e)
                    u[dofOf(e)] = solution(e, first + j);
                system.gm->expandToG(u);
                const std::array<double, assembly::MAX_SUPPORT_DOFS> rigid = system.relief->rigidMotionOf(u);
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    for (std::size_t a = 0; a < phi.columns(); ++a)
                        u[dof] -= phi(dof, a) * rigid[a];
            }
            system.kgg->multiply(displacements.view().block(0, 0, gSize, count),
                                 products.view().block(0, 0, gSize, count), pool);
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::span<double> product(&products(0, j), gSize);
                system.gm->reduceToN(product);
                for (std::size_t e = 0; e < equations; ++e)
                    steps(e, first + j) = system.relievedLoads(e, first + j) - product[dofOf(e)];
            }
        }
        solver.solve(steps.view(), cancellation);
        const linalg::IterativeRefinement::Step step =
            refinement.record(linalg::IterativeRefinement::largestRelative(steps.view(), solution));
        if (step.isAccepted)
            for (std::size_t b = 0; b < k; ++b)
                for (std::size_t e = 0; e < equations; ++e)
                    solution(e, b) += steps(e, b);
        if (!step.continues)
            break;
    }
    return refinement.corrections();
}

// The inertia relief of a constraint set kept for the sensitivities (KeptRelief): D, M·D, (Dᵀ·M·D)⁻¹, the
// accelerations of its cases (their basis vectors' combined with their coefficients, as the checks give them) and how
// far D is from rigid-body motions (the largest support reaction of D over the largest stiffness of a support DOF).
std::shared_ptr<const KeptRelief> keptReliefOf(
    const assembly::InertiaRelief& relief, const linalg::BlockSparseMatrix& kgg, std::span<const std::uint8_t> isHeld,
    const std::vector<base::DofIndex>& supports, const linalg::DenseMatrix& coefficients,
    const std::vector<std::array<double, assembly::MAX_SUPPORT_DOFS>>& basisAccelerations, base::MemoryBudget& budget)
{
    auto kept = std::make_shared<KeptRelief>();
    const linalg::DenseMatrix& modes = relief.supportModes();
    const std::size_t g = modes.rows();
    const std::size_t r = modes.columns();
    kept->supportModes = linalg::DenseMatrix(budget, g, r, "kept inertia relief D");
    kept->massModes = linalg::DenseMatrix(budget, g, r, "kept inertia relief M D");
    for (std::size_t j = 0; j < r; ++j)
        for (std::size_t dof = 0; dof < g; ++dof)
        {
            kept->supportModes(dof, j) = modes(dof, j);
            kept->massModes(dof, j) = relief.massModes()(dof, j);
        }
    // (Dᵀ·M·D)⁻¹ by Gauss-Jordan on the r × r reduced mass (symmetric positive definite: the relief refused it
    // otherwise).
    std::array<double, assembly::MAX_SUPPORT_DOFS * 2 * assembly::MAX_SUPPORT_DOFS> augmented{};
    const std::size_t width = 2 * r;
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t k = 0; k < r; ++k)
            augmented[i * width + k] = relief.reducedMass()(i, k);
        augmented[i * width + r + i] = 1.0;
    }
    for (std::size_t p = 0; p < r; ++p)
    {
        const double pivot = augmented[p * width + p];
        for (std::size_t k = 0; k < width; ++k)
            augmented[p * width + k] /= pivot;
        for (std::size_t i = 0; i < r; ++i)
        {
            if (i == p)
                continue;
            const double factor = augmented[i * width + p];
            for (std::size_t k = 0; k < width; ++k)
                augmented[i * width + k] -= factor * augmented[p * width + k];
        }
    }
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t k = 0; k < r; ++k)
            kept->reducedMassInverse[i * r + k] = augmented[i * width + r + k];
    // The accelerations of the cases: Σ_b α(b, c)·ü_b.
    for (std::size_t c = 0; c < coefficients.columns(); ++c)
    {
        std::array<double, assembly::MAX_SUPPORT_DOFS> acceleration{};
        for (std::size_t b = 0; b < coefficients.rows(); ++b)
            for (std::size_t k = 0; k < r; ++k)
                acceleration[k] += coefficients(b, c) * basisAccelerations[b][k];
        kept->accelerations.push_back(acceleration);
    }
    // How far D is from rigid motions: its reactions at the held DOFs (the s and r sets) against the stiffness of the
    // support DOFs.
    double reaction = 0.0;
    double stiffness = 0.0;
    base::BudgetedBuffer<double> column(budget, g, "kept inertia relief D column");
    base::BudgetedBuffer<double> product(budget, g, "kept inertia relief K D");
    for (std::size_t j = 0; j < r; ++j)
    {
        for (std::size_t dof = 0; dof < g; ++dof)
            column[dof] = modes(dof, j);
        kgg.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < g; ++dof)
            if (isHeld[dof] != 0)
                reaction = std::max(reaction, std::abs(product[dof]));
        std::fill(column.data(), column.data() + g, 0.0);
        column[static_cast<std::size_t>(supports[j])] = 1.0;
        kgg.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < g; ++dof)
            stiffness = std::max(stiffness, std::abs(product[dof]));
    }
    kept->rigidResidual = stiffness > 0.0 ? reaction / stiffness : 0.0;
    return kept;
}

// The load basis handed over (StaticOptions::keepsLoadBasis): the basis of each constraint set, chosen
// before any factorisation or condensation (the full path's StaticLoadCases, the same on both paths), and the mass of
// its inertial loads on the pattern of the model's elements (one per mass option of the sets: COUPMASS, WTMASS).
struct LoadBasisPlan
{
    std::unique_ptr<linalg::NodeGraph> massGraph;
    std::unique_ptr<linalg::SparsePattern> massPattern;
    struct Mass
    {
        assembly::MassAssemblyOptions options;
        std::optional<base::EntityId> selection;
        std::unique_ptr<linalg::BlockSparseMatrix> matrix;
    };
    std::vector<Mass> masses;
    std::vector<std::unique_ptr<StaticLoadCases>> cases; // per constraint set
    std::uint64_t estimatedBytes = 0;                    // what the units keep and hold while made
};

// The vectors per unit a load basis holds while it is made, beside those it keeps (the g-set displacements of its units
// on the full path; on the condensed path the interface's displacements, the boundary basis solutions, on the plan's
// estimate of the interface's DOFs): the right-hand sides of the full path; the interface's loads and right-hand sides
// of the condensed path. The units' loads on the equations of the condensed units they load are reserved when made.
constexpr std::uint64_t FULL_BASIS_TRANSIENT_VECTORS = 1;
constexpr std::uint64_t CONDENSED_BASIS_TRANSIENT_VECTORS = 2;

std::string megabytesOf(std::uint64_t bytes)
{
    constexpr double MEGABYTE = 1024.0 * 1024.0;
    std::ostringstream text;
    text.setf(std::ios::fixed);
    text.precision(1);
    text << static_cast<double>(bytes) / MEGABYTE << " MB";
    return text.str();
}

// Chooses the basis of each constraint set (`groups`, indices into `subcases`) and checks what its units keep against
// the share of the budget the result tables may take (StaticOptions::outputLimitBytes, else OUTPUT_BUDGET_FRACTION of
// what is available): the basis replaces those tables. Refuses on the condensed path, naming the subcases, PARAM
// INREL = −2 and enforced displacements (SolverError). Throws MemoryBudgetError with the estimate.
LoadBasisPlan planLoadBasis(const input::ModelSnapshot& snapshot, const StaticOptions& options,
                            const AnalysisModel& model, const assembly::LoadVectorBuilder& loadBuilder,
                            std::span<const StaticSubcase> subcases, std::span<const assembly::SpcSet> spcSets,
                            const std::vector<std::vector<std::size_t>>& groups,
                            std::span<const assembly::InertiaReliefMode> reliefModes, base::MemoryBudget& budget,
                            base::DiagnosticSink& sink)
{
    const assembly::DofMap& dofMap = model.dofMap();
    const bool isCondensedPath = options.condensation && !options.condensation->isFullPath();
    if (isCondensedPath)
    {
        for (const std::vector<std::size_t>& group : groups)
        {
            // Inertia relief is taken but PARAM INREL = −2, which the condensed path refuses.
            if (reliefModes[group.front()] == assembly::InertiaReliefMode::Automatic)
                sink.report(subcaseProblem(
                    subcases[group.front()].label, base::Severity::Error, "load-basis-condensed-relief",
                    "the load basis of the condensed path takes PARAM InertiaRelief = -1 with FreeBodySupport, "
                    "not PARAM InertiaRelief = -2; give the "
                    "supports with FreeBodySupport, or ask for the load basis without condensation"));
            for (const std::size_t s : group)
                if (!spcSets[s].enforcedDisplacements().empty())
                    sink.report(subcaseProblem(
                        subcases[s].label, base::Severity::Error, "load-basis-condensed-enforced",
                        "the load basis of the condensed path does not take enforced displacements in this version "
                        "(SinglePointConstr D, EnforcedMotionConstr); ask for the load basis without condensation"));
        }
        stopOnErrors(sink, "the load basis");
    }
    LoadBasisPlan plan;
    const input::ParamTable* params = model.params();
    for (const std::vector<std::size_t>& group : groups)
    {
        const linalg::BlockSparseMatrix* mass = nullptr;
        bool needsMass = false;
        for (const std::size_t s : group)
            needsMass = needsMass || (subcases[s].load && loadBuilder.needsMass(*subcases[s].load));
        if (needsMass)
        {
            // The options of the set's first subcase, as the full path's mass; their problems are reported there.
            base::DiagnosticSink reported;
            const assembly::MassAssemblyOptions massOptions =
                assembly::MassAssemblyOptions::fromParams(params, subcases[group.front()].label, reported);
            const auto found = std::find_if(plan.masses.begin(), plan.masses.end(),
                                            [&](const auto& entry)
                                            {
                                                return entry.options.formulation == massOptions.formulation &&
                                                       entry.options.weightToMass == massOptions.weightToMass &&
                                                       entry.selection == subcases[group.front()].nonStructuralMass;
                                            });
            if (found != plan.masses.end())
                mass = found->matrix.get();
            else
            {
                if (!plan.massPattern)
                {
                    const std::vector<linalg::NodeGroupList> nodeGroups{model.connectivity().groups()};
                    plan.massGraph =
                        std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), nodeGroups, budget, options.pool);
                    plan.massPattern =
                        std::make_unique<linalg::SparsePattern>(*plan.massGraph, model.nodeDofCounts(), budget);
                }
                auto matrix = std::make_unique<linalg::BlockSparseMatrix>(*plan.massPattern, budget, "load basis Mgg");
                const assembly::NonStructuralMassShares shares(model.elementSet(),
                                                               subcases[group.front()].nonStructuralMass, budget, sink);
                stopOnErrors(sink, "the selected non-structural mass");
                assembly::MassAssembler(model.elementSet(), dofMap, shares, massOptions)
                    .assemble(*matrix, options.pool, options.cancellation);
                mass = matrix.get();
                plan.masses.push_back({massOptions, subcases[group.front()].nonStructuralMass, std::move(matrix)});
            }
        }
        StaticLoadCases::Inputs caseInputs;
        caseInputs.snapshot = &snapshot;
        caseInputs.points = &model.index().points();
        caseInputs.dofMap = &dofMap;
        caseInputs.loads = &loadBuilder;
        for (const std::size_t s : group)
        {
            caseInputs.cases.push_back(&subcases[s]);
            caseInputs.spcs.push_back(&spcSets[s]);
        }
        caseInputs.mass = mass;
        caseInputs.tolerance = options.loadBasisTolerance;
        caseInputs.pool = options.pool;
        caseInputs.kernel = options.kernel;
        plan.cases.push_back(std::make_unique<StaticLoadCases>(std::move(caseInputs), budget, sink));
    }
    stopOnErrors(sink, "the load basis");

    // What the units keep, and the largest set's vectors while they are made.
    const auto gSize = static_cast<std::uint64_t>(dofMap.gSetSize());
    const std::uint64_t keptRows =
        isCondensedPath ? static_cast<std::uint64_t>(options.condensation->interfaceEstimate().dofCount) : gSize;
    std::uint64_t unitCount = 0;
    std::uint64_t largestSet = 0;
    for (const std::unique_ptr<StaticLoadCases>& cases : plan.cases)
    {
        unitCount += cases->basisSize();
        largestSet = std::max<std::uint64_t>(largestSet, cases->basisSize());
    }
    const std::uint64_t keptBytes =
        unitCount * keptRows * sizeof(double) + BasisFixedRows::estimateBytes(options.basisFixedRows, gSize, unitCount);
    const std::uint64_t transientBytes =
        (isCondensedPath ? CONDENSED_BASIS_TRANSIENT_VECTORS : FULL_BASIS_TRANSIENT_VECTORS) * largestSet * keptRows *
        sizeof(double);
    const std::uint64_t estimate = keptBytes + transientBytes;
    const std::uint64_t limit = options.outputLimitBytes.value_or(
        static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available())));
    plan.estimatedBytes = estimate;
    if (estimate > limit)
    {
        const std::string message = base::concatenate(
            {"the load basis keeps the displacements of its ", std::to_string(unitCount), " unit",
             unitCount == 1 ? "" : "s", " on ", std::to_string(keptRows), isCondensedPath ? " interface DOFs" : " DOFs",
             ": about ", megabytesOf(estimate), " (", megabytesOf(keptBytes), " kept, ", megabytesOf(transientBytes),
             " while they are made), more than the ", megabytesOf(limit),
             " the results may take in memory; nothing was solved. What to do:\n  - give the analysis more memory"
             "\n  - ask for fewer subcases, or one constraint set at a time",
             isCondensedPath ? ""
                             : "\n  - condense the model: the basis then keeps the displacements "
                               "of the interface only"});
        throw base::MemoryBudgetError(
            message, base::MemoryShortfall{
                         estimate,
                         budget.available(),
                         budget.limit(),
                         {{"load basis displacements", keptBytes}, {"load basis vectors while made", transientBytes}}});
    }
    return plan;
}

} // namespace

std::vector<base::EntityId> StaticSolution::subcasesToSolve() const
{
    base::DiagnosticSink ignored;
    std::vector<base::EntityId> labels;
    for (const StaticSubcase& subcase : staticSubcasesOf(*m_snapshot, m_options.subcases, ignored))
        labels.push_back(subcase.label);
    return labels;
}

StaticResults StaticSolution::run(base::MemoryBudget& budget) const
{
    const bool isCondensedPath = m_options.condensation && !m_options.condensation->isFullPath();
    try
    {
        StaticResults results = runOnPath(budget);
        if (m_options.structures != nullptr)
        {
            results.isStructureCacheUsed = true;
            results.structureCacheBytes = m_options.structures->bytes();
        }
        return results;
    }
    catch (const base::SingularityError& error)
    {
        if (!isCondensedPath || !error.mechanisms().empty())
            throw;
        throw withShapesRemedy(error);
    }
}

StaticResults StaticSolution::runOnPath(base::MemoryBudget& budget) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    const base::CancellationToken* cancellation = m_options.cancellation;
    base::ThreadPool* pool = m_options.pool;
    // The multiPointConstrForce rows of the grid point forces split by relation for an O_o_Reader of data contract 56
    // or later (the snapshot's tables say which); one row a point below.
    const bool isMpcSplit = snapshot.versions().dataContractVersion >= input::FIRST_CONTRACT_WITH_SPLIT_MPC_ROWS;
    BasisFixedRowRequest basisFixedRows = m_options.basisFixedRows;
    basisFixedRows.isMpcSplit = isMpcSplit;
    StaticResults results;
    Stopwatch watch;
    base::DiagnosticSink sink;
    // A run that stops after the memory plan or the condensation of the units (StaticOptions::condensedStage) makes no
    // table: the output limit does not apply to it.
    const bool isStagedRun = m_options.condensedStage != CondensedStage::Complete;
    // The Interface stage goes on to the constraint sets (their interfaces handed over), and the caller's values
    // replace the interface's solves (StaticOptions::givenBoundaries).
    const bool isInterfaceStage = m_options.condensedStage == CondensedStage::Interface;
    const bool hasGivenBoundaries = !m_options.givenBoundaries.empty();
    if (hasGivenBoundaries && (isStagedRun || m_options.keepsLoadBasis || m_options.keepsSystems))
        throw std::invalid_argument(
            "StaticSolution: the interface's solve replaced by given values is a complete static run that keeps no "
            "system nor load basis (a bug of the caller that asked it: report it)");
    // GivenLocal (one unit's boundary): its unit, whose outputs alone the run makes.
    std::optional<std::int64_t> givenUnit;
    for (const StaticOptions::GivenBoundary& given : m_options.givenBoundaries)
        if (given.mode == reduction::BoundarySolveMode::GivenLocal)
            givenUnit = given.unitId;
    if (isStagedRun && (!m_options.condensation || m_options.condensation->isFullPath()))
        throw std::invalid_argument(
            "StaticSolution: a run that stops after the memory plan or the condensation needs a "
            "condensation plan that condenses a unit; give one, or run every stage");
    if (m_options.keepsLoadBasis &&
        (!m_options.sectionLoads.isEmpty() || !m_options.zones.isEmpty() || !m_options.envelopes.empty() ||
         !m_options.monitorPoints3.empty() || !m_options.externalUnits.empty() || isStagedRun))
        throw std::invalid_argument(
            "StaticSolution: the load basis is kept without sections, zones, envelopes, MonitorNodeForceSum3, external "
            "units or a "
            "staged run; ask for those in another run (a bug of the caller that gave them: report it)");
    // The full path streamed keeps a load basis only with declared quantities or the fixed rows of declared cuts
    // its regions are gone after the solve.
    const bool isStreamedBasis = m_options.isStreamed && m_options.keepsLoadBasis &&
                                 (m_options.basisQuantities != nullptr || m_options.basisFixedRows.isRequested);
    if (m_options.isStreamed &&
        ((m_options.condensation && !m_options.condensation->isFullPath()) || m_options.keepsSystems ||
         (m_options.keepsLoadBasis && !isStreamedBasis) || !m_options.sectionLoads.isEmpty() ||
         !m_options.zones.isEmpty() || !m_options.monitorPoints3.empty() || !m_options.externalUnits.empty() ||
         !m_options.keptDisplacements.empty() || isStagedRun))
        throw std::invalid_argument(
            "StaticSolution: the full path streamed takes no condensation plan, kept systems, load basis handed over, "
            "sections, zones, MonitorNodeForceSum3, external units or kept displacements (its regions keep nothing "
            "after the "
            "solve); ask for those on the full path or the condensed path");
    if (m_options.keepsThermalUnits && (!m_options.keepsLoadBasis || m_options.isStreamed ||
                                        (m_options.condensation && !m_options.condensation->isFullPath())))
        throw std::invalid_argument(
            "StaticSolution: the thermal units of a load basis are kept on the full path only, not streamed nor "
            "condensed (a bug of the caller that asked it: report it)");
    std::vector<base::EntityId> excludedCases; // with keepsLoadBasis: the subcases left out (a thermal load)

    // 1. What the model asks for is supported, every element can be computed, the subcases and their output plans are
    // read (every problem named in one pass); 2. the volume of the requested outputs, before any element matrix is
    // computed; then the DOF map and the connectivity (AnalysisModel).
    std::vector<StaticSubcase> subcases;
    output::EnvelopeAccumulator envelopes(m_options.envelopes);
    std::vector<output::OutputPlan> plans; // bounded by the subcases
    // The plans without StaticOptions::knownOutputs, which decide the units the condensed path recovers;
    // empty when no known output is asked.
    std::vector<output::OutputPlan> recoveryPlans;
    const bool hasKnownOutputs =
        std::any_of(m_options.knownOutputs.begin(), m_options.knownOutputs.end(),
                    [](const output::OutputRequestPlan& request) { return request.isRequested; });
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* fastenerPiercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    base::ProgressCounter* progress = nullptr;
    // The section and cut loads of the options (output/sections/SectionLoadRun), made with the element set.
    std::unique_ptr<output::SectionLoadRun> sectionRun;
    // The contour data of the zones of the options (output/sections/ZoneRun), made with the element set.
    std::unique_ptr<output::ZoneRun> zoneRun;
    // The MonitorNodeForceSum3 of the options, made with the element set.
    std::vector<output::MonitorPoint3> monitors3; // bounded by the MonitorNodeForceSum3 of the model
    std::uint64_t resultTableBytes = 0;           // the result tables held in memory (the condensed path's memory plan)
    // The external units (ExternalUnitPath), made with the element set before the connectivity pass.
    std::unique_ptr<ExternalUnitPath> externalPath;
    AnalysisModelOptions modelOptions;
    modelOptions.checkCases = [&](base::DiagnosticSink& caseSink)
    {
        subcases = staticSubcasesOf(snapshot, m_options.subcases, caseSink);
        for (const StaticSubcase& subcase : subcases)
        {
            plans.push_back(output::OutputPlan::forSubcase(snapshot, subcase.tableId, caseSink));
            if (m_options.usesCaseControlOutputs)
                plans.back().addRequired(m_options.requiredOutputs);
            else
                plans.back().restrictTo(m_options.requiredOutputs);
            if (hasKnownOutputs)
            {
                recoveryPlans.push_back(plans.back());
                plans.back().addRequired(m_options.knownOutputs);
            }
            if (plans.back().modalEffectiveMass().isRequested)
                caseSink.report(
                    subcaseProblem(subcase.label, base::Severity::Warning, "output-not-produced",
                                   "ModalEffectiveMassOutput is an output of a real eigenvalue analysis (SOL 103): a "
                                   "static subcase does not produce it"));
        }
        if (m_options.keepsLoadBasis)
        {
            // The load basis handed over superposes linear results: a case with a thermal or deformation load (or a
            // SUBCOM that combines one) is not the superposition of its units' element results (a DEFORM is an
            // initial strain as a temperature is).
            for (const StaticSubcase& subcase : subcases)
            {
                bool isThermal = subcase.hasInitialStrains();
                for (const base::EntityId term : subcase.combined)
                    isThermal =
                        isThermal || std::find(excludedCases.begin(), excludedCases.end(), term) != excludedCases.end();
                // With thermal units a solved subcase keeps its thermal and deformation loads (its
                // thermal part is the unit of its key of temperature and deformation sets); a SUBCOM that has a
                // thermal or deformation load or combines a thermal subcase is not the combination of its subcases'
                // element results (its initial strains are those of its own TEMPERATURE(LOAD) and DEFORM), so it is
                // still refused.
                if (m_options.keepsThermalUnits)
                {
                    if (!subcase.isCombination)
                        continue;
                    bool combinesThermal = subcase.hasInitialStrains();
                    for (std::size_t t = 0; t < subcase.combined.size(); ++t)
                        for (const StaticSubcase& other : subcases)
                            combinesThermal =
                                combinesThermal || (other.label == subcase.combined[t] && other.hasInitialStrains() &&
                                                    subcase.coefficients[t] != 0.0);
                    if (combinesThermal)
                        caseSink.report(subcaseProblem(
                            subcase.label, base::Severity::Error, "load-basis-thermal",
                            "expected a SubcaseCombination without a thermal or deformation load and "
                            "combining no thermal subcase "
                            "in the load basis with thermal units: its element results take the initial strains of "
                            "its own TemperatureSelection(LOAD) and DeformationLoadSelection, not "
                            "those of its subcases; "
                            "remove the SubcaseCombination from the basis's subcases or leave the thermal "
                            "subcases out"));
                    continue;
                }
                if (!isThermal)
                    continue;
                if (m_options.excludesThermalCases)
                    excludedCases.push_back(subcase.label);
                else
                    caseSink.report(subcaseProblem(subcase.label, base::Severity::Error, "load-basis-thermal",
                                                   "expected a subcase without a thermal load "
                                                   "(TemperatureSelection(LOAD)) or element deformations "
                                                   "(DeformationLoadSelection) in the load basis handed "
                                                   "over, whose units superpose linear results; leave it out, "
                                                   "keep its initial strains as a unit, or solve it in a static run"));
            }
            for (std::size_t s = subcases.size(); s-- > 0;)
            {
                if (std::find(excludedCases.begin(), excludedCases.end(), subcases[s].label) == excludedCases.end())
                    continue;
                subcases.erase(subcases.begin() + static_cast<std::ptrdiff_t>(s));
                plans.erase(plans.begin() + static_cast<std::ptrdiff_t>(s));
                if (!recoveryPlans.empty())
                    recoveryPlans.erase(recoveryPlans.begin() + static_cast<std::ptrdiff_t>(s));
            }
        }
    };
    modelOptions.checkBeforeConnectivity = [&](const assembly::ElementSet& elementSet, const assembly::DofMap& dofMap)
    {
        std::vector<base::EntityId> checkedMassSets;
        for (const StaticSubcase& subcase : subcases)
        {
            if (!subcase.nonStructuralMass || std::find(checkedMassSets.begin(), checkedMassSets.end(),
                                                        *subcase.nonStructuralMass) != checkedMassSets.end())
                continue;
            const assembly::NonStructuralMassShares shares(elementSet, subcase.nonStructuralMass, budget, sink);
            checkedMassSets.push_back(*subcase.nonStructuralMass);
        }
        stopOnErrors(sink, "the selected non-structural mass");
        output::OutputVolumeEstimate estimate(elementSet, dofMap);
        const bool hasMset = (rigidElements != nullptr && rigidElements->reader().rowCount() > 0) ||
                             (constraints != nullptr && constraints->mpcReader().rowCount() > 0) ||
                             fastenerPiercings != nullptr;
        for (const output::OutputPlan& plan : plans)
            estimate.addCase(plan, hasMset, [&](const std::string& kind)
                             { return m_options.keepsCaseTables || !envelopes.takes(kind); });
        const std::uint64_t limit = m_options.outputLimitBytes.value_or(
            static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available())));
        // Tables that do not fit in memory go to the result sink, when there is one, within what it can take.
        results.areTablesStreamed = !isStagedRun && m_options.resultSink && estimate.totalBytes() > limit;
        if (results.areTablesStreamed && m_options.sinkLimitBytes)
            estimate.requireWithinDestination(*m_options.sinkLimitBytes);
        else if (!results.areTablesStreamed && !isStagedRun && !m_options.keepsLoadBasis)
            estimate.requireWithin(limit);
        // The section and cut loads: their definitions checked with the model, before any
        // factorisation, and their rows made.
        if (!m_options.sectionLoads.isEmpty())
        {
            sectionRun = std::make_unique<output::SectionLoadRun>(m_options.sectionLoads, snapshot, elementSet, dofMap,
                                                                  subcases.size(), budget, sink, m_options.pool);
            stopOnErrors(sink, "the section and cut loads");
        }
        // The zones of local models: their contours resolved with the model, before any
        // factorisation, the constraints of the solved subcases deciding the partial ones.
        if (!m_options.zones.isEmpty())
        {
            std::vector<output::ZoneSubcaseConstraint> zoneSubcases;
            for (const StaticSubcase& subcase : subcases)
                if (!subcase.isCombination)
                    zoneSubcases.push_back({subcase.label, subcase.spc, subcase.load});
            zoneRun = std::make_unique<output::ZoneRun>(m_options.zones, zoneSubcases, snapshot, elementSet, dofMap,
                                                        subcases.size(), budget, sink);
            stopOnErrors(sink, "the zones of local models");
        }
        // The MonitorNodeForceSum3: their sets checked with the model before any factorisation (SolverError naming the
        // point).
        for (const output::MonitorPoint3Definition& definition : m_options.monitorPoints3)
            monitors3.emplace_back(definition, snapshot, elementSet, dofMap);
        // What the tables hold in memory until the end: all of them, or the subcases a streaming sink has in flight.
        resultTableBytes = results.areTablesStreamed
                               ? estimate.totalBytes() /
                                     static_cast<std::uint64_t>(std::max<std::int64_t>(estimate.caseCount(), 1)) *
                                     STREAMED_SUBCASES_IN_FLIGHT
                               : estimate.totalBytes();
        progress = beginStage(m_options.monitor, "connectivity", 1);
    };
    // PERFORMANCE: the element stiffness matrices are computed once: the connectivity pass keeps them (within
    // KEPT_STIFFNESS_BUDGET_FRACTION of the budget left) for the assembly of Kgg, which adds the same matrices in the
    // same order (the same bits); the store is released after the last assembly, before the factorisation.
    // Measured: setup + assembly of the 1 M DOF plate 1.57 s -> 1.2 s (KEPT_STIFFNESS_BUDGET_FRACTION).
    // The condensed path assembles no Kgg (each unit computes its own matrices): nothing is kept for it, so that the
    // memory plan of the condensation does not start with element matrices no one reads.
    const bool isCondensedPath = m_options.condensation && !m_options.condensation->isFullPath();
    modelOptions.keptStiffnessBudgetFraction =
        isCondensedPath || m_options.isStreamed ? 0.0 : KEPT_STIFFNESS_BUDGET_FRACTION;
    if (!m_options.externalUnits.empty())
    {
        // On the condensed path an external unit is a unit of the interface problem (CondensedStatics).
        modelOptions.replacedElements = [&](const assembly::ElementSet& plainSet, const assembly::DofMap& plainDofs)
        {
            ExternalUnitPath::Inputs inputs;
            inputs.snapshot = &snapshot;
            inputs.elements = &plainSet;
            inputs.dofMap = &plainDofs;
            inputs.requests = m_options.externalUnits;
            inputs.subcases = subcases;
            externalPath = std::make_unique<ExternalUnitPath>(inputs, budget, sink);
            refuseReplacedRequests(externalPath->replacedElements(), m_options, sink);
            stopOnErrors(sink, "the external units");
            bool isEnergyRequested = false;
            for (output::OutputPlan& plan : plans)
            {
                isEnergyRequested = isEnergyRequested || plan.of(output::OutputKind::StrainEnergy).isRequested;
                plan = externalPath->restrictPlan(plan, plainSet);
            }
            for (output::OutputPlan& plan : recoveryPlans)
                plan = externalPath->restrictPlan(plan, plainSet);
            if (isEnergyRequested)
            {
                base::Diagnostic dropped;
                dropped.severity = base::Severity::Warning;
                dropped.code = "external-unit-strain-energy";
                dropped.card = "ElemStrainEnergyOutput";
                dropped.message =
                    "the element strain energies are not written in an analysis with external units: "
                    "their percentages need the energy of the plugins' models, which the analysis does "
                    "not have; take the zone's energy from its plugin (½·u_aᵀ·K*·u_a)";
                sink.report(std::move(dropped));
            }
            return externalPath->replacedElements();
        };
    }
    AnalysisModel model(snapshot, m_options.analysis, budget, pool, cancellation, sink, modelOptions);
    results.timings.parts += model.timings();
    if (progress != nullptr)
        progress->advance();
    const elements::SnapshotIndex& index = model.index();
    const assembly::ElementSet& elementSet = model.elementSet();
    const assembly::DofMap& dofMap = model.dofMap();
    const input::ParamTable* params = model.params();
    results.gSetSize = dofMap.gSetSize();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const linalg::DenseKernels kernels(m_options.kernel, linalg::DenseKernelOptions{pool});
    results.kernel = std::string(linalg::kernelName(m_options.kernel));
    const assembly::LoadVectorBuilder loadBuilder(elementSet, dofMap, budget);
    const output::NodalResultBuilder tableBuilder(dofMap);
    const output::ElementResultBuilder elementTableBuilder(elementSet, dofMap, m_options.recoveryRangeElements);
    const output::StiffnessRecovery stiffnessRecovery(elementSet, dofMap, pool);
    if (m_options.keepsLoadBasis)
    {
        // A bar or beam under a line load (PLOAD1) is recovered in the station layout of its loaded case, which the
        // superposition of the units' tables does not give: refused.
        for (const StaticSubcase& subcase : subcases)
        {
            if (subcase.isCombination || !subcase.load)
                continue;
            const auto lineLoads = loadBuilder.lineLoadsOf(*subcase.load);
            if (std::any_of(lineLoads.begin(), lineLoads.end(), [](const auto& batch) { return !batch.empty(); }))
                sink.report(subcaseProblem(
                    subcase.label, base::Severity::Error, "load-basis-line-loads",
                    "expected a subcase without line loads (BeamDistributedLoad) in the load basis handed over: its "
                    "bar and beam "
                    "forces are written in the station layout of the loaded case, not superposed from its units; "
                    "solve it in a static run"));
        }
        stopOnErrors(sink, "the load basis");
        results.loadBasis = std::make_shared<KeptLoadBasis>();
        results.loadBasis->index = model.sharedIndex();
        results.loadBasis->dofMap = model.sharedDofMap();
        results.loadBasis->excludedCases = excludedCases;
        for (const StaticSubcase& subcase : subcases)
            if (subcase.isCombination)
                results.loadBasis->combinations.push_back({subcase.label, subcase.combined, subcase.coefficients});
    }
    if (sectionRun)
    {
        // A bar or beam of a section under a line load is refused for its subcase (SectionLoadRun::checkLineLoads).
        for (const StaticSubcase& subcase : subcases)
            if (!subcase.isCombination && subcase.load)
                sectionRun->checkLineLoads(subcase.label, loadBuilder.lineLoadsOf(*subcase.load), sink);
        stopOnErrors(sink, "the section and cut loads");
    }
    // The initial strains of a case with a thermal or deformation load, for that part of its section and cut loads.
    const auto sectionTemperaturesOf = [&](const StaticSubcase& subcase) -> std::unique_ptr<assembly::TemperatureField>
    {
        if (!sectionRun)
            return nullptr;
        return initialStrainsOf(snapshot, index.points(), subcase, budget, sink);
    };
    const auto offerSectionCase = [&](const StaticSubcase& subcase, std::span<const double> displacements)
    {
        if (!sectionRun)
            return;
        const std::unique_ptr<assembly::TemperatureField> temperatures = sectionTemperaturesOf(subcase);
        sectionRun->offerCase(subcase.label, displacements, temperatures ? &temperatures->temperatures() : nullptr,
                              budget);
    };
    // The initial strains of a case with a thermal or deformation load, for that part of its contour forces.
    const auto zoneTemperaturesOf = [&](const StaticSubcase& subcase) -> std::unique_ptr<assembly::TemperatureField>
    {
        if (!zoneRun)
            return nullptr;
        return initialStrainsOf(snapshot, index.points(), subcase, budget, sink);
    };
    // The contour data of a case known by its vectors (a SUBCOM, the condensed path).
    // On the condensed path, the zones that are units not recovered take their contour forces from the units'
    // boundary forces.
    const auto offerZoneCase = [&](const StaticSubcase& subcase, std::span<const double> displacements,
                                   std::span<const double> applied, std::span<const double> spcForces,
                                   std::span<const double> mpcForces,
                                   std::span<const CondensedStatics::UnitBoundaryForces> unitForces = {})
    {
        if (!zoneRun)
            return;
        const auto column = [](std::span<const double> values) -> linalg::ConstDenseMatrixView
        { return {values.data(), values.size(), values.empty() ? 0U : 1U, values.size()}; };
        std::vector<output::ZoneGivenForces> given; // bounded by the zones
        for (const output::ZoneDefinition& zone : m_options.zones.zones)
            for (const CondensedStatics::UnitBoundaryForces& unit : unitForces)
                if (zone.unit != 0 && unit.unitId == zone.unit)
                    given.push_back({zone.id, unit.dofs, unit.forces, unit.hasInteriorReactions});
        const std::unique_ptr<assembly::TemperatureField> temperatures = zoneTemperaturesOf(subcase);
        zoneRun->offerCase(subcase.label,
                           {column(displacements), column(applied), column(spcForces), column(mpcForces)},
                           temperatures ? &temperatures->temperatures() : nullptr, budget, given);
    };

    // PARAM GRDPNT, and the reference point of the resultants (that grid point, the origin otherwise).
    const std::int32_t weightReferenceNode =
        assembly::findIntegerParam(params, "WeightReferenceNode", std::nullopt, sink)
            .value_or(NO_WEIGHT_REFERENCE_NODE);
    base::Vec3 referencePoint{};
    if (weightReferenceNode > 0)
    {
        if (const std::optional<std::int32_t> row = index.points().findGrid(weightReferenceNode))
            referencePoint = dofMap.positionInBasic(*row);
        else
            sink.report(assembly::badParamDiagnostic("WeightReferenceNode", "-1, 0 or a grid point of the model"));
    }

    // 3. The constraint sets of the solved subcases (SPC, MPC, BCONTACT, FreeBodySupport and PARAM INREL), and the
    // groups of subcases that share one: one assembly and one factorisation per group.
    std::vector<std::size_t> solved; // indices into subcases
    for (std::size_t s = 0; s < subcases.size(); ++s)
        if (!subcases[s].isCombination)
            solved.push_back(s);
    std::vector<assembly::SpcSet> spcSets;
    spcSets.reserve(subcases.size());
    std::vector<std::vector<base::DofIndex>> supports(subcases.size());
    std::vector<assembly::InertiaReliefMode> reliefModes(subcases.size(), assembly::InertiaReliefMode::None);
    // A subcase whose SPC set and EnforcedMotionConstr terms an earlier one has takes that one's constraints
    // (SpcSet::withLoadSet: the same arrays, its diagnostics already reported) instead of a pass over the g set of its
    // own.
    const std::vector<base::EntityId> enforcedSets = assembly::SpcSet::enforcedSetIds(constraints);
    struct SpcKey
    {
        std::optional<base::EntityId> spc;
        std::vector<input::LoadTerm> terms;
        std::size_t subcase = 0;
    };
    std::vector<SpcKey> madeSpcSets; // bounded by the subcases
    for (std::size_t s = 0; s < subcases.size(); ++s)
    {
        std::vector<input::LoadTerm> terms =
            assembly::SpcSet::enforcedTermsOf(snapshot.findLoads(), enforcedSets, subcases[s].load);
        const auto made =
            std::find_if(madeSpcSets.begin(), madeSpcSets.end(),
                         [&](const SpcKey& key)
                         {
                             return key.spc == subcases[s].spc &&
                                    std::equal(key.terms.begin(), key.terms.end(), terms.begin(), terms.end(),
                                               [](const input::LoadTerm& a, const input::LoadTerm& b)
                                               { return a.set == b.set && a.factor == b.factor; });
                         });
        if (made != madeSpcSets.end())
            spcSets.push_back(spcSets[made->subcase].withLoadSet(subcases[s].load));
        else
        {
            spcSets.emplace_back(constraints, snapshot.findLoads(), dofMap, subcases[s].spc, subcases[s].load, budget,
                                 sink);
            madeSpcSets.push_back({subcases[s].spc, std::move(terms), s});
        }
        if (subcases[s].isCombination)
            continue;
        supports[s] = assembly::supportDofsOf(snapshot, dofMap, subcases[s].tableId, sink);
        reliefModes[s] = reliefModeOf(params, subcases[s].label, sink);
    }
    stopOnErrors(sink, "the assembly");
    std::vector<std::vector<std::size_t>> groups; // bounded by the subcases
    for (const std::size_t s : solved)
    {
        bool isPlaced = false;
        for (std::vector<std::size_t>& group : groups)
        {
            const std::size_t first = group.front();
            if (subcases[first].mpc == subcases[s].mpc && subcases[first].contact == subcases[s].contact &&
                subcases[first].nonStructuralMass == subcases[s].nonStructuralMass &&
                spcSets[first].hasSameConstrainedDofs(spcSets[s]) && supports[first] == supports[s] &&
                reliefModes[first] == reliefModes[s])
            {
                group.push_back(s);
                isPlaced = true;
                break;
            }
        }
        if (!isPlaced)
            groups.push_back({s});
    }
    // An external unit without M* in an analysis that uses the mass (the reduced mass is required when the analysis
    // uses mass: inertial loads, inertia relief).
    if (externalPath)
    {
        bool isInertial = false;
        for (const std::size_t s : solved)
            isInertial = isInertial || reliefModes[s] != assembly::InertiaReliefMode::None ||
                         (subcases[s].load && loadBuilder.needsMass(*subcases[s].load));
        externalPath->checkMass(isInertial, weightReferenceNode != NO_WEIGHT_REFERENCE_NODE,
                                "inertial loads or inertia relief", sink);
        stopOnErrors(sink, "the external units");
    }
    // The load basis handed over: the basis of each constraint set chosen now, before any factorisation
    // or condensation, and the memory its units keep checked against the budget (planLoadBasis).
    LoadBasisPlan basisPlan;
    // (On the full path streamed the basis of each set is chosen by StreamedStatics, its memory in the streamed plan.)
    if (results.loadBasis && !m_options.isStreamed)
    {
        basisPlan = planLoadBasis(snapshot, m_options, model, loadBuilder, subcases, spcSets, groups, reliefModes,
                                  budget, sink);
        results.loadBasis->estimatedBytes = basisPlan.estimatedBytes;
        results.timings.loadBasis += watch.lap();
    }
    // The subcases a SUBCOM combines keep their vectors.
    std::vector<std::uint8_t> isCombined(subcases.size(), 0);
    for (const StaticSubcase& subcase : subcases)
        for (const base::EntityId term : subcase.combined)
            for (std::size_t s = 0; s < subcases.size(); ++s)
                if (subcases[s].label == term)
                    isCombined[s] = 1;
    // The relations' forces of a case whose grid point forces split their multiPointConstrForce rows, or that a SUBCOM
    // combines (its own may).
    const auto takeRelationForces = [&](CaseState& state, std::size_t s, const assembly::RigidElimination* gm)
    {
        if (!isMpcSplit || gm == nullptr || state.mpcForces.empty() ||
            !(state.plan->of(output::OutputKind::GridPointForces).isRequested || isCombined[s] != 0))
            return;
        state.isMpcSplit = true;
        state.mpcRelationForces = gm->relationForces(state.mpcForces, budget);
    };
    results.timings.setup += watch.lap();

    std::vector<SubcaseTables> tablesOfSubcase(subcases.size());
    std::vector<StoredCase> stored; // bounded by the subcases combined
    bool hasGridPointWeight = false;

    // The condensed path: the units of the plan condensed once for every subcase, the
    // outputs of the units not recovered left out of the plans (and listed). A plan that condenses no unit is the full
    // path.
    std::unique_ptr<CondensedStatics> condensed;
    std::vector<output::OutputPlan> condensedPlans;   // per subcase, with a condensed path
    std::vector<output::OutputPlan> basisOutputPlans; // the load basis handed over: no output (the condensed path's)
    if (isCondensedPath)
    {
        progress = beginStage(m_options.monitor, "condensation", 1);
        CondensedStatics::Inputs condensedInputs;
        condensedInputs.snapshot = &snapshot;
        condensedInputs.model = &model;
        condensedInputs.plan = m_options.condensation.get();
        condensedInputs.loads = &loadBuilder;
        condensedInputs.subcases = subcases;
        condensedInputs.spcSets = spcSets;
        condensedInputs.solved = solved;
        condensedInputs.plans = hasKnownOutputs ? std::span<const output::OutputPlan>(recoveryPlans)
                                                : std::span<const output::OutputPlan>(plans);
        condensedInputs.recoveredUnits = m_options.recoveredUnits;
        // The units of the section and cut elements are recovered: their forces need their displacements.
        for (const output::SectionDefinition& section : m_options.sectionLoads.sections)
            for (const output::SectionMember& member : section.members)
                condensedInputs.elementsToRecover.push_back(member.elementId);
        for (const output::CutDefinition& cut : m_options.sectionLoads.cuts)
            condensedInputs.elementsToRecover.insert(condensedInputs.elementsToRecover.end(), cut.elements.begin(),
                                                     cut.elements.end());
        // The units of the members of the zones too: their contour forces need the displacements of their elements.
        // A zone that is a condensed unit of the plan gives its contour data from the unit's boundary, its u_a and
        // K*·u_a − f*·α, with no recovery.
        for (const output::ZoneDefinition& zone : m_options.zones.zones)
        {
            if (isCondensedUnitZone(zone, *m_options.condensation))
                continue;
            condensedInputs.elementsToRecover.insert(condensedInputs.elementsToRecover.end(), zone.elements.begin(),
                                                     zone.elements.end());
            for (const output::EnclosedElement& enclosed : zone.enclosed)
                condensedInputs.elementsToRecover.push_back(enclosed.elementId);
        }
        // And the units of the elements of a MonitorNodeForceSum3: their grid point forces need their displacements.
        for (const output::MonitorPoint3Definition& definition : m_options.monitorPoints3)
            condensedInputs.elementsToRecover.insert(condensedInputs.elementsToRecover.end(),
                                                     definition.elements.begin(), definition.elements.end());
        std::sort(condensedInputs.elementsToRecover.begin(), condensedInputs.elementsToRecover.end());
        condensedInputs.tolerance = m_options.loadBasisTolerance;
        for (const std::size_t s : solved)
            condensedInputs.hasInertiaRelief =
                condensedInputs.hasInertiaRelief ||
                (reliefModes[s] != assembly::InertiaReliefMode::None && !supports[s].empty());
        condensedInputs.kernels = &kernels;
        condensedInputs.pool = pool;
        condensedInputs.cancellation = cancellation;
        condensedInputs.cache = m_options.cache;
        condensedInputs.cacheModelTag = m_options.cacheModelTag;
        condensedInputs.designStamps = m_options.designStamps;
        // The static solution reads no equation of a unit beside what CondensedStatics reads, so that a
        // unit taken from the cache and not recovered is built without them (CondensedStatics decides with the rest of
        // the analysis: no inertia relief, no system kept); the units' boundary data kept by the structure cache; the
        // stores of the cache finished when the analysis ends, off its critical path.
        condensedInputs.structures = m_options.structures;
        condensedInputs.keptFactors = m_options.keptFactors;
        condensedInputs.takesHitsWithoutEquations = true;
        condensedInputs.verifiesCacheHits = m_options.verifiesCacheHits;
        condensedInputs.defersCacheStores = true;
        // The memory plan counts what the results hold to the end: the tables, the vectors of the subcases a SUBCOM
        // combines and those kept for the caller.
        std::uint64_t storedCases = m_options.keptDisplacements.size();
        for (const std::uint8_t combined : isCombined)
            storedCases += combined;
        condensedInputs.resultBytes =
            resultTableBytes + storedCases * STORED_CASE_VECTORS * static_cast<std::uint64_t>(gSize) * sizeof(double);
        condensedInputs.largeUnitMinimumFlops = m_options.largeUnitMinimumFlops;
        condensedInputs.memoryMode = m_options.memoryMode;
        // A free body under inertia relief solved on conditioned supports, but by an analysis whose
        // interface factor serves the sensitivities of a design point.
        condensedInputs.conditionsReliefSupports = !m_options.keepsSystems;
        condensedInputs.stage = m_options.condensedStage;
        condensedInputs.keepsSystems = m_options.keepsSystems;
        condensedInputs.recoversDrivenUnits = m_options.recoversDrivenUnits;
        condensedInputs.alsoRecoveredUnits = m_options.alsoRecoveredUnits;
        condensedInputs.externalUnits = externalPath.get();
        if (results.loadBasis)
        {
            // The load basis handed over: no case, so no output and no unit recovered (its blocks recover the
            // interiors they read); the constraint sets kept with their interface factors, and the interior factors of
            // the units the cases load kept while the budget allows (the basis vectors are condensed with them).
            basisOutputPlans.assign(subcases.size(), output::OutputPlan{});
            condensedInputs.plans = basisOutputPlans;
            condensedInputs.recoveredUnits = std::vector<std::int64_t>{};
            condensedInputs.elementsToRecover.clear();
            condensedInputs.resultBytes = 0;
            condensedInputs.keepsSystems = true;
            condensedInputs.handsOverLoadBasis = true;
            // The memory mode applies to the basis (a block reads up to the largest set's basis at
            // once), but not to a run whose kept sets also serve the sensitivities (they read the units' equations).
            for (const std::unique_ptr<StaticLoadCases>& cases : basisPlan.cases)
                condensedInputs.loadBasisColumnCount =
                    std::max(condensedInputs.loadBasisColumnCount, static_cast<std::int64_t>(cases->basisSize()));
            if (m_options.keepsSystems)
                condensedInputs.memoryMode = reduction::MemoryMode::Speed;
        }
        // The units condensed with their mass for their M* and Craig-Bampton matrices (the Condensation stage).
        if (m_options.condensedStage == CondensedStage::Condensation &&
            (m_options.condensesWithMass || m_options.unitModes.count > 0 ||
             m_options.unitModes.cutoffEigenvalue > 0.0))
            condensedInputs.isCraigBampton = true;
        if (m_options.condensedStage == CondensedStage::Condensation || isInterfaceStage)
        {
            // No unit recovered and no table kept: the run stops after the condensation (or after the interfaces). (The
            // MemoryPlan stage keeps the inputs of a complete run: its estimate is the complete run's.)
            condensedInputs.recoveredUnits = std::vector<std::int64_t>{};
            condensedInputs.elementsToRecover.clear();
            condensedInputs.resultBytes = 0;
        }
        condensed = std::make_unique<CondensedStatics>(condensedInputs, budget, sink);
        if (results.loadBasis)
        {
            // The elements of each condensed unit, by the unit's index in the kept sets (the plan's condensed units in
            // order): the blocks recover the interiors of the units whose elements they read.
            std::uint32_t unitIndex = 0;
            for (const reduction::UnitPlan& unit : m_options.condensation->units())
            {
                if (unit.decision != reduction::UnitDecision::Condensed)
                    continue;
                for (const base::EntityId elementId : unit.elementIds)
                    results.loadBasis->condensedElements.push_back({elementId, unitIndex});
                ++unitIndex;
            }
            std::sort(results.loadBasis->condensedElements.begin(), results.loadBasis->condensedElements.end(),
                      [](const KeptLoadBasis::CondensedElement& a, const KeptLoadBasis::CondensedElement& b)
                      { return a.elementId < b.elementId; });
        }
        if (isStagedRun && !isInterfaceStage)
        {
            results.path = AnalysisPath::Condensed;
            // The fixed-interface modes asked of the Condensation stage, their modal parts stored with the rest.
            const reduction::ModeRequest& modes = m_options.unitModes;
            if (m_options.condensedStage == CondensedStage::Condensation &&
                (modes.count > 0 || modes.cutoffEigenvalue > 0.0))
            {
                CraigBamptonRecord request;
                request.modeCount = modes.count;
                request.cutoffEigenvalue = modes.isCount() ? 0.0 : modes.cutoffEigenvalue;
                request.reference = modes.isCount() ? "count" : "cut-off";
                condensed->reduceModes(modes, request, false, budget, sink);
            }
            condensed->finishCacheStores(sink, pool);
            results.condensed = condensed->record();
            results.timings.factorisation += watch.lap();
            if (m_options.condensedStage == CondensedStage::Condensation)
            {
                results.reducedUnits = condensed->reducedUnits(budget);
                results.reducedCaseLabels = condensed->caseLabels();
                results.largestPivotRatio = condensed->largestPivotRatio();
                results.solvedVectorCount = condensed->solvedVectorCount();
            }
            results.loadCaseCount = static_cast<std::int64_t>(solved.size());
            results.diagnostics = sink.diagnostics();
            return results;
        }
        for (const output::OutputPlan& plan : plans)
            condensedPlans.push_back(givenUnit
                                         ? condensed->unitOnlyOutputPlanOf(condensed->outputPlanOf(plan), *givenUnit)
                                         : condensed->outputPlanOf(plan));
        if (progress != nullptr)
            progress->advance();
        results.path = AnalysisPath::Condensed;
        results.timings.factorisation += watch.lap();
    }

    // The tables, checks and envelopes of one case.
    // `vectorised`: the case's element tables made with the others of its block (ElementResultBuilder::buildCases),
    // or null for the per-case recovery.
    const auto recoverCase = [&](std::size_t s, const CaseState& state, output::SubcaseCheck check,
                                 std::vector<output::ElementResultTable>* vectorised = nullptr)
    {
        const StaticSubcase& subcase = *state.subcase;
        const output::OutputPlan& plan = *state.plan;
        SubcaseTables& tables = tablesOfSubcase[s];
        const auto isKept = [&](std::string_view kind)
        { return m_options.keepsCaseTables || !envelopes.takes(std::string(kind)); };
        const auto addNodal = [&](output::OutputKind kind, std::span<const double> values)
        {
            const output::OutputRequestPlan& request = plan.of(kind);
            if (!request.isRequested || values.empty())
                return;
            output::NodalResultTable table = tableBuilder.build(std::string(output::tableKindOf(kind)), subcase.label,
                                                                values, budget, request.isAll, request.members);
            envelopes.offer(table);
            if (isKept(table.kind))
                tables.nodal.push_back(std::move(table));
        };
        addNodal(output::OutputKind::Displacement, state.displacements);
        addNodal(output::OutputKind::AppliedLoads, state.applied);
        addNodal(output::OutputKind::SpcForces, state.spcForces);
        addNodal(output::OutputKind::MpcForces, state.mpcForces);

        const std::unique_ptr<assembly::TemperatureField> temperatures =
            initialStrainsOf(snapshot, index.points(), subcase, budget, sink);
        const elements::ElementTemperatures* temperatureView = temperatures ? &temperatures->temperatures() : nullptr;
        const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
        std::vector<output::ElementResultTable> elementTables;
        if (vectorised != nullptr)
            elementTables = std::move(*vectorised);
        else
            elementTableBuilder.build(subcase.label, state.elementDisplacementsOrAll(), state.lineLoads, budget,
                                      elementTables, pool, cancellation, temperatureView, &selection);
        if (plan.of(output::OutputKind::StrainEnergy).isRequested)
            stiffnessRecovery.strainEnergy(subcase.label, state.elementDisplacementsOrAll(),
                                           plan.of(output::OutputKind::StrainEnergy), budget, elementTables,
                                           cancellation);
        for (output::ElementResultTable& table : elementTables)
        {
            envelopes.offer(table);
            if (isKept(table.kind))
                tables.element.push_back(std::move(table));
        }
        if (plan.of(output::OutputKind::GridPointForces).isRequested)
        {
            output::StiffnessRecovery::NodalState nodal;
            nodal.displacements = state.elementDisplacementsOrAll();
            nodal.appliedLoads = state.applied;
            nodal.spcForces = state.spcForces;
            nodal.mpcForces = state.mpcForces;
            nodal.isConstrained = state.isConstrained;
            nodal.temperatures = temperatureView;
            nodal.isMpcSplit = state.isMpcSplit;
            nodal.mpcRelationForces = state.mpcRelationForces.span();
            const output::OutputRequestPlan& request = plan.of(output::OutputKind::GridPointForces);
            tables.gridPointForces.push_back(
                state.gridPointForceUnits != nullptr
                    ? stiffnessRecovery.gridPointForces(subcase.label, nodal, request, *state.gridPointForceUnits,
                                                        budget, cancellation)
                    : stiffnessRecovery.gridPointForces(subcase.label, nodal, request, budget, cancellation));
        }
        if (!monitors3.empty())
        {
            // The MonitorNodeForceSum3 of the case (output::MonitorPoint3): the element rows of its ELEMSET with the
            // thermal part of its load, and the rows appliedLoad, singlePointConstrForce and multiPointConstrForce that
            // XFLAG keeps at its GRIDSET.
            const output::MonitorNodalLoads nodal{state.applied, state.spcForces, state.mpcForces};
            MonitorPoint3Loads loads;
            loads.subcase = subcase.label;
            for (const output::MonitorPoint3& monitor : monitors3)
                loads.loads.push_back(
                    monitor.loadsOf(elementSet, state.elementDisplacementsOrAll(), nodal, temperatureView));
            tables.monitorPoints3 = std::move(loads);
        }
        check.subcase = subcase.label;
        check.appliedLoads = output::LoadResultant::of(dofMap, state.applied, referencePoint);
        check.spcForces = output::LoadResultant::of(dofMap, state.spcForces, referencePoint);
        addResultant(check.spcForces, state.unrecoveredReactions, referencePoint);
        tables.check = std::move(check);
        if (results.areTablesStreamed)
        {
            // The tables leave as soon as they are made: nothing of them stays for the next cases.
            output::ResultArrays batch;
            batch.nodalTables = std::move(tables.nodal);
            batch.elementTables = std::move(tables.element);
            batch.gridPointForceTables = std::move(tables.gridPointForces);
            tables.nodal.clear();
            tables.element.clear();
            tables.gridPointForces.clear();
            m_options.resultSink(std::move(batch));
        }
        if (std::find(m_options.keptDisplacements.begin(), m_options.keptDisplacements.end(), subcase.label) !=
            m_options.keptDisplacements.end())
        {
            KeptDisplacements kept;
            kept.subcase = subcase.label;
            kept.values = base::BudgetedBuffer<double>(budget, gSize, "kept displacements");
            std::copy(state.displacements.begin(), state.displacements.end(), kept.values.data());
            results.keptDisplacements.push_back(std::move(kept));
        }
        if (isCombined[s] != 0)
        {
            StoredCase kept;
            kept.label = subcase.label;
            const auto copy = [&](std::span<const double> values, const char* label)
            {
                base::BudgetedBuffer<double> buffer(budget, gSize, label);
                if (!values.empty())
                    std::copy(values.begin(), values.end(), buffer.data());
                return buffer;
            };
            kept.displacements = copy(state.displacements, "SUBCOM term displacements");
            kept.applied = copy(state.applied, "SUBCOM term applied loads");
            kept.spcForces = copy(state.spcForces, "SUBCOM term SPC forces");
            kept.mpcForces = copy(state.mpcForces, "SUBCOM term MPC forces");
            if (state.isMpcSplit)
            {
                kept.mpcRelationForces = base::BudgetedBuffer<assembly::RelationForce>(
                    budget, state.mpcRelationForces.size(), "SUBCOM term MPC forces by relation");
                std::copy(state.mpcRelationForces.data(),
                          state.mpcRelationForces.data() + state.mpcRelationForces.size(),
                          kept.mpcRelationForces.data());
            }
            kept.isConstrained = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "SUBCOM term constrained DOFs");
            std::copy(state.isConstrained.begin(), state.isConstrained.end(), kept.isConstrained.data());
            kept.unrecoveredReactions = state.unrecoveredReactions;
            kept.unitForces.assign(state.unitForces.begin(), state.unitForces.end());
            if (state.gridPointForceUnits != nullptr)
                kept.unitRows.assign(state.gridPointForceUnits->unitRows.begin(),
                                     state.gridPointForceUnits->unitRows.end());
            stored.push_back(std::move(kept));
        }
    };

    // Whether the cases of a condensed constraint set take their recovered units' results
    // unit by unit (CondensedStatics::GroupInputs::visitUnit, each unit's recovery made once for every case rather than
    // in every batch): the units recovered in sequence, and every output of the cases taken either by the envelopes
    // from the units' rows (the displacements and the element tables, then never kept per case) or whole from the
    // cases' vectors, which hold every unit's (the applied loads, the SPC forces, the checks); so no MPC force, grid
    // point force, strain energy, MonitorNodeForceSum3, section, zone, SUBCOM term, kept displacements, thermal or line
    // load, external unit, cases with other plans, tables recovered case by case on request, nor what the units keep
    // per case beyond a quarter of the memory left. When it holds, the envelopes' rows are defined from the whole
    // tables of the first case's plan at rest (EnvelopeAccumulator::defineRows), so that the units' rows take their
    // places: the rows, their order and their extremes of the cases recovered batch by batch, to the bit.
    const auto takesUnitByUnit = [&](std::span<const std::size_t> group) -> bool
    {
        if (!condensed || !condensed->recoversInSequence() || m_options.keepsCaseTables || envelopes.isEmpty() ||
            externalPath || sectionRun || zoneRun || !monitors3.empty() || m_options.isRecoveredCaseByCase)
            return false;
        const StaticSubcase& firstCase = subcases[group.front()];
        const output::OutputPlan& first = condensedPlans[group.front()];
        const output::OutputRequestPlan& firstDisplacement = first.of(output::OutputKind::Displacement);
        const output::ElementOutputSelection firstSelection = output::elementSelectionOf(first);
        for (const std::size_t s : group)
        {
            const StaticSubcase& subcase = subcases[s];
            const output::OutputPlan& plan = condensedPlans[s];
            if (isCombined[s] != 0 || subcase.hasInitialStrains() ||
                std::find(m_options.keptDisplacements.begin(), m_options.keptDisplacements.end(), subcase.label) !=
                    m_options.keptDisplacements.end())
                return false;
            if (subcase.load)
            {
                const auto lineLoads = loadBuilder.lineLoadsOf(*subcase.load);
                if (std::any_of(lineLoads.begin(), lineLoads.end(), [](const auto& batch) { return !batch.empty(); }))
                    return false;
            }
            for (const output::OutputKind kind :
                 {output::OutputKind::MpcForces, output::OutputKind::GridPointForces, output::OutputKind::StrainEnergy})
                if (plan.of(kind).isRequested)
                    return false;
            const output::OutputRequestPlan& displacement = plan.of(output::OutputKind::Displacement);
            if (displacement.isRequested != firstDisplacement.isRequested ||
                displacement.isAll != firstDisplacement.isAll || displacement.members != firstDisplacement.members ||
                !output::elementSelectionOf(plan).selectsSameAs(firstSelection))
                return false;
        }
        if (firstDisplacement.isRequested && !envelopes.takes("displacements"))
            return false;
        if (condensed->unitByUnitBytes(group.size()) > budget.available() / 4)
            return false;
        // The whole tables of the first case's plan at rest: every element table taken by the envelopes.
        base::BudgetedBuffer<double> zeros(budget, gSize, "unit by unit tables at rest");
        std::vector<output::ElementResultTable> wholeTables;
        if (firstSelection.stress.isRequested || firstSelection.strain.isRequested || firstSelection.forces.isRequested)
            elementTableBuilder.build(firstCase.label, zeros.span(), {}, budget, wholeTables, pool, cancellation,
                                      nullptr, &firstSelection);
        for (const output::ElementResultTable& table : wholeTables)
            if (!envelopes.takes(table.kind))
                return false;
        if (firstDisplacement.isRequested)
            envelopes.defineRows(tableBuilder.build("displacements", firstCase.label, zeros.span(), budget,
                                                    firstDisplacement.isAll, firstDisplacement.members));
        for (const output::ElementResultTable& table : wholeTables)
            envelopes.defineRows(table);
        return true;
    };

    // The load basis handed over of a constraint set: its cases, its units named and their coefficients
    // (the displacements, or the condensed path's parts, follow).
    const auto keptBasisSetOf =
        [&](std::size_t groupIndex, const std::vector<std::size_t>& group, const StaticLoadCases& loadCases)
    {
        KeptLoadBasisSet kept;
        kept.constraintSet = static_cast<std::int32_t>(groupIndex);
        for (const std::size_t s : group)
            kept.cases.push_back(subcases[s].label);
        for (std::size_t b = 0; b < loadCases.basisSize(); ++b)
        {
            const StaticLoadCases::VectorName name = loadCases.nameOf(b);
            LoadBasisUnit unit;
            unit.family = name.family;
            unit.kind = name.kind;
            unit.source = name.source;
            unit.loadSet = name.loadSet;
            unit.subcase = name.subcase;
            // A thermal unit: the key of temperature and deformation sets of the case it is the
            // thermal part of (StaticLoadCases: one vector per key), whose initial strains its element tables are
            // recovered with (its DEFORM with its temperatures).
            if (name.family == assembly::LoadFamily::Thermal)
                for (const std::size_t s : group)
                    if (subcases[s].label == name.subcase && subcases[s].hasInitialStrains())
                    {
                        const InitialStrainKey key = initialStrainKeyOf(subcases[s]);
                        unit.temperatureLoad = key.temperatureLoad.value_or(0);
                        unit.temperatureInitial = key.temperatureInitial;
                        unit.deformation = key.deformation.value_or(0);
                        unit.loadSet = unit.temperatureLoad;
                    }
            if (name.dof >= 0)
            {
                const assembly::DofLabel label = dofMap.labelOf(name.dof);
                unit.grid = dofMap.pointIdOf(name.dof);
                unit.component = label.kind == input::PointKind::Grid ? label.component : 0;
            }
            kept.units.push_back(unit);
        }
        const linalg::DenseMatrix& coefficients = loadCases.coefficients();
        kept.coefficients =
            linalg::DenseMatrix(budget, coefficients.rows(), coefficients.columns(), "load basis coefficients");
        for (std::size_t c = 0; c < coefficients.columns(); ++c)
            for (std::size_t i = 0; i < coefficients.rows(); ++i)
                kept.coefficients(i, c) = coefficients(i, c);
        return kept;
    };
    // The reports of a constraint set's load basis (StaticResults::loadBases).
    const auto recordLoadBases =
        [&](std::size_t groupIndex, const std::vector<std::size_t>& group, const StaticLoadCases& loadCases)
    {
        for (const assembly::LoadBasisReport& report : loadCases.reports())
        {
            LoadBasisRecord record;
            record.constraintSet = static_cast<std::int32_t>(groupIndex);
            for (const std::size_t s : group)
                record.subcases.push_back(subcases[s].label);
            record.report = report;
            results.loadBases.push_back(std::move(record));
        }
    };

    for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
    {
        const std::vector<std::size_t>& group = groups[groupIndex];
        checkCancelled(cancellation);
        const StaticSubcase& first = subcases[group.front()];
        const std::optional<std::int32_t> paramSubcase = first.label;
        const assembly::InertiaReliefMode reliefMode = reliefModes[group.front()];
        const std::vector<base::DofIndex>& userSupports = supports[group.front()];

        // 4. Gm, the sets, the pattern, Kgg and Knn (ConstrainedSystem), with the r set of FreeBodySupport or the
        // supports PARAM INREL = −2 chooses.
        const assembly::SpcSet& groupSpc = spcSets[group.front()];
        if (reliefMode == assembly::InertiaReliefMode::Automatic)
        {
            if (!userSupports.empty())
                sink.report(
                    subcaseProblem(first.label, base::Severity::Warning, "free-body-support-ignored",
                                   "PARAM InertiaRelief = -2 chooses its own supports: the FreeBodySupport points are "
                                   "ignored"));
            // GRID PS removes components from the model (a plane model's out-of-plane DOFs) and is accepted; an SPC
            // would hold the body.
            if (!groupSpc.sbDofs().empty())
                sink.report(
                    subcaseProblem(first.label, base::Severity::Error, "inertia-relief-constrained",
                                   "PARAM InertiaRelief = -2 is the inertia relief of a free body, found single-point "
                                   "constraints of the subcase (SinglePointConstr, SinglePointConstrList); remove "
                                   "them, or use FreeBodySupport with "
                                   "PARAM InertiaRelief = -1"));
        }
        else
        {
            if (reliefMode == assembly::InertiaReliefMode::OnSupports && userSupports.empty())
                sink.report(subcaseProblem(
                    first.label, base::Severity::Error, "inertia-relief-without-free-body-support",
                    "PARAM InertiaRelief = -1 needs the FreeBodySupport or FreeBodySupportSet DOFs of the r set, found "
                    "none (dofSetsTable); add FreeBodySupport or use "
                    "PARAM InertiaRelief = -2"));
            if (reliefMode == assembly::InertiaReliefMode::OnSupports &&
                userSupports.size() > assembly::MAX_SUPPORT_DOFS)
                sink.report(subcaseProblem(
                    first.label, base::Severity::Error, "inertia-relief-too-many-supports",
                    base::concatenate({"inertia relief takes at most six FreeBodySupport DOFs (a "
                                       "statically determinate support), found ",
                                       std::to_string(userSupports.size()), "; keep one DOF per rigid-body motion"})));
            if (reliefMode == assembly::InertiaReliefMode::None && !userSupports.empty())
                sink.report(subcaseProblem(
                    first.label, base::Severity::Warning, "free-body-support-as-constraint",
                    "FreeBodySupport without PARAM InertiaRelief: the r-set DOFs are held as constraints (no "
                    "inertia relief); their reactions are in the SPC forces"));
        }
        if (reliefMode != assembly::InertiaReliefMode::None)
            for (const std::size_t s : group)
                if (!spcSets[s].enforcedDisplacements().empty())
                    sink.report(subcaseProblem(
                        subcases[s].label, base::Severity::Error, "inertia-relief-enforced",
                        "inertia relief with enforced displacements (SinglePointConstr D, EnforcedMotionConstr) is not "
                        "supported by this version of O_o_Solver; remove one of them"));
        results.timings.setup += watch.lap();

        if (m_options.isStreamed)
        {
            // The full path streamed: the set solved without its factor, its outputs made region
            // by region (StreamedStatics).
            for (const std::size_t s : group)
                if (isCombined[s] != 0 && !results.loadBasis)
                    sink.report(
                        subcaseProblem(subcases[s].label, base::Severity::Error, "streamed-unsupported",
                                       "expected no SubcaseCombination on the full path streamed (its regions keep no "
                                       "case's vectors to combine); solve it on the full path "
                                       "(not streamed) or the condensed path"));
            stopOnErrors(sink, "the assembly");
            // The load basis: no output of the cases, the declared quantities and cuts of the vectors.
            const std::vector<output::OutputPlan> noOutputs(results.loadBasis ? subcases.size() : 0);
            StreamedStatics::GroupInputs streamedInputs;
            streamedInputs.model = &model;
            streamedInputs.loads = &loadBuilder;
            streamedInputs.elementTables = &elementTableBuilder;
            streamedInputs.nodalTables = &tableBuilder;
            streamedInputs.subcases = subcases;
            streamedInputs.cases = group;
            streamedInputs.plans = plans;
            streamedInputs.spc = &groupSpc;
            streamedInputs.supports = userSupports;
            streamedInputs.relief = reliefMode;
            streamedInputs.selection =
                ConstraintSelection{first.label, first.spc, first.mpc, first.load, first.contact};
            streamedInputs.pivotPolicy = pivotPolicyOf(params, paramSubcase, sink);
            streamedInputs.envelopes = envelopes.isEmpty() ? nullptr : &envelopes;
            streamedInputs.keepsCaseTables = m_options.keepsCaseTables;
            streamedInputs.resultBytes = resultTableBytes;
            streamedInputs.constraintSet = static_cast<std::int32_t>(groupIndex);
            streamedInputs.kernels = &kernels;
            streamedInputs.pool = pool;
            streamedInputs.cancellation = cancellation;
            streamedInputs.tolerance = m_options.loadBasisTolerance;
            streamedInputs.kernel = m_options.kernel;
            if (results.loadBasis)
            {
                streamedInputs.plans = noOutputs;
                // No result table either: the deck's output requests are not made for the cases of a basis.
                streamedInputs.resultBytes = 0;
                streamedInputs.keepsLoadBasis = true;
                streamedInputs.basisQuantities = m_options.basisQuantities.get();
                streamedInputs.basisSelection = m_options.basisSelection;
                streamedInputs.basisFixedRows = basisFixedRows;
                std::int32_t unitsBefore = 0;
                for (const KeptLoadBasisSet& kept : results.loadBasis->sets)
                    unitsBefore += static_cast<std::int32_t>(kept.units.size());
                streamedInputs.firstUnit = unitsBefore + 1;
            }
            // The partial factor retention: none under MemoryMode::LowMemory unless it is given.
            StreamedOptions streamedOptions = m_options.streamed;
            if (streamedOptions.retention == StreamedRetention::Auto &&
                m_options.memoryMode == reduction::MemoryMode::LowMemory)
                streamedOptions.retention = StreamedRetention::None;
            StreamedStatics::GroupResult streamed =
                StreamedStatics::solveGroup(streamedInputs, streamedOptions, budget, sink);
            if (!streamed.singularityTable.empty())
                results.singularityTables.push_back(streamed.singularityTable);
            if (results.loadBasis)
            {
                // The set's units, coefficients and what its vectors gave (no displacement).
                recordLoadBases(groupIndex, group, *streamed.loadCases);
                KeptLoadBasisSet kept = keptBasisSetOf(groupIndex, group, *streamed.loadCases);
                kept.streamed = streamed.basis;
                kept.fixedRows = streamed.fixedRows;
                results.loadBasis->sets.push_back(std::move(kept));
                streamed.loadCases.reset();
            }
            for (std::size_t c = 0; c < group.size(); ++c)
            {
                SubcaseTables& tables = tablesOfSubcase[group[c]];
                for (output::NodalResultTable& table : streamed.nodal[c])
                    tables.nodal.push_back(std::move(table));
                for (output::ElementResultTable& table : streamed.element[c])
                    tables.element.push_back(std::move(table));
            }
            results.largestFsetSize = std::max(results.largestFsetSize, streamed.record.equations);
            results.largestPivotRatio = std::max(results.largestPivotRatio, streamed.record.largestPivotRatio);
            results.solvedVectorCount += streamed.record.basisVectors;
            results.loadCaseCount += static_cast<std::int64_t>(group.size());
            ++results.constraintSetCount;
            results.streamed.push_back(std::move(streamed.record));
            results.path = AnalysisPath::Streamed;
            results.timings.solve += watch.lap();
            continue;
        }

        if (condensed)
        {
            stopOnErrors(sink, "the assembly");
            CondensedStatics::GroupInputs groupInputs;
            groupInputs.constraintSet = static_cast<std::int32_t>(groupIndex);
            groupInputs.cases = group;
            groupInputs.spc = &spcSets[group.front()];
            groupInputs.supports = userSupports;
            groupInputs.relief = reliefMode;
            groupInputs.selection = ConstraintSelection{first.label, first.spc, first.mpc, first.load, first.contact};
            groupInputs.pivotPolicy = pivotPolicyOf(params, paramSubcase, sink);
            groupInputs.singularity = assembly::SingularityOptions::fromParams(params, paramSubcase, sink);
            groupInputs.referencePoint = referencePoint;
            if (isInterfaceStage)
            {
                // The set's interface problem handed over (solved too when asked), no case.
                groupInputs.exportsInterface = true;
                groupInputs.solvesExport = m_options.solvesInterfaceExport;
                const std::string table = condensed->solveGroup(
                    groupInputs, [](std::size_t, const CondensedStatics::CaseResult&) {}, budget, sink);
                if (!table.empty())
                    results.singularityTables.push_back(table);
                ++results.constraintSetCount;
                results.loadCaseCount += static_cast<std::int64_t>(group.size());
                results.timings.factorisation += watch.lap();
                continue;
            }
            // The interface's solve replaced by the caller's values (StaticOptions::givenBoundaries).
            linalg::DenseMatrix givenColumns; // GivenLocal: the columns of the set's cases (bounded by the unit)
            if (hasGivenBoundaries)
            {
                const auto entry = std::find_if(m_options.givenBoundaries.begin(), m_options.givenBoundaries.end(),
                                                [&](const StaticOptions::GivenBoundary& given)
                                                {
                                                    return given.mode == reduction::BoundarySolveMode::GivenLocal ||
                                                           given.constraintSet == static_cast<std::int32_t>(groupIndex);
                                                });
                if (entry == m_options.givenBoundaries.end())
                    throw std::invalid_argument(base::concatenate(
                        {"recover: no solution given for constraint set ", std::to_string(groupIndex), " (subcase ",
                         std::to_string(first.label),
                         "); give the solution of every constraint set the subcases asked for make"}));
                groupInputs.boundaryMode = entry->mode;
                groupInputs.isGivenPerCase = entry->isPerCase;
                groupInputs.givenUnitId = entry->unitId;
                groupInputs.given = entry->values;
                if (entry->mode == reduction::BoundarySolveMode::GivenLocal)
                {
                    // Its columns are labelled by their subcases: those of this set, in its order.
                    if (entry->caseLabels.size() != entry->values.columns)
                        throw std::invalid_argument(base::concatenate(
                            {"recover: the boundary displacements have ", std::to_string(entry->values.columns),
                             " columns for ", std::to_string(entry->caseLabels.size()),
                             " subcases; give one column per subcase"}));
                    givenColumns =
                        linalg::DenseMatrix(budget, entry->values.rows, group.size(), "given boundary cases");
                    for (std::size_t c = 0; c < group.size(); ++c)
                    {
                        const base::EntityId label = subcases[group[c]].label;
                        const auto column = std::find(entry->caseLabels.begin(), entry->caseLabels.end(), label);
                        if (column == entry->caseLabels.end())
                            throw std::invalid_argument(base::concatenate(
                                {"recover: no boundary displacements given for subcase ", std::to_string(label),
                                 "; give a column for every subcase the analysis solves"}));
                        const auto position = static_cast<std::size_t>(column - entry->caseLabels.begin());
                        for (std::size_t i = 0; i < entry->values.rows; ++i)
                            givenColumns(i, c) = entry->values(i, position);
                    }
                    groupInputs.given = givenColumns.view();
                }
            }
            if (results.loadBasis)
            {
                // The load basis handed over: the set's interface factorised and kept, no case; the basis vectors of
                // the set (the full path's) solved on it (solveCondensedBasis).
                groupInputs.solvesCases = false;
                const std::string table = condensed->solveGroup(
                    groupInputs, [](std::size_t, const CondensedStatics::CaseResult&) {}, budget, sink);
                if (!table.empty())
                    results.singularityTables.push_back(table);
                ++results.constraintSetCount;
                results.loadCaseCount += static_cast<std::int64_t>(group.size());
                results.timings.factorisation += watch.lap();
                const StaticLoadCases& loadCases = *basisPlan.cases[groupIndex];
                recordLoadBases(groupIndex, group, loadCases);
                progress = beginStage(m_options.monitor, "solve", loadCases.basisSize());
                KeptLoadBasisSet kept = keptBasisSetOf(groupIndex, group, loadCases);
                const std::shared_ptr<const CondensedFactorisedSet> keptSet = condensed->lastKeptSet();
                // The fixed rows of the units' grid point forces, on the set's Gm and held DOFs.
                const std::shared_ptr<KeptFixedRows> fixedRows = BasisFixedRows::plan(
                    basisFixedRows, dofMap, keptSet->gm(), keptSet->isConstrained(), loadCases.basisSize(), budget);
                kept.condensed =
                    solveCondensedBasis(keptSet, loadCases, kernels, budget, sink, cancellation, fixedRows.get());
                kept.fixedRows = fixedRows;
                stopOnErrors(sink, "the solve");
                results.loadBasis->sets.push_back(std::move(kept));
                results.solvedVectorCount += static_cast<std::int64_t>(loadCases.basisSize());
                if (progress != nullptr)
                    progress->advance(loadCases.basisSize());
                results.timings.solve += watch.lap();
                continue;
            }
            progress = beginStage(m_options.monitor, "recovery", group.size());
            // The recovered units' results unit by unit (takesUnitByUnit): the cases' own
            // recovery then takes the rows of no recovered unit (CondensedStatics::interfaceOutputPlanOf), each unit
            // giving its rows to the envelopes (GroupInputs::visitUnit, below).
            const bool isUnitByUnit = takesUnitByUnit(group);
            std::vector<output::OutputPlan> unitByUnitPlans; // by subcase, those of the set's cases replaced
            if (isUnitByUnit)
            {
                unitByUnitPlans = condensedPlans;
                for (const std::size_t s : group)
                    unitByUnitPlans[s] = condensed->interfaceOutputPlanOf(condensedPlans[s]);
            }
            const std::vector<output::OutputPlan>& casePlans = isUnitByUnit ? unitByUnitPlans : condensedPlans;
            // The element tables of the block cases of each batch (blockCasesOf), made when the batch's displacements
            // are known (GroupInputs::visitBatch) and taken by each case's recovery; not in a streamed run, whose batch
            // of cases would hold the tables of every case of the batch at once.
            // PERFORMANCE: the condensed path's cases recovered by blocks rather than each alone from the g-set vector
            // of its batch.
            // Measured: 400 plates, 32 cases, 40 recovered, 5.1 → 1.9 s of recovery.
            const BlockCases condensedBlocks =
                blockCasesOf(group, subcases, casePlans, loadBuilder, snapshot, index.points(),
                             m_options.isRecoveredCaseByCase || results.areTablesStreamed, budget);
            // The rows of each recovered unit (unit by unit): its element tables, made as the batches make the cases'
            // (by blocks of cases, or case by case), and its interior points' displacements, offered to the envelopes
            // in the order of the cases. The plan of a unit made at its first batch.
            std::vector<std::unique_ptr<const output::OutputPlan>> unitPlans; // bounded by the condensed units
            std::vector<output::ElementOutputSelection> unitSelections;
            if (isUnitByUnit)
                groupInputs.visitUnit = [&](const CondensedStatics::GroupInputs::UnitCases& unitCases)
                {
                    const std::size_t u = unitCases.unit;
                    if (unitPlans.size() <= u)
                    {
                        unitPlans.resize(u + 1);
                        unitSelections.resize(u + 1);
                    }
                    if (!unitPlans[u])
                    {
                        unitPlans[u] = std::make_unique<const output::OutputPlan>(
                            condensed->unitOutputPlanOf(condensedPlans[group.front()], u));
                        unitSelections[u] = output::elementSelectionOf(*unitPlans[u]);
                    }
                    const output::ElementOutputSelection& selection = unitSelections[u];
                    const output::OutputRequestPlan& displacement = unitPlans[u]->of(output::OutputKind::Displacement);
                    const linalg::ConstDenseMatrixView& displacements = unitCases.displacements;
                    const std::size_t count = displacements.columns;
                    std::vector<std::int32_t> labels; // bounded by a batch
                    for (std::size_t c = 0; c < count; ++c)
                        labels.push_back(subcases[group[unitCases.firstCaseInGroup + c]].label);
                    const auto columnOf = [&](std::size_t c)
                    { return std::span<const double>(displacements.pointer(0, c), displacements.rows); };
                    std::vector<std::vector<output::ElementResultTable>> tables(count);
                    if (selection.stress.isRequested || selection.strain.isRequested || selection.forces.isRequested)
                    {
                        if (condensedBlocks.count > 0)
                            elementTableBuilder.buildCases(labels, displacements, condensedBlocks.count, budget, tables,
                                                           pool, cancellation, &selection);
                        else
                            for (std::size_t c = 0; c < count; ++c)
                                elementTableBuilder.build(labels[c], columnOf(c), {}, budget, tables[c], pool,
                                                          cancellation, nullptr, &selection);
                    }
                    for (std::size_t c = 0; c < count; ++c)
                    {
                        if (displacement.isRequested)
                            envelopes.offer(tableBuilder.build("displacements", labels[c], columnOf(c), budget,
                                                               displacement.isAll, displacement.members));
                        for (const output::ElementResultTable& table : tables[c])
                            envelopes.offer(table);
                    }
                };
            std::vector<std::vector<output::ElementResultTable>> batchTables; // per case of the batch
            std::size_t batchFirst = 0;
            if (condensedBlocks.count > 0)
                groupInputs.visitBatch = [&](std::size_t firstCaseInGroup, linalg::ConstDenseMatrixView displacements)
                {
                    batchFirst = firstCaseInGroup;
                    recoverBlockCases(elementTableBuilder, condensedBlocks, subcases, group, firstCaseInGroup,
                                      displacements, displacements.columns, budget, pool, cancellation, batchTables);
                };
            const std::string table = condensed->solveGroup(
                groupInputs,
                [&](std::size_t caseInGroup, const CondensedStatics::CaseResult& result)
                {
                    const std::size_t s = group[caseInGroup];
                    CaseState state;
                    state.subcase = &subcases[s];
                    state.plan = &casePlans[s];
                    state.displacements = result.displacements;
                    state.applied = result.applied;
                    state.spcForces = result.spcForces;
                    state.mpcForces = result.mpcForces;
                    state.isConstrained = result.isConstrained;
                    takeRelationForces(state, s, result.mpcRelations);
                    if (subcases[s].load)
                        state.lineLoads = loadBuilder.lineLoadsOf(*subcases[s].load);
                    state.gridPointForceUnits = result.gridPointForceUnits;
                    state.unrecoveredReactions = result.unrecoveredReactions;
                    state.unitForces = result.unitBoundaryForces;
                    // The external units: their contour data recorded, their grid point force rows beside
                    // those of the condensed units not recovered.
                    MergedUnitRows merged;
                    if (externalPath)
                    {
                        merged = mergeUnitRows(
                            result.gridPointForceUnits, externalPath->replacedElements(),
                            externalPath->recordCase(s, state.displacements, state.unrecoveredReactions, budget));
                        merged.view = {merged.excluded, merged.rows};
                        state.gridPointForceUnits = &merged.view;
                    }
                    output::SubcaseCheck check;
                    check.epsilon = result.epsilon;
                    check.externalWork = result.externalWork;
                    check.supportAccelerations = result.supportAccelerations;
                    offerSectionCase(subcases[s], result.displacements);
                    offerZoneCase(subcases[s], result.displacements, result.applied, result.spcForces, result.mpcForces,
                                  result.unitBoundaryForces);
                    recoverCase(s, state, std::move(check),
                                condensedBlocks.isBlockCase[caseInGroup] != 0 ? &batchTables[caseInGroup - batchFirst]
                                                                              : nullptr);
                    if (progress != nullptr)
                        progress->advance();
                },
                budget, sink);
            if (!table.empty())
                results.singularityTables.push_back(table);
            ++results.constraintSetCount;
            results.loadCaseCount += static_cast<std::int64_t>(group.size());
            results.timings.recovery += watch.lap();
            continue;
        }

        progress = beginStage(m_options.monitor, "assembly", 1);
        ConstrainedSystemOptions systemOptions;
        systemOptions.spc = &groupSpc;
        systemOptions.choosesSupports = reliefMode == assembly::InertiaReliefMode::Automatic;
        if (!systemOptions.choosesSupports)
            systemOptions.supportDofs = userSupports;
        systemOptions.referencePoint = referencePoint;
        systemOptions.structures = m_options.structures;
        systemOptions.keptStiffness = &model.keptStiffness();
        systemOptions.releasesKeptStiffness = groupIndex + 1 == groups.size();
        if (externalPath)
        {
            systemOptions.heldDofs = externalPath->heldDofs();
            systemOptions.extraCouplings = externalPath->couplings();
            systemOptions.addStiffness = [&](linalg::BlockSparseMatrix& kgg) { externalPath->addStiffness(kgg); };
        }
        const ConstraintSelection selection{first.label, first.spc, first.mpc, first.load, first.contact};
        const ConstrainedSystem system(model, selection, budget, pool, cancellation, sink, systemOptions);
        results.timings.parts += system.timings();
        if (m_options.structures != nullptr)
            ++(system.isPatternReused() ? results.structureReuse.patternsReused : results.structureReuse.patternsBuilt);
        const assembly::RigidElimination& gm = system.gm();
        const assembly::DofSets& solveSets = system.solveSets();
        const linalg::BlockSparseMatrix& kgg = system.kgg();
        const linalg::BlockSparseMatrix& knn = system.knn();
        const std::vector<base::DofIndex>& rSet = system.supportDofs();
        // The fixed rows of the units' grid point forces the load basis keeps, chosen before any solve.
        const std::shared_ptr<KeptFixedRows> fixedRows =
            results.loadBasis ? BasisFixedRows::plan(basisFixedRows, dofMap, gm, system.isConstrained(),
                                                     basisPlan.cases[groupIndex]->basisSize(), budget)
                              : nullptr;
        if (!system.singularityTable().empty())
            results.singularityTables.push_back(system.singularityTable());
        // PARAM USETPRT >= 0 (-1, the default, prints nothing) asks for the USET table of the DOF sets solved.
        if (assembly::findIntegerParam(params, "DofSetPrint", paramSubcase, sink).value_or(-1) >= 0)
        {
            constexpr std::array<assembly::DofSet, 6> PRINTED_SETS{assembly::DofSet::G, assembly::DofSet::M,
                                                                   assembly::DofSet::S, assembly::DofSet::F,
                                                                   assembly::DofSet::R, assembly::DofSet::L};
            std::ostringstream uset;
            assembly::Uset::write(uset, solveSets, dofMap, PRINTED_SETS);
            results.usetTables.push_back(uset.str());
        }
        if (progress != nullptr)
            progress->advance();
        results.timings.assembly += watch.lap();

        // Inertia relief on a body free apart from six FreeBodySupport DOFs, in a static analysis that keeps no factor
        // for the sensitivities: the relief made on the rigid-body motions Φ from the geometry (rigidSupportModes, the
        // test of a free body from Knn before the factorisation), the body solved held by six translations of the
        // farthest points (assembly::chooseConditionedSupports) rather than by the FreeBodySupport DOFs, the solution
        // refined (refineRelievedSolution), the element results made from it, the displacements then moved by the rigid
        // motion that brings the FreeBodySupport DOFs back to rest (u ← u − D·u_r, D = Φ·C): the balanced loads give
        // the same deformation on any statically determinate support, and a thin body held at one point is
        // ill-conditioned. Otherwise (a body a constraint holds besides its supports, fewer than six of them, INREL =
        // −2, a design point's analysis) the FreeBodySupport DOFs held and D solved from the stiffness.
        const bool hasRelief = reliefMode != assembly::InertiaReliefMode::None && !rSet.empty();
        RigidSupports rigid;
        if (hasRelief && reliefMode == assembly::InertiaReliefMode::OnSupports && !m_options.keepsSystems)
            rigid = rigidSupportModes(dofMap, knn, gm, system.isConstrained(), rSet, budget);
        std::optional<linalg::DenseMatrix>& rigidModes = rigid.modes;
        if (hasRelief)
        {
            ReliefRecord reliefRecord;
            reliefRecord.constraintSet = static_cast<std::int32_t>(groupIndex);
            reliefRecord.reaction = rigid.reaction;
            results.reliefs.push_back(reliefRecord);
        }
        std::vector<base::DofIndex> solveSupports; // the conditioned supports (empty: the r set held)
        std::unique_ptr<assembly::DofSets> conditionedSets;
        if (rigidModes)
        {
            base::BudgetedBuffer<std::uint8_t> isCandidate(budget, gSize, "relief support candidates");
            for (std::size_t dof = 0; dof < gSize; ++dof)
            {
                const assembly::ExclusiveSet set = solveSets.exclusiveSetOf(static_cast<base::DofIndex>(dof));
                isCandidate[dof] = set == assembly::ExclusiveSet::B || set == assembly::ExclusiveSet::R ? 1 : 0;
            }
            solveSupports = assembly::chooseConditionedSupports(dofMap, isCandidate.span());
            if (solveSupports.size() == assembly::MAX_SUPPORT_DOFS)
                conditionedSets = std::make_unique<assembly::DofSets>(solveSets.withSupports(solveSupports));
            else
            {
                solveSupports.clear();
                rigidModes.reset();
            }
        }
        if (hasRelief)
            results.reliefs.back().isRigid = rigidModes.has_value();
        for (const base::DofIndex dof : solveSupports)
            results.reliefs.back().solveSupports.push_back({dofMap.pointIdOf(dof), dofMap.labelOf(dof).component});
        const assembly::DofSets& factorSets = conditionedSets ? *conditionedSets : solveSets;

        // The l-set factorisation.
        FsetSolverOptions solverOptions;
        solverOptions.pivotPolicy = pivotPolicyOf(params, paramSubcase, sink);
        solverOptions.cancellation = cancellation;
        solverOptions.structures = m_options.structures;
        const auto lSetSize = static_cast<std::uint64_t>(factorSets.count(assembly::DofSet::L));
        solverOptions.progress = beginStage(m_options.monitor, "factorisation", lSetSize);
        std::unique_ptr<FsetSolver> factorised =
            factoriseWithShapes(knn, system.graph(), dofMap, factorSets, kernels, solverOptions, gm, budget, pool);
        const FsetSolver& solver = *factorised;
        results.timings.parts += solver.timings();
        if (m_options.structures != nullptr)
        {
            ++(solver.reuse().isOrderingReused ? results.structureReuse.orderingsReused
                                               : results.structureReuse.orderingsBuilt);
            if (solver.equationCount() > 0)
            {
                ++(solver.reuse().isPlanReused ? results.structureReuse.plansReused
                                               : results.structureReuse.plansBuilt);
                ++(solver.reuse().isStorageReused ? results.structureReuse.storagesReused
                                                  : results.structureReuse.storagesAllocated);
            }
        }
        results.largestFsetSize = std::max<std::int64_t>(results.largestFsetSize, solver.equationCount());
        results.largestPivotRatio = std::max(results.largestPivotRatio, solver.pivotDiagnostics().largestRatio());
        reportKeptPivots(
            solver.pivotDiagnostics(), [&solver](std::int32_t equation) { return solver.nameOf(equation); },
            first.label, sink);
        ++results.constraintSetCount;
        results.timings.factorisation += watch.lap();
        const auto equations = static_cast<std::size_t>(solver.equationCount());

        // The mass, for the inertial loads, inertia relief and the grid point weight generator.
        bool needsMass = hasRelief || (weightReferenceNode != NO_WEIGHT_REFERENCE_NODE && !hasGridPointWeight);
        for (const std::size_t s : group)
            needsMass =
                needsMass || (!results.loadBasis && subcases[s].load && loadBuilder.needsMass(*subcases[s].load));
        std::unique_ptr<linalg::BlockSparseMatrix> mass;
        if (needsMass)
        {
            mass = std::make_unique<linalg::BlockSparseMatrix>(system.pattern(), budget, "mass Mgg");
            const assembly::NonStructuralMassShares shares(elementSet, first.nonStructuralMass, budget, sink);
            stopOnErrors(sink, "the selected non-structural mass");
            assembly::MassAssembler(elementSet, dofMap, shares,
                                    assembly::MassAssemblyOptions::fromParams(params, paramSubcase, sink))
                .assemble(*mass, pool, cancellation);
            // The reduced mass of the external units, as given (in consistent units: WTMASS scales the elements'
            // mass only, and a mass given as a matrix is added after it, unscaled).
            if (externalPath)
                externalPath->addMass(*mass);
        }
        if (weightReferenceNode != NO_WEIGHT_REFERENCE_NODE && !hasGridPointWeight)
        {
            // PARAM GRDPNT: the rigid-body mass properties, in the units of the input masses (WTMASS removed).
            const double weightToMass =
                assembly::findRealParam(params, "WeightToMass", paramSubcase, sink).value_or(1.0);
            const assembly::RigidBodyModes modes(dofMap, referencePoint, budget);
            base::MatN<6> rigidMass = modes.rigidBodyMass(*mass, budget);
            if (weightToMass != 0.0)
                rigidMass *= 1.0 / weightToMass;
            results.gridPointWeight = assembly::GridPointWeight::fromRigidBodyMass(
                rigidMass, std::max(weightReferenceNode, 0), referencePoint);
            hasGridPointWeight = true;
        }

        // Inertia relief: the rigid-body motions of a unit motion of each support DOF, exact for a body free apart from
        // six supports (rigidSupportModes), otherwise from the stiffness, D_l = −K_ll⁻¹·K_lr, D_r = I,
        // u_m = Gm·u_n, zero on the s set.
        std::unique_ptr<assembly::InertiaRelief> relief;
        if (rigidModes)
            relief = std::make_unique<assembly::InertiaRelief>(std::move(*rigidModes), *mass, budget);
        else if (hasRelief)
        {
            linalg::DenseMatrix rightHandSides(budget, equations, rSet.size(), "inertia relief K_lr");
            base::BudgetedBuffer<double> unit(budget, gSize, "inertia relief unit motion");
            for (std::size_t j = 0; j < rSet.size(); ++j)
            {
                std::fill(unit.data(), unit.data() + gSize, 0.0);
                unit[static_cast<std::size_t>(rSet[j])] = 1.0;
                const base::BudgetedBuffer<double> column = multiply(knn, unit.span(), budget, "inertia relief K e");
                for (std::size_t e = 0; e < equations; ++e)
                    rightHandSides(e, j) =
                        -column[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))];
            }
            solver.solve(rightHandSides.view(), cancellation);
            linalg::DenseMatrix modes(budget, gSize, rSet.size(), "inertia relief support modes");
            base::BudgetedBuffer<double> mode(budget, gSize, "inertia relief support mode");
            for (std::size_t j = 0; j < rSet.size(); ++j)
            {
                std::fill(mode.data(), mode.data() + gSize, 0.0);
                for (std::size_t e = 0; e < equations; ++e)
                    mode[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))] =
                        rightHandSides(e, j);
                mode[static_cast<std::size_t>(rSet[j])] = 1.0;
                gm.expandToG(mode.span());
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    modes(dof, j) = mode[dof];
            }
            relief = std::make_unique<assembly::InertiaRelief>(std::move(modes), *mass, budget);
        }

        // 5. The load cases of the group and their basis (the load basis handed over: chosen before, planLoadBasis).
        std::unique_ptr<StaticLoadCases> ownCases;
        if (!results.loadBasis)
        {
            StaticLoadCases::Inputs caseInputs;
            caseInputs.snapshot = &snapshot;
            caseInputs.points = &index.points();
            caseInputs.dofMap = &dofMap;
            caseInputs.loads = &loadBuilder;
            for (const std::size_t s : group)
            {
                caseInputs.cases.push_back(&subcases[s]);
                caseInputs.spcs.push_back(&spcSets[s]);
            }
            caseInputs.mass = mass.get();
            caseInputs.tolerance = m_options.loadBasisTolerance;
            caseInputs.pool = pool;
            caseInputs.kernel = m_options.kernel;
            ownCases = std::make_unique<StaticLoadCases>(std::move(caseInputs), budget, sink);
        }
        const StaticLoadCases& loadCases = ownCases ? *ownCases : *basisPlan.cases[groupIndex];
        // The loads of the zones of the external units go to their plugins, which give their condensed loads f* per
        // basis vector.
        if (externalPath)
        {
            externalPath->condenseLoads(static_cast<std::int32_t>(groupIndex), group, loadCases, budget, sink);
            stopOnErrors(sink, "the external units");
        }
        recordLoadBases(groupIndex, group, loadCases);
        results.timings.loadBasis += watch.lap();

        // The basis vectors: their loads (inertia relief applied), reduced to the l set with the enforced
        // displacements (Pf = P̄n,l + (Gmᵀ·Pm)_l − K_ls·u_s).
        const std::size_t k = loadCases.basisSize();
        results.solvedVectorCount += static_cast<std::int64_t>(k);
        results.loadCaseCount += static_cast<std::int64_t>(group.size());
        const bool hasMset = gm.rowCount() > 0;
        const bool removesRigidMotion = reliefMode == assembly::InertiaReliefMode::Automatic && hasRelief;
        // A free body on its conditioned supports: its element results made from the displacements as solved
        // on those supports (SolvedDisplacements), not from those of the FreeBodySupport DOFs' gauge, whose rigid
        // motion's round-off reaches the stiff elements (with fasteners 1e8 times the skins, 1e-6 of their forces
        // between two FreeBodySupport points): the element results do not depend on the FreeBodySupport DOFs.
        const bool isRegauged = !solveSupports.empty();
        // The load basis handed over keeps the displacements alone (planLoadBasis counted them, and the right-hand
        // sides); the enforced displacements only when the set has some.
        const bool isBasisOnly = results.loadBasis != nullptr;
        bool hasGroupEnforced = !isBasisOnly;
        for (const std::size_t s : group)
            hasGroupEnforced = hasGroupEnforced || !spcSets[s].enforcedDisplacements().empty();
        std::vector<linalg::DenseMatrix> basisResults; // the quantities of the basis vectors, g × k each
        std::array<bool, QUANTITY_COUNT> isUsed{};
        for (std::size_t q = 0; q < QUANTITY_COUNT; ++q)
        {
            isUsed[q] = isBasisOnly ? q == Displacements
                                    : (q != MpcForces || hasMset) &&
                                          (q != SolvedDisplacements || removesRigidMotion || isRegauged);
            basisResults.emplace_back(budget, gSize, isUsed[q] ? k : 0, "load basis results", pool);
        }
        // PERFORMANCE: the results of the basis vectors zeroed on the pool: g × k each.
        // Measured: 3.6 GB in all for 64 cases at 1 M DOFs, whose first touch took about 1 s on one thread.
        linalg::DenseMatrix enforcedOfBasis(budget, gSize, hasGroupEnforced ? k : 0,
                                            "load basis enforced displacements", pool);
        linalg::DenseMatrix rightHandSides(budget, equations, k, "static right-hand sides", pool);
        std::vector<std::array<double, assembly::MAX_SUPPORT_DOFS>> accelerations(k); // bounded by the basis
        // A free body solved on its conditioned supports is refined (refineRelievedSolution).
        const bool isRefined = !solveSupports.empty();
        // Basis vectors [first, end): their loads (inertia relief applied), reduced to the l set with the enforced
        // displacements, written in their own columns, on three g-set work vectors of the caller's.
        const auto makeVectors = [&](std::size_t first, std::size_t end)
        {
            base::BudgetedBuffer<double> loads(budget, gSize, "static loads Pg", base::FOR_OVERWRITE);
            base::BudgetedBuffer<double> enforced(budget, gSize, "enforced displacements", base::FOR_OVERWRITE);
            base::BudgetedBuffer<double> reduced(budget, gSize, "static loads Pn", base::FOR_OVERWRITE);
            for (std::size_t b = first; b < end; ++b)
            {
                checkCancelled(cancellation);
                // Only a vector of the enforced family enforces a displacement (StaticLoadCases::basisVector): the
                // others leave the work vector and their zeroed column alone, a third of a vector's passes over the g
                // set saved (the loop is bound by the memory's bandwidth).
                const bool isEnforcedFamily = loadCases.familyOf(b) == assembly::LoadFamily::Enforced;
                loadCases.basisVector(b, loads.span(), isEnforcedFamily ? enforced.span() : std::span<double>(), budget,
                                      sink);
                if (externalPath)
                    externalPath->applyToBasisVector(b, loads.span());
                if (relief)
                {
                    accelerations[b] = relief->accelerationsOf(loads.span());
                    relief->relieve(loads.span());
                }
                // appliedLoad as statics writes it (inertia relief's loads included), at the DOFs the basis keeps.
                if (fixedRows)
                    BasisFixedRows::take(*fixedRows, b, loads.span());
                std::copy(loads.data(), loads.data() + gSize, reduced.data());
                gm.reduceToN(reduced.span());
                bool hasEnforced = false;
                for (std::size_t dof = 0; dof < gSize && isEnforcedFamily; ++dof)
                    hasEnforced = hasEnforced || enforced[dof] != 0.0;
                if (hasEnforced)
                {
                    if (!hasGroupEnforced)
                        throw std::logic_error(
                            "StaticSolution: a basis vector enforces a displacement in a set "
                            "without enforced displacements (a bug of O_o_Solver: report it)");
                    const base::BudgetedBuffer<double> kus = multiply(knn, enforced.span(), budget, "Knn us");
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        reduced[dof] -= kus[dof];
                }
                for (std::size_t dof = 0; dof < gSize; ++dof)
                {
                    if (isUsed[Applied])
                        basisResults[Applied](dof, b) = loads[dof];
                    if (hasGroupEnforced && isEnforcedFamily)
                        enforcedOfBasis(dof, b) = enforced[dof];
                }
                for (std::size_t e = 0; e < equations; ++e)
                {
                    const auto dof = static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)));
                    rightHandSides(e, b) = reduced[dof];
                    if (isUsed[ReducedLoads])
                        basisResults[ReducedLoads](dof, b) = reduced[dof];
                }
                // The conditioned supports are l-set DOFs of the case: their loads enter the external work of the
                // displacements in the gauge of the FreeBodySupport DOFs (the work of the balanced loads is that of any
                // gauge).
                for (const base::DofIndex dof : solveSupports)
                    if (isUsed[ReducedLoads])
                        basisResults[ReducedLoads](static_cast<std::size_t>(dof), b) =
                            reduced[static_cast<std::size_t>(dof)];
            }
        };
        // PERFORMANCE: the basis vectors made in parallel by pieces of BASIS_VECTORS_PER_PIECE, each vector by the
        // code of one thread on the work vectors of its piece and written in its own columns, so that every value has
        // the bits of the serial loop. What they call is safe concurrently: StaticLoadCases,
        // assembly::LoadVectorBuilder, TemperatureField, InertialLoads, InertiaRelief, RigidElimination and the element
        // families' load facets hold no mutable state (their const methods only read their tables and write the
        // caller's vectors), the memory budget and the diagnostic sink are thread-safe and the sink orders what it
        // gets. An external unit's plugin (called on the calling thread) and a budget that cannot hold a piece's work
        // vectors per thread keep the serial loop.
        const std::size_t threadCount = pool != nullptr ? pool->threadCount() : 1;
        const std::uint64_t pieceBytes = 3 * static_cast<std::uint64_t>(gSize) * sizeof(double);
        const bool isParallel = threadCount > 1 && k > BASIS_VECTORS_PER_PIECE && externalPath == nullptr &&
                                budget.available() / BASIS_WORK_MEMORY_DIVISOR >= threadCount * pieceBytes;
        base::parallelFor(isParallel ? pool : nullptr, k,
                          isParallel ? BASIS_VECTORS_PER_PIECE : std::max<std::size_t>(k, 1), makeVectors,
                          cancellation);
        stopOnErrors(sink, "the solve");
        results.timings.assembly += watch.lap();

        // All the basis vectors at once.
        progress = beginStage(m_options.monitor, "solve", k);
        linalg::DenseMatrix relievedLoads(budget, equations, isRefined ? k : 0, "relief relieved loads");
        for (std::size_t b = 0; b < relievedLoads.columns(); ++b)
            std::copy_n(&rightHandSides(0, b), equations, &relievedLoads(0, b));
        solver.solve(rightHandSides.view(), cancellation);
        if (isRefined)
        {
            const RelievedSystem relieved{&solver, &kgg, &gm, relief.get(), relievedLoads.view()};
            results.reliefs.back().refinementCorrections =
                refineRelievedSolution(relieved, rightHandSides.view(), budget, pool, cancellation);
            // The accelerations of the relief made on Φ, ü_Φ, as those of the FreeBodySupport DOFs: ü_D = C⁻¹·ü_Φ =
            // Φ_S·ü_Φ.
            for (std::array<double, assembly::MAX_SUPPORT_DOFS>& acceleration : accelerations)
                acceleration = supportAccelerationsOf(*relief, rSet, acceleration);
        }
        if (progress != nullptr)
            progress->advance(k);
        results.timings.solve += watch.lap();

        // The results of the basis vectors: ug, the residual of the solve, the SPC and MPC forces. The thermal loads
        // are not applied loads of OLOAD (OLOAD reports the mechanical loads; the SPC forces balance both).
        const std::span<const std::uint8_t> isConstrained = system.isConstrained();
        {
            // PERFORMANCE: the products of Knn and Kgg with the basis vectors made a group of vectors at a time
            // (BlockSparseMatrix::multiply of a block: the matrix read once per group of columns, the columns shared
            // between the threads) in place of one product per vector on one thread; each vector's terms come in the
            // order of its own product, so the same bits whatever the group or the threads.
            // Measured: the 1 M DOF plate with 64 cases.
            const std::size_t group = basisProductGroup(gSize, k, isUsed[Residuals], budget);
            linalg::DenseMatrix solvedPart(budget, gSize, isUsed[Residuals] ? group : 0, "displacements of the l set");
            linalg::DenseMatrix products(budget, gSize, group, "stiffness products of the basis vectors");
            base::BudgetedBuffer<double> mpcForces(budget, isBasisOnly ? 0 : gSize, "MPC forces");
            linalg::DenseMatrix& displacements = basisResults[Displacements];
            for (std::size_t first = 0; first < k; first += group)
            {
                checkCancelled(cancellation);
                const std::size_t count = std::min(group, k - first);
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t b = first + j;
                    std::fill_n(&displacements(0, b), gSize, 0.0);
                    if (isUsed[Residuals])
                        std::fill_n(&solvedPart(0, j), gSize, 0.0);
                    for (std::size_t e = 0; e < equations; ++e)
                    {
                        const auto dof = static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)));
                        displacements(dof, b) = rightHandSides(e, b);
                        if (isUsed[Residuals])
                            solvedPart(dof, j) = rightHandSides(e, b);
                    }
                }
                // EPSILON: δP = Pf − K_ll·u_l on the l set; not for the load basis handed over.
                if (isUsed[Residuals])
                {
                    knn.multiply(solvedPart.view().block(0, 0, gSize, count), products.view().block(0, 0, gSize, count),
                                 pool);
                    for (std::size_t j = 0; j < count; ++j)
                        for (std::size_t e = 0; e < equations; ++e)
                        {
                            const auto dof =
                                static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)));
                            basisResults[Residuals](dof, first + j) =
                                basisResults[ReducedLoads](dof, first + j) - products(dof, j);
                        }
                }
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t b = first + j;
                    const std::span<double> ug(&displacements(0, b), gSize);
                    if (hasGroupEnforced)
                        for (std::size_t dof = 0; dof < gSize; ++dof)
                            if (enforcedOfBasis(dof, b) != 0.0)
                                ug[dof] = enforcedOfBasis(dof, b);
                    gm.expandToG(ug);
                    if (isRegauged)
                    {
                        if (isUsed[SolvedDisplacements])
                            std::copy(ug.begin(), ug.end(), &basisResults[SolvedDisplacements](0, b));
                        toSupportGauge(*relief, rigid.coefficients, rSet, ug);
                    }
                    if (removesRigidMotion)
                    {
                        if (isUsed[SolvedDisplacements])
                            std::copy(ug.begin(), ug.end(), &basisResults[SolvedDisplacements](0, b));
                        relief->removeRigidMotion(ug, budget);
                    }
                }
                if (isBasisOnly)
                    continue;
                kgg.multiply(displacements.view().block(0, first, gSize, count),
                             products.view().block(0, 0, gSize, count), pool);
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t b = first + j;
                    const std::span<const double> loads(&basisResults[Applied](0, b), gSize);
                    const std::span<const double> kug(&products(0, j), gSize);
                    gm.computeMpcForces(kug, loads, mpcForces.span());
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                    {
                        if (isConstrained[dof] != 0)
                            basisResults[SpcForces](dof, b) = kug[dof] - loads[dof] - mpcForces[dof];
                        if (hasMset)
                            basisResults[MpcForces](dof, b) = mpcForces[dof];
                    }
                    if (loadCases.familyOf(b) == assembly::LoadFamily::Thermal)
                        for (std::size_t dof = 0; dof < gSize; ++dof)
                            basisResults[Applied](dof, b) = 0.0;
                }
            }
        }

        // The load basis handed over: the units, their coefficients and their displacements kept, no
        // case recovered.
        if (results.loadBasis)
        {
            KeptLoadBasisSet kept = keptBasisSetOf(groupIndex, group, loadCases);
            kept.displacements = std::move(basisResults[Displacements]);
            kept.fixedRows = fixedRows;
            results.loadBasis->sets.push_back(std::move(kept));
            results.timings.recovery += watch.lap();
            continue;
        }

        // The section and cut loads of the basis vectors (computed on the load basis, then combined).
        std::optional<linalg::DenseMatrix> sectionBasis;
        if (sectionRun)
            sectionBasis.emplace(sectionRun->basisValues(
                basisResults[isRegauged ? SolvedDisplacements : Displacements].view(), budget, pool));
        // The contour data of the basis vectors (computed on the load basis, then combined), kept with
        // the coefficients of the cases for a linear plugin.
        std::optional<output::ZoneRun::BasisValues> zoneBasis;
        if (zoneRun)
        {
            zoneBasis.emplace(zoneRun->basisValues({basisResults[Displacements].view(), basisResults[Applied].view(),
                                                    basisResults[SpcForces].view(), basisResults[MpcForces].view()},
                                                   budget, pool));
            if (zoneRun->keepsBasis())
            {
                std::vector<std::int32_t> labels;
                bool hasThermalCases = false;
                for (const std::size_t s : group)
                {
                    labels.push_back(subcases[s].label);
                    hasThermalCases = hasThermalCases || subcases[s].hasInitialStrains();
                }
                linalg::DenseMatrix coefficients(budget, loadCases.coefficients().rows(),
                                                 loadCases.coefficients().columns(), "zone basis coefficients");
                for (std::size_t c = 0; c < coefficients.columns(); ++c)
                    for (std::size_t i = 0; i < coefficients.rows(); ++i)
                        coefficients(i, c) = loadCases.coefficients()(i, c);
                zoneRun->keepBasis(static_cast<std::int32_t>(groupIndex), labels, *zoneBasis, coefficients,
                                   hasThermalCases, budget);
            }
        }

        // 6. The cases: their linear results combined from the basis, batch by batch, then each recovered.
        progress = beginStage(m_options.monitor, "recovery", group.size());
        const output::ResultCombination combination(
            kernels,
            [&]
            {
                linalg::DenseMatrix copy(budget, loadCases.coefficients().rows(), loadCases.coefficients().columns(),
                                         "load case coefficients");
                for (std::size_t c = 0; c < copy.columns(); ++c)
                    for (std::size_t i = 0; i < copy.rows(); ++i)
                        copy(i, c) = loadCases.coefficients()(i, c);
                return copy;
            }());
        const std::size_t batch = combination.batchCaseCount(gSize, QUANTITY_COUNT, budget.available());
        // The cases whose element tables are recovered by blocks (blockCasesOf).
        const BlockCases blockCases = blockCasesOf(group, subcases, plans, loadBuilder, snapshot, index.points(),
                                                   m_options.isRecoveredCaseByCase, budget);
        // A streamed run holds the tables of STREAMED_SUBCASES_IN_FLIGHT subcases: its blocks stay within them.
        const std::size_t blockLimit = results.areTablesStreamed ? STREAMED_SUBCASES_IN_FLIGHT - 1 : batch;
        std::vector<linalg::DenseMatrix> combined; // per quantity, g × batch
        for (std::size_t q = 0; q < QUANTITY_COUNT; ++q)
            combined.emplace_back(budget, gSize, isUsed[q] ? batch : 0, "combined case results");
        for (std::size_t firstCase = 0; firstCase < group.size(); firstCase += batch)
        {
            const std::size_t count = std::min(batch, group.size() - firstCase);
            for (std::size_t q = 0; q < QUANTITY_COUNT; ++q)
                if (isUsed[q])
                    combination.combine(basisResults[q].view(), firstCase, count, combined[q].view(), budget);
            if (sectionRun)
            {
                // The section and cut loads of the batch from those of the basis, with the thermal part of each case
                // that has a thermal load.
                std::vector<std::int32_t> labels(count); // bounded by the batch
                for (std::size_t j = 0; j < count; ++j)
                    labels[j] = subcases[group[firstCase + j]].label;
                sectionRun->offerBatch(
                    *sectionBasis, combination, firstCase, labels,
                    [&](std::size_t j) { return sectionTemperaturesOf(subcases[group[firstCase + j]]); }, budget);
            }
            if (zoneRun)
            {
                // The contour data of the batch from those of the basis, with the thermal part of each thermal case.
                std::vector<std::int32_t> labels(count); // bounded by the batch
                for (std::size_t j = 0; j < count; ++j)
                    labels[j] = subcases[group[firstCase + j]].label;
                zoneRun->offerBatch(
                    *zoneBasis, combination, firstCase, labels,
                    [&](std::size_t j) { return zoneTemperaturesOf(subcases[group[firstCase + j]]); }, budget);
            }
            // The element tables of the block cases of the batch, made by blocks of at most blockLimit cases.
            std::vector<std::vector<output::ElementResultTable>> blockTables; // per case of the batch
            std::size_t blockEnd = 0;
            for (std::size_t j = 0; j < count; ++j)
            {
                checkCancelled(cancellation);
                const std::size_t s = group[firstCase + j];
                if (blockCases.isBlockCase[firstCase + j] != 0 && j >= blockEnd)
                {
                    // The block cases from j, up to blockLimit of them (recoverBlockCases cuts them by selection).
                    std::size_t end = j;
                    std::size_t taken = 0;
                    while (end < count && taken < blockLimit)
                        taken += blockCases.isBlockCase[firstCase + end++];
                    std::vector<std::vector<output::ElementResultTable>> made;
                    recoverBlockCases(
                        elementTableBuilder, blockCases, subcases, group, firstCase + j,
                        combined[isRegauged ? SolvedDisplacements : Displacements].view().block(0, j, gSize, end - j),
                        blockLimit, budget, pool, cancellation, made);
                    blockTables.resize(count);
                    for (std::size_t i = j; i < end; ++i)
                        blockTables[i] = std::move(made[i - j]);
                    blockEnd = end;
                }
                std::vector<output::ElementResultTable>* vectorised =
                    blockCases.isBlockCase[firstCase + j] != 0 ? &blockTables[j] : nullptr;
                const auto column = [&](Quantity q) -> std::span<const double>
                {
                    if (combined[q].columns() == 0)
                        return {};
                    return {&combined[q](0, j), gSize};
                };
                CaseState state;
                state.subcase = &subcases[s];
                state.plan = &plans[s];
                state.displacements = column(Displacements);
                if (isRegauged)
                    state.elementDisplacements = column(SolvedDisplacements);
                state.applied = column(Applied);
                state.spcForces = column(SpcForces);
                state.mpcForces = column(MpcForces);
                state.isConstrained = isConstrained;
                takeRelationForces(state, s, &gm);
                if (subcases[s].load)
                    state.lineLoads = loadBuilder.lineLoadsOf(*subcases[s].load);
                output::SubcaseCheck check;
                // EPSILON and the external work on the l set, with the displacements of the solve.
                const std::span<const double> solvedU =
                    removesRigidMotion || isRegauged ? column(SolvedDisplacements) : column(Displacements);
                const std::span<const double> reduced = column(ReducedLoads);
                const std::span<const double> residual = column(Residuals);
                double work = 0.0;
                double residualWork = 0.0;
                for (std::size_t dof = 0; dof < gSize; ++dof)
                {
                    work += solvedU[dof] * reduced[dof];
                    residualWork += solvedU[dof] * residual[dof];
                }
                check.epsilon = work != 0.0 ? residualWork / work : 0.0;
                check.externalWork = EXTERNAL_WORK_FACTOR * work;
                if (relief)
                    for (std::size_t r = 0; r < rSet.size(); ++r)
                    {
                        double acceleration = 0.0;
                        for (std::size_t b = 0; b < k; ++b)
                            acceleration += combination.coefficients()(b, firstCase + j) * accelerations[b][r];
                        check.supportAccelerations.push_back(acceleration);
                    }
                std::vector<output::StiffnessRecovery::UnitRow> externalRows; // bounded by the contours
                output::StiffnessRecovery::CondensedUnits externalUnits;
                if (externalPath)
                {
                    externalRows = externalPath->recordCase(s, state.displacements, state.unrecoveredReactions, budget);
                    externalUnits = {externalPath->replacedElements(), externalRows};
                    state.gridPointForceUnits = &externalUnits;
                }
                recoverCase(s, state, std::move(check), vectorised);
                if (progress != nullptr)
                    progress->advance();
            }
        }

        // 7. With the deck tolerance, the worst cases solved on their own and compared.
        for (const assembly::LoadBasisReport& report : loadCases.reports())
        {
            for (std::size_t w = 0; w < report.worstCases.size(); ++w)
            {
                const auto c = static_cast<std::size_t>(report.worstCases[w]);
                base::BudgetedBuffer<double> loads(budget, gSize, "deck check loads");
                base::BudgetedBuffer<double> enforced(budget, gSize, "deck check enforced displacements");
                loadCases.exactCase(c, loads.span(), enforced.span(), budget, sink);
                if (externalPath)
                    externalPath->applyToExactCase(loads.span(), budget);
                if (relief)
                    relief->relieve(loads.span());
                gm.reduceToN(loads.span());
                const base::BudgetedBuffer<double> kus = multiply(knn, enforced.span(), budget, "Knn us");
                linalg::DenseMatrix single(budget, equations, 1, "deck check right-hand side");
                for (std::size_t e = 0; e < equations; ++e)
                {
                    const auto dof = static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)));
                    single(e, 0) = loads[dof] - kus[dof];
                }
                // A free body on its conditioned supports: refined and brought to the gauge of its FreeBodySupport DOFs
                // as the basis vectors are.
                linalg::DenseMatrix singleLoads(budget, equations, isRefined ? 1 : 0, "deck check relieved loads");
                if (isRefined)
                    std::copy_n(&single(0, 0), equations, &singleLoads(0, 0));
                solver.solve(single.view(), cancellation);
                if (isRefined)
                {
                    const RelievedSystem relieved{&solver, &kgg, &gm, relief.get(), singleLoads.view()};
                    static_cast<void>(refineRelievedSolution(relieved, single.view(), budget, pool, cancellation));
                }
                base::BudgetedBuffer<double> alone(budget, gSize, "deck check displacements");
                for (std::size_t e = 0; e < equations; ++e)
                    alone[static_cast<std::size_t>(solver.dofOfEquation(static_cast<std::int32_t>(e)))] = single(e, 0);
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    if (enforced[dof] != 0.0)
                        alone[dof] = enforced[dof];
                gm.expandToG(alone.span());
                if (isRefined)
                    toSupportGauge(*relief, rigid.coefficients, rSet, alone.span());
                if (removesRigidMotion)
                    relief->removeRigidMotion(alone.span(), budget);
                double largest = 0.0;
                double deviation = 0.0;
                for (std::size_t dof = 0; dof < gSize; ++dof)
                {
                    double combinedValue = 0.0;
                    for (std::size_t b = 0; b < k; ++b)
                        combinedValue += combination.coefficients()(b, c) * basisResults[Displacements](dof, b);
                    largest = std::max(largest, std::abs(alone[dof]));
                    deviation = std::max(deviation, std::abs(combinedValue - alone[dof]));
                }
                results.deckDeviations.push_back({subcases[group[c]].label, report.family, report.worstResiduals[w],
                                                  largest > 0.0 ? deviation / largest : 0.0});
            }
        }
        if (m_options.keepsSystems)
        {
            FactorisedSetParts parts;
            parts.index = model.sharedIndex();
            parts.dofMap = model.sharedDofMap();
            parts.gm = system.sharedGm();
            parts.solveSets = system.sharedSolveSets();
            parts.pattern = system.sharedPattern();
            parts.kgg = system.sharedKgg();
            parts.solver = std::move(factorised);
            if (relief)
                parts.relief = keptReliefOf(*relief, kgg, system.isConstrained(), rSet, loadCases.coefficients(),
                                            accelerations, budget);
            parts.kernel = m_options.kernel;
            for (const std::size_t s : group)
                parts.subcases.push_back(subcases[s].label);
            results.systems.push_back(std::make_shared<const FactorisedSet>(std::move(parts)));
        }
        results.timings.recovery += watch.lap();
    }

    // The Interface stage ends with the interfaces handed over (no case, no SUBCOM, no table).
    if (isInterfaceStage)
    {
        condensed->finishCacheStores(sink, pool);
        results.condensed = condensed->record();
        results.interfaces = condensed->takeInterfaceExports();
        results.reducedCaseLabels = condensed->caseLabels();
        results.largestPivotRatio = std::max(results.largestPivotRatio, condensed->largestPivotRatio());
        results.diagnostics = sink.diagnostics();
        return results;
    }

    // 8. The SUBCOMs: the combinations of the results of the subcases they name.
    for (std::size_t s = 0; s < subcases.size(); ++s)
    {
        const StaticSubcase& subcase = subcases[s];
        if (!subcase.isCombination || results.loadBasis)
            continue;
        checkCancelled(cancellation);
        base::BudgetedBuffer<double> ug(budget, gSize, "SUBCOM displacements");
        base::BudgetedBuffer<double> applied(budget, gSize, "SUBCOM applied loads");
        base::BudgetedBuffer<double> spcForces(budget, gSize, "SUBCOM SPC forces");
        base::BudgetedBuffer<double> mpcForces(budget, gSize, "SUBCOM MPC forces");
        base::BudgetedBuffer<std::uint8_t> isConstrained(budget, gSize, "SUBCOM constrained DOFs");
        CaseState state;
        state.subcase = &subcase;
        state.plan = condensed ? &condensedPlans[s] : &plans[s];
        std::vector<output::StiffnessRecovery::UnitRow> unitRows;     // the condensed path: combined per point and unit
        std::vector<CondensedStatics::UnitBoundaryForces> unitForces; // the same: combined per unit
        if (snapshot.hasTable(input::TableRole::Loads))
            state.lineLoads.resize(elementSet.batches().size());
        for (std::size_t t = 0; t < subcase.combined.size(); ++t)
        {
            const double coefficient = subcase.coefficients[t];
            const auto found = std::find_if(stored.begin(), stored.end(),
                                            [&](const StoredCase& kept) { return kept.label == subcase.combined[t]; });
            if (found == stored.end())
                continue; // reported by staticSubcasesOf
            for (std::size_t dof = 0; dof < gSize; ++dof)
            {
                ug[dof] += coefficient * found->displacements[dof];
                applied[dof] += coefficient * found->applied[dof];
                spcForces[dof] += coefficient * found->spcForces[dof];
                mpcForces[dof] += coefficient * found->mpcForces[dof];
                isConstrained[dof] = static_cast<std::uint8_t>(isConstrained[dof] | found->isConstrained[dof]);
            }
            for (std::size_t k = 0; k < state.unrecoveredReactions.size(); ++k)
                state.unrecoveredReactions[k] += coefficient * found->unrecoveredReactions[k];
            for (const CondensedStatics::UnitBoundaryForces& unit : found->unitForces)
            {
                auto existing = std::find_if(unitForces.begin(), unitForces.end(),
                                             [&](const CondensedStatics::UnitBoundaryForces& entry)
                                             { return entry.unitId == unit.unitId; });
                if (existing == unitForces.end())
                {
                    unitForces.push_back({unit.unitId, unit.dofs, std::vector<double>(unit.forces.size(), 0.0),
                                          unit.hasInteriorReactions});
                    existing = unitForces.end() - 1;
                }
                for (std::size_t k = 0; k < unit.forces.size(); ++k)
                    existing->forces[k] += coefficient * unit.forces[k];
            }
            for (const output::StiffnessRecovery::UnitRow& row : found->unitRows)
            {
                auto existing =
                    std::find_if(unitRows.begin(), unitRows.end(), [&](const output::StiffnessRecovery::UnitRow& entry)
                                 { return entry.node == row.node && entry.unitId == row.unitId; });
                if (existing == unitRows.end())
                {
                    unitRows.push_back({row.node, row.unitId, {}});
                    existing = unitRows.end() - 1;
                }
                for (std::size_t c = 0; c < row.values.size(); ++c)
                    existing->values[c] += coefficient * row.values[c];
            }
            // The line loads of the term, scaled, for the station layout of the bars it loads.
            for (std::size_t term = 0; term < subcases.size(); ++term)
            {
                if (subcases[term].label != subcase.combined[t] || !subcases[term].load)
                    continue;
                std::vector<std::vector<elements::ElementLineLoad>> loads =
                    loadBuilder.lineLoadsOf(*subcases[term].load);
                for (std::size_t b = 0; b < loads.size(); ++b)
                    for (elements::ElementLineLoad load : loads[b])
                    {
                        load.p1 *= coefficient;
                        load.p2 *= coefficient;
                        state.lineLoads[b].push_back(load);
                    }
            }
        }
        for (std::vector<elements::ElementLineLoad>& loads : state.lineLoads)
            std::stable_sort(loads.begin(), loads.end(),
                             [](const elements::ElementLineLoad& left, const elements::ElementLineLoad& right)
                             { return left.elementRow < right.elementRow; });
        state.displacements = ug.span();
        state.applied = applied.span();
        state.spcForces = spcForces.span();
        bool hasMpcForces = false;
        for (std::size_t dof = 0; dof < gSize && !hasMpcForces; ++dof)
            hasMpcForces = mpcForces[dof] != 0.0;
        if (hasMpcForces)
            state.mpcForces = mpcForces.span();
        state.isConstrained = isConstrained.span();
        // The relations' forces of its terms, combined (each term's list times its coefficient): split rows when every
        // term has them, one row a point otherwise.
        if (hasMpcForces && isMpcSplit && state.plan->of(output::OutputKind::GridPointForces).isRequested)
        {
            std::size_t count = 0;
            bool isEveryTermSplit = true;
            for (std::size_t t = 0; t < subcase.combined.size(); ++t)
            {
                const auto found = std::find_if(stored.begin(), stored.end(), [&](const StoredCase& kept)
                                                { return kept.label == subcase.combined[t]; });
                if (found == stored.end())
                    continue;
                isEveryTermSplit = isEveryTermSplit && (found->mpcRelationForces.size() != 0 ||
                                                        std::all_of(found->mpcForces.data(),
                                                                    found->mpcForces.data() + found->mpcForces.size(),
                                                                    [](double value) { return value == 0.0; }));
                count += found->mpcRelationForces.size();
            }
            if (isEveryTermSplit)
            {
                state.isMpcSplit = true;
                state.mpcRelationForces =
                    base::BudgetedBuffer<assembly::RelationForce>(budget, count, "SUBCOM MPC forces by relation");
                std::size_t next = 0;
                for (std::size_t t = 0; t < subcase.combined.size(); ++t)
                {
                    const auto found = std::find_if(stored.begin(), stored.end(), [&](const StoredCase& kept)
                                                    { return kept.label == subcase.combined[t]; });
                    if (found == stored.end())
                        continue;
                    for (std::size_t k = 0; k < found->mpcRelationForces.size(); ++k)
                    {
                        assembly::RelationForce force = found->mpcRelationForces[k];
                        force.force *= subcase.coefficients[t];
                        state.mpcRelationForces[next++] = force;
                    }
                }
            }
        }
        std::stable_sort(unitRows.begin(), unitRows.end(),
                         [](const output::StiffnessRecovery::UnitRow& a, const output::StiffnessRecovery::UnitRow& b)
                         { return a.node < b.node; });
        output::StiffnessRecovery::CondensedUnits combinedUnits;
        if (condensed && condensed->hasUnrecoveredUnits())
        {
            combinedUnits = condensed->gridPointForceUnits(unitRows);
            state.gridPointForceUnits = &combinedUnits;
        }
        std::vector<output::StiffnessRecovery::UnitRow> externalRows; // bounded by the contours
        MergedUnitRows mergedCombination;
        if (externalPath)
        {
            std::vector<std::size_t> terms; // bounded by the subcases
            std::vector<double> termCoefficients;
            for (std::size_t t = 0; t < subcase.combined.size(); ++t)
                for (std::size_t term = 0; term < subcases.size(); ++term)
                    if (subcases[term].label == subcase.combined[t] && !subcases[term].isCombination)
                    {
                        terms.push_back(term);
                        termCoefficients.push_back(subcase.coefficients[t]);
                        break;
                    }
            externalPath->combineCase(s, terms, termCoefficients);
            externalRows = externalPath->recordCase(s, state.displacements, state.unrecoveredReactions, budget);
            if (state.gridPointForceUnits != nullptr)
            {
                // The condensed path: beside the rows of the condensed units not recovered.
                mergedCombination =
                    mergeUnitRows(state.gridPointForceUnits, externalPath->replacedElements(), externalRows);
                mergedCombination.view = {mergedCombination.excluded, mergedCombination.rows};
                combinedUnits = mergedCombination.view;
            }
            else
                combinedUnits = {externalPath->replacedElements(), externalRows};
            state.gridPointForceUnits = &combinedUnits;
        }
        output::SubcaseCheck check;
        check.isCombination = true;
        offerSectionCase(subcase, state.displacements);
        offerZoneCase(subcase, state.displacements, state.applied, state.spcForces, state.mpcForces, unitForces);
        recoverCase(s, state, std::move(check));
    }
    results.timings.recovery += watch.lap();

    for (SubcaseTables& tables : tablesOfSubcase)
    {
        for (output::NodalResultTable& table : tables.nodal)
            results.arrays.nodalTables.push_back(std::move(table));
        for (output::ElementResultTable& table : tables.element)
            results.arrays.elementTables.push_back(std::move(table));
        for (output::GridPointForceTable& table : tables.gridPointForces)
            results.arrays.gridPointForceTables.push_back(std::move(table));
        if (tables.check)
            results.checks.push_back(std::move(*tables.check));
        if (tables.monitorPoints3)
            results.monitorPoints3Loads.push_back(std::move(*tables.monitorPoints3));
    }
    results.monitorPoints3 = m_options.monitorPoints3;
    if (condensed)
    {
        condensed->finishCacheStores(sink, pool);
        results.condensed = condensed->record();
        for (const CondensedUnitRecord& unit : results.condensed->units)
            results.structureReuse.unitBoundariesReused += unit.isBoundaryDataReused ? 1 : 0;
        results.structureReuse.unitBoundariesKept += results.condensed->keptUnitBoundaryCount;
        results.condensedSystems = condensed->takeKeptSets();
        results.largestPivotRatio = std::max(results.largestPivotRatio, condensed->largestPivotRatio());
        results.solvedVectorCount += condensed->solvedVectorCount();
        // The factors the units still hold, kept for the later analyses of the process.
        if (m_options.sharedBudget.get() == &budget)
            condensed->offerInteriorFactors(m_options.sharedBudget);
        // (The load basis handed over reports the bases of its sets, the full path's: recordLoadBases.)
        for (const CondensedSetRecord& set : results.condensed->constraintSets)
            for (const assembly::LoadBasisReport& report : results.condensed->interfaceLoadBases)
            {
                if (results.loadBasis)
                    break;
                LoadBasisRecord record;
                record.constraintSet = set.constraintSet;
                record.report = report;
                results.loadBases.push_back(std::move(record));
            }
    }
    // The full path streamed's memory plan alone (stage "memory_plan") stops before any output: no envelope.
    if (!envelopes.isEmpty() && !(m_options.isStreamed && m_options.streamed.isMemoryPlanOnly))
        results.envelopes = envelopes.tables();
    if (sectionRun)
    {
        results.sectionLoads = sectionRun->takeSectionTable();
        results.cutLoads = sectionRun->takeCutTable();
    }
    if (zoneRun)
        results.zones = zoneRun->takeTables();
    if (externalPath)
        results.externalUnits = externalPath->takeRecords(budget);
    results.diagnostics = sink.diagnostics();
    return results;
}

} // namespace oos::solutions
