#include "oos/design/sensitivity/SensitivitySystem.hpp"

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/assemble/StiffnessAssembler.hpp"
#include "oos/assembly/constraints/MsetElimination.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/assembly/singularities/NodeDiagonalBlockSet.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::design
{

namespace
{

// The error of the problems of the model the system meets (the analysis of the same design point met them first).
void stopOnErrors(const base::DiagnosticSink& sink)
{
    if (!sink.hasErrors())
        return;
    std::string message = "the sensitivity system of the design point cannot be built:";
    for (const base::Diagnostic& diagnostic : sink.diagnostics())
        if (diagnostic.severity == base::Severity::Error)
            message.append("\n  ").append(base::formatDiagnostic(diagnostic));
    throw base::SolverError(message);
}

// PARAM MAXRATIO (default 1e7) and BAILOUT (default 0), as the static solution reads them.
linalg::PivotPolicy pivotPolicyOf(const input::ParamTable* params, std::optional<std::int32_t> subcase,
                                  base::DiagnosticSink& sink)
{
    linalg::PivotPolicy policy;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", subcase, sink);
        ratio && *ratio > 0.0)
        policy.maxRatio = *ratio;
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", subcase, sink))
        policy.bailout = *bailout;
    return policy;
}

// A value for a message: three significant digits (a residual of 3e-7 is not "0.000000").
std::string valueText(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.3g", value);
    return text.data();
}

// How the sensitivity system holds the inertia relief of its subcases.
enum class ReliefSource : std::uint8_t
{
    Kept,      // the full path's kept set, with its relief (or none)
    OwnFactor, // a factor made again by the system: no relief kept
    Condensed  // the condensed path
};

// The subcases whose sensitivities this version does not compute (OptimisationError naming the subcase).
void refuseUnsupported(const std::vector<solutions::StaticSubcase>& subcases, const input::ParamTable* params,
                       base::DiagnosticSink& sink, ReliefSource source, const solutions::KeptRelief* relief)
{
    if (subcases.empty())
        throw std::invalid_argument("SensitivitySystem: expected the subcases of one constraint set, found none");
    for (const solutions::StaticSubcase& subcase : subcases)
    {
        // A SUBCOM is solved on its terms' set (SensitivityComputation::prepareCombinations).
        if (subcase.isCombination)
            throw std::invalid_argument(
                base::concatenate({"SensitivitySystem: expected the SubcaseStarts of one constraint set, found "
                                   "SubcaseCombination ",
                                   std::to_string(subcase.label),
                                   " (a SubcaseCombination is solved on the set of its terms; a bug of O_o_Solver: "
                                   "report it)"}));
        const std::int32_t inertiaRelief =
            assembly::findIntegerParam(params, "InertiaRelief", subcase.label, sink).value_or(0);
        if (inertiaRelief == 0)
            continue;
        const std::string name = base::concatenate({"the sensitivities of subcase ", std::to_string(subcase.label)});
        if (inertiaRelief != -1)
            throw base::OptimisationError(base::concatenate(
                {name, " under PARAM InertiaRelief = ", std::to_string(inertiaRelief),
                 " are not computed by this version (PARAM InertiaRelief = -1 with FreeBodySupport or "
                 "FreeBodySupportSet only, "
                 "the "
                 "inertia relief the loads of the sizing loop write); give the supports with FreeBodySupport and "
                 "PARAM InertiaRelief = -1"}));
        if (relief == nullptr && source == ReliefSource::Condensed)
            throw base::OptimisationError(base::concatenate(
                {name,
                 " under inertia relief (PARAM InertiaRelief = -1) need the relief of the condensed analysis, kept "
                 "with its constraint set; evaluate the point with its factors kept (keep_analysis=True, the "
                 "default)"}));
        if (relief == nullptr)
            throw base::OptimisationError(base::concatenate(
                {name,
                 " under inertia relief (PARAM InertiaRelief = -1) need the relief of the analysis, kept with its "
                 "factor; evaluate the point with its factors kept (keep_analysis=True, the default)"}));
        if (relief->rigidResidual > solutions::KeptRelief::RIGID_SUPPORT_TOLERANCE)
            throw base::OptimisationError(base::concatenate(
                {name,
                 " under inertia relief: the motions of its supports are not rigid-body motions (their "
                 "reactions are ",
                 valueText(relief->rigidResidual),
                 " of the stiffness of a support), so that they depend on the design; hold the body by its "
                 "FreeBodySupport "
                 "DOFs alone (no SinglePointConstr, no grounded spring), a statically determinate support"}));
    }
}

const input::ParamTable* paramsOf(const input::ModelSnapshot& snapshot)
{
    return snapshot.hasTable(input::TableRole::Params) ? &snapshot.params() : nullptr;
}

// The relief of a condensed constraint set as the sensitivities read it (solutions::KeptRelief): D = Φ·C on
// the g set (Φ the rigid-body motions about the basic origin, as the condensed analysis fits C), M·D with the mass of
// the design point's elements (assembled once here: the condensed path keeps no Mgg; the options of the units' mass,
// the PARAMs of the bulk data), (Dᵀ·M·D)⁻¹ and the accelerations of the analysis. Rigid by construction (the analysis
// refused a misfit).
std::shared_ptr<const solutions::KeptRelief> condensedReliefOf(const solutions::CondensedFactorisedSet& set,
                                                               const assembly::ElementSet& elementSet,
                                                               const input::ParamTable* params,
                                                               base::MemoryBudget& budget, base::ThreadPool* pool)
{
    const solutions::CondensedRelief& relief = *set.relief();
    const assembly::DofMap& dofMap = set.dofMap();
    const auto g = static_cast<std::size_t>(dofMap.gSetSize());
    const std::size_t r = relief.supportCount;
    auto kept = std::make_shared<solutions::KeptRelief>();
    const assembly::RigidBodyModes rigid(dofMap, {0.0, 0.0, 0.0}, budget);
    const linalg::DenseMatrix& phi = rigid.modes();
    kept->supportModes = linalg::DenseMatrix(budget, g, r, "condensed relief D");
    for (std::size_t j = 0; j < r; ++j)
        for (std::size_t a = 0; a < assembly::RigidBodyModes::MODE_COUNT; ++a)
        {
            const double c = relief.fit[j * assembly::RigidBodyModes::MODE_COUNT + a];
            if (c == 0.0)
                continue;
            for (std::size_t dof = 0; dof < g; ++dof)
                kept->supportModes(dof, j) += phi(dof, a) * c;
        }
    const assembly::ElementConnectivity connectivity(elementSet, dofMap, budget, pool);
    const std::vector<linalg::NodeGroupList> groups{connectivity.groups()};
    const linalg::NodeGraph graph(dofMap.nodeCount(), groups, budget, pool);
    const base::BudgetedBuffer<std::uint8_t> counts = dofMap.nodeDofCounts(budget);
    const linalg::SparsePattern pattern(graph, counts.span(), budget);
    linalg::BlockSparseMatrix mass(pattern, budget, "condensed relief Mgg");
    base::DiagnosticSink ignored; // the analysis reported the problems of its parameters
    const assembly::NonStructuralMassShares shares(elementSet);
    assembly::MassAssembler(elementSet, dofMap, shares,
                            assembly::MassAssemblyOptions::fromParams(params, std::nullopt, ignored))
        .assemble(mass, pool);
    kept->massModes = linalg::DenseMatrix(budget, g, r, "condensed relief M D");
    base::BudgetedBuffer<double> column(budget, g, "condensed relief D column");
    base::BudgetedBuffer<double> product(budget, g, "condensed relief M D column");
    for (std::size_t j = 0; j < r; ++j)
    {
        for (std::size_t dof = 0; dof < g; ++dof)
            column[dof] = kept->supportModes(dof, j);
        mass.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < g; ++dof)
            kept->massModes(dof, j) = product[dof];
    }
    // (Dᵀ·M·D)⁻¹ by Gauss-Jordan.
    std::array<double, assembly::MAX_SUPPORT_DOFS * 2 * assembly::MAX_SUPPORT_DOFS> augmented{};
    const std::size_t width = 2 * r;
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t k = 0; k < r; ++k)
        {
            double sum = 0.0;
            for (std::size_t dof = 0; dof < g; ++dof)
                sum += kept->supportModes(dof, i) * kept->massModes(dof, k);
            augmented[i * width + k] = sum;
        }
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
    for (const std::vector<double>& acceleration : relief.accelerations)
    {
        std::array<double, assembly::MAX_SUPPORT_DOFS> values{};
        std::copy(acceleration.begin(), acceleration.end(), values.begin());
        kept->accelerations.push_back(values);
    }
    return kept;
}

} // namespace

SensitivitySystem::SensitivitySystem(std::shared_ptr<const solutions::FactorisedSet> kept,
                                     std::vector<solutions::StaticSubcase> subcases, linalg::KernelKind kernel,
                                     base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_budget(&budget), m_subcases(std::move(subcases)), m_kept(std::move(kept))
{
    if (!m_kept)
        throw std::invalid_argument("SensitivitySystem: expected a kept constraint set, found none");
    base::DiagnosticSink sink;
    refuseUnsupported(m_subcases, paramsOf(m_kept->index().snapshot()), sink, ReliefSource::Kept, m_kept->relief());
    for (const solutions::StaticSubcase& subcase : m_subcases)
        if (!m_kept->solves(subcase.label))
            throw std::invalid_argument(base::concatenate(
                {"SensitivitySystem: expected subcases solved on the kept constraint set, found subcase ",
                 std::to_string(subcase.label), ", which was not; build the system from the element set instead"}));
    m_dofMap = &m_kept->dofMap();
    m_kernels = std::make_unique<linalg::DenseKernels>(kernel, linalg::DenseKernelOptions{pool});
    m_gmView = &m_kept->gm();
    m_setsView = &m_kept->solveSets();
    m_kggView = &m_kept->kgg();
    m_patternView = &m_kept->pattern();
}

SensitivitySystem::SensitivitySystem(const assembly::ElementSet& elementSet, const assembly::DofMap& dofMap,
                                     std::vector<solutions::StaticSubcase> subcases, linalg::KernelKind kernel,
                                     base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_dofMap(&dofMap), m_budget(&budget), m_subcases(std::move(subcases))
{
    const input::ModelSnapshot& snapshot = elementSet.index().snapshot();
    const input::ParamTable* params = paramsOf(snapshot);
    base::DiagnosticSink sink;
    refuseUnsupported(m_subcases, params, sink, ReliefSource::OwnFactor, nullptr);
    const solutions::StaticSubcase& first = m_subcases.front();

    // Gm and the sets, as StaticSolution step 4 makes them.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* fastenerPiercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    m_kernels = std::make_unique<linalg::DenseKernels>(kernel, linalg::DenseKernelOptions{pool});
    m_mpc = std::make_unique<assembly::MpcEquations>(constraints, dofMap, first.mpc, budget, sink);
    m_fasteners = std::make_unique<assembly::FastenerConstraints>(fastenerPiercings, dofMap, budget, sink);
    m_spc = std::make_unique<assembly::SpcSet>(constraints, snapshot.findLoads(), dofMap, first.spc, first.load, budget,
                                               sink);
    // The glued contact of the subcases (StaticSolution groups them by BCONTACT; the design subcases share one).
    const assembly::GlueContext glueContext{rigidElements, m_mpc.get(), m_fasteners.get(), m_spc->sbDofs(),
                                            m_spc->sgDofs()};
    m_glue = std::make_unique<assembly::GlueConstraints>(snapshot, first.contact, first.label, dofMap, glueContext,
                                                         budget, sink);
    m_gm = std::make_unique<assembly::RigidElimination>(rigidElements, m_mpc.get(), dofMap, budget, sink,
                                                        m_fasteners.get(), m_glue.get());
    // FreeBodySupport without inertia relief: the r set is held as constraints (StaticSolution's
    // "free-body-support-as-constraint").
    const std::vector<base::DofIndex> rSet = assembly::supportDofsOf(snapshot, dofMap, first.tableId, sink);
    const assembly::DofSets sets(
        dofMap, assembly::DofSetInputs{m_gm->mDofs(), m_spc->sbDofs(), m_spc->sgDofs(), {}, rSet}, budget, sink);
    stopOnErrors(sink);

    m_connectivity = std::make_unique<assembly::ElementConnectivity>(elementSet, dofMap, budget, pool);
    m_couplings = std::make_unique<assembly::MsetCouplings>(m_connectivity->groups(), *m_gm, dofMap, budget);
    const std::vector<linalg::NodeGroupList> groupLists{m_connectivity->groups(), m_couplings->groups()};
    m_graph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), groupLists, budget, pool);
    const base::BudgetedBuffer<std::uint8_t> nodeDofCounts = dofMap.nodeDofCounts(budget);
    m_pattern = std::make_unique<linalg::SparsePattern>(*m_graph, nodeDofCounts.span(), budget);
    m_kgg = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "sensitivity stiffness Kgg");
    assembly::StiffnessAssembler(elementSet, dofMap).assemble(*m_kgg, pool, nullptr);
    m_knn = std::make_unique<linalg::BlockSparseMatrix>(*m_pattern, budget, "sensitivity stiffness Knn");
    assembly::MsetElimination::reduce(*m_kgg, *m_graph, *m_gm, dofMap, *m_knn);
    const assembly::NodeDiagonalBlockSet diagonalBlocks(*m_knn, dofMap, budget);
    const assembly::GridPointSingularities singularities(
        dofMap, sets, diagonalBlocks.blocks(), assembly::SingularityOptions::fromParams(params, first.label, sink),
        budget, sink);
    m_solveSets = std::make_unique<assembly::DofSets>(sets.withAutoSpc(singularities.autoSpcDofs()));
    stopOnErrors(sink);

    solutions::FsetSolverOptions options;
    options.pivotPolicy = pivotPolicyOf(params, first.label, sink);
    m_solver = std::make_unique<solutions::FsetSolver>(*m_knn, *m_graph, dofMap, *m_solveSets, *m_kernels, options,
                                                       budget, pool);
    m_gmView = m_gm.get();
    m_setsView = m_solveSets.get();
    m_kggView = m_kgg.get();
    m_patternView = m_pattern.get();
}

SensitivitySystem::SensitivitySystem(std::shared_ptr<const solutions::CondensedFactorisedSet> kept,
                                     const assembly::ElementSet& elementSet,
                                     std::vector<solutions::StaticSubcase> subcases, linalg::KernelKind kernel,
                                     base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_budget(&budget),
      m_subcases(std::move(subcases)),
      m_condensedSet(std::move(kept)),
      m_elementSet(&elementSet),
      m_pool(pool)
{
    if (!m_condensedSet)
        throw std::invalid_argument("SensitivitySystem: expected a kept condensed constraint set, found none");
    base::DiagnosticSink sink;
    if (m_condensedSet->relief() != nullptr)
        m_condensedRelief =
            condensedReliefOf(*m_condensedSet, elementSet, paramsOf(elementSet.index().snapshot()), budget, pool);
    refuseUnsupported(m_subcases, paramsOf(elementSet.index().snapshot()), sink, ReliefSource::Condensed,
                      m_condensedRelief.get());
    for (const solutions::StaticSubcase& subcase : m_subcases)
        if (!m_condensedSet->solves(subcase.label))
            throw std::invalid_argument(base::concatenate(
                {"SensitivitySystem: expected subcases solved on the kept condensed constraint set, found subcase ",
                 std::to_string(subcase.label), ", which was not"}));
    m_dofMap = &m_condensedSet->dofMap();
    m_kernels = std::make_unique<linalg::DenseKernels>(kernel, linalg::DenseKernelOptions{pool});
    m_gmView = &m_condensedSet->gm();
    m_condensed = std::make_unique<BoundaryReducedSensitivity>(m_condensedSet, kernel, budget, pool);
}

SensitivitySystem::~SensitivitySystem() = default;

void SensitivitySystem::assembleStiffness() const
{
    if (m_lazyKgg)
        return;
    const assembly::DofMap& dofMap = *m_dofMap;
    base::MemoryBudget& budget = *m_budget;
    m_lazyConnectivity = std::make_unique<assembly::ElementConnectivity>(*m_elementSet, dofMap, budget, m_pool);
    m_lazyCouplings =
        std::make_unique<assembly::MsetCouplings>(m_lazyConnectivity->groups(), *m_gmView, dofMap, budget);
    // The contour points of each external unit coupled (its K* is a clique), as the static solution couples them.
    std::vector<std::int64_t> offsets{0}; // bounded by the external units
    std::vector<std::int32_t> nodes;      // bounded by their contours
    for (const std::shared_ptr<const reduction::ExternalUnitAssembly>& placement : m_condensedSet->externalUnits())
    {
        const std::vector<std::int32_t> contour = placement->contourNodes(dofMap);
        nodes.insert(nodes.end(), contour.begin(), contour.end());
        offsets.push_back(static_cast<std::int64_t>(nodes.size()));
    }
    std::vector<linalg::NodeGroupList> lists{m_lazyConnectivity->groups(), m_lazyCouplings->groups()};
    if (!nodes.empty())
        lists.push_back(linalg::NodeGroupList{offsets, nodes});
    m_lazyGraph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), lists, budget, m_pool);
    const base::BudgetedBuffer<std::uint8_t> nodeDofCounts = dofMap.nodeDofCounts(budget);
    m_lazyPattern = std::make_unique<linalg::SparsePattern>(*m_lazyGraph, nodeDofCounts.span(), budget);
    m_lazyKgg =
        std::make_unique<linalg::BlockSparseMatrix>(*m_lazyPattern, budget, "sensitivity stiffness Kgg (condensed)");
    assembly::StiffnessAssembler(*m_elementSet, dofMap).assemble(*m_lazyKgg, m_pool, nullptr);
    for (const std::shared_ptr<const reduction::ExternalUnitAssembly>& placement : m_condensedSet->externalUnits())
        placement->addStiffnessTo(*m_lazyKgg, dofMap);
}

const linalg::BlockSparseMatrix& SensitivitySystem::kgg() const
{
    if (!m_condensed)
        return *m_kggView;
    assembleStiffness();
    return *m_lazyKgg;
}

const linalg::SparsePattern& SensitivitySystem::pattern() const
{
    if (!m_condensed)
        return *m_patternView;
    assembleStiffness();
    return *m_lazyPattern;
}

bool SensitivitySystem::isConstrained(base::DofIndex dof) const noexcept
{
    if (m_condensed)
        return m_condensedSet->isConstrained()[static_cast<std::size_t>(dof)] != 0;
    return m_setsView->contains(assembly::DofSet::S, dof) ||
           m_setsView->exclusiveSetOf(dof) == assembly::ExclusiveSet::R;
}

void SensitivitySystem::toEquations(std::span<const double> vectorG, std::span<double> equations) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (vectorG.size() != gSize || equations.size() != static_cast<std::size_t>(equationCount()))
        throw std::invalid_argument("SensitivitySystem::toEquations: expected a g-set vector and the equations");
    if (m_condensed)
    {
        // The condensed solve applies Tᵀ itself (the units' and the interface's relations).
        std::copy(vectorG.begin(), vectorG.end(), equations.begin());
        return;
    }
    base::BudgetedBuffer<double> reduced(*m_budget, gSize, "sensitivity reduced vector");
    std::copy(vectorG.begin(), vectorG.end(), reduced.data());
    m_gmView->reduceToN(reduced.span());
    for (std::size_t e = 0; e < equations.size(); ++e)
        equations[e] = reduced[static_cast<std::size_t>(dofOfEquation(static_cast<std::int32_t>(e)))];
}

void SensitivitySystem::toEquationsInPlace(std::span<double> vectorG, std::span<double> equations) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (vectorG.size() != gSize || equations.size() != static_cast<std::size_t>(equationCount()))
        throw std::invalid_argument("SensitivitySystem::toEquationsInPlace: expected a g-set vector and the equations");
    if (m_condensed)
    {
        std::copy(vectorG.begin(), vectorG.end(), equations.begin());
        return;
    }
    m_gmView->reduceToN(vectorG);
    for (std::size_t e = 0; e < equations.size(); ++e)
        equations[e] = vectorG[static_cast<std::size_t>(dofOfEquation(static_cast<std::int32_t>(e)))];
}

void SensitivitySystem::fromEquations(std::span<const double> equations, std::span<double> vectorG) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (vectorG.size() != gSize || equations.size() != static_cast<std::size_t>(equationCount()))
        throw std::invalid_argument("SensitivitySystem::fromEquations: expected the equations and a g-set vector");
    if (m_condensed)
    {
        std::copy(equations.begin(), equations.end(), vectorG.begin());
        return;
    }
    std::fill(vectorG.begin(), vectorG.end(), 0.0);
    for (std::size_t e = 0; e < equations.size(); ++e)
        vectorG[static_cast<std::size_t>(dofOfEquation(static_cast<std::int32_t>(e)))] = equations[e];
    m_gmView->expandToG(vectorG);
}

void SensitivitySystem::solve(linalg::DenseMatrixView rightHandSides, std::span<const std::uint8_t> isReadDof) const
{
    if (m_condensed)
        m_condensed->solve(rightHandSides, isReadDof);
    else if (m_kept)
        m_kept->solve(rightHandSides, *m_kernels);
    else
        m_solver->solve(rightHandSides);
    m_solveCount += static_cast<std::int64_t>(rightHandSides.columns);
}

} // namespace oos::design
