#include "oos/solutions/superelements/PartReduction.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/singularities/GridPointSingularities.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/solutions/common/CommonConstraints.hpp"
#include "oos/solutions/common/EigenMethod.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::solutions
{

namespace
{

base::Diagnostic partProblem(std::int32_t superelementId, std::string code, std::string message,
                             base::Severity severity = base::Severity::Error)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "BEGIN SUPER";
    diagnostic.id = superelementId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// A qualified neutral field of the eigenvalue method used by a part:
// `lanczosField` on a LanczosEigenMethod, `realField` on a RealEigenMethod.
std::string methodField(const EigenMethod& method, std::string_view lanczosField, std::string_view realField)
{
    const std::string_view field = method.card == "RealEigenMethod" ? realField : lanczosField;
    return base::concatenate({"", method.card, ".", field, ""});
}

std::string partName(std::int32_t superelementId)
{
    return base::concatenate({"part superelement ", std::to_string(superelementId)});
}

// The g-set DOF of a point's component (from 1; 0 for a scalar point), or nothing.
std::optional<base::DofIndex> dofOfPoint(const assembly::DofMap& dofMap, base::EntityId nodeId, int component)
{
    return dofMap.findDof(nodeId, component);
}

} // namespace

PartReduction::~PartReduction() = default;

PartReduction::PartReduction(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    if (inputs.part == nullptr || !inputs.part->snapshot || inputs.kernels == nullptr)
        throw std::invalid_argument("PartReduction: expected a part with its snapshot, and the kernels");
    const SuperelementPart& part = *inputs.part;
    m_superelementId = part.superelementId;
    m_kernels = inputs.kernels;
    m_cancellation = inputs.cancellation;
    m_model = std::make_unique<AnalysisModel>(*part.snapshot, inputs.analysis, budget, inputs.pool, inputs.cancellation,
                                              sink);
    selectSubcases(inputs, sink);
    buildConstraints(budget, sink);
    describe(part, sink);
    stopOnErrors(sink, base::concatenate({"the reduction of ", partName(m_superelementId)}));

    m_unitModel.elements = &m_model->elementSet();
    m_unitModel.dofMap = &m_model->dofMap();
    m_unitModel.elementIndex = m_elementIndex.get();
    m_unitModel.unitRelations = m_relations.get();
    m_unitModel.isConstrainedEverywhere = m_isHeld.span();
    m_unitModel.nonStructuralMass = m_shares.get();
    const input::ParamTable* params = m_model->params();
    m_unitModel.massOptions = assembly::MassAssemblyOptions::fromParams(params, std::nullopt, sink);
    m_unitModel.singularity = assembly::SingularityOptions::fromParams(params, std::nullopt, sink);
    m_unitModel.keepsMass = inputs.analysis == AnalysisKind::Modes;
    m_system = std::make_shared<const reduction::UnitSystem>(m_unitModel, m_description, budget, inputs.pool);
    checkBoundaryRoles(sink);
    stopOnErrors(sink, base::concatenate({"the reduction of ", partName(m_superelementId)}));

    // The partial factorisation of the part's interior (positive definite: the part held at its boundary), with the
    // MAXRATIO and BAILOUT of its own PARAMs.
    m_factorOptions.pivoting = linalg::PivotingMode::PositiveDefinite;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", std::nullopt, sink);
        ratio && *ratio > 0.0)
        m_factorOptions.pivotPolicy.maxRatio = *ratio;
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", std::nullopt, sink))
        m_factorOptions.pivotPolicy.bailout = *bailout;
    m_factorOptions.cancellation = inputs.cancellation;
    m_factorOptions.equationNamer = [system = m_system.get()](std::int32_t equation)
    {
        const reduction::UnitDofLabel label = system->labelOf(system->localDofOfEquation(equation));
        return base::GridComponent{label.id, label.component};
    };

    buildLoads(inputs, budget, sink);
    stopOnErrors(sink, base::concatenate({"the loads of ", partName(m_superelementId)}));
    const reduction::UnitPatterns patterns{m_patternLoads.view(), m_patternEnforced.view(), 0};
    m_condensation =
        std::make_unique<reduction::StaticCondensation>(m_system, patterns, *inputs.kernels, m_factorOptions, budget);
    m_noLoads = linalg::DenseMatrix(budget, m_system->boundaryLabels().size(), 0, "superelement no loads");
    if (inputs.analysis == AnalysisKind::Modes)
        reduceModes(inputs, budget, sink);
}

void PartReduction::selectSubcases(const Inputs& inputs, base::DiagnosticSink& sink)
{
    const std::vector<StaticSubcase> all = staticSubcasesOf(m_model->snapshot(), {}, sink);
    for (const base::EntityId label : inputs.subcases)
    {
        const auto found =
            std::find_if(all.begin(), all.end(), [&](const StaticSubcase& subcase) { return subcase.label == label; });
        if (found == all.end())
            throw std::invalid_argument(base::concatenate(
                {"PartReduction: ", partName(m_superelementId), " is given subcase ", std::to_string(label),
                 ", which the Case Control does not have; give subcases of the Case Control"}));
        if (found->isCombination)
        {
            sink.report(partProblem(
                m_superelementId, "superelement-subcase-combination",
                base::concatenate({"SubcaseCombination ", std::to_string(label), " applies to ",
                                   partName(m_superelementId),
                                   ": combinations of subcases are not supported in a deck with "
                                   "superelements by this version; remove the SubcaseCombination, or combine "
                                   "the results of its subcases afterwards"})));
            continue;
        }
        if (found->temperatureLoad)
            sink.report(
                partProblem(m_superelementId, "superelement-thermal",
                            base::concatenate({"subcase ", std::to_string(label),
                                               " has a thermal load (TemperatureSelection(LOAD)): thermal loads in a "
                                               "deck with superelements are not supported by this version; "
                                               "remove the thermal load"})));
        if (found->deformation)
            sink.report(partProblem(
                m_superelementId, "superelement-thermal",
                base::concatenate({"subcase ", std::to_string(label),
                                   " has element deformations (DeformationLoadSelection): deformation loads in a "
                                   "deck with superelements are not supported by this version; "
                                   "remove the DeformationLoadSelection selection"})));
        m_subcases.push_back(*found);
    }
}

void PartReduction::buildConstraints(base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const input::ModelSnapshot& snapshot = m_model->snapshot();
    const assembly::DofMap& dofMap = m_model->dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const CommonConstraints common = CommonConstraints::of(snapshot, dofMap, m_subcases, budget, sink);
    // One reduction serves every subcase of the part: its constraints must not vary between them.
    std::int64_t varying = 0;
    base::EntityId firstVarying = 0;
    m_isHeld = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "superelement held DOFs");
    for (std::size_t dof = 0; dof < gSize; ++dof)
    {
        const std::int32_t count = common.constrainedCount[dof];
        m_isHeld[dof] = common.isHeldEverywhere(static_cast<base::DofIndex>(dof)) ? 1 : 0;
        if (count > 0 && count < common.subcaseCount)
        {
            if (varying == 0)
                firstVarying = dofMap.pointIdOf(static_cast<base::DofIndex>(dof));
            ++varying;
        }
    }
    if (varying > 0)
        sink.report(partProblem(
            m_superelementId, "superelement-varying-constraints",
            base::concatenate({partName(m_superelementId), ": ", std::to_string(varying),
                               " DOFs are constrained in some of its subcases only (the first at point ",
                               std::to_string(firstVarying),
                               "): one reduction of a part serves every subcase in this version; select the same "
                               "SinglePointConstrSelection set in every subcase that applies to the part"})));
    if (!common.varyingMpcSets.empty())
        sink.report(
            partProblem(m_superelementId, "superelement-varying-mpc",
                        base::concatenate({partName(m_superelementId),
                                           ": its subcases select different MultiPointConstrSelection sets; select the "
                                           "same MultiPointConstrSelection set "
                                           "in every subcase that applies to the part"})));
    if (common.commonContact)
        sink.report(
            partProblem(m_superelementId, "superelement-contact",
                        base::concatenate({partName(m_superelementId),
                                           ": glued contact (ContactSelection) inside a part superelement is not "
                                           "supported by this version; glue the bodies in the residual "
                                           "structure, or merge the part into it"})));
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* piercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    m_mpc = std::make_unique<assembly::MpcEquations>(constraints, dofMap, common.commonMpcSet(), common.commonMpcCards,
                                                     budget, sink);
    m_fasteners = std::make_unique<assembly::FastenerConstraints>(piercings, dofMap, budget, sink);
    m_relations = std::make_unique<assembly::RigidElimination>(rigidElements, m_mpc.get(), dofMap, budget, sink,
                                                               m_fasteners.get());
    m_elementIndex = std::make_unique<reduction::UnitElementIndex>(m_model->elementSet(), budget);
    m_shares = std::make_unique<assembly::NonStructuralMassShares>(m_model->elementSet());
}

void PartReduction::describe(const SuperelementPart& part, base::DiagnosticSink& sink)
{
    const assembly::DofMap& dofMap = m_model->dofMap();
    // The q set: scalar points of the part that no element uses (their values are the generalised coordinates).
    m_modalPoints = part.modalPoints;
    std::sort(m_modalPoints.begin(), m_modalPoints.end());
    m_modalPoints.erase(std::unique(m_modalPoints.begin(), m_modalPoints.end()), m_modalPoints.end());
    std::vector<std::int32_t> modalNodes; // bounded by the q set
    for (const base::EntityId nodeId : m_modalPoints)
    {
        const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
        if (!point || point->kind != input::PointKind::Scalar)
        {
            sink.report(partProblem(
                m_superelementId, "superelement-q-set",
                base::concatenate({partName(m_superelementId), ": point ", std::to_string(nodeId),
                                   " of its q set (GeneralizedDofSet) is not a scalar point of the part; the "
                                   "generalised coordinates of a part are scalar points (ScalarPoint) "
                                   "of its Bulk Data"})));
            continue;
        }
        modalNodes.push_back(dofMap.gridCount() + point->row);
        m_modalDofs.push_back(dofMap.scalarDof(point->row));
    }
    std::sort(modalNodes.begin(), modalNodes.end());
    const linalg::NodeGroupList groups = m_model->connectivity().groups();
    for (std::size_t g = 0; g < groups.groupCount(); ++g)
        for (const std::int32_t node : groups.group(g))
            if (std::binary_search(modalNodes.begin(), modalNodes.end(), node))
            {
                sink.report(partProblem(m_superelementId, "superelement-q-set",
                                        base::concatenate({partName(m_superelementId), ": scalar point ",
                                                           std::to_string(dofMap.scalarId(node - dofMap.gridCount())),
                                                           " of its q set is used by an element; the generalised "
                                                           "coordinates must be free scalar points"})));
                return;
            }

    // The unit: every point of the part but the q set, every element; its boundary, the part's boundary points.
    m_description.unitId = part.superelementId;
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
        m_description.nodeIds.push_back(dofMap.gridId(row));
    for (std::int32_t row = 0; row < dofMap.scalarCount(); ++row)
        if (!std::binary_search(modalNodes.begin(), modalNodes.end(), dofMap.gridCount() + row))
            m_description.nodeIds.push_back(dofMap.scalarId(row));
    std::sort(m_description.nodeIds.begin(), m_description.nodeIds.end());
    for (const assembly::AssemblyBatch& batch : m_model->elementSet().batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            m_description.elementIds.push_back(batch.view.elementId(row));
    std::sort(m_description.elementIds.begin(), m_description.elementIds.end());
    for (const SuperelementBoundaryPoint& point : part.boundary)
    {
        if (std::binary_search(m_modalPoints.begin(), m_modalPoints.end(), point.nodeId))
            continue; // a generalised coordinate joined to the residual structure: not a boundary point
        if (!std::binary_search(m_description.nodeIds.begin(), m_description.nodeIds.end(), point.nodeId))
        {
            sink.report(partProblem(
                m_superelementId, "superelement-boundary",
                base::concatenate({partName(m_superelementId), ": boundary point ", std::to_string(point.nodeId),
                                   " is not a grid or scalar point of the part; define it in the "
                                   "part's Bulk Data"})));
            continue;
        }
        m_description.boundaryNodeIds.push_back(point.nodeId);
    }
    std::sort(m_description.boundaryNodeIds.begin(), m_description.boundaryNodeIds.end());
    m_description.boundaryNodeIds.erase(
        std::unique(m_description.boundaryNodeIds.begin(), m_description.boundaryNodeIds.end()),
        m_description.boundaryNodeIds.end());
}

void PartReduction::checkBoundaryRoles(base::DiagnosticSink& sink)
{
    const reduction::UnitSystem& system = *m_system;
    const assembly::DofMap& dofMap = m_model->dofMap();
    for (const base::EntityId nodeId : m_description.boundaryNodeIds)
    {
        const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
        const int components = point->kind == input::PointKind::Grid ? 6 : 1;
        for (int c = 1; c <= components; ++c)
        {
            const int component = point->kind == input::PointKind::Grid ? c : 0;
            const std::optional<base::DofIndex> dof = dofOfPoint(dofMap, nodeId, component);
            const std::optional<std::int64_t> local = dof ? system.localDofOf(*dof) : std::nullopt;
            if (!local)
                continue;
            const reduction::UnitDofRole role = system.roles()[static_cast<std::size_t>(*local)];
            if (role == reduction::UnitDofRole::Dependent)
                sink.report(partProblem(
                    m_superelementId, "superelement-dependent-boundary",
                    base::concatenate(
                        {partName(m_superelementId), ": component ", std::to_string(component), " of boundary point ",
                         std::to_string(nodeId),
                         " is made dependent by a relation of the part (an MultiPointConstr or a rigid element): a "
                         "boundary DOF cannot be in the m set; make that point the independent point of "
                         "the relation, or move the relation to the residual structure"})));
            else if (role == reduction::UnitDofRole::Constrained)
                m_heldBoundary.push_back(system.labelOf(*local));
        }
    }
}

void PartReduction::buildLoads(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const reduction::UnitSystem& system = *m_system;
    const auto localDofs = static_cast<std::size_t>(system.localDofCount());
    const assembly::DofMap& dofMap = m_model->dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    if (inputs.analysis != AnalysisKind::Statics || m_subcases.empty())
    {
        m_patternLoads = linalg::DenseMatrix(budget, localDofs, 0, "superelement pattern loads");
        m_patternEnforced = linalg::DenseMatrix(budget, localDofs, 0, "superelement pattern enforced");
        m_appliedLoads = linalg::DenseMatrix(budget, gSize, 0, "superelement applied loads");
        return;
    }
    const input::ModelSnapshot& snapshot = m_model->snapshot();
    const assembly::ElementSet& elementSet = m_model->elementSet();
    const assembly::LoadVectorBuilder loadBuilder(elementSet, dofMap, budget);
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    std::vector<assembly::SpcSet> spcSets; // bounded by the subcases
    spcSets.reserve(m_subcases.size());
    bool needsMass = false;
    for (const StaticSubcase& subcase : m_subcases)
    {
        spcSets.emplace_back(constraints, snapshot.findLoads(), dofMap, subcase.spc, subcase.load, budget, sink);
        needsMass = needsMass || (subcase.load && loadBuilder.needsMass(*subcase.load));
    }
    // The inertial loads are the part's Mgg times an acceleration field (LoadVectorBuilder).
    std::unique_ptr<linalg::NodeGraph> massGraph;
    std::unique_ptr<linalg::SparsePattern> massPattern;
    std::unique_ptr<linalg::BlockSparseMatrix> mass;
    if (needsMass)
    {
        const std::vector<linalg::NodeGroupList> groups{m_model->connectivity().groups()};
        massGraph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), groups, budget, inputs.pool);
        massPattern = std::make_unique<linalg::SparsePattern>(*massGraph, m_model->nodeDofCounts(), budget);
        mass = std::make_unique<linalg::BlockSparseMatrix>(*massPattern, budget, "superelement mass Mgg");
        assembly::MassAssembler(elementSet, dofMap, *m_shares, m_unitModel.massOptions)
            .assemble(*mass, inputs.pool, inputs.cancellation);
    }
    StaticLoadCases::Inputs caseInputs;
    caseInputs.snapshot = &snapshot;
    caseInputs.points = &m_model->index().points();
    caseInputs.dofMap = &dofMap;
    caseInputs.loads = &loadBuilder;
    for (std::size_t s = 0; s < m_subcases.size(); ++s)
    {
        caseInputs.cases.push_back(&m_subcases[s]);
        caseInputs.spcs.push_back(&spcSets[s]);
    }
    caseInputs.mass = mass.get();
    caseInputs.pool = inputs.pool;
    const StaticLoadCases cases(std::move(caseInputs), budget, sink);

    const std::size_t caseCount = m_subcases.size();
    m_patternLoads = linalg::DenseMatrix(budget, localDofs, caseCount, "superelement pattern loads");
    m_appliedLoads = linalg::DenseMatrix(budget, gSize, caseCount, "superelement applied loads");
    linalg::DenseMatrix enforcedLocal(budget, localDofs, caseCount, "superelement pattern enforced");
    base::BudgetedBuffer<double> loadsG(budget, gSize, "superelement loads");
    base::BudgetedBuffer<double> enforcedG(budget, gSize, "superelement enforced displacements");
    base::BudgetedBuffer<double> local(budget, localDofs, "superelement local loads");
    bool hasEnforced = false;
    for (std::size_t c = 0; c < caseCount; ++c)
    {
        cases.exactCase(c, loadsG.span(), enforcedG.span(), budget, sink);
        for (std::size_t dof = 0; dof < gSize; ++dof)
            m_appliedLoads(dof, c) = loadsG[dof];
        // Every local DOF takes its loads, the boundary's included (f_b of f*): a load on a boundary point of the part
        // is the part's load, reduced with it.
        system.gatherLocal(loadsG.span(), local.span());
        for (std::size_t dof = 0; dof < localDofs; ++dof)
            m_patternLoads(dof, c) = local[dof];
        system.gatherLocal(enforcedG.span(), local.span());
        for (std::size_t dof = 0; dof < localDofs; ++dof)
        {
            const double value = system.roles()[dof] == reduction::UnitDofRole::Constrained ? local[dof] : 0.0;
            enforcedLocal(dof, c) = value;
            hasEnforced = hasEnforced || value != 0.0;
        }
    }
    m_patternEnforced = hasEnforced ? std::move(enforcedLocal)
                                    : linalg::DenseMatrix(budget, localDofs, 0, "superelement pattern enforced");
}

void PartReduction::reduceModes(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const linalg::DenseKernels& kernels = *inputs.kernels;
    m_boundaryMass =
        reduction::CraigBampton::projectOnBoundary(*m_condensation, m_system->mass(), kernels, budget, m_cancellation);
    // No mode yet: the recovery of a part reduced statically (Guyan) is u_o = Ψ·u_b alone.
    m_modes = linalg::DenseMatrix(budget, static_cast<std::size_t>(m_system->interiorDofCount()), 0,
                                  "superelement fixed-interface modes");
    if (m_modalPoints.empty())
        return;

    // The method of the fixed-interface modes: the METHOD of the part's subcases, read in the part's Bulk Data.
    const input::ModelSnapshot& snapshot = m_model->snapshot();
    std::optional<std::int32_t> methodId;
    base::EntityId methodSubcase = 0;
    for (const StaticSubcase& subcase : m_subcases)
    {
        const std::optional<std::int32_t> selected = snapshot.hasTable(input::TableRole::Subcases)
                                                         ? snapshot.subcases().selectedSet(subcase.tableId, "method")
                                                         : std::nullopt;
        if (!selected)
            continue;
        if (methodId && *methodId != *selected)
        {
            sink.report(partProblem(
                m_superelementId, "superelement-method",
                base::concatenate({partName(m_superelementId), ": its subcases select EigenMethodSelection ",
                                   std::to_string(*methodId), " and ", std::to_string(*selected),
                                   "; the fixed-interface modes of a part come from one method: "
                                   "select one EigenMethodSelection in the subcases that apply to it"})));
            return;
        }
        methodId = selected;
        methodSubcase = subcase.label;
    }
    if (!methodId)
    {
        sink.report(partProblem(
            m_superelementId, "superelement-method",
            base::concatenate({partName(m_superelementId), " has a q set of ", std::to_string(m_modalPoints.size()),
                               " points but none of its subcases selects a EigenMethodSelection: add "
                               "EigenMethodSelection = "
                               "the ID of an LanczosEigenMethod of the part's Bulk Data to its subcase "
                               "(SuperelementSelection selects the subcase of a part)"})));
        stopOnErrors(sink, base::concatenate({"the modes of ", partName(m_superelementId)}));
        return;
    }
    const std::optional<EigenMethod> method =
        readEigenMethod(snapshot, *methodId, AnalysisKind::Modes, methodSubcase, sink);
    stopOnErrors(sink, base::concatenate({"the modes of ", partName(m_superelementId)}));
    const linalg::EigenRequest request = eigenRequestOf(*method, AnalysisKind::Modes);
    if (!request.startsFromLowest())
    {
        sink.report(partProblem(m_superelementId, "superelement-method",
                                base::concatenate({partName(m_superelementId), ": its method ", method->card, " ",
                                                   std::to_string(method->setId), " starts its range above 0 (",
                                                   methodField(*method, "lowerFrequency", "frequency1"),
                                                   "); the fixed-interface modes of a part are the lowest ones: leave ",
                                                   methodField(*method, "lowerFrequency", "frequency1"), " blank"})));
        stopOnErrors(sink, base::concatenate({"the modes of ", partName(m_superelementId)}));
    }
    // As many modes as the q set has points (its generalised coordinates) are kept, the lowest of the method's.
    const std::size_t qSize = m_modalPoints.size();
    reduction::ModeRequest modeRequest;
    if (request.modeCount > 0)
        modeRequest.count = static_cast<std::int32_t>(std::min(request.modeCount, qSize));
    else if (request.hasUpperBound())
        modeRequest.cutoffEigenvalue = request.upperBound;
    else
        modeRequest.count = static_cast<std::int32_t>(qSize);
    reduction::FixedInterfaceModes modes =
        reduction::CraigBampton::computeModes(*m_system, modeRequest, kernels, m_factorOptions, budget);
    std::size_t kept = modes.modes.columns();
    if (kept > qSize)
        kept = qSize;
    if (kept < qSize)
    {
        sink.report(partProblem(
            m_superelementId, "superelement-residual-vectors",
            base::concatenate({partName(m_superelementId), ": its q set has ", std::to_string(qSize),
                               " points but its method ", method->card, " ", std::to_string(method->setId), " gives ",
                               std::to_string(kept),
                               " fixed-interface modes; the other generalised coordinates would need residual "
                               "vectors, which this version does not compute: give the q set as many points as the "
                               "method gives modes (",
                               methodField(*method, "desiredModeCount", "desiredModeCount"), ")"})));
        stopOnErrors(sink, base::concatenate({"the modes of ", partName(m_superelementId)}));
    }
    const std::size_t interior = modes.modes.rows();
    m_modes = linalg::DenseMatrix(budget, interior, kept, "superelement fixed-interface modes");
    linalg::DenseMatrix eigenvalues(budget, kept, 1, "superelement fixed-interface eigenvalues");
    // The interior equations in the order of the part's DOFs (its points in the order of their IDs, then their
    // components), the order in which a tie is decided.
    std::vector<std::size_t> inDofOrder(interior); // bounded by the part's interior
    for (std::size_t e = 0; e < interior; ++e)
        inDofOrder[e] = e;
    std::sort(inDofOrder.begin(), inDofOrder.end(),
              [&](std::size_t a, std::size_t b)
              {
                  return m_system->localDofOfEquation(static_cast<std::int32_t>(a)) <
                         m_system->localDofOfEquation(static_cast<std::int32_t>(b));
              });
    // The first equation (in the DOF order) whose value is within FIXED_INTERFACE_TIE_TOLERANCE of the largest
    // magnitude of mode j; nothing for a zero mode.
    const auto peakOf = [&](std::size_t j) -> std::optional<std::size_t>
    {
        double largest = 0.0;
        for (const std::size_t e : inDofOrder)
            largest = std::max(largest, std::abs(modes.modes(e, j)));
        if (largest == 0.0)
            return std::nullopt;
        for (const std::size_t e : inDofOrder)
            if (std::abs(modes.modes(e, j)) >= (1.0 - FIXED_INTERFACE_TIE_TOLERANCE) * largest)
                return e;
        return std::nullopt;
    };
    for (std::size_t j = 0; j < kept; ++j)
    {
        // The normalisation of the part's method (EIGRL NORM). MASS keeps the unit generalised mass of the Lanczos
        // vectors and makes the component of largest magnitude positive (the first one on a near-tie, in the order of
        // the part's DOFs: the convention of scaleEigenvectors, decided whatever the order of the equations). The
        // format fixes no sign for a mode, so the generalised coordinates of the q set, which carry it, are defined up
        // to the sign of each mode; this rule makes them reproducible. MAX makes the component of largest magnitude +1;
        // POINT is taken as MAX (a warning).
        const bool isMass = method->normalization == EigenNormalization::Mass;
        const std::optional<std::size_t> peak = peakOf(j);
        double scale = 1.0;
        if (peak)
            scale = isMass ? (modes.modes(*peak, j) < 0.0 ? -1.0 : 1.0) : 1.0 / modes.modes(*peak, j);
        for (std::size_t i = 0; i < interior; ++i)
            m_modes(i, j) = scale * modes.modes(i, j);
        eigenvalues(j, 0) = modes.eigenvalues(j, 0);
    }
    if (method->normalization == EigenNormalization::Point)
        sink.report(partProblem(
            m_superelementId, "superelement-norm-point",
            base::concatenate({partName(m_superelementId), ": ", methodField(*method, "norm", "normalization"),
                               " = POINT of its method is taken as MAX for "
                               "its fixed-interface modes"}),
            base::Severity::Warning));
    m_modal = std::make_unique<reduction::ModalReduction>(
        reduction::CraigBampton::reduceModes(*m_condensation, m_modes, eigenvalues, kernels, budget, m_cancellation));
}

std::span<const reduction::UnitDofLabel> PartReduction::boundaryLabels() const noexcept
{
    return m_system->boundaryLabels();
}

const linalg::DenseMatrix& PartReduction::stiffness() const noexcept
{
    return m_condensation->reducedUnit().stiffness();
}

const linalg::DenseMatrix& PartReduction::loads() const noexcept
{
    return m_patternLoads.columns() > 0 ? m_condensation->reducedUnit().patterns() : m_noLoads;
}

std::int64_t PartReduction::interiorDofCount() const noexcept
{
    return m_system->interiorDofCount();
}

std::span<const double> PartReduction::appliedLoads(std::size_t s) const noexcept
{
    if (s >= m_appliedLoads.columns())
        return {};
    return {m_appliedLoads.data() + s * m_appliedLoads.leadingDimension(), m_appliedLoads.rows()};
}

linalg::DenseMatrix PartReduction::toGSet(linalg::ConstDenseMatrixView localValues,
                                          linalg::ConstDenseMatrixView modalValues, base::MemoryBudget& budget) const
{
    const reduction::UnitSystem& system = *m_system;
    const auto gSize = static_cast<std::size_t>(m_model->dofMap().gSetSize());
    linalg::DenseMatrix valuesG(budget, gSize, localValues.columns, "superelement displacements ug");
    for (std::size_t local = 0; local < localValues.rows; ++local)
    {
        const std::optional<base::DofIndex> dof = system.modelDofOf(static_cast<std::int64_t>(local));
        if (!dof)
            continue;
        for (std::size_t c = 0; c < localValues.columns; ++c)
            valuesG(static_cast<std::size_t>(*dof), c) = localValues(local, c);
    }
    for (std::size_t q = 0; q < m_modalDofs.size() && q < modalValues.rows; ++q)
        for (std::size_t c = 0; c < localValues.columns; ++c)
            valuesG(static_cast<std::size_t>(m_modalDofs[q]), c) = modalValues(q, c);
    return valuesG;
}

linalg::DenseMatrix PartReduction::recoverStatic(linalg::ConstDenseMatrixView boundaryValues,
                                                 std::span<const std::size_t> subcaseOfColumn,
                                                 base::MemoryBudget& budget) const
{
    const reduction::UnitSystem& system = *m_system;
    const std::size_t columns = subcaseOfColumn.size();
    if (boundaryValues.rows != system.boundaryLabels().size() || boundaryValues.columns != columns)
        throw std::invalid_argument(
            "PartReduction::recoverStatic: the boundary values do not match the part; give one row per boundary "
            "label of the part and one column per subcase");
    const auto localDofs = static_cast<std::size_t>(system.localDofCount());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    const bool hasEnforced = m_patternEnforced.columns() > 0;
    linalg::DenseMatrix loadsLocal(budget, localDofs, columns, "superelement recovery loads");
    linalg::DenseMatrix enforcedLocal(budget, localDofs, hasEnforced ? columns : 0, "superelement recovery enforced");
    for (std::size_t c = 0; c < columns; ++c)
    {
        const std::size_t s = subcaseOfColumn[c];
        if (s >= m_patternLoads.columns())
            throw std::invalid_argument(
                "PartReduction::recoverStatic: a subcase index out of range; give the index of a subcase of the part");
        for (std::size_t local = 0; local < localDofs; ++local)
        {
            loadsLocal(local, c) = m_patternLoads(local, s);
            if (hasEnforced)
                enforcedLocal(local, c) = m_patternEnforced(local, s);
        }
    }
    linalg::DenseMatrix equationValues(budget, equations, columns, "superelement recovery equations");
    system.equationLoads(loadsLocal.view(), enforcedLocal.view(), equationValues.view(), budget);
    m_condensation->recover(equationValues.view(), boundaryValues, *m_kernels, m_cancellation);
    linalg::DenseMatrix displacementsLocal(budget, localDofs, columns, "superelement recovery displacements");
    system.expandSolution(equationValues.view(), enforcedLocal.view(), displacementsLocal.view());
    const linalg::DenseMatrix noModal(budget, 0, columns, "superelement no modal values");
    return toGSet(displacementsLocal.view(), noModal.view(), budget);
}

linalg::DenseMatrix PartReduction::recoverModes(linalg::ConstDenseMatrixView boundaryValues,
                                                linalg::ConstDenseMatrixView modalValues,
                                                base::MemoryBudget& budget) const
{
    const reduction::UnitSystem& system = *m_system;
    const std::size_t columns = boundaryValues.columns;
    const std::size_t modeCount = m_modal ? m_modal->modeCount() : 0;
    if (boundaryValues.rows != system.boundaryLabels().size() || modalValues.rows != modeCount ||
        modalValues.columns != columns)
        throw std::invalid_argument(
            "PartReduction::recoverModes: the boundary or modal values do not match the part; give one row per "
            "boundary label and per mode of the part");
    const auto localDofs = static_cast<std::size_t>(system.localDofCount());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    linalg::DenseMatrix equationValues(budget, equations, columns, "superelement mode equations");
    reduction::CraigBampton::recover(*m_condensation, m_modes, boundaryValues, modalValues, equationValues.view(),
                                     *m_kernels, m_cancellation);
    linalg::DenseMatrix displacementsLocal(budget, localDofs, columns, "superelement mode displacements");
    const linalg::DenseMatrix noEnforced(budget, localDofs, 0, "superelement no enforced values");
    system.expandSolution(equationValues.view(), noEnforced.view(), displacementsLocal.view());
    return toGSet(displacementsLocal.view(), modalValues, budget);
}

} // namespace oos::solutions
