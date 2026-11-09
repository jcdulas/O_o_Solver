#include "oos/solutions/sol144/TrimSolution.hpp"

#include "TrimStructure.hpp"
#include "oos/aero/trim/TrimCorrections.hpp"

namespace oos::solutions
{

using namespace trim;

namespace
{

// The counter of a new stage of the run ("assembly", "aerodynamics", "condensation", "factorisation", "flexibility",
// "trim"), or null without a monitor.
base::ProgressCounter* beginStage(ProgressMonitor* monitor, const char* stage, std::uint64_t total)
{
    return monitor == nullptr ? nullptr : &monitor->beginStage(stage, total);
}

void advance(base::ProgressCounter* counter) noexcept
{
    if (counter != nullptr)
        counter->advance();
}

} // namespace

TrimResults TrimSolution::run(base::MemoryBudget& budget) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    const base::CancellationToken* cancellation = m_options.cancellation;
    base::ThreadPool* pool = m_options.pool;
    ProgressMonitor* monitor = m_options.monitor;
    advance(beginStage(monitor, "assembly", 1));
    TrimResults results;
    Stopwatch watch;
    base::DiagnosticSink sink;
    // The parts of the run, kept by a TrimLinearisation when the options ask for it, dropped at the return
    // otherwise.
    auto kept = std::make_unique<TrimLinearisation::State>();

    // 1. The support check, the elements, the subcases and their trim cases (every problem named in one pass); the DOF
    // map and the connectivity (AnalysisModel).
    std::vector<StaticSubcase>& subcases = kept->staticSubcases;
    std::vector<std::optional<aero::TrimCase>> trimCases;
    std::vector<output::OutputPlan>& plans = kept->outputPlans; // bounded by the subcases
    AnalysisModelOptions modelOptions;
    modelOptions.checkCases = [&](base::DiagnosticSink& caseSink)
    {
        subcases = staticSubcasesOf(snapshot, m_options.subcases, caseSink);
        for (const StaticSubcase& subcase : subcases)
        {
            trimCases.emplace_back();
            plans.push_back(output::OutputPlan::forSubcase(snapshot, subcase.tableId, caseSink));
            plans.back().addRequired(m_options.requiredOutputs);
            if (subcase.isCombination)
            {
                caseSink.report(
                    subcaseProblem(subcase.label, base::Severity::Error, "trim-subcase-combination",
                                   "a SubcaseCombination has no meaning in a static aeroelastic analysis (SOL 144); "
                                   "remove it"));
                continue;
            }
            // The applied loads of the subcase's LOAD: on the full path of an analysis; thermal loads, the
            // condensed path and the design derivatives of a trim do not take them yet.
            if (subcase.temperatureLoad)
                caseSink.report(
                    subcaseProblem(subcase.label, base::Severity::Error, "trim-applied-loads",
                                   "thermal loads in a trim subcase (TemperatureSelection(LOAD)) are not supported "
                                   "by this version; remove the selection"));
            // Case Control DEFORM: element deformations apply to SOLs 101, 105, 114 and 200 only (an error in the
            // other solution sequences).
            if (subcase.deformation)
                caseSink.report(
                    subcaseProblem(subcase.label, base::Severity::Error, "trim-applied-loads",
                                   "element deformations in a trim subcase (DeformationLoadSelection) are not "
                                   "applied in a static aeroelastic analysis "
                                   "(DeformationLoadSelection is for SOL 101, 105, 114 and 200 "
                                   "only); remove the DeformationLoadSelection selection"));
            if (subcase.load && m_options.condensation)
                caseSink.report(
                    subcaseProblem(subcase.label, base::Severity::Error, "trim-applied-loads",
                                   "applied loads (StaticLoadSelection) in a trim subcase are taken on the full path "
                                   "only in this version; solve without a condensation plan, or remove "
                                   "the selection"));
            if (subcase.load && m_options.keepsLinearisation)
                caseSink.report(subcaseProblem(subcase.label, base::Severity::Error, "trim-applied-loads",
                                               "the design derivatives of a trim subcase with applied loads "
                                               "(StaticLoadSelection) are not given by this version; remove "
                                               "the StaticLoadSelection selection of the "
                                               "SAERO subcase"));
            trimCases.back() =
                aero::TrimCase::forSubcase(snapshot, subcase.label, subcase.tableId, m_options.links, caseSink);
            if (!trimCases.back())
                caseSink.report(subcaseProblem(subcase.label, base::Severity::Error, "trim-missing",
                                               "a static aeroelastic subcase needs a TrimSelection selection in its "
                                               "Case Control (TrimSelection = n), found none"));
        }
    };
    // The volume of the requested structural outputs, before any element matrix is computed.
    modelOptions.checkBeforeConnectivity = [&](const assembly::ElementSet& elementSet, const assembly::DofMap& dofMap)
    {
        output::OutputVolumeEstimate estimate(elementSet, dofMap);
        const bool hasMset =
            snapshot.hasTable(input::TableRole::RigidElements) ||
            (snapshot.hasTable(input::TableRole::Spc) && snapshot.constraints().mpcReader().rowCount() > 0);
        for (const output::OutputPlan& plan : plans)
            estimate.addCase(plan, hasMset, [](const std::string&) { return true; });
        estimate.requireWithin(m_options.outputLimitBytes.value_or(
            static_cast<std::uint64_t>(OUTPUT_BUDGET_FRACTION * static_cast<double>(budget.available()))));
    };
    kept->model = std::make_unique<const AnalysisModel>(snapshot, AnalysisKind::StaticAeroelastic, budget, pool,
                                                        cancellation, sink, modelOptions);
    const AnalysisModel& model = *kept->model;
    const output::NodalResultBuilder tableBuilder(model.dofMap());
    const output::ElementResultBuilder elementTableBuilder(model.elementSet(), model.dofMap());
    const output::StiffnessRecovery stiffnessRecovery(model.elementSet(), model.dofMap(), pool);
    const assembly::ElementSet& elementSet = model.elementSet();
    const assembly::DofMap& dofMap = model.dofMap();
    const input::ParamTable* params = model.params();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    results.gSetSize = dofMap.gSetSize();
    const linalg::DenseKernels kernels(m_options.kernel, linalg::DenseKernelOptions{pool});

    // The aerodynamic model: the AeroStaticParams, the boxes, the trim variables, the splines and the hinge lines.
    advance(beginStage(monitor, "aerodynamics", 1));
    kept->reference = aero::AeroReference::read(snapshot);
    const aero::AeroReference& reference = kept->reference;
    kept->boxes = std::make_unique<const aero::AeroBoxes>(snapshot, reference, budget);
    const aero::AeroBoxes& boxes = *kept->boxes;
    const aero::TrimVariables variables(snapshot, reference, boxes);
    kept->splines = std::make_unique<const aero::SplineMatrix>(snapshot, reference, boxes, budget);
    const aero::SplineMatrix& splines = *kept->splines;
    const aero::HingeMoments hinges(boxes, variables);
    const std::size_t n = boxes.size();
    // The direct matrix input: DownwashMatrix's normalwash and AeroPressureMatrix's pressures, a fixed trim variable
    // INTERCEPT of value 1; nothing below is computed without them. Their design derivatives are the linearisation's as
    // built: w_0 and Δcp_e do not depend on the structure, so they drop out of the differentiated trim equations, taken
    // at the trimmed state, which holds their part.
    const aero::TrimCorrections corrections(snapshot, boxes, budget);
    const bool hasIntercept = !corrections.isEmpty();
    for (std::size_t j = 0; j < n; ++j)
        results.boxIds.push_back(boxes[j].id);
    for (std::size_t s = 0; s < hinges.surfaceCount(); ++s)
        results.surfaces.push_back(hinges.labelOf(s));
    results.surfaceDefinitions = variables.controlSurfaces();
    results.referenceChord = reference.chord;
    results.referenceSpan = reference.span;
    results.referenceArea = reference.area;
    results.symmetryXz = reference.symmetryXz;
    results.symmetryXy = reference.symmetryXy;
    // {X}_REF = Rᵀ·({X}_BAS − o), R the axes of RCSID (columns) and o its origin.
    results.basicToReference = reference.referenceFrame.axes.transposed();
    results.referenceOffset = -1.0 * (results.basicToReference * reference.referenceFrame.origin);

    // The monitor points: the caller's MonitorIntegratedLoad1, then its own: the whole vehicle about the RCSID origin
    // in its axes (structural: every grid point; aerodynamic: every box), and the hinge moment of each
    // AeroControlSurface (its boxes about the origin of CID1, the moment about its y axis).
    results.configuration = m_options.configuration;
    results.monitorPoints = m_options.monitorPoints;
    for (const bool isAerodynamic : {false, true})
    {
        output::MonitorPoint1Definition vehicle;
        vehicle.name = m_options.configuration;
        vehicle.label = "Full Vehicle Integrated Loads";
        vehicle.pointClass = "vehicleCoefficients";
        vehicle.isAerodynamic = isAerodynamic;
        vehicle.positionFrameId = reference.rigidBodyFrameId;
        vehicle.displacementFrameId = reference.rigidBodyFrameId;
        if (isAerodynamic)
            vehicle.ids = results.boxIds;
        else
            for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
                vehicle.ids.push_back(dofMap.gridId(row));
        results.monitorPoints.push_back(std::move(vehicle));
    }
    for (const aero::TrimVariables::ControlSurface& surface : variables.controlSurfaces())
    {
        output::MonitorPoint1Definition hinge;
        hinge.name = surface.label;
        hinge.label = surface.label + "   - Control Surface Hinge Moment";
        hinge.component = std::to_string(surface.aeroListId1 != 0 ? surface.aeroListId1 : surface.id);
        hinge.pointClass = "hingeMoment";
        hinge.axes = 5; // the moment about the hinge line, the y axis of CID1
        hinge.isAerodynamic = true;
        hinge.positionFrameId = surface.frameId1;
        hinge.displacementFrameId = surface.frameId1;
        for (const std::size_t j : surface.boxes)
            hinge.ids.push_back(boxes[j].id);
        results.monitorPoints.push_back(std::move(hinge));
    }
    std::vector<output::MonitorPoint1> monitors; // bounded by the monitor points
    for (const output::MonitorPoint1Definition& definition : results.monitorPoints)
        monitors.emplace_back(definition, snapshot, dofMap, reference, boxes);
    // The MonitorNodeForceSum3: the grid point forces of their elements and the loads XFLAG keeps at their grid points,
    // in the trimmed state (output::MonitorPoint3).
    results.monitorPoints3 = m_options.monitorPoints3;
    std::vector<output::MonitorPoint3> monitors3; // bounded by the monitor points
    for (const output::MonitorPoint3Definition& definition : results.monitorPoints3)
        monitors3.emplace_back(definition, snapshot, elementSet, dofMap);
    for (const std::optional<aero::TrimCase>& trimCase : trimCases)
        if (trimCase)
            checkVariables(*trimCase, variables, sink);

    // The structural results of a trimmed state (the tables its output plan asks for): the displacements, the applied
    // loads (the aerodynamic loads on the structure and the inertial loads, whose resultant is
    // zero in the supported directions), the SPC forces (K·u − P − the MPC forces on the s and r sets: the reactions of
    // the SPCs, zero at the supports to round-off), the MPC forces, the element results, the grid point forces and the
    // strain energy.
    const auto recoverStructure = [&](base::EntityId subcase, const output::OutputPlan& plan,
                                      std::span<const double> displacements, std::span<const double> applied,
                                      std::span<const double> supportAccelerations, const TrimStructure& structure,
                                      std::span<const std::vector<elements::ElementLineLoad>> lineLoads,
                                      base::MemoryBudget& recoveryBudget, output::ResultArrays& arrays,
                                      std::vector<std::array<double, 6>>& monitor3Loads)
    {
        base::BudgetedBuffer<double> elementForces(recoveryBudget, gSize, "trimmed K u");
        std::vector<output::StiffnessRecovery::UnitRow> unitRows; // the condensed path's units not recovered
        structure.elementForces(displacements, supportAccelerations, elementForces.span(), unitRows, recoveryBudget);
        base::BudgetedBuffer<double> mpcForces(recoveryBudget, gSize, "trimmed MPC forces");
        structure.gm().computeMpcForces(elementForces.span(), applied, mpcForces.span());
        base::BudgetedBuffer<double> spcForces(recoveryBudget, gSize, "trimmed SPC forces");
        const std::span<const std::uint8_t> isConstrained = structure.isConstrained();
        for (std::size_t dof = 0; dof < gSize; ++dof)
            spcForces[dof] = isConstrained[dof] != 0 ? elementForces[dof] - applied[dof] - mpcForces[dof] : 0.0;
        const bool hasMset = structure.gm().rowCount() > 0;
        // The MonitorNodeForceSum3 sum the rows appliedLoad, singlePointConstrForce and multiPointConstrForce of the
        // grid point forces with the element rows.
        const output::MonitorNodalLoads nodal{
            applied, spcForces.span(), hasMset ? std::span<const double>(mpcForces.span()) : std::span<const double>()};
        for (const output::MonitorPoint3& monitor : monitors3)
            monitor3Loads.push_back(monitor.loadsOf(elementSet, displacements, nodal));
        const auto addNodal = [&](output::OutputKind kind, std::span<const double> values)
        {
            const output::OutputRequestPlan& request = plan.of(kind);
            if (!request.isRequested)
                return;
            arrays.nodalTables.push_back(tableBuilder.build(std::string(output::tableKindOf(kind)), subcase, values,
                                                            recoveryBudget, request.isAll, request.members));
        };
        addNodal(output::OutputKind::Displacement, displacements);
        addNodal(output::OutputKind::AppliedLoads, applied);
        addNodal(output::OutputKind::SpcForces, spcForces.span());
        if (hasMset)
            addNodal(output::OutputKind::MpcForces, mpcForces.span());
        const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
        elementTableBuilder.build(subcase, displacements, lineLoads, recoveryBudget, arrays.elementTables, pool,
                                  cancellation, nullptr, &selection);
        if (plan.of(output::OutputKind::StrainEnergy).isRequested)
            stiffnessRecovery.strainEnergy(subcase, displacements, plan.of(output::OutputKind::StrainEnergy),
                                           recoveryBudget, arrays.elementTables, cancellation);
        if (plan.of(output::OutputKind::GridPointForces).isRequested)
        {
            output::StiffnessRecovery::NodalState nodal;
            nodal.displacements = displacements;
            nodal.appliedLoads = applied;
            nodal.spcForces = spcForces.span();
            nodal.mpcForces = mpcForces.span();
            nodal.isConstrained = isConstrained;
            const output::OutputRequestPlan& request = plan.of(output::OutputKind::GridPointForces);
            const std::optional<output::StiffnessRecovery::CondensedUnits> units = structure.condensedUnits(unitRows);
            arrays.gridPointForceTables.push_back(
                units ? stiffnessRecovery.gridPointForces(subcase, nodal, request, *units, recoveryBudget, cancellation)
                      : stiffnessRecovery.gridPointForces(subcase, nodal, request, recoveryBudget, cancellation));
        }
    };

    // The resultants of the boxes on the rigid configuration (unsplined): the force A_j n_j at the force point.
    const base::Vec3 origin = reference.referenceOriginInAeroFrame();
    linalg::DenseMatrix boxResultant(budget, 6, n, "box resultants");
    for (std::size_t j = 0; j < n; ++j)
    {
        const base::Vec3 force = boxes[j].area * boxes[j].normal;
        storeResultant({force, base::cross(boxes[j].forcePoint - origin, force)}, boxResultant, j);
    }
    kept->rigidModes =
        std::make_unique<const assembly::RigidBodyModes>(dofMap, reference.referenceFrame.origin, budget);
    const assembly::RigidBodyModes& rigidModes = *kept->rigidModes;

    // The constraint sets of the subcases and their supports: one assembly and factorisation per set.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    std::vector<std::vector<base::DofIndex>> supports(subcases.size());
    for (std::size_t s = 0; s < subcases.size(); ++s)
        supports[s] = assembly::supportDofsOf(snapshot, dofMap, subcases[s].tableId, sink);
    stopOnErrors(sink, "the trim cases");
    std::vector<std::vector<std::size_t>> groups; // bounded by the subcases
    for (std::size_t s = 0; s < subcases.size(); ++s)
    {
        bool isPlaced = false;
        for (std::vector<std::size_t>& group : groups)
        {
            const StaticSubcase& first = subcases[group.front()];
            if (first.spc == subcases[s].spc && first.mpc == subcases[s].mpc && first.contact == subcases[s].contact &&
                supports[group.front()] == supports[s])
            {
                group.push_back(s);
                isPlaced = true;
                break;
            }
        }
        if (!isPlaced)
            groups.push_back({s});
    }
    results.timings.setup += watch.lap();

    // The condensed path: the units of the plan condensed once for every constraint set, with
    // their six inertial patterns when a set has supports (the inertial loads of the trim are rigid-body accelerations,
    // CondensedStructure), the units whose outputs are asked for and those of the elements of a MonitorNodeForceSum3
    // recovered.
    kept->plan = m_options.condensation;
    std::unique_ptr<assembly::LoadVectorBuilder>& loadBuilder = kept->loadBuilder;
    if (!m_options.condensation &&
        std::any_of(subcases.begin(), subcases.end(), [](const StaticSubcase& subcase) { return subcase.load; }))
        loadBuilder = std::make_unique<assembly::LoadVectorBuilder>(elementSet, dofMap, budget);
    std::vector<assembly::SpcSet>& spcSets = kept->spcSets; // per subcase
    std::vector<std::size_t>& solved = kept->solved;        // bounded by the subcases
    std::unique_ptr<CondensedStatics>& condensed = kept->condensed;
    std::vector<output::OutputPlan> condensedPlans; // per subcase
    if (m_options.condensation)
    {
        loadBuilder = std::make_unique<assembly::LoadVectorBuilder>(elementSet, dofMap, budget);
        spcSets.reserve(subcases.size());
        for (std::size_t s = 0; s < subcases.size(); ++s)
        {
            spcSets.emplace_back(constraints, snapshot.findLoads(), dofMap, subcases[s].spc, subcases[s].load, budget,
                                 sink);
            solved.push_back(s);
        }
        stopOnErrors(sink, "the constraints");
        CondensedStatics::Inputs condensedInputs;
        condensedInputs.snapshot = &snapshot;
        condensedInputs.model = &model;
        condensedInputs.plan = m_options.condensation.get();
        condensedInputs.loads = loadBuilder.get();
        condensedInputs.subcases = subcases;
        condensedInputs.spcSets = spcSets;
        condensedInputs.solved = solved;
        condensedInputs.plans = plans;
        condensedInputs.recoveredUnits = m_options.recoveredUnits;
        for (const output::MonitorPoint3Definition& definition : m_options.monitorPoints3)
            condensedInputs.elementsToRecover.insert(condensedInputs.elementsToRecover.end(),
                                                     definition.elements.begin(), definition.elements.end());
        std::sort(condensedInputs.elementsToRecover.begin(), condensedInputs.elementsToRecover.end());
        condensedInputs.hasInertiaRelief = std::any_of(
            supports.begin(), supports.end(), [](const std::vector<base::DofIndex>& set) { return !set.empty(); });
        condensedInputs.kernels = &kernels;
        condensedInputs.pool = pool;
        condensedInputs.cancellation = cancellation;
        condensedInputs.cache = m_options.cache;
        condensedInputs.cacheModelTag = m_options.cacheModelTag;
        advance(beginStage(monitor, "condensation", 1));
        condensed = std::make_unique<CondensedStatics>(condensedInputs, budget, sink);
        for (const output::OutputPlan& plan : plans)
            condensedPlans.push_back(condensed->outputPlanOf(plan));
        results.path = AnalysisPath::Condensed;
        results.timings.assembly += watch.lap();
    }

    for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
    {
        const std::vector<std::size_t>& group = groups[groupIndex];
        checkCancelled(cancellation);
        advance(beginStage(monitor, "factorisation", 1));
        const StaticSubcase& first = subcases[group.front()];
        const std::vector<base::DofIndex>& supportDofs = supports[group.front()];
        if (supportDofs.size() > assembly::MAX_SUPPORT_DOFS)
            sink.report(subcaseProblem(
                first.label, base::Severity::Error, "trim-too-many-supports",
                base::concatenate({"the trim takes at most six FreeBodySupport DOFs (a statically "
                                   "determinate support), found ",
                                   std::to_string(supportDofs.size()), "; keep one DOF per rigid-body motion"})));
        kept->groups.push_back(std::make_unique<TrimGroup>());
        TrimGroup& keptGroup = *kept->groups.back();
        std::optional<assembly::SpcSet>& groupSpc = keptGroup.spc;
        if (!condensed)
            groupSpc.emplace(constraints, snapshot.findLoads(), dofMap, first.spc, std::nullopt, budget, sink);
        const assembly::SpcSet& spc = condensed ? spcSets[group.front()] : *groupSpc;
        if (!spc.enforcedDisplacements().empty())
            sink.report(subcaseProblem(
                first.label, base::Severity::Error, "trim-enforced",
                "enforced displacements (SinglePointConstr D) in a trim subcase are not supported by this "
                "version; remove them"));
        stopOnErrors(sink, "the assembly");

        const ConstraintSelection selection{first.label, first.spc, first.mpc, std::nullopt, first.contact};
        std::optional<ConstrainedSystem>& system = keptGroup.system;
        std::optional<linalg::BlockSparseMatrix>& mass = keptGroup.mass;
        std::optional<FsetSolver>& solver = keptGroup.solver;
        std::unique_ptr<TrimStructure>& structure = keptGroup.structure;
        if (condensed)
        {
            // The interface problem of the set with its supports held (CondensedStatics::prepareLoadGroup).
            CondensedStatics::GroupInputs groupInputs;
            groupInputs.constraintSet = static_cast<std::int32_t>(groupIndex);
            groupInputs.cases = group;
            groupInputs.spc = &spc;
            groupInputs.supports = supportDofs;
            groupInputs.selection = selection;
            groupInputs.pivotPolicy = pivotPolicyOf(params, first.label, sink);
            groupInputs.singularity = assembly::SingularityOptions::fromParams(params, first.label, sink);
            groupInputs.referencePoint = reference.referenceFrame.origin;
            CondensedStatics::LoadGroup loadGroup = condensed->prepareLoadGroup(groupInputs, budget, sink);
            if (!loadGroup.singularityTable.empty())
                results.singularityTables.push_back(loadGroup.singularityTable);
            results.largestLsetSize =
                std::max<std::int64_t>(results.largestLsetSize, loadGroup.interface->equationCount());
            results.timings.factorisation += watch.lap();
            structure = std::make_unique<CondensedStructure>(*condensed, std::move(loadGroup), supportDofs, rigidModes,
                                                             reference.referenceFrame.axes, budget);
        }
        else
        {
            ConstrainedSystemOptions systemOptions;
            systemOptions.spc = &spc;
            systemOptions.supportDofs = supportDofs;
            system.emplace(model, selection, budget, pool, cancellation, sink, systemOptions);
            if (!system->singularityTable().empty())
                results.singularityTables.push_back(system->singularityTable());
            stopOnErrors(sink, "the assembly");
            mass.emplace(system->pattern(), budget, "mass Mgg");
            const assembly::NonStructuralMassShares shares(elementSet);
            assembly::MassAssembler(elementSet, dofMap, shares,
                                    assembly::MassAssemblyOptions::fromParams(params, first.label, sink))
                .assemble(*mass, pool, cancellation);
            results.timings.assembly += watch.lap();

            FsetSolverOptions solverOptions;
            solverOptions.pivotPolicy = pivotPolicyOf(params, first.label, sink);
            solverOptions.cancellation = cancellation;
            solver.emplace(system->knn(), system->graph(), dofMap, system->solveSets(), kernels, solverOptions, budget,
                           pool);
            results.largestLsetSize = std::max<std::int64_t>(results.largestLsetSize, solver->equationCount());
            results.timings.factorisation += watch.lap();
            structure = std::make_unique<FullStructure>(*system, *solver, *mass, rigidModes,
                                                        reference.referenceFrame.axes, cancellation, budget);
        }

        advance(beginStage(monitor, "flexibility", 1));
        keptGroup.flexibility =
            flexibilityOf(*structure, dofMap, reference, boxes, splines, rigidModes, cancellation, budget);
        const BoxFlexibility& flexibility = keptGroup.flexibility;
        const std::size_t r = structure->supportCount();
        results.timings.flexibility += watch.lap();

        // E_R = E − (S·K⁻¹·M·D_r)·m_rr⁻¹·(D_rᵀ·T), the slopes of the relieved loads; m_rr⁻¹·(D_rᵀ·T) kept for the
        // inertia of the free body.
        linalg::DenseMatrix relievedSlopes;
        linalg::DenseMatrix& supportAccelerationsOfBoxes = keptGroup.supportAccelerationsOfBoxes; // r × n
        std::unique_ptr<linalg::DenseLu>& supportMass = keptGroup.supportMass;
        if (r > 0)
        {
            linalg::DenseMatrix copy(budget, r, r, "support mass factor");
            for (std::size_t j = 0; j < r; ++j)
                for (std::size_t i = 0; i < r; ++i)
                    copy(i, j) = flexibility.supportMass(i, j);
            supportMass = std::make_unique<linalg::DenseLu>(std::move(copy), budget);
            if (supportMass->isSingular())
                throw base::SolverError(base::concatenate({"subcase ", std::to_string(first.label),
                                                           ": the FreeBodySupport DOFs move no mass (D_rᵀ·M·D_r is "
                                                           "singular); support the rigid-body motions at "
                                                           "DOFs whose motion moves the mass of the model"}));
            supportAccelerationsOfBoxes = linalg::DenseMatrix(budget, r, n, "support accelerations of boxes");
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < r; ++k)
                    supportAccelerationsOfBoxes(k, j) = flexibility.supportLoads(k, j);
            supportMass->solve(supportAccelerationsOfBoxes.view());
            relievedSlopes = linalg::DenseMatrix(budget, n, n, "relieved slopes E_R");
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t i = 0; i < n; ++i)
                {
                    double correction = 0.0;
                    for (std::size_t k = 0; k < r; ++k)
                        correction += flexibility.inertialSlopes(i, k) * supportAccelerationsOfBoxes(k, j);
                    relievedSlopes(i, j) = flexibility.slopes(i, j) - correction;
                }
        }

        std::optional<double> factorMach;
        linalg::DenseMatrix normalwash; // D of the doublet lattice at factorMach
        base::ProgressCounter* trimmed = beginStage(monitor, "trim", group.size());
        for (const std::size_t s : group)
        {
            checkCancelled(cancellation);
            advance(trimmed);
            const aero::TrimCase& trimCase = *trimCases[s];
            const double q = trimCase.dynamicPressure;
            const double feedback = trimCase.aeqr;
            const std::size_t m = trimCase.settings.size();
            const double aunits =
                assembly::findRealParam(params, "AccelUnitsFactor", subcases[s].label, sink).value_or(1.0);
            if (aunits <= 0.0)
                throw base::SolverError(
                    base::concatenate({"PARAM AccelUnitsFactor: expected a positive real number (1/g in the "
                                       "units of the model), found ",
                                       std::to_string(aunits), "; set PARAM AccelUnitsFactor to 1/g, or remove it"}));
            if (!factorMach || *factorMach != trimCase.mach)
            {
                normalwash = aero::DoubletLatticeSteady::normalwashFactors(boxes, trimCase.mach, reference.symmetryXz,
                                                                           reference.symmetryXy, budget);
                factorMach = trimCase.mach;
            }

            // The normalwash of each variable and the support accelerations of each rigid acceleration (divided by
            // PARAM AUNITS, the accelerations of the TRIM in units of g).
            linalg::DenseMatrix rigid(budget, n, m, "rigid pressures");
            linalg::DenseMatrix accelerations(budget, r, m, "support accelerations of the variables");
            bool hasAcceleration = false;
            for (std::size_t x = 0; x < m; ++x)
            {
                const aero::TrimSetting& setting = trimCase.settings[x];
                if (const std::optional<std::size_t> axis = indexIn(ACCELERATION_LABELS, setting.label))
                {
                    for (std::size_t k = 0; k < r; ++k)
                        accelerations(k, x) = structure->referenceMotion()(k, *axis) / aunits;
                    hasAcceleration =
                        hasAcceleration || setting.state != aero::TrimVariableState::Fixed || setting.value != 0.0;
                    continue;
                }
                bool isAerodynamic = indexIn(AERODYNAMIC_LABELS, setting.label).has_value();
                for (const aero::TrimVariables::ControlSurface& surface : variables.controlSurfaces())
                    isAerodynamic = isAerodynamic || surface.label == setting.label;
                if (isAerodynamic)
                    variables.normalwashOf(setting.label, std::span<double>(&rigid(0, x), n));
            }
            if (r == 0 && hasAcceleration)
                sink.report(subcaseProblem(subcases[s].label, base::Severity::Warning, "trim-acceleration-unsupported",
                                           "rigid accelerations (RigidAccelX ...) without FreeBodySupport DOFs have no "
                                           "effect: the structure is "
                                           "held by its constraints (the inertial loads are those of "
                                           "the supported motions); give the weight of a restrained model as a "
                                           "GravityLoad in the subcase's StaticLoadSelection"));
            // The restrained right-hand sides w_x − AEQR·S·K⁻¹·M·D_r·ü_r(x) (f = −M·D_r·ü_r), the unrestrained ones w_x
            // (the inertia of the free body relieves the loads of an acceleration entirely).
            linalg::DenseMatrix restrained(budget, n, m, "restrained pressures");
            linalg::DenseMatrix unrestrained(budget, n, m, "unrestrained pressures");
            std::unique_ptr<linalg::DenseLu> restrainedFactor; // D − AEQR·q·E
            std::unique_ptr<linalg::DenseLu> freeFactor;       // D − AEQR·q·E_R
            for (std::size_t x = 0; x < m; ++x)
                for (std::size_t j = 0; j < n; ++j)
                {
                    double inertial = 0.0;
                    for (std::size_t k = 0; k < r; ++k)
                        inertial += flexibility.inertialSlopes(j, k) * accelerations(k, x);
                    restrained(j, x) = rigid(j, x) - feedback * inertial;
                    unrestrained(j, x) = rigid(j, x);
                }
            // The intercept's normalwash: w_0, and with AeroPressureMatrix the elastic normalwash of its loads
            // q·T·Δcp_e, AEQR·q·E·Δcp_e restrained and AEQR·q·E_R·Δcp_e free; its pressures are the solves plus Δcp_e.
            linalg::DenseMatrix rigidIntercept(budget, hasIntercept ? n : 0, 1, "rigid intercept pressures");
            linalg::DenseMatrix restrainedIntercept(budget, hasIntercept ? n : 0, 1, "restrained intercept pressures");
            linalg::DenseMatrix unrestrainedIntercept(budget, hasIntercept ? n : 0, 1,
                                                      "unrestrained intercept pressures");
            if (hasIntercept)
            {
                const std::span<const double> interceptNormalwash = corrections.normalwash();
                const std::span<const double> fixedPressures = corrections.pressures();
                for (std::size_t j = 0; j < n; ++j)
                {
                    double restrainedFeedback = 0.0;
                    double freeFeedback = 0.0;
                    for (std::size_t l = 0; l < n; ++l)
                    {
                        restrainedFeedback += flexibility.slopes(j, l) * fixedPressures[l];
                        if (r > 0)
                            freeFeedback += relievedSlopes(j, l) * fixedPressures[l];
                    }
                    rigidIntercept(j, 0) = interceptNormalwash[j];
                    restrainedIntercept(j, 0) = interceptNormalwash[j] + feedback * q * restrainedFeedback;
                    unrestrainedIntercept(j, 0) = r > 0 ? interceptNormalwash[j] + feedback * q * freeFeedback : 0.0;
                }
            }
            {
                const linalg::DenseLu rigidFactor(copyOf(normalwash, budget, "D"), budget);
                aero::solvePressures(rigidFactor, rigid.view());
                if (hasIntercept)
                    aero::solvePressures(rigidFactor, rigidIntercept.view());
                restrainedFactor =
                    pressureFactor(minusScaled(normalwash, feedback * q, flexibility.slopes, budget, "D - q E"),
                                   trimCase, "restrained", budget);
                restrainedFactor->solve(restrained.view());
                if (hasIntercept)
                    restrainedFactor->solve(restrainedIntercept.view());
                if (r > 0)
                {
                    freeFactor =
                        pressureFactor(minusScaled(normalwash, feedback * q, relievedSlopes, budget, "D - q E_R"),
                                       trimCase, "free", budget);
                    freeFactor->solve(unrestrained.view());
                    if (hasIntercept)
                        freeFactor->solve(unrestrainedIntercept.view());
                }
            }
            if (hasIntercept)
            {
                const std::span<const double> fixedPressures = corrections.pressures();
                for (std::size_t j = 0; j < n; ++j)
                {
                    rigidIntercept(j, 0) += fixedPressures[j];
                    restrainedIntercept(j, 0) += fixedPressures[j];
                    if (r > 0)
                        unrestrainedIntercept(j, 0) += fixedPressures[j];
                }
            }
            results.timings.aerodynamics += watch.lap();

            TrimSubcaseResult result;
            result.subcase = subcases[s].label;
            result.trimSet = trimCase.setId;
            result.mach = trimCase.mach;
            result.dynamicPressure = q;
            result.hasUnrestrained = r > 0;

            // The trim equations D_rᵀ·(q·T·Δcp + f) = 0: column x, q·(D_rᵀ·T)·Δcp_x − m_rr·ü_r(x).
            linalg::DenseMatrix equations(budget, r, m, "trim equations");
            for (std::size_t x = 0; x < m; ++x)
                for (std::size_t k = 0; k < r; ++k)
                {
                    double aerodynamic = 0.0;
                    for (std::size_t j = 0; j < n; ++j)
                        aerodynamic += flexibility.supportLoads(k, j) * restrained(j, x);
                    double inertial = 0.0;
                    for (std::size_t l = 0; l < r; ++l)
                        inertial += flexibility.supportMass(k, l) * accelerations(l, x);
                    equations(k, x) = q * aerodynamic - inertial;
                }
            // The applied loads P of the subcase's LOAD: the structure deformed by them adds the normalwash
            // AEQR·S·K_ll⁻¹·P, whose pressures Δcp_P add to those of the variables, and the trim equations take
            // D_rᵀ·(q·T·Δcp_P + P) as their constant part.
            base::BudgetedBuffer<double> applied(budget, subcases[s].load ? gSize : 0, "trim applied loads");
            std::vector<double> appliedPressures; // bounded by the boxes, empty without applied loads
            std::vector<double> constant;         // bounded by the supports
            std::vector<std::vector<elements::ElementLineLoad>> lineLoads; // bounded by the batches
            if (subcases[s].load)
            {
                std::fill(applied.data(), applied.data() + gSize, 0.0);
                loadBuilder->addLoads(*subcases[s].load, &*mass, applied.span(), budget, sink);
                stopOnErrors(sink, "the applied loads");
                lineLoads = loadBuilder->lineLoadsOf(*subcases[s].load);
                linalg::DenseMatrix appliedColumn(budget, gSize, 1, "trim applied load column");
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    appliedColumn(dof, 0) = applied[dof];
                const linalg::DenseMatrix noAccelerations(budget, r, 0, "no support accelerations");
                linalg::DenseMatrix appliedDisplacements(budget, gSize, 1, "trim applied load displacements");
                structure->solve(appliedColumn.view(), noAccelerations.view(), appliedDisplacements.view(), false,
                                 budget);
                linalg::DenseMatrix pressures(budget, n, 1, "trim applied load pressures");
                slopesOf(splines, dofMap, std::span<const double>(&appliedDisplacements(0, 0), gSize), pressures.view(),
                         0);
                for (std::size_t j = 0; j < n; ++j)
                    pressures(j, 0) *= feedback;
                restrainedFactor->solve(pressures.view());
                appliedPressures.assign(&pressures(0, 0), &pressures(0, 0) + n);
                constant.assign(r, 0.0);
                for (std::size_t k = 0; k < r; ++k)
                {
                    double aerodynamic = 0.0;
                    for (std::size_t j = 0; j < n; ++j)
                        aerodynamic += flexibility.supportLoads(k, j) * appliedPressures[j];
                    double work = 0.0;
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        work += structure->supportModes()(dof, k) * applied[dof];
                    constant[k] = q * aerodynamic + work;
                }
            }
            // The intercept's resultant on the supported motions, q·D_rᵀ·T·Δcp_0, in the constant part.
            if (hasIntercept && r > 0)
            {
                constant.resize(r, 0.0);
                for (std::size_t k = 0; k < r; ++k)
                {
                    double aerodynamic = 0.0;
                    for (std::size_t j = 0; j < n; ++j)
                        aerodynamic += flexibility.supportLoads(k, j) * restrainedIntercept(j, 0);
                    constant[k] += q * aerodynamic;
                }
            }
            const std::vector<double> values = aero::TrimSystem::solve(trimCase, equations, constant, budget);

            // The stability derivatives, per unit q (aero::ReferenceCoefficients); the intercept's row (the
            // reference row) from its pressures, the variable of value 1 with no acceleration.
            const auto derivativeRowOf =
                [&](std::span<const double> rigidColumn, std::span<const double> restrainedColumn,
                    std::span<const double> unrestrainedColumn, std::span<const double> restrainedAcceleration)
            {
                StabilityDerivativeRow row;
                row.rigidUnsplined = aero::ReferenceCoefficients::of(reference, combine(boxResultant, rigidColumn));
                row.rigidSplined =
                    aero::ReferenceCoefficients::of(reference, combine(flexibility.structuralResultant, rigidColumn));
                row.elasticRestrained = aero::ReferenceCoefficients::of(
                    reference, combine(flexibility.structuralResultant, restrainedColumn));
                if (r > 0)
                {
                    row.elasticUnrestrained = aero::ReferenceCoefficients::of(
                        reference, combine(flexibility.structuralResultant, unrestrainedColumn));
                    std::vector<double> relievedAcceleration(r, 0.0); // M·D_r·m_rr⁻¹·D_rᵀ·T·Δcp unrestrained
                    for (std::size_t k = 0; k < r; ++k)
                        for (std::size_t j = 0; j < n; ++j)
                            relievedAcceleration[k] += supportAccelerationsOfBoxes(k, j) * unrestrainedColumn[j];
                    row.inertialRestrained = aero::ReferenceCoefficients::of(
                        reference, scaled(combine(flexibility.inertialResultant, restrainedAcceleration), 1.0 / q));
                    row.inertialUnrestrained = aero::ReferenceCoefficients::of(
                        reference, combine(flexibility.inertialResultant, relievedAcceleration));
                }
                return row;
            };
            if (hasIntercept)
            {
                const std::vector<double> noAcceleration(r, 0.0);
                result.reference =
                    derivativeRowOf(std::span<const double>(&rigidIntercept(0, 0), n),
                                    std::span<const double>(&restrainedIntercept(0, 0), n),
                                    std::span<const double>(&unrestrainedIntercept(0, 0), n), noAcceleration);
            }
            for (std::size_t x = 0; x < m; ++x)
            {
                // The inertia of the supported motion: M·D_r·ü_r(x) / q restrained, M·D_r·m_rr⁻¹·D_rᵀ·T·Δcp
                // unrestrained.
                std::vector<double> restrainedAcceleration(r, 0.0);
                for (std::size_t k = 0; k < r; ++k)
                    restrainedAcceleration[k] = accelerations(k, x);
                StabilityDerivativeRow row = derivativeRowOf(
                    std::span<const double>(&rigid(0, x), n), std::span<const double>(&restrained(0, x), n),
                    std::span<const double>(&unrestrained(0, x), n), restrainedAcceleration);
                row.label = trimCase.settings[x].label;
                result.derivatives.push_back(std::move(row));
            }
            for (std::size_t surface = 0; surface < hinges.surfaceCount(); ++surface)
            {
                HingeMomentDerivatives hinge;
                hinge.surface = hinges.labelOf(surface);
                result.hingeMomentDerivatives.push_back(std::move(hinge));
            }
            if (hasIntercept)
            {
                // The reference row: the hinge moment coefficients of the intercept's pressures.
                const std::vector<double> rigidHinge =
                    hinges.coefficientsOf(std::span<const double>(&rigidIntercept(0, 0), n));
                const std::vector<double> restrainedHinge =
                    hinges.coefficientsOf(std::span<const double>(&restrainedIntercept(0, 0), n));
                const std::vector<double> unrestrainedHinge =
                    r > 0 ? hinges.coefficientsOf(std::span<const double>(&unrestrainedIntercept(0, 0), n))
                          : std::vector<double>(hinges.surfaceCount(), 0.0);
                for (std::size_t surface = 0; surface < hinges.surfaceCount(); ++surface)
                    result.hingeMomentDerivatives[surface].atReference = {rigidHinge[surface], restrainedHinge[surface],
                                                                          unrestrainedHinge[surface]};
            }
            for (std::size_t x = 0; x < m; ++x)
            {
                const std::vector<double> rigidHinge = hinges.coefficientsOf(std::span<const double>(&rigid(0, x), n));
                const std::vector<double> restrainedHinge =
                    hinges.coefficientsOf(std::span<const double>(&restrained(0, x), n));
                const std::vector<double> unrestrainedHinge =
                    r > 0 ? hinges.coefficientsOf(std::span<const double>(&unrestrained(0, x), n))
                          : std::vector<double>(hinges.surfaceCount(), 0.0);
                for (std::size_t surface = 0; surface < hinges.surfaceCount(); ++surface)
                    result.hingeMomentDerivatives[surface].perVariable.push_back(
                        {rigidHinge[surface], restrainedHinge[surface], unrestrainedHinge[surface]});
            }

            // The trimmed state: Δcp = Σ_x u_x·Δcp_x (restrained), ü_r = Σ_x u_x·ü_r(x), and the displacements
            // u_l = K_ll⁻¹·(q·T·Δcp − M·D_r·ü_r).
            for (std::size_t x = 0; x < m; ++x)
                result.variables.push_back({variables.idOf(trimCase.settings[x].label), trimCase.settings[x].label,
                                            trimCase.settings[x].kind, trimCase.settings[x].state, values[x]});
            result.printsPressures = plans[s].isAeroPressureRequested();
            result.printsForces = plans[s].isAeroForceRequested();
            result.pressures = appliedPressures;
            result.pressures.resize(n, 0.0);
            result.supportAccelerations.assign(r, 0.0);
            for (std::size_t x = 0; x < m; ++x)
            {
                for (std::size_t j = 0; j < n; ++j)
                    result.pressures[j] += restrained(j, x) * values[x];
                for (std::size_t k = 0; k < r; ++k)
                    result.supportAccelerations[k] += accelerations(k, x) * values[x];
            }
            if (hasIntercept)
                for (std::size_t j = 0; j < n; ++j)
                    result.pressures[j] += restrainedIntercept(j, 0);
            result.hingeMoments = hinges.momentsOf(result.pressures, q);
            {
                std::vector<double> boxForces(n, 0.0); // bounded by the boxes
                for (std::size_t j = 0; j < n; ++j)
                    boxForces[j] = q * result.pressures[j] * boxes[j].area;
                linalg::DenseMatrix boxLoads(budget, gSize, 1, "trimmed box loads");
                addPointLoads(splines.structuralLoads(boxForces), dofMap, std::span<double>(&boxLoads(0, 0), gSize));
                if (subcases[s].load)
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        boxLoads(dof, 0) += applied[dof];
                linalg::DenseMatrix supportAccelerations(budget, r, 1, "trimmed support accelerations");
                for (std::size_t k = 0; k < r; ++k)
                    supportAccelerations(k, 0) = result.supportAccelerations[k];
                linalg::DenseMatrix trimmed(budget, gSize, 1, "trimmed displacements of the solve");
                structure->solve(boxLoads.view(), supportAccelerations.view(), trimmed.view(), true, budget);
                result.displacements = base::BudgetedBuffer<double>(budget, gSize, "trimmed displacements");
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    result.displacements[dof] = trimmed(dof, 0);
                // The applied loads of the outputs, the box loads and P − M·D_r·ü_r.
                base::BudgetedBuffer<double> loads(budget, gSize, "trimmed loads");
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    loads[dof] = boxLoads(dof, 0);
                for (std::size_t k = 0; k < r; ++k)
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        loads[dof] -= structure->massModes()(dof, k) * result.supportAccelerations[k];
                recoverStructure(result.subcase, condensed ? condensedPlans[s] : plans[s], result.displacements.span(),
                                 loads.span(), result.supportAccelerations, *structure, lineLoads, budget,
                                 results.arrays, result.monitor3Loads);
            }
            // The forces of the boxes at their aerodynamic grid points, the box centers (AEROF): the normal force
            // q·Δcp·A and its moment about the spanwise axis of the box, F·(x_c − x_f) (positive for a force along
            // the normal ahead of the center).
            for (std::size_t j = 0; j < n; ++j)
            {
                const double force = q * result.pressures[j] * boxes[j].area;
                result.boxForces.push_back(force);
                result.boxMoments.push_back(force * base::dot(boxes[j].center - boxes[j].forcePoint, FLOW_AXIS));
            }
            if (!monitors.empty())
            {
                // The rigid air: the pressures of the rigid configuration at the trimmed variables; the elastic
                // restrained: the trimmed pressures; their loads on the structure through the splines; the inertia
                // M·D_r·ü_r (the inertial loads, the mass times the acceleration).
                std::vector<double> rigidForces(n, 0.0); // bounded by the boxes
                for (std::size_t x = 0; x < m; ++x)
                    for (std::size_t j = 0; j < n; ++j)
                        rigidForces[j] += q * rigid(j, x) * values[x] * boxes[j].area;
                if (hasIntercept)
                    for (std::size_t j = 0; j < n; ++j)
                        rigidForces[j] += q * rigidIntercept(j, 0) * boxes[j].area;
                const auto structuralLoadsOf = [&](std::span<const double> boxForces)
                {
                    base::BudgetedBuffer<double> loads(budget, gSize, "monitor point loads");
                    std::fill(loads.data(), loads.data() + gSize, 0.0);
                    addPointLoads(splines.structuralLoads(boxForces), dofMap, loads.span());
                    return loads;
                };
                const base::BudgetedBuffer<double> rigidLoads = structuralLoadsOf(rigidForces);
                const base::BudgetedBuffer<double> elasticLoads = structuralLoadsOf(result.boxForces);
                base::BudgetedBuffer<double> inertialLoads(budget, gSize, "monitor point inertia");
                std::fill(inertialLoads.data(), inertialLoads.data() + gSize, 0.0);
                for (std::size_t k = 0; k < r; ++k)
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        inertialLoads[dof] += structure->massModes()(dof, k) * result.supportAccelerations[k];
                for (const output::MonitorPoint1& monitor : monitors)
                {
                    std::array<std::array<double, 6>, output::MONITOR_CONTRIBUTION_COUNT> loads{};
                    if (monitor.definition().isAerodynamic)
                    {
                        loads[output::RigidAir] = monitor.loadsOfBoxes(rigidForces);
                        loads[output::ElasticRestrained] = monitor.loadsOfBoxes(result.boxForces);
                    }
                    else
                    {
                        loads[output::RigidAir] = monitor.loadsOfStructure(rigidLoads.span());
                        loads[output::ElasticRestrained] = monitor.loadsOfStructure(elasticLoads.span());
                        loads[output::Inertial] = monitor.loadsOfStructure(inertialLoads.span());
                        if (subcases[s].load)
                        {
                            // The applied loads (rigid and restrained: the same loads, not
                            // aerodynamic).
                            loads[output::RigidApplied] = monitor.loadsOfStructure(applied.span());
                            loads[output::RestrainedApplied] = loads[output::RigidApplied];
                        }
                    }
                    result.monitorLoads.push_back(loads);
                }
            }
            if (m_options.keepsLinearisation)
            {
                // What the derivatives of the subcase read (TrimLinearisation).
                TrimSubcaseState state;
                state.label = result.subcase;
                state.group = groupIndex;
                state.dynamicPressure = q;
                state.feedback = feedback;
                state.trimCase = trimCase;
                state.equations = std::move(equations);
                state.restrained = std::move(restrained);
                state.unrestrained = r > 0 ? std::move(unrestrained) : linalg::DenseMatrix();
                state.accelerations = std::move(accelerations);
                state.restrainedFactor = std::move(restrainedFactor);
                state.freeFactor = std::move(freeFactor);
                state.supportAccelerations = result.supportAccelerations;
                state.displacements.assign(result.displacements.data(), result.displacements.data() + gSize);
                state.supportMotion.assign(gSize, 0.0);
                for (std::size_t k = 0; k < r; ++k)
                    for (std::size_t dof = 0; dof < gSize; ++dof)
                        state.supportMotion[dof] += structure->supportModes()(dof, k) * result.supportAccelerations[k];
                kept->subcases.push_back(std::move(state));
            }
            results.subcases.push_back(std::move(result));
            results.timings.trim += watch.lap();
        }
    }
    if (condensed)
    {
        results.condensed = condensed->record();
    }
    results.diagnostics = sink.diagnostics();
    if (m_options.keepsLinearisation)
    {
        // The later solves on the calling thread (the run's pool and cancellation token may be gone).
        kept->kernels = std::make_unique<const linalg::DenseKernels>(m_options.kernel, linalg::DenseKernelOptions{});
        for (const std::unique_ptr<TrimGroup>& group : kept->groups)
            group->structure->rebind(*kept->kernels);
        results.linearisation = std::make_shared<TrimLinearisation>(std::move(kept));
    }
    return results;
}

std::vector<base::EntityId> trimRetainedPoints(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
{
    const aero::AeroReference reference = aero::AeroReference::read(snapshot);
    const aero::AeroBoxes boxes(snapshot, reference, budget);
    const aero::SplineMatrix splines(snapshot, reference, boxes, budget);
    std::vector<base::EntityId> points;
    for (const aero::SplineMatrix::Spline& spline : splines.splines())
        points.insert(points.end(), spline.nodes.begin(), spline.nodes.end());
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

} // namespace oos::solutions
