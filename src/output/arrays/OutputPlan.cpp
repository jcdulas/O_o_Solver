#include "oos/output/arrays/OutputPlan.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::output
{

namespace
{

// The request of each kind by the neutral name O_o_Reader gives the command in outputRequestsTable (it turns the
// keyword, its aliases and its abbreviations into this name: DISPLACEMENT, VECTOR and DISP give DisplacementOutput).
struct RequestName
{
    OutputKind kind;
    std::string_view name;
};
constexpr std::array<RequestName, 9> REQUEST_NAMES{{{OutputKind::Displacement, "DisplacementOutput"},
                                                    {OutputKind::SpcForces, "SinglePointConstrForceOutput"},
                                                    {OutputKind::MpcForces, "MultiPointConstrForceOutput"},
                                                    {OutputKind::AppliedLoads, "AppliedLoadOutput"},
                                                    {OutputKind::Stress, "StressOutput"},
                                                    {OutputKind::Strain, "StrainOutput"},
                                                    {OutputKind::Force, "ElemForceOutput"},
                                                    {OutputKind::GridPointForces, "NodeForceOutput"},
                                                    {OutputKind::StrainEnergy, "ElemStrainEnergyOutput"}}};
// The describers (by their neutral names) with no effect on the results of a linear static run: the place of the
// output (PRINT, PLOT: the results hold every requested table), the default sort (SORT1), and the options of the
// complex, frequency and random outputs (REAL, IMAG, PHASE, PSDF, ATOC, CRMS, RMS, RALL, RPRINT, NORPRINT, RPUNCH;
// ABS and REL of enforced motion in dynamics), which mean nothing for a static solution.
constexpr std::array<std::string_view, 16> NO_EFFECT_DESCRIBERS{"PrintListing",
                                                                "PlotOnly",
                                                                "SortByPoint",
                                                                "RealPart",
                                                                "RealImaginary",
                                                                "MagnitudePhase",
                                                                "SpectralDensityFunction",
                                                                "AutocorrelationFunction",
                                                                "CumulativeRootMeanSquare",
                                                                "RootMeanSquare",
                                                                "AllRandomResults",
                                                                "RandomPrint",
                                                                "NoRandomPrint",
                                                                "RandomPunch",
                                                                "AbsoluteMotion",
                                                                "RelativeMotion"};

// The describers that change the element tables, by their neutral names.
constexpr std::array<std::string_view, 4> CORNER_DESCRIBERS{"CornerResults", "BilinearCornerStress",
                                                            "CubicCornerStress", "StrainGageCornerStress"};
constexpr std::string_view CENTER_DESCRIBER = "CenterOnly";
constexpr std::string_view FIBER_DESCRIBER = "FiberDistance";
constexpr std::string_view CURVATURE_DESCRIBER = "StrainCurvature";
constexpr std::string_view VON_MISES_DESCRIBER = "VonMisesStress";
constexpr std::array<std::string_view, 2> MAX_SHEAR_DESCRIBERS{"MaxShearStress", "ShearComponents"};
constexpr std::string_view SORT_BY_POINT_DESCRIBER = "SortByPoint";
constexpr std::string_view SORT_BY_TIME_DESCRIBER = "SortByTime";
constexpr std::string_view PUNCH_DESCRIBER = "PunchFile";

bool isOneOf(std::string_view describer, std::span<const std::string_view> names)
{
    return std::find(names.begin(), names.end(), describer) != names.end();
}

std::optional<OutputKind> kindOf(std::string_view request)
{
    for (const RequestName& entry : REQUEST_NAMES)
        if (request == entry.name)
            return entry.kind;
    return std::nullopt;
}

base::Diagnostic requestProblem(base::Severity severity, std::string code, std::int32_t subcase, std::string message);

// The Case Control request MEFFMASS and its reference point describer GRID = gid (a describer O_o_Reader keeps as the
// deck writes it, with its value).
constexpr std::string_view MODAL_EFFECTIVE_MASS = "ModalEffectiveMassOutput";
// The aerodynamic outputs of SOL 144 (APRESSURE, AEROF).
constexpr std::string_view AERO_PRESSURES = "AeroPressureOutput";
constexpr std::string_view AERO_FORCES = "AeroForceOutput";
// DSAPRT: the print and export of the design sensitivities, not an output of the analysis; SOL 200 honours its END.
constexpr std::string_view DESIGN_SENSITIVITY_PRINT = "DesignSensitivityOutput";
constexpr std::string_view REFERENCE_NODE_DESCRIBER = "ReferenceNode=";

ModalEffectiveMassRequest modalEffectiveMassOf(const input::OutputRequest& request, std::int32_t subcase,
                                               base::DiagnosticSink& sink)
{
    ModalEffectiveMassRequest result;
    result.isRequested = request.scope != input::OutputRequest::Scope::None;
    for (const std::string& describer : request.describers)
    {
        std::string compact;
        for (const char character : describer)
            if (character != ' ')
                compact.push_back(character);
        if (compact.rfind(REFERENCE_NODE_DESCRIBER, 0) != 0)
            continue;
        try
        {
            result.referencePoint = std::stoi(compact.substr(REFERENCE_NODE_DESCRIBER.size()));
        }
        catch (const std::exception&)
        {
            sink.report(
                requestProblem(base::Severity::Warning, "output-describer-ignored", subcase,
                               base::concatenate({std::string(MODAL_EFFECTIVE_MASS), "(", describer,
                                                  "): expected ReferenceNode = a grid point ID; the origin of the "
                                                  "basic system is the reference point"})));
        }
    }
    return result;
}

base::Diagnostic requestProblem(base::Severity severity, std::string code, std::int32_t subcase, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "SubcaseStart";
    diagnostic.id = subcase;
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

std::string_view requestNameOf(OutputKind kind) noexcept
{
    switch (kind)
    {
    case OutputKind::Displacement:
        return "DisplacementOutput";
    case OutputKind::SpcForces:
        return "SinglePointConstrForceOutput";
    case OutputKind::MpcForces:
        return "MultiPointConstrForceOutput";
    case OutputKind::AppliedLoads:
        return "AppliedLoadOutput";
    case OutputKind::Stress:
        return "StressOutput";
    case OutputKind::Strain:
        return "StrainOutput";
    case OutputKind::Force:
        return "ElemForceOutput";
    case OutputKind::GridPointForces:
        return "NodeForceOutput";
    case OutputKind::StrainEnergy:
        return "ElemStrainEnergyOutput";
    }
    return "";
}

std::string_view tableKindOf(OutputKind kind) noexcept
{
    switch (kind)
    {
    case OutputKind::Displacement:
        return "displacements";
    case OutputKind::SpcForces:
        return "singlePointConstrForces";
    case OutputKind::MpcForces:
        return "multiPointConstrForces";
    case OutputKind::AppliedLoads:
        return "appliedLoads";
    case OutputKind::Stress:
        return "stress";
    case OutputKind::Strain:
        return "strain";
    case OutputKind::Force:
        return "forces";
    case OutputKind::GridPointForces:
        return "nodePointForces";
    case OutputKind::StrainEnergy:
        return "strainEnergy";
    }
    return "";
}

ElementOutputSelection elementSelectionOf(const OutputPlan& plan)
{
    const auto kindOf = [](const OutputRequestPlan& request)
    {
        ElementKindSelection selection;
        selection.isRequested = request.isRequested;
        selection.isAll = request.isAll;
        selection.members = request.members;
        selection.isCentreOnly = !request.isCorner;
        return selection;
    };
    ElementOutputSelection selection;
    selection.stress = kindOf(plan.of(OutputKind::Stress));
    selection.strain = kindOf(plan.of(OutputKind::Strain));
    selection.forces = kindOf(plan.of(OutputKind::Force));
    selection.isStrainAtFibres = plan.of(OutputKind::Strain).isFiber;
    selection.isMaximumShear = false;
    return selection;
}

bool OutputRequestPlan::takes(std::int32_t id) const
{
    if (!isRequested)
        return false;
    return isAll || std::binary_search(members.begin(), members.end(), id);
}

void OutputPlan::restrictTo(const std::array<OutputRequestPlan, OUTPUT_KIND_COUNT>& required)
{
    for (std::size_t k = 0; k < OUTPUT_KIND_COUNT; ++k)
    {
        const auto kind = static_cast<OutputKind>(k);
        if (!required[k].isRequested)
        {
            setRequest(kind, OutputRequestPlan{});
            continue;
        }
        OutputRequestPlan restricted = of(kind);
        if (!restricted.isRequested)
        {
            setRequest(kind, required[k]);
            continue;
        }
        restricted.isAll = required[k].isAll;
        restricted.members = required[k].members;
        setRequest(kind, std::move(restricted));
    }
}

void OutputPlan::addRequired(const std::array<OutputRequestPlan, OUTPUT_KIND_COUNT>& required)
{
    for (std::size_t k = 0; k < OUTPUT_KIND_COUNT; ++k)
    {
        const OutputRequestPlan& extra = required[k];
        if (!extra.isRequested)
            continue;
        const auto kind = static_cast<OutputKind>(k);
        OutputRequestPlan merged = of(kind);
        if (!merged.isRequested)
        {
            setRequest(kind, extra);
            continue;
        }
        if (merged.isAll || extra.isAll)
        {
            merged.isAll = true;
            merged.members.clear();
        }
        else
        {
            std::vector<std::int32_t> members;
            std::set_union(merged.members.begin(), merged.members.end(), extra.members.begin(), extra.members.end(),
                           std::back_inserter(members));
            merged.members = std::move(members);
        }
        setRequest(kind, std::move(merged));
    }
}

void OutputPlan::setRequest(OutputKind kind, OutputRequestPlan request)
{
    m_requests[static_cast<std::size_t>(kind)] = std::move(request);
}

OutputPlan OutputPlan::everything()
{
    OutputPlan plan;
    for (const OutputKind kind : {OutputKind::Displacement, OutputKind::SpcForces, OutputKind::MpcForces,
                                  OutputKind::Stress, OutputKind::Strain, OutputKind::Force})
    {
        OutputRequestPlan request;
        request.isRequested = true;
        request.isCorner = true;
        request.isVonMises = true;
        plan.setRequest(kind, std::move(request));
    }
    plan.m_isAeroPressureRequested = true;
    plan.m_isAeroForceRequested = true;
    return plan;
}

OutputPlan OutputPlan::forSubcase(const input::ModelSnapshot& snapshot, std::int32_t subcase,
                                  base::DiagnosticSink& sink)
{
    // A Case Control without any output request (a model built in memory, a snapshot without the table) gives every
    // output, so that the caller always gets results; a deck with requests gets those of each subcase. O_o_Reader
    // gives output_requests_table for every model since data contract 31, with no row when there is no request.
    if (!snapshot.hasTable(input::TableRole::OutputRequests) ||
        snapshot.outputRequests().requestReader().rowCount() == 0)
        return everything();
    OutputPlan plan;
    plan.m_isFromCaseControl = true;
    const input::OutputRequests& requests = snapshot.outputRequests();

    for (const input::OutputRequest& request : requests.requestsOf(subcase))
    {
        // The request and its describers by their neutral names; a message names them as the deck writes them.
        const std::string requestName = std::string(request.request);
        if (request.request == MODAL_EFFECTIVE_MASS)
        {
            plan.m_modalEffectiveMass = modalEffectiveMassOf(request, subcase, sink);
            continue;
        }
        if (request.request == AERO_PRESSURES || request.request == AERO_FORCES)
        {
            const bool isOn = request.scope != input::OutputRequest::Scope::None;
            (request.request == AERO_FORCES ? plan.m_isAeroForceRequested : plan.m_isAeroPressureRequested) = isOn;
            continue;
        }
        if (request.request == DESIGN_SENSITIVITY_PRINT)
        {
            sink.report(requestProblem(base::Severity::Warning, "design-sensitivity-print-not-produced", subcase,
                                       base::concatenate({requestName,
                                                          ": its END=SENS is honoured by SOL 200, its print and "
                                                          "export of the design sensitivities are not produced by "
                                                          "this version of O_o_Solver (the gradients are given by "
                                                          "the design module)"})));
            continue;
        }
        const std::optional<OutputKind> kind = kindOf(request.request);
        if (!kind)
        {
            sink.report(requestProblem(base::Severity::Warning, "output-not-produced", subcase,
                                       base::concatenate({"the output request ", requestName,
                                                          " is not produced by this version of O_o_Solver "
                                                          "(DisplacementOutput, "
                                                          "SinglePointConstrForceOutput, "
                                                          "MultiPointConstrForceOutput, "
                                                          "AppliedLoadOutput, StressOutput, "
                                                          "StrainOutput, ElemForceOutput, "
                                                          "NodeForceOutput, "
                                                          "ElemStrainEnergyOutput are); it is ignored"})));
            continue;
        }
        OutputRequestPlan entry;
        entry.isRequested = request.scope != input::OutputRequest::Scope::None;
        for (const std::string& describer : request.describers)
        {
            if (isOneOf(describer, CORNER_DESCRIBERS))
                entry.isCorner = true;
            else if (describer == CENTER_DESCRIBER)
                entry.isCorner = false;
            else if (describer == FIBER_DESCRIBER)
                entry.isFiber = true;
            else if (describer == CURVATURE_DESCRIBER)
                entry.isFiber = false;
            else if (describer == VON_MISES_DESCRIBER)
                entry.isVonMises = true;
            else if (isOneOf(describer, MAX_SHEAR_DESCRIBERS))
                entry.isVonMises = false;
            else if (describer == SORT_BY_TIME_DESCRIBER)
                sink.report(requestProblem(
                    base::Severity::Warning, "output-sort-by-time", subcase,
                    base::concatenate({requestName, "(", describer, ") is given in ", SORT_BY_POINT_DESCRIBER,
                                       " (one table per subcase): O_o_Solver writes no table sorted by time"})));
            else if (describer == PUNCH_DESCRIBER)
                sink.report(requestProblem(base::Severity::Warning, "output-punch-file", subcase,
                                           base::concatenate({requestName, "(", describer,
                                                              "): O_o_Solver writes no result file; the table is "
                                                              "in the attached results"})));
            else if (!isOneOf(describer, NO_EFFECT_DESCRIBERS))
                // Nothing silently ignored: a describer this version does not apply is named.
                sink.report(requestProblem(base::Severity::Warning, "output-describer-ignored", subcase,
                                           base::concatenate({requestName, "(", describer,
                                                              "): this describer is not applied by this version of "
                                                              "O_o_Solver; the table is given without it"})));
        }
        if (request.scope == input::OutputRequest::Scope::Set)
        {
            const std::optional<std::span<const std::int32_t>> members = requests.setMembers(request.setId, subcase);
            if (!members)
            {
                sink.report(requestProblem(
                    base::Severity::Error, "output-unknown-set", subcase,
                    base::concatenate({"expected Case Control SetDefinition ", std::to_string(request.setId), " of ",
                                       requestName, " in subcase ", std::to_string(subcase),
                                       " or in the global section, found none; define the "
                                       "SetDefinition or write the request with = ALL"})));
                continue;
            }
            entry.isAll = false;
            entry.members.assign(members->begin(), members->end());
            std::sort(entry.members.begin(), entry.members.end());
            entry.members.erase(std::unique(entry.members.begin(), entry.members.end()), entry.members.end());
        }
        plan.setRequest(*kind, std::move(entry));
    }
    // The corner option of one element output request applies to the others of the subcase (STRESS(BILIN) also gives
    // the corner forces of FORCE(PLOT)), so that the element tables of a subcase have the same places.
    bool isCorner = false;
    for (const OutputKind kind : {OutputKind::Stress, OutputKind::Strain, OutputKind::Force})
        isCorner = isCorner || plan.of(kind).isCorner;
    for (const OutputKind kind : {OutputKind::Stress, OutputKind::Strain, OutputKind::Force})
        plan.m_requests[static_cast<std::size_t>(kind)].isCorner = isCorner;
    return plan;
}

} // namespace oos::output
