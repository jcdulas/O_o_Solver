#include "oos/solutions/common/RunCatalog.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/tables/ConstraintTable.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace oos::solutions
{

namespace
{

// The cards a static run ignores (damping, dynamic or thermal-only data, plot elements), restricted to the cards the
// snapshot can hold, and the eigenvalue methods; sorted for the binary search.
constexpr std::array<std::string_view, 23> IGNORED_CARDS{
    "BucklingEigenMethod", "BushPropFreq", "ComplexEigenMethod", "ComplexEigenPoles", "DamperDofInline",
    "DamperDofProperty", "DamperProp", "DamperPropFreq", "DamperScalarInline", "DamperScalarProperty",
    "HeatAnisoMat",        "HeatCapacity2", "HeatCapacityProp5",  "HeatIsoMat",        "LanczosEigenMethod",
    "PlotLine2",        "PlotQuad4",      "PlotTria3",       "RealEigenMethod",   "SpringPropFreq",
    "TableModalDamping1",  "ViscousDamper2", "ViscousDamperProp"};
static_assert(std::is_sorted(IGNORED_CARDS.begin(), IGNORED_CARDS.end()));

// A PARAM the input format defines but that changes nothing in this version (by its neutral name), with the reason
// given to the user.
struct IgnoredParameter
{
    std::string_view name;
    std::string_view reason;
};

// K6ROT: the scale of a drilling penalty on shells, which no shell of this version has: the ShellQuad4Drilling and
// ShellTria3Drilling take a membrane with a drilling DOF, the other shells no stiffness for the rotation about their
// normal.
constexpr std::array<IgnoredParameter, 1> IGNORED_PARAMETERS{
    IgnoredParameter{"ShellDrillingStiffness",
                     "no shell element of O_o_Solver has a drilling penalty: the ShellQuad4, ShellTria3, ShellQuad8 and "
                     "ShellTria6 have no stiffness for the rotation about their normal (the automatic constraint "
                     "of singular degrees of freedom, PARAM AutoSingularityConstr, removes it on a flat mesh), and the "
                     "ShellQuad4Drilling and ShellTria3Drilling have their own drilling membrane"}};

// One warning per ignored PARAM the model gives, in the Bulk Data or in any Case Control section.
void reportIgnoredParameters(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& ignored)
{
    if (!snapshot.hasTable(input::TableRole::Params))
        return;
    const base::DictionaryView names = snapshot.params().reader().dictionaryTexts("name");
    for (const IgnoredParameter& parameter : IGNORED_PARAMETERS)
    {
        bool isGiven = false;
        for (std::int64_t row = 0; row < names.size() && !isGiven; ++row)
            isGiven = names[row] == parameter.name;
        if (!isGiven)
            continue;
        const std::string label = std::string(parameter.name);
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "ignored-parameter";
        diagnostic.card = base::concatenate({"PARAM ", label});
        diagnostic.message = base::concatenate({"PARAM ", label, " has no effect and is ignored: ", parameter.reason,
                                                "; remove the parameter, nothing else to do"});
        ignored.push_back(std::move(diagnostic));
    }
}

} // namespace

input::SupportCatalog staticSupportCatalog(const elements::ElementRegistry& registry)
{
    input::SupportCatalog catalog = input::SupportCatalog::current();
    registry.addSupportTo(catalog);
    constexpr std::array<const char*, 5> CONSTRAINT_CARDS{"SinglePointConstr", "SinglePointConstrList",
                                                          "EnforcedMotionConstr",
                                                          input::PERMANENT_CONSTRAINT_CARD.data(), "MultiPointConstr"};
    // The temperature cards of TEMPERATURE(LOAD) and (INITIAL) and the element deformations of DEFORM
    // (TemperatureField; the element families check what they can take, LoadVectorBuilder the elements a DEFORM names).
    // The inertial loads (InertialLoads; ACCEL, AccelNodeLoad, RotationalForceLoad need inertialLoadsTable, and are
    // refused at the load assembly without it) and the supports of inertia relief (SUPORT, SUPORT1: dofSetsTable).
    constexpr std::array<const char*, 22> LOAD_CARDS{
        "PointForceLoad",         "PointForceLoad1",      "PointForceLoad2",         "PointMomentLoad",
        "PointMomentLoad1",       "PointMomentLoad2",     "ScalarPointLoad",         "GravityLoad",
        "AccelFieldLoad",         "AccelNodeLoad",        "RotationalForceLoad",     "PointsPressureLoad",
        "BeamDistributedLoad",    "ShellPressureLoad",    "FacePressureLoad",        "NodeTemperatureLoad",
        "DefaultTemperatureLoad", "PlateTemperatureLoad", "LineElemTemperatureLoad", "FreeBodySupport",
        "FreeBodySupportSet",     "AxialDeformationLoad"};
    for (const char* card : CONSTRAINT_CARDS)
        catalog.addCard(card);
    for (const char* card : LOAD_CARDS)
        catalog.addCard(card);
    // The rigid elements are element batches of the snapshot without a family: RigidElimination reads them from
    // rigidElementsTable and refuses what it cannot eliminate.
    constexpr std::array<const char*, 5> RIGID_CARDS{"RigidBodyMulti", "RigidBodyStar", "RigidInterpolation",
                                                     "RigidBar", "RigidRod"};
    for (const char* card : RIGID_CARDS)
        catalog.addCard(card);
    return catalog;
}

bool isIgnoredByStaticRun(std::string_view card) noexcept
{
    return std::binary_search(IGNORED_CARDS.begin(), IGNORED_CARDS.end(), card);
}

base::SupportReport checkStaticSupport(const input::ModelSnapshot& snapshot, const elements::ElementRegistry& registry)
{
    return checkSupport(snapshot, registry, AnalysisKind::Statics);
}

base::SupportReport checkSupport(const input::ModelSnapshot& snapshot, const elements::ElementRegistry& registry,
                                 AnalysisKind analysis)
{
    input::SupportCatalog catalog = staticSupportCatalog(registry);
    // The method cards the analysis reads (EIGRL, RealEigenMethod, BucklingEigenMethod: METHOD selects RealEigenMethod
    // or LanczosEigenMethod in SOL 103, BucklingEigenMethod or LanczosEigenMethod in SOL 105).
    if (analysis == AnalysisKind::Modes)
    {
        catalog.addCard("LanczosEigenMethod");
        catalog.addCard("RealEigenMethod");
    }
    else if (analysis == AnalysisKind::Buckling)
    {
        catalog.addCard("LanczosEigenMethod");
        catalog.addCard("BucklingEigenMethod");
    }
    const char* analysisName = analysis == AnalysisKind::Statics    ? "a linear static analysis"
                               : analysis == AnalysisKind::Modes    ? "a real eigenvalue analysis (SOL 103)"
                               : analysis == AnalysisKind::Buckling ? "a linear buckling analysis (SOL 105)"
                                                                    : "a static aeroelastic analysis (SOL 144)";
    base::SupportReport report = catalog.check(snapshot);
    std::vector<base::Diagnostic> refusals;
    for (base::Diagnostic& diagnostic : report.refusals)
    {
        // The diagnostics of the catalog name the card by its deck keyword (EIGRL): the list is of neutral names.
        if (diagnostic.code != "unsupported-card" || !isIgnoredByStaticRun(diagnostic.card))
        {
            refusals.push_back(std::move(diagnostic));
            continue;
        }
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "ignored-card";
        diagnostic.message =
            base::concatenate({"the card ", diagnostic.card, " has no effect on ", analysisName,
                               " and is ignored, like all the Bulk Data this analysis does not read (damping, "
                               "dynamic or thermal-only data, plot elements, eigenvalue methods); nothing to do"});
        report.ignored.push_back(std::move(diagnostic));
    }
    // The element formulations the model chose that the registry could not follow (a name a card does not have).
    for (const base::Diagnostic& problem : registry.selectionProblems())
        refusals.push_back(problem);
    report.refusals = std::move(refusals);
    reportIgnoredParameters(snapshot, report.ignored);
    return report;
}

} // namespace oos::solutions
