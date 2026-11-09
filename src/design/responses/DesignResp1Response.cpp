#include "oos/design/responses/DesignResp1Response.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/aids/LinearityIndicator.hpp"
#include "oos/design/aids/SelfAdjointResponses.hpp"
#include "oos/design/model/DesignCodes.hpp"
#include "oos/design/responses/composite/StrengthRatio.hpp"
#include "oos/design/responses/weight_volume/MassProperties.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/input/tables/ParamTable.hpp"
#include "oos/output/sections/CutLoads.hpp"
#include "oos/output/sections/SectionLoads.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

namespace oos::design
{

namespace
{

// The ownerKind of a DesignResp1 whose attributes are element IDs (otherwise a property card: ShellProp, RodProp).
constexpr std::string_view ELEMENTS_OWNER = "Elements";

constexpr const char* CARD = "DesignResp1";
// DesignResp1: ATTA and ATTB of WEIGHT default to 3 (the weight along z), rows and columns 1 to 6.
constexpr std::int32_t DEFAULT_WEIGHT_ENTRY = 3;
constexpr std::int32_t RIGID_BODY_SIZE = 6;
// DesignResp1: the lamina number of a composite response defaults to 1.
constexpr std::int32_t DEFAULT_LAMINA = 1;
// The item codes tried when a caller's response names its items by a column (the largest item code that O_o_Solver
// maps is 111, the Beam2 end B margin).
constexpr std::int32_t LARGEST_ITEM_CODE = 111;
// PARAM GRDPNT: −1 (the default) or a non-grid value puts the reference at the origin of the basic system.
constexpr std::int32_t NO_WEIGHT_REFERENCE_NODE = -1;
// Item code 4 of ESE is the element strain energy density.
constexpr std::int32_t STRAIN_ENERGY_DENSITY_ITEM = 4;
// The column of the element strain energy in the strain energy table (StiffnessRecovery), whence the density.
constexpr std::string_view STRAIN_ENERGY_COLUMN = "strainEnergy";
// DesignResp1: the approximation codes ATTB of EIGN, FREQ and LAMA (blank: the Rayleigh quotient for EIGN and
// FREQ, the direct linearisation for LAMA).
constexpr double EIGEN_APPROXIMATION_DIRECT = 1.0;
constexpr double EIGEN_APPROXIMATION_INVERSE = 2.0;

base::Diagnostic problem(const char* code, const DesignResp1Definition& definition, std::string field,
                         std::string message)
{
    return designProblem(code, CARD, definition.id, std::move(field), std::move(message));
}

bool isComposite(ResponseType type)
{
    return type == ResponseType::CompositeStress || type == ResponseType::CompositeStrain ||
           type == ResponseType::CompositeFailure || type == ResponseType::CompositeStrengthRatio;
}

output::OutputKind outputKindOf(ResponseType type)
{
    switch (type)
    {
    case ResponseType::Displacement:
    case ResponseType::DeflectionToThickness:
    case ResponseType::Rotation:
        return output::OutputKind::Displacement;
    case ResponseType::MeanDisplacement:
        return output::OutputKind::AppliedLoads;
    case ResponseType::SpcForce:
        return output::OutputKind::SpcForces;
    case ResponseType::Strain:
    case ResponseType::CompositeStrain:
        return output::OutputKind::Strain;
    case ResponseType::Force:
        return output::OutputKind::Force;
    case ResponseType::StrainEnergy:
        return output::OutputKind::StrainEnergy;
    default:
        // STRESS, and the ply stresses and failure indices, which ElementResultBuilder makes with STRESS.
        return output::OutputKind::Stress;
    }
}

// The column names of the table an item location reads, for the caller's selector by name.
std::vector<std::string> columnNames(const elements::ElementFamily& family, const ItemLocation& location)
{
    if (location.tableKind == "strainEnergy")
        return {"strainEnergy", "percent", "strainEnergyDensity"};
    if (location.isComposite)
    {
        const elements::LayerOutput output = location.tableKind == "stress"   ? elements::LayerOutput::Stresses
                                             : location.tableKind == "strain" ? elements::LayerOutput::Strains
                                                                              : elements::LayerOutput::FailureIndices;
        return family.layerOutputColumns(output);
    }
    if (location.tableKind == "forces")
        return family.outputColumns(elements::ElementOutput::Forces);
    if (location.tableKind == "strain")
    {
        std::vector<std::string> names = family.outputColumns(elements::ElementOutput::StrainCurvatures);
        return names.empty() ? family.outputColumns(elements::ElementOutput::Strains) : names;
    }
    return family.outputColumns(elements::ElementOutput::Stresses);
}

void addMembers(output::OutputRequestPlan& request, const std::vector<std::int32_t>& ids)
{
    if (request.isRequested && request.isAll)
        return;
    request.isRequested = true;
    request.isAll = false;
    std::vector<std::int32_t> merged;
    std::vector<std::int32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    std::set_union(request.members.begin(), request.members.end(), sorted.begin(), sorted.end(),
                   std::back_inserter(merged));
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    request.members = std::move(merged);
}

} // namespace

DesignResp1Response::DesignResp1Response(DesignResp1Definition definition, const input::ModelSnapshot& snapshot,
                                         const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems)
    : m_definition(std::move(definition))
{
    const ResponseType type = m_definition.type;
    const bool takesAttributeB = type == ResponseType::Weight || isComposite(type) || isEigenType(type) ||
                                 type == ResponseType::StabilityDerivative;
    if (m_definition.attributeB && !takesAttributeB)
        problems.push_back(
            problem(design_code::UNSUPPORTED_RESPONSE, m_definition, "attributeB",
                    base::concatenate({"expected DesignResp1.attributeB blank on a ", responseTypeName(type),
                                       " response of a static analysis, found a mode number; the eigenvector "
                                       "responses of a MODES analysis (the DISP, STRESS ... of a mode) are not "
                                       "supported by this version: constrain the eigenvalues (EIGN, FREQ) instead"})));
    if (m_definition.region && (isGlobalType(type) || isEigenType(type) || isAeroelasticType(type)))
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "regionId",
                    base::concatenate({"expected DesignResp1.regionId blank on a ", responseTypeName(type),
                                       " response (no DesignResp1.regionId for this type), found ",
                                       std::to_string(*m_definition.region)})));
    switch (type)
    {
    case ResponseType::Weight:
    case ResponseType::Volume:
        if (type == ResponseType::Weight)
        {
            m_weightRow = m_definition.attributeA.value_or(DEFAULT_WEIGHT_ENTRY);
            const double column = m_definition.attributeB.value_or(DEFAULT_WEIGHT_ENTRY);
            m_weightColumn = static_cast<std::int32_t>(column);
            if (m_weightRow < 1 || m_weightRow > RIGID_BODY_SIZE || column != std::floor(column) ||
                m_weightColumn < 1 || m_weightColumn > RIGID_BODY_SIZE)
                problems.push_back(
                    problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                            base::concatenate({"expected DesignResp1.attributeA and "
                                               "DesignResp1.attributeB of WEIGHT from 1 to 6 (a row "
                                               "and a column of the rigid-body weight matrix), found ",
                                               std::to_string(m_weightRow), " and ", std::to_string(column)})));
        }
        for (const std::int32_t superelementId : m_definition.targets)
            if (superelementId != 0)
                problems.push_back(problem(
                    design_code::UNSUPPORTED_RESPONSE, m_definition, "attribute1",
                    base::concatenate({"expected DesignResp1.attributes blank, ALL or 0 (the residual), found "
                                       "superelement ",
                                       std::to_string(superelementId), "; superelements are not supported yet"})));
        break;
    case ResponseType::FractionalMass:
        resolveMassTargets(catalog, problems);
        break;
    case ResponseType::Displacement:
    case ResponseType::SpcForce:
        resolveGridTargets(snapshot, problems);
        break;
    case ResponseType::Compliance:
    case ResponseType::MeanDisplacement:
        if (!m_definition.targets.empty() || m_definition.attributeA || !m_definition.componentName.empty())
            problems.push_back(
                problem(design_code::INVALID_RESPONSE, m_definition, "attributes",
                        base::concatenate({"expected no grid point, element or component for ", responseTypeName(type),
                                           " (one item per subcase, from its loads), found some"})));
        break;
    case ResponseType::DeflectionToThickness:
    case ResponseType::Rotation:
        resolvePointTargets(snapshot, problems);
        break;
    case ResponseType::SectionLoad:
    case ResponseType::CutLoad:
        resolveSections(snapshot, problems);
        break;
    case ResponseType::ContourDisplacement:
    case ResponseType::ContourForce:
        resolveZones(problems);
        break;
    case ResponseType::Eigenvalue:
    case ResponseType::Frequency:
    case ResponseType::BucklingFactor:
        resolveEigen(problems);
        break;
    case ResponseType::TrimVariable:
    case ResponseType::StabilityDerivative:
        resolveAeroelastic(problems);
        break;
    default:
        resolveElementTargets(catalog, problems);
        break;
    }
}

void DesignResp1Response::resolveAeroelastic(std::vector<base::Diagnostic>& problems)
{
    const ResponseType type = m_definition.type;
    if (!m_definition.attributeA || *m_definition.attributeA < 1)
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                    base::concatenate({"expected the ID of the AeroRigidTrimVariable or AeroControlSurface of a trim "
                                       "variable in DesignResp1.attributeA of ",
                                       responseTypeName(type), ", found ",
                                       m_definition.attributeA ? std::to_string(*m_definition.attributeA) : "blank"})));
    if (!m_definition.propertyType.empty())
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "ownerKind",
                                   base::concatenate({"expected DesignResp1.ownerKind blank for ",
                                                      responseTypeName(type), ", found ", m_definition.propertyType})));
    if (type == ResponseType::TrimVariable)
    {
        if (!m_definition.targets.empty() || !m_definition.componentName.empty())
            problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "attribute1",
                                       "expected no DesignResp1.attributes for TRIM (one item per subcase: the "
                                       "value of the trim variable of DesignResp1.attributeA), "
                                       "found some"));
        return;
    }
    // STABDER: ATTB the restraint flag (0 unrestrained, 1 restrained), ATT1 the component.
    const double flag = m_definition.attributeB.value_or(0.0);
    if (flag != 0.0 && flag != 1.0)
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "attributeB",
                                   base::concatenate({"expected DesignResp1.attributeB of STABDER blank or 0 "
                                                      "(unrestrained) or 1 (restrained), found ",
                                                      std::to_string(flag)})));
    m_isRestrained = flag == 1.0;
    if (m_definition.targets.size() != 1 || m_definition.targets.front() < 1 || m_definition.targets.front() > 6)
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attribute1",
                    base::concatenate({"expected one component of STABDER in DesignResp1.attribute1, 1 to 6 (CX, CY, "
                                       "CZ, CMX, CMY, CMZ), "
                                       "found ",
                                       std::to_string(m_definition.targets.size()), " DesignResp1.attributes",
                                       m_definition.targets.empty()
                                           ? std::string("; give the component in DesignResp1.attribute1")
                                           : base::concatenate({" from ", std::to_string(m_definition.targets.front()),
                                                                "; give one component in DesignResp1.attribute1"})})));
    else
        m_stabilityComponent = m_definition.targets.front();
}

void DesignResp1Response::aeroelasticValues(std::int32_t subcase, ResponseContext& context,
                                            ResponseValues& values) const
{
    const ResponseType type = m_definition.type;
    const auto name = [&]
    {
        return base::concatenate({"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, ", ",
                                  responseTypeName(type), ")"});
    };
    const solutions::TrimSubcaseResult* trimmed = nullptr;
    if (context.trim != nullptr)
        for (const solutions::TrimSubcaseResult& candidate : context.trim->subcases)
            if (candidate.subcase == subcase)
                trimmed = &candidate;
    if (trimmed == nullptr)
        throw base::OptimisationError(
            base::concatenate({name(), ": expected the trim of subcase ", std::to_string(subcase),
                               ", which is not a SAERO subcase of the analysis; select the subcases of its analysis "
                               "(SubcaseDesignConstrSelection, "
                               "subcases=)"}));
    const std::int32_t variable = m_definition.attributeA.value_or(0);
    const auto found = std::find_if(trimmed->variables.begin(), trimmed->variables.end(),
                                    [&](const solutions::TrimVariableValue& value) { return value.id == variable; });
    if (found == trimmed->variables.end())
        throw base::OptimisationError(
            base::concatenate({name(), ": AeroRigidTrimVariable or AeroControlSurface ", std::to_string(variable),
                               " is not a variable of the AeroTrim of subcase ", std::to_string(subcase),
                               "; name in DesignResp1.attributeA the ID of one of its variables"}));
    if (type == ResponseType::TrimVariable)
    {
        values.items.push_back({subcase, variable, 0, 0});
        values.values.push_back(found->value);
        return;
    }
    const auto row = std::find_if(trimmed->derivatives.begin(), trimmed->derivatives.end(),
                                  [&](const solutions::StabilityDerivativeRow& r) { return r.label == found->label; });
    if (row == trimmed->derivatives.end())
        throw base::OptimisationError(base::concatenate({name(), ": the trim of subcase ", std::to_string(subcase),
                                                         " has no stability derivatives of variable ", found->label,
                                                         "; give the STABDER a variable of the subcase's AeroTrim"}));
    if (!m_isRestrained && !trimmed->hasUnrestrained)
        throw base::OptimisationError(base::concatenate(
            {name(),
             ": the unrestrained derivatives (DesignResp1.attributeB 0) need the FreeBodySupport DOFs of a "
             "free trim, and subcase ",
             std::to_string(subcase), " has none; ask the restrained one (DesignResp1.attributeB 1)"}));
    const std::array<double, 6>& coefficients = m_isRestrained ? row->elasticRestrained : row->elasticUnrestrained;
    values.items.push_back({subcase, variable, m_stabilityComponent, m_isRestrained ? 1 : 0});
    values.values.push_back(coefficients[static_cast<std::size_t>(m_stabilityComponent - 1)]);
}

void DesignResp1Response::resolveEigen(std::vector<base::Diagnostic>& problems)
{
    const ResponseType type = m_definition.type;
    if (!m_definition.attributeA || *m_definition.attributeA < 1)
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                    base::concatenate({"expected the mode number in DesignResp1.attributeA of ", responseTypeName(type),
                                       " (an integer of at least 1), found ",
                                       m_definition.attributeA ? std::to_string(*m_definition.attributeA) : "blank"})));
    if (m_definition.attributeB)
    {
        // The approximation code, blank, 1 (direct) or 2 (inverse).
        const double code = *m_definition.attributeB;
        if (code != EIGEN_APPROXIMATION_DIRECT && code != EIGEN_APPROXIMATION_INVERSE)
            problems.push_back(problem(
                design_code::INVALID_RESPONSE, m_definition, "attributeB",
                base::concatenate({"expected DesignResp1.attributeB of ", responseTypeName(type),
                                   " blank, 1 (direct linearisation) or 2 (inverse), the approximation code, found ",
                                   std::to_string(code)})));
    }
    // DesignResp1: the ownerKind of Eigenvalue and Frequency may name the source of the mode, Structure (the default)
    // or Fluid.
    const std::string& source = m_definition.propertyType;
    if (!source.empty() && !(isModalType(type) && source == "Structure"))
        problems.push_back(
            problem(design_code::UNSUPPORTED_RESPONSE, m_definition, "ownerKind",
                    base::concatenate({"expected DesignResp1.ownerKind of ", responseTypeName(type),
                                       isModalType(type) ? " blank or Structure (O_o_Solver has no fluid)" : " blank",
                                       ", found ", source})));
    if (!m_definition.targets.empty() || !m_definition.componentName.empty() || !m_definition.plies.empty())
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attribute1",
                    base::concatenate(
                        {"expected no DesignResp1.attributes for ", responseTypeName(type),
                         " (one item per subcase: the root of the mode of DesignResp1.attributeA), found some"})));
}

void DesignResp1Response::eigenValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const
{
    const ResponseType type = m_definition.type;
    const char* analysis = type == ResponseType::BucklingFactor ? "buckling" : "modes";
    const auto table = std::find_if(context.eigenvalueTables.begin(), context.eigenvalueTables.end(),
                                    [&](const output::EigenvalueTable* candidate)
                                    { return candidate->subcase == subcase && candidate->analysis == analysis; });
    if (table == context.eigenvalueTables.end())
        throw base::OptimisationError(base::concatenate(
            {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, ", ", responseTypeName(type),
             "): expected the eigenvalues of subcase ", std::to_string(subcase), ", which is not a ",
             type == ResponseType::BucklingFactor ? "Buckling" : "NormalModes",
             " subcase of the analysis; select the subcases of its analysis (SubcaseDesignConstrSelection, "
             "subcases=)"}));
    const std::int32_t asked = m_definition.attributeA.value_or(1);
    std::int32_t mode = asked;
    if (const auto tracked = context.trackedModes.find({m_definition.id, subcase});
        tracked != context.trackedModes.end())
        mode = tracked->second;
    const std::vector<output::EigenvalueRow>& rows = (*table)->rows;
    if (mode < 1 || static_cast<std::size_t>(mode) > rows.size())
        throw base::OptimisationError(
            base::concatenate({"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, ", ",
                               responseTypeName(type), "): mode ", std::to_string(mode), " of subcase ",
                               std::to_string(subcase), " is not among the ", std::to_string(rows.size()),
                               " roots the analysis found; ask the method of the subcase (LanczosEigenMethod "
                               "LanczosEigenMethod.desiredModeCount, "
                               "LanczosEigenMethod.upperFrequency) for more roots"}));
    const output::EigenvalueRow& row = rows[static_cast<std::size_t>(mode - 1)];
    values.items.push_back({subcase, asked, 0, mode});
    values.values.push_back(type == ResponseType::Frequency ? row.cycles : row.eigenvalue);
}

void DesignResp1Response::resolveZones(std::vector<base::Diagnostic>& problems)
{
    if (m_definition.attributeA || !m_definition.targets.empty() || !m_definition.propertyType.empty())
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attributes",
                    base::concatenate({"expected no item code, DesignResp1.ownerKind nor DesignResp1.attributes for ",
                                       responseTypeName(m_definition.type),
                                       " (the contours of its zones give its items), found some"})));
    if (m_definition.zones.empty())
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "zone",
                                   base::concatenate({"expected the zones of ", responseTypeName(m_definition.type),
                                                      ", found none; give zone="})));
    if (m_definition.componentName.empty())
        return;
    const auto found =
        std::find(output::ZONE_COMPONENTS.begin(), output::ZONE_COMPONENTS.end(), m_definition.componentName);
    if (found == output::ZONE_COMPONENTS.end())
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "component",
                                   base::concatenate({"expected a component t1, t2, t3, r1, r2 or r3 of the zone's "
                                                      "system, found '",
                                                      m_definition.componentName, "'"})));
    else
        m_zoneComponent = static_cast<std::int32_t>(found - output::ZONE_COMPONENTS.begin()) + 1;
}

void DesignResp1Response::zoneValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const
{
    const bool isForce = m_definition.type == ResponseType::ContourForce;
    for (const std::int32_t zone : m_definition.zones)
    {
        const auto table = std::find_if(context.zones.begin(), context.zones.end(),
                                        [&](const output::ZoneTable& candidate) { return candidate.id == zone; });
        if (table == context.zones.end() || !table->hasValues)
            throw base::OptimisationError(base::concatenate(
                {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, ", ",
                 responseTypeName(m_definition.type), "): expected the contour data of zone ", std::to_string(zone),
                 " in the analysis (DesignProblem::evaluateStatics computes them), found none; evaluate the response "
                 "on an analysis of the design problem that declares the zone"}));
        const auto column = static_cast<std::size_t>(std::find(table->cases.begin(), table->cases.end(), subcase) -
                                                     table->cases.begin());
        if (column == table->cases.size())
            throw base::OptimisationError(base::concatenate(
                {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label,
                 "): the analysis did not solve subcase ", std::to_string(subcase), " for zone ", std::to_string(zone),
                 "; analyse every subcase the response reads (DesignProblem(subcases=...))"}));
        const std::size_t rows = table->contourDofCount();
        const base::BudgetedBuffer<double>& data = isForce ? table->forces : table->displacements;
        for (std::size_t r = 0; r < rows; ++r)
        {
            if (m_zoneComponent != 0 && table->contourComponents[r] != m_zoneComponent)
                continue;
            values.items.push_back({subcase, zone, table->contourComponents[r], table->contourGrids[r]});
            values.values.push_back(data[column * rows + r]);
        }
    }
}

void DesignResp1Response::resolveSections(const input::ModelSnapshot& snapshot, std::vector<base::Diagnostic>& problems)
{
    const bool isSection = m_definition.type == ResponseType::SectionLoad;
    const char* what = isSection ? "section" : "cut";
    if (m_definition.attributeA || !m_definition.targets.empty() || !m_definition.propertyType.empty())
        problems.push_back(problem(
            design_code::INVALID_RESPONSE, m_definition, "attributes",
            base::concatenate({"expected no item code, DesignResp1.ownerKind nor DesignResp1.attributes for ",
                               responseTypeName(m_definition.type), " (its ", what, "s give its items), found some"})));
    if ((isSection ? m_definition.sections.size() : m_definition.cuts.size()) == 0 ||
        !(isSection ? m_definition.cuts.empty() : m_definition.sections.empty()))
    {
        problems.push_back(problem(
            design_code::INVALID_RESPONSE, m_definition, isSection ? "sections" : "cuts",
            base::concatenate({"expected the ", what, "s of ", responseTypeName(m_definition.type), " (and no ",
                               isSection ? "cut" : "section", "), found ", std::to_string(m_definition.sections.size()),
                               " sections and ", std::to_string(m_definition.cuts.size()), " cuts"})));
        return;
    }
    std::set<std::int32_t> ids;
    for (const output::SectionDefinition& section : m_definition.sections)
        if (!ids.insert(section.id).second)
            problems.push_back(problem(
                design_code::INVALID_RESPONSE, m_definition, "sections",
                base::concatenate({"expected distinct section IDs, found ", std::to_string(section.id), " twice"})));
    for (const output::CutDefinition& cut : m_definition.cuts)
        if (!ids.insert(cut.id).second)
            problems.push_back(
                problem(design_code::INVALID_RESPONSE, m_definition, "cuts",
                        base::concatenate({"expected distinct cut IDs, found ", std::to_string(cut.id), " twice"})));
    // The definitions checked against the model as the analysis checks them (output/sections), at once.
    base::MemoryBudget budget(base::MemoryBudget::UNLIMITED);
    const elements::SnapshotIndex index(snapshot, budget);
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const assembly::ElementSet set(index, registry);
    const assembly::DofMap dofMap(snapshot.nodes(), index.points());
    base::DiagnosticSink sink;
    const auto addItems = [&](std::int32_t id, std::span<const std::string_view> components)
    {
        bool isFound = false;
        for (std::size_t k = 0; k < components.size(); ++k)
            if (m_definition.componentName.empty() || m_definition.componentName == components[k])
            {
                m_sectionItems.push_back({id, static_cast<std::int32_t>(k + 1), components[k]});
                isFound = true;
            }
        if (!isFound)
            problems.push_back(
                problem(design_code::INVALID_RESPONSE, m_definition, "component",
                        base::concatenate({"expected a component of ", what, " ", std::to_string(id), ", found '",
                                           m_definition.componentName, "', which it does not have"})));
    };
    if (isSection)
    {
        const output::SectionLoads loads(m_definition.sections, snapshot, set, dofMap, sink);
        for (const output::SectionLoads::Section& section : loads.sections())
            addItems(m_definition.sections[section.definition].id, output::sectionComponents(section.family));
    }
    else
    {
        const output::CutLoads loads(m_definition.cuts, snapshot, set, dofMap, sink);
        for (const output::CutLoads::Cut& cut : loads.cuts())
            addItems(m_definition.cuts[cut.definition].id, output::CUT_COMPONENTS);
    }
    for (const base::Diagnostic& diagnostic : sink.diagnostics())
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, isSection ? "sections" : "cuts", diagnostic.message));
}

void DesignResp1Response::sectionValues(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const
{
    const bool isSection = m_definition.type == ResponseType::SectionLoad;
    const output::SectionLoadTable* table = isSection ? context.sectionLoads : context.cutLoads;
    const auto refuse = [&](const std::string& what)
    {
        // The remedy given by each caller, in `what`.
        throw base::OptimisationError(base::concatenate({"DesignResp1 ", std::to_string(m_definition.id), " (",
                                                         m_definition.label, ", ", responseTypeName(m_definition.type),
                                                         ") in subcase ", std::to_string(subcase), ": ", what}));
    };
    if (table == nullptr || !table->hasValues)
        refuse(
            "expected the section and cut loads of the analysis (DesignProblem::evaluateStatics computes them), "
            "found none; evaluate the response on an analysis of the design problem that declares it");
    const auto column =
        static_cast<std::size_t>(std::find(table->cases.begin(), table->cases.end(), subcase) - table->cases.begin());
    if (column == table->cases.size())
        refuse(
            "the analysis did not solve this subcase; analyse every subcase the response reads "
            "(DesignProblem(subcases=...))");
    // The first row of each section or cut of this response in the table.
    const ResponseContext::FirstSectionRows& firstRowOf = context.firstSectionRows(*table, m_definition.id);
    for (const SectionItem& item : m_sectionItems)
    {
        const auto found = firstRowOf.find(item.id);
        if (found == firstRowOf.end())
            refuse(
                base::concatenate({"expected the loads of ", isSection ? "section " : "cut ", std::to_string(item.id),
                                   " in the analysis, found none; declare it in sections= or cuts= of the "
                                   "response"}));
        const std::size_t row = found->second + static_cast<std::size_t>(item.component - 1);
        values.items.push_back({subcase, item.id, item.component, 0});
        values.values.push_back(table->values[column * table->rowCount() + row]);
    }
}

void DesignResp1Response::resolvePointTargets(const input::ModelSnapshot& snapshot,
                                              std::vector<base::Diagnostic>& problems)
{
    const bool isDeflection = m_definition.type == ResponseType::DeflectionToThickness;
    // The candidates: the corners of the shells (deflection over thickness), every grid point (rotation).
    std::vector<std::int32_t> candidates;
    if (isDeflection)
        candidates = LinearityIndicator::shellGridIds(snapshot);
    else
    {
        const std::span<const std::int32_t> ids = snapshot.nodes().ids();
        candidates.assign(ids.begin(), ids.end());
        std::sort(candidates.begin(), candidates.end());
    }
    if (m_definition.targets.empty())
    {
        m_grids = std::move(candidates);
        if (m_grids.empty())
            problems.push_back(problem(
                design_code::INVALID_RESPONSE, m_definition, "attributes",
                base::concatenate({"expected ", isDeflection ? "a shell grid point" : "a grid point",
                                   " in the model for ", responseTypeName(m_definition.type), ", found none"})));
        return;
    }
    const std::unordered_set<std::int32_t> known(candidates.begin(), candidates.end());
    for (const std::int32_t grid : m_definition.targets)
    {
        if (known.count(grid) == 0)
        {
            problems.push_back(
                problem(design_code::MISSING_RESPONSE_TARGET, m_definition, "attributes",
                        base::concatenate({"expected ", isDeflection ? "a corner of a shell element" : "a grid point",
                                           " of the model, found ", std::to_string(grid), ", which is not one"})));
            continue;
        }
        m_grids.push_back(grid);
    }
}

void DesignResp1Response::resolveGridTargets(const input::ModelSnapshot& snapshot,
                                             std::vector<base::Diagnostic>& problems)
{
    const std::int32_t digits = m_definition.attributeA.value_or(0);
    std::set<std::int32_t> seen;
    for (std::int32_t rest = digits; rest > 0; rest /= 10)
    {
        const std::int32_t component = rest % 10;
        if (component < 1 || component > RIGID_BODY_SIZE || !seen.insert(component).second)
        {
            seen.clear();
            break;
        }
    }
    if (seen.empty())
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                    base::concatenate({"expected the components of ", responseTypeName(m_definition.type),
                                       " as unique digits 1 to 6, found ", std::to_string(digits)})));
    m_components.assign(seen.begin(), seen.end());
    if (m_definition.targets.empty())
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "attribute1",
                    base::concatenate({"expected the grid points of ", responseTypeName(m_definition.type),
                                       " in DesignResp1.attributes, found none"})));
    const std::span<const std::int32_t> ids = snapshot.nodes().ids();
    const std::unordered_set<std::int32_t> grids(ids.begin(), ids.end());
    for (const std::int32_t grid : m_definition.targets)
    {
        if (grids.count(grid) == 0)
        {
            problems.push_back(problem(design_code::MISSING_RESPONSE_TARGET, m_definition, "attributes",
                                       base::concatenate({"expected a grid point of the model, found ",
                                                          std::to_string(grid), ", which the model does not have"})));
            continue;
        }
        m_grids.push_back(grid);
    }
}

void DesignResp1Response::resolveMassTargets(const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems)
{
    if (m_definition.targets.empty())
        return;
    if (m_definition.propertyType.empty() || m_definition.propertyType == ELEMENTS_OWNER)
    {
        problems.push_back(
            problem(design_code::INVALID_RESPONSE, m_definition, "ownerKind",
                    "expected the property card of the DesignResp1.attributes of FRMASS in "
                    "DesignResp1.ownerKind, "
                    "found " +
                        (m_definition.propertyType.empty() ? std::string("blank") : m_definition.propertyType)));
        return;
    }
    for (const std::int32_t propId : m_definition.targets)
    {
        if (!catalog.hasProperty(m_definition.propertyType, propId))
            problems.push_back(
                problem(design_code::MISSING_RESPONSE_TARGET, m_definition, "attributes",
                        base::concatenate({"expected a ", m_definition.propertyType, " of the model, found ",
                                           std::to_string(propId), ", which the model does not have"})));
        else
            m_properties.emplace_back(m_definition.propertyType, propId);
    }
}

void DesignResp1Response::resolveElementTargets(const ElementCatalog& catalog, std::vector<base::Diagnostic>& problems)
{
    const ResponseType type = m_definition.type;
    const bool usesName = !m_definition.componentName.empty();
    if (!usesName && !m_definition.attributeA)
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                                   base::concatenate({"expected the item code of ", responseTypeName(type),
                                                      " in DesignResp1.attributeA, found it blank"})));
    if (m_definition.targets.empty())
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "attribute1",
                                   base::concatenate({"expected the elements or properties of ", responseTypeName(type),
                                                      " in DesignResp1.attributes, found none"})));
    if (m_definition.propertyType.empty())
        problems.push_back(problem(design_code::INVALID_RESPONSE, m_definition, "ownerKind",
                                   "expected DesignResp1.ownerKind = ELEM or a property card for an element response, "
                                   "found it blank"));
    if (isComposite(type))
    {
        if (!m_definition.plies.empty())
            m_plies = m_definition.plies;
        else
            m_plies.push_back(static_cast<std::int32_t>(m_definition.attributeB.value_or(DEFAULT_LAMINA)));
    }
    // The elements, in the order of the ATTi (the elements of a property in increasing ID).
    std::vector<const ElementEntry*> selected;
    const bool areElements = m_definition.propertyType == ELEMENTS_OWNER;
    for (const std::int32_t target : m_definition.targets)
    {
        if (areElements)
        {
            const ElementEntry* entry = catalog.find(target);
            if (entry == nullptr)
                problems.push_back(
                    problem(design_code::MISSING_RESPONSE_TARGET, m_definition, "attributes",
                            base::concatenate({"expected an element of the model, found ", std::to_string(target),
                                               ", which the model does not have"})));
            else
                selected.push_back(entry);
            continue;
        }
        if (m_definition.propertyType.empty())
            continue;
        if (!catalog.hasProperty(m_definition.propertyType, target))
        {
            problems.push_back(
                problem(design_code::MISSING_RESPONSE_TARGET, m_definition, "attributes",
                        base::concatenate({"expected a ", m_definition.propertyType, " of the model, found ",
                                           std::to_string(target), ", which the model does not have"})));
            continue;
        }
        const std::vector<const ElementEntry*> elements = catalog.elementsOfProperty(m_definition.propertyType, target);
        selected.insert(selected.end(), elements.begin(), elements.end());
    }
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(catalog.snapshot());
    for (const ElementEntry* entry : selected)
    {
        ElementTarget target{entry->elementId, entry->card, entry->batch, entry->row, {}, {}};
        const elements::ElementFamily* family = registry.find(entry->card);
        if (family == nullptr)
        {
            problems.push_back(problem(design_code::UNSUPPORTED_RESPONSE, m_definition, "attributes",
                                       base::concatenate({"expected an element type O_o_Solver computes, found ",
                                                          entry->card, " ", std::to_string(entry->elementId)})));
            continue;
        }
        target.elementType = isComposite(type) ? family->layerResultElementType() : family->resultElementType(false);
        if (isComposite(type) && entry->propertyCard != "CompProp" && entry->propertyCard != "CompPropGlobal")
        {
            problems.push_back(
                problem(design_code::INVALID_RESPONSE, m_definition, "attributes",
                        base::concatenate({"expected a CompProp or CompPropGlobal element for ", responseTypeName(type),
                                           ", found ", entry->card, " ", std::to_string(entry->elementId), " of ",
                                           entry->propertyCard.empty() ? "no property" : entry->propertyCard})));
            continue;
        }
        std::vector<std::int32_t> codes;
        if (!usesName)
            codes.push_back(m_definition.attributeA.value_or(0));
        else
            for (std::int32_t code = 2; code <= LARGEST_ITEM_CODE; ++code)
            {
                const auto located = locateItem(type, entry->card, code);
                const ItemLocation* location = std::get_if<ItemLocation>(&located);
                if (location == nullptr)
                    continue;
                const std::vector<std::string> names = columnNames(*family, *location);
                const std::optional<std::size_t> column = location->columnIn(names);
                if (column && names[*column] == m_definition.componentName)
                    codes.push_back(code);
            }
        if (usesName && codes.empty())
            problems.push_back(
                problem(design_code::INVALID_RESPONSE, m_definition, "component",
                        base::concatenate({"expected a column of the ", responseTypeName(type), " results of ",
                                           entry->card, " ", std::to_string(entry->elementId), ", found '",
                                           m_definition.componentName, "', which its tables do not have"})));
        for (const std::int32_t code : codes)
        {
            auto located = locateItem(type, entry->card, code);
            if (const std::string* refusal = std::get_if<std::string>(&located))
            {
                problems.push_back(
                    problem(design_code::INVALID_RESPONSE, m_definition, "attributeA",
                            base::concatenate({*refusal, " (element ", std::to_string(entry->elementId), ")"})));
                continue;
            }
            target.items.emplace_back(code, std::get<ItemLocation>(located));
        }
        if (!target.items.empty())
            m_elements.push_back(std::move(target));
    }
}

void DesignResp1Response::addRequiredOutputs(
    std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT>& outputs) const
{
    const ResponseType type = m_definition.type;
    if (isGlobalType(type) || type == ResponseType::Compliance || isEigenType(type))
        return; // the compliance reads the external work, which every analysis gives; the roots are no output
    if (type == ResponseType::MeanDisplacement)
    {
        // The applied forces of every grid point (W = Σ|F_g|).
        output::OutputRequestPlan& request = outputs[static_cast<std::size_t>(output::OutputKind::AppliedLoads)];
        request.isRequested = true;
        request.isAll = true;
        request.members.clear();
        return;
    }
    std::vector<std::int32_t> ids = m_grids;
    for (const ElementTarget& target : m_elements)
        ids.push_back(target.elementId);
    addMembers(outputs[static_cast<std::size_t>(outputKindOf(type))], ids);
    if (type == ResponseType::CompositeStrengthRatio)
        addMembers(outputs[static_cast<std::size_t>(output::OutputKind::Stress)], ids); // the ply stresses
}

ElementFilter DesignResp1Response::massFilter() const
{
    if (!m_properties.empty())
        return [properties = m_properties](std::string_view, std::int32_t, std::int32_t propId)
        {
            return std::any_of(properties.begin(), properties.end(),
                               [propId](const auto& property) { return property.second == propId; });
        };
    return [](std::string_view card, std::int32_t, std::int32_t) { return !MassProperties::isConcentratedMass(card); };
}

base::Vec3 DesignResp1Response::weightReference(const input::ModelSnapshot& snapshot)
{
    // The reference point of WEIGHT: the grid point of PARAM GRDPNT, the origin when it is not a grid point.
    base::Vec3 reference{0.0, 0.0, 0.0};
    if (snapshot.hasTable(input::TableRole::Params))
        if (const std::optional<input::ParamValue> weightReferenceNode = snapshot.params().find("WeightReferenceNode");
            weightReferenceNode && weightReferenceNode->kind == input::ParamValue::Kind::Integer &&
            weightReferenceNode->integer != NO_WEIGHT_REFERENCE_NODE && weightReferenceNode->integer > 0)
        {
            const std::span<const std::int32_t> ids = snapshot.nodes().ids();
            const auto found = std::find(ids.begin(), ids.end(), weightReferenceNode->integer);
            if (found != ids.end())
            {
                const auto row = static_cast<std::int64_t>(found - ids.begin());
                const std::span<const double> position = snapshot.nodes().positionsInBasic()[row];
                reference = {position[0], position[1], position[2]};
            }
        }
    return reference;
}

double DesignResp1Response::massOfSelection(const input::ModelSnapshot& snapshot, ResponseContext& context) const
{
    return MassProperties::rigidBodyMass(snapshot, {0.0, 0.0, 0.0}, context.massFormulation, massFilter(),
                                         *context.budget)(0, 0);
}

double DesignResp1Response::elementItem(const ElementTarget& target, std::int32_t code, const ItemLocation& location,
                                        std::int32_t lamina, std::int32_t subcase, ResponseContext& context) const
{
    const auto missing = [&](const std::string& what)
    {
        throw base::OptimisationError(
            base::concatenate({"DesignResp1 ",
                               std::to_string(m_definition.id),
                               " (",
                               m_definition.label,
                               "): ",
                               what,
                               " for item ",
                               std::to_string(code),
                               " of ",
                               target.card,
                               " ",
                               std::to_string(target.elementId),
                               lamina != 0 ? ", lamina " + std::to_string(lamina) : std::string(),
                               " in subcase ",
                               std::to_string(subcase),
                               " (table ",
                               location.tableKind,
                               " of ",
                               target.elementType,
                               "); the analysis did not give it: check the element's property (a ply with "
                               "CompProp.plyStressOutput = NO has no "
                               "row) and the item code"}));
    };
    if (m_definition.type == ResponseType::StrainEnergy && code == STRAIN_ENERGY_DENSITY_ITEM)
    {
        // Item code 4, the element strain energy density: the energy over the element's volume (the
        // density column of O_o_Solver's strain energy table is empty).
        ItemLocation energy = location;
        energy.column = STRAIN_ENERGY_COLUMN;
        const std::optional<double> value =
            context.results->elementValue(energy, target.elementType, subcase, target.elementId, 0);
        if (!context.elementVolumes)
            context.elementVolumes =
                MassProperties::elementVolumes(*context.snapshot, context.snapshotOwner, {}, *context.budget);
        const auto volume = context.elementVolumes->find(target.elementId);
        if (!value || volume == context.elementVolumes->end() || !(volume->second > 0.0))
            missing("expected the strain energy and a positive volume of the element, found none");
        return *value / volume->second;
    }
    const std::optional<double> value =
        context.results->elementValue(location, target.elementType, subcase, target.elementId, location.layer + lamina);
    if (!value || !std::isfinite(*value))
        missing("expected a finite result value, found none");
    if (!location.isStrengthRatio)
        return *value;
    // CSTRAT: the strength ratio of the failure index column (StrengthRatio), from the ply stresses and the
    // allowables of the laminate of the element's property on the overlaid snapshot.
    if (location.column == BOND_FAILURE_INDEX_COLUMN)
        return StrengthRatio::ofBond(*value);
    const std::span<const double> stresses =
        context.results->elementRow("stress", target.elementType, subcase, target.elementId, ItemPlace::Centre, lamina);
    if (stresses.size() < 3)
        missing("expected the ply stresses of the strength ratio, found none");
    const elements::SnapshotIndex& index = context.snapshotIndex();
    const elements::ElementBatchView view(context.snapshot->elementBatches()[target.batch], index);
    const elements::Resolution<elements::ShellSection> section =
        elements::resolveShellSection(view, elements::ShellSectionColumns(view), target.row);
    if (!section || !section.value->isLaminate())
        missing("expected the laminate of the element's CompProp, found none");
    const elements::Laminate& laminate = *section.value->laminate;
    const auto ply =
        std::find_if(laminate.plies.begin(), laminate.plies.end(),
                     [lamina](const elements::LaminatePly& candidate) { return candidate.number == lamina; });
    if (ply == laminate.plies.end())
        missing("expected the lamina in the laminate, found none");
    return StrengthRatio::ofDirectStresses(laminate.failureTheory, ply->material,
                                           {stresses[0], stresses[1], stresses[2]}, *value);
}

double DesignResp1Response::selfAdjointValue(std::int32_t subcase, ResponseContext& context) const
{
    const auto refuse = [&](const std::string& what)
    {
        // The remedy given by each caller, in `what`.
        throw base::OptimisationError(base::concatenate({"DesignResp1 ", std::to_string(m_definition.id), " (",
                                                         m_definition.label, ", ", responseTypeName(m_definition.type),
                                                         ") in subcase ", std::to_string(subcase), ": ", what}));
    };
    if (context.checks.empty())
        refuse(
            "expected the checks of the analysis (its external work), found none; evaluate the response on an "
            "analysis of the design problem, which computes them");
    if (context.hasEnforcedDisplacements(subcase))
        refuse(
            "the subcase enforces displacements (SinglePointConstr D or EnforcedMotionConstr), so the work of its "
            "loads is not the "
            "compliance "
            "and the displacement is not its adjoint (design/aids/SelfAdjointResponses); constrain displacements "
            "instead, or remove the enforced values");
    const double compliance = SelfAdjointResponses::compliance(context.checks, subcase);
    if (m_definition.type == ResponseType::Compliance)
        return compliance;
    if (context.arrays == nullptr)
        refuse(
            "expected the applied loads of the analysis, found no result arrays; evaluate the response on an "
            "analysis of the design problem, which keeps them");
    const double magnitude =
        SelfAdjointResponses::forceMagnitude(SelfAdjointResponses::appliedForces(*context.arrays, subcase));
    if (!(magnitude > 0.0))
        refuse(
            "expected an applied force (PointForceLoad, PointsPressureLoad, GravityLoad...) in the subcase, found "
            "none: the "
            "mean displacement "
            "is weighted by the forces; use compliance");
    return compliance / magnitude;
}

double DesignResp1Response::pointValue(std::int32_t grid, std::int32_t subcase, ResponseContext& context) const
{
    const bool isDeflection = m_definition.type == ResponseType::DeflectionToThickness;
    std::array<double, 3> values{};
    for (std::int32_t c = 0; c < 3; ++c)
    {
        const std::int32_t component = isDeflection ? c + 1 : c + 4;
        const std::optional<double> value = context.results->nodalValue("displacements", subcase, grid, component);
        if (!value)
            throw base::OptimisationError(
                base::concatenate({"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label,
                                   "): expected the displacements of grid point ", std::to_string(grid), " in subcase ",
                                   std::to_string(subcase),
                                   ", found none in the results of the analysis; evaluate the response on an analysis "
                                   "of the design problem, which recovers the grid points of its responses"}));
        values[static_cast<std::size_t>(c)] = *value;
    }
    if (!isDeflection)
        return LinearityIndicator::rotationMagnitude(values);
    const ShellPointGeometry* point = context.linearityIndicator().findShellPoint(grid);
    if (point == nullptr)
        throw base::OptimisationError(base::concatenate(
            {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, "): grid point ",
             std::to_string(grid),
             " has no shell element of positive area and thickness around it at this design point; its deflection "
             "over thickness is not defined: leave the grid point out of nodes= or give its shells a thickness"}));
    return LinearityIndicator::deflectionRatio(*point, values);
}

void DesignResp1Response::evaluate(std::int32_t subcase, ResponseContext& context, ResponseValues& values) const
{
    const ResponseType type = m_definition.type;
    const auto append = [&values](ResponseItem item, double value)
    {
        values.items.push_back(item);
        values.values.push_back(value);
    };
    switch (type)
    {
    case ResponseType::Weight:
    {
        const base::Vec3 reference = weightReference(*context.snapshot);
        const base::MatN<6> mass =
            MassProperties::rigidBodyMass(*context.snapshot, reference, context.massFormulation, {}, *context.budget);
        append({0, 0, m_weightRow * 10 + m_weightColumn, 0},
               mass(static_cast<std::size_t>(m_weightRow - 1), static_cast<std::size_t>(m_weightColumn - 1)));
        return;
    }
    case ResponseType::Volume:
        append({0, 0, 0, 0}, MassProperties::volume(
                                 *context.snapshot, context.snapshotOwner,
                                 [](std::string_view, std::int32_t, std::int32_t) { return true; }, *context.budget));
        return;
    case ResponseType::FractionalMass:
    {
        if (context.initialSnapshot == nullptr)
            throw base::OptimisationError(base::concatenate(
                {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label,
                 "): FRMASS needs the mass at the initial design, which this evaluation was not given; evaluate it "
                 "through the design problem, which keeps the initial design"}));
        const double initial = massOfSelection(*context.initialSnapshot, context);
        if (initial <= 0.0)
            throw base::OptimisationError(base::concatenate(
                {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label,
                 "): expected a positive mass of the selected elements at the initial design (the denominator of "
                 "FRMASS), found ",
                 std::to_string(initial), "; select elements with a density"}));
        append({0, 0, 0, 0}, massOfSelection(*context.snapshot, context) / initial);
        return;
    }
    case ResponseType::Displacement:
    case ResponseType::SpcForce:
    {
        const char* kind = type == ResponseType::Displacement ? "displacements" : "singlePointConstrForces";
        for (const std::int32_t grid : m_grids)
            for (const std::int32_t component : m_components)
            {
                const std::optional<double> value = context.results->nodalValue(kind, subcase, grid, component);
                if (!value)
                    throw base::OptimisationError(base::concatenate(
                        {"DesignResp1 ", std::to_string(m_definition.id), " (", m_definition.label, "): expected the ",
                         kind, " of grid point ", std::to_string(grid), " in subcase ", std::to_string(subcase),
                         ", found none in the results of the analysis; evaluate the response on an analysis of the "
                         "design problem, which recovers the grid points of its responses"}));
                append({subcase, grid, component, 0}, *value);
            }
        return;
    }
    case ResponseType::Compliance:
    case ResponseType::MeanDisplacement:
        append({subcase, 0, 0, 0}, selfAdjointValue(subcase, context));
        return;
    case ResponseType::DeflectionToThickness:
    case ResponseType::Rotation:
        for (const std::int32_t grid : m_grids)
            append({subcase, grid, 0, 0}, pointValue(grid, subcase, context));
        return;
    case ResponseType::SectionLoad:
    case ResponseType::CutLoad:
        sectionValues(subcase, context, values);
        return;
    case ResponseType::ContourDisplacement:
    case ResponseType::ContourForce:
        zoneValues(subcase, context, values);
        return;
    case ResponseType::Eigenvalue:
    case ResponseType::Frequency:
    case ResponseType::BucklingFactor:
        eigenValues(subcase, context, values);
        return;
    case ResponseType::TrimVariable:
    case ResponseType::StabilityDerivative:
        aeroelasticValues(subcase, context, values);
        return;
    default:
        break;
    }
    const std::vector<std::int32_t> laminae = isComposite(type) ? m_plies : std::vector<std::int32_t>{0};
    for (const ElementTarget& target : m_elements)
        for (const std::int32_t lamina : laminae)
            for (const auto& [code, location] : target.items)
                append({subcase, target.elementId, code, lamina},
                       elementItem(target, code, location, lamina, subcase, context));
}

} // namespace oos::design
