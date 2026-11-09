#include "oos/input/snapshot/ModelSnapshot.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::input
{

namespace
{

template <typename Table>
const Table* findCard(const std::vector<Table>& tables, std::string_view card) noexcept
{
    for (const Table& table : tables)
    {
        if (table.card() == card)
            return &table;
    }
    return nullptr;
}

// The place of an aeroelastic table in ModelSnapshot::m_aero, or AERO_TABLE_COUNT for another role.
std::size_t aeroIndexOf(TableRole role) noexcept
{
    for (std::size_t i = 0; i < AERO_TABLE_COUNT; ++i)
        if (AERO_TABLE_ROLES[i] == role)
            return i;
    return AERO_TABLE_COUNT;
}

template <typename Table>
const Table& require(const std::optional<Table>& table, TableRole role)
{
    if (!table)
        throw missingTableError(role);
    return *table;
}

constexpr std::string_view ELEMENTS_PREFIX = "elements/";
constexpr std::string_view PROPERTIES_PREFIX = "properties/";
constexpr std::string_view METHODS_PREFIX = "methods/";
constexpr std::string_view MATERIALS_PREFIX = "materials/";
constexpr std::string_view LOAD_CARDS_PREFIX = "loadCards/";
constexpr std::string_view AERO_CARDS_PREFIX = "aeroCards/";

} // namespace

ModelSnapshot::ModelSnapshot(Parts parts)
    : m_versions(parts.versions),
      m_sourceCards(std::move(parts.sourceCards)),
      m_offsetDefinition(parts.offsetDefinition),
      m_unsupportedOffsetDefinition(std::move(parts.unsupportedOffsetDefinition)),
      m_elementFormulations(std::move(parts.elementFormulations)),
      m_nodes(std::move(*parts.nodes)),
      m_elementBatches(std::move(parts.elementBatches)),
      m_propertyTables(std::move(parts.propertyTables)),
      m_methodTables(std::move(parts.methodTables)),
      m_materialCardTables(std::move(parts.materialCardTables)),
      m_loadCardTables(std::move(parts.loadCardTables)),
      m_aeroCardTables(std::move(parts.aeroCardTables)),
      m_materials(std::move(parts.materials)),
      m_constraints(std::move(parts.constraints)),
      m_rigidElements(std::move(parts.rigidElements)),
      m_loads(std::move(parts.loads)),
      m_subcases(std::move(parts.subcases)),
      m_params(std::move(parts.params)),
      m_units(std::move(parts.units)),
      m_masses(std::move(parts.masses)),
      m_outputRequests(std::move(parts.outputRequests)),
      m_coordinateSystems(std::move(parts.coordinateSystems)),
      m_laminatePlies(std::move(parts.laminatePlies)),
      m_laminateSummary(std::move(parts.laminateSummary)),
      m_sections(std::move(parts.sections)),
      m_fastenerPiercings(std::move(parts.fastenerPiercings)),
      m_supports(std::move(parts.supports)),
      m_connectionNodes(std::move(parts.connectionNodes)),
      m_connectionClusters(std::move(parts.connectionClusters)),
      m_contact(std::move(parts.contact)),
      m_spiders(std::move(parts.spiders)),
      m_connectionSharing(std::move(parts.connectionSharing)),
      m_deformations(std::move(parts.deformations)),
      m_aero(std::move(parts.aero))
{
}

const ElementBatch* ModelSnapshot::findElementBatch(std::string_view card) const noexcept
{
    return findCard(m_elementBatches, card);
}

const CardTable* ModelSnapshot::findPropertyTable(std::string_view card) const noexcept
{
    return findCard(m_propertyTables, card);
}

const CardTable* ModelSnapshot::findMethodTable(std::string_view card) const noexcept
{
    return findCard(m_methodTables, card);
}

const CardTable* ModelSnapshot::findMaterialCardTable(std::string_view card) const noexcept
{
    return findCard(m_materialCardTables, card);
}

const CardTable* ModelSnapshot::findAeroCardTable(std::string_view card) const noexcept
{
    return findCard(m_aeroCardTables, card);
}

bool ModelSnapshot::hasTable(TableRole role) const noexcept
{
    switch (role)
    {
    case TableRole::Nodes:
        return true;
    case TableRole::ScalarPoints:
        return m_nodes.scalarPointReader() != nullptr;
    case TableRole::Materials:
        return m_materials.has_value();
    case TableRole::Spc:
    case TableRole::Mpc:
        return m_constraints.has_value();
    case TableRole::RigidElements:
        return m_rigidElements.has_value();
    case TableRole::Loads:
    case TableRole::LineLoads:
    case TableRole::Temperatures:
        return m_loads.has_value();
    case TableRole::Pressures:
        return m_loads && !m_loads->isPressureFields();
    case TableRole::PressureFields:
        return m_loads && m_loads->isPressureFields();
    case TableRole::InertialLoads:
        return m_loads && m_loads->hasInertialLoads();
    case TableRole::LoadCombinations:
        return m_loads && m_loads->hasCombinationTerms();
    case TableRole::Subcases:
        return m_subcases.has_value();
    case TableRole::Params:
        return m_params.has_value();
    case TableRole::Components:
    case TableRole::ComponentElements:
        return m_units.has_value();
    case TableRole::Boundaries:
        return m_units && m_units->hasBoundaries();
    case TableRole::LoadGroups:
        return m_units && m_units->hasLoadGroups();
    case TableRole::NonStructuralMass:
        return m_masses.has_value();
    case TableRole::OutputRequests:
    case TableRole::CaseSets:
        return m_outputRequests.has_value();
    case TableRole::CoordinateSystems:
        return m_coordinateSystems.has_value();
    case TableRole::LaminatePlies:
        return m_laminatePlies.has_value();
    case TableRole::LaminateSummary:
        return m_laminateSummary.has_value();
    case TableRole::Sections:
        return m_sections.has_value();
    case TableRole::FastenerPiercings:
        return m_fastenerPiercings.has_value();
    case TableRole::Supports:
        return m_supports.has_value();
    case TableRole::ConnectionNodes:
        return m_connectionNodes.has_value();
    case TableRole::ConnectionClusters:
        return m_connectionClusters.has_value();
    case TableRole::ContactNodePairs:
    case TableRole::ContactPairs:
        return m_contact.has_value();
    case TableRole::ContactTables:
        return m_contact && m_contact->hasContactTables();
    case TableRole::Spiders:
        return m_spiders.has_value();
    case TableRole::ConnectionSharing:
        return m_connectionSharing.has_value();
    case TableRole::Deformations:
        return m_deformations.has_value();
    case TableRole::AeroReference:
    case TableRole::AeroPanels:
    case TableRole::AeroBoxes:
    case TableRole::ControlSurfaces:
    case TableRole::TrimCases:
    case TableRole::Splines:
    case TableRole::SplineNodes:
    case TableRole::DirectMatrixShapes:
    case TableRole::DirectMatrices:
        return m_aero[aeroIndexOf(role)].has_value();
    }
    return false;
}

const MaterialTable& ModelSnapshot::materials() const
{
    return require(m_materials, TableRole::Materials);
}

const ConstraintTable& ModelSnapshot::constraints() const
{
    return require(m_constraints, TableRole::Spc);
}

const RigidTable& ModelSnapshot::rigidElements() const
{
    return require(m_rigidElements, TableRole::RigidElements);
}

const LoadTable& ModelSnapshot::loads() const
{
    return require(m_loads, TableRole::Loads);
}

const SubcaseTable& ModelSnapshot::subcases() const
{
    return require(m_subcases, TableRole::Subcases);
}

const ParamTable& ModelSnapshot::params() const
{
    return require(m_params, TableRole::Params);
}

const UnitTable& ModelSnapshot::units() const
{
    return require(m_units, TableRole::Components);
}

const MassTable& ModelSnapshot::masses() const
{
    return require(m_masses, TableRole::NonStructuralMass);
}

const OutputRequests& ModelSnapshot::outputRequests() const
{
    return require(m_outputRequests, TableRole::OutputRequests);
}

const CoordinateSystemTable& ModelSnapshot::coordinateSystems() const
{
    return require(m_coordinateSystems, TableRole::CoordinateSystems);
}

const base::ContractReader& ModelSnapshot::laminatePlies() const
{
    return require(m_laminatePlies, TableRole::LaminatePlies);
}

const base::ContractReader& ModelSnapshot::laminateSummary() const
{
    return require(m_laminateSummary, TableRole::LaminateSummary);
}

const SectionTable& ModelSnapshot::sections() const
{
    return require(m_sections, TableRole::Sections);
}

const FastenerPiercingTable& ModelSnapshot::fastenerPiercings() const
{
    return require(m_fastenerPiercings, TableRole::FastenerPiercings);
}

const base::ContractReader& ModelSnapshot::supports() const
{
    return require(m_supports, TableRole::Supports);
}

const base::ContractReader& ModelSnapshot::connectionNodes() const
{
    return require(m_connectionNodes, TableRole::ConnectionNodes);
}

const base::ContractReader& ModelSnapshot::connectionClusters() const
{
    return require(m_connectionClusters, TableRole::ConnectionClusters);
}

const ContactTable& ModelSnapshot::contact() const
{
    return require(m_contact, TableRole::ContactNodePairs);
}

const base::ContractReader& ModelSnapshot::spiders() const
{
    return require(m_spiders, TableRole::Spiders);
}

const base::ContractReader& ModelSnapshot::connectionSharing() const
{
    return require(m_connectionSharing, TableRole::ConnectionSharing);
}

const base::ContractReader& ModelSnapshot::deformations() const
{
    return require(m_deformations, TableRole::Deformations);
}

const base::ContractReader& ModelSnapshot::aeroTable(TableRole role) const
{
    const std::size_t index = aeroIndexOf(role);
    if (index == AERO_TABLE_COUNT)
        throw std::invalid_argument(
            base::concatenate({"ModelSnapshot::aeroTable: expected a table of the aeroelastic "
                               "model, found the role '",
                               tableName(role), "'"}));
    return require(m_aero[index], role);
}

const base::ContractReader* ModelSnapshot::findTable(std::string_view name) const
{
    if (name.starts_with(ELEMENTS_PREFIX))
    {
        const ElementBatch* batch = findElementBatch(name.substr(ELEMENTS_PREFIX.size()));
        return batch ? &batch->reader() : nullptr;
    }
    if (name.starts_with(PROPERTIES_PREFIX))
    {
        const CardTable* table = findPropertyTable(name.substr(PROPERTIES_PREFIX.size()));
        return table ? &table->reader() : nullptr;
    }
    if (name.starts_with(METHODS_PREFIX))
    {
        const CardTable* table = findMethodTable(name.substr(METHODS_PREFIX.size()));
        return table ? &table->reader() : nullptr;
    }
    if (name.starts_with(MATERIALS_PREFIX))
    {
        const CardTable* table = findMaterialCardTable(name.substr(MATERIALS_PREFIX.size()));
        return table ? &table->reader() : nullptr;
    }
    if (name.starts_with(AERO_CARDS_PREFIX))
    {
        const CardTable* table = findAeroCardTable(name.substr(AERO_CARDS_PREFIX.size()));
        return table ? &table->reader() : nullptr;
    }
    if (name.starts_with(LOAD_CARDS_PREFIX))
    {
        const CardTable* table = findCard(m_loadCardTables, name.substr(LOAD_CARDS_PREFIX.size()));
        return table ? &table->reader() : nullptr;
    }
    const std::optional<TableRole> role = findTableRole(name);
    if (!role || !hasTable(*role))
        return nullptr;
    switch (*role)
    {
    case TableRole::Nodes:
        return &m_nodes.reader();
    case TableRole::ScalarPoints:
        return m_nodes.scalarPointReader();
    case TableRole::Materials:
        return &m_materials->reader();
    case TableRole::Spc:
        return &m_constraints->spcReader();
    case TableRole::Mpc:
        return &m_constraints->mpcReader();
    case TableRole::RigidElements:
        return &m_rigidElements->reader();
    case TableRole::Loads:
        return &m_loads->loads();
    case TableRole::Pressures:
    case TableRole::PressureFields:
        return &m_loads->pressures();
    case TableRole::LineLoads:
        return &m_loads->lineLoads();
    case TableRole::Temperatures:
        return &m_loads->temperatures();
    case TableRole::InertialLoads:
        return &m_loads->inertialLoads();
    case TableRole::LoadCombinations:
        return &m_loads->combinations();
    case TableRole::Subcases:
        return &m_subcases->reader();
    case TableRole::Params:
        return &m_params->reader();
    case TableRole::Components:
        return &m_units->components();
    case TableRole::ComponentElements:
        return &m_units->componentElements();
    case TableRole::Boundaries:
        return &m_units->boundaries();
    case TableRole::LoadGroups:
        return &m_units->loadGroups();
    case TableRole::NonStructuralMass:
        return &m_masses->reader();
    case TableRole::OutputRequests:
        return &m_outputRequests->requestReader();
    case TableRole::CaseSets:
        return &m_outputRequests->caseSetReader();
    case TableRole::CoordinateSystems:
        return &m_coordinateSystems->reader();
    case TableRole::LaminatePlies:
        return &*m_laminatePlies;
    case TableRole::LaminateSummary:
        return &*m_laminateSummary;
    case TableRole::Sections:
        return &m_sections->reader();
    case TableRole::FastenerPiercings:
        return &m_fastenerPiercings->reader();
    case TableRole::Supports:
        return &*m_supports;
    case TableRole::ConnectionNodes:
        return &*m_connectionNodes;
    case TableRole::ConnectionClusters:
        return &*m_connectionClusters;
    case TableRole::ContactNodePairs:
        return &m_contact->nodePairReader();
    case TableRole::ContactPairs:
        return &m_contact->pairReader();
    case TableRole::ContactTables:
        return m_contact->tableReader();
    case TableRole::Spiders:
        return &*m_spiders;
    case TableRole::ConnectionSharing:
        return &*m_connectionSharing;
    case TableRole::Deformations:
        return &*m_deformations;
    case TableRole::AeroReference:
    case TableRole::AeroPanels:
    case TableRole::AeroBoxes:
    case TableRole::ControlSurfaces:
    case TableRole::TrimCases:
    case TableRole::Splines:
    case TableRole::SplineNodes:
    case TableRole::DirectMatrixShapes:
    case TableRole::DirectMatrices:
        return &*m_aero[aeroIndexOf(*role)];
    }
    return nullptr;
}

std::vector<LoadTerm> ModelSnapshot::loadTermsOf(std::optional<std::int32_t> loadSetId) const
{
    if (!loadSetId)
        return {};
    if (!m_loads)
        return {LoadTerm{*loadSetId, 1.0}};
    return m_loads->termsOf(*loadSetId);
}

std::vector<std::string> ModelSnapshot::tableNames() const
{
    std::vector<std::string> names;
    for (const TableRole role : ALL_TABLE_ROLES)
    {
        if (hasTable(role))
            names.emplace_back(tableName(role));
    }
    for (const ElementBatch& batch : m_elementBatches)
        names.push_back(std::string(ELEMENTS_PREFIX).append(batch.card()));
    for (const CardTable& table : m_propertyTables)
        names.push_back(std::string(PROPERTIES_PREFIX).append(table.card()));
    for (const CardTable& table : m_methodTables)
        names.push_back(std::string(METHODS_PREFIX).append(table.card()));
    for (const CardTable& table : m_materialCardTables)
        names.push_back(std::string(MATERIALS_PREFIX).append(table.card()));
    for (const CardTable& table : m_loadCardTables)
        names.push_back(std::string(LOAD_CARDS_PREFIX).append(table.card()));
    for (const CardTable& table : m_aeroCardTables)
        names.push_back(std::string(AERO_CARDS_PREFIX).append(table.card()));
    return names;
}

} // namespace oos::input
