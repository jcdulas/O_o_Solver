#include "oos/input/snapshot/SnapshotBuilder.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/ContractVersion.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::input
{

namespace
{

void requireNewCard(const std::vector<std::string>& cards, const std::string& card, std::string_view kind)
{
    for (const std::string& existing : cards)
    {
        if (existing == card)
            throw std::invalid_argument(base::concatenate(
                {"SnapshotBuilder: the ", kind, " table of ", card, " was already given; give each card type once"}));
    }
}

template <typename Entry>
std::vector<std::string> cardsOf(const std::vector<Entry>& entries)
{
    std::vector<std::string> cards;
    for (const Entry& entry : entries)
        cards.push_back(entry.card);
    return cards;
}

// The card of the enforced displacements (SPCD), whose rows singlePointConstrTable holds with the constraints.
constexpr std::string_view ENFORCED_DISPLACEMENT_CARD = "EnforcedMotionConstr";

// With loadCombinationsTable the SPCD rows of singlePointConstrTable list each load set once, as the load tables do: an
// SPCD row whose set is not its card's (a combination expanded) would be added twice, under its LOAD and through the
// set the LOAD reaches (SpcSet reads the SPCD of a LOAD by its terms, LoadTable::termsOf).
void requireOwnEnforcedRows(const ConstraintTable& constraints)
{
    const base::ContractReader& table = constraints.spcReader();
    const base::DictionaryView cards = constraints.spcCards();
    const std::span<const std::int32_t> setIds = table.values<std::int32_t>("setId", base::NullPolicy::Allow);
    const std::span<const std::int32_t> sources = table.values<std::int32_t>("sourceSetId", base::NullPolicy::Allow);
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        if (cards[row] != ENFORCED_DISPLACEMENT_CARD || setIds[index] == sources[index])
            continue;
        throw base::ContractError(base::concatenate(
            {"model snapshot: expected the enforced displacement rows of singlePointConstrTable to list each load set "
             "once with "
             "loadCombinationsTable (O_o_Reader's expandCombinations=False), found row ",
             std::to_string(row), " of set ", std::to_string(setIds[index]), " whose card belongs to set ",
             std::to_string(sources[index]), "; take singlePointConstrTable with expandCombinations=False too"}));
    }
}

// The column that keys the material card tables (IsoMat, OrthoMat2D...) and the load card tables (PointForceLoad...).
constexpr std::string_view MATERIAL_ID_COLUMN = "id";
constexpr std::string_view LOAD_SET_ID_COLUMN = "id";

} // namespace

void SnapshotBuilder::setVersions(std::int64_t structureVersion, std::int64_t contentVersion)
{
    m_versions.structureVersion = structureVersion;
    m_versions.contentVersion = contentVersion;
}

void SnapshotBuilder::checkVersion(const base::ImportedTable& table, const std::string& name)
{
    const std::int64_t version = requireAcceptedTableVersion(table, name);
    if (version == 0)
        return;
    if (m_versions.dataContractVersion != 0 && version != m_versions.dataContractVersion)
        throw base::ContractError(
            base::concatenate({"table '", name, "': expected the data_contract_version ",
                               std::to_string(m_versions.dataContractVersion), " of the tables given before, found ",
                               std::to_string(version), "; take every table of a snapshot from the same O_o_Reader"}));
    m_versions.dataContractVersion = version;
}

void SnapshotBuilder::setTable(TableRole role, base::ImportedTable table)
{
    if (m_tables.contains(role))
        throw std::invalid_argument(base::concatenate(
            {"SnapshotBuilder: the table '", tableName(role), "' was already given; give each table once"}));
    checkVersion(table, std::string(tableName(role)));
    m_tables.emplace(role, std::move(table));
}

void SnapshotBuilder::setOffsetDefinitionValue(std::string_view value)
{
    if (const std::optional<OffsetDefinition> definition = offsetDefinitionOf(value))
    {
        setOffsetDefinition(*definition);
        return;
    }
    m_offsetDefinition = OffsetDefinition::ElementOffsets;
    m_unsupportedOffsetDefinition = std::string(value);
}

void SnapshotBuilder::setElementFormulation(std::string_view card, std::string_view formulation)
{
    if (card.empty() || formulation.empty())
        throw std::invalid_argument(base::concatenate(
            {"SnapshotBuilder: expected an element card and a formulation name, found the card '", card,
             "' and the name '", formulation, "'; give both (\"ShellQuad4\", \"ClassicFormulation\")"}));
    // Keep the exact card name; the element registry validates that the solver supports it.
    ElementFormulationChoice choice{std::string(card), std::string(formulation)};
    const auto place = std::lower_bound(m_elementFormulations.begin(), m_elementFormulations.end(), choice,
                                        [](const ElementFormulationChoice& a, const ElementFormulationChoice& b)
                                        { return a.card < b.card; });
    if (place != m_elementFormulations.end() && place->card == choice.card)
        place->formulation = std::move(choice.formulation);
    else
        m_elementFormulations.insert(place, std::move(choice));
}

void SnapshotBuilder::setCdRotations(BorrowedArray<double> rotations)
{
    m_cdRotations = std::move(rotations);
}

void SnapshotBuilder::addElementBatch(std::string card, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_elementTables), card, "element");
    checkVersion(table, card);
    m_elementTables.push_back({std::move(card), "", std::move(table)});
}

void SnapshotBuilder::addPropertyTable(std::string card, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_propertyTables), card, "property");
    checkVersion(table, card);
    std::string idColumn(PROPERTY_ID_COLUMN);
    m_propertyTables.push_back({std::move(card), std::move(idColumn), std::move(table)});
}

void SnapshotBuilder::addMethodTable(std::string card, std::string idColumn, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_methodTables), card, "method");
    checkVersion(table, card);
    m_methodTables.push_back({std::move(card), std::move(idColumn), std::move(table)});
}

void SnapshotBuilder::addMaterialCardTable(std::string card, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_materialCardTables), card, "material");
    checkVersion(table, card);
    m_materialCardTables.push_back({std::move(card), std::string(MATERIAL_ID_COLUMN), std::move(table)});
}

void SnapshotBuilder::addLoadCardTable(std::string card, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_loadCardTables), card, "load");
    checkVersion(table, card);
    m_loadCardTables.push_back({std::move(card), std::string(LOAD_SET_ID_COLUMN), std::move(table)});
}

void SnapshotBuilder::addAeroCardTable(std::string card, std::string idColumn, base::ImportedTable table)
{
    requireNewCard(cardsOf(m_aeroCardTables), card, "aeroelastic");
    checkVersion(table, card);
    m_aeroCardTables.push_back({std::move(card), std::move(idColumn), std::move(table)});
}

std::optional<base::ImportedTable> SnapshotBuilder::take(TableRole role)
{
    const auto found = m_tables.find(role);
    if (found == m_tables.end())
        return std::nullopt;
    base::ImportedTable table = std::move(found->second);
    m_tables.erase(found);
    return table;
}

void SnapshotBuilder::checkGroup(std::initializer_list<TableRole> roles) const
{
    std::string present;
    std::string missing;
    for (const TableRole role : roles)
    {
        std::string& list = m_tables.contains(role) ? present : missing;
        list.append(list.empty() ? "" : ", ").append(tableName(role));
    }
    if (!present.empty() && !missing.empty())
        throw base::ContractError(
            base::concatenate({"model snapshot: the tables [", present, "] were given without [", missing,
                               "], which belong with them; give the whole group or none of it"}));
}

void SnapshotBuilder::checkPresence() const
{
    if (!m_tables.contains(TableRole::Nodes))
        throw base::ContractError(
            "model snapshot: expected the node table 'nodes' (cardTable(deck, \"nodes\") of O_o_Reader), found none; "
            "every snapshot needs it: hand it over with the other tables");
    checkGroup({TableRole::Spc, TableRole::Mpc});
    if (m_tables.contains(TableRole::Pressures) && m_tables.contains(TableRole::PressureFields))
        throw base::ContractError(
            "model snapshot: expected the pressures in one form, found both 'pressureTable' and 'pressureFields'; "
            "give pressureTable(detail=\"fields\") or its rows, not both");
    checkGroup({TableRole::Loads,
                m_tables.contains(TableRole::PressureFields) ? TableRole::PressureFields : TableRole::Pressures,
                TableRole::LineLoads, TableRole::Temperatures});
    if (m_tables.contains(TableRole::InertialLoads) && !m_tables.contains(TableRole::Loads))
        throw base::ContractError(
            "model snapshot: the table 'inertialLoadsTable' was given without the static load tables "
            "(loadsTable, pressureTable, lineLoadsTable, temperatureTable); give them together");
    if (m_tables.contains(TableRole::LoadCombinations) && !m_tables.contains(TableRole::Loads))
        throw base::ContractError(
            "model snapshot: the table 'loadCombinationsTable' was given without the static load tables "
            "(loadsTable, pressureTable, lineLoadsTable, temperatureTable); give them together");
    checkGroup({TableRole::Components, TableRole::ComponentElements});
    if ((m_tables.contains(TableRole::Boundaries) || m_tables.contains(TableRole::LoadGroups)) &&
        !m_tables.contains(TableRole::Components))
        throw base::ContractError(
            "model snapshot: component boundaries or load case groups were given without the components "
            "(connectedComponents and its elements); give them together");
    checkGroup({TableRole::OutputRequests, TableRole::CaseSets});
    checkGroup({TableRole::ContactNodePairs, TableRole::ContactPairs});
    if (m_tables.contains(TableRole::ContactTables) && !m_tables.contains(TableRole::ContactPairs))
        throw base::ContractError(
            "model snapshot: the table 'contactTablesTable' was given without the contact pairs "
            "(contactNodePairs, contactPairsTable); give them together");
}

std::optional<std::vector<base::ImportedTable>> SnapshotBuilder::takeGroup(std::initializer_list<TableRole> roles)
{
    if (!m_tables.contains(*roles.begin()))
        return std::nullopt; // checkPresence: the group is all absent
    std::vector<base::ImportedTable> tables;
    for (const TableRole role : roles)
        tables.push_back(std::move(*take(role)));
    return tables;
}

ModelSnapshot SnapshotBuilder::build()
{
    checkPresence();
    try
    {
        ModelSnapshot snapshot = takeAll();
        *this = SnapshotBuilder();
        return snapshot;
    }
    catch (...)
    {
        // A table refused by its contract: the builder is left empty rather than half emptied.
        *this = SnapshotBuilder();
        throw;
    }
}

ModelSnapshot SnapshotBuilder::takeAll()
{
    ModelSnapshot::Parts parts;
    parts.versions = m_versions;
    parts.sourceCards = std::move(m_sourceCards);
    parts.offsetDefinition = m_offsetDefinition;
    parts.unsupportedOffsetDefinition = m_unsupportedOffsetDefinition;
    parts.elementFormulations = std::move(m_elementFormulations);

    std::optional<base::ImportedTable> nodes = take(TableRole::Nodes);
    parts.nodes.emplace(std::move(*nodes), take(TableRole::ScalarPoints), std::move(m_cdRotations));

    for (CardEntry& entry : m_elementTables)
        parts.elementBatches.emplace_back(std::move(entry.card), std::move(entry.table));
    for (CardEntry& entry : m_propertyTables)
        parts.propertyTables.emplace_back(std::move(entry.card), std::move(entry.idColumn), std::move(entry.table));
    for (CardEntry& entry : m_methodTables)
        parts.methodTables.emplace_back(std::move(entry.card), std::move(entry.idColumn), std::move(entry.table));
    for (CardEntry& entry : m_materialCardTables)
        parts.materialCardTables.emplace_back(std::move(entry.card), std::move(entry.idColumn), std::move(entry.table));
    for (CardEntry& entry : m_loadCardTables)
        parts.loadCardTables.emplace_back(std::move(entry.card), std::move(entry.idColumn), std::move(entry.table));
    for (CardEntry& entry : m_aeroCardTables)
        parts.aeroCardTables.emplace_back(std::move(entry.card), std::move(entry.idColumn), std::move(entry.table));

    if (auto materials = take(TableRole::Materials))
        parts.materials.emplace(std::move(*materials));
    if (auto constraints = takeGroup({TableRole::Spc, TableRole::Mpc}))
        parts.constraints.emplace(std::move((*constraints)[0]), std::move((*constraints)[1]));
    if (auto rigid = take(TableRole::RigidElements))
        parts.rigidElements.emplace(std::move(*rigid));
    const bool isPressureFields = m_tables.contains(TableRole::PressureFields);
    if (auto loads = takeGroup({TableRole::Loads, isPressureFields ? TableRole::PressureFields : TableRole::Pressures,
                                TableRole::LineLoads, TableRole::Temperatures}))
        parts.loads.emplace(std::move((*loads)[0]), std::move((*loads)[1]), std::move((*loads)[2]),
                            std::move((*loads)[3]), take(TableRole::InertialLoads), take(TableRole::LoadCombinations),
                            isPressureFields);
    if (parts.loads && parts.loads->hasCombinationTerms() && parts.constraints)
        requireOwnEnforcedRows(*parts.constraints);
    if (auto subcases = take(TableRole::Subcases))
        parts.subcases.emplace(std::move(*subcases));
    if (auto params = take(TableRole::Params))
        parts.params.emplace(std::move(*params));
    if (auto units = takeGroup({TableRole::Components, TableRole::ComponentElements}))
        parts.units.emplace(std::move((*units)[0]), std::move((*units)[1]), take(TableRole::Boundaries),
                            take(TableRole::LoadGroups));
    if (auto masses = take(TableRole::NonStructuralMass))
        parts.masses.emplace(std::move(*masses));
    if (auto requests = takeGroup({TableRole::OutputRequests, TableRole::CaseSets}))
        parts.outputRequests.emplace(std::move((*requests)[0]), std::move((*requests)[1]));
    if (auto systems = take(TableRole::CoordinateSystems))
        parts.coordinateSystems.emplace(std::move(*systems));
    if (auto plies = take(TableRole::LaminatePlies))
        parts.laminatePlies.emplace(readTable(TableRole::LaminatePlies, std::move(*plies)));
    if (auto summary = take(TableRole::LaminateSummary))
        parts.laminateSummary.emplace(readTable(TableRole::LaminateSummary, std::move(*summary)));
    if (auto sections = take(TableRole::Sections))
        parts.sections.emplace(std::move(*sections));
    if (auto piercings = take(TableRole::FastenerPiercings))
        parts.fastenerPiercings.emplace(std::move(*piercings));
    if (auto supports = take(TableRole::Supports))
        parts.supports.emplace(readTable(TableRole::Supports, std::move(*supports)));
    if (auto nodes = take(TableRole::ConnectionNodes))
        parts.connectionNodes.emplace(readTable(TableRole::ConnectionNodes, std::move(*nodes)));
    if (auto clusters = take(TableRole::ConnectionClusters))
        parts.connectionClusters.emplace(readTable(TableRole::ConnectionClusters, std::move(*clusters)));
    if (auto spiders = take(TableRole::Spiders))
        parts.spiders.emplace(readTable(TableRole::Spiders, std::move(*spiders)));
    if (auto sharing = take(TableRole::ConnectionSharing))
        parts.connectionSharing.emplace(readTable(TableRole::ConnectionSharing, std::move(*sharing)));
    if (auto deformations = take(TableRole::Deformations))
        parts.deformations.emplace(readTable(TableRole::Deformations, std::move(*deformations)));
    for (std::size_t i = 0; i < AERO_TABLE_COUNT; ++i)
        if (auto table = take(AERO_TABLE_ROLES[i]))
            parts.aero[i].emplace(readTable(AERO_TABLE_ROLES[i], std::move(*table)));
    if (auto contact = takeGroup({TableRole::ContactNodePairs, TableRole::ContactPairs}))
        parts.contact.emplace(std::move((*contact)[0]), std::move((*contact)[1]), take(TableRole::ContactTables));
    return ModelSnapshot(std::move(parts));
}

} // namespace oos::input
