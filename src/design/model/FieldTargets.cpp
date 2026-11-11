#include "oos/design/model/FieldTargets.hpp"

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DerivedProperties.hpp"
#include "oos/design/model/DesignCodes.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/tables/NestedListView.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace oos::design
{

namespace
{

// The property cards whose fields the elements do not read from the property table itself but from a table
// O_o_Reader derives from it (laminatePliesTable for the composites, sectionsTable for the library sections): a
// design point overlays those tables too (DerivedProperties).
constexpr std::array<std::string_view, 4> DERIVED_PROPERTY_CARDS{"CompProp", "CompPropGlobal", "BarShapeProp",
                                                                 "BeamShapeProp"};

const std::string MATERIALS_TABLE = "materialsTable";
// The column of the node table that holds the positions in the basic system (NodeTable::positionsInBasic).
const std::string NODE_POSITION_COLUMN = "xyzGlobal";

// The material summary uses compact engineering symbols; relations from card fields use their explicit names.
// Overlay both representations, since rods and beams read the summary while shells also read material card tables.
[[nodiscard]] std::string materialSummaryField(const std::string& field)
{
    static constexpr std::pair<const char*, const char*> FIELDS[] = {
        {"youngsModulus", "e"},         {"shearModulus", "g"},      {"poissonRatio", "nu"},
        {"thermalExpansion", "a"},      {"tensionAllowable", "st"}, {"compressionAllowable", "sc"},
        {"shearAllowable", "ss"},       {"youngsModulus1", "e1"},   {"youngsModulus2", "e2"},
        {"poissonRatio12", "nu12"},     {"shearModulus12", "g12"},  {"tensionStrength1", "xt"},
        {"compressionStrength1", "xc"}, {"tensionStrength2", "yt"}, {"compressionStrength2", "yc"},
        {"shearStrength", "s"}};
    for (const auto& [name, summary] : FIELDS)
        if (field == name)
            return summary;
    return field;
}

[[nodiscard]] bool isMaterialIdColumn(const std::string& name)
{
    // matId, matId1 ... and the materials a shell names by their role (membraneMatId, bendingMatId, ...); not
    // matFrameId, a coordinate system.
    return name == "matId" || name.ends_with("MatId") ||
           (name.size() == 6 && name.starts_with("matId") && name[5] >= '1' && name[5] <= '9');
}

// The row of an ID in an int32 column, or -1.
[[nodiscard]] std::int64_t rowOfId(std::span<const std::int32_t> ids, std::int32_t id)
{
    const auto sorted = std::lower_bound(ids.begin(), ids.end(), id);
    auto found = sorted != ids.end() && *sorted == id ? sorted : std::find(ids.begin(), ids.end(), id);
    return found == ids.end() ? -1 : static_cast<std::int64_t>(found - ids.begin());
}

// Binds the relations of a model to a snapshot, collecting the problems.
class Binder
{
public:
    Binder(const DesignModel& model, const input::ModelSnapshot& snapshot) : m_model(model), m_snapshot(snapshot) {}

    void bind(std::vector<FieldTarget>& targets, std::vector<std::int64_t>& unitIds, DerivedProperties& derived)
    {
        m_derivedProperties = &derived;
        prepareUnits(unitIds);
        targets.resize(m_model.relations().size());
        for (std::size_t index = 0; index < m_model.relations().size(); ++index)
        {
            const DesignRelation& relation = m_model.relations()[index];
            FieldTarget& target = targets[index];
            switch (relation.targetKind)
            {
            case TargetKind::Property:
                m_relationIndex = index;
                bindCardTable(relation, "properties/", target);
                break;
            case TargetKind::Element:
                bindCardTable(relation, "elements/", target);
                break;
            case TargetKind::Material:
                bindMaterial(relation, target);
                break;
            }
            if (!target.cells.empty() && m_hasUnits)
                target.units = unitsOf(relation);
        }
    }

    // The grid points of the shape variables: their rows, base positions, units, and the refusals (FieldTargets.hpp);
    // the directions and frames their moves change (DerivedGeometry).
    void bindShapes(std::vector<ShapeTarget>& targets, std::vector<std::int64_t>& movedRows,
                    std::vector<double>& basePositions, std::size_t unitCount, DerivedGeometry& derived)
    {
        const std::vector<ShapeBasis>& bases = m_model.shapes().bases();
        if (bases.empty())
            return;
        const input::NodeTable& nodes = m_snapshot.nodes();
        const std::span<const std::int32_t> ids = nodes.ids();
        std::set<std::int32_t> moving; // the grid points that really move (a non-zero move)
        for (const ShapeBasis& basis : bases)
            for (const ShapeMove& move : basis.moves)
                if (move.move != std::array<double, 3>{})
                    moving.insert(move.gridId);
        prepareShapeRefusals(moving);
        // The rows of each move, then the moved rows in increasing order.
        std::vector<std::vector<std::int64_t>> rowsOfBasis(bases.size());
        std::set<std::int64_t> rows;
        for (std::size_t b = 0; b < bases.size(); ++b)
            for (const ShapeMove& move : bases[b].moves)
            {
                const std::int64_t row = rowOfId(ids, move.gridId);
                rowsOfBasis[b].push_back(row);
                if (row < 0)
                {
                    m_diagnostics.push_back(
                        designProblem(design_code::MISSING_TARGET, "DesignNodeMove", bases[b].variableId, "nodeId",
                                      base::concatenate({"expected grid point ", std::to_string(move.gridId),
                                                         " in the model, found none; correct the "
                                                         "DesignNodeMove.nodeId of the DesignNodeMove"})));
                    continue;
                }
                rows.insert(row);
                if (move.move != std::array<double, 3>{})
                    refuseMove(bases[b].variableId, move.gridId, row);
            }
        movedRows.assign(rows.begin(), rows.end());
        const base::FixedListView<double> positions = nodes.positionsInBasic();
        for (const std::int64_t row : movedRows)
        {
            const std::span<const double> xyz = positions[row];
            basePositions.insert(basePositions.end(), xyz.begin(), xyz.begin() + 3);
        }
        derived = DerivedGeometry(m_model.shapes(), m_snapshot, m_diagnostics);
        // The rows whose units a basis changes: its moved grid points, and the grid points of the concentrated loads
        // whose directions it changes.
        std::vector<std::vector<std::int64_t>> unitRowsOfBasis = rowsOfBasis;
        std::set<std::int64_t> unitRows(rows.begin(), rows.end());
        for (std::size_t b = 0; b < bases.size(); ++b)
            for (const std::int32_t grid : derived.loadGridsOf(b))
                if (const std::int64_t row = rowOfId(ids, grid); row >= 0)
                {
                    unitRowsOfBasis[b].push_back(row);
                    unitRows.insert(row);
                }
        const std::map<std::int64_t, std::size_t> unitOfRow =
            unitsOfMovedRows(std::vector<std::int64_t>(unitRows.begin(), unitRows.end()));
        targets.resize(bases.size());
        for (std::size_t b = 0; b < bases.size(); ++b)
        {
            std::set<std::size_t> units;
            for (const std::int64_t row : rowsOfBasis[b])
            {
                if (row < 0)
                    continue;
                targets[b].points.push_back(static_cast<std::size_t>(
                    std::lower_bound(movedRows.begin(), movedRows.end(), row) - movedRows.begin()));
            }
            // A basis that moves a system changes what any element may read (every unit); otherwise the units of its
            // rows.
            const bool isEveryUnit =
                !m_snapshot.hasTable(input::TableRole::ConnectionNodes) || derived.changesEveryUnit(b);
            for (const std::int64_t row : unitRowsOfBasis[b])
            {
                if (row < 0 || !m_hasUnits)
                    continue;
                if (isEveryUnit)
                {
                    for (std::size_t unit = 0; unit < unitCount; ++unit)
                        units.insert(unit);
                }
                else if (const auto found = unitOfRow.find(row); found != unitOfRow.end())
                    units.insert(found->second);
            }
            targets[b].units.assign(units.begin(), units.end());
        }
    }

    [[nodiscard]] std::vector<base::Diagnostic>& diagnostics() noexcept { return m_diagnostics; }

private:
    void problem(const char* code, const DesignRelation& relation, const std::string& message)
    {
        m_diagnostics.push_back(designProblem(code, relation.card, relation.id, "fieldName", message));
    }

    // Checks that a column can hold the field at `row`; adds the problem otherwise.
    bool isFieldColumn(const DesignRelation& relation, const base::ContractReader& reader, const std::string& table,
                       std::int64_t row)
    {
        const std::string type = reader.columnType(relation.field);
        if (type.starts_with(NESTED_REAL_LIST))
        {
            // A BeamProp station field (station_section: A, I1, I2, I12, J, NSM; station_stress: C1 to F2), its item
            // flattened as O_o_Reader's field_item: station · width + component, within the stations of the row.
            const input::NestedListView view(reader, relation.field);
            const std::int64_t items = view.itemCount(row) * view.width();
            if (relation.fieldItem >= 0 && relation.fieldItem < items)
                return true;
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected an item from 0 to ", std::to_string(items - 1), " of the column '",
                                       relation.field, "' of ", table, " (", std::to_string(view.itemCount(row)),
                                       " stations of ", std::to_string(view.width()),
                                       " values: station · width + component), found ",
                                       std::to_string(relation.fieldItem)}));
            return false;
        }
        if (type == "list<float64>")
        {
            // The plies of a CompProp or CompPropGlobal, the dimensions of a BarShapeProp or of the stations of a
            // BeamShapeProp: the item a position in the list of the row.
            const auto items = static_cast<std::int64_t>(reader.lists<double>(relation.field)[row].size());
            if (relation.fieldItem >= 0 && relation.fieldItem < items)
                return true;
            problem(
                design_code::MISSING_TARGET, relation,
                base::concatenate({"expected an item from 0 to ", std::to_string(items - 1), " of the column '",
                                   relation.field, "' of ", table, ", found ", std::to_string(relation.fieldItem)}));
            return false;
        }
        if (type == "float64")
        {
            if (relation.fieldItem < 0)
                return true;
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected no item for the float64 column '", relation.field, "' of ", table,
                                       ", found item ", std::to_string(relation.fieldItem)}));
            return false;
        }
        if (type.starts_with("fixed_list<float64>"))
        {
            const std::int64_t width = reader.fixedLists<double>(relation.field, base::NullPolicy::Allow).width();
            if (relation.fieldItem >= 0 && relation.fieldItem < width)
                return true;
            problem(
                design_code::MISSING_TARGET, relation,
                base::concatenate({"expected an item from 0 to ", std::to_string(width - 1), " of the column '",
                                   relation.field, "' of ", table, ", found ", std::to_string(relation.fieldItem)}));
            return false;
        }
        problem(design_code::UNSUPPORTED_TARGET, relation,
                base::concatenate({"expected a real field, found the column '", relation.field, "' of ", table,
                                   " of type ", type, "; only real fields can be designed"}));
        return false;
    }

    static constexpr std::string_view NESTED_REAL_LIST = "list<fixed_list<float64>";

    void bindCardTable(const DesignRelation& relation, const std::string& prefix, FieldTarget& target)
    {
        const std::string table = prefix + relation.targetCard;
        const bool isDerived = relation.targetKind == TargetKind::Property &&
                               std::find(DERIVED_PROPERTY_CARDS.begin(), DERIVED_PROPERTY_CARDS.end(),
                                         relation.targetCard) != DERIVED_PROPERTY_CARDS.end();
        const base::ContractReader* reader = m_snapshot.findTable(table);
        if (reader == nullptr)
        {
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected a ", relation.targetCard, " ", std::to_string(relation.targetId),
                                       ", found no ", relation.targetCard, " in the model"}));
            return;
        }
        const std::span<const std::int32_t> ids = idsOf(relation);
        const std::int64_t row = rowOfId(ids, relation.targetId);
        if (row < 0)
        {
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected a ", relation.targetCard, " ", std::to_string(relation.targetId),
                                       ", found none of that ID in the model"}));
            return;
        }
        if (!reader->hasColumn(relation.field))
        {
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected a field '", relation.field, "' of ", relation.targetCard,
                                       " (a column of its table in the data contract), found none"}));
            return;
        }
        if (!isFieldColumn(relation, *reader, table, row))
            return;
        // The OD2 of a TubeProp alone would taper the tube, which the Tube2 of this version does not take: a blank
        // OD2 follows a designed OD (below), a given one equal to OD is designed with it.
        if (relation.targetKind == TargetKind::Property && relation.targetCard == "TubeProp" &&
            relation.field == "outerDiameter2")
        {
            problem(design_code::UNSUPPORTED_TARGET, relation,
                    "the TubeProp.outerDiameter2 of a TubeProp as a target would taper the tube, which this "
                    "version does not analyse; design "
                    "TubeProp.outerDiameter (a blank TubeProp.outerDiameter2, or one equal to "
                    "TubeProp.outerDiameter, follows it)");
            return;
        }
        if (isDerived)
        {
            // The ply stack or the section properties follow the card's cell (DerivedProperties).
            std::string why;
            if (!m_derivedProperties->bind(m_snapshot, m_relationIndex, relation.targetCard, relation.targetId, table,
                                           relation.field, row, relation.fieldItem, why))
            {
                problem(design_code::UNSUPPORTED_TARGET, relation, why);
                return;
            }
        }
        target.cells.push_back({table, relation.field, row, relation.fieldItem});
        // A TubeProp's blank OD2 is OD (a constant tube) and its blank T is OD/2 (a solid rod); a blank one follows a
        // designed OD (as O_o_Reader's model.design.apply). From data contract 54 the card says which were given
        // (od2_given, t_given); before it, T is taken as given. An OD2 equal to OD follows it either way.
        if (relation.targetCard == "TubeProp" && relation.field == "outerDiameter")
        {
            const auto isBlank = [&](const char* flag) -> std::optional<bool>
            {
                if (!reader->hasColumn(flag) || reader->columnType(flag) != "uint8")
                    return std::nullopt;
                return reader->values<std::uint8_t>(flag, base::NullPolicy::Allow)[static_cast<std::size_t>(row)] == 0;
            };
            if (reader->hasColumn("outerDiameter2") && reader->columnType("outerDiameter2") == "float64")
            {
                const std::span<const double> outer = reader->values<double>("outerDiameter", base::NullPolicy::Allow);
                const std::span<const double> outer2 =
                    reader->values<double>("outerDiameter2", base::NullPolicy::Allow);
                // A given OD2 equal to OD follows it too: left behind, the tube would taper, which this version
                // does not analyse.
                const bool follows = isBlank("outerDiameter2Given").value_or(false) ||
                                     outer2[static_cast<std::size_t>(row)] == outer[static_cast<std::size_t>(row)];
                if (follows)
                    target.cells.push_back({table, "outerDiameter2", row, relation.fieldItem});
            }
            if (isBlank("thicknessGiven").value_or(false) && reader->hasColumn("thickness"))
                target.cells.push_back({table, "thickness", row, relation.fieldItem, 0.5});
        }
    }

    [[nodiscard]] std::span<const std::int32_t> idsOf(const DesignRelation& relation) const
    {
        if (relation.targetKind == TargetKind::Element)
            return m_snapshot.findElementBatch(relation.targetCard)->ids();
        return m_snapshot.findPropertyTable(relation.targetCard)->ids();
    }

    void bindMaterial(const DesignRelation& relation, FieldTarget& target)
    {
        bool hasTable = false;
        bool hasColumn = false;
        if (m_snapshot.hasTable(input::TableRole::Materials))
        {
            const input::MaterialTable& materials = m_snapshot.materials();
            const std::span<const std::int32_t> ids = materials.ids();
            const base::DictionaryView types = materials.types();
            for (std::int64_t row = 0; row < materials.rowCount(); ++row)
            {
                if (ids[static_cast<std::size_t>(row)] != relation.targetId || types[row] != relation.targetCard)
                    continue;
                hasTable = true;
                const std::string column = materialSummaryField(relation.field);
                if (materials.reader().hasColumn(column))
                {
                    hasColumn = true;
                    auto summaryRelation = relation;
                    summaryRelation.field = column;
                    if (isFieldColumn(summaryRelation, materials.reader(), MATERIALS_TABLE, row))
                        target.cells.push_back({MATERIALS_TABLE, column, row, relation.fieldItem});
                }
                break;
            }
        }
        if (const input::CardTable* card = m_snapshot.findMaterialCardTable(relation.targetCard))
        {
            const std::int64_t row = rowOfId(card->ids(), relation.targetId);
            if (row >= 0)
            {
                hasTable = true;
                if (card->reader().hasColumn(relation.field))
                {
                    hasColumn = true;
                    const std::string table = "materials/" + relation.targetCard;
                    if (isFieldColumn(relation, card->reader(), table, row))
                        target.cells.push_back({table, relation.field, row, relation.fieldItem});
                }
            }
        }
        if (!hasTable)
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected a ", relation.targetCard, " ", std::to_string(relation.targetId),
                                       ", found none of that ID and type in the model"}));
        else if (!hasColumn)
            problem(design_code::MISSING_TARGET, relation,
                    base::concatenate({"expected a field '", relation.field, "' of ", relation.targetCard,
                                       " (a column of materialsTable or of its card table), found none"}));
    }

    void prepareUnits(std::vector<std::int64_t>& unitIds)
    {
        m_hasUnits = m_snapshot.hasTable(input::TableRole::Components) &&
                     m_snapshot.hasTable(input::TableRole::ComponentElements);
        if (!m_hasUnits)
            return;
        const input::UnitTable& units = m_snapshot.units();
        const std::span<const std::int64_t> ids = units.componentIds();
        unitIds.assign(ids.begin(), ids.end());
        std::unordered_map<std::int64_t, std::size_t> unitOfComponent;
        for (std::size_t index = 0; index < unitIds.size(); ++index)
            unitOfComponent.emplace(unitIds[index], index);

        const base::ContractReader& elements = units.componentElements();
        const std::span<const std::int32_t> elementIds = elements.values<std::int32_t>("elementId");
        const base::DictionaryView types = elements.dictionaryTexts("type", base::NullPolicy::Allow);
        const std::span<const std::int32_t> propIds = elements.values<std::int32_t>("propId", base::NullPolicy::Allow);
        const base::ValidityView propIdValid = elements.validity("propId");
        // O_o_Reader leaves component_id null for an element in no component (a rigid element of a single grid
        // point): it changes no unit.
        const std::span<const std::int64_t> components =
            elements.values<std::int64_t>("componentId", base::NullPolicy::Allow);
        const base::ValidityView componentValid = elements.validity("componentId");
        for (std::int64_t row = 0; row < elements.rowCount(); ++row)
        {
            if (!componentValid.isValid(row))
                continue;
            const auto found = unitOfComponent.find(components[static_cast<std::size_t>(row)]);
            if (found == unitOfComponent.end())
                continue;
            m_unitOfElement[{std::string(types[row]), elementIds[static_cast<std::size_t>(row)]}] = found->second;
            if (propIdValid.isValid(row))
                m_unitsOfProperty[propIds[static_cast<std::size_t>(row)]].insert(found->second);
        }
    }

    // The properties and elements that name each material, built at the first material relation.
    void prepareMaterialUsers()
    {
        if (m_areMaterialUsersReady)
            return;
        m_areMaterialUsersReady = true;
        const auto scan = [](const base::ContractReader& reader, std::span<const std::int32_t> ids,
                             const std::function<void(std::int32_t matId, std::int32_t id)>& add)
        {
            for (const std::string& column : reader.table().columnNames())
            {
                if (!isMaterialIdColumn(column) || reader.columnType(column) != "int32")
                    continue;
                const std::span<const std::int32_t> matIds =
                    reader.values<std::int32_t>(column, base::NullPolicy::Allow);
                const base::ValidityView valid = reader.validity(column);
                for (std::int64_t row = 0; row < reader.rowCount(); ++row)
                {
                    if (valid.isValid(row))
                        add(matIds[static_cast<std::size_t>(row)], ids[static_cast<std::size_t>(row)]);
                }
            }
        };
        for (const input::CardTable& table : m_snapshot.propertyTables())
            scan(table.reader(), table.ids(),
                 [&](std::int32_t matId, std::int32_t propId) { m_propertiesOfMaterial[matId].insert(propId); });
        for (const input::ElementBatch& batch : m_snapshot.elementBatches())
            scan(batch.reader(), batch.ids(), [&](std::int32_t matId, std::int32_t elementId)
                 { m_elementsOfMaterial[matId].insert({batch.card(), elementId}); });
        if (m_snapshot.hasTable(input::TableRole::LaminatePlies))
        {
            const base::ContractReader& plies = m_snapshot.laminatePlies();
            scan(plies, plies.values<std::int32_t>("propId"),
                 [&](std::int32_t matId, std::int32_t propId) { m_propertiesOfMaterial[matId].insert(propId); });
        }
    }

    [[nodiscard]] std::vector<std::size_t> unitsOf(const DesignRelation& relation)
    {
        std::set<std::size_t> units;
        const auto addProperty = [&](std::int32_t propId)
        {
            const auto found = m_unitsOfProperty.find(propId);
            if (found != m_unitsOfProperty.end())
                units.insert(found->second.begin(), found->second.end());
        };
        const auto addElement = [&](const std::pair<std::string, std::int32_t>& element)
        {
            const auto found = m_unitOfElement.find(element);
            if (found != m_unitOfElement.end())
                units.insert(found->second);
        };
        switch (relation.targetKind)
        {
        case TargetKind::Property:
            addProperty(relation.targetId);
            break;
        case TargetKind::Element:
            addElement({relation.targetCard, relation.targetId});
            break;
        case TargetKind::Material:
            prepareMaterialUsers();
            if (const auto found = m_propertiesOfMaterial.find(relation.targetId);
                found != m_propertiesOfMaterial.end())
                for (const std::int32_t propId : found->second)
                    addProperty(propId);
            if (const auto found = m_elementsOfMaterial.find(relation.targetId); found != m_elementsOfMaterial.end())
                for (const auto& element : found->second)
                    addElement(element);
            break;
        }
        return {units.begin(), units.end()};
    }

    // Among the grid points `moving`, those whose positions a table derived by O_o_Reader reads (the elements
    // Fastener and Weld pierce, glued contact), with the first thing found that reads them, for the refusals of
    // refuseMove.
    void prepareShapeRefusals(const std::set<std::int32_t>& moving)
    {
        const auto add = [&](std::int32_t grid, const auto& describe)
        {
            if (moving.count(grid) != 0 && m_derivedReaders.count(grid) == 0)
                m_derivedReaders.emplace(grid, describe());
        };
        if (m_snapshot.hasTable(input::TableRole::FastenerPiercings))
        {
            const input::FastenerPiercingTable& piercings = m_snapshot.fastenerPiercings();
            const base::ValidityView isGiven = piercings.isPointGridGiven();
            const base::ListView<std::int32_t> grids = piercings.grids();
            for (std::int64_t row = 0; row < piercings.rowCount(); ++row)
            {
                const auto reader = [&]
                {
                    return base::concatenate({"the pierced element of ", std::string(piercings.card(row)), " ",
                                              std::to_string(piercings.fastenerIds()[static_cast<std::size_t>(row)]),
                                              " (fastenerPiercings: projections and shape functions O_o_Reader "
                                              "computes from the positions)"});
                };
                for (const std::int32_t grid : grids[row])
                    add(grid, reader);
                if (isGiven.isValid(row))
                    add(piercings.pointGrids()[static_cast<std::size_t>(row)], reader);
            }
        }
        for (const char* card : {"Fastener", "Weld"})
        {
            const input::ElementBatch* batch = m_snapshot.findElementBatch(card);
            if (batch == nullptr)
                continue;
            const base::ContractReader& reader = batch->reader();
            for (const char* column : {"spotNode", "nodeA", "nodeB"})
            {
                if (!reader.hasColumn(column) || reader.columnType(column) != "int32")
                    continue;
                const std::span<const std::int32_t> grids =
                    reader.values<std::int32_t>(column, base::NullPolicy::Allow);
                const base::ValidityView valid = reader.validity(column);
                for (std::int64_t row = 0; row < reader.rowCount(); ++row)
                    if (valid.isValid(row))
                        add(grids[static_cast<std::size_t>(row)],
                            [&]
                            {
                                return base::concatenate(
                                    {card, " ", std::to_string(batch->ids()[static_cast<std::size_t>(row)]),
                                     " (its piercing points come from fastenerPiercings, computed by O_o_Reader from "
                                     "the positions)"});
                            });
            }
        }
        if (m_snapshot.hasTable(input::TableRole::ContactNodePairs))
        {
            const input::ContactTable& contact = m_snapshot.contact();
            for (std::int64_t row = 0; row < contact.nodePairCount(); ++row)
            {
                const input::ContactNodePair pair = contact.nodePair(row);
                const auto reader = [&]
                {
                    return base::concatenate({"the glued contact of grid point ", std::to_string(pair.nodeId),
                                              " (contactNodePairs: the projection and local coordinates O_o_Reader "
                                              "computes from the positions)"});
                };
                add(pair.nodeId, reader);
                for (const std::int32_t grid : pair.face)
                    if (grid != 0)
                        add(grid, reader);
            }
        }
    }

    // Refuses the move of a grid point that the overlay of its position cannot make right (FieldTargets.hpp).
    void refuseMove(std::int32_t variableId, std::int32_t gridId, std::int64_t row)
    {
        const auto refuse = [&](const std::string& why)
        {
            m_diagnostics.push_back(designProblem(
                design_code::UNSUPPORTED_TARGET, "DesignNodeMove", variableId, "nodeId",
                base::concatenate({"grid point ", std::to_string(gridId), " cannot be moved by a shape variable: ", why,
                                   "; the design point would be analysed with the table as read. Leave "
                                   "the grid point out of the shape, or move it in the model and take a new "
                                   "snapshot"})));
        };
        if (const auto found = m_derivedReaders.find(gridId); found != m_derivedReaders.end())
            refuse(base::concatenate({"its position is read by ", found->second}));
        const std::int32_t displacementFrameId =
            m_snapshot.nodes().displacementFrameIds()[static_cast<std::size_t>(row)];
        if (displacementFrameId == 0)
            return;
        if (!m_snapshot.hasTable(input::TableRole::CoordinateSystems))
        {
            refuse(base::concatenate({"its displacement system Node.displacementFrameId ",
                                      std::to_string(displacementFrameId),
                                      " cannot be checked (the snapshot has no coordsTable); a cylindrical or "
                                      "spherical Node.displacementFrameId turns with the position"}));
            return;
        }
        const input::CoordinateSystemTable& systems = m_snapshot.coordinateSystems();
        const std::int64_t systemRow = rowOfId(systems.ids(), displacementFrameId);
        if (systemRow >= 0 && systems.frameAt(systemRow).kind == input::CoordinateFrame::Kind::Rectangular)
            return;
        refuse(
            base::concatenate({"its displacement system Node.displacementFrameId ", std::to_string(displacementFrameId),
                               " is cylindrical or spherical, so its directions (displacementRotations, computed by "
                               "O_o_Reader at the read position) turn with the position"}));
    }

    // The unit of each moved grid point (componentNodes), when the snapshot has the units and that table.
    [[nodiscard]] std::map<std::int64_t, std::size_t> unitsOfMovedRows(const std::vector<std::int64_t>& movedRows) const
    {
        std::map<std::int64_t, std::size_t> unitOfRow;
        if (!m_hasUnits || !m_snapshot.hasTable(input::TableRole::ConnectionNodes))
            return unitOfRow;
        std::unordered_map<std::int64_t, std::size_t> unitOfComponent;
        const std::span<const std::int64_t> componentIds = m_snapshot.units().componentIds();
        for (std::size_t index = 0; index < componentIds.size(); ++index)
            unitOfComponent.emplace(componentIds[index], index);
        std::unordered_map<std::int32_t, std::int64_t> rowOfGrid;
        const std::span<const std::int32_t> ids = m_snapshot.nodes().ids();
        for (const std::int64_t row : movedRows)
            rowOfGrid.emplace(ids[static_cast<std::size_t>(row)], row);
        const base::ContractReader& nodes = m_snapshot.connectionNodes();
        const std::span<const std::int32_t> nodeIds = nodes.values<std::int32_t>("nodeId");
        const std::span<const std::int64_t> components =
            nodes.values<std::int64_t>("componentId", base::NullPolicy::Allow);
        const base::ValidityView valid = nodes.validity("componentId");
        for (std::int64_t r = 0; r < nodes.rowCount(); ++r)
        {
            const auto row = rowOfGrid.find(nodeIds[static_cast<std::size_t>(r)]);
            if (row == rowOfGrid.end() || !valid.isValid(r))
                continue;
            if (const auto unit = unitOfComponent.find(components[static_cast<std::size_t>(r)]);
                unit != unitOfComponent.end())
                unitOfRow[row->second] = unit->second;
        }
        return unitOfRow;
    }

    const DesignModel& m_model;
    const input::ModelSnapshot& m_snapshot;
    DerivedProperties* m_derivedProperties = nullptr;
    std::size_t m_relationIndex = 0; // the relation being bound
    std::vector<base::Diagnostic> m_diagnostics;
    std::map<std::int32_t, std::string> m_derivedReaders; // a moving grid point → what reads it in a derived table
    bool m_hasUnits = false;
    bool m_areMaterialUsersReady = false;
    std::map<std::pair<std::string, std::int32_t>, std::size_t> m_unitOfElement;
    std::unordered_map<std::int32_t, std::set<std::size_t>> m_unitsOfProperty;
    std::unordered_map<std::int32_t, std::set<std::int32_t>> m_propertiesOfMaterial;
    std::unordered_map<std::int32_t, std::set<std::pair<std::string, std::int32_t>>> m_elementsOfMaterial;
};

} // namespace

FieldTargets::FieldTargets(const DesignModel& model, const input::ModelSnapshot& snapshot,
                           std::shared_ptr<const SectionLibrary> sections)
    : m_sections(std::move(sections))
{
    Binder binder(model, snapshot);
    binder.bind(m_targets, m_unitIds, m_derivedProperties);
    binder.bindShapes(m_shapeTargets, m_movedRows, m_basePositions, m_unitIds.size(), m_derived);
    throwDesignDiagnostics("the design model does not fit the model", binder.diagnostics());
    m_derivedMaterials.bind(model, snapshot);
}

std::vector<double> FieldTargets::movedPositions(const ShapeRelations& relations, std::span<const double> shapes) const
{
    std::vector<double> positions = m_basePositions;
    const std::vector<ShapeBasis>& bases = relations.bases();
    for (std::size_t b = 0; b < m_shapeTargets.size(); ++b)
    {
        const ShapeTarget& target = m_shapeTargets[b];
        for (std::size_t m = 0; m < target.points.size(); ++m)
            for (std::size_t k = 0; k < 3; ++k)
                positions[3 * target.points[m] + k] += shapes[b] * bases[b].moves[m].move[k];
    }
    return positions;
}

input::ColumnOverlay FieldTargets::overlayOf(const ShapeRelations& relations, std::span<const double> fields,
                                             std::span<const double> shapes, std::span<const double> reference) const
{
    input::ColumnOverlay overlay;
    for (std::size_t index = 0; index < m_targets.size(); ++index)
        for (const TargetCell& cell : m_targets[index].cells)
            overlay.setValue(cell.table, cell.column, {cell.row, cell.item, cell.scale * fields[index]});
    // The ply stacks and section properties of the designed laminates and library sections.
    if (!m_derivedProperties.isEmpty())
        m_derivedProperties.addCells(overlay, fields, std::nullopt, m_sections.get(), nullptr, reference);
    m_derivedMaterials.addCells(overlay, fields);
    if (m_movedRows.empty())
        return overlay;
    const std::vector<double> positions = movedPositions(relations, shapes);
    const std::string nodes(input::tableName(input::TableRole::Nodes));
    for (std::size_t p = 0; p < m_movedRows.size(); ++p)
        for (std::int32_t k = 0; k < 3; ++k)
            overlay.setValue(nodes, NODE_POSITION_COLUMN,
                             {m_movedRows[p], k, positions[3 * p + static_cast<std::size_t>(k)]});
    // The directions and frames the moved positions define. A perturbation of a field (no shapes given)
    // leaves them at the base positions, where the base snapshot holds them already.
    if (!shapes.empty())
        m_derived.addCells(shapes, overlay);
    return overlay;
}

void FieldTargets::addDerivedCells(input::ColumnOverlay& overlay, std::span<const double> fields, std::size_t relation,
                                   const SectionLinearisation* linear) const
{
    m_derivedMaterials.addCells(overlay, fields, relation);
    if (const std::optional<std::size_t> group = m_derivedProperties.groupOf(relation))
        m_derivedProperties.addCells(overlay, fields, group, m_sections.get(), linear);
}

std::vector<base::Diagnostic> checkFieldTargets(const DesignModel& model, const input::ModelSnapshot& snapshot)
{
    Binder binder(model, snapshot);
    std::vector<FieldTarget> targets;
    std::vector<std::int64_t> unitIds;
    DerivedProperties derivedProperties;
    binder.bind(targets, unitIds, derivedProperties);
    std::vector<ShapeTarget> shapeTargets;
    std::vector<std::int64_t> movedRows;
    std::vector<double> basePositions;
    DerivedGeometry derived;
    binder.bindShapes(shapeTargets, movedRows, basePositions, unitIds.size(), derived);
    return std::move(binder.diagnostics());
}

} // namespace oos::design
