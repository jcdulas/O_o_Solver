#pragma once

#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/BorrowedArray.hpp"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// Assembles a ModelSnapshot from the tables the caller gathers from a O_o_Reader model (the table packages of the
// O_o_Reader WebAssembly module): the tables of the roles, one table per element, property and method card type, the
// CD rotations and the version stamps. Each table is taken over as it comes (its contract version is checked then);
// build() checks every table against its contract and hands everything to the snapshot without copying a column.
//
// Tables that belong together must come together: single-point and multipoint constraints; the four static load
// tables; the components and the component of each element; the output requests and the case sets. The node table is
// required.
//
// Thread-safety: none (one builder per snapshot, on one thread). Ownership: owns the tables until build().
class SnapshotBuilder
{
public:
    SnapshotBuilder() = default;
    SnapshotBuilder(const SnapshotBuilder&) = delete;
    SnapshotBuilder& operator=(const SnapshotBuilder&) = delete;
    SnapshotBuilder(SnapshotBuilder&&) noexcept = default;
    SnapshotBuilder& operator=(SnapshotBuilder&&) noexcept = default;
    ~SnapshotBuilder() = default;

    void setVersions(std::int64_t structureVersion, std::int64_t contentVersion);

    // Throws std::invalid_argument when the role was already given, ContractError when the table carries a contract
    // version that is not accepted or differs from that of the tables given before.
    void setTable(TableRole role, base::ImportedTable table);
    void setCdRotations(BorrowedArray<double> rotations);
    // The model's MDLPRM OFFDEF (ModelSnapshot::offsetDefinition); ELMOFF, the default, without this call.
    void setOffsetDefinition(OffsetDefinition definition)
    {
        m_offsetDefinition = definition;
        m_unsupportedOffsetDefinition.clear();
    }
    // The model's MDLPRM OFFDEF as its value: one offsetDefinitionOf takes, or another one, kept for the support report
    // to refuse (ModelSnapshot::unsupportedOffsetDefinition) with the default definition.
    void setOffsetDefinitionValue(std::string_view value);
    // The formulation of the elements of one card (the card by its neutral name, "ShellQuad4"; the formulation
    // "ClassicFormulation": ModelSnapshot::elementFormulations), the formulation compared exactly; a second call for
    // the same card replaces the first. The card name is kept exactly; the element registry reports unsupported cards.
    // Model formulation parameters are read from the parameter table. The support report refuses a formulation
    // the card has no formulation for. Throws std::invalid_argument for an empty card or name.
    void setElementFormulation(std::string_view card, std::string_view formulation);
    // The card tables, each by the neutral card name ("ShellQuad4", "ShellProp"). Throws std::invalid_argument when the
    // card type was already given, ContractError as setTable(). An element batch is keyed by ELEMENT_ID_COLUMN, a
    // property table by PROPERTY_ID_COLUMN ("id" both).
    void addElementBatch(std::string card, base::ImportedTable table);
    void addPropertyTable(std::string card, base::ImportedTable table);
    void addMethodTable(std::string card, std::string idColumn, base::ImportedTable table);
    // A material card table (IsoMat, AnisoMat2D, OrthoMat2D: the card table of O_o_Reader), keyed by "id".
    void addMaterialCardTable(std::string card, base::ImportedTable table);
    // A load card table with the field formats of its cards (PointForceLoad, ShellPressureLoad, LoadCombination...: the
    // card table of O_o_Reader), keyed by "setId".
    void addLoadCardTable(std::string card, base::ImportedTable table);
    // An aeroelastic card table read field by field (AeroBeamSpline2, AeroBeamSpline5 and AeroBoxList keyed by "id":
    // the card table of O_o_Reader).
    void addAeroCardTable(std::string card, std::string idColumn, base::ImportedTable table);

    // Checks every table against its contract and builds the snapshot; the builder is left empty. Throws
    // ContractError for a table that breaks its contract, a missing node table or name table, or an incomplete group
    // of tables.
    // Exception safety: the node table and the groups are checked before any table is taken, so such an error leaves
    // the builder unchanged; a table refused by its contract leaves it empty (the tables taken so far are released),
    // never half emptied.
    [[nodiscard]] ModelSnapshot build();

private:
    struct CardEntry
    {
        std::string card;
        std::string idColumn;
        base::ImportedTable table;
    };

    void checkVersion(const base::ImportedTable& table, const std::string& name);
    [[nodiscard]] std::optional<base::ImportedTable> take(TableRole role);
    // Throws ContractError when a group of tables is incomplete (checkGroup) or the node table is missing.
    void checkPresence() const;
    // ContractError naming the missing tables when some but not all of the group were given.
    void checkGroup(std::initializer_list<TableRole> roles) const;
    // Takes the tables of a group (checked by checkPresence): all present (returned) or all absent (nothing).
    [[nodiscard]] std::optional<std::vector<base::ImportedTable>> takeGroup(std::initializer_list<TableRole> roles);
    [[nodiscard]] ModelSnapshot takeAll();

    ModelVersions m_versions;
    OffsetDefinition m_offsetDefinition = OffsetDefinition::ElementOffsets;
    std::string m_unsupportedOffsetDefinition;
    std::vector<ElementFormulationChoice> m_elementFormulations; // sorted by card
    std::map<TableRole, base::ImportedTable> m_tables;
    BorrowedArray<double> m_cdRotations;
    std::vector<CardEntry> m_elementTables;
    std::vector<CardEntry> m_propertyTables;
    std::vector<CardEntry> m_methodTables;
    std::vector<CardEntry> m_materialCardTables;
    std::vector<CardEntry> m_loadCardTables;
    std::vector<CardEntry> m_aeroCardTables;
};

} // namespace oos::input
