#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/requests/OutputRequests.hpp"
#include "oos/input/snapshot/OffsetDefinition.hpp"
#include "oos/input/tables/CardTable.hpp"
#include "oos/input/tables/ConstraintTable.hpp"
#include "oos/input/tables/ContactTable.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"
#include "oos/input/tables/ElementBatch.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"
#include "oos/input/tables/LoadTable.hpp"
#include "oos/input/tables/MassTable.hpp"
#include "oos/input/tables/MaterialTable.hpp"
#include "oos/input/tables/NodeTable.hpp"
#include "oos/input/tables/ParamTable.hpp"
#include "oos/input/tables/RigidTable.hpp"
#include "oos/input/tables/SectionTable.hpp"
#include "oos/input/tables/SubcaseTable.hpp"
#include "oos/input/tables/UnitTable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// The formulation the program that built the snapshot chose for the elements of one card: the neutral card name
// (ShellQuad4) and the formulation name. The element layer reads it (elements::ElementRegistry::forModel), before the
// model's <card>Formulation parameter.
struct ElementFormulationChoice
{
    std::string card;
    std::string formulation;

    friend bool operator==(const ElementFormulationChoice&, const ElementFormulationChoice&) = default;
};

// The tables of the aeroelastic model and of its direct matrix input (the DownwashMatrix and AeroPressureMatrix of a
// trim), in the order ModelSnapshot keeps them.
inline constexpr std::size_t AERO_TABLE_COUNT = 9;
inline constexpr std::array<TableRole, AERO_TABLE_COUNT> AERO_TABLE_ROLES{
    TableRole::AeroReference,   TableRole::AeroPanels,         TableRole::AeroBoxes,
    TableRole::ControlSurfaces, TableRole::TrimCases,          TableRole::Splines,
    TableRole::SplineNodes,     TableRole::DirectMatrixShapes, TableRole::DirectMatrices};

// The version stamps of the model a snapshot was taken from (O_o_Reader: model.structure_version, which changes when
// cards are added or removed and invalidates the row-index columns; model.content_version, which changes with any
// edit), and the data contract version its tables carry (0 when none carried one).
struct ModelVersions
{
    std::int64_t structureVersion = 0;
    std::int64_t contentVersion = 0;
    std::int64_t dataContractVersion = 0;
};

// Everything the solver reads from a O_o_Reader model, in one immutable object: the tables of the data
// contract, each checked against its contract and read in place. The snapshot borrows the producer's buffers: it owns
// the Arrow structures of every table (whose release lets O_o_Reader free its buffers) and the owners of the borrowed
// arrays, and never copies a column it can read in place. O_o_Reader's columns are copy-on-write, so an edit of the
// model after the snapshot leaves the snapshot's buffers as they were: a new snapshot is taken after an edit, and
// SnapshotDiff tells what changed.
//
// The node table is always there; the other tables are there when the snapshot was taken with them (SnapshotBuilder),
// and an accessor of an absent table throws ContractError naming the table and what it holds.
//
// Thread-safety: immutable; concurrent reads from any number of threads are safe. Ownership: owns its tables; movable,
// not copyable (share it through a std::shared_ptr). Built by SnapshotBuilder.
class ModelSnapshot
{
public:
    ModelSnapshot(const ModelSnapshot&) = delete;
    ModelSnapshot& operator=(const ModelSnapshot&) = delete;
    ModelSnapshot(ModelSnapshot&&) noexcept = default;
    ModelSnapshot& operator=(ModelSnapshot&&) noexcept = default;
    ~ModelSnapshot() = default;

    [[nodiscard]] const ModelVersions& versions() const noexcept { return m_versions; }
    [[nodiscard]] const NodeTable& nodes() const noexcept { return m_nodes; }
    // The element formulations chosen by the program that built the snapshot, one per card at most, sorted by card
    // (empty: the deck's PARAMs and the defaults decide).
    [[nodiscard]] const std::vector<ElementFormulationChoice>& elementFormulations() const noexcept
    {
        return m_elementFormulations;
    }
    // The model's MDLPRM OFFDEF (ELMOFF, the default, when the model gives none): where the mass of an offset
    // element is (OffsetDefinition).
    [[nodiscard]] OffsetDefinition offsetDefinition() const noexcept { return m_offsetDefinition; }
    // The model's MDLPRM OFFDEF value when O_o_Solver does not take it (offsetDefinitionOf gives nothing: NOMASS ...),
    // which SupportCatalog::check refuses; empty otherwise (offsetDefinition() is then the value's).
    [[nodiscard]] const std::string& unsupportedOffsetDefinition() const noexcept
    {
        return m_unsupportedOffsetDefinition;
    }

    // One batch per element card type present in the model, in the order they were given.
    [[nodiscard]] const std::vector<ElementBatch>& elementBatches() const noexcept { return m_elementBatches; }
    [[nodiscard]] const ElementBatch* findElementBatch(std::string_view card) const noexcept;
    // One table per property card type present (ShellProp, BarProp...), and per method card type
    // (LanczosEigenMethod, RealEigenMethod, BucklingEigenMethod), by the neutral card name.
    [[nodiscard]] const std::vector<CardTable>& propertyTables() const noexcept { return m_propertyTables; }
    [[nodiscard]] const CardTable* findPropertyTable(std::string_view card) const noexcept;
    [[nodiscard]] const std::vector<CardTable>& methodTables() const noexcept { return m_methodTables; }
    [[nodiscard]] const CardTable* findMethodTable(std::string_view card) const noexcept;
    // One table per material card type the elements read field by field (IsoMat, AnisoMat2D, OrthoMat2D: the card
    // tables of O_o_Reader, keyed by "id"), for the constants materialsTable does not give (the transverse shear moduli
    // and expansion coefficients of an OrthoMat2D, the matrix and expansion coefficients of an AnisoMat2D).
    [[nodiscard]] const std::vector<CardTable>& materialCardTables() const noexcept { return m_materialCardTables; }
    [[nodiscard]] const CardTable* findMaterialCardTable(std::string_view card) const noexcept;
    // One table per load card type whose field formats the deck tolerance reads (PointForceLoad, ShellPressureLoad,
    // LoadCombination...: the card tables of O_o_Reader with their field formats, keyed by "setId"; the
    // LoadCombination table with its "termLoad" column too).
    [[nodiscard]] const std::vector<CardTable>& loadCardTables() const noexcept { return m_loadCardTables; }
    // The card tables of the aeroelastic model (AeroBeamSpline2, AeroBeamSpline5, AeroBoxList,
    // AeroRigidTrimVariable ...).
    [[nodiscard]] const std::vector<CardTable>& aeroCardTables() const noexcept { return m_aeroCardTables; }
    // One table per aeroelastic card type the aero module reads field by field (AeroBeamSpline2, AeroBeamSpline5 and
    // AeroBoxList keyed by "id": the card tables of O_o_Reader).
    [[nodiscard]] const CardTable* findAeroCardTable(std::string_view card) const noexcept;

    // Whether the snapshot holds the table of a role.
    [[nodiscard]] bool hasTable(TableRole role) const noexcept;

    // The typed views; each throws ContractError when its table is absent (missingTableError).
    [[nodiscard]] const MaterialTable& materials() const;
    [[nodiscard]] const ConstraintTable& constraints() const;
    [[nodiscard]] const RigidTable& rigidElements() const;
    [[nodiscard]] const LoadTable& loads() const;
    // The load tables, or null when the snapshot was taken without them (SpcSet reads its SPCD through them).
    [[nodiscard]] const LoadTable* findLoads() const noexcept { return m_loads ? &*m_loads : nullptr; }
    [[nodiscard]] const SubcaseTable& subcases() const;
    [[nodiscard]] const ParamTable& params() const;
    [[nodiscard]] const UnitTable& units() const;
    [[nodiscard]] const MassTable& masses() const;
    [[nodiscard]] const OutputRequests& outputRequests() const;
    [[nodiscard]] const CoordinateSystemTable& coordinateSystems() const;
    // The ply stacks of the laminates (laminatePliesTable) and their summary (laminateSummary), read by the shells.
    [[nodiscard]] const base::ContractReader& laminatePlies() const;
    [[nodiscard]] const base::ContractReader& laminateSummary() const;
    // The section properties of PBARL and PBEAML (sectionsTable) and the fastener piercings (the piercing points of
    // fastenerPiercings with the shape functions of the elements pierced).
    [[nodiscard]] const SectionTable& sections() const;
    [[nodiscard]] const FastenerPiercingTable& fastenerPiercings() const;
    // The SUPORT and SUPORT1 points (dofSetsTable), read through its contract by the solution drivers.
    [[nodiscard]] const base::ContractReader& supports() const;
    // The connection system of the units (componentNodes, connectionList), read through
    // their contracts by the reduction module.
    [[nodiscard]] const base::ContractReader& connectionNodes() const;
    [[nodiscard]] const base::ContractReader& connectionClusters() const;
    // The contact between bodies (contactNodePairs, contactPairsTable and, when given, contactTablesTable),
    // read by the glued contact of the assembly.
    [[nodiscard]] const ContactTable& contact() const;
    // The spiders of the components and the connections that touch each grid point (spiders,
    // connectionSharing), read through their contracts by the spider absorption of the reduction module.
    [[nodiscard]] const base::ContractReader& spiders() const;
    [[nodiscard]] const base::ContractReader& connectionSharing() const;
    // The DEFORM cards (deformations: setId, element, deformation), read by assembly::TemperatureField for the element
    // deformation set a subcase's DEFORM selects.
    [[nodiscard]] const base::ContractReader& deformations() const;
    // The aeroelastic model (aeroReferenceTable, aeroPanelsTable, aeroBoxesTable,
    // controlSurfacesTable, aeroTrimCasesTable, splinesTable, splineNodes) and its direct matrix input
    // (directMatrixShapes, directMatricesTable), read through their contracts by the aero module.
    [[nodiscard]] const base::ContractReader& aeroTable(TableRole role) const;

    // The terms of the loads a subcase's LOAD selects (LoadTable::termsOf): none without a set, the set alone with the
    // factor 1 without load tables. What the readers of the load tables by set (the load vectors, the inertial loads,
    // the SPCD of SpcSet) go through, so that the two forms of the tables give the same loads.
    [[nodiscard]] std::vector<LoadTerm> loadTermsOf(std::optional<std::int32_t> loadSetId) const;

    // A table by name, for checks and reports that walk every table: a role name ("singlePointConstrTable"),
    // "elements/<card>", "properties/<card>", "methods/<card>", "materials/<card>", "loadCards/<card>" or
    // "aeroCards/<card>", the card by its neutral name ("properties/ShellProp"); nullptr when the snapshot has no such
    // table.
    [[nodiscard]] const base::ContractReader* findTable(std::string_view name) const;
    // The names findTable() accepts for the tables the snapshot holds, roles first, then the card tables.
    [[nodiscard]] std::vector<std::string> tableNames() const;

private:
    friend class SnapshotBuilder;

    // What SnapshotBuilder assembles; the snapshot takes it whole.
    struct Parts
    {
        ModelVersions versions;
        OffsetDefinition offsetDefinition = OffsetDefinition::ElementOffsets;
        std::string unsupportedOffsetDefinition;
        std::vector<ElementFormulationChoice> elementFormulations;
        std::optional<NodeTable> nodes;
        std::vector<ElementBatch> elementBatches;
        std::vector<CardTable> propertyTables;
        std::vector<CardTable> methodTables;
        std::vector<CardTable> materialCardTables;
        std::vector<CardTable> loadCardTables;
        std::vector<CardTable> aeroCardTables;
        std::optional<MaterialTable> materials;
        std::optional<ConstraintTable> constraints;
        std::optional<RigidTable> rigidElements;
        std::optional<LoadTable> loads;
        std::optional<SubcaseTable> subcases;
        std::optional<ParamTable> params;
        std::optional<UnitTable> units;
        std::optional<MassTable> masses;
        std::optional<OutputRequests> outputRequests;
        std::optional<CoordinateSystemTable> coordinateSystems;
        std::optional<base::ContractReader> laminatePlies;
        std::optional<base::ContractReader> laminateSummary;
        std::optional<SectionTable> sections;
        std::optional<FastenerPiercingTable> fastenerPiercings;
        std::optional<base::ContractReader> supports;
        std::optional<base::ContractReader> connectionNodes;
        std::optional<base::ContractReader> connectionClusters;
        std::optional<ContactTable> contact;
        std::optional<base::ContractReader> spiders;
        std::optional<base::ContractReader> connectionSharing;
        std::optional<base::ContractReader> deformations;
        std::array<std::optional<base::ContractReader>, AERO_TABLE_COUNT> aero;
    };

    // Precondition: parts.nodes holds a value.
    explicit ModelSnapshot(Parts parts);

    ModelVersions m_versions;
    OffsetDefinition m_offsetDefinition = OffsetDefinition::ElementOffsets;
    std::string m_unsupportedOffsetDefinition;
    std::vector<ElementFormulationChoice> m_elementFormulations;
    NodeTable m_nodes;
    std::vector<ElementBatch> m_elementBatches;
    std::vector<CardTable> m_propertyTables;
    std::vector<CardTable> m_methodTables;
    std::vector<CardTable> m_materialCardTables;
    std::vector<CardTable> m_loadCardTables;
    std::vector<CardTable> m_aeroCardTables;
    std::optional<MaterialTable> m_materials;
    std::optional<ConstraintTable> m_constraints;
    std::optional<RigidTable> m_rigidElements;
    std::optional<LoadTable> m_loads;
    std::optional<SubcaseTable> m_subcases;
    std::optional<ParamTable> m_params;
    std::optional<UnitTable> m_units;
    std::optional<MassTable> m_masses;
    std::optional<OutputRequests> m_outputRequests;
    std::optional<CoordinateSystemTable> m_coordinateSystems;
    std::optional<base::ContractReader> m_laminatePlies;
    std::optional<base::ContractReader> m_laminateSummary;
    std::optional<SectionTable> m_sections;
    std::optional<FastenerPiercingTable> m_fastenerPiercings;
    std::optional<base::ContractReader> m_supports;
    std::optional<base::ContractReader> m_connectionNodes;
    std::optional<base::ContractReader> m_connectionClusters;
    std::optional<ContactTable> m_contact;
    std::optional<base::ContractReader> m_spiders;
    std::optional<base::ContractReader> m_connectionSharing;
    std::optional<base::ContractReader> m_deformations;
    std::array<std::optional<base::ContractReader>, AERO_TABLE_COUNT> m_aero; // in the order of AERO_TABLE_ROLES
};

} // namespace oos::input
