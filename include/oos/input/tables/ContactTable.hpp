#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// The glue parameters in force for one pair of contact bodies (the glue fields of BCONPRG and of a BCTABLE section, as
// O_o_Reader resolves them from the pair's BCONPRG/BCONPRP, the fields of its BCTABLE section and the BCPARA defaults).
// Nothing for a field O_o_Reader leaves null; an IGLUE left null is blank, that is 0: touching contact.
struct GlueParameters
{
    std::optional<std::int32_t> glueOption;
    std::optional<std::int32_t> glueSeparation;
    std::optional<std::int32_t> coordinateCheck;
    std::optional<std::int32_t> searchOption;
    std::optional<double> errorTolerance;
    std::optional<double> toleranceBias;
    // the card that sets IGLUE ("ContactParams 7", "ContactTable 5", "default"); may be empty
    std::string glueOptionSource;
};

// One row of contactNodePairs: a grid point of the secondary surface of a pair of bodies and a face of the primary
// surface near it, as O_o_Reader's own contact rule (rule="components") finds them. The glue reads these rows as the
// definition of the two surfaces near each other (the candidate secondary grid points, the primary faces), never as
// the tie itself: GlueConstraints projects every candidate point on every candidate face of its pair itself
// (CalculiX's tie). `face` holds the grid points of the face in the element's order: the corners in their cyclic order,
// then, on a quadratic face, the mid-side points in the order of the edges (corner 1-2, 2-3, ...), 0 for a mid-side
// point the element leaves out.
struct ContactNodePair
{
    std::int64_t row = 0;
    std::string_view table; // the card and ID that pair the bodies; empty for a pairing without table
    std::int32_t secondaryBody = 0;
    std::int32_t primaryBody = 0;
    std::int32_t nodeId = 0;
    std::optional<std::int32_t> primaryElementId; // nothing on a BLSEG segment
    std::span<const std::int32_t> face;
    bool isSearchedFromPrimary = false; // a row of the search from the primary body: its bodies are swapped
};

// The contact between bodies as O_o_Reader gives it, read for glued contact: the node pairs of every pair of bodies
// (contactNodePairs, the surfaces near each other), the pairs with their kind and glue parameters
// (contactPairsTable), and, when given, the pairs each BCONTACT ID selects (contactTablesTable). Contract 38 gives
// the pairs and the node pairs without the columns the glue needs (the face, the search side, IGLUE):
// missingGlueColumns() lists those absent, and a glued contact is refused with them (assembly::requireGlueContract),
// never guessed.
//
// Which pairs a subcase glues: BCONTACT = n selects the contact table n of the Bulk Data: a BCTABLE n, whose pairs are
// those whose `table` is "ContactTable n", or a BCTABL1 n, which lists BCONECT entries, whose pairs only
// contactTablesTable names; without that table, a BCONTACT that matches no BCTABLE while the model has BCONECT pairs
// is a ContractError naming contactTablesTable (selectedPairRows).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables (buffers borrowed).
class ContactTable
{
public:
    // Throws ContractError when a table breaks its contract.
    ContactTable(base::ImportedTable nodePairs, base::ImportedTable pairs, std::optional<base::ImportedTable> tables);

    [[nodiscard]] std::int64_t pairCount() const noexcept { return m_pairs.rowCount(); }
    [[nodiscard]] std::int64_t nodePairCount() const noexcept { return m_nodePairs.rowCount(); }
    [[nodiscard]] bool hasContactTables() const noexcept { return m_tables.has_value(); }

    // The columns a glued contact needs that the tables lack, as "table.column", in a fixed order; empty when
    // every one is there.
    [[nodiscard]] std::vector<std::string> missingGlueColumns() const;

    // Pair row `row`: its table text (empty when null), its bodies, its kind, its node pair count, its glue parameters.
    [[nodiscard]] std::string_view pairTable(std::int64_t row) const;
    [[nodiscard]] std::int32_t pairSecondaryBody(std::int64_t row) const;
    [[nodiscard]] std::int32_t pairPrimaryBody(std::int64_t row) const;
    [[nodiscard]] std::string_view pairKind(std::int64_t row) const;
    [[nodiscard]] std::int64_t pairNodeCount(std::int64_t row) const;
    // Precondition: missingGlueColumns() is empty (else the parameters are all nothing).
    [[nodiscard]] GlueParameters glueParameters(std::int64_t row) const;

    // Node pair row `row`. Precondition: missingGlueColumns() is empty (else the face is empty).
    [[nodiscard]] ContactNodePair nodePair(std::int64_t row) const;
    // Whether node pair `row` names a face (a null primaryFace: O_o_Reader found none).
    [[nodiscard]] bool hasFace(std::int64_t row) const;

    // The pair rows a BCONTACT ID selects, increasing. Throws ContractError (naming contactTablesTable) when the
    // model has BCONECT pairs and no contactTablesTable, and the ID is not that of a BCTABLE.
    [[nodiscard]] std::vector<std::int64_t> selectedPairRows(std::int32_t bcontactId) const;

    [[nodiscard]] const base::ContractReader& nodePairReader() const noexcept { return m_nodePairs; }
    [[nodiscard]] const base::ContractReader& pairReader() const noexcept { return m_pairs; }
    [[nodiscard]] const base::ContractReader* tableReader() const noexcept { return m_tables ? &*m_tables : nullptr; }

private:
    base::ContractReader m_nodePairs;
    base::ContractReader m_pairs;
    std::optional<base::ContractReader> m_tables;
};

// The card and ID of a contact table text ("ContactTable 5" gives {"ContactTable", 5}, the neutral card name); nothing
// for a text of another form.
struct ContactTableName
{
    std::string card;
    std::int32_t id = 0;
};
[[nodiscard]] std::optional<ContactTableName> parseContactTableName(std::string_view text);

} // namespace oos::input
