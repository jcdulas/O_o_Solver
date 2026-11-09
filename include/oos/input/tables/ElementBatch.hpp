#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/input/tables/ConnectivityView.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::input
{

// The elements of one card type (elements are processed in batches of one type, as the columns of O_o_Reader): the
// table of cardTable(deck, "<card>") (the neutral card name: ShellQuad4) with the card's fields and the row-index
// columns O_o_Reader computes (nodesRows: the row of each grid point in the node table; propIdRow and propType: the
// row of the property in the table of its card type). The columns every batch may have are checked at construction
// (elementBatchContract); the element's own code reads its other fields through reader().
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class ElementBatch
{
public:
    // Throws ContractError when the table breaks the element batch contract of `card` or has a null ID.
    ElementBatch(std::string card, base::ImportedTable table);

    [[nodiscard]] const std::string& card() const noexcept { return m_card; }
    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const;

    // The property of each element, for the cards that have one (not CONM2, CELAS2, RBE2...).
    [[nodiscard]] bool hasProperty() const;
    [[nodiscard]] std::span<const std::int32_t> propertyIds() const;
    // The row of each element's property in the table of its type, and that type; for the cards with a property.
    [[nodiscard]] bool hasPropertyRows() const { return m_reader.hasColumn("propIdRow"); }
    [[nodiscard]] std::span<const std::int32_t> propertyRows() const;
    [[nodiscard]] base::DictionaryView propertyTypes() const;

    // The grid point IDs of each element, for the cards with a "nodes" column (CBAR and the springs name theirs GA,
    // GB, G1, G2: their own code reads them). Nulls are allowed here (validity() of the reader tells them).
    [[nodiscard]] bool hasNodes() const { return m_reader.hasColumn("nodes"); }
    [[nodiscard]] ConnectivityView nodes() const { return connectivity("nodes"); }
    // The row of each grid point in the node table (NodeTable), valid for the structure_version of the snapshot.
    [[nodiscard]] bool hasNodeRows() const { return m_reader.hasColumn("nodesRows"); }
    [[nodiscard]] ConnectivityView nodeRows() const { return connectivity("nodesRows"); }

    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    [[nodiscard]] ConnectivityView connectivity(const std::string& column) const;

    std::string m_card;
    std::string m_idColumn;
    std::string m_propertyColumn;
    base::ContractReader m_reader;
};

} // namespace oos::input
