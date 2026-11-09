#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::input
{

// The table of one card type keyed by an integer ID: a property card (ShellProp by "id", BarProp by "propId"), a
// method card (LanczosEigenMethod by "setId"); card() is the neutral card name. The ID column is checked at
// construction; the other fields are read through reader() by the code that knows the card (an element's property
// reader), each with its own column contract.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (its buffers stay the
// producer's, borrowed).
class CardTable
{
public:
    // Throws ContractError when the table has no int32 column `idColumn`, or nulls in it.
    CardTable(std::string card, std::string idColumn, base::ImportedTable table);

    [[nodiscard]] const std::string& card() const noexcept { return m_card; }
    [[nodiscard]] const std::string& idColumn() const noexcept { return m_idColumn; }
    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_reader.values<std::int32_t>(m_idColumn); }
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    std::string m_card;
    std::string m_idColumn;
    base::ContractReader m_reader;
};

} // namespace oos::input
