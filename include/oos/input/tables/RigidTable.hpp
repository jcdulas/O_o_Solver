#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <span>

namespace oos::input
{

// The rigid elements of the model, expanded by O_o_Reader (rigidElementsTable): one row per grid point of each
// RBAR, RBAR1, RBE1, RBE2, RBE3, RROD, RTRPLT... with its role ("independent", "dependent", and for an RBE3 the
// weighted legs), its components as a text of digits, and the RBE3 weight (null for the other cards). The card fields
// themselves (ALPHA, TREF) are in the element batch of the card.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class RigidTable
{
public:
    explicit RigidTable(base::ImportedTable rigidElements);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_reader.values<std::int32_t>("elementId"); }
    [[nodiscard]] base::DictionaryView types() const { return m_reader.dictionaryTexts("type"); }
    [[nodiscard]] base::DictionaryView roles() const { return m_reader.dictionaryTexts("role"); }
    [[nodiscard]] std::span<const std::int32_t> nodeIds() const { return m_reader.values<std::int32_t>("nodeId"); }
    // The components of each row; empty for a row without (the independent end of an RROD, whose dof is null).
    [[nodiscard]] base::Utf8View components() const { return m_reader.texts("dof", base::NullPolicy::Allow); }
    [[nodiscard]] std::span<const double> weights() const
    {
        return m_reader.values<double>("weight", base::NullPolicy::Allow);
    }
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
