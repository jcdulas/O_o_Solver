#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <span>

namespace oos::input
{

// The materials of the model (materialsTable of O_o_Reader): one row per material card of any type, with the
// constants of MAT1 (e, g, nu, rho, a, tref, ge) and of MAT8 (e1, e2, nu12, g12) in shared columns, null where a type
// has no such constant. The element code reads the constants it needs with NullPolicy::Allow and refuses a null it
// cannot accept, as a diagnostic.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class MaterialTable
{
public:
    explicit MaterialTable(base::ImportedTable materials);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_reader.values<std::int32_t>("matId"); }
    [[nodiscard]] base::DictionaryView types() const { return m_reader.dictionaryTexts("type"); }
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
