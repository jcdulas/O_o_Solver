#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace oos::input
{

// The subcases of the Case Control, effective (subcasesTable of O_o_Reader): one row per subcase with what it
// selects (SPC, MPC, LOAD, METHOD, TEMP(LOAD), TEMP(INIT) set IDs, null when not selected), the global section
// included as subcase 0 of kind "global" and its selections inherited by the subcases (O_o_Reader applies the
// inheritance: a subcase takes every global selection it does not override). The kinds are camelCase names: "subcase"
// (SUBCASE), "subcaseCombination" (SUBCOM), "auxiliary" (AUXCASE)... A subcaseCombination row gives the subcases it
// combines and their coefficients (SUBSEQ; columns combinedSubcases and coefficients, optional in the contract).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class SubcaseTable
{
public:
    explicit SubcaseTable(base::ImportedTable subcases);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_reader.values<std::int32_t>("subcase"); }
    [[nodiscard]] base::DictionaryView kinds() const { return m_reader.dictionaryTexts("kind"); }

    // The set a subcase selects with a Case Control command ("spc", "mpc", "load", "dynamicLoadSelection", "method",
    // "temperatureLoad", "temperatureInitial", "nonStructMass"), or nothing when it selects none. Throws
    // std::invalid_argument for another command or a subcase that does not exist.
    [[nodiscard]] std::optional<std::int32_t> selectedSet(std::int32_t subcase, const std::string& command) const;

    // The set the subcase of row `row` selects (same commands as selectedSet), or nothing. Throws
    // std::invalid_argument for another command or a row out of range.
    [[nodiscard]] std::optional<std::int32_t> selectedSetAtRow(std::int64_t row, const std::string& command) const;

    // Whether the table gives the contact table of each subcase (column bcontact).
    [[nodiscard]] bool hasContactColumn() const { return m_reader.hasColumn("contactSelection"); }
    // The contact table the subcase of row `row` selects with BCONTACT (O_o_Reader gives it effective),
    // or nothing when it selects none or the table has no bcontact column (hasContactColumn says which). Throws
    // std::invalid_argument for a row out of range.
    [[nodiscard]] std::optional<std::int32_t> contactTableAtRow(std::int64_t row) const;

    // The element deformation set the subcase of row `row` selects with DEFORM (effective,
    // the global section's inherited), or nothing when it selects none or the table has no deform column (an
    // int32 column, or the float64 one the snapshot adds from the Case Control, NaN for none). Throws
    // std::invalid_argument for a row out of range.
    [[nodiscard]] std::optional<std::int32_t> deformationSetAtRow(std::int64_t row) const;

    // The row of a subcase, or -1 when there is none.
    [[nodiscard]] std::int64_t findRow(std::int32_t subcase) const;

    // The subcases a SUBCOM row combines and their coefficients (SUBSEQ), as O_o_Reader gives them (both
    // empty for another row, or when the table has no such columns). Throws ContractError when the two lists of a
    // row differ in length.
    struct Combination
    {
        std::span<const std::int32_t> subcases;
        std::span<const double> coefficients;
    };
    [[nodiscard]] Combination combinationAtRow(std::int64_t row) const;

    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
