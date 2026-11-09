#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace oos::input
{

// The value of one PARAM, in the column its kind names.
struct ParamValue
{
    enum class Kind
    {
        Integer,
        Real,
        Text
    };
    Kind kind = Kind::Integer;
    std::int32_t integer = 0;
    double real = 0.0;
    std::string_view text; // valid as long as the ParamTable lives
};

// The PARAM cards of the model, typed (O_o_Reader paramsTable, data contract 31): one row per
// PARAM name, by its neutral name (WeightToMass for WTMASS; a name the dictionary of the deck does not know as the deck
// writes it, in upper case), of the Bulk Data (subcase null) or of a Case Control section (subcase 0 the global
// section), with its kind
// ("integer", "real", "text", "complex", "blank") and its value in the matching column. A blank PARAM reads as not
// given; a complex one by its real part.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class ParamTable
{
public:
    explicit ParamTable(base::ImportedTable params);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }

    // The value of a parameter for a subcase: the PARAM of that subcase's Case Control if it has one, else that of
    // the global Case Control section, else that of the Bulk Data (a Case Control PARAM overrides the Bulk Data
    // one); nothing when the parameter is not given (the caller applies the default). Without a subcase, the global
    // Case Control or Bulk Data value. `name` is the neutral name, compared exactly ("WeightToMass").
    [[nodiscard]] std::optional<ParamValue> find(std::string_view name,
                                                 std::optional<std::int32_t> subcase = std::nullopt) const;
    // The first subcase whose own Case Control gives the parameter a value (not the global section); nothing when
    // none does. For the parameters that hold for the whole model, which a subcase cannot change.
    [[nodiscard]] std::optional<std::int32_t> findSubcaseGiving(std::string_view name) const;

    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    [[nodiscard]] ParamValue valueAt(std::int64_t row) const;

    base::ContractReader m_reader;
};

} // namespace oos::input
