#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <span>

namespace oos::input
{

// The rigid elements of the model, expanded by O_o_Model (rigidElementsTable): one row per grid point of each
// RigidBar, RigidBar1, RigidBodyMulti, RigidBodyStar, RigidInterpolation, RigidRod, RigidTriaPlate... with its role
// ("independent", "dependent", and for a RigidInterpolation the weighted legs), its components as a text of digits,
// and the RigidInterpolation weight (null for the other cards). The optional column `alpha` is the thermal expansion
// coefficient of the element (its ALPHA), repeated on each of its rows (null for a card that has none; absent from a
// table of an earlier O_o_Model or built by hand): the solver does not compute the expansion of a rigid element, and
// reads it only to refuse it under a thermal load (assembly::LoadVectorBuilder::checkThermalLoads).
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
    // The components of each row; empty for a row without (the independent end of a RigidRod, whose dof is null).
    [[nodiscard]] base::Utf8View components() const { return m_reader.texts("dof", base::NullPolicy::Allow); }
    [[nodiscard]] std::span<const double> weights() const
    {
        return m_reader.values<double>("weight", base::NullPolicy::Allow);
    }
    // Whether the table gives the thermal expansion coefficient of the elements (its optional column `alpha`), and,
    // when it does, the coefficient of the element of each row with the rows that have one.
    [[nodiscard]] bool hasExpansionCoefficients() const { return m_reader.hasColumn("alpha"); }
    [[nodiscard]] std::span<const double> expansionCoefficients() const
    {
        return m_reader.values<double>("alpha", base::NullPolicy::Allow);
    }
    [[nodiscard]] base::ValidityView expansionValidity() const { return m_reader.validity("alpha"); }
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
