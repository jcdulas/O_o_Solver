#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace oos::input
{

// The constraints of the model, expanded by O_o_Reader:
// - singlePointConstrTable: one row per constrained grid point and SPC set (SPC, SPC1, SPCD, SPCADD expanded), and
//   the permanent constraints of the grid points (card PERMANENT_CONSTRAINT_CARD, set null: they apply to every set);
//   the card by its neutral name (SinglePointConstr, SinglePointConstrList, EnforcedMotionConstr); the components as
//   a text of digits ("123456"), the enforced value (null for a SinglePointConstrList and the permanent constraints);
// - multiPointConstrTable: one row per term of each MPC equation (MPCADD resolved), the dependent term flagged.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables (buffers borrowed).
// The card of the rows of singlePointConstrTable that hold the permanent constraints of the grid points (their PS
// field): not a card of the deck, so the name table has no keyword for it.
inline constexpr std::string_view PERMANENT_CONSTRAINT_CARD = "gridPs";

class ConstraintTable
{
public:
    ConstraintTable(base::ImportedTable spc, base::ImportedTable mpc);

    [[nodiscard]] std::int64_t spcRowCount() const noexcept { return m_spc.rowCount(); }
    // The set of each row; null for the permanent constraints (validity() of spcReader()).
    [[nodiscard]] std::span<const std::int32_t> spcSetIds() const
    {
        return m_spc.values<std::int32_t>("setId", base::NullPolicy::Allow);
    }
    [[nodiscard]] std::span<const std::int32_t> spcNodeIds() const { return m_spc.values<std::int32_t>("nodeId"); }
    [[nodiscard]] base::Utf8View spcComponents() const { return m_spc.texts("dof"); }
    [[nodiscard]] base::DictionaryView spcCards() const { return m_spc.dictionaryTexts("card"); }
    // The enforced values; read validity() of spcReader() for the rows without one.
    [[nodiscard]] std::span<const double> enforcedValues() const
    {
        return m_spc.values<double>("enforced", base::NullPolicy::Allow);
    }

    [[nodiscard]] std::int64_t mpcRowCount() const noexcept { return m_mpc.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> mpcSetIds() const { return m_mpc.values<std::int32_t>("setId"); }
    [[nodiscard]] std::span<const std::int32_t> mpcEquations() const { return m_mpc.values<std::int32_t>("equation"); }
    // The position of each term in its equation, from 1; term 1 is the dependent DOF (isDependent).
    [[nodiscard]] std::span<const std::int32_t> mpcTerms() const { return m_mpc.values<std::int32_t>("term"); }
    [[nodiscard]] std::span<const std::int32_t> mpcNodeIds() const { return m_mpc.values<std::int32_t>("nodeId"); }
    // The component of each term: one digit 1 to 6, 0 for a scalar point.
    [[nodiscard]] std::span<const std::int32_t> mpcComponents() const { return m_mpc.values<std::int32_t>("dof"); }
    [[nodiscard]] std::span<const double> mpcCoefficients() const { return m_mpc.values<double>("coefficient"); }

    [[nodiscard]] const base::ContractReader& spcReader() const noexcept { return m_spc; }
    [[nodiscard]] const base::ContractReader& mpcReader() const noexcept { return m_mpc; }

private:
    base::ContractReader m_spc;
    base::ContractReader m_mpc;
};

} // namespace oos::input
