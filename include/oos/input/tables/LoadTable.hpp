#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::input
{

// A load set whose cards a subcase's LOAD brings, and the factor of their values: a LOAD entry defines the load
// S·Σ Si·Li, so the set Li comes with S·Si (through a nested LOAD entry, the product of the factors on the way;
// summed when the entry reaches the set twice): O_o_Reader's combinedFactor.
struct LoadTerm
{
    std::int32_t set = 0;
    double factor = 1.0;
};

// The static loads of the model, scaled by O_o_Reader, one reader per table:
// - loadsTable: one row per elementary load of each load set (`scale` the product of the LOAD factors, `sourceSetId`
//   the set of the card): FORCE, MOMENT and their variants per grid point with the force and moment in the basic
//   system, GRAV as an acceleration in the basic system, the first pressure of PLOAD*, RFORCE with no values;
// - pressureTable: PLOAD, PLOAD2, PLOAD4 per element face, or its fields (O_o_Reader's detail="fields": a row
//   per card and its element runs; isPressureFields(), read row by row through assembly::PressureRows);
// - lineLoadsTable: PLOAD1;
// - temperatureTable: TEMP, TEMPD, TEMPP1, ...;
// - inertialLoadsTable (optional): GRAV, ACCEL, ACCEL1, RFORCE with all their fields;
// - loadCombinationsTable (optional): the terms of every LOAD entry.
//
// Two forms. Without loadCombinationsTable the tables expand every LOAD combination: the rows of a
// combination are listed under its SID with their factors applied (`scale`), and a set's loads are its rows. With it
// (O_o_Reader's expandCombinations=False) each load set is listed once, its own cards with the scale 1, and the loads
// a LOAD selects are made from its terms (termsOf): the rows of each set it reaches times the set's factor. The SPCD
// rows of singlePointConstrTable take the same form as these tables (ConstraintTable; SnapshotBuilder checks that they
// agree).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables (buffers borrowed).
class LoadTable
{
public:
    // Throws ContractError when loadCombinationsTable is not sorted by `load`, or when it is given with tables that
    // expand the combinations (a row of loadsTable or inertialLoadsTable whose `setId` is not its `sourceSetId`).
    // `isPressureFields`: `pressures` is pressureTable(detail="fields") (TableRole::PressureFields).
    LoadTable(base::ImportedTable loads, base::ImportedTable pressures, base::ImportedTable lineLoads,
              base::ImportedTable temperatures, std::optional<base::ImportedTable> inertialLoads,
              std::optional<base::ImportedTable> combinations, bool isPressureFields = false);

    [[nodiscard]] std::int64_t loadRowCount() const noexcept { return m_loads.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> loadSetIds() const { return m_loads.values<std::int32_t>("setId"); }
    [[nodiscard]] base::DictionaryView loadCards() const { return m_loads.dictionaryTexts("card"); }
    [[nodiscard]] std::span<const double> loadScales() const { return m_loads.values<double>("scale"); }

    [[nodiscard]] const base::ContractReader& loads() const noexcept { return m_loads; }
    // pressureTable in its form (isPressureFields()).
    [[nodiscard]] const base::ContractReader& pressures() const noexcept { return m_pressures; }
    [[nodiscard]] bool isPressureFields() const noexcept { return m_isPressureFields; }
    [[nodiscard]] const base::ContractReader& lineLoads() const noexcept { return m_lineLoads; }
    [[nodiscard]] const base::ContractReader& temperatures() const noexcept { return m_temperatures; }

    [[nodiscard]] bool hasInertialLoads() const noexcept { return m_inertialLoads.has_value(); }
    // Throws ContractError naming inertialLoadsTable when the snapshot has no inertial load table.
    [[nodiscard]] const base::ContractReader& inertialLoads() const;

    // Whether the tables list each load set once, the LOAD combinations as terms (loadCombinationsTable given).
    [[nodiscard]] bool hasCombinationTerms() const noexcept { return m_combinations.has_value(); }
    // Throws ContractError naming loadCombinationsTable when the snapshot has no combination table.
    [[nodiscard]] const base::ContractReader& combinations() const;
    // Whether `setId` is a LOAD entry of the combination table (false with expanded tables).
    [[nodiscard]] bool isCombination(std::int32_t setId) const;
    // The terms whose rows make up the loads that load set `setId` selects, in the order their values are added: the
    // set itself with the factor 1 first (its own rows: a set of its own, the expanded rows of a combination, or the
    // cards a LOAD set holds besides its entry, which the input format does not foresee), then, with the combination
    // table, each set the LOAD entry `setId` reaches with its combinedFactor, in the table's order (by set). Bounded by
    // the sets of one combination; O(log n) in the table's rows.
    [[nodiscard]] std::vector<LoadTerm> termsOf(std::int32_t setId) const;

private:
    bool m_isPressureFields = false;
    base::ContractReader m_loads;
    base::ContractReader m_pressures;
    base::ContractReader m_lineLoads;
    base::ContractReader m_temperatures;
    std::optional<base::ContractReader> m_inertialLoads;
    std::optional<base::ContractReader> m_combinations;
    // The columns of the combination table, read once (empty without it).
    std::span<const std::int32_t> m_combinationLoads;
    std::span<const std::int32_t> m_combinationSets;
    std::span<const double> m_combinationFactors;
};

} // namespace oos::input
