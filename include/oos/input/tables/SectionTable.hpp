#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::input
{

// The rows [first, last) of one property in the sections table.
struct SectionRows
{
    std::int64_t first = 0;
    std::int64_t last = 0;

    [[nodiscard]] std::int64_t count() const noexcept { return last - first; }
};

// The section properties of the PBARL and PBEAML cards, computed from the dimensions of their section type
// (O_o_Reader sectionsTable(types=["PBARL", "PBEAML"]), data contract 31): one row per PBARL, one
// row per PBEAML station, in the element system (y, z) about the neutral axis, with the shear factors K1, K2 and the
// stress recovery points C, D, E, F as (y, z) pairs (the columns c1 c2 d1 d2 e1 e2 f1 f2).
//
// Invariant (checked at construction): the rows are sorted by propId, then by xxb, so that the stations of a property
// are contiguous and in increasing order, and findProperty() is a binary search (no index is built).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class SectionTable
{
public:
    // Throws ContractError when the table breaks its contract or its rows are not sorted by propId, then xxb.
    explicit SectionTable(base::ImportedTable sections);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> propIds() const { return m_reader.values<std::int32_t>("propId"); }
    // X/XB of each row: 0 at end A, 1 at end B (0 for a PBARL).
    [[nodiscard]] std::span<const double> positions() const { return m_reader.values<double>("xxb"); }
    [[nodiscard]] std::span<const double> areas() const { return m_reader.values<double>("area"); }
    [[nodiscard]] std::span<const double> inertias1() const { return m_reader.values<double>("i1"); }
    [[nodiscard]] std::span<const double> inertias2() const { return m_reader.values<double>("i2"); }
    [[nodiscard]] std::span<const double> productsOfInertia() const { return m_reader.values<double>("i12"); }
    [[nodiscard]] std::span<const double> torsionalConstants() const { return m_reader.values<double>("j"); }
    [[nodiscard]] std::span<const double> nonStructuralMasses() const
    {
        return m_reader.values<double>("nonStructMass");
    }
    [[nodiscard]] std::span<const double> shearFactors1() const { return m_reader.values<double>("k1"); }
    [[nodiscard]] std::span<const double> shearFactors2() const { return m_reader.values<double>("k2"); }
    // N1, N2: the neutral axis of each row from its shear centre, or empty spans when the table has no such
    // columns (every shear centre on its neutral axis).
    [[nodiscard]] std::span<const double> neutralAxes1() const { return optionalValues("n1"); }
    [[nodiscard]] std::span<const double> neutralAxes2() const { return optionalValues("n2"); }
    // C1, C2, D1, D2, E1, E2, F1, F2 of each row (8 values per row, gathered from their eight columns at
    // construction).
    [[nodiscard]] base::FixedListView<double> recoveryPoints() const
    {
        return {m_recoveryPoints.data(), rowCount(), RECOVERY_VALUE_COUNT};
    }

    // The rows of a property, or nothing when the table has no row for it.
    [[nodiscard]] std::optional<SectionRows> findProperty(std::int32_t propId) const;

    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    // The four recovery points C, D, E, F as (y, z): eight values per row.
    static constexpr std::int64_t RECOVERY_VALUE_COUNT = 8;

    [[nodiscard]] std::span<const double> optionalValues(const char* name) const
    {
        return m_reader.hasColumn(name) ? m_reader.values<double>(name) : std::span<const double>();
    }

    base::ContractReader m_reader;
    // The recovery points row after row. A copy, so that the elements read them as one fixed list per row (the table
    // gives them as eight columns): 64 bytes per section row, bounded by the PBARL and PBEAML stations (std::vector).
    std::vector<double> m_recoveryPoints;
};

} // namespace oos::input
