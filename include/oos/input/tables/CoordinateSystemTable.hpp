#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"

#include <cstdint>
#include <span>

namespace oos::input
{

// One coordinate system, resolved to the basic system by O_o_Reader (CORD1x, CORD2x and their RID chains): its kind,
// its origin and its axes in the basic system. For a cylindrical (C) or spherical (S) system the axes are those of the
// rectangular system the card defines (i, j, k from the points A, B, C of a CORD2C); the local directions at a point
// (radial, tangential, axial) depend on the point.
struct CoordinateFrame
{
    enum class Kind : std::uint8_t
    {
        Rectangular,
        Cylindrical,
        Spherical
    };

    Kind kind = Kind::Rectangular;
    base::Vec3 origin;
    // Columns are the axes i, j, k in the basic system: v_basic = axes · v_local for a rectangular system.
    base::Mat3 axes = base::Mat3::identity();
};

// The coordinate systems of the model (coordsTable of O_o_Reader): one row per CORDij card, with the origin and the
// axes i, j, k already resolved to the basic system. The elements read it for the systems their cards name (CBUSH CID
// and OCID, CONM1 and CONM2 CID); the basic system (CID 0) has no row.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table (buffers borrowed).
class CoordinateSystemTable
{
public:
    // Throws ContractError when the table breaks its contract or holds a kind other than "R", "C", "S".
    explicit CoordinateSystemTable(base::ImportedTable systems);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_reader.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_reader.values<std::int32_t>("frameId"); }
    [[nodiscard]] CoordinateFrame frameAt(std::int64_t row) const;
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
};

} // namespace oos::input
