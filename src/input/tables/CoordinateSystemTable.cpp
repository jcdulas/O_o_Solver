#include "oos/input/tables/CoordinateSystemTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace oos::input
{

namespace
{

CoordinateFrame::Kind kindOf(std::string_view text)
{
    if (text == "R")
        return CoordinateFrame::Kind::Rectangular;
    if (text == "C")
        return CoordinateFrame::Kind::Cylindrical;
    return CoordinateFrame::Kind::Spherical;
}

} // namespace

CoordinateSystemTable::CoordinateSystemTable(base::ImportedTable systems)
    : m_reader(std::move(systems), contractOf(TableRole::CoordinateSystems))
{
    const base::DictionaryView kinds = m_reader.dictionaryTexts("type");
    const std::span<const std::int32_t> frameIds = ids();
    for (std::int64_t row = 0; row < rowCount(); ++row)
    {
        const std::string_view kind = kinds[row];
        if (kind != "R" && kind != "C" && kind != "S")
            throw base::ContractError(
                base::concatenate({"coordsTable: expected the kind R, C or S, found '", kind,
                                   "' for coordinate system ", std::to_string(frameIds[static_cast<std::size_t>(row)]),
                                   "; the table does not come from O_o_Reader's coordsTable query: take the "
                                   "snapshot again from O_o_Reader's query"}));
    }
}

CoordinateFrame CoordinateSystemTable::frameAt(std::int64_t row) const
{
    const auto value = [&](const char* column)
    { return m_reader.values<double>(column)[static_cast<std::size_t>(row)]; };
    CoordinateFrame frame;
    frame.kind = kindOf(m_reader.dictionaryTexts("type")[row]);
    frame.origin = {value("originX"), value("originY"), value("originZ")};
    const std::array<base::Vec3, 3> axes{base::Vec3{value("iX"), value("iY"), value("iZ")},
                                         base::Vec3{value("jX"), value("jY"), value("jZ")},
                                         base::Vec3{value("kX"), value("kY"), value("kZ")}};
    // The axes are the columns of the rotation (local to basic), hence the transpose of the matrix of rows.
    frame.axes = base::matrixFromRows(axes[0], axes[1], axes[2]).transposed();
    return frame;
}

} // namespace oos::input
