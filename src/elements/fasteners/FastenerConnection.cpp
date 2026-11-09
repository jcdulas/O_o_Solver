#include "oos/elements/fasteners/FastenerConnection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/fasteners/PiercingInterpolation.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <variant>

namespace oos::elements
{

namespace
{

constexpr std::array<const char*, 2> SIDE_FIELDS{"patchIdA", "patchIdB"};

// The local index of a grid row in `rows`, appended when new.
std::size_t localIndex(std::vector<std::int32_t>& rows, std::int32_t gridRow)
{
    const auto found = std::find(rows.begin(), rows.end(), gridRow);
    if (found != rows.end())
        return static_cast<std::size_t>(found - rows.begin());
    rows.push_back(gridRow);
    return rows.size() - 1;
}

Resolution<SideInterpolation> resolveSide(const ElementBatchView& view, std::int64_t row,
                                          const input::FastenerPiercingTable* table, input::PiercingRows rows,
                                          std::size_t side, std::int32_t directGrid)
{
    using Result = Resolution<SideInterpolation>;
    const char* sideName = side == 0 ? "A" : "B";
    if (table == nullptr || rows.isEmpty())
    {
        if (directGrid == 0)
        {
            // The reader's reason when it did not find the point: the field of the card and its tolerance.
            const input::SideFailure* failure =
                table != nullptr ? table->findFailure(view.elementId(row), side == 0) : nullptr;
            if (failure != nullptr)
                return Result::failure(elementProblem(
                    view, row, problem_code::MISSING_POINT, failure->field,
                    base::concatenate({"no piercing point on side ", sideName, ": ", failure->message})));
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, SIDE_FIELDS[side],
                base::concatenate({"expected the piercing point of side ", sideName,
                                   " in fastenerPiercings with the grid points of the shell element it pierces and "
                                   "their shape functions, found no row for it; take the snapshot again with the "
                                   "fastenerPiercings table of the same model"})));
        }
        Resolution<std::int32_t> gridRow = resolveGridRow(view, row, directGrid, side == 0 ? "nodeA" : "nodeB");
        if (!gridRow)
            return Result::failure(std::move(gridRow.problem));
        SideInterpolation result;
        result.point = view.positionInBasic(*gridRow.value);
        result.gridRows = {*gridRow.value};
        result.map = directGridMap(view.displacementFrame(*gridRow.value));
        return Result::success(std::move(result));
    }
    const SideGridLookup findGrid = [&view](std::int32_t gridId) -> std::optional<SideGrid>
    {
        const std::optional<std::int32_t> gridRow = view.index().points().findGrid(gridId);
        if (!gridRow)
            return std::nullopt;
        return SideGrid{*gridRow, view.positionInBasic(*gridRow), view.displacementFrame(*gridRow)};
    };
    std::variant<SideInterpolation, std::string> interpolation = interpolateSide(*table, rows, findGrid);
    if (const auto* problem = std::get_if<std::string>(&interpolation))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, SIDE_FIELDS[side],
                           base::concatenate({"pierced element of side ", sideName, ": ", *problem})));
    return Result::success(std::move(std::get<SideInterpolation>(interpolation)));
}

} // namespace

std::vector<ElementDof> FastenerConnection::dofs() const
{
    std::vector<ElementDof> result;
    result.reserve(dofCount());
    for (const std::int32_t gridRow : gridRows)
        for (int c = 1; c <= 6; ++c)
            result.push_back(ElementDof::grid(gridRow, c));
    return result;
}

std::vector<double> FastenerConnection::condense(const base::MatN<12>& k) const
{
    const std::size_t n = dofCount();
    // k·M first (12 × n), then Mᵀ·(k·M), summed in a fixed order.
    std::vector<double> km(12 * n, 0.0);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 12; ++r)
                sum += k(i, r) * map[r * n + j];
            km[i * n + j] = sum;
        }
    std::vector<double> result(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < 12; ++r)
                sum += map[r * n + i] * km[r * n + j];
            result[i * n + j] = sum;
        }
    // Exactly symmetric: the two triangles round differently and the assembler may read one only.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            result[i * n + j] = result[j * n + i] = 0.5 * (result[i * n + j] + result[j * n + i]);
    return result;
}

std::array<double, 12> FastenerConnection::pointMotions(const NodalDisplacements& displacements) const
{
    const std::size_t n = dofCount();
    std::vector<double> u(n);
    for (std::size_t g = 0; g < gridRows.size(); ++g)
        for (int c = 0; c < 6; ++c)
            u[6 * g + static_cast<std::size_t>(c)] = displacements.valueAt(ElementDof::grid(gridRows[g], c + 1));
    std::array<double, 12> w{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            sum += map[i * n + j] * u[j];
        w[i] = sum;
    }
    return w;
}

Resolution<FastenerConnection> resolveFastenerConnection(const ElementBatchView& view, std::int64_t row,
                                                         std::array<std::int32_t, 2> directGrids)
{
    using Result = Resolution<FastenerConnection>;
    const bool isDirect = directGrids[0] != 0 && directGrids[1] != 0;
    if (!isDirect && !view.snapshot().hasTable(input::TableRole::FastenerPiercings))
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "propId",
            base::concatenate({"expected the piercing points of the fastener in the snapshot's fastenerPiercings "
                               "table (O_o_Reader data contract 60 or later), found no such table: hand the table "
                               "over to the snapshot"})));
    // An ALIGN Weld reads no table: both sides connect directly.
    const input::FastenerPiercingTable* table = isDirect ? nullptr : &view.snapshot().fastenerPiercings();
    const std::optional<input::FastenerPiercingRows> rows =
        table != nullptr ? table->findFastener(view.elementId(row)) : std::nullopt;
    const input::FastenerPiercingRows sides = rows.value_or(input::FastenerPiercingRows{});
    Resolution<SideInterpolation> sideA = resolveSide(view, row, table, sides.sideA, 0, directGrids[0]);
    if (!sideA)
        return Result::failure(std::move(sideA.problem));
    Resolution<SideInterpolation> sideB = resolveSide(view, row, table, sides.sideB, 1, directGrids[1]);
    if (!sideB)
        return Result::failure(std::move(sideB.problem));
    if (base::norm(sideB.value->point - sideA.value->point) <= FASTENER_COINCIDENT_DISTANCE)
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, "nodeA",
            "expected distinct piercing points A and B, found them coincident; the fastener axis is undefined: "
            "separate the sheets or give Fastener.nodeA and Fastener.nodeB"));

    FastenerConnection connection;
    connection.pointA = sideA.value->point;
    connection.pointB = sideB.value->point;
    std::array<std::vector<std::size_t>, 2> localOf;
    for (std::size_t s = 0; s < 2; ++s)
        for (const std::int32_t gridRow : (s == 0 ? sideA : sideB).value->gridRows)
            localOf[s].push_back(localIndex(connection.gridRows, gridRow));
    const std::size_t n = connection.dofCount();
    connection.map.assign(12 * n, 0.0);
    for (std::size_t s = 0; s < 2; ++s)
    {
        const PiercingMap& map = (s == 0 ? sideA : sideB).value->map;
        for (std::size_t i = 0; i < 6; ++i)
            for (std::size_t g = 0; g < map.gridCount; ++g)
                for (std::size_t c = 0; c < 6; ++c)
                    connection.map[(6 * s + i) * n + 6 * localOf[s][g] + c] += map.at(i, 6 * g + c);
    }
    return Result::success(std::move(connection));
}

base::Mat3 fastenerAxes(base::Vec3 pointA, base::Vec3 pointB)
{
    const base::Vec3 xAxis = base::normalized(pointB - pointA);
    std::size_t least = 0;
    for (std::size_t k = 1; k < 3; ++k)
        if (std::abs(xAxis[k]) < std::abs(xAxis[least]))
            least = k;
    base::Vec3 basic{};
    basic[least] = 1.0;
    const base::Vec3 yAxis = base::normalized(basic - (basic[least] * xAxis[least]) * xAxis);
    const base::Vec3 zAxis = base::cross(xAxis, yAxis);
    return base::matrixFromRows(xAxis, yAxis, zAxis).transposed();
}

} // namespace oos::elements
