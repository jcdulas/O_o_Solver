#include "oos/reduction/absorption/FastenerPiercing.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace oos::reduction
{

namespace
{

constexpr std::size_t DOFS_PER_GRID = 6;
constexpr std::size_t TRANSLATION_COUNT = 3;
// A pierced element whose grid points all sit on P has no length to scale rotations by (as the RigidInterpolation's Lc
// rule).
constexpr double UNIT_LENGTH = 1.0;

std::string sideName(FastenerSide side)
{
    return side == FastenerSide::A ? "A" : "B";
}

} // namespace

std::variant<SpiderRelation, std::string> FastenerPiercing::relation(const input::FastenerPiercingTable& piercings,
                                                                     const assembly::DofMap& dofMap,
                                                                     base::EntityId elementId, FastenerSide side)
{
    const std::optional<input::FastenerPiercingRows> rows = piercings.findFastener(elementId);
    if (!rows || side == FastenerSide::None)
        return base::concatenate({"fastener ", std::to_string(elementId), " has no piercing rows"});
    const input::PiercingRows& sideRows = side == FastenerSide::A ? rows->sideA : rows->sideB;
    if (sideRows.isEmpty())
        return base::concatenate(
            {"fastener ", std::to_string(elementId), " has no piercing point on side ", sideName(side)});

    // The interpolation of the piercing point, as assembly::FastenerConstraints and the elements build it.
    const elements::SideGridLookup findGrid = [&dofMap](std::int32_t gridId) -> std::optional<elements::SideGrid>
    {
        const std::optional<std::int32_t> gridRow = dofMap.points().findGrid(gridId);
        if (!gridRow)
            return std::nullopt;
        return elements::SideGrid{*gridRow, dofMap.positionInBasic(*gridRow), dofMap.cdToBasic(*gridRow)};
    };
    std::variant<elements::SideInterpolation, std::string> resolved =
        elements::interpolateSide(piercings, sideRows, findGrid);
    if (const auto* problem = std::get_if<std::string>(&resolved))
        return base::concatenate({"fastener ", std::to_string(elementId), " side ", sideName(side), ": ", *problem});
    const elements::SideInterpolation& resolvedSide = std::get<elements::SideInterpolation>(resolved);
    const std::vector<std::int32_t>& gridRows = resolvedSide.gridRows;
    const elements::PiercingMap& interpolation = resolvedSide.map;
    const base::Vec3 origin = resolvedSide.point;

    // The columns M uses, by increasing g-set DOF.
    std::vector<std::pair<base::DofIndex, std::size_t>> columns; // (DOF, column of M)
    for (std::size_t g = 0; g < gridRows.size(); ++g)
        for (std::size_t c = 0; c < DOFS_PER_GRID; ++c)
        {
            const std::size_t column = g * DOFS_PER_GRID + c;
            bool isUsed = false;
            for (std::size_t row = 0; row < DOFS_PER_GRID && !isUsed; ++row)
                isUsed = interpolation.at(row, column) != 0.0;
            if (isUsed)
                columns.emplace_back(dofMap.gridDof(gridRows[g], static_cast<int>(c) + 1), column);
        }
    std::sort(columns.begin(), columns.end());

    SpiderRelation relation;
    SpiderWeightMatrix& w = relation.weights;
    w.referenceDofCount = DOFS_PER_GRID;
    w.legDofCount = columns.size();
    w.values.assign(w.referenceDofCount * w.legDofCount, 0.0);
    for (std::size_t row = 0; row < DOFS_PER_GRID; ++row)
    {
        w.referenceKinds.push_back(row < TRANSLATION_COUNT ? SpiderDofKind::Translation : SpiderDofKind::Rotation);
        for (std::size_t k = 0; k < columns.size(); ++k)
            w.values[row * w.legDofCount + k] = interpolation.at(row, columns[k].second);
    }
    for (const auto& [dof, column] : columns)
    {
        relation.legDofs.push_back(dof);
        w.legKinds.push_back(dofKindOf(dofMap, dof));
    }
    double total = 0.0;
    for (const std::int32_t gridRow : gridRows)
    {
        const base::Vec3 arm = dofMap.positionInBasic(gridRow) - origin;
        total += std::sqrt(arm[0] * arm[0] + arm[1] * arm[1] + arm[2] * arm[2]);
    }
    w.characteristicLength = total > 0.0 ? total / static_cast<double>(gridRows.size()) : UNIT_LENGTH;
    return relation;
}

} // namespace oos::reduction
