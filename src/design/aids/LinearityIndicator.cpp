#include "oos/design/aids/LinearityIndicator.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/ElementVolumes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace oos::design
{

namespace
{

// A value for a message: four significant digits.
std::string formatValue(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.4g", value);
    return text.data();
}

double signOf(double value) noexcept
{
    // The right derivative at 0, as ABS of DesignEquation (design/expressions).
    return value < 0.0 ? -1.0 : 1.0;
}

// The volume of each element of `geometry` in `snapshot` (assembly::ElementVolumes of its shell batches), NaN for an
// element whose card gives none.
std::vector<double> volumesOf(const LinearityGeometry& geometry, const input::ModelSnapshot& snapshot,
                              base::MemoryBudget& budget, base::ThreadPool* pool)
{
    std::vector<double> volumes(geometry.elements().size(), std::numeric_limits<double>::quiet_NaN());
    if (geometry.elements().empty())
        return volumes;
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const elements::SnapshotIndex index(snapshot, budget);
    const assembly::ElementSet set(index, registry);
    // The batch of the set of each snapshot batch, and the shell batches wanted.
    const std::span<const input::ElementBatch> batches = snapshot.elementBatches();
    std::vector<std::size_t> setBatchOf(batches.size(), set.batches().size()); // bounded by the card types
    std::vector<std::uint8_t> isWanted(set.batches().size(), 0);               // bounded by the card types
    for (std::size_t b = 0; b < set.batches().size(); ++b)
        for (std::size_t k = 0; k < batches.size(); ++k)
            if (&batches[k] == &set.batches()[b].view.batch())
            {
                setBatchOf[k] = b;
                isWanted[b] = LinearityGeometry::cornerCountOf(batches[k].card()) > 0 ? 1 : 0;
            }
    const assembly::ElementVolumes elementVolumes(set, isWanted, budget, pool);
    for (std::size_t e = 0; e < geometry.elements().size(); ++e)
    {
        const LinearityGeometry::Element& element = geometry.elements()[e];
        const std::size_t b = setBatchOf[element.batch];
        if (b == set.batches().size())
            continue;
        const std::span<const double> ofBatch = elementVolumes.volumesOf(b);
        if (static_cast<std::size_t>(element.row) < ofBatch.size())
            volumes[e] = ofBatch[static_cast<std::size_t>(element.row)];
    }
    return volumes;
}

} // namespace

bool LinearityIndicator::isShellCard(std::string_view card) noexcept
{
    return LinearityGeometry::cornerCountOf(card) != 0;
}

std::vector<std::int32_t> LinearityIndicator::shellGridIds(const input::ModelSnapshot& snapshot)
{
    const input::NodeTable& nodes = snapshot.nodes();
    const std::span<const std::int32_t> ids = nodes.ids();
    std::set<std::int32_t> grids;
    for (const input::ElementBatch& batch : snapshot.elementBatches())
    {
        const std::size_t corners = LinearityGeometry::cornerCountOf(batch.card());
        if (corners == 0)
            continue;
        for (std::int64_t row = 0; row < batch.rowCount(); ++row)
            for (const std::int32_t node : LinearityGeometry::cornerRowsOf(batch, row, corners, nodes.rowCount()))
                grids.insert(ids[static_cast<std::size_t>(node)]);
    }
    return {grids.begin(), grids.end()};
}

LinearityIndicator::LinearityIndicator(const input::ModelSnapshot& snapshot,
                                       const std::shared_ptr<const void>& keepAlive, base::MemoryBudget& budget,
                                       base::ThreadPool* pool)
    : LinearityIndicator(std::make_shared<const LinearityGeometry>(snapshot, budget), snapshot, keepAlive, budget, pool)
{
}

LinearityIndicator::LinearityIndicator(std::shared_ptr<const LinearityGeometry> geometry,
                                       const input::ModelSnapshot& snapshot,
                                       const std::shared_ptr<const void>& keepAlive, base::MemoryBudget& budget,
                                       base::ThreadPool* pool)
    : m_reservation(budget.reserve(LinearityGeometry::estimateBytes(snapshot), "linearity indicator thicknesses")),
      m_geometry(std::move(geometry))
{
    // The volumes are those of the snapshot's own elements (their unit-density overlay lives in ElementVolumes alone).
    static_cast<void>(keepAlive);
    const std::vector<double> volumes = volumesOf(*m_geometry, snapshot, budget, pool);
    // The elements of positive volume, in the order of the geometry (the order of the batches and rows).
    constexpr std::size_t LEFT_OUT = static_cast<std::size_t>(-1);
    std::vector<std::size_t> kept(volumes.size(), LEFT_OUT); // the index of each geometry element in m_elements
    for (std::size_t e = 0; e < volumes.size(); ++e)
    {
        if (!(volumes[e] > 0.0))
            continue;
        const LinearityGeometry::Element& from = m_geometry->elements()[e];
        ShellElementGeometry element;
        element.card = from.card;
        element.elementId = from.elementId;
        element.cornerRows = from.cornerRows;
        element.normal = from.normal;
        element.area = from.area;
        element.thickness = volumes[e] / from.area;
        kept[e] = m_elements.size();
        m_elements.push_back(std::move(element));
    }
    // The grid points of a kept element, by increasing ID: the mean normal of their elements, each oriented as the
    // first one (the first one alone if they cancel: a fold), and their mean thickness.
    for (const LinearityGeometry::Point& from : m_geometry->points())
    {
        std::vector<std::size_t> around; // bounded by the elements of one grid point
        for (const std::size_t e : from.elements)
            if (kept[e] != LEFT_OUT)
                around.push_back(kept[e]);
        if (around.empty())
            continue;
        ShellPointGeometry point;
        point.grid = from.grid;
        point.row = from.row;
        const base::Vec3 first = m_elements[around.front()].normal;
        base::Vec3 sum;
        double thickness = 0.0;
        for (const std::size_t e : around)
        {
            const base::Vec3 normal = m_elements[e].normal;
            sum = sum + (base::dot(normal, first) < 0.0 ? -normal : normal);
            thickness += m_elements[e].thickness;
        }
        const double length = base::norm(sum);
        point.normal = length > 0.0 ? sum / length : first;
        point.thickness = thickness / static_cast<double>(around.size());
        point.displacementFrame = from.displacementFrame;
        point.elements = std::move(around);
        m_pointOfGrid.emplace(point.grid, m_points.size());
        m_points.push_back(std::move(point));
    }
}

const ShellPointGeometry* LinearityIndicator::findShellPoint(std::int32_t grid) const
{
    const auto found = m_pointOfGrid.find(grid);
    return found == m_pointOfGrid.end() ? nullptr : &m_points[found->second];
}

double LinearityIndicator::deflectionRatio(const ShellPointGeometry& point, std::array<double, 3> translationInCd)
{
    const base::Vec3 inBasic =
        point.displacementFrame * base::Vec3{translationInCd[0], translationInCd[1], translationInCd[2]};
    return std::abs(base::dot(point.normal, inBasic)) / point.thickness;
}

std::array<double, 3> LinearityIndicator::deflectionWeights(const ShellPointGeometry& point,
                                                            std::array<double, 3> translationInCd)
{
    const base::Mat3& frame = point.displacementFrame;
    const base::Vec3 inBasic = frame * base::Vec3{translationInCd[0], translationInCd[1], translationInCd[2]};
    const double scale = signOf(base::dot(point.normal, inBasic)) / point.thickness;
    // n·R·t = (Rᵀ·n)·t: the weights are the normal written in CD.
    std::array<double, 3> weights{};
    for (std::size_t c = 0; c < 3; ++c)
        weights[c] =
            scale * (frame(0, c) * point.normal[0] + frame(1, c) * point.normal[1] + frame(2, c) * point.normal[2]);
    return weights;
}

double LinearityIndicator::rotationMagnitude(std::array<double, 3> rotation) noexcept
{
    return std::sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2]);
}

std::array<double, 3> LinearityIndicator::rotationWeights(std::array<double, 3> rotation) noexcept
{
    const double magnitude = rotationMagnitude(rotation);
    if (magnitude == 0.0)
        return {0.0, 0.0, 0.0};
    return {rotation[0] / magnitude, rotation[1] / magnitude, rotation[2] / magnitude};
}

LinearityReport LinearityIndicator::report(const output::ResultArrays& arrays,
                                           std::span<const std::int32_t> subcases) const
{
    LinearityReport report;
    for (const std::int32_t subcase : subcases)
    {
        const auto table = std::find_if(arrays.nodalTables.begin(), arrays.nodalTables.end(),
                                        [subcase](const output::NodalResultTable& candidate)
                                        { return candidate.kind == "displacements" && candidate.subcase == subcase; });
        if (table == arrays.nodalTables.end())
            continue;
        LinearitySubcase result;
        result.subcase = subcase;
        for (std::size_t row = 0; row < table->rowCount(); ++row)
        {
            if (table->pointTypes[row] != static_cast<std::int8_t>(output::PointType::Grid))
                continue;
            const double* values = table->values.data() + row * output::NodalResultTable::COMPONENT_COUNT;
            const std::int32_t grid = table->ids[row];
            const double rotation = rotationMagnitude({values[3], values[4], values[5]});
            result.grids.push_back(grid);
            result.rotations.push_back(rotation);
            if (rotation > result.largestRotation || result.rotationGrid == 0)
            {
                result.largestRotation = rotation;
                result.rotationGrid = grid;
            }
            if (const ShellPointGeometry* point = findShellPoint(grid))
            {
                const double ratio = deflectionRatio(*point, {values[0], values[1], values[2]});
                result.shellGrids.push_back(grid);
                result.deflectionRatios.push_back(ratio);
                if (ratio > result.largestDeflectionRatio || result.deflectionGrid == 0)
                {
                    result.largestDeflectionRatio = ratio;
                    result.deflectionGrid = grid;
                }
            }
        }
        if (result.deflectionGrid != 0 && result.largestDeflectionRatio > report.deflectionLimit)
            report.warnings.push_back(base::concatenate(
                {"subcase ", std::to_string(subcase), ": the normal displacement of shell grid point ",
                 std::to_string(result.deflectionGrid), " is ", formatValue(result.largestDeflectionRatio),
                 " times the thickness around it, above LINEARITY_DEFLECTION_RATIO = ",
                 formatValue(report.deflectionLimit),
                 ": the linear answer is more than 5 % off the geometrically non-linear one (membrane stiffening); "
                 "thicken the shell, lower the load, or constrain deflection_to_thickness"}));
        if (result.rotationGrid != 0 && result.largestRotation > report.rotationLimit)
            report.warnings.push_back(base::concatenate(
                {"subcase ", std::to_string(subcase), ": grid point ", std::to_string(result.rotationGrid),
                 " rotates by ", formatValue(result.largestRotation),
                 " rad, above LINEARITY_ROTATION = ", formatValue(report.rotationLimit),
                 " rad: the linear answer is more than 5 % off the geometrically non-linear one (large rotation); "
                 "stiffen the structure, lower the load, or constrain the rotation"}));
        report.subcases.push_back(std::move(result));
    }
    return report;
}

} // namespace oos::design
