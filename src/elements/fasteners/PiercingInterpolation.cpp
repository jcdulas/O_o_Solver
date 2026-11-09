#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

constexpr std::size_t UNKNOWNS = 6; // the translation and the rotation of the piercing point

// Adds factor·block to the 3 × 3 block of M at (blockRow, column).
void addBlock(PiercingMap& map, std::size_t blockRow, std::size_t column, const base::Mat3& block, double factor)
{
    const std::size_t width = 6 * map.gridCount;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            map.values[(3 * blockRow + i) * width + column + j] += factor * block(i, j);
}

// The smallest pivot of the Gaussian elimination of a symmetric positive semi-definite matrix (no row exchange: the
// pivots of such a matrix are its LDLᵀ factors), relative to its largest diagonal term; 0 for a zero matrix.
double smallestRelativePivot(base::MatN<UNKNOWNS> a)
{
    double largest = 0.0;
    for (std::size_t k = 0; k < UNKNOWNS; ++k)
        largest = std::max(largest, a(k, k));
    if (!(largest > 0.0))
        return 0.0;
    double smallest = largest;
    for (std::size_t k = 0; k < UNKNOWNS; ++k)
    {
        smallest = std::min(smallest, a(k, k));
        if (!(a(k, k) > 0.0))
            return 0.0;
        for (std::size_t i = k + 1; i < UNKNOWNS; ++i)
        {
            const double factor = a(i, k) / a(k, k);
            for (std::size_t j = k; j < UNKNOWNS; ++j)
                a(i, j) -= factor * a(k, j);
        }
    }
    return smallest / largest;
}

// The weights w_i = max(N_i, f·max N) of the interpolation (PIERCING_WEIGHT_FLOOR), or a problem.
std::variant<std::vector<double>, std::string> weightsOf(std::span<const PiercedGrid> grids)
{
    double sum = 0.0;
    double largest = 0.0;
    for (const PiercedGrid& grid : grids)
    {
        sum += grid.shapeValue;
        largest = std::max(largest, grid.shapeValue);
    }
    if (!(std::abs(sum - 1.0) <= PIERCING_WEIGHT_SUM_TOLERANCE))
        return base::concatenate({"expected the shape functions of the pierced element at the piercing point to sum to "
                                  "1, found ",
                                  std::to_string(sum), "; the piercing table does not hold an interpolation"});
    std::vector<double> weights; // one per grid point of the element (at most 8)
    for (const PiercedGrid& grid : grids)
        weights.push_back(std::max(grid.shapeValue, PIERCING_WEIGHT_FLOOR * largest));
    return weights;
}

} // namespace

PiercingMap directGridMap(const base::Mat3& gridFrame)
{
    PiercingMap map;
    map.gridCount = 1;
    map.values.assign(36, 0.0);
    addBlock(map, 0, 0, gridFrame, 1.0);
    addBlock(map, 1, 3, gridFrame, 1.0);
    return map;
}

std::variant<PiercingMap, std::string> interpolatePiercing(base::Vec3 piercingPoint, std::span<const PiercedGrid> grids)
{
    if (grids.empty())
        return std::string("expected the grid points of the pierced element, found none");
    std::variant<std::vector<double>, std::string> weighting = weightsOf(grids);
    if (const auto* problem = std::get_if<std::string>(&weighting))
        return *problem;
    const std::vector<double>& weights = std::get<std::vector<double>>(weighting);

    // Lc, the mean distance of the grid points to P: the unknowns are q' = (u_P, Lc·θ_P), the rows
    // B'_i = [I, −[ρ_i]×] with ρ_i = r_i / Lc, so that every term of A' = Σ w_i·B'_iᵀ·B'_i is of the order of the
    // weights.
    double characteristicLength = 0.0;
    for (const PiercedGrid& grid : grids)
        characteristicLength += base::norm(grid.position - piercingPoint);
    characteristicLength /= static_cast<double>(grids.size());
    if (!(characteristicLength > 0.0))
        return std::string("expected the grid points of the pierced element around the piercing point, found them all "
                           "on it");
    std::vector<base::MatN<3, UNKNOWNS>> rows(grids.size());
    base::MatN<UNKNOWNS> normal;
    for (std::size_t i = 0; i < grids.size(); ++i)
    {
        const base::Mat3 arm = crossProductMatrix((grids[i].position - piercingPoint) / characteristicLength);
        for (std::size_t r = 0; r < 3; ++r)
        {
            rows[i](r, r) = 1.0;
            for (std::size_t c = 0; c < 3; ++c)
                rows[i](r, 3 + c) = -arm(r, c);
        }
        normal = normal + weights[i] * (rows[i].transposed() * rows[i]);
    }
    if (smallestRelativePivot(normal) <= PIERCING_PIVOT_TOLERANCE)
        return std::string("expected grid points of the pierced element that fix the rotation of the piercing point, "
                           "found them collinear or coincident");

    // Grid i contributes G_i = A'⁻¹·B'_iᵀ·w_i on its translations in basic, R_i·u_i; the rotations unscaled after.
    PiercingMap map;
    map.gridCount = grids.size();
    map.values.assign(36 * map.gridCount, 0.0);
    const base::MatN<UNKNOWNS> inverse = inverted(normal);
    for (std::size_t i = 0; i < grids.size(); ++i)
    {
        const base::MatN<UNKNOWNS, 3> gain = inverse * rows[i].transposed();
        for (std::size_t part = 0; part < 2; ++part)
        {
            const double unscale = part == 0 ? 1.0 : 1.0 / characteristicLength;
            base::Mat3 block;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    block(r, c) = gain(3 * part + r, c);
            addBlock(map, part, 6 * i, block * grids[i].frame, weights[i] * unscale);
        }
    }
    return map;
}

std::variant<SideInterpolation, std::string> interpolateSide(const input::FastenerPiercingTable& table,
                                                             input::PiercingRows rows, const SideGridLookup& findGrid)
{
    if (rows.isEmpty())
        return std::string("expected the row of its piercing point in the fastener piercing table, found none");
    const std::int64_t row = rows.first;
    SideInterpolation side;
    const std::span<const double> xyz = table.piercingPoints()[row];
    side.point = {xyz[0], xyz[1], xyz[2]};
    const std::span<const std::int32_t> ids = table.grids()[row];
    const std::span<const double> values = table.weights()[row];
    std::vector<PiercedGrid> grids; // the grid points of one element (at most 8)
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        const std::optional<SideGrid> grid = findGrid(ids[k]);
        if (!grid)
            return base::concatenate({"expected grid point ", std::to_string(ids[k]),
                                      " of the pierced element in the model, found none; the piercing table does not "
                                      "come from the same model: take the snapshot again"});
        side.gridRows.push_back(grid->row);
        grids.push_back({grid->position, grid->frame, values[k]});
    }
    std::variant<PiercingMap, std::string> map = interpolatePiercing(side.point, grids);
    if (auto* problem = std::get_if<std::string>(&map))
        return std::move(*problem);
    side.map = std::move(std::get<PiercingMap>(map));
    return side;
}

} // namespace oos::elements
