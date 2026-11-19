#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/FormulationModel.hpp"
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
        return base::concatenate(
            {"expected the shape functions of the pierced element at the piercing point to sum to "
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

namespace
{

// The gains of the weighted least-squares rigid motion of the translations t_i of points x_i about P,
// (u_P, θ_P) = Σ G_i·t_i (PiercingInterpolation.hpp), or a problem.
std::variant<std::vector<base::MatN<UNKNOWNS, 3>>, std::string> rigidMotionGains(base::Vec3 piercingPoint,
                                                                                 std::span<const base::Vec3> positions,
                                                                                 std::span<const double> weights)
{
    // Lc, the mean distance of the points to P: the unknowns are q' = (u_P, Lc·θ_P), the rows
    // B'_i = [I, −[ρ_i]×] with ρ_i = r_i / Lc, so that every term of A' = Σ w_i·B'_iᵀ·B'_i is of the order of the
    // weights.
    double characteristicLength = 0.0;
    for (const base::Vec3& position : positions)
        characteristicLength += base::norm(position - piercingPoint);
    characteristicLength /= static_cast<double>(positions.size());
    if (!(characteristicLength > 0.0))
        return std::string(
            "expected the grid points of the pierced element around the piercing point, found them all "
            "on it");
    std::vector<base::MatN<3, UNKNOWNS>> rows(positions.size());
    base::MatN<UNKNOWNS> normal;
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const base::Mat3 arm = crossProductMatrix((positions[i] - piercingPoint) / characteristicLength);
        for (std::size_t r = 0; r < 3; ++r)
        {
            rows[i](r, r) = 1.0;
            for (std::size_t c = 0; c < 3; ++c)
                rows[i](r, 3 + c) = -arm(r, c);
        }
        normal = normal + weights[i] * (rows[i].transposed() * rows[i]);
    }
    if (smallestRelativePivot(normal) <= PIERCING_PIVOT_TOLERANCE)
        return std::string(
            "expected grid points of the pierced element that fix the rotation of the piercing point, "
            "found them collinear or coincident");

    // G_i = A'⁻¹·B'_iᵀ·w_i, the rotations unscaled after.
    std::vector<base::MatN<UNKNOWNS, 3>> gains; // one per point (at most 8 grid points, or 4 corners)
    const base::MatN<UNKNOWNS> inverse = inverted(normal);
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        base::MatN<UNKNOWNS, 3> gain = inverse * rows[i].transposed();
        for (std::size_t r = 0; r < UNKNOWNS; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                gain(r, c) *= r < 3 ? weights[i] : weights[i] / characteristicLength;
        gains.push_back(gain);
    }
    return gains;
}

// What the grid points give to the three parts of the motion of a piercing point P, each a 3 × 3 block per grid
// point on its translations or its rotations (in its displacement system):
//   mean:    ū = Σ mean_g·u_g, the translation of the point X̄ of the sheet under P (its shape functions);
//   bending: β = Σ bending_g·θ_g, the rotation of the shell there (the field of rotations of its elements);
//   fit:     θ_fit = Σ fit_g·u_g, the rotation of the rigid motion fitted to the translations.
struct TieTerms
{
    explicit TieTerms(std::size_t grids) : mean(grids), bending(grids), fit(grids) {}

    std::vector<base::Mat3> mean;
    std::vector<base::Mat3> bending;
    std::vector<base::Mat3> fit;
};

// The 3 × 3 block of rows `part` (0: translation, 1: rotation) of a gain.
base::Mat3 blockOf(const base::MatN<UNKNOWNS, 3>& gain, std::size_t part)
{
    base::Mat3 block;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            block(r, c) = gain(3 * part + r, c);
    return block;
}

// The map of the piercing point from the terms: its rotation is that of the shell about the axes of its plane and
// that of the fit about its normal n,
//     θ_P = (I − n·nᵀ)·β + n·nᵀ·θ_fit,
// and its translation that of the sheet carried rigidly by θ_P from X̄ to P: u_P = ū + θ_P × (P − X̄).
PiercingMap composeTie(const TieTerms& terms, base::Vec3 normal, base::Vec3 fromSheetToPoint)
{
    base::Mat3 aboutNormal;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            aboutNormal(r, c) = normal[r] * normal[c];
    const base::Mat3 inPlane = base::Mat3::identity() + (-1.0) * aboutNormal;
    // u_P = ū − [P − X̄]×·θ_P
    const base::Mat3 arm = (-1.0) * crossProductMatrix(fromSheetToPoint);
    PiercingMap map;
    map.gridCount = terms.mean.size();
    map.values.assign(36 * map.gridCount, 0.0);
    for (std::size_t g = 0; g < map.gridCount; ++g)
    {
        const base::Mat3 rotationOfTranslations = aboutNormal * terms.fit[g];
        const base::Mat3 rotationOfRotations = inPlane * terms.bending[g];
        addBlock(map, 1, 6 * g, rotationOfTranslations, 1.0);
        addBlock(map, 1, 6 * g + 3, rotationOfRotations, 1.0);
        addBlock(map, 0, 6 * g, terms.mean[g] + arm * rotationOfTranslations, 1.0);
        addBlock(map, 0, 6 * g + 3, arm * rotationOfRotations, 1.0);
    }
    return map;
}

} // namespace

std::variant<PiercingMap, std::string> interpolatePiercing(base::Vec3 piercingPoint, std::span<const PiercedGrid> grids)
{
    if (grids.empty())
        return std::string("expected the grid points of the pierced element, found none");
    std::variant<std::vector<double>, std::string> weighting = weightsOf(grids);
    if (const auto* problem = std::get_if<std::string>(&weighting))
        return *problem;
    const std::vector<double>& weights = std::get<std::vector<double>>(weighting);
    std::vector<base::Vec3> positions; // one per grid point of the patch
    for (const PiercedGrid& grid : grids)
        positions.push_back(grid.position);
    auto gains = rigidMotionGains(piercingPoint, positions, weights);
    if (const auto* problem = std::get_if<std::string>(&gains))
        return *problem;
    // The normal of the sheet: that of the first three grid points, the corners of the pierced element.
    const base::Vec3 across = grids.size() >= 3
                                  ? base::cross(positions[1] - positions[0], positions[2] - positions[0])
                                  : base::Vec3{};
    if (!(base::norm(across) > 0.0))
        return std::string(
            "expected grid points of the pierced element that fix the rotation of the piercing point, "
            "found them collinear or coincident");

    // The point of the sheet under P and its motion: the shape functions of the pierced element over the
    // translations and the rotations of its grid points; the rotation about the normal from the rigid fit.
    TieTerms terms(grids.size());
    base::Vec3 onSheet;
    for (std::size_t i = 0; i < grids.size(); ++i)
    {
        onSheet = onSheet + grids[i].shapeValue * grids[i].position;
        terms.mean[i] = grids[i].shapeValue * grids[i].frame;
        terms.bending[i] = grids[i].shapeValue * grids[i].frame;
        terms.fit[i] = blockOf(std::get<0>(gains)[i], 1) * grids[i].frame;
    }
    return composeTie(terms, base::normalized(across), piercingPoint - onSheet);
}

std::variant<PiercingMap, std::string> interpolateFootprint(base::Vec3 piercingPoint, std::span<const SideGrid> grids,
                                                            std::span<const AuxiliaryPoint> points)
{
    if (points.empty())
        return std::string("expected the auxiliary points of the fastener on the shells, found none");
    // The auxiliary points where the shape functions put them, X_k = Σ N_kj·x_j: a rigid motion of the grid points
    // is then the same rigid motion of the auxiliary points.
    std::vector<base::Vec3> positions; // the corners of one footprint
    std::vector<double> weights;
    for (const AuxiliaryPoint& point : points)
    {
        base::Vec3 position;
        for (std::size_t j = 0; j < point.gridIndices.size(); ++j)
            position = position + point.shapeValues[j] * grids[point.gridIndices[j]].position;
        positions.push_back(position);
        weights.push_back(point.weight);
    }
    auto gains = rigidMotionGains(piercingPoint, positions, weights);
    if (const auto* problem = std::get_if<std::string>(&gains))
        return *problem;
    // The normal of the square: across its diagonals (across two sides of three points).
    base::Vec3 across;
    if (points.size() >= 4)
        across = base::cross(positions[2] - positions[0], positions[3] - positions[1]);
    else if (points.size() == 3)
        across = base::cross(positions[1] - positions[0], positions[2] - positions[0]);
    if (!(base::norm(across) > 0.0))
        return std::string(
            "expected grid points of the pierced element that fix the rotation of the piercing point, "
            "found them collinear or coincident");

    // Auxiliary point k gives its share w_k of the mean translation and of the mean rotation of the shell, through
    // the shape functions N_kj of its element, and the gain of its translation in the rigid fit.
    TieTerms terms(grids.size());
    base::Vec3 onSheet;
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        onSheet = onSheet + points[k].weight * positions[k];
        for (std::size_t j = 0; j < points[k].gridIndices.size(); ++j)
        {
            const std::size_t grid = points[k].gridIndices[j];
            const base::Mat3 share = (points[k].weight * points[k].shapeValues[j]) * grids[grid].frame;
            terms.mean[grid] = terms.mean[grid] + share;
            terms.bending[grid] = terms.bending[grid] + share;
            terms.fit[grid] = terms.fit[grid] +
                              points[k].shapeValues[j] * (blockOf(std::get<0>(gains)[k], 1) * grids[grid].frame);
        }
    }
    return composeTie(terms, base::normalized(across), piercingPoint - onSheet);
}

namespace
{

std::string missingGrid(std::int32_t id)
{
    return base::concatenate({"expected grid point ", std::to_string(id),
                              " of the pierced element in the model, found none; the piercing table does not "
                              "come from the same model: take the snapshot again"});
}

// The side tied over the cross-section of its fastener: the auxiliary points of `corners` (row `row`).
std::variant<SideInterpolation, std::string> interpolateCorners(const input::FootprintSamples& corners,
                                                                std::int64_t row, SideInterpolation side,
                                                                const SideGridLookup& findGrid)
{
    std::vector<std::int32_t> ids; // the grid points under the corners (those of four elements at most)
    std::vector<SideGrid> grids;
    std::vector<AuxiliaryPoint> points;
    for (std::int64_t k = 0; k < corners.count(row); ++k)
    {
        const input::FootprintSample sample = corners.at(row, k);
        AuxiliaryPoint point;
        point.weight = sample.weight;
        point.shapeValues.assign(sample.shapeValues.begin(), sample.shapeValues.end());
        for (const std::int32_t id : sample.grids)
        {
            auto known = std::find(ids.begin(), ids.end(), id);
            if (known == ids.end())
            {
                const std::optional<SideGrid> grid = findGrid(id);
                if (!grid)
                    return missingGrid(id);
                grids.push_back(*grid);
                side.gridRows.push_back(grid->row);
                ids.push_back(id);
                known = ids.end() - 1;
            }
            point.gridIndices.push_back(static_cast<std::size_t>(known - ids.begin()));
        }
        points.push_back(std::move(point));
    }
    std::variant<PiercingMap, std::string> map = interpolateFootprint(side.point, grids, points);
    if (auto* problem = std::get_if<std::string>(&map))
        return std::move(*problem);
    side.map = std::move(std::get<PiercingMap>(map));
    return side;
}

} // namespace

std::variant<SideInterpolation, std::string> interpolateSide(const input::FastenerPiercingTable& table,
                                                             input::PiercingRows rows, const SideGridLookup& findGrid)
{
    if (rows.isEmpty())
        return std::string("expected the row of its piercing point in the fastener piercing table, found none");
    const std::int64_t row = rows.first;
    SideInterpolation side;
    const std::span<const double> xyz = table.piercingPoints()[row];
    side.point = {xyz[0], xyz[1], xyz[2]};
    if (table.cornerFootprint().count(row) > 0)
        return interpolateCorners(table.cornerFootprint(), row, std::move(side), findGrid);
    // The grid points of the pierced element, or of the shell elements around a point on a grid point or a side of
    // it (FastenerPiercingTable::patchGrids): the rotation of such a point is then that of the sheet on every side
    // of it, not the slope of one element.
    const std::span<const std::int32_t> ids = table.patchGrids()[row];
    const std::span<const double> values = table.patchWeights()[row];
    std::vector<PiercedGrid> grids; // the grid points of the elements around one point
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        const std::optional<SideGrid> grid = findGrid(ids[k]);
        if (!grid)
            return missingGrid(ids[k]);
        side.gridRows.push_back(grid->row);
        grids.push_back({grid->position, grid->frame, values[k]});
    }
    std::variant<PiercingMap, std::string> map = interpolatePiercing(side.point, grids);
    if (auto* problem = std::get_if<std::string>(&map))
        return std::move(*problem);
    side.map = std::move(std::get<PiercingMap>(map));
    return side;
}

SideMotion sideMotionOf(const SideInterpolation& side)
{
    SideMotion motion;
    motion.point = side.point;
    for (const std::int32_t gridRow : side.gridRows)
        for (int c = 1; c <= 6; ++c)
            motion.dofs.push_back(ElementDof::grid(gridRow, c));
    const std::size_t n = motion.dofs.size();
    motion.map.assign(6 * n, 0.0);
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < n; ++j)
            motion.map[i * n + j] = side.map.at(i, j);
    return motion;
}

std::variant<SideMotion, std::string> resolveSideMotion(const input::FastenerPiercingTable& table,
                                                        input::PiercingRows rows, const SideGridLookup& findGrid,
                                                        const FormulationModel* formulation)
{
    if (formulation != nullptr && !rows.isEmpty() && table.elementIds()[static_cast<std::size_t>(rows.first)] != 0)
    {
        const std::int64_t row = rows.first;
        const std::span<const double> xyz = table.piercingPoints()[row];
        const base::Vec3 point{xyz[0], xyz[1], xyz[2]};
        // The samples of the shells: the footprint of the fastener, or without one the piercing point alone.
        const input::FootprintSamples& footprint = table.meanFootprint();
        std::vector<input::FootprintSample> read; // the Gauss points of one footprint
        for (std::int64_t k = 0; k < footprint.count(row); ++k)
            read.push_back(footprint.at(row, k));
        if (read.empty())
            read.push_back({table.elementIds()[static_cast<std::size_t>(row)], 1.0, table.grids()[row],
                            table.weights()[row]});
        std::vector<std::int32_t> gridRows; // the grid points of the samples, sample after sample
        for (const input::FootprintSample& sample : read)
            for (const std::int32_t id : sample.grids)
            {
                const std::optional<SideGrid> grid = findGrid(id);
                if (!grid)
                    return missingGrid(id);
                gridRows.push_back(grid->row);
            }
        std::vector<FormulationModel::ShellSample> samples;
        std::size_t first = 0;
        for (const input::FootprintSample& sample : read)
        {
            samples.push_back({sample.elementId, sample.weight,
                               std::span<const std::int32_t>(gridRows).subspan(first, sample.grids.size()),
                               sample.shapeValues});
            first += sample.grids.size();
        }
        const std::optional<FormulationModel::PointMotion> motion =
            formulation->shellPointMotion(table.elementIds()[static_cast<std::size_t>(row)], point, samples);
        if (motion)
        {
            SideMotion side;
            side.point = point;
            // the DOFs in the order the rows first name them
            for (const std::vector<FormulationRelationTerm>& terms : motion->rows)
                for (const FormulationRelationTerm& term : terms)
                    if (std::find(side.dofs.begin(), side.dofs.end(), term.dof) == side.dofs.end())
                        side.dofs.push_back(term.dof);
            const std::size_t n = side.dofs.size();
            side.map.assign(6 * n, 0.0);
            for (std::size_t i = 0; i < 6; ++i)
                for (const FormulationRelationTerm& term : motion->rows[i])
                {
                    const auto at = std::find(side.dofs.begin(), side.dofs.end(), term.dof) - side.dofs.begin();
                    side.map[i * n + static_cast<std::size_t>(at)] += term.coefficient;
                }
            return side;
        }
    }
    std::variant<SideInterpolation, std::string> interpolation = interpolateSide(table, rows, findGrid);
    if (auto* problem = std::get_if<std::string>(&interpolation))
        return std::move(*problem);
    return sideMotionOf(std::get<SideInterpolation>(interpolation));
}

} // namespace oos::elements
