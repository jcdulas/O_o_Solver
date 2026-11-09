#include "oos/aero/splines/SplineMatrix.hpp"

#include "oos/aero/splines/BeamSpline.hpp"
#include "oos/aero/splines/SurfaceSpline.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string_view>

namespace oos::aero
{

namespace
{

const base::Vec3 FLOW{1.0, 0.0, 0.0};
// AeroBeamSpline2 and AeroBeamSpline5 defaults: DTOR = 1, DTHX = DTHY = −1 (rotations not attached).
constexpr double DEFAULT_TORSION_RATIO = 1.0;
constexpr double NOT_ATTACHED = -1.0;

// The kind of interpolation of a spline card and its method (SPLINE1, AeroBeamSpline2, AeroSurfaceSpline4,
// AeroBeamSpline5).
enum class SplineKind : std::uint8_t
{
    Surface,
    Beam
};

SplineKind kindOf(std::string_view card, std::string_view method, std::int32_t elementId)
{
    const bool isSurfaceCard = card == "AeroSurfaceSpline1" || card == "AeroSurfaceSpline4";
    const bool isBeamCard = card == "AeroBeamSpline2" || card == "AeroBeamSpline5";
    if (isSurfaceCard && (method.empty() || method == "InfinitePlateSpline" || method == "ThinPlateSpline"))
        return SplineKind::Surface;
    if (isBeamCard && (method.empty() || method == "Beam"))
        return SplineKind::Beam;
    const std::string what =
        method.empty() ? std::string(card) : base::concatenate({card, " AeroSurfaceSpline1.method ", method});
    throw base::UnsupportedCardError(base::concatenate(
        {what, " ", std::to_string(elementId),
         ": expected the infinite plate spline of AeroSurfaceSpline1 or AeroSurfaceSpline4 "
         "(AeroSurfaceSpline1.method InfinitePlateSpline or ThinPlateSpline) or the "
         "beam spline of AeroBeamSpline2 "
         "or AeroBeamSpline5, found this one (the finite plate spline FiniteSurfaceSpline, the radial interpolation "
         "RigidSurfaceSpline "
         "and "
         "the AeroSplineEquation3, AeroFiniteSurfaceSpline6, AeroFiniteBeamSpline7 and AeroUserSpline are not "
         "supported); use AeroSurfaceSpline1.method = InfinitePlateSpline or a beam spline"}));
}

// The row of `id` in an int32 column, or nothing.
std::optional<std::int64_t> rowOf(std::span<const std::int32_t> ids, std::int32_t id)
{
    for (std::size_t row = 0; row < ids.size(); ++row)
        if (ids[row] == id)
            return static_cast<std::int64_t>(row);
    return std::nullopt;
}

double realOr(const base::ContractReader& table, const char* column, std::int64_t row, double fallback)
{
    if (!table.validity(column).isValid(row))
        return fallback;
    const double value = table.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
    return std::isfinite(value) ? value : fallback;
}

// The box IDs of an AeroBoxList (its THRU runs expanded).
std::vector<std::int32_t> aelistItems(const input::ModelSnapshot& snapshot, std::int32_t setId, std::int32_t spline)
{
    const input::CardTable* table = snapshot.findAeroCardTable("AeroBoxList");
    const std::optional<std::int64_t> row = table != nullptr ? rowOf(table->ids(), setId) : std::nullopt;
    if (!row)
        throw base::SolverError(base::concatenate({"spline ", std::to_string(spline), ": expected its AeroBoxList ",
                                                   std::to_string(setId), ", found none; add the AeroBoxList card"}));
    const base::NestedListView<std::int32_t> runs = table->reader().nestedLists<std::int32_t>("runs");
    std::vector<std::int32_t> items;
    for (std::int64_t k = 0; k < runs.itemCount(*row); ++k)
    {
        const std::span<const std::int32_t> run = runs.item(*row, k);
        for (std::int32_t id = run[0]; id <= run[1]; ++id)
            items.push_back(id);
    }
    return items;
}

} // namespace

SplineMatrix::SplineMatrix(const input::ModelSnapshot& snapshot, const AeroReference& reference, const AeroBoxes& boxes,
                           base::MemoryBudget& budget)
    : m_boxes(&boxes), m_reference(reference)
{
    if (!snapshot.hasTable(input::TableRole::Splines))
        return;
    const base::ContractReader& table = snapshot.aeroTable(input::TableRole::Splines);
    const base::ContractReader& nodeTable = snapshot.aeroTable(input::TableRole::SplineNodes);
    const std::span<const std::int32_t> elementIds = table.values<std::int32_t>("elementId");
    const base::DictionaryView types = table.dictionaryTexts("type");
    const std::span<const std::int32_t> caeros = table.values<std::int32_t>("aeroPanelId", base::NullPolicy::Allow);
    const std::span<const std::int32_t> firsts = table.values<std::int32_t>("firstBox", base::NullPolicy::Allow);
    const std::span<const std::int32_t> lasts = table.values<std::int32_t>("lastBox", base::NullPolicy::Allow);
    const std::span<const std::int32_t> aelists = table.values<std::int32_t>("aeroBoxListId", base::NullPolicy::Allow);
    const base::ValidityView hasAelist = table.validity("aeroBoxListId");
    const base::DictionaryView methods = table.dictionaryTexts("method", base::NullPolicy::Allow);
    const base::ValidityView hasMethod = table.validity("method");
    const base::DictionaryView usages = table.dictionaryTexts("usage", base::NullPolicy::Allow);
    const base::ValidityView hasUsage = table.validity("usage");
    const std::span<const std::int32_t> nodeSplines = nodeTable.values<std::int32_t>("spline");
    const std::span<const std::int32_t> nodeIds = nodeTable.values<std::int32_t>("nodeId");
    const input::NodeTable& nodes = snapshot.nodes();
    const std::span<const std::int32_t> gridIds = nodes.ids();
    const base::FixedListView<double> positions = nodes.positionsInBasic();
    std::vector<std::uint8_t> isMoved(boxes.size(), 0);
    std::vector<std::uint8_t> isLoaded(boxes.size(), 0);

    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        Spline spline;
        spline.elementId = elementIds[index];
        spline.card = std::string(types[row]);
        const std::string_view method = hasMethod.isValid(row) ? methods[row] : std::string_view{};
        const SplineKind kind = kindOf(spline.card, method, spline.elementId);
        const std::string_view usage = hasUsage.isValid(row) ? usages[row] : std::string_view{"Both"};
        spline.movesBoxes = usage != "Force";
        spline.loadsStructure = usage != "Displacement";
        const std::string name = base::concatenate({spline.card, " ", std::to_string(spline.elementId)});

        // The boxes: BOX1 to BOX2 of a AeroSurfaceSpline1 or AeroBeamSpline2, the items of the AeroBoxList of a
        // AeroSurfaceSpline4 or AeroBeamSpline5.
        std::vector<std::int32_t> boxIds;
        if (hasAelist.isValid(row))
            boxIds = aelistItems(snapshot, aelists[index], spline.elementId);
        else
            for (std::int32_t id = firsts[index]; id <= lasts[index]; ++id)
                boxIds.push_back(id);
        for (const std::int32_t id : boxIds)
            if (const std::optional<std::size_t> box = boxes.find(id))
                spline.boxes.push_back(*box);
        if (spline.boxes.empty())
            throw base::SolverError(
                base::concatenate({name, ": expected boxes of AeroLiftingPanel1 ", std::to_string(caeros[index]),
                                   ", found none; check its AeroSurfaceSpline1.firstBox, "
                                   "AeroSurfaceSpline1.lastBox or AeroBoxList"}));
        for (const std::size_t box : spline.boxes)
        {
            if ((spline.movesBoxes && isMoved[box] != 0) || (spline.loadsStructure && isLoaded[box] != 0))
                throw base::SolverError(base::concatenate(
                    {name, ": box ", std::to_string(boxes[box].id),
                     " is already splined by another spline of the same usage (a box is splined once); "
                     "remove it from one of them"}));
            if (spline.movesBoxes)
                isMoved[box] = 1;
            if (spline.loadsStructure)
                isLoaded[box] = 1;
        }

        // The structural points, in the aerodynamic frame.
        std::vector<std::int32_t> points;
        for (std::size_t k = 0; k < nodeSplines.size(); ++k)
            if (nodeSplines[k] == spline.elementId)
                points.push_back(nodeIds[k]);
        std::vector<base::Vec3> pointPositions;
        for (const std::int32_t nodeId : points)
        {
            const auto found = std::lower_bound(gridIds.begin(), gridIds.end(), nodeId);
            if (found == gridIds.end() || *found != nodeId)
                throw base::SolverError(base::concatenate(
                    {name, ": expected grid points in its ", spline.card, ".nodeSetId, found ", std::to_string(nodeId),
                     ", which is not a grid point of the model; correct its IdListSet1"}));
            const std::span<const double> p = positions[found - gridIds.begin()];
            pointPositions.push_back(reference.pointInAeroFrame({p[0], p[1], p[2]}));
        }

        // The frame of the spline: z the normal of its panel, x the flow in that plane (a surface spline) or y the axis
        // of its CID in that plane (a beam spline), from an origin on it.
        const base::Vec3 z = boxes[spline.boxes.front()].normal;
        base::Vec3 x = base::normalized(FLOW - base::dot(FLOW, z) * z);
        base::Vec3 y = base::cross(z, x);
        base::Vec3 origin = boxes[spline.boxes.front()].center;
        BeamSpline::Flexibility flexibility;
        if (kind == SplineKind::Beam)
        {
            const input::CardTable* card = snapshot.findAeroCardTable(spline.card);
            const std::optional<std::int64_t> cardRow =
                card != nullptr ? rowOf(card->ids(), spline.elementId) : std::nullopt;
            if (!cardRow)
                throw base::SolverError(base::concatenate(
                    {name, ": expected its card's fields (", spline.card, ".frameId, ", spline.card,
                     ".bendingTorsionRatio, ", spline.card, ".rotationFlexibilityX, ", spline.card,
                     ".rotationFlexibilityY) in the snapshot, found none; take the snapshot with the aero cards"}));
            const base::ContractReader& fields = card->reader();
            const std::int32_t frameId =
                fields.validity("frameId").isValid(*cardRow)
                    ? fields.values<std::int32_t>("frameId",
                                                  base::NullPolicy::Allow)[static_cast<std::size_t>(*cardRow)]
                    : 0;
            const std::string frameLabel = base::concatenate({"SPLINE ", spline.card, ".frameId"});
            const input::CoordinateFrame frame = frameOf(snapshot, frameId, frameLabel.c_str());
            const base::Vec3 axis =
                reference.directionInAeroFrame({frame.axes(0, 1), frame.axes(1, 1), frame.axes(2, 1)});
            y = base::normalized(axis - base::dot(axis, z) * z);
            x = base::cross(y, z);
            origin = reference.pointInAeroFrame(frame.origin);
            flexibility.dtor = realOr(fields, "bendingTorsionRatio", *cardRow, DEFAULT_TORSION_RATIO);
            flexibility.dthx = realOr(fields, "rotationFlexibilityX", *cardRow, NOT_ATTACHED);
            flexibility.dthy = realOr(fields, "rotationFlexibilityY", *cardRow, NOT_ATTACHED);
        }
        const double dz = realOr(table, "dz", row, 0.0);
        flexibility.dz = dz;
        const auto local = [&](base::Vec3 p)
        { return std::array<double, 2>{base::dot(p - origin, x), base::dot(p - origin, y)}; };
        std::vector<std::array<double, 2>> targets;
        for (const std::size_t box : spline.boxes)
            targets.push_back(local(boxes[box].center));
        // The slope along the flow in the plane of the box: ∂h/∂x along the spline's x, ∂h/∂y along its y.
        const base::Vec3 flowInPlane = base::normalized(FLOW - base::dot(FLOW, z) * z);
        const double alongX = base::dot(flowInPlane, x);
        const double alongY = base::dot(flowInPlane, y);

        if (kind == SplineKind::Surface)
        {
            std::vector<std::array<double, 2>> coordinates;
            for (const base::Vec3& p : pointPositions)
                coordinates.push_back(local(p));
            SurfaceSpline::Interpolation interpolation = SurfaceSpline::interpolate(coordinates, dz, targets, budget);
            spline.matrix = linalg::DenseMatrix(budget, 2 * spline.boxes.size(), points.size(), "spline matrix");
            for (std::size_t b = 0; b < spline.boxes.size(); ++b)
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    spline.matrix(2 * b, i) = interpolation.plunge(b, i);
                    spline.matrix(2 * b + 1, i) = interpolation.slope(b, i);
                }
            spline.nodes = points;
            spline.directions.assign(points.size(), z);
            spline.isRotation.assign(points.size(), 0);
        }
        else
        {
            std::vector<double> stations;
            for (const base::Vec3& p : pointPositions)
                stations.push_back(base::dot(p - origin, y));
            BeamSpline::Interpolation interpolation = BeamSpline::interpolate(stations, flexibility, targets, budget);
            const std::size_t unknowns = 3 * points.size();
            spline.matrix = linalg::DenseMatrix(budget, 2 * spline.boxes.size(), unknowns, "spline matrix");
            for (std::size_t b = 0; b < spline.boxes.size(); ++b)
                for (std::size_t u = 0; u < unknowns; ++u)
                {
                    spline.matrix(2 * b, u) = interpolation.plunge(b, u);
                    spline.matrix(2 * b + 1, u) =
                        alongX * interpolation.slope(b, u) + alongY * interpolation.spanSlope(b, u);
                }
            for (const std::int32_t nodeId : points)
            {
                spline.nodes.insert(spline.nodes.end(), {nodeId, nodeId, nodeId});
                spline.directions.insert(spline.directions.end(), {z, x, y});
                spline.isRotation.insert(spline.isRotation.end(), {0, 1, 1});
            }
        }
        m_splines.push_back(std::move(spline));
    }
}

BoxMotion SplineMatrix::boxMotion(const std::function<PointMotion(std::int32_t)>& motionOf) const
{
    BoxMotion motion{std::vector<double>(m_boxes->size(), 0.0), std::vector<double>(m_boxes->size(), 0.0)};
    for (const Spline& spline : m_splines)
    {
        if (!spline.movesBoxes)
            continue;
        for (std::size_t u = 0; u < spline.nodes.size(); ++u)
        {
            const PointMotion point = motionOf(spline.nodes[u]);
            const base::Vec3 vector =
                m_reference.directionInAeroFrame(spline.isRotation[u] != 0 ? point.rotation : point.translation);
            const double value = base::dot(spline.directions[u], vector);
            if (value == 0.0)
                continue;
            for (std::size_t b = 0; b < spline.boxes.size(); ++b)
            {
                motion.plunge[spline.boxes[b]] += spline.matrix(2 * b, u) * value;
                motion.slope[spline.boxes[b]] += spline.matrix(2 * b + 1, u) * value;
            }
        }
    }
    return motion;
}

std::vector<SplineMatrix::PointLoad> SplineMatrix::structuralLoads(std::span<const double> boxForces) const
{
    const AeroBoxes& boxes = *m_boxes;
    std::vector<PointLoad> loads;
    for (const Spline& spline : m_splines)
    {
        if (!spline.loadsStructure)
            continue;
        for (std::size_t u = 0; u < spline.nodes.size(); ++u)
        {
            double load = 0.0;
            for (std::size_t b = 0; b < spline.boxes.size(); ++b)
            {
                const AeroBox& box = boxes[spline.boxes[b]];
                // The force at the force point is the force and the moment F (x_f − x_c) at the center, along the flow
                // in the plane of the box.
                const base::Vec3 flowInPlane = base::normalized(FLOW - base::dot(FLOW, box.normal) * box.normal);
                const double arm = base::dot(box.forcePoint - box.center, flowInPlane);
                const double force = boxForces[spline.boxes[b]];
                load += spline.matrix(2 * b, u) * force + spline.matrix(2 * b + 1, u) * force * arm;
            }
            PointLoad point;
            point.nodeId = spline.nodes[u];
            const base::Vec3 vector = m_reference.aeroFrame.axes * (load * spline.directions[u]);
            if (spline.isRotation[u] != 0)
                point.moment = vector;
            else
                point.force = vector;
            loads.push_back(point);
        }
    }
    return loads;
}

} // namespace oos::aero
