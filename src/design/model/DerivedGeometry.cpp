#include "oos/design/model/DerivedGeometry.hpp"

#include "../common/TableColumns.hpp"
#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/design/model/DesignCodes.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string_view>

namespace oos::design
{

namespace
{

using table_columns::NullableColumn;
using BasisSet = std::vector<std::size_t>; // shape bases, increasing
using Point = DerivedGeometry::Point;
using Frame = DerivedGeometry::Frame;
using Load = DerivedGeometry::Load;
using LoadForm = DerivedGeometry::LoadForm;

// The card and field a refusal names: the shape variables are DesignNodeMove-like terms whatever card gave them.
constexpr const char* SHAPE_CARD = "DesignNodeMove";
constexpr const char* SHAPE_FIELD = "nodeId";
constexpr double NO_SIZE = std::numeric_limits<double>::infinity();

const std::string LOADS_TABLE(input::tableName(input::TableRole::Loads));
const std::string INERTIAL_TABLE(input::tableName(input::TableRole::InertialLoads));
const std::string COORDINATES_TABLE(input::tableName(input::TableRole::CoordinateSystems));
const std::array<std::string, 3> FORCE_COLUMNS{"fx", "fy", "fz"};
const std::array<std::string, 3> MOMENT_COLUMNS{"mx", "my", "mz"};
const std::array<std::string, 3> ACCELERATION_COLUMNS{"ax", "ay", "az"};
const std::array<std::string, 3> DIRECTION_COLUMNS{"directionX", "directionY", "directionZ"};
const std::array<std::string, 3> ORIGIN_COLUMNS{"originX", "originY", "originZ"};
const std::array<std::array<std::string, 3>, 3> AXIS_COLUMNS{std::array<std::string, 3>{"iX", "iY", "iZ"},
                                                             std::array<std::string, 3>{"jX", "jY", "jZ"},
                                                             std::array<std::string, 3>{"kX", "kY", "kZ"}};

[[nodiscard]] BasisSet unite(const BasisSet& left, const BasisSet& right)
{
    BasisSet both;
    std::set_union(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(both));
    return both;
}

[[nodiscard]] bool contains(const BasisSet& set, std::size_t basis)
{
    return std::binary_search(set.begin(), set.end(), basis);
}

// The row of an ID in an int32 column sorted or not, or -1.
[[nodiscard]] std::int64_t rowOfId(std::span<const std::int32_t> ids, std::int32_t id)
{
    const auto sorted = std::lower_bound(ids.begin(), ids.end(), id);
    auto found = sorted != ids.end() && *sorted == id ? sorted : std::find(ids.begin(), ids.end(), id);
    return found == ids.end() ? -1 : static_cast<std::int64_t>(found - ids.begin());
}

// A frame: its origin and its axes (the columns i, j, k in the basic system).
struct Placement
{
    base::Vec3 origin;
    base::Mat3 axes;
};

// RectFrame1: the origin A = G1, the z axis from A to B = G2, C = G3 in the x-z plane: k = (B − A)/|B − A|,
// j = k × (C − A)/|k × (C − A)|, i = j × k (CORD1C and SphericalFrame1 define their reference frames alike).
[[nodiscard]] Placement frameFromPoints(base::Vec3 a, base::Vec3 b, base::Vec3 c)
{
    const base::Vec3 k = base::normalized(b - a);
    const base::Vec3 j = base::normalized(base::cross(k, c - a));
    const base::Vec3 i = base::cross(j, k);
    return {a, base::matrixFromRows(i, j, k).transposed()};
}

// The direction of a PointForceLoad1 or PointMomentLoad1 (from G1 to G2) or of a PointForceLoad2 or PointMomentLoad2
// ((G2 − G1) × (G4 − G3)), not made unit: its length tells a degenerate direction.
[[nodiscard]] base::Vec3 directionOf(LoadForm form, const std::array<base::Vec3, 4>& points)
{
    if (form == LoadForm::TwoPoints)
        return points[1] - points[0];
    return base::cross(points[1] - points[0], points[3] - points[2]);
}

[[nodiscard]] double largestAbsDifference(const base::Mat3& left, const base::Mat3& right)
{
    double largest = 0.0;
    for (std::size_t k = 0; k < left.values.size(); ++k)
        largest = std::max(largest, std::abs(left.values[k] - right.values[k]));
    return largest;
}

[[nodiscard]] std::string vectorText(base::Vec3 v)
{
    return base::concatenate({"(", std::to_string(v[0]), ", ", std::to_string(v[1]), ", ", std::to_string(v[2]), ")"});
}

[[nodiscard]] base::Vec3 column(const base::Mat3& m, std::size_t j)
{
    return {m(0, j), m(1, j), m(2, j)};
}

// Binds a DerivedGeometry: the moving grid points, the moved systems, the derived loads and the refusals.
class Binder
{
public:
    Binder(const ShapeRelations& relations, const input::ModelSnapshot& snapshot,
           std::vector<base::Diagnostic>& problems, std::vector<Point>& points, std::vector<Frame>& frames,
           std::vector<Load>& loads)
        : m_relations(relations),
          m_snapshot(snapshot),
          m_problems(problems),
          m_points(points),
          m_frames(frames),
          m_loads(loads)
    {
        const std::vector<ShapeBasis>& bases = relations.bases();
        for (std::size_t b = 0; b < bases.size(); ++b)
            for (const ShapeMove& move : bases[b].moves)
                if (move.move != std::array<double, 3>{})
                    m_movesOf[move.gridId].emplace_back(b, base::Vec3{move.move[0], move.move[1], move.move[2]});
    }

    void bind()
    {
        if (m_movesOf.empty())
            return;
        bindFrames();
        refuseFrameUsers();
        bindLoadTable();
        bindInertialTable();
        if (!m_frames.empty() &&
            (m_snapshot.hasTable(input::TableRole::AeroBoxes) || m_snapshot.hasTable(input::TableRole::AeroPanels)))
            refuse(m_frames.front().bases,
                   base::concatenate({"the shape variables move coordinate system ",
                                      std::to_string(m_frames.front().frameId),
                                      ", and the model has an aerodynamic model, whose boxes O_o_Reader derives from "
                                      "the CAERO cards in their systems when it reads the model; leave the grid "
                                      "points that define the system out of the shapes"}));
    }

private:
    // The bases that move a grid point (empty when none does).
    [[nodiscard]] BasisSet basesMoving(std::int32_t gridId) const
    {
        BasisSet bases;
        if (const auto found = m_movesOf.find(gridId); found != m_movesOf.end())
            for (const auto& [basis, move] : found->second)
                bases.push_back(basis);
        std::sort(bases.begin(), bases.end());
        bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
        return bases;
    }

    // The index of a grid point among the derived points (added at its first use); nothing when the model has no such
    // grid point (FieldTargets reports the moves of unknown grid points).
    [[nodiscard]] std::optional<std::size_t> pointOf(std::int32_t gridId)
    {
        if (const auto found = m_pointIndex.find(gridId); found != m_pointIndex.end())
            return found->second;
        const input::NodeTable& nodes = m_snapshot.nodes();
        const std::int64_t row = rowOfId(nodes.ids(), gridId);
        if (row < 0)
            return std::nullopt;
        Point point;
        point.gridId = gridId;
        const std::span<const double> xyz = nodes.positionsInBasic()[row];
        point.base = {xyz[0], xyz[1], xyz[2]};
        if (const auto moving = m_movesOf.find(gridId); moving != m_movesOf.end())
            point.moves = moving->second;
        m_points.push_back(std::move(point));
        m_pointIndex.emplace(gridId, m_points.size() - 1);
        return m_points.size() - 1;
    }

    void refuse(const BasisSet& bases, const std::string& why)
    {
        const std::int32_t variableId = bases.empty() ? 0 : m_relations.bases()[bases.front()].variableId;
        m_problems.push_back(designProblem(
            design_code::UNSUPPORTED_TARGET, SHAPE_CARD, variableId, SHAPE_FIELD,
            base::concatenate({why,
                               " (O_o_Reader resolves it when it reads the model; it "
                               "gives the grid points of the PointForceLoad1, PointForceLoad2, PointMomentLoad1, "
                               "PointMomentLoad2 and CORD1x cards, "
                               "from which O_o_Solver derives their directions and frames at a design point)"})));
    }

    // The systems the shapes move: the CORD1x whose G1, G2 or G3 moves, then those defined in a moved one (RID), to
    // the end of the chains; a frame's parent comes before it.
    void bindFrames()
    {
        if (!m_snapshot.hasTable(input::TableRole::CoordinateSystems))
            return;
        const input::CoordinateSystemTable& systems = m_snapshot.coordinateSystems();
        const base::ContractReader& reader = systems.reader();
        const std::span<const std::int32_t> frameIds = systems.ids();
        const NullableColumn<std::int32_t> definition(reader, "definition");
        const NullableColumn<std::int32_t> referenceFrame(reader, "referenceFrameId");
        const std::array<NullableColumn<std::int32_t>, 3> grids{NullableColumn<std::int32_t>(reader, "g1"),
                                                                NullableColumn<std::int32_t>(reader, "g2"),
                                                                NullableColumn<std::int32_t>(reader, "g3")};
        const bool hasGrids = reader.hasColumn("g1") && reader.hasColumn("g2") && reader.hasColumn("g3");
        for (std::int64_t row = 0; row < systems.rowCount(); ++row)
        {
            const std::int32_t frameId = frameIds[static_cast<std::size_t>(row)];
            if (!hasGrids)
            {
                if (definition[row].value_or(0) == 1)
                    refuse({}, base::concatenate({"coordinate system ", std::to_string(frameId),
                                                  " is a CORD1x, whose grid points coordsTable does not give (a "
                                                  "O_o_Reader before data contract 44), so a shape cannot be checked "
                                                  "against it; install a O_o_Reader of data contract 44 or later"}));
                continue;
            }
            if (!grids[0][row] || !grids[1][row] || !grids[2][row])
                continue;
            BasisSet bases;
            for (const NullableColumn<std::int32_t>& grid : grids)
                bases = unite(bases, basesMoving(*grid[row]));
            if (bases.empty())
                continue;
            bindPointFrame(systems, row, frameId, bases, {*grids[0][row], *grids[1][row], *grids[2][row]});
        }
        // The systems defined in moved ones, by passes until none is added (a chain is finite: O_o_Reader refuses a
        // circular one).
        bool isGrown = !m_frames.empty();
        while (isGrown)
        {
            isGrown = false;
            for (std::int64_t row = 0; row < systems.rowCount(); ++row)
            {
                const std::int32_t frameId = frameIds[static_cast<std::size_t>(row)];
                const std::optional<std::int32_t> parent = referenceFrame[row];
                if (!parent || *parent == 0 || m_frameIndex.count(frameId) != 0)
                    continue;
                const auto found = m_frameIndex.find(*parent);
                if (found == m_frameIndex.end())
                    continue;
                Frame frame;
                frame.frameId = frameId;
                frame.row = row;
                frame.isFromPoints = false;
                frame.parent = found->second;
                frame.bases = m_frames[found->second].bases;
                const input::CoordinateFrame placed = systems.frameAt(row);
                frame.isRectangular = placed.kind == input::CoordinateFrame::Kind::Rectangular;
                frame.origin = placed.origin;
                frame.axes = placed.axes;
                m_frameIndex.emplace(frameId, m_frames.size());
                m_frames.push_back(std::move(frame));
                isGrown = true;
            }
        }
    }

    // A CORD1x whose grid points move: bound when the frame from its base points is O_o_Reader's.
    void bindPointFrame(const input::CoordinateSystemTable& systems, std::int64_t row, std::int32_t frameId,
                        const BasisSet& bases, const std::array<std::int32_t, 3>& gridIds)
    {
        Frame frame;
        frame.frameId = frameId;
        frame.row = row;
        frame.bases = bases;
        const input::CoordinateFrame placed = systems.frameAt(row);
        frame.isRectangular = placed.kind == input::CoordinateFrame::Kind::Rectangular;
        frame.origin = placed.origin;
        frame.axes = placed.axes;
        for (std::size_t k = 0; k < 3; ++k)
        {
            const std::optional<std::size_t> point = pointOf(gridIds[k]);
            if (!point)
                return; // a grid point the model does not have: O_o_Reader could not resolve the system either
            frame.points[k] = *point;
        }
        const base::Vec3 a = m_points[frame.points[0]].base;
        const base::Vec3 b = m_points[frame.points[1]].base;
        const base::Vec3 c = m_points[frame.points[2]].base;
        frame.size = std::min(base::norm(b - a), base::norm(c - a));
        const double width = base::norm(base::cross(b - a, c - a));
        bool isSame = frame.size > 0.0 && width > 0.0;
        if (isSame)
        {
            const Placement derived = frameFromPoints(a, b, c);
            const double scale = std::max({1.0, base::norm(a), frame.size});
            isSame = base::norm(derived.origin - placed.origin) <= DERIVED_CHECK_TOLERANCE * scale &&
                     largestAbsDifference(derived.axes, placed.axes) <= DERIVED_CHECK_TOLERANCE;
        }
        if (!isSame)
        {
            refuse(bases,
                   base::concatenate(
                       {"the shape variables move the grid points of coordinate system ", std::to_string(frameId),
                        " (a RectFrame1, CylindricalFrame1 or SphericalFrame1), whose frame "
                        "O_o_Solver derives "
                        "from G1, G2, G3 as the format defines it (origin "
                        "G1, z along G1 to G2, G3 in the x-z plane) and finds ",
                        frame.size > 0.0 && width > 0.0 ? "another frame than" : "undefined (aligned points), unlike",
                        " the one O_o_Reader resolved (origin ", vectorText(placed.origin), ", z axis ",
                        vectorText(column(placed.axes, 2)), "); leave its grid points out of the shapes"}));
            return;
        }
        m_frameIndex.emplace(frameId, m_frames.size());
        m_frames.push_back(std::move(frame));
    }

    // The grid points located in or displaced in a moved system, and the DesignNodeMove terms along one (the class
    // comment).
    void refuseFrameUsers()
    {
        if (m_frames.empty())
            return;
        const input::NodeTable& nodes = m_snapshot.nodes();
        const std::span<const std::int32_t> ids = nodes.ids();
        const std::span<const std::int32_t> positionFrameIds = nodes.positionFrameIds();
        const std::span<const std::int32_t> displacementFrameIds = nodes.displacementFrameIds();
        for (std::int64_t row = 0; row < nodes.rowCount(); ++row)
        {
            const auto r = static_cast<std::size_t>(row);
            if (const auto found = m_frameIndex.find(positionFrameIds[r]); found != m_frameIndex.end())
                refuse(m_frames[found->second].bases,
                       base::concatenate({"grid point ", std::to_string(ids[r]),
                                          " is located (Node.positionFrameId) in coordinate system ",
                                          std::to_string(positionFrameIds[r]),
                                          ", which the shape variables move: its position in the basic system would "
                                          "follow the system; give it in another system, or leave the grid points "
                                          "that define the system out of the shapes"}));
            if (const auto found = m_frameIndex.find(displacementFrameIds[r]); found != m_frameIndex.end())
                refuse(
                    m_frames[found->second].bases,
                    base::concatenate({"grid point ", std::to_string(ids[r]),
                                       " has its displacements (Node.displacementFrameId) in coordinate system ",
                                       std::to_string(displacementFrameIds[r]),
                                       ", which the shape variables move (displacementRotations would turn with "
                                       "it); give it another Node.displacementFrameId, or leave the grid points that "
                                       "define the system "
                                       "out of the shapes"}));
        }
        for (const ShapeTerm& term : m_relations.terms())
            if (const auto found = m_frameIndex.find(term.frameId); term.frameId != 0 && found != m_frameIndex.end())
                refuse(m_frames[found->second].bases,
                       base::concatenate({"the DesignNodeMove of DesignVar ", std::to_string(term.variableId),
                                          " on grid point ", std::to_string(term.gridId),
                                          " gives its vector in coordinate system ", std::to_string(term.frameId),
                                          ", which the shape variables move (O_o_Reader takes N along "
                                          "DesignNodeMove.frameId when it "
                                          "reads the model); give N in a system no shape moves"}));
    }

    // Whether a system of the model is rectangular; nothing for an unknown CID.
    [[nodiscard]] std::optional<bool> isRectangular(std::int32_t frameId) const
    {
        if (!m_snapshot.hasTable(input::TableRole::CoordinateSystems))
            return std::nullopt;
        const input::CoordinateSystemTable& systems = m_snapshot.coordinateSystems();
        const std::int64_t row = rowOfId(systems.ids(), frameId);
        if (row < 0)
            return std::nullopt;
        return systems.frameAt(row).kind == input::CoordinateFrame::Kind::Rectangular;
    }

    [[nodiscard]] std::string loadText(std::string_view card, std::int32_t gridId, std::int64_t row) const
    {
        const std::string set = std::to_string(m_snapshot.loads().loadSetIds()[static_cast<std::size_t>(row)]);
        if (gridId == 0)
            return base::concatenate({card, " of load set ", set});
        return base::concatenate({card, " on grid point ", std::to_string(gridId), " (load set ", set, ")"});
    }

    // loadsTable: PointForceLoad1, PointForceLoad2, PointMomentLoad1, PointMomentLoad2 from their moved grid points;
    // FORCE, MOMENT and GravityLoad given in a moved system.
    void bindLoadTable()
    {
        if (!m_snapshot.hasTable(input::TableRole::Loads))
            return;
        const base::ContractReader& table = m_snapshot.loads().loads();
        const base::DictionaryView cards = m_snapshot.loads().loadCards();
        const NullableColumn<std::int32_t> targets(table, "targetId");
        const NullableColumn<std::int32_t> frameIds(table, "frameId");
        const std::array<NullableColumn<std::int32_t>, 4> grids{
            NullableColumn<std::int32_t>(table, "g1"), NullableColumn<std::int32_t>(table, "g2"),
            NullableColumn<std::int32_t>(table, "g3"), NullableColumn<std::int32_t>(table, "g4")};
        const bool hasGrids =
            table.hasColumn("g1") && table.hasColumn("g2") && table.hasColumn("g3") && table.hasColumn("g4");
        std::map<std::string, NullableColumn<double>> reals;
        for (const auto* names : {&FORCE_COLUMNS, &MOMENT_COLUMNS, &ACCELERATION_COLUMNS})
            for (const std::string& name : *names)
                reals.emplace(name, NullableColumn<double>(table, name));
        const auto vectorAt = [&](const std::array<std::string, 3>& columns, std::int64_t row)
        {
            base::Vec3 v;
            for (std::size_t k = 0; k < 3; ++k)
                v[k] = reals.at(columns[k])[row].value_or(0.0);
            return v;
        };
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            const std::string_view card = cards[row];
            const std::int32_t gridId = targets[row].value_or(0);
            const bool isOnePair = card == "PointForceLoad1" || card == "PointMomentLoad1";
            const bool isTwoPairs = card == "PointForceLoad2" || card == "PointMomentLoad2";
            const std::array<std::string, 3>& columns =
                card.starts_with("PointMomentLoad") ? MOMENT_COLUMNS
                                                    : (card == "GravityLoad" ? ACCELERATION_COLUMNS : FORCE_COLUMNS);
            if (isOnePair || isTwoPairs)
            {
                if (!hasGrids)
                {
                    refuse({}, base::concatenate({"the direction of ", loadText(card, gridId, row),
                                                  " comes from grid points that loadsTable does not give (a "
                                                  "O_o_Reader before data contract 44), so a shape cannot be checked "
                                                  "against it; install a O_o_Reader of data contract 44 or later"}));
                    continue;
                }
                std::array<std::optional<std::int32_t>, 4> defining{grids[0][row], grids[1][row], grids[2][row],
                                                                    grids[3][row]};
                bindDirection(row, card, isTwoPairs ? LoadForm::FourPoints : LoadForm::TwoPoints, defining, gridId,
                              columns, vectorAt(columns, row));
                continue;
            }
            const std::optional<std::int32_t> frameId = frameIds[row];
            const bool isNodal = card == "PointForceLoad" || card == "PointMomentLoad";
            if (!frameId || *frameId == 0 || (!isNodal && card != "GravityLoad"))
                continue;
            const auto frame = m_frameIndex.find(*frameId);
            if (frame == m_frameIndex.end())
            {
                // FORCE: the components in CID at the grid point, whose curvilinear directions turn with it.
                const BasisSet moving = isNodal ? basesMoving(gridId) : BasisSet{};
                const std::optional<bool> rectangular = isRectangular(*frameId);
                if (!moving.empty() && rectangular.has_value() && !*rectangular)
                    refuse(moving, base::concatenate({loadText(card, gridId, row),
                                                      " is given in the curvilinear system ", std::to_string(*frameId),
                                                      ", whose directions at the point turn when the shape moves it; "
                                                      "give the load in a rectangular system"}));
                continue;
            }
            const Frame& moved = m_frames[frame->second];
            if (!moved.isRectangular)
            {
                refuse(moved.bases, base::concatenate({loadText(card, gridId, row),
                                                       " is given in the curvilinear system ", std::to_string(*frameId),
                                                       ", which the shape variables move; give it in a rectangular "
                                                       "system"}));
                continue;
            }
            Load load;
            load.form = LoadForm::InFrame;
            load.table = LOADS_TABLE;
            load.row = row;
            load.columns = columns;
            load.frame = frame->second;
            load.vector = vectorAt(columns, row);
            load.gridId = isNodal ? gridId : 0;
            load.isConcentrated = isNodal;
            load.bases = moved.bases;
            load.size = NO_SIZE;
            m_loads.push_back(std::move(load));
        }
    }

    // A PointForceLoad1, PointForceLoad2, PointMomentLoad1 or PointMomentLoad2 whose grid points move: bound when its
    // direction from the base points is O_o_Reader's.
    void bindDirection(std::int64_t row, std::string_view card, LoadForm form,
                       const std::array<std::optional<std::int32_t>, 4>& defining, std::int32_t gridId,
                       const std::array<std::string, 3>& columns, base::Vec3 vector)
    {
        const std::size_t count = form == LoadForm::FourPoints ? 4 : 2;
        BasisSet bases;
        for (std::size_t k = 0; k < count; ++k)
            if (defining[k])
                bases = unite(bases, basesMoving(*defining[k]));
        if (bases.empty())
            return;
        const std::string what = loadText(card, gridId, row);
        Load load;
        load.form = form;
        load.table = LOADS_TABLE;
        load.row = row;
        load.columns = columns;
        load.gridId = gridId;
        load.bases = bases;
        std::array<base::Vec3, 4> positions{};
        for (std::size_t k = 0; k < count; ++k)
        {
            const std::optional<std::size_t> point = defining[k] ? pointOf(*defining[k]) : std::nullopt;
            if (!point)
            {
                refuse(bases, base::concatenate({what, ": its G", std::to_string(k + 1),
                                                 defining[k] ? " is not a grid point of the model" : " is blank",
                                                 ", so O_o_Solver cannot derive its direction again; give every grid "
                                                 "point of the card"}));
                return;
            }
            load.points[k] = *point;
            positions[k] = m_points[*point].base;
        }
        load.size = base::norm(positions[1] - positions[0]);
        if (form == LoadForm::FourPoints)
            load.size = std::min(load.size, base::norm(positions[3] - positions[2]));
        const double length = base::norm(vector);
        if (length == 0.0)
            return; // a load of zero magnitude has no direction to derive
        const base::Vec3 direction = directionOf(form, positions);
        const double directionLength = base::norm(direction);
        if (!(directionLength > 0.0))
        {
            refuse(bases, base::concatenate({what,
                                             ": its direction is undefined at the base positions (coincident or "
                                             "parallel grid points); correct the card"}));
            return;
        }
        const base::Vec3 unit = direction / directionLength;
        load.magnitude = base::dot(vector, unit);
        if (base::norm(vector - load.magnitude * unit) > DERIVED_CHECK_TOLERANCE * length)
        {
            refuse(bases, base::concatenate({what,
                                             ": O_o_Solver derives its direction from its grid points as the format "
                                             "defines it and finds ",
                                             vectorText(unit), ", unlike the load O_o_Reader gives, ",
                                             vectorText(vector), "; leave its grid points out of the shapes"}));
            return;
        }
        m_loads.push_back(std::move(load));
    }

    // inertialLoadsTable: the direction of a GravityLoad, ACCEL, AccelNodeLoad or RotationalForceLoad given in a moved
    // system.
    void bindInertialTable()
    {
        if (m_frames.empty() || !m_snapshot.hasTable(input::TableRole::InertialLoads) ||
            !m_snapshot.loads().hasInertialLoads())
            return;
        const base::ContractReader& table = m_snapshot.loads().inertialLoads();
        const base::DictionaryView cards = table.dictionaryTexts("card");
        const NullableColumn<std::int32_t> frameIds(table, "frameId");
        const std::array<NullableColumn<double>, 3> directions{NullableColumn<double>(table, DIRECTION_COLUMNS[0]),
                                                               NullableColumn<double>(table, DIRECTION_COLUMNS[1]),
                                                               NullableColumn<double>(table, DIRECTION_COLUMNS[2])};
        for (std::int64_t row = 0; row < table.rowCount(); ++row)
        {
            const std::optional<std::int32_t> frameId = frameIds[row];
            if (!frameId || *frameId == 0)
                continue;
            const auto frame = m_frameIndex.find(*frameId);
            if (frame == m_frameIndex.end())
                continue;
            const Frame& moved = m_frames[frame->second];
            if (!moved.isRectangular)
            {
                refuse(moved.bases,
                       base::concatenate({cards[row], " is given in the curvilinear system ", std::to_string(*frameId),
                                          ", which the shape variables move; give it in a rectangular "
                                          "system"}));
                continue;
            }
            Load load;
            load.form = LoadForm::InFrame;
            load.table = INERTIAL_TABLE;
            load.row = row;
            load.columns = DIRECTION_COLUMNS;
            load.frame = frame->second;
            for (std::size_t k = 0; k < 3; ++k)
                load.vector[k] = directions[k][row].value_or(0.0);
            load.gridId = 0;
            load.isConcentrated = false;
            load.bases = moved.bases;
            load.size = NO_SIZE;
            m_loads.push_back(std::move(load));
        }
    }

    const ShapeRelations& m_relations;
    const input::ModelSnapshot& m_snapshot;
    std::vector<base::Diagnostic>& m_problems;
    std::vector<Point>& m_points;
    std::vector<Frame>& m_frames;
    std::vector<Load>& m_loads;
    std::map<std::int32_t, std::vector<std::pair<std::size_t, base::Vec3>>> m_movesOf; // grid → (basis, move)
    std::map<std::int32_t, std::size_t> m_pointIndex;                                  // grid → index of its point
    std::map<std::int32_t, std::size_t> m_frameIndex;                                  // CID → index of its frame
};

} // namespace

DerivedGeometry::DerivedGeometry(const ShapeRelations& relations, const input::ModelSnapshot& snapshot,
                                 std::vector<base::Diagnostic>& problems)
{
    Binder(relations, snapshot, problems, m_points, m_frames, m_loads).bind();
}

std::vector<std::int32_t> DerivedGeometry::systemsMovedBy(std::size_t basis) const
{
    std::vector<std::int32_t> frameIds;
    for (const Frame& frame : m_frames)
        if (contains(frame.bases, basis))
            frameIds.push_back(frame.frameId);
    std::sort(frameIds.begin(), frameIds.end());
    return frameIds;
}

bool DerivedGeometry::changesConcentratedLoads(std::size_t basis) const
{
    return std::any_of(m_loads.begin(), m_loads.end(),
                       [&](const Load& load) { return load.isConcentrated && contains(load.bases, basis); });
}

bool DerivedGeometry::changesEveryUnit(std::size_t basis) const
{
    return std::any_of(m_frames.begin(), m_frames.end(),
                       [&](const Frame& frame) { return contains(frame.bases, basis); });
}

std::vector<std::int32_t> DerivedGeometry::loadGridsOf(std::size_t basis) const
{
    std::vector<std::int32_t> grids;
    for (const Load& load : m_loads)
        if (load.isConcentrated && load.gridId != 0 && contains(load.bases, basis))
            grids.push_back(load.gridId);
    std::sort(grids.begin(), grids.end());
    grids.erase(std::unique(grids.begin(), grids.end()), grids.end());
    return grids;
}

double DerivedGeometry::smallestSize(std::size_t basis) const
{
    double smallest = NO_SIZE;
    for (const Frame& frame : m_frames)
        if (frame.isFromPoints && contains(frame.bases, basis))
            smallest = std::min(smallest, frame.size);
    for (const Load& load : m_loads)
        if (load.form != LoadForm::InFrame && contains(load.bases, basis))
            smallest = std::min(smallest, load.size);
    return smallest;
}

void DerivedGeometry::addCells(std::span<const double> shapes, input::ColumnOverlay& overlay) const
{
    if (isEmpty())
        return;
    // The positions g = g0 + Σ_v s_v·T_v(g), the bases added in order (as FieldTargets::movedPositions).
    std::vector<base::Vec3> positions; // bounded by the derived points
    positions.reserve(m_points.size());
    for (const Point& point : m_points)
    {
        base::Vec3 position = point.base;
        for (const auto& [basis, move] : point.moves)
            for (std::size_t k = 0; k < 3; ++k)
                position[k] += shapes[basis] * move[k];
        positions.push_back(position);
    }
    // The frames, each parent before its children (the class comment gives the formulas).
    std::vector<Placement> placed; // bounded by the moved systems
    placed.reserve(m_frames.size());
    for (const Frame& frame : m_frames)
    {
        if (frame.isFromPoints)
        {
            placed.push_back(
                frameFromPoints(positions[frame.points[0]], positions[frame.points[1]], positions[frame.points[2]]));
        }
        else
        {
            const Frame& parent = m_frames[frame.parent];
            const Placement& moved = placed[frame.parent];
            const base::Mat3 turn = moved.axes * parent.axes.transposed();
            placed.push_back({moved.origin + turn * (frame.origin - parent.origin), turn * frame.axes});
        }
        const Placement& now = placed.back();
        for (std::size_t k = 0; k < 3; ++k)
            overlay.setValue(COORDINATES_TABLE, ORIGIN_COLUMNS[k],
                             {frame.row, input::CellValue::SCALAR_ITEM, now.origin[k]});
        for (std::size_t axis = 0; axis < 3; ++axis)
            for (std::size_t k = 0; k < 3; ++k)
                overlay.setValue(COORDINATES_TABLE, AXIS_COLUMNS[axis][k],
                                 {frame.row, input::CellValue::SCALAR_ITEM, now.axes(k, axis)});
    }
    for (const Load& load : m_loads)
    {
        base::Vec3 vector;
        if (load.form == LoadForm::InFrame)
        {
            // v' = A'·Aᵀ·v: the vector keeps its components in the system that turns.
            const base::Mat3 turn = placed[load.frame].axes * m_frames[load.frame].axes.transposed();
            vector = turn * load.vector;
        }
        else
        {
            std::array<base::Vec3, 4> points{};
            const std::size_t count = load.form == LoadForm::FourPoints ? 4 : 2;
            for (std::size_t k = 0; k < count; ++k)
                points[k] = positions[load.points[k]];
            vector = load.magnitude * base::normalized(directionOf(load.form, points));
        }
        for (std::size_t k = 0; k < 3; ++k)
            overlay.setValue(load.table, load.columns[k], {load.row, input::CellValue::SCALAR_ITEM, vector[k]});
    }
}

} // namespace oos::design
