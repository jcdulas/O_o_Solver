#include "oos/assembly/constraints/GlueConstraints.hpp"

#include "oos/assembly/constraints/FaceShapeFunctions.hpp"
#include "oos/assembly/constraints/GlueSearch.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

namespace oos::assembly
{

namespace
{

constexpr int TRANSLATIONS = 3;
constexpr std::uint8_t TRANSLATION_MASK = 0b000111;

// What another constraint does to a DOF.
constexpr std::uint8_t FREE = 0;
constexpr std::uint8_t HELD = 1;      // SPC, GRID PS
constexpr std::uint8_t DEPENDENT = 2; // rigid element, fastener, MPC

// The card whose glue fields (IGLUE, JGLUE, ERROR) a message names, for every card that sets them.
constexpr std::string_view CONTACT_TABLE_CARD = "ContactTable";

// One tied point (model-sized: budgeted, so trivially copyable).
struct GlueRecord
{
    std::int32_t pairIndex = 0;
    std::int32_t pointRow = 0; // grid row of the secondary point
    std::int32_t facePointCount = 0;
    std::array<std::int32_t, MAX_GLUE_FACE_POINTS> faceRows{}; // −1 for an absent mid-side point
    std::array<double, MAX_GLUE_FACE_POINTS> shapeValues{};
    std::uint8_t componentMask = 0; // bit c − 1: component c of the point is tied
};

// One face of a row of contactNodePairs, keyed by its pair (sorted, then made distinct).
struct FaceKey
{
    std::int32_t pairIndex = 0;
    std::int32_t pointCount = 0;
    std::array<std::int32_t, MAX_GLUE_FACE_POINTS> gridIds{};

    [[nodiscard]] auto tie() const { return std::tie(pairIndex, pointCount, gridIds); }
};

// One candidate secondary point of a row, keyed by its pair.
struct PointKey
{
    std::int32_t pairIndex = 0;
    std::int32_t nodeId = 0;
};

// Why a candidate point is not tied although a face takes it, per glued pair.
enum class LeftOut : std::size_t
{
    Constrained,
    GluedTwice,
    Loop
};
constexpr std::size_t LEFT_OUT_REASONS = 3;

struct PairTally
{
    std::array<std::int64_t, LEFT_OUT_REASONS> counts{};
    std::array<std::int32_t, LEFT_OUT_REASONS> firstPoint{};
    std::string firstConstraint;      // what holds the first constrained point
    std::int64_t partlyTiedCount = 0; // points of which some components keep another constraint
    std::int32_t firstPartlyTied = 0;
    std::string firstPartConstraint;
    std::int64_t invalidFaceCount = 0; // faces that cannot interpolate
    std::string firstInvalidFace;
    std::int64_t unknownPointCount = 0; // candidate grid points the model does not define
    std::int32_t firstUnknownPoint = 0;
    std::int32_t firstOutside = 0;     // the first point no face takes
    double firstOutsideDistance = 0.0; // its distance to the nearest face looked at (infinity: none near)
    double largestGap = 0.0;           // the largest distance from a tied point to its face
    std::int32_t largestGapPoint = 0;

    void leaveOut(LeftOut reason, std::int32_t nodeId)
    {
        const auto r = static_cast<std::size_t>(reason);
        if (counts[r]++ == 0)
            firstPoint[r] = nodeId;
    }
};

// Calls visit(begin, end) for the rows of each rigid element of the table (one elementId, contiguous).
template <typename Visit>
void forEachRigidElement(const input::RigidTable& table, Visit visit)
{
    const std::span<const std::int32_t> elementIds = table.ids();
    std::int64_t begin = 0;
    while (begin < table.rowCount())
    {
        std::int64_t end = begin + 1;
        while (end < table.rowCount() &&
               elementIds[static_cast<std::size_t>(end)] == elementIds[static_cast<std::size_t>(begin)])
            ++end;
        visit(begin, end);
        begin = end;
    }
}

// What the other constraints of the subcase do to each g-set DOF (FREE, HELD, DEPENDENT).
base::BudgetedBuffer<std::uint8_t> constraintStates(const GlueContext& context, const DofMap& dofMap,
                                                    base::MemoryBudget& budget)
{
    base::BudgetedBuffer<std::uint8_t> states(budget, static_cast<std::size_t>(dofMap.gSetSize()),
                                              "glue: constrained DOFs");
    for (const base::DofIndex dof : context.spcDofs)
        states[static_cast<std::size_t>(dof)] = HELD;
    for (const base::DofIndex dof : context.permanentDofs)
        states[static_cast<std::size_t>(dof)] = HELD;
    if (context.mpc != nullptr)
        for (std::size_t i = 0; i < context.mpc->size(); ++i)
            states[static_cast<std::size_t>(context.mpc->dependentDof(i))] = DEPENDENT;
    if (context.fasteners != nullptr)
        for (std::size_t i = 0; i < context.fasteners->rowCount(); ++i)
            states[static_cast<std::size_t>(context.fasteners->dependentDof(i))] = DEPENDENT;
    if (context.rigidElements != nullptr)
    {
        // The rigid elements' dependents as RigidElimination will take them (their problems are reported there).
        const RigidElementEquations equations(dofMap);
        LocalConstraintRows local;
        base::DiagnosticSink scratch;
        forEachRigidElement(*context.rigidElements,
                            [&](std::int64_t begin, std::int64_t end)
                            {
                                if (!equations.build(*context.rigidElements, begin, end, local, scratch))
                                    return;
                                for (const base::DofIndex dof : local.dependents)
                                    states[static_cast<std::size_t>(dof)] = DEPENDENT;
                            });
    }
    return states;
}

ConstraintSource sourceOfCard(std::string_view card)
{
    return card == "ContactPair" ? ConstraintSource::ContactPair : ConstraintSource::ContactTable;
}

base::Diagnostic pairDiagnostic(base::Severity severity, std::string code, const GluedPair& pair, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = pair.card;
    diagnostic.id = pair.tableId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

std::string pairName(const GluedPair& pair)
{
    return base::concatenate({"the glued pair of bodies ", std::to_string(pair.secondaryBody), " (secondary) and ",
                              std::to_string(pair.primaryBody), " (primary) of ", pair.table});
}

// The GlueFace of a face key, or why it cannot interpolate: a number of points other than 3, 4, 6, 8 (a
// SlidelineSegments segment has 2), an absent corner, a grid point the model does not define, coincident corners.
std::variant<GlueFace, std::string> glueFaceOf(const FaceKey& key, const DofMap& dofMap)
{
    if (key.pointCount == 2)
        return std::string("a segment of a SlidelineSegments line, which the tie of a surface does not take");
    if (key.pointCount != 3 && key.pointCount != 4 && key.pointCount != 6 && key.pointCount != 8)
        return base::concatenate(
            {"expected a face of 3, 4, 6 or 8 grid points, found ", std::to_string(key.pointCount)});
    const std::span<const std::int32_t> grids(key.gridIds.data(), static_cast<std::size_t>(key.pointCount));
    if (const auto shape = faceShapeFunctions(grids, 0.0, 0.0); std::holds_alternative<std::string>(shape))
        return std::get<std::string>(shape);
    GlueFace face;
    face.pointCount = key.pointCount;
    face.gridIds = key.gridIds;
    for (std::size_t k = 0; k < grids.size(); ++k)
    {
        if (grids[k] == 0)
            continue; // an absent mid-side point (never a corner: checked above)
        const std::optional<std::int32_t> row = dofMap.points().findGrid(grids[k]);
        if (!row)
            return base::concatenate({"its grid point ", std::to_string(grids[k]), " is not in the model"});
        face.positions[k] = dofMap.positionInBasic(*row);
    }
    const std::size_t corners = face.cornerCount();
    for (std::size_t a = 0; a < corners; ++a)
        for (std::size_t b = a + 1; b < corners; ++b)
            if (!(base::norm(face.positions[a] - face.positions[b]) > 0.0))
                return std::string("two of its corners are at the same place");
    return face;
}

// The diagnostics of one pair once its points are processed. The glue fields are named by their keywords (`names`);
// the fields of a contact table stand for those of every card that sets them.
void reportPair(const GluedPairReport& report, const PairTally& tally, base::EntityId subcase,
                base::DiagnosticSink& sink)
{
    const GluedPair& pair = report.pair;
    const std::string glueOptionLabel = std::string("GlueOption");
    const auto count = [&](LeftOut reason) { return tally.counts[static_cast<std::size_t>(reason)]; };
    const auto first = [&](LeftOut reason)
    { return std::to_string(tally.firstPoint[static_cast<std::size_t>(reason)]); };
    if (tally.invalidFaceCount > 0)
        sink.report(pairDiagnostic(
            base::Severity::Error, "invalid-contact-face", pair,
            base::concatenate({pairName(pair), ": ", std::to_string(tally.invalidFaceCount),
                               " faces of the primary surface (contactNodePairs) cannot tie a grid point; the "
                               "first: ",
                               tally.firstInvalidFace,
                               "; glue the bodies by the faces of shells or solids, or take the snapshot again"})));
    if (tally.unknownPointCount > 0)
        sink.report(pairDiagnostic(
            base::Severity::Error, "invalid-contact-node-pair", pair,
            base::concatenate({pairName(pair), ": ", std::to_string(tally.unknownPointCount),
                               " grid points of the secondary surface (contactNodePairs) are not in the model (the "
                               "first, ",
                               std::to_string(tally.firstUnknownPoint),
                               "); the contact tables do not belong to this model: take the snapshot again"})));
    if (report.outsidePointCount > 0)
    {
        const std::string distance =
            std::isfinite(tally.firstOutsideDistance)
                ? base::concatenate({"at ", std::to_string(tally.firstOutsideDistance), " from the nearest face"})
                : std::string("with no face near");
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-points-outside-tolerance", pair,
            base::concatenate({pairName(pair), ": ", std::to_string(report.outsidePointCount), " of the ",
                               std::to_string(report.candidatePointCount),
                               " grid points of the secondary surface near the primary surface are not tied: no "
                               "primary face is within the position tolerance ",
                               std::to_string(report.positionTolerance),
                               " (orthogonal distance) and the in-plane "
                               "tolerance ",
                               std::to_string(report.inPlaneTolerance), " (the first, grid point ",
                               std::to_string(tally.firstOutside), ", ", distance,
                               "); give the pair an ERROR to tie farther points"})));
    }
    std::string leftOut;
    if (count(LeftOut::Constrained) > 0)
        leftOut.append(base::concatenate({std::to_string(count(LeftOut::Constrained)),
                                          " already dependent or held by another constraint (the first, grid point ",
                                          first(LeftOut::Constrained), ", by ", tally.firstConstraint, ")"}));
    if (count(LeftOut::GluedTwice) > 0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-point-twice", pair,
            base::concatenate({pairName(pair), ": ", std::to_string(count(LeftOut::GluedTwice)),
                               " grid points are tied by an earlier pair already (the first, grid point ",
                               first(LeftOut::GluedTwice),
                               "); a DOF is dependent once only: the earlier tie is kept"})));
    if (count(LeftOut::Loop) > 0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-dependency-cycle", pair,
            base::concatenate({pairName(pair), ": the glue equations of ", std::to_string(count(LeftOut::Loop)),
                               " grid points would close a loop of dependencies (the first, grid point ",
                               first(LeftOut::Loop),
                               ", depends through the faces it meets on itself): they are not tied; glue the bodies "
                               "one way only"})));
    if (!leftOut.empty())
        sink.report(
            pairDiagnostic(base::Severity::Warning, "glue-points-left-out", pair,
                           base::concatenate({pairName(pair), " in subcase ", std::to_string(subcase),
                                              ": grid points of the secondary surface left out of the glue: ", leftOut,
                                              "; the other constraint keeps them; check that this is meant"})));
    if (tally.partlyTiedCount > 0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-components-left", pair,
            base::concatenate({pairName(pair), " in subcase ", std::to_string(subcase), ": ",
                               std::to_string(tally.partlyTiedCount),
                               " grid points of the secondary surface have components held or dependent already (the "
                               "first, grid point ",
                               std::to_string(tally.firstPartlyTied), ", by ", tally.firstPartConstraint,
                               "): those components keep their constraint, the others are tied"})));
    if (pair.asksForRotations() && report.tiedPointCount > 0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-rotations-not-tied", pair,
            base::concatenate({pairName(pair), ": ", glueOptionLabel, " ", std::to_string(pair.glueOption),
                               " asks for a glue of the shell rotations too; O_o_Solver ties the translations of the "
                               "grid points only (the tie of a surface; the rotations of a shell grid point stay "
                               "free): a moment passes through tie forces at points apart, not at a single line of "
                               "points"})));
    if (tally.largestGap > 0.0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-gap-kept", pair,
            base::concatenate({pairName(pair), ": tied grid points lie off their faces, the largest by ",
                               std::to_string(tally.largestGap), " (grid point ", std::to_string(tally.largestGapPoint),
                               "); O_o_Solver never moves a grid point, so the gap is kept: a rigid rotation of the "
                               "bodies strains the glue by the rotation times the gap"})));
    if (report.tiedPointCount > 0 && report.tiedPointCount <= 2)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glued-pair-hinge", pair,
            base::concatenate({pairName(pair), " in subcase ", std::to_string(subcase), " ties ",
                               std::to_string(report.tiedPointCount),
                               " grid points by their translations only: the bodies can turn about the line through "
                               "them (a mechanism the grid point singularity processor reports, never hidden by a "
                               "stiffness)"})));
    if (report.tiedPointCount == 0 && report.candidatePointCount == 0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glued-pair-ties-nothing", pair,
            base::concatenate({pairName(pair), " in subcase ", std::to_string(subcase),
                               ": contactNodePairs gives no grid point of the secondary surface near the primary "
                               "surface; nothing is tied (the tie goes from the secondary body to the primary one "
                               "only): check the bodies and their order in the table"})));
    std::string parameters = base::concatenate({glueOptionLabel, " ", std::to_string(pair.glueOption)});
    if (!pair.glueOptionSource.empty())
        parameters.append(" (").append(pair.glueOptionSource).append(")");
    if (pair.glueSeparation != 0)
        parameters.append(", ")
            .append(std::string("GlueSeparationOption"))
            .append(" ")
            .append(std::to_string(pair.glueSeparation))
            .append(" (permanent)");
    const std::string tolerance =
        pair.errorTolerance && *pair.errorTolerance > 0.0
            ? base::concatenate(
                  {std::string("ContactDistanceTolerance"), " ", std::to_string(report.positionTolerance)})
            : base::concatenate({std::to_string(report.positionTolerance), ", 2.5 % of the typical element size ",
                                 std::to_string(report.typicalElementSize)});
    sink.report(pairDiagnostic(
        base::Severity::Information, "glued-pair", pair,
        base::concatenate({pairName(pair),
                           " in subcase ",
                           std::to_string(subcase),
                           ", ",
                           parameters,
                           ", position tolerance ",
                           tolerance,
                           ": ",
                           std::to_string(report.tiedPointCount),
                           " grid points tied by their translations to ",
                           std::to_string(report.faceCornerCount),
                           " face corners (mesh ratio ",
                           std::to_string(report.faceCornerCount > 0 ? static_cast<double>(report.tiedPointCount) /
                                                                           static_cast<double>(report.faceCornerCount)
                                                                     : 0.0),
                           "), ",
                           std::to_string(report.outsidePointCount),
                           " outside the tolerance, ",
                           std::to_string(report.leftOutPointCount),
                           " left out, of ",
                           std::to_string(report.candidatePointCount),
                           " candidate points and ",
                           std::to_string(report.primaryFaceCount),
                           " primary faces"})));
}

} // namespace

GlueConstraints::GlueConstraints(const input::ModelSnapshot& snapshot, std::optional<base::EntityId> contactId,
                                 base::EntityId subcase, const DofMap& dofMap, const GlueContext& context,
                                 base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    for (GluedPair& pair : selectGluedPairs(snapshot, contactId, subcase, sink))
    {
        GluedPairReport report;
        report.pair = std::move(pair);
        m_pairs.push_back(std::move(report));
    }
    if (m_pairs.empty())
        return;
    const input::ContactTable& contact = snapshot.contact();
    std::vector<PairTally> tallies(m_pairs.size());

    // 1. The surfaces of each pair from contactNodePairs: the distinct secondary grid points of the rows searched
    //    from the secondary body, the distinct faces of those rows (GlueConstraints.hpp, "the surfaces").
    std::map<std::tuple<std::string_view, std::int32_t, std::int32_t>, std::int32_t> pairOf;
    for (std::size_t p = 0; p < m_pairs.size(); ++p)
        pairOf.emplace(std::make_tuple(std::string_view(m_pairs[p].pair.table), m_pairs[p].pair.secondaryBody,
                                       m_pairs[p].pair.primaryBody),
                       static_cast<std::int32_t>(p));
    const auto rowCount = static_cast<std::size_t>(contact.nodePairCount());
    base::BudgetedBuffer<PointKey> points(budget, rowCount, "glue: candidate points");
    base::BudgetedBuffer<FaceKey> faceKeys(budget, rowCount, "glue: candidate faces");
    std::size_t pointCount = 0;
    std::size_t faceKeyCount = 0;
    for (std::int64_t row = 0; row < contact.nodePairCount(); ++row)
    {
        const input::ContactNodePair node = contact.nodePair(row);
        if (node.isSearchedFromPrimary)
            continue; // the search the other way: the tie is from the secondary body only
        const auto found = pairOf.find(std::make_tuple(node.table, node.secondaryBody, node.primaryBody));
        if (found == pairOf.end())
            continue; // a pair the subcase does not glue
        points[pointCount++] = {found->second, node.nodeId};
        if (node.face.empty())
            continue;
        FaceKey key;
        key.pairIndex = found->second;
        key.pointCount = static_cast<std::int32_t>(node.face.size()); // more than 8: refused by glueFaceOf
        std::copy_n(node.face.begin(), std::min(node.face.size(), MAX_GLUE_FACE_POINTS), key.gridIds.begin());
        faceKeys[faceKeyCount++] = key;
    }
    std::sort(points.data(), points.data() + pointCount, [](const PointKey& a, const PointKey& b)
              { return a.pairIndex != b.pairIndex ? a.pairIndex < b.pairIndex : a.nodeId < b.nodeId; });
    pointCount = static_cast<std::size_t>(std::unique(points.data(), points.data() + pointCount,
                                                      [](const PointKey& a, const PointKey& b)
                                                      { return a.pairIndex == b.pairIndex && a.nodeId == b.nodeId; }) -
                                          points.data());
    std::sort(faceKeys.data(), faceKeys.data() + faceKeyCount,
              [](const FaceKey& a, const FaceKey& b) { return a.tie() < b.tie(); });
    faceKeyCount =
        static_cast<std::size_t>(std::unique(faceKeys.data(), faceKeys.data() + faceKeyCount,
                                             [](const FaceKey& a, const FaceKey& b) { return a.tie() == b.tie(); }) -
                                 faceKeys.data());
    // The faces that can interpolate, in pair order; the others reported.
    base::BudgetedBuffer<GlueFace> faces(budget, faceKeyCount, "glue: primary faces");
    std::vector<std::size_t> faceBegin(m_pairs.size() + 1, 0); // bounded by the pairs of one contact table
    std::size_t faceCount = 0;
    for (std::size_t k = 0; k < faceKeyCount; ++k)
    {
        const FaceKey& key = faceKeys[k];
        PairTally& tally = tallies[static_cast<std::size_t>(key.pairIndex)];
        std::variant<GlueFace, std::string> face = glueFaceOf(key, dofMap);
        if (auto* problem = std::get_if<std::string>(&face))
        {
            if (tally.invalidFaceCount++ == 0)
                tally.firstInvalidFace = std::move(*problem);
            continue;
        }
        faces[faceCount++] = std::get<GlueFace>(face);
        faceBegin[static_cast<std::size_t>(key.pairIndex) + 1] = faceCount;
    }
    for (std::size_t p = 1; p <= m_pairs.size(); ++p)
        faceBegin[p] = std::max(faceBegin[p], faceBegin[p - 1]);

    // 2. The search and the tie of every candidate point, pair by pair, points by increasing ID.
    const base::BudgetedBuffer<std::uint8_t> states = constraintStates(context, dofMap, budget);
    const auto gridCount = static_cast<std::size_t>(dofMap.gridCount());
    base::BudgetedBuffer<std::int32_t> glueOwner(budget, gridCount, "glue: record of each grid point");
    for (std::size_t g = 0; g < gridCount; ++g)
        glueOwner[g] = -1;
    base::BudgetedBuffer<GlueRecord> records(budget, pointCount, "glue: tied points");
    std::size_t recordCount = 0;
    base::BudgetedBuffer<std::int64_t> visited(budget, gridCount, "glue: loop search");
    base::BudgetedBuffer<std::int32_t> stack(budget, gridCount, "glue: loop search");
    std::int64_t stamp = 0;

    // Whether the equation of `pointRow` on these face rows closes a loop: a face point that depends, through the glue
    // rows accepted so far, on the point itself. Each grid point is pushed once per search (marked when pushed).
    const auto closesLoop = [&](std::int32_t pointRow, const GlueRecord& candidate)
    {
        ++stamp;
        std::size_t top = 0;
        const auto push = [&](std::int32_t row)
        {
            if (row < 0 || visited[static_cast<std::size_t>(row)] == stamp)
                return;
            visited[static_cast<std::size_t>(row)] = stamp;
            stack[top++] = row;
        };
        for (std::int32_t k = 0; k < candidate.facePointCount; ++k)
            push(candidate.faceRows[static_cast<std::size_t>(k)]);
        while (top > 0)
        {
            const std::int32_t row = stack[--top];
            if (row == pointRow)
                return true;
            const std::int32_t owner = glueOwner[static_cast<std::size_t>(row)];
            if (owner < 0)
                continue;
            const GlueRecord& record = records[static_cast<std::size_t>(owner)];
            for (std::int32_t k = 0; k < record.facePointCount; ++k)
                push(record.faceRows[static_cast<std::size_t>(k)]);
        }
        return false;
    };

    std::size_t nextPoint = 0;
    for (std::size_t p = 0; p < m_pairs.size(); ++p)
    {
        GluedPairReport& report = m_pairs[p];
        PairTally& tally = tallies[p];
        const std::span<const GlueFace> pairFaces(faces.data() + faceBegin[p], faceBegin[p + 1] - faceBegin[p]);
        report.primaryFaceCount = static_cast<std::int64_t>(pairFaces.size());
        report.typicalElementSize = typicalElementSize(pairFaces);
        report.inPlaneTolerance = DEFAULT_TOLERANCE_FRACTION * report.typicalElementSize;
        report.positionTolerance = report.pair.errorTolerance && *report.pair.errorTolerance > 0.0
                                       ? *report.pair.errorTolerance
                                       : report.inPlaneTolerance;
        const GlueTolerances tolerances{report.positionTolerance, report.inPlaneTolerance};
        const GlueSearch search(pairFaces, tolerances.position + tolerances.inPlane, budget);
        // The grid points of the primary faces, sorted: a candidate among them is shared by the bodies.
        std::size_t primaryGridCount = 0;
        for (const GlueFace& face : pairFaces)
            primaryGridCount += static_cast<std::size_t>(face.pointCount);
        base::BudgetedBuffer<std::int32_t> primaryGrids(budget, primaryGridCount, "glue: primary grid points");
        std::size_t filled = 0;
        for (const GlueFace& face : pairFaces)
            for (const std::int32_t nodeId : face.grids())
                primaryGrids[filled++] = nodeId;
        std::sort(primaryGrids.data(), primaryGrids.data() + filled);
        const auto isPrimaryGrid = [&](std::int32_t nodeId)
        { return std::binary_search(primaryGrids.data(), primaryGrids.data() + filled, nodeId); };

        for (; nextPoint < pointCount && points[nextPoint].pairIndex == static_cast<std::int32_t>(p); ++nextPoint)
        {
            const std::int32_t nodeId = points[nextPoint].nodeId;
            ++report.candidatePointCount;
            const std::optional<std::int32_t> pointRow = dofMap.points().findGrid(nodeId);
            if (!pointRow)
            {
                if (tally.unknownPointCount++ == 0)
                    tally.firstUnknownPoint = nodeId;
                continue;
            }
            if (isPrimaryGrid(nodeId))
                continue; // the bodies share the point: nothing to tie
            const base::Vec3 position = dofMap.positionInBasic(*pointRow);
            const GlueProjection projection = search.project(position, tolerances);
            if (projection.face < 0)
            {
                if (report.outsidePointCount++ == 0)
                {
                    tally.firstOutside = nodeId;
                    tally.firstOutsideDistance = projection.nearestDistance;
                }
                continue;
            }
            const GlueFace& face = pairFaces[static_cast<std::size_t>(projection.face)];
            GlueRecord candidate;
            candidate.pairIndex = static_cast<std::int32_t>(p);
            candidate.pointRow = *pointRow;
            candidate.facePointCount = face.pointCount;
            for (std::size_t k = 0; k < static_cast<std::size_t>(face.pointCount); ++k)
            {
                candidate.shapeValues[k] = projection.shape.values[k];
                candidate.faceRows[k] = face.gridIds[k] == 0 ? -1 : *dofMap.points().findGrid(face.gridIds[k]);
            }
            const double gap = base::norm(position - face.pointAt(projection.coordinates));
            if (gap > GLUE_GAP_TOLERANCE * report.typicalElementSize && gap > tally.largestGap)
            {
                tally.largestGap = gap;
                tally.largestGapPoint = nodeId;
            }

            // The translations tied; a component another constraint already holds or makes dependent keeps that
            // constraint (one exclusive set per DOF).
            if (glueOwner[static_cast<std::size_t>(*pointRow)] >= 0)
            {
                tally.leaveOut(LeftOut::GluedTwice, nodeId);
                ++report.leftOutPointCount;
                continue;
            }
            std::uint8_t mask = 0;
            std::string takenText;
            for (int component = 1; component <= TRANSLATIONS; ++component)
            {
                const std::uint8_t state = states[static_cast<std::size_t>(dofMap.gridDof(*pointRow, component))];
                if (state == FREE)
                {
                    mask = static_cast<std::uint8_t>(mask | (1U << (component - 1)));
                    continue;
                }
                if (takenText.empty())
                    takenText =
                        base::concatenate({state == HELD ? "an SinglePointConstr or Node Node.permanentConstraints"
                                                         : "a rigid element, fastener or MultiPointConstr",
                                           " on component ", std::to_string(component)});
            }
            if ((mask & TRANSLATION_MASK) == 0)
            {
                if (tally.counts[static_cast<std::size_t>(LeftOut::Constrained)] == 0)
                    tally.firstConstraint = takenText;
                tally.leaveOut(LeftOut::Constrained, nodeId);
                ++report.leftOutPointCount;
                continue;
            }
            if (!takenText.empty() && tally.partlyTiedCount++ == 0)
            {
                tally.firstPartlyTied = nodeId;
                tally.firstPartConstraint = takenText;
            }
            candidate.componentMask = mask;
            if (closesLoop(*pointRow, candidate))
            {
                tally.leaveOut(LeftOut::Loop, nodeId);
                ++report.leftOutPointCount;
                continue;
            }
            records[recordCount] = candidate;
            glueOwner[static_cast<std::size_t>(*pointRow)] = static_cast<std::int32_t>(recordCount);
            ++recordCount;
            ++report.tiedPointCount;
        }
    }

    // The distinct face corners each pair's points depend on (the mesh ratio of the report): a mark per grid point,
    // the pair's index + 1 once counted. The records are in pair order.
    base::BudgetedBuffer<std::int32_t> counted(budget, gridCount, "glue: face corners counted");
    for (std::size_t r = 0; r < recordCount; ++r)
    {
        const GlueRecord& record = records[r];
        for (std::int32_t k = 0; k < record.facePointCount; ++k)
        {
            const std::int32_t faceRow = record.faceRows[static_cast<std::size_t>(k)];
            if (faceRow < 0 || counted[static_cast<std::size_t>(faceRow)] == record.pairIndex + 1)
                continue;
            counted[static_cast<std::size_t>(faceRow)] = record.pairIndex + 1;
            ++m_pairs[static_cast<std::size_t>(record.pairIndex)].faceCornerCount;
        }
    }
    for (std::size_t p = 0; p < m_pairs.size(); ++p)
        reportPair(m_pairs[p], tallies[p], subcase, sink);

    // 3. The rows, by record then component; a row of component c of the point:
    //     u_s,c = Σ_i Σ_k N_i·(R_sᵀ·R_i)(c, k)·u_i,k     (k over the translations).
    // The coefficients are computed twice (to size the buffers exactly, then to write them): cheap against the memory
    // of an upper bound (8 face points × 3 terms per row).
    const auto forEachTerm = [&](const GlueRecord& record, int component, auto visit)
    {
        const auto c = static_cast<std::size_t>(component - 1);
        const base::Mat3 toPoint = dofMap.cdToBasic(record.pointRow).transposed();
        for (std::int32_t i = 0; i < record.facePointCount; ++i)
        {
            const std::int32_t faceRow = record.faceRows[static_cast<std::size_t>(i)];
            const double shapeValue = record.shapeValues[static_cast<std::size_t>(i)];
            if (faceRow < 0 || shapeValue == 0.0)
                continue;
            const base::Mat3 transfer = toPoint * dofMap.cdToBasic(faceRow);
            for (std::size_t k = 0; k < 3; ++k)
            {
                const double coefficient = shapeValue * transfer(c, k);
                if (coefficient != 0.0)
                    visit(dofMap.gridDof(faceRow, static_cast<int>(k) + 1), coefficient);
            }
        }
    };
    const auto isTied = [](const GlueRecord& record, int component)
    { return (record.componentMask & (1U << (component - 1))) != 0; };
    std::size_t rowCapacity = 0;
    std::int64_t termCapacity = 0;
    for (std::size_t r = 0; r < recordCount; ++r)
        for (int component = 1; component <= TRANSLATIONS; ++component)
        {
            if (!isTied(records[r], component))
                continue;
            ++rowCapacity;
            forEachTerm(records[r], component, [&](base::DofIndex, double) { ++termCapacity; });
        }
    m_dependents = base::BudgetedBuffer<base::DofIndex>(budget, rowCapacity, "glue constraint rows");
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, rowCapacity + 1, "glue constraint rows");
    m_terms =
        base::BudgetedBuffer<ConstraintTerm>(budget, static_cast<std::size_t>(termCapacity), "glue constraint terms");
    m_sources = base::BudgetedBuffer<std::uint8_t>(budget, rowCapacity, "glue constraint rows");
    m_sourceIds = base::BudgetedBuffer<base::EntityId>(budget, rowCapacity, "glue constraint rows");
    m_offsets[0] = 0;
    for (std::size_t r = 0; r < recordCount; ++r)
    {
        const GlueRecord& record = records[r];
        const GluedPair& pair = m_pairs[static_cast<std::size_t>(record.pairIndex)].pair;
        for (int component = 1; component <= TRANSLATIONS; ++component)
        {
            if (!isTied(record, component))
                continue;
            m_dependents[m_rowCount] = dofMap.gridDof(record.pointRow, component);
            m_sources[m_rowCount] = static_cast<std::uint8_t>(sourceOfCard(pair.card));
            m_sourceIds[m_rowCount] = pair.tableId;
            forEachTerm(record, component, [&](base::DofIndex dof, double coefficient)
                        { m_terms[static_cast<std::size_t>(m_termCount++)] = {dof, coefficient}; });
            ++m_rowCount;
            m_offsets[m_rowCount] = m_termCount;
        }
    }
}

} // namespace oos::assembly
