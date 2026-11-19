#include "oos/assembly/constraints/GlueConstraints.hpp"

#include "oos/assembly/constraints/FaceShapeFunctions.hpp"
#include "oos/assembly/constraints/GlueSearch.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/input/tables/PiercingFootprint.hpp"

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
constexpr int COMPONENTS = 6;
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
    // The face is that of a shell of the ClassicFormulation: the point follows the normal fibre of the shell at the
    // tied point (GlueConstraints.hpp, "the tie"), with:
    std::uint8_t readsShell = 0;
    base::Vec3 normal; // n, the unit normal of the face at the tied point
    base::Vec3 arm;    // from the tied point of the face to the secondary point
    // w_i: the rotation of the sheet in its plane at the tied point, ω = Σ w_i·u_i (u_i in the basic system)
    std::array<base::Vec3, MAX_GLUE_FACE_POINTS> turn{};
};

// One face of a row of contactNodePairs, keyed by its pair (sorted, then made distinct).
struct FaceKey
{
    std::int32_t pairIndex = 0;
    std::int32_t pointCount = 0;
    std::array<std::int32_t, MAX_GLUE_FACE_POINTS> gridIds{};
    std::int32_t elementId = 0; // the element the face is of (0: not given); not part of the key

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
    std::int64_t gapFollowedCount = 0; // tied points off their faces that follow the rotation of a shell

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

// The IDs of the shell elements of the model, sorted (4 bytes per shell: budgeted).
base::BudgetedBuffer<std::int32_t> shellElementIds(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget)
{
    std::size_t count = 0;
    for (const std::string_view card : input::PIERCED_SHELL_CARDS)
        if (const input::ElementBatch* batch = snapshot.findElementBatch(card))
            count += batch->ids().size();
    base::BudgetedBuffer<std::int32_t> ids(budget, count, "glue: shell elements");
    std::size_t filled = 0;
    for (const std::string_view card : input::PIERCED_SHELL_CARDS)
        if (const input::ElementBatch* batch = snapshot.findElementBatch(card))
            for (const std::int32_t id : batch->ids())
                ids[filled++] = id;
    std::sort(ids.data(), ids.data() + filled);
    return ids;
}

// Fills what a record reads of a shell face at its tied point (GlueRecord): the normal n = x_,ξ × x_,η, the arm to
// the secondary point, and the vectors w_i of the rotation of the sheet in its plane,
//     ω = ½·n·(∇ × u) = Σ_i w_i·u_i,   w_i = ½·n × ∇N_i,   ∇N_i = N_i,ξ·a¹ + N_i,η·a²,
// a¹, a² the contravariant vectors of the tangents (a^α·x_,β = δ_αβ, in the tangent plane). False for a face whose
// tangents are parallel there (no normal).
bool readShellFace(GlueRecord& record, const GlueFace& face, FaceCoordinates at, base::Vec3 secondaryPoint)
{
    const FaceShapeDerivatives d = faceShapeDerivatives(face.grids(), at.xi, at.eta);
    base::Vec3 alongXi;
    base::Vec3 alongEta;
    for (std::size_t k = 0; k < static_cast<std::size_t>(face.pointCount); ++k)
    {
        alongXi = alongXi + d.dXi[k] * face.positions[k];
        alongEta = alongEta + d.dEta[k] * face.positions[k];
    }
    const base::Vec3 across = base::cross(alongXi, alongEta);
    const double area = base::norm(across);
    if (!(area > 0.0))
        return false;
    record.normal = across / area;
    record.arm = secondaryPoint - face.pointAt(at);
    // a¹ = (x_,η × n) / (x_,ξ·(x_,η × n)),   a² = (n × x_,ξ) / (x_,η·(n × x_,ξ)): both denominators are the area
    const base::Vec3 first = base::cross(alongEta, record.normal) / area;
    const base::Vec3 second = base::cross(record.normal, alongXi) / area;
    for (std::size_t k = 0; k < static_cast<std::size_t>(face.pointCount); ++k)
        record.turn[k] = 0.5 * base::cross(record.normal, d.dXi[k] * first + d.dEta[k] * second);
    record.readsShell = 1;
    return true;
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
    if (pair.asksForRotations() && report.tiedPointCount > report.rotationTiedPointCount)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-rotations-not-tied", pair,
            base::concatenate({pairName(pair), ": ", glueOptionLabel, " ", std::to_string(pair.glueOption),
                               " asks for a glue of the shell rotations too; ",
                               std::to_string(report.tiedPointCount - report.rotationTiedPointCount), " of the ",
                               std::to_string(report.tiedPointCount),
                               " tied grid points have their translations tied only: the rotations are tied to the "
                               "rotation of a shell (those of the grid point in the ClassicFormulation, that of the "
                               "edges between tied grid points in a formulation with unknowns along its edges), and "
                               "their face is not one (the face of a solid), no edge joins them to another tied grid "
                               "point, or their rotations are held or dependent already. There a moment passes "
                               "through tie forces at points apart, not at a single line of points"})));
    // A gap is followed where the face is a shell: only the points that do not follow it are reported.
    if (tally.largestGap > 0.0)
        sink.report(pairDiagnostic(
            base::Severity::Warning, "glue-gap-kept", pair,
            base::concatenate({pairName(pair), ": tied grid points lie off their faces, the largest by ",
                               std::to_string(tally.largestGap), " (grid point ", std::to_string(tally.largestGapPoint),
                               "), on faces that have no rotation (solids); O_o_Solver never moves a grid point, so "
                               "the gap is kept: a rigid rotation of the bodies strains the glue by the rotation "
                               "times the gap"})));
    if (report.tiedPointCount > 0 && report.tiedPointCount <= 2 && report.rotationTiedPointCount == 0)
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
                           " grid points tied by their translations (",
                           std::to_string(report.rotationTiedPointCount),
                           " by their rotations too, ",
                           std::to_string(tally.gapFollowedCount),
                           " off their faces following the rotation of a shell) to ",
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
                           " primary faces",
                           report.tiedEdgeCount + report.freeEdgeCount > 0
                               ? base::concatenate({"; ", std::to_string(report.tiedEdgeCount),
                                                    " edges of the glued surface tied along their length (",
                                                    std::to_string(report.rotationTiedEdgeCount),
                                                    " by their rotation too), ",
                                                    std::to_string(report.freeEdgeCount),
                                                    " left to their own unknowns (a point of theirs is off the "
                                                    "primary faces)"})
                               : std::string()})));
}

// The rows of the edges `traces` of the glued surface of a pair (those between two of its tied grid points), with an
// element formulation that has unknowns along its edges (GlueConstraints.hpp, "the edges of such a formulation"):
// appended to `dependents`, `offsets` and `terms`. Counts the edges in the report of the pair and marks in
// `turnsWithFace` the grid points at the ends of the edges tied by their rotation.
void tieEdges(GluedPairReport& report, std::span<const elements::FormulationModel::EdgeTrace* const> traces,
              std::span<const GlueFace> faces, const DofMap& dofMap, const GlueContext& context,
              std::span<const std::uint8_t> states, base::MemoryBudget& budget,
              std::vector<base::DofIndex>& dependents, std::vector<std::int64_t>& offsets,
              std::vector<ConstraintTerm>& terms, std::span<std::uint8_t> turnsWithFace)
{
    const GlueTolerances tolerances{report.positionTolerance, report.inPlaneTolerance};
    const GlueSearch search(faces, tolerances.position + tolerances.inPlane, budget);
    // A row is summed over the g set: the sums, and the DOFs it reaches.
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    base::BudgetedBuffer<double> sums(budget, gSize, "glue: row of a glued edge");
    base::BudgetedBuffer<std::uint8_t> isReached(budget, gSize, "glue: row of a glued edge");
    std::vector<base::DofIndex> reached; // bounded by one row
    const auto add = [&](const std::vector<elements::FormulationRelationTerm>& rowTerms, double factor)
    {
        if (factor == 0.0)
            return;
        for (const elements::FormulationRelationTerm& term : rowTerms)
        {
            const auto dof = static_cast<std::size_t>(dofMap.dofOf(term.dof));
            if (isReached[dof] == 0)
            {
                isReached[dof] = 1;
                reached.push_back(static_cast<base::DofIndex>(dof));
            }
            sums[dof] += factor * term.coefficient;
        }
    };

    std::vector<elements::FormulationModel::PointMotion> motions; // of the points of one edge
    std::vector<base::Mat3> carries;                              // u(P) + carry·θ(P) at each of them
    for (const elements::FormulationModel::EdgeTrace* const edge : traces)
    {
        const elements::FormulationModel::EdgeTrace& trace = *edge;
        // The motion of the face under each point of the edge; an edge that leaves the primary faces, or lies on a
        // face no element of the formulation has, keeps its own unknowns.
        motions.clear();
        carries.clear();
        bool hasRotation = true;
        for (const base::Vec3& point : trace.points)
        {
            const GlueProjection projection = search.project(point, tolerances);
            if (projection.face < 0)
                break;
            const GlueFace& face = faces[static_cast<std::size_t>(projection.face)];
            const auto count = static_cast<std::size_t>(face.pointCount);
            std::array<std::int32_t, MAX_GLUE_FACE_POINTS> faceRows{};
            for (std::size_t k = 0; k < count; ++k)
                faceRows[k] = face.gridIds[k] == 0 ? -1 : *dofMap.points().findGrid(face.gridIds[k]);
            std::optional<elements::FormulationModel::PointMotion> motion =
                context.formulation->facePointMotion({faceRows.data(), count}, {projection.shape.values.data(), count});
            if (!motion)
                break;
            hasRotation = hasRotation && motion->point.has_value();
            carries.push_back(motion->point ? (-1.0) * elements::crossProductMatrix(point - *motion->point)
                                            : base::Mat3{});
            motions.push_back(std::move(*motion));
        }
        if (motions.size() != trace.points.size())
        {
            ++report.freeEdgeCount;
            continue;
        }
        const bool tiesRotation = report.pair.asksForRotations() && hasRotation;
        bool isTied = false;
        bool isTurned = false;
        for (const elements::FormulationModel::EdgeMoment& moment : trace.moments)
        {
            if (moment.isRotation && !tiesRotation)
                continue;
            const base::DofIndex dependent = dofMap.dofOf(moment.dof);
            if (states[static_cast<std::size_t>(dependent)] != FREE)
                continue; // held by a support or dependent already: it keeps that constraint
            for (std::size_t g = 0; g < motions.size(); ++g)
            {
                const base::Vec3& weight = moment.weights[g];
                if (moment.isRotation)
                {
                    for (std::size_t b = 0; b < 3; ++b)
                        add(motions[g].rows[3 + b], weight[b]);
                    continue;
                }
                // w·(u + carry·θ) = w·u + (carryᵀ·w)·θ
                for (std::size_t b = 0; b < 3; ++b)
                    add(motions[g].rows[b], weight[b]);
                for (std::size_t e = 0; e < 3; ++e)
                    add(motions[g].rows[3 + e],
                        weight[0] * carries[g](0, e) + weight[1] * carries[g](1, e) + weight[2] * carries[g](2, e));
            }
            std::sort(reached.begin(), reached.end());
            for (const base::DofIndex dof : reached)
            {
                const auto at = static_cast<std::size_t>(dof);
                if (sums[at] != 0.0)
                    terms.push_back({dof, sums[at]});
                sums[at] = 0.0;
                isReached[at] = 0;
            }
            reached.clear();
            dependents.push_back(dependent);
            offsets.push_back(static_cast<std::int64_t>(terms.size()));
            isTied = true;
            isTurned = isTurned || moment.isRotation;
        }
        if (isTied)
            ++report.tiedEdgeCount;
        if (isTurned)
        {
            ++report.rotationTiedEdgeCount;
            for (const std::int32_t gridRow : trace.gridRows)
                turnsWithFace[static_cast<std::size_t>(gridRow)] = 1;
        }
    }
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
        key.elementId = node.primaryElementId.value_or(0);
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
    // The faces that can interpolate, in pair order; the others reported. A face of a shell of the ClassicFormulation
    // is marked: a point tied to it follows the rotation of the shell (an element of a formulation with entities
    // gives its own motion, FormulationModel::facePointMotion).
    const base::BudgetedBuffer<std::int32_t> shellIds = shellElementIds(snapshot, budget);
    const auto isClassicShell = [&](std::int32_t elementId)
    {
        return context.formulation == nullptr && elementId != 0 &&
               std::binary_search(shellIds.data(), shellIds.data() + shellIds.size(), elementId);
    };
    base::BudgetedBuffer<std::uint8_t> faceIsShell(budget, faceKeyCount, "glue: primary faces");
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
        faceIsShell[faceCount] = isClassicShell(key.elementId) ? 1 : 0;
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
    // The pair (its index + 1; 0: none) of each grid point of a glued surface that lies on a face: the points tied,
    // and those left to a support or to another relation that holds every translation of theirs. The edges between
    // two of them are those of the glued surface (2b).
    base::BudgetedBuffer<std::int32_t> surfaceOf(budget, gridCount, "glue: points of the glued surfaces");
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
            // A shell face: the point follows the normal fibre of the shell, gap included. An element of a
            // formulation with entities that gives its rotation there does too (forEachTerm).
            const bool followsRotation =
                (faceIsShell[faceBegin[p] + static_cast<std::size_t>(projection.face)] != 0 &&
                 readShellFace(candidate, face, projection.coordinates, position)) ||
                (context.formulation != nullptr &&
                 [&]
                 {
                     const auto count = static_cast<std::size_t>(candidate.facePointCount);
                     const auto motion = context.formulation->facePointMotion(
                         {candidate.faceRows.data(), count}, {candidate.shapeValues.data(), count});
                     return motion && motion->point.has_value();
                 }());
            const double gap = base::norm(position - face.pointAt(projection.coordinates));
            if (gap > GLUE_GAP_TOLERANCE * report.typicalElementSize)
            {
                if (followsRotation)
                    ++tally.gapFollowedCount;
                else if (gap > tally.largestGap)
                {
                    tally.largestGap = gap;
                    tally.largestGapPoint = nodeId;
                }
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
                surfaceOf[static_cast<std::size_t>(*pointRow)] = static_cast<std::int32_t>(p) + 1;
                continue;
            }
            if (!takenText.empty() && tally.partlyTiedCount++ == 0)
            {
                tally.firstPartlyTied = nodeId;
                tally.firstPartConstraint = takenText;
            }
            // The rotations too when the pair asks for them and the face is a shell (those that are free).
            bool hasRotationTied = false;
            if (report.pair.asksForRotations() && candidate.readsShell != 0)
                for (int component = TRANSLATIONS + 1; component <= COMPONENTS; ++component)
                    if (states[static_cast<std::size_t>(dofMap.gridDof(*pointRow, component))] == FREE)
                    {
                        mask = static_cast<std::uint8_t>(mask | (1U << (component - 1)));
                        hasRotationTied = true;
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
            surfaceOf[static_cast<std::size_t>(*pointRow)] = static_cast<std::int32_t>(p) + 1;
            ++recordCount;
            ++report.tiedPointCount;
            if (hasRotationTied)
                ++report.rotationTiedPointCount;
        }
    }

    // 2b. The edges of the glued surface between two tied grid points, with an element formulation that has unknowns
    //     along its edges (GlueConstraints.hpp, "the edges of such a formulation"). Their rows are gathered in
    //     vectors (their number is not known before the faces are searched) and copied to budgeted buffers.
    if (context.formulation != nullptr && recordCount > 0)
    {
        std::vector<std::int32_t> surfaceRows; // increasing; bounded by the points of the glued surfaces
        for (std::size_t g = 0; g < gridCount; ++g)
            if (surfaceOf[g] != 0)
                surfaceRows.push_back(static_cast<std::int32_t>(g));
        std::vector<elements::FormulationModel::EdgeTrace> traces; // as the formulation gives them
        context.formulation->edgeTraces(surfaceRows, traces);
        // The unknowns of these edges, and the edges of each pair: those between two grid points of its glued
        // surface (an edge between the points of two pairs keeps its own unknowns).
        std::vector<base::DofIndex> edgeDofs;
        std::vector<std::vector<const elements::FormulationModel::EdgeTrace*>> tracesOf(m_pairs.size());
        for (const elements::FormulationModel::EdgeTrace& trace : traces)
        {
            for (const elements::FormulationModel::EdgeMoment& moment : trace.moments)
                edgeDofs.push_back(dofMap.dofOf(moment.dof));
            const std::int32_t first = surfaceOf[static_cast<std::size_t>(trace.gridRows[0])];
            const std::int32_t second = surfaceOf[static_cast<std::size_t>(trace.gridRows[1])];
            if (first == second)
                tracesOf[static_cast<std::size_t>(first - 1)].push_back(&trace);
        }
        std::sort(edgeDofs.begin(), edgeDofs.end());
        edgeDofs.erase(std::unique(edgeDofs.begin(), edgeDofs.end()), edgeDofs.end());
        m_edgeDofs = base::BudgetedBuffer<base::DofIndex>(budget, edgeDofs.size(), "glue edge unknowns");
        std::copy(edgeDofs.begin(), edgeDofs.end(), m_edgeDofs.data());

        std::vector<base::DofIndex> edgeDependents;
        std::vector<std::int64_t> edgeOffsets{0};
        std::vector<ConstraintTerm> edgeTerms;
        base::BudgetedBuffer<std::uint8_t> turnsWithFace(budget, gridCount, "glue: points of the turned edges");
        for (std::size_t p = 0; p < m_pairs.size(); ++p)
        {
            if (tracesOf[p].empty())
                continue;
            const std::span<const GlueFace> pairFaces(faces.data() + faceBegin[p], faceBegin[p + 1] - faceBegin[p]);
            tieEdges(m_pairs[p], tracesOf[p], pairFaces, dofMap, context, states.span(), budget, edgeDependents,
                     edgeOffsets, edgeTerms, turnsWithFace.span());
        }
        // the tied grid points whose rotation the edges carry
        for (std::size_t r = 0; r < recordCount; ++r)
            if (turnsWithFace[static_cast<std::size_t>(records[r].pointRow)] != 0)
                ++m_pairs[static_cast<std::size_t>(records[r].pairIndex)].rotationTiedPointCount;
        m_edgeRowCount = edgeDependents.size();
        m_edgeDependents = base::BudgetedBuffer<base::DofIndex>(budget, edgeDependents.size(), "glue edge rows");
        m_edgeOffsets = base::BudgetedBuffer<std::int64_t>(budget, edgeOffsets.size(), "glue edge rows");
        m_edgeTerms = base::BudgetedBuffer<ConstraintTerm>(budget, edgeTerms.size(), "glue edge terms");
        std::copy(edgeDependents.begin(), edgeDependents.end(), m_edgeDependents.data());
        std::copy(edgeOffsets.begin(), edgeOffsets.end(), m_edgeOffsets.data());
        std::copy(edgeTerms.begin(), edgeTerms.end(), m_edgeTerms.data());
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

    // 3. The rows, by record then component (GlueConstraints.hpp, "the tie"). With (u_P, θ_P) the motion of the face
    //    at the tied point over the DOFs it reads, in the basic system, and d the arm to the secondary point,
    //        u_s = R_sᵀ·(u_P + θ_P × d),   θ_s = R_sᵀ·θ_P;
    //    a face without rotation gives u_s = R_sᵀ·u_P. The coefficients are computed twice (to size the buffers
    //    exactly, then to write them): cheap against the memory of an upper bound.
    const auto faceMotionOf = [&](const GlueRecord& record) -> std::optional<elements::FormulationModel::PointMotion>
    {
        if (context.formulation == nullptr)
            return std::nullopt;
        const auto count = static_cast<std::size_t>(record.facePointCount);
        return context.formulation->facePointMotion({record.faceRows.data(), count},
                                                    {record.shapeValues.data(), count});
    };
    const auto forEachTerm = [&](const GlueRecord& record,
                                 const std::optional<elements::FormulationModel::PointMotion>& motion, int component,
                                 auto visit)
    {
        const bool isRotation = component > TRANSLATIONS;
        const auto c = static_cast<std::size_t>(isRotation ? component - TRANSLATIONS - 1 : component - 1);
        const base::Mat3 toPoint = dofMap.cdToBasic(record.pointRow).transposed();
        if (motion)
        {
            // An element of the formulation: its rows of u_P, and of θ_P × d = −[d]×·θ_P when it gives its rotation.
            std::map<base::DofIndex, double> sums; // one term per DOF
            const auto add = [&](const std::vector<elements::FormulationRelationTerm>& terms, double factor)
            {
                if (factor != 0.0)
                    for (const elements::FormulationRelationTerm& term : terms)
                        sums[dofMap.dofOf(term.dof)] += factor * term.coefficient;
            };
            base::Mat3 carry; // u_P + carry·θ_P
            if (motion->point)
                carry = (-1.0) * elements::crossProductMatrix(dofMap.positionInBasic(record.pointRow) - *motion->point);
            for (std::size_t b = 0; b < 3; ++b)
            {
                add(motion->rows[b], toPoint(c, b));
                for (std::size_t e = 0; e < 3; ++e)
                    add(motion->rows[3 + e], toPoint(c, b) * carry(b, e));
            }
            for (const auto& [dof, coefficient] : sums)
                if (coefficient != 0.0)
                    visit(dof, coefficient);
            return;
        }
        // A face of grid points. A shell (readsShell) has the rotation
        //     θ_P = Σ_i [ N_i·(I − n·nᵀ)·R_i·θ_i + n·(w_i·R_i·u_i) ]:
        // that of its grid points about the axes of its plane, that of the sheet in its plane about its normal.
        base::Mat3 inPlane = base::Mat3::identity();
        base::Mat3 carry;
        if (record.readsShell != 0)
        {
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t k = 0; k < 3; ++k)
                    inPlane(r, k) -= record.normal[r] * record.normal[k];
            carry = (-1.0) * elements::crossProductMatrix(record.arm);
        }
        for (std::int32_t i = 0; i < record.facePointCount; ++i)
        {
            const std::int32_t faceRow = record.faceRows[static_cast<std::size_t>(i)];
            const double shapeValue = record.shapeValues[static_cast<std::size_t>(i)];
            if (faceRow < 0)
                continue;
            const base::Mat3 toBasic = dofMap.cdToBasic(faceRow);
            // The blocks of the row on the translations and on the rotations of the face point, in the basic system.
            base::Mat3 onTranslations;
            base::Mat3 onRotations;
            if (record.readsShell != 0)
            {
                base::Mat3 aboutNormal; // n·w_iᵀ: the rotation in the plane, from the translations
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t k = 0; k < 3; ++k)
                        aboutNormal(r, k) = record.normal[r] * record.turn[static_cast<std::size_t>(i)][k];
                const base::Mat3 aboutPlane = shapeValue * inPlane;
                onTranslations = isRotation ? aboutNormal : shapeValue * base::Mat3::identity() + carry * aboutNormal;
                onRotations = isRotation ? aboutPlane : carry * aboutPlane;
            }
            else
                onTranslations = shapeValue * base::Mat3::identity();
            const base::Mat3 translationTransfer = toPoint * onTranslations * toBasic;
            const base::Mat3 rotationTransfer = toPoint * onRotations * toBasic;
            for (std::size_t k = 0; k < 3; ++k)
            {
                if (translationTransfer(c, k) != 0.0)
                    visit(dofMap.gridDof(faceRow, static_cast<int>(k) + 1), translationTransfer(c, k));
                if (record.readsShell != 0 && rotationTransfer(c, k) != 0.0)
                    visit(dofMap.gridDof(faceRow, static_cast<int>(k) + 4), rotationTransfer(c, k));
            }
        }
    };
    const auto isTied = [](const GlueRecord& record, int component)
    { return (record.componentMask & (1U << (component - 1))) != 0; };
    std::size_t rowCapacity = 0;
    std::int64_t termCapacity = 0;
    for (std::size_t r = 0; r < recordCount; ++r)
    {
        const std::optional<elements::FormulationModel::PointMotion> motion = faceMotionOf(records[r]);
        for (int component = 1; component <= COMPONENTS; ++component)
        {
            if (!isTied(records[r], component))
                continue;
            ++rowCapacity;
            forEachTerm(records[r], motion, component, [&](base::DofIndex, double) { ++termCapacity; });
        }
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
        const std::optional<elements::FormulationModel::PointMotion> motion = faceMotionOf(record);
        for (int component = 1; component <= COMPONENTS; ++component)
        {
            if (!isTied(record, component))
                continue;
            m_dependents[m_rowCount] = dofMap.gridDof(record.pointRow, component);
            m_sources[m_rowCount] = static_cast<std::uint8_t>(sourceOfCard(pair.card));
            m_sourceIds[m_rowCount] = pair.tableId;
            forEachTerm(record, motion, component, [&](base::DofIndex dof, double coefficient)
                        { m_terms[static_cast<std::size_t>(m_termCount++)] = {dof, coefficient}; });
            ++m_rowCount;
            m_offsets[m_rowCount] = m_termCount;
        }
    }
}

} // namespace oos::assembly
