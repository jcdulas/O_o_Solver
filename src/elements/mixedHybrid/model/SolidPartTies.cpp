#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPartProjections.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

// The ties of the solids with the shells and the lines alone (spec section 3.4, ties 1 to 4; document 34 of the
// reference, section 8.2; SolidPart.hpp): what each edge and face of the solids follows, the geometry it takes from
// it, and the rows of the ties.
//
// Every tie is a projection that a rigid motion and a uniform strain pass through exactly: an affine field u = b + A x
// gives the shell the displacement b + A X(ξ, η) of its own surface, and the solid edge or face that follows the shell
// has the geometry of that same surface projected the same way, so the row of each function gives A times its
// geometry coefficient, the value SolidPart::affineField gives it. One exception: the rotation about an edge read
// over the thickness of a shell (fibreRotation) passes a rigid motion exactly and reads, in a uniform strain, the
// rotation of the fibre of the shell on the skin, which is the mean rotation of the strain (the value of
// ShellPart::affineField) plus its shear between the fibre and the conormal: a shell follows the surface it stands
// on.
//
// Thread-safety: findTies and followTies build the part; ties() is a pure function of it.

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t COMPONENTS = 3;
constexpr std::size_t QUADRILATERAL = 4;
constexpr std::size_t TRIANGLE = 3;
// The halves of a quadrilateral shell (three of its corners) that may cover a triangle of the solids.
constexpr std::array<std::array<std::size_t, 3>, 4> QUADRILATERAL_HALVES{{{0, 1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 3}}};
// The points of the rule of the projection of the functions of a line alone on those of the solid edge: k + 4, the
// products of two polynomials of degree k integrated exactly (the reference's edge_rule(k + 4)).
constexpr std::size_t ALONE_PROJECTION_EXTRA_POINTS = 4;
// The points of the rule of the rotation unknowns of an edge (the reference's edge_rule(12)): for a rigid rotation ω,
// W² |X_s| ω · t P_l = ω · (P_s W − P W_s) P_l (X = P / W along the edge) is a polynomial of degree at most 2k along a
// conic edge, integrated exactly.
constexpr std::size_t ROTATION_RULE_POINTS = 12;
// The displacement functions of an edge alone: 1 − s and s at its grids come first (AloneEdge::functions).
constexpr std::size_t ALONE_GRID_FUNCTIONS = 2;
// The points of the rule on each piece of the fibre of a shell between two sides of the faces of the skin it crosses
// (readFibre): the displacement of the skin along the fibre is a polynomial of the trace degree on each face, and its
// moment about the middle of the fibre one degree more.
constexpr std::size_t FIBRE_PIECE_POINTS = 5;
// The fibre crosses a side of a face when it passes within this fraction of the thickness of its chord (a side that
// is no crossing only cuts a piece in two).
constexpr double FIBRE_CROSSING_TOLERANCE = 1.0e-2;
// A piece of the fibre shorter than this fraction of the thickness is left out.
constexpr double FIBRE_MIN_PIECE = 1.0e-9;
// A value of a function of a solid below this is not written in a row: the functions of the other faces vanish on a
// face to round-off.
constexpr double FUNCTION_ROUND_OFF = 1.0e-13;
// A point of the fibre is read on a face of the skin when it lies on it within this fraction of the thickness.
constexpr double FIBRE_ON_SKIN_TOLERANCE = 1.0e-6;
// The points read must span at least this fraction of the thickness (their variance against that of the whole
// fibre), or the slope of the displacement along the fibre is not read.
constexpr double FIBRE_MIN_SPAN = 0.25;
// The closest point of a face: the steps of its Gauss-Newton iteration, the step of the differences that give the
// tangents of the face, the step below which it has converged, the margin of the face coordinates, and the margin
// within which the coordinates on the face of straight sides must be for the point to be sought on the face itself.
constexpr std::size_t CLOSEST_POINT_STEPS = 30;
constexpr double FACE_TANGENT_STEP = 1.0e-6;
constexpr double CLOSEST_POINT_CONVERGED = 1.0e-13;
constexpr double FACE_COORDINATE_MARGIN = 1.0e-9;
constexpr double STRAIGHT_FACE_MARGIN = 0.1;
// The sphere about the corners of a face of the skin, enlarged by this factor, holds the face (a curved face leaves
// the sphere of its corners).
constexpr double FACE_SPHERE_MARGIN = 1.5;
// The functions of an edge of the solids are written from the mean of the skin over the fibre of a shell when their
// own share of that mean is not below this (the entries of the inverse of that share at most its inverse): the faces
// beside the edge are then no narrower than about that fraction of the thickness.
constexpr double FIBRE_MIN_EDGE_SHARE = 1.0e-3;

using Row = std::map<TieUnknown, double>;

TieUnknown gridUnknown(std::size_t grid, std::size_t component)
{
    return {TieUnknown::Kind::Grid, grid, component};
}

TieUnknown entityUnknown(std::size_t entity, std::size_t dof)
{
    return {TieUnknown::Kind::SolidEntity, entity, dof};
}

// A local unknown of a solid as an unknown of a tie.
TieUnknown unknownOf(const SolidLocalUnknown& local)
{
    return local.onGrid ? gridUnknown(local.index, local.component) : entityUnknown(local.index, local.component);
}

// out += c × row.
void addRow(Row& out, const Row& row, double c)
{
    for (const auto& [unknown, value] : row)
        out[unknown] += c * value;
}

// The local edge of a shell that joins the two grids of `key`.
std::optional<std::size_t> localEdgeOf(const ShellCorners& corners, const EdgeKey& key)
{
    for (std::size_t k = 0; k < corners.count; ++k)
        if (EdgeKey::of(corners.at(k), corners.at(k + 1)) == key)
            return k;
    return std::nullopt;
}

// The corner of a shell at grid `grid` (one of its corners).
std::size_t cornerAt(const ShellCorners& corners, std::size_t grid)
{
    const std::span<const std::size_t> view = corners.view();
    return static_cast<std::size_t>(std::find(view.begin(), view.end(), grid) - view.begin());
}

// A solid around an edge (tie 4): its kernel, its local unknowns, and the reference points of the lower and the
// higher grid of the edge.
struct SolidAroundEdge
{
    SolidPart::SharedKernel kernel;
    std::vector<SolidLocalUnknown> local;
    SolidPoint lower{};
    SolidPoint higher{};
};

// The solids of an edge whose kernel is accepted (a refused one has its problem reported by validate()), with the
// reference points of the two grids of the edge in each.
std::vector<SolidAroundEdge> solidsAround(const SolidPart& part, const SolidPartEdge& edge)
{
    std::vector<SolidAroundEdge> out;
    for (const std::size_t is : edge.solids)
    {
        std::variant<SolidPart::SharedKernel, std::string> kernel =
            part.kernelOf(is, EntityLayout::Kind::Static, SolidPart::SolidKernelRequest{});
        if (!std::holds_alternative<SolidPart::SharedKernel>(kernel))
            continue;
        const std::vector<std::size_t>& corners = part.solid(is).corners;
        const auto localOf = [&](std::size_t grid)
        { return static_cast<std::size_t>(std::find(corners.begin(), corners.end(), grid) - corners.begin()); };
        const std::span<const SolidPoint> vertices = shapeVertices(part.shapeOf(is));
        out.push_back({std::get<SolidPart::SharedKernel>(std::move(kernel)),
                       part.localUnknowns(is, EntityLayout::Kind::Static), vertices[localOf(edge.grids.lower)],
                       vertices[localOf(edge.grids.higher)]});
    }
    return out;
}

// The rotation of the solids about the unit vector t at the parameter s of their edge (from its lower grid): half the
// curl of their displacement, the mean over the solids of the edge, as a row. With u = Σ_j N_j q_j,
// curl (N e_c) = ∇N × e_c, so ω · t = ½ Σ_jc q_jc (t × ∇N_j)_c.
// Reference: src/tdnns/solid.rs::Solids::rotation_about
Row rotationAbout(std::span<const SolidAroundEdge> around, double s, const SolidPoint& t)
{
    Row out;
    const double count = static_cast<double>(around.size());
    for (const SolidAroundEdge& solid : around)
    {
        const SolidPoint r = add3(scl3(solid.lower, 1.0 - s), scl3(solid.higher, s));
        const std::vector<SolidPoint> gradients = solid.kernel->gradients(r);
        for (std::size_t j = 0; j < gradients.size(); ++j)
        {
            const SolidPoint w = cross3(t, gradients[j]);
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                if (w[c] != 0.0)
                    out[unknownOf(solid.local[COMPONENTS * j + c])] += 0.5 * w[c] / count;
        }
    }
    return out;
}

} // namespace

// ---- the displacement of the skin of the solids at a point (ties 1 and 4 over the thickness of a shell)

// The faces of the skin of the solids, and the displacement of the skin at a point of them as rows on the unknowns of
// the solids. Each face is kept with its corners and the sphere that holds it; the faces are listed when the first
// edge asks for those near it (a model without a shell on its solids lists none), and the kernels of the solids are
// asked when a point is read on them (SolidPart::kernelOf keeps them).
class SkinFaces
{
public:
    explicit SkinFaces(const SolidPart& part) : m_part(&part) {}

    // The faces that come within `reach` of the segment from a to b.
    [[nodiscard]] std::vector<std::size_t> near(const SolidPoint& a, const SolidPoint& b, double reach)
    {
        if (!m_isListed)
            listFaces();
        std::vector<std::size_t> out;
        const SolidPoint chord = sub3(b, a);
        const double length2 = dot3(chord, chord);
        for (std::size_t f = 0; f < m_faces.size(); ++f)
        {
            // the point of the segment closest to the centre of the face
            const SolidPoint toCentre = sub3(m_faces[f].centre, a);
            const double along = length2 > 0.0 ? std::clamp(dot3(toCentre, chord) / length2, 0.0, 1.0) : 0.0;
            if (norm3(sub3(toCentre, scl3(chord, along))) <= m_faces[f].radius + reach)
                out.push_back(f);
        }
        return out;
    }

    // The displacement of the skin at the point p (three rows, basic components), read on the first of the faces
    // `candidates` that holds it within `tolerance`; empty when none does.
    [[nodiscard]] std::optional<std::array<Row, 3>> displacementAt(const SolidPoint& p,
                                                                   std::span<const std::size_t> candidates,
                                                                   double tolerance)
    {
        for (const std::size_t f : candidates)
        {
            const Face& face = m_faces[f];
            if (norm3(sub3(p, face.centre)) > face.radius + tolerance)
                continue;
            const std::optional<std::array<double, 2>> guess = straightCoordinates(face, p);
            if (!guess || !isInside(face, *guess, STRAIGHT_FACE_MARGIN))
                continue;
            const Solid* solid = solidOf(face.solid);
            if (solid == nullptr)
                continue;
            const std::optional<std::vector<double>> functions = functionsAt(*solid, face, p, *guess, tolerance);
            if (!functions)
                continue;
            std::array<Row, 3> rows;
            for (std::size_t j = 0; j < functions->size(); ++j)
                if (std::fabs((*functions)[j]) > FUNCTION_ROUND_OFF)
                    for (std::size_t c = 0; c < COMPONENTS; ++c)
                        rows[c][unknownOf(solid->local[COMPONENTS * j + c])] += (*functions)[j];
            return rows;
        }
        return std::nullopt;
    }

    // The distances z, between `low` and `high`, at which the line origin + z direction (a unit vector) crosses a side
    // of one of the faces `candidates` (the chord of the side, within `tolerance`), increasing and once each.
    [[nodiscard]] std::vector<double> crossings(const SolidPoint& origin, const SolidPoint& direction, double low,
                                                double high, std::span<const std::size_t> candidates,
                                                double tolerance) const
    {
        std::vector<double> out;
        for (const std::size_t f : candidates)
        {
            const Face& face = m_faces[f];
            const std::size_t count = face.corners.cornerCount;
            for (std::size_t k = 0; k < count; ++k)
            {
                // the closest points of the line and of the side A + τ e: z d·d − τ d·e = d·w, −z d·e + τ e·e = −e·w
                const SolidPoint& a = face.positions[k];
                const SolidPoint e = sub3(face.positions[(k + 1) % count], a);
                const SolidPoint w = sub3(a, origin);
                const double de = dot3(direction, e);
                const double ee = dot3(e, e);
                const double determinant = ee - de * de;
                if (!(determinant > PARALLEL_SIDE_RATIO * ee))
                    continue;
                const double z = (dot3(direction, w) * ee - de * dot3(e, w)) / determinant;
                const double tau = (de * dot3(direction, w) - dot3(e, w)) / determinant;
                if (!(z > low && z < high) || tau < -FACE_COORDINATE_MARGIN || tau > 1.0 + FACE_COORDINATE_MARGIN)
                    continue;
                const SolidPoint gap = sub3(add3(origin, scl3(direction, z)), add3(a, scl3(e, tau)));
                if (norm3(gap) <= tolerance)
                    out.push_back(z);
            }
        }
        std::sort(out.begin(), out.end());
        const double same = FIBRE_MIN_PIECE * (high - low);
        const auto isSame = [same](double left, double right) { return right - left <= same; };
        out.erase(std::unique(out.begin(), out.end(), isSame), out.end());
        return out;
    }

private:
    // A side of a face is not crossed by a line nearly parallel to it: sin² of their angle below this.
    static constexpr double PARALLEL_SIDE_RATIO = 1.0e-12;

    struct Face
    {
        std::size_t solid = 0;
        ShapeFace corners;                     // local to the shape of the solid
        std::array<SolidPoint, 4> positions{}; // of its corners, in that order
        SolidPoint centre{};
        double radius = 0.0;
    };
    struct Solid
    {
        SolidPart::SharedKernel kernel;
        std::vector<SolidLocalUnknown> local;
    };

    // Lists the faces of the skin (the faces of one solid), each with its corners in the order of its solid and the
    // sphere about them.
    void listFaces()
    {
        m_isListed = true;
        const std::span<const SolidPoint> x = m_part->gridPositions();
        for (const SolidPartFace& face : m_part->faces())
        {
            if (!face.onSkin() || !m_part->usable(face.solids.front()))
                continue;
            const std::size_t solid = face.solids.front();
            const std::optional<std::size_t> local = m_part->localFaceOf(solid, face.view());
            if (!local)
                continue;
            Face out;
            out.solid = solid;
            out.corners = shapeFaces(m_part->shapeOf(solid))[*local];
            const std::size_t count = out.corners.cornerCount;
            const std::vector<std::size_t>& grids = m_part->solid(solid).corners;
            for (std::size_t k = 0; k < count; ++k)
            {
                out.positions[k] = x[grids[out.corners.corners[k]]];
                out.centre = add3(out.centre, scl3(out.positions[k], 1.0 / static_cast<double>(count)));
            }
            for (std::size_t k = 0; k < count; ++k)
                out.radius = std::max(out.radius, FACE_SPHERE_MARGIN * norm3(sub3(out.positions[k], out.centre)));
            m_faces.push_back(out);
        }
    }

    // The kernel and the unknowns of a solid, asked once; null for a solid whose kernel is refused.
    [[nodiscard]] const Solid* solidOf(std::size_t solid)
    {
        const auto found = m_solids.find(solid);
        if (found != m_solids.end())
            return found->second ? &*found->second : nullptr;
        std::optional<Solid> entry;
        std::variant<SolidPart::SharedKernel, std::string> kernel =
            m_part->kernelOf(solid, EntityLayout::Kind::Static, SolidPart::SolidKernelRequest{});
        if (std::holds_alternative<SolidPart::SharedKernel>(kernel))
            entry = Solid{std::get<SolidPart::SharedKernel>(std::move(kernel)),
                          m_part->localUnknowns(solid, EntityLayout::Kind::Static)};
        const auto inserted = m_solids.emplace(solid, std::move(entry)).first;
        return inserted->second ? &*inserted->second : nullptr;
    }

    [[nodiscard]] static bool isInside(const Face& face, const std::array<double, 2>& c, double margin)
    {
        if (face.corners.cornerCount == TRIANGLE)
            return c[0] >= -margin && c[1] >= -margin && c[0] + c[1] <= 1.0 + margin;
        return c[0] >= -margin && c[0] <= 1.0 + margin && c[1] >= -margin && c[1] <= 1.0 + margin;
    }

    // The point of the face of straight sides through the corners of `face` at the coordinates c (faceWeights), and
    // its two tangents.
    static void straightPoint(const Face& face, const std::array<double, 2>& c, SolidPoint& point,
                              std::array<SolidPoint, 2>& tangent)
    {
        const std::array<SolidPoint, 4>& x = face.positions;
        if (face.corners.cornerCount == TRIANGLE)
        {
            tangent = {sub3(x[1], x[0]), sub3(x[2], x[0])};
            point = add3(x[0], add3(scl3(tangent[0], c[0]), scl3(tangent[1], c[1])));
            return;
        }
        const double s = c[0];
        const double t = c[1];
        point = add3(add3(scl3(x[0], (1.0 - s) * (1.0 - t)), scl3(x[1], s * (1.0 - t))),
                     add3(scl3(x[2], s * t), scl3(x[3], (1.0 - s) * t)));
        tangent = {add3(scl3(sub3(x[1], x[0]), 1.0 - t), scl3(sub3(x[2], x[3]), t)),
                   add3(scl3(sub3(x[3], x[0]), 1.0 - s), scl3(sub3(x[2], x[1]), s))};
    }

    // One Gauss-Newton step towards the point of a surface closest to p: (Jᵀ J) δ = −Jᵀ r, r the residual and J the
    // two tangents; empty when the tangents do not span a plane.
    [[nodiscard]] static std::optional<std::array<double, 2>> closestPointStep(const SolidPoint& residual,
                                                                               const std::array<SolidPoint, 2>& tangent)
    {
        const double a00 = dot3(tangent[0], tangent[0]);
        const double a01 = dot3(tangent[0], tangent[1]);
        const double a11 = dot3(tangent[1], tangent[1]);
        const double b0 = -dot3(tangent[0], residual);
        const double b1 = -dot3(tangent[1], residual);
        const double determinant = a00 * a11 - a01 * a01;
        if (!(determinant > 0.0))
            return std::nullopt;
        return std::array<double, 2>{(a11 * b0 - a01 * b1) / determinant, (a00 * b1 - a01 * b0) / determinant};
    }

    // The coordinates of the point closest to p on the face of straight sides through the corners of `face`; empty
    // when the iteration leaves the face far behind or the face is degenerate.
    [[nodiscard]] static std::optional<std::array<double, 2>> straightCoordinates(const Face& face, const SolidPoint& p)
    {
        std::array<double, 2> c = face.corners.cornerCount == TRIANGLE ? std::array<double, 2>{1.0 / 3.0, 1.0 / 3.0}
                                                                       : std::array<double, 2>{0.5, 0.5};
        for (std::size_t step = 0; step < CLOSEST_POINT_STEPS; ++step)
        {
            SolidPoint point;
            std::array<SolidPoint, 2> tangent;
            straightPoint(face, c, point, tangent);
            const std::optional<std::array<double, 2>> delta = closestPointStep(sub3(point, p), tangent);
            if (!delta)
                return std::nullopt;
            c = {c[0] + (*delta)[0], c[1] + (*delta)[1]};
            if (std::fabs(c[0]) > 2.0 || std::fabs(c[1]) > 2.0)
                return std::nullopt;
            if (std::fabs((*delta)[0]) + std::fabs((*delta)[1]) < CLOSEST_POINT_CONVERGED)
                break;
        }
        return c;
    }

    // The values of the functions of the solid at the point of its face closest to p, when that point is inside the
    // face and within `tolerance` of p. The coordinates `guess` of the face of straight sides are those of the face
    // itself when it is straight (the functions are then evaluated once); on a curved face a Gauss-Newton iteration
    // follows, X(c) the point of the element at the coordinates c of the face and its tangents by differences.
    [[nodiscard]] std::optional<std::vector<double>> functionsAt(const Solid& solid, const Face& face,
                                                                 const SolidPoint& p,
                                                                 const std::array<double, 2>& guess,
                                                                 double tolerance) const
    {
        const VolumeShape shape = m_part->shapeOf(face.solid);
        const VolumeGeometry& geometry = solid.kernel->geometry();
        const std::span<const std::size_t> corners = face.corners.cornerSpan();
        const auto functionsOf = [&](const std::array<double, 2>& c)
        { return geometry.functions(facePoint(shape, corners, c)); };
        const auto positionOf = [&](const std::vector<double>& functions)
        {
            SolidPoint x{};
            for (std::size_t j = 0; j < functions.size(); ++j)
                for (std::size_t m = 0; m < COMPONENTS; ++m)
                    x[m] += functions[j] * geometry.coordinate(j, m);
            return x;
        };
        std::array<double, 2> c = guess;
        std::vector<double> functions = functionsOf(c);
        for (std::size_t step = 0; step < CLOSEST_POINT_STEPS; ++step)
        {
            const SolidPoint residual = sub3(positionOf(functions), p);
            if (norm3(residual) <= FUNCTION_ROUND_OFF * tolerance)
                break; // on the face: the straight face is the face
            std::array<SolidPoint, 2> tangent;
            for (std::size_t a = 0; a < 2; ++a)
            {
                std::array<double, 2> forward = c;
                std::array<double, 2> backward = c;
                forward[a] += FACE_TANGENT_STEP;
                backward[a] -= FACE_TANGENT_STEP;
                tangent[a] = scl3(sub3(positionOf(functionsOf(forward)), positionOf(functionsOf(backward))),
                                  0.5 / FACE_TANGENT_STEP);
            }
            const std::optional<std::array<double, 2>> delta = closestPointStep(residual, tangent);
            if (!delta)
                return std::nullopt;
            c = {c[0] + (*delta)[0], c[1] + (*delta)[1]};
            // a point far outside the face is not on it: its coordinates are not followed further
            if (std::fabs(c[0]) > 2.0 || std::fabs(c[1]) > 2.0)
                return std::nullopt;
            functions = functionsOf(c);
            if (std::fabs((*delta)[0]) + std::fabs((*delta)[1]) < CLOSEST_POINT_CONVERGED)
                break;
        }
        if (!isInside(face, c, FACE_COORDINATE_MARGIN) || norm3(sub3(positionOf(functions), p)) > tolerance)
            return std::nullopt;
        return functions;
    }

    const SolidPart* m_part;
    bool m_isListed = false;
    std::vector<Face> m_faces;                            // the faces of the skin, once listed
    std::map<std::size_t, std::optional<Solid>> m_solids; // by solid, the kernels asked so far
};

namespace
{

// The displacement of the skin of the solids read along the fibre of a shell: at the distance z along the normal, with
// the weight of its point in the integral over the thickness, the three components as rows.
struct FibreSample
{
    double z = 0.0;
    double weight = 0.0;
    std::array<Row, 3> displacement;
};

// What the skin holds of a fibre: the points read, and whether every point of the rule was (the skin holds the whole
// fibre).
struct FibreReading
{
    std::vector<FibreSample> samples; // bounded by the pieces of the fibre times the rule
    bool isComplete = true;
};

// Reads the skin along the fibre of a shell, from offset − h / 2 to offset + h / 2 along its normal: the fibre is cut
// where it crosses the sides of the faces of the skin (the displacement of the skin is a polynomial on each face, with
// a kink across a side), and each piece takes a Gauss rule, so that the integrals over the thickness of the
// displacement and of its moment are those of the skin itself: a rule that does not follow the faces applies the
// force of the shell at its points, as so many forces on lines.
FibreReading readFibre(SkinFaces& skin, std::span<const std::size_t> candidates, const ShellEdgeFibre& fibre)
{
    FibreReading reading;
    const double h = fibre.thickness;
    const double low = fibre.offset - 0.5 * h;
    const double high = fibre.offset + 0.5 * h;
    std::vector<double> ends{low};
    for (const double z :
         skin.crossings(fibre.point, fibre.normal, low, high, candidates, FIBRE_CROSSING_TOLERANCE * h))
        ends.push_back(z);
    ends.push_back(high);
    const std::vector<LineRulePoint> rule = unitIntervalRule(FIBRE_PIECE_POINTS);
    for (std::size_t piece = 0; piece + 1 < ends.size(); ++piece)
    {
        const double length = ends[piece + 1] - ends[piece];
        if (length <= FIBRE_MIN_PIECE * h)
            continue;
        for (const LineRulePoint& point : rule)
        {
            const double z = ends[piece] + point.point * length;
            std::optional<std::array<Row, 3>> rows = skin.displacementAt(
                add3(fibre.point, scl3(fibre.normal, z)), candidates, FIBRE_ON_SKIN_TOLERANCE * h);
            if (!rows)
            {
                reading.isComplete = false;
                continue;
            }
            reading.samples.push_back({z, point.weight * length, std::move(*rows)});
        }
    }
    return reading;
}

// The rotation about the unit tangent t of the fibre of a shell that the solids hold on their skin (a plate that
// continues the solids, a fin that stands on them), as a row: the slope along the fibre of the displacement of the skin
// across it. A rotation ω moves the point at the distance z along the normal n by z ω × n, and
// (ω × n) · (t × n) = ω · t, so with v = u · (t × n) read over the thickness (readFibre),
//   ω · t = ∫ (z − z̄) v dz / ∫ (z − z̄)² dz,   z̄ the middle of what is read,
// the least squares slope of v, which is the moment of the fibre's displacement about its middle over its second
// moment: the couple of the shell enters the solids as the traction linear across its thickness, 12 M z / h³, where
// the mean rotation of the solids at the line of the edge (rotationAbout) takes it as a couple on that line, which a
// continuum resists the less the finer its mesh. Exact for a rigid motion and for a uniform strain, of which it
// reads the rotation of the fibre (the mean rotation plus the shear between the fibre and the conormal). Empty when
// the points of the fibre the skin holds do not span FIBRE_MIN_SPAN of the thickness (a shell that does not lie in
// the skin at its edge, an edge inside the solids). Not in the reference.
std::optional<Row> fibreRotation(SkinFaces& skin, std::span<const std::size_t> candidates, const ShellEdgeFibre& fibre,
                                 const SolidPoint& t)
{
    if (!(fibre.thickness > 0.0))
        return std::nullopt;
    const FibreReading reading = readFibre(skin, candidates, fibre);
    double weight = 0.0;
    double mean = 0.0;
    for (const FibreSample& sample : reading.samples)
    {
        weight += sample.weight;
        mean += sample.weight * sample.z;
    }
    if (!(weight > 0.0))
        return std::nullopt;
    mean /= weight;
    double secondMoment = 0.0;
    for (const FibreSample& sample : reading.samples)
        secondMoment += sample.weight * (sample.z - mean) * (sample.z - mean);
    // the variance of the whole fibre is h² / 12
    const double span = FIBRE_MIN_SPAN * fibre.thickness;
    if (secondMoment < weight * span * span / 12.0)
        return std::nullopt;
    const SolidPoint across = cross3(t, fibre.normal);
    Row out;
    for (const FibreSample& sample : reading.samples)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            addRow(out, sample.displacement[c], across[c] * sample.weight * (sample.z - mean) / secondMoment);
    return out;
}

// The mean over the fibre of a shell of the displacement of the skin of the solids (three rows, basic components):
// (1 / h) ∫ u dz (readFibre). Empty unless the skin holds the whole fibre and the fibre is centred on the surface of
// the grid points (no offset): the mean of an affine field is then its value at the point of the edge. Not in the
// reference.
std::optional<std::array<Row, 3>> fibreMean(SkinFaces& skin, std::span<const std::size_t> candidates,
                                            const ShellEdgeFibre& fibre)
{
    if (!(fibre.thickness > 0.0) || std::fabs(fibre.offset) > FIBRE_ON_SKIN_TOLERANCE * fibre.thickness)
        return std::nullopt;
    const FibreReading reading = readFibre(skin, candidates, fibre);
    if (!reading.isComplete || reading.samples.empty())
        return std::nullopt;
    std::array<Row, 3> mean;
    for (const FibreSample& sample : reading.samples)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            addRow(mean[c], sample.displacement[c], sample.weight / fibre.thickness);
    return mean;
}

} // namespace

// ---- what the edges and faces follow

// Reference: src/tdnns/solid.rs::build (edge_kind, face_kind)
void SolidPart::findTies()
{
    const ShellPart* shells = m_input.shells != nullptr && m_input.shells->usable() ? m_input.shells : nullptr;
    findEdgeTies(shells);
    if (shells != nullptr)
        for (std::size_t ie = 0; ie < shells->input().shells.size(); ++ie)
            coverFaces(ie);
    for (SolidPartEdge& edge : m_edges)
        edge.follows = edge.tie.kind != SolidEdgeTie::Kind::Own;
    for (SolidPartFace& face : m_faces)
        face.follows = face.tie.coveredByShell;
}

// An edge of shells follows the first shell of the edge (by increasing element ID); an edge of lines alone is
// followed by its lines.
// Reference: src/tdnns/solid.rs::build (edge_kind)
void SolidPart::findEdgeTies(const ShellPart* shells)
{
    std::map<EdgeKey, std::size_t> aloneByKey;
    for (std::size_t i = 0; i < m_input.aloneEdges.size(); ++i)
        aloneByKey.emplace(m_input.aloneEdges[i].grids, i);
    for (SolidPartEdge& edge : m_edges)
    {
        if (shells != nullptr)
        {
            const Topology& topology = shells->topology();
            const auto found = topology.edgeIndex.find(edge.grids);
            if (found != topology.edgeIndex.end() && !topology.edgeElements[found->second].empty())
            {
                const std::size_t shell = topology.edgeElements[found->second].front();
                const std::optional<std::size_t> localEdge =
                    localEdgeOf(shells->input().shells[shell].corners, edge.grids);
                OOS_ASSERT(localEdge.has_value());
                edge.tie.kind = SolidEdgeTie::Kind::Shell;
                edge.tie.shell = shell;
                edge.tie.localEdge = *localEdge;
                continue;
            }
        }
        const auto alone = aloneByKey.find(edge.grids);
        if (alone != aloneByKey.end())
        {
            edge.tie.kind = SolidEdgeTie::Kind::Alone;
            edge.tie.aloneEdge = alone->second;
        }
    }
}

// The faces that shell `shell` covers: the face of its corners, or for a quadrilateral the faces of its four halves
// (the first shell of a face wins).
// Reference: src/tdnns/solid.rs::build (face_kind)
void SolidPart::coverFaces(std::size_t shell)
{
    const ShellCorners& corners = m_input.shells->input().shells[shell].corners;
    // the corner sets: the whole shell, then the halves of a quadrilateral
    std::vector<std::vector<std::size_t>> sets(1);
    for (std::size_t k = 0; k < corners.count; ++k)
        sets.front().push_back(k);
    if (corners.count == QUADRILATERAL)
        for (const std::array<std::size_t, 3>& half : QUADRILATERAL_HALVES)
            sets.emplace_back(half.begin(), half.end());
    for (const std::vector<std::size_t>& set : sets)
    {
        std::vector<std::size_t> grids;
        for (const std::size_t k : set)
            grids.push_back(corners.at(k));
        std::sort(grids.begin(), grids.end());
        const auto found = m_faceIndex.find(grids);
        if (found == m_faceIndex.end() || m_faces[found->second].tie.coveredByShell)
            continue;
        SolidPartFace& face = m_faces[found->second];
        face.tie.coveredByShell = true;
        face.tie.shell = shell;
        for (std::size_t i = 0; i < face.count; ++i)
            face.tie.corners[i] = cornerAt(corners, face.grids[i]);
        if (corners.count == QUADRILATERAL && face.count == TRIANGLE)
            followDiagonal(found->second);
    }
}

// A triangle of the solids under half of a quadrilateral shell has the diagonal of the quadrilateral for an edge (its
// two grids at opposite corners): that edge follows the shell where nothing else carries it.
// Reference: src/tdnns/solid.rs::build (EdgeKind::Diagonal)
void SolidPart::followDiagonal(std::size_t face)
{
    const SolidPartFace& f = m_faces[face];
    for (std::size_t i = 0; i < TRIANGLE; ++i)
    {
        const std::size_t j = (i + 1) % TRIANGLE;
        const std::size_t ca = f.tie.corners[i];
        const std::size_t cb = f.tie.corners[j];
        if ((ca + 2) % QUADRILATERAL != cb)
            continue;
        SolidPartEdge& diagonal = m_edges[m_edgeIndex.at(EdgeKey::of(f.grids[i], f.grids[j]))];
        if (diagonal.tie.kind != SolidEdgeTie::Kind::Own)
            continue;
        const bool fromLower = f.grids[i] < f.grids[j];
        diagonal.tie.kind = SolidEdgeTie::Kind::Diagonal;
        diagonal.tie.shell = f.tie.shell;
        diagonal.tie.lowerCorner = fromLower ? ca : cb;
        diagonal.tie.higherCorner = fromLower ? cb : ca;
    }
}

// Reference: src/tdnns/solid.rs::build (the geometry of what follows the shells and the lines alone)
void SolidPart::followTies(const EdgeProjection& projection)
{
    const std::span<const SolidPoint> x = m_input.gridPositions;
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        SolidPartEdge& edge = m_edges[e];
        // an edge of lines alone is straight, whatever its mid-side grid and the skin around it
        if (edge.tie.kind == SolidEdgeTie::Kind::Alone)
        {
            edge.geometry.assign(edge.geometry.size(), SolidPoint{});
            continue;
        }
        if (!edge.follows || !takesCurves() || edge.geometry.empty())
            continue;
        // the curve of the shell along the edge, projected as the skin is
        const std::vector<SolidPoint> values =
            shellPoints(edge.tie.shell, followedEdgePoints(e, projection.parameters));
        edge.geometry = projection.project(values, x[edge.grids.lower], x[edge.grids.higher]);
    }
    if (!takesCurves())
        return;
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        SolidPartFace& face = m_faces[f];
        if (!face.follows || face.geometry.empty())
            continue;
        // the surface of the shell over the face, less what its grids and its edges (as they now are) carry
        std::vector<base::EntityId> order;
        std::vector<SolidPoint> corners;
        std::vector<std::vector<SolidPoint>> edgeCoefficients;
        for (std::size_t i = 0; i < face.count; ++i)
        {
            order.push_back(static_cast<base::EntityId>(face.grids[i]));
            corners.push_back(x[face.grids[i]]);
            const EdgeKey key = EdgeKey::of(face.grids[i], face.grids[(i + 1) % face.count]);
            edgeCoefficients.push_back(m_edges[m_edgeIndex.at(key)].geometry);
        }
        const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
        const std::vector<SolidPoint> values = shellPoints(face.tie.shell, followedFacePoints(f, fp.points));
        face.geometry = fp.project(values, corners, edgeCoefficients);
    }
}

// ---- the points and the rows of the shells

// Reference: src/tdnns/solid.rs::Context::edge_point_of, and the diagonal of build and ties
std::vector<ReferencePoint> SolidPart::followedEdgePoints(std::size_t edge, std::span<const double> parameters) const
{
    const SolidEdgeTie& tie = m_edges[edge].tie;
    std::vector<ReferencePoint> out;
    for (const double s : parameters)
    {
        if (tie.kind == SolidEdgeTie::Kind::Shell)
        {
            out.push_back(m_input.shells->edgeReferencePoint(tie.shell, tie.localEdge, s));
            continue;
        }
        // the diagonal of a quadrilateral: the straight line of the reference square from corner to corner
        const ReferencePoint a = referenceCorner(QUADRILATERAL, tie.lowerCorner);
        const ReferencePoint b = referenceCorner(QUADRILATERAL, tie.higherCorner);
        out.push_back({(1.0 - s) * a[0] + s * b[0], (1.0 - s) * a[1] + s * b[1]});
    }
    return out;
}

// The point of the face at its coordinates c is Σ_i w_i(c) corner i (faceWeights); the same weights of the corners of
// the shell there give its reference point.
// Reference: src/tdnns/solid.rs::Context::face_point_of
std::vector<ReferencePoint> SolidPart::followedFacePoints(std::size_t face,
                                                          std::span<const std::array<double, 2>> points) const
{
    const SolidPartFace& f = m_faces[face];
    const std::size_t cornerCount = m_input.shells->input().shells[f.tie.shell].corners.count;
    std::vector<ReferencePoint> out;
    for (const std::array<double, 2>& c : points)
    {
        const std::vector<double> w = faceWeights(f.count, c);
        ReferencePoint r{};
        for (std::size_t i = 0; i < f.count; ++i)
        {
            const ReferencePoint corner = referenceCorner(cornerCount, f.tie.corners[i]);
            r[0] += w[i] * corner[0];
            r[1] += w[i] * corner[1];
        }
        out.push_back(r);
    }
    return out;
}

// The points of the surface of a shell, X = Σ_i x_i N_i / W.
// Reference: src/tdnns/solid.rs::Context::shell_rows (its point, no function kept)
std::vector<SolidPoint> SolidPart::shellPoints(std::size_t shell, std::span<const ReferencePoint> points) const
{
    const auto none = [](const FunctionKind&) { return false; };
    std::vector<SolidPoint> out;
    for (const ShellDisplacementRows& at : m_input.shells->displacementRowsAt(shell, points, none))
        out.push_back(at.point);
    return out;
}

// Reference: src/tdnns/solid.rs::Context::shell_rows
std::vector<std::array<SolidPart::TieRow, 3>> SolidPart::shellRows(std::size_t shell,
                                                                   std::span<const ReferencePoint> points,
                                                                   const std::optional<std::size_t>& onEdge) const
{
    const ShellPart& shells = *m_input.shells;
    std::function<bool(const FunctionKind&)> keep;
    if (onEdge)
    {
        // on an edge of the shell, the functions of its two corners and of that edge (the others vanish there)
        const std::size_t edge = *onEdge;
        const std::size_t next = (edge + 1) % shells.input().shells[shell].corners.count;
        keep = [edge, next](const FunctionKind& kind)
        {
            if (kind.place == FunctionKind::Place::Vertex)
                return kind.entity == edge || kind.entity == next;
            return kind.place == FunctionKind::Place::Edge && kind.entity == edge;
        };
    }
    // a grid translation of the shells is the translation of the grid; the other unknowns are the shells' own
    const std::size_t firstEntity = shells.firstEntityUnknown();
    constexpr std::size_t GRID_SLOTS = 6;
    std::vector<std::array<TieRow, 3>> out;
    for (const ShellDisplacementRows& at : shells.displacementRowsAt(shell, points, keep))
    {
        std::array<TieRow, 3> rows;
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            for (const ShellPartTerm& term : at.rows[c])
            {
                const TieUnknown unknown = term.unknown < firstEntity
                                               ? gridUnknown(term.unknown / GRID_SLOTS, term.unknown % GRID_SLOTS)
                                               : TieUnknown{TieUnknown::Kind::ShellUnknown, term.unknown, 0};
                rows[c][unknown] += term.coefficient;
            }
        out.push_back(std::move(rows));
    }
    return out;
}

// ---- the rows of the ties

// Reference: src/tdnns/solid.rs::Solids::ties
std::vector<SolidTie> SolidPart::ties(const std::function<bool(const TieUnknown&)>& isHeld) const
{
    std::vector<PendingTie> pending;
    const EdgeProjection projection = EdgeProjection::of(m_traceDegree);
    SkinFaces skin(*this);
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        if (m_edges[e].tie.kind == SolidEdgeTie::Kind::Alone)
            addAloneEdgeTies(e, pending);
        else
            addEdgeTies(e, projection, skin, pending);
    }
    for (std::size_t f = 0; f < m_faces.size(); ++f)
        addFaceTies(f, pending);
    for (std::size_t e = 0; e < m_edges.size(); ++e)
        addRotationTies(e, skin, pending);

    // a dependent held by a support is left out; the first row of a dependent wins; no zero coefficient
    std::set<TieUnknown> written;
    std::vector<SolidTie> out;
    for (const PendingTie& tie : pending)
    {
        if ((isHeld && isHeld(tie.dependent)) || !written.insert(tie.dependent).second)
            continue;
        SolidTie row{tie.dependent, {}};
        for (const auto& [unknown, coefficient] : tie.row)
            if (coefficient != 0.0)
                row.terms.push_back({unknown, coefficient});
        out.push_back(std::move(row));
    }
    return out;
}

// Tie 1: the functions of an edge that follows a shell are the projection of the trace of the shell less its linear
// part, coefficient m = Σ_q a(m, q) u(s_q) + a0[m] u(lower grid) + a1[m] u(higher grid) (EdgeProjection); where the
// shells of the edge stand on the solids, the functions are written from the mean of the skin over their thickness
// instead (addFibreEdgeTies).
// Reference: src/tdnns/solid.rs::Solids::ties (EdgeKind::Shell, EdgeKind::Diagonal)
void SolidPart::addEdgeTies(std::size_t edge, const EdgeProjection& projection, SkinFaces& skin,
                            std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const bool followsShell = e.tie.kind == SolidEdgeTie::Kind::Shell || e.tie.kind == SolidEdgeTie::Kind::Diagonal;
    if (!followsShell || !m_edgeEntities[edge])
        return;
    if (e.tie.kind == SolidEdgeTie::Kind::Shell && addFibreEdgeTies(edge, skin, out))
        return;
    // on an edge of the shell its corners and that edge only; on the diagonal every function of the shell
    const std::optional<std::size_t> onEdge =
        e.tie.kind == SolidEdgeTie::Kind::Shell ? std::optional<std::size_t>(e.tie.localEdge) : std::nullopt;
    const std::vector<std::array<TieRow, 3>> field =
        shellRows(e.tie.shell, followedEdgePoints(edge, projection.parameters), onEdge);
    for (std::size_t m = 0; m < e.geometry.size(); ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            TieRow row;
            for (std::size_t q = 0; q < field.size(); ++q)
                addRow(row, field[q][c], projection.weights(m, q));
            row[gridUnknown(e.grids.lower, c)] += projection.lowerWeights[m];
            row[gridUnknown(e.grids.higher, c)] += projection.upperWeights[m];
            out.push_back({entityUnknown(*m_edgeEntities[edge], COMPONENTS * m + c), std::move(row)});
        }
}

// Whether shell `shell` of edge `e` lies on a face of the solids: one of its other corners makes a face of the solids
// with the edge.
bool SolidPart::liesOnFace(std::size_t shell, const SolidPartEdge& e) const
{
    for (const std::size_t g : m_input.shells->input().shells[shell].corners.view())
    {
        if (g == e.grids.lower || g == e.grids.higher)
            continue;
        std::array<std::size_t, 3> three{e.grids.lower, e.grids.higher, g};
        std::sort(three.begin(), three.end());
        if (m_faceByThree.contains(three))
            return true;
    }
    return false;
}

// Tie 1 over the thickness of the shells, for an edge none of whose shells lies on a face of the solids (a plate
// that continues the solids, a fin that stands on them) and whose fibre the skin holds all along it: the moments of
// the displacement of the edge of the shells are those of the mean of the skin over the fibre,
//   q_shell(l, c) = ∫ W ū_c P_l(2s − 1) ds,   ū = (1 / h) ∫ u dz  (fibreMean),
// in place of the trace of the solids on the line of the edge: the force of the shells then enters the solids as a
// traction spread over their thickness, where the line takes it as a force on a line, under which a continuum dents
// the more the finer its mesh. The relation is written on the functions of the edge of the solids, as tie 1 is: with
// G_lm the share of function m of the edge in the moment l of the mean (the same for each component),
//   q_edge(m, c) = Σ_l G⁻¹_ml (q_shell(l, c) − the other terms of the moment l of ū_c).
// An affine field passes through it exactly (its mean over the fibre is its value on the edge). False, and nothing
// written, where it does not apply: the edge then follows the trace of the shell (addEdgeTies). Not in the reference.
bool SolidPart::addFibreEdgeTies(std::size_t edge, SkinFaces& skin, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const ShellPart& shells = *m_input.shells;
    const std::size_t topologyEdge = shells.topology().edgeIndex.at(e.grids);
    const std::vector<std::size_t>& owners = shells.topology().edgeElements[topologyEdge];
    if (std::any_of(owners.begin(), owners.end(), [&](std::size_t shell) { return liesOnFace(shell, e); }))
        return false;
    const ShellPart::EdgeMoments moments = shells.edgeMoments(topologyEdge);
    const std::size_t functionCount = e.geometry.size();
    if (functionCount == 0 || moments.displacementWeights.size() != functionCount)
        return false;
    // the fibres of the first shell of the edge at the points of the moments, and the mean of the skin over each
    const std::optional<std::size_t> localEdge = localEdgeOf(shells.input().shells[owners.front()].corners, e.grids);
    OOS_ASSERT(localEdge.has_value());
    std::vector<ShellEdgeFibre> fibres;
    double reach = 0.0;
    for (const double s : moments.parameters)
    {
        fibres.push_back(shells.edgeFibre(owners.front(), *localEdge, s));
        reach = std::max(reach, 0.5 * fibres.back().thickness);
    }
    const std::span<const SolidPoint> x = m_input.gridPositions;
    const std::vector<std::size_t> candidates = skin.near(x[e.grids.lower], x[e.grids.higher], reach);
    std::vector<std::array<TieRow, 3>> means;
    for (const ShellEdgeFibre& fibre : fibres)
    {
        std::optional<std::array<TieRow, 3>> mean = fibreMean(skin, candidates, fibre);
        if (!mean)
            return false;
        means.push_back(std::move(*mean));
    }
    // the moments of the mean, and the share of the functions of the edge in them
    const std::size_t entity = *m_edgeEntities[edge];
    std::vector<std::array<TieRow, 3>> rows(functionCount);
    for (std::size_t l = 0; l < functionCount; ++l)
        for (std::size_t g = 0; g < means.size(); ++g)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                addRow(rows[l][c], means[g][c], moments.displacementWeights[l][g]);
    Matrix share(functionCount, functionCount);
    for (std::size_t l = 0; l < functionCount; ++l)
        for (std::size_t m = 0; m < functionCount; ++m)
        {
            const auto found = rows[l][0].find(entityUnknown(entity, COMPONENTS * m));
            share(l, m) = found == rows[l][0].end() ? 0.0 : found->second;
        }
    const std::optional<Matrix> shareInverse = inverse(share);
    if (!shareInverse)
        return false;
    for (std::size_t m = 0; m < functionCount; ++m)
        for (std::size_t l = 0; l < functionCount; ++l)
            if (!(std::fabs((*shareInverse)(m, l)) <= 1.0 / FIBRE_MIN_EDGE_SHARE))
                return false;
    for (std::size_t m = 0; m < functionCount; ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            TieRow row;
            for (std::size_t l = 0; l < functionCount; ++l)
            {
                const double coefficient = (*shareInverse)(m, l);
                row[{TieUnknown::Kind::ShellUnknown, moments.displacementUnknowns[COMPONENTS * l + c], 0}] +=
                    coefficient;
                for (const auto& [unknown, value] : rows[l][c])
                    if (!(unknown.kind == TieUnknown::Kind::SolidEntity && unknown.index == entity))
                        row[unknown] -= coefficient * value;
            }
            out.push_back({entityUnknown(entity, COMPONENTS * m + c), std::move(row)});
        }
    return true;
}

// Tie 3: both the line alone and the solid edge are linear between the grids plus functions that vanish there; those
// of the line, f_l = s (1 − s) P_l(2s − 1) along the axes of its frame, follow the L2 projection of those of the solid,
// g_m along the basic axes: q_line(l, axis) = Σ_m C(l, m) axis · q_solid(m), C = G⁻¹ R, G_li = ∫ f_l f_i,
// R_lm = ∫ f_l g_m (the same functions when k and the trace degree are 4).
// Reference: src/tdnns/solid.rs::Solids::ties (EdgeKind::Alone)
void SolidPart::addAloneEdgeTies(std::size_t edge, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const SolidPartAloneEdge& alone = m_input.aloneEdges[e.tie.aloneEdge];
    const std::size_t k = alone.alone.degree;
    const std::size_t lineFunctions = k - 1;
    const std::size_t solidFunctions = e.geometry.size();
    Matrix gram(lineFunctions, lineFunctions);
    Matrix cross(lineFunctions, solidFunctions);
    for (const LineRulePoint& point : unitIntervalRule(k + ALONE_PROJECTION_EXTRA_POINTS))
    {
        const std::vector<double> line = alone.alone.functions(point.point).values;
        const std::vector<double> solid = edgeFunctions(m_traceDegree, point.point);
        for (std::size_t i = 0; i < lineFunctions; ++i)
        {
            const double fi = line[ALONE_GRID_FUNCTIONS + i];
            for (std::size_t j = 0; j < lineFunctions; ++j)
                gram(i, j) += point.weight * fi * line[ALONE_GRID_FUNCTIONS + j];
            for (std::size_t m = 0; m < solidFunctions; ++m)
                cross(i, m) += point.weight * fi * solid[m];
        }
    }
    const std::optional<Matrix> gramInverse = inverse(gram);
    if (!gramInverse)
        throw base::InternalError("the Gram matrix of the functions of an edge alone is singular");
    const Matrix c = *gramInverse * cross;
    for (std::size_t l = 0; l < lineFunctions; ++l)
        for (std::size_t axis = 0; axis < COMPONENTS; ++axis)
        {
            TieRow row;
            for (std::size_t m = 0; m < solidFunctions; ++m)
                for (std::size_t cc = 0; cc < COMPONENTS; ++cc)
                    row[entityUnknown(*m_edgeEntities[edge], COMPONENTS * m + cc)] +=
                        c(l, m) * alone.alone.frame[axis][cc];
            out.push_back({{TieUnknown::Kind::AloneEdge, alone.edge, COMPONENTS * l + axis}, std::move(row)});
        }
}

// Tie 2: the functions of a face that a shell covers are the projection of the displacement of the shell less what
// the grids and the edges of the face carry, coefficient m = Σ_q b(m, q) (u(c_q) − Σ_i φ_i(c_q) u_i − Σ_kn φ_kn(c_q)
// q_kn) (FaceProjection).
// Reference: src/tdnns/solid.rs::Solids::ties (the faces)
void SolidPart::addFaceTies(std::size_t face, std::vector<PendingTie>& out) const
{
    const SolidPartFace& f = m_faces[face];
    if (!f.follows || !m_faceEntities[face])
        return;
    const std::size_t nf = f.count;
    std::vector<base::EntityId> order;
    std::vector<std::optional<std::size_t>> edgeEntities;
    for (std::size_t i = 0; i < nf; ++i)
    {
        order.push_back(static_cast<base::EntityId>(f.grids[i]));
        edgeEntities.push_back(m_edgeEntities[m_edgeIndex.at(EdgeKey::of(f.grids[i], f.grids[(i + 1) % nf]))]);
    }
    const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
    const std::vector<std::array<TieRow, 3>> field =
        shellRows(f.tie.shell, followedFacePoints(face, fp.points), std::nullopt);
    for (std::size_t m = 0; m < fp.interiorCount; ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            TieRow row;
            for (std::size_t q = 0; q < fp.points.size(); ++q)
            {
                const double b = fp.weights(m, q);
                const std::vector<double>& phi = fp.functionValues[q];
                addRow(row, field[q][c], b);
                for (std::size_t i = 0; i < nf; ++i)
                    row[gridUnknown(f.grids[i], c)] -= b * phi[i];
                for (std::size_t k = 0; k < nf; ++k)
                    for (std::size_t n = 0; n < fp.perEdge; ++n)
                        row[entityUnknown(*edgeEntities[k], COMPONENTS * n + c)] -= b * phi[nf + k * fp.perEdge + n];
            }
            out.push_back({entityUnknown(*m_faceEntities[face], COMPONENTS * m + c), std::move(row)});
        }
}

// Tie 4: the rotation unknowns l of an edge of shells or of lines alone that the solids have, where a shell of the
// edge does not lie on a face of the solids, follow the rotation of the solids about the edge: the Legendre
// coefficients (2l + 1) ∫ W² |X_s| (ω · t) P_l(2s − 1) ds of the rotation of the edge (ShellEdgeRotationFrame; W = 1
// and |X_s| = L on a straight edge of lines alone). ω · t is the rotation of the fibre of that shell read on the skin
// of the solids over its thickness (fibreRotation; not in the reference), or, where the skin does not hold the fibre
// and for the lines alone, the mean rotation of the solids at the line of the edge (rotationAbout, the reference's).
// Reference: src/tdnns/solid.rs::Solids::ties (the rotations about the edges)
void SolidPart::addRotationTies(std::size_t edge, SkinFaces& skin, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const std::vector<LineRulePoint> rule = unitIntervalRule(ROTATION_RULE_POINTS);
    // the rotation unknowns of the edge, its tangent and the measure of its rotation at the points of the rule
    std::vector<TieUnknown> dependents;
    std::vector<ShellEdgeRotationFrame> frames;
    // the rotation of the solids at the points of the rule, read over the thickness of a shell (every point, or none)
    std::vector<TieRow> fibreRotations;
    if (e.tie.kind == SolidEdgeTie::Kind::Shell)
    {
        const ShellPart& shells = *m_input.shells;
        const std::size_t topologyEdge = shells.topology().edgeIndex.at(e.grids);
        const std::vector<std::size_t>& owners = shells.topology().edgeElements[topologyEdge];
        const auto lies = [&](std::size_t shell) { return liesOnFace(shell, e); };
        const auto standing = std::find_if_not(owners.begin(), owners.end(), lies);
        if (standing == owners.end())
            return;
        const std::size_t first = shells.entities()[shells.edgeEntity(topologyEdge)].firstUnknown;
        for (std::size_t l = 0; l < shells.degree(); ++l)
            dependents.push_back({TieUnknown::Kind::ShellUnknown, first + shells.edgeRotationDof(l), 0});
        for (const LineRulePoint& point : rule)
            frames.push_back(shells.edgeRotationFrame(e.tie.shell, e.tie.localEdge, point.point));
        // The rotation over the thickness of the first shell that does not lie on a face of the solids, where the
        // skin holds its fibre all along the edge (fibreRotation); elsewhere the mean rotation of the solids at the
        // line of the edge, below.
        const std::optional<std::size_t> standingEdge = localEdgeOf(shells.input().shells[*standing].corners, e.grids);
        OOS_ASSERT(standingEdge.has_value());
        std::vector<ShellEdgeFibre> fibres;
        double reach = 0.0;
        for (const LineRulePoint& point : rule)
        {
            fibres.push_back(shells.edgeFibre(*standing, *standingEdge, point.point));
            reach = std::max(reach, std::fabs(fibres.back().offset) + 0.5 * fibres.back().thickness);
        }
        const std::span<const SolidPoint> x = m_input.gridPositions;
        const std::vector<std::size_t> candidates = skin.near(x[e.grids.lower], x[e.grids.higher], reach);
        for (std::size_t q = 0; q < rule.size() && fibreRotations.size() == q; ++q)
            if (std::optional<TieRow> row = fibreRotation(skin, candidates, fibres[q], frames[q].tangent))
                fibreRotations.push_back(std::move(*row));
    }
    else if (e.tie.kind == SolidEdgeTie::Kind::Alone)
    {
        const SolidPartAloneEdge& alone = m_input.aloneEdges[e.tie.aloneEdge];
        const std::size_t k = alone.alone.degree;
        for (std::size_t l = 0; l < k; ++l)
            dependents.push_back({TieUnknown::Kind::AloneEdge, alone.edge, COMPONENTS * (k - 1) + l});
        const SolidPoint chord = sub3(alone.alone.upperPoint, alone.alone.lowerPoint);
        frames.assign(rule.size(), ShellEdgeRotationFrame{unit3(chord), norm3(chord)});
    }
    else
    {
        return;
    }

    std::vector<TieRow> rotations;
    if (fibreRotations.size() == rule.size())
    {
        rotations = std::move(fibreRotations);
    }
    else
    {
        const std::vector<SolidAroundEdge> around = solidsAround(*this, e);
        if (around.empty())
            return;
        for (std::size_t q = 0; q < rule.size(); ++q)
            rotations.push_back(rotationAbout(around, rule[q].point, frames[q].tangent));
    }
    for (std::size_t l = 0; l < dependents.size(); ++l)
    {
        TieRow row;
        for (std::size_t q = 0; q < rule.size(); ++q)
            addRow(row, rotations[q],
                   static_cast<double>(2 * l + 1) * rule[q].weight * legendreValue(l, 2.0 * rule[q].point - 1.0) *
                       frames[q].measure);
        out.push_back({dependents[l], std::move(row)});
    }
}

// ---- the functions of the skin from a field known at points

std::vector<SolidSkinTrace> SolidPart::skinTraces(std::span<const std::size_t> grids) const
{
    std::vector<SolidSkinTrace> out;
    const auto isGiven = [&](std::size_t grid) { return std::binary_search(grids.begin(), grids.end(), grid); };
    const std::span<const SolidPoint> x = m_input.gridPositions;
    const EdgeProjection projection = EdgeProjection::of(m_traceDegree);
    // An edge: the curve X(s) = (1 − s) X_a + s X_b + Σ_n g_n b_n(s), and the projection of the field less its
    // linear part (tie 1): coefficient m = Σ_p a(m, p) u(s_p) + a0[m] u(a) + a1[m] u(b).
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        const SolidPartEdge& edge = m_edges[e];
        if (!m_edgeEntities[e] || !edge.onSkin || edge.follows || edge.geometry.empty() ||
            !isGiven(edge.grids.lower) || !isGiven(edge.grids.higher))
            continue;
        SolidSkinTrace trace;
        trace.grids = {edge.grids.lower, edge.grids.higher};
        trace.entity = *m_edgeEntities[e];
        for (const double s : projection.parameters)
        {
            const std::vector<double> functions = edgeFunctions(m_traceDegree, s);
            SolidPoint point = add3(scl3(x[edge.grids.lower], 1.0 - s), scl3(x[edge.grids.higher], s));
            for (std::size_t n = 0; n < edge.geometry.size(); ++n)
                point = add3(point, scl3(edge.geometry[n], functions[n]));
            trace.points.push_back(point);
        }
        trace.weights = projection.weights;
        for (std::size_t m = 0; m < edge.geometry.size(); ++m)
            trace.terms.push_back({{gridUnknown(edge.grids.lower, 0), projection.lowerWeights[m]},
                                   {gridUnknown(edge.grids.higher, 0), projection.upperWeights[m]}});
        out.push_back(std::move(trace));
    }
    // A face: the surface X(c) = Σ_i φ_i(c) X_i + Σ_kn φ_kn(c) g_kn + Σ_m φ_m(c) g_m (its grids, its edges, its own
    // functions), and the projection of the field less what its grids and its edges carry (tie 2):
    // coefficient m = Σ_p b(m, p) (u(c_p) − Σ_i φ_i(c_p) u_i − Σ_kn φ_kn(c_p) q_kn).
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        const SolidPartFace& face = m_faces[f];
        const std::span<const std::size_t> cycle = face.view();
        if (!m_faceEntities[f] || !face.onSkin() || face.follows || face.geometry.empty() ||
            !std::all_of(cycle.begin(), cycle.end(), isGiven))
            continue;
        const std::size_t nf = face.count;
        std::vector<base::EntityId> order;
        std::vector<std::size_t> edges;
        bool hasEdgeFunctions = true;
        for (std::size_t i = 0; i < nf; ++i)
        {
            order.push_back(static_cast<base::EntityId>(face.grids[i]));
            edges.push_back(m_edgeIndex.at(EdgeKey::of(face.grids[i], face.grids[(i + 1) % nf])));
            hasEdgeFunctions = hasEdgeFunctions && m_edgeEntities[edges.back()].has_value();
        }
        if (!hasEdgeFunctions)
            continue;
        const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
        if (fp.interiorCount != face.geometry.size())
            throw base::InternalError("the functions of a face of the skin are not those of its projection");
        SolidSkinTrace trace;
        trace.grids.assign(cycle.begin(), cycle.end());
        trace.entity = *m_faceEntities[f];
        const std::size_t firstInterior = nf + nf * fp.perEdge;
        for (std::size_t p = 0; p < fp.points.size(); ++p)
        {
            const std::vector<double>& phi = fp.functionValues[p];
            SolidPoint point{};
            for (std::size_t i = 0; i < nf; ++i)
                point = add3(point, scl3(x[face.grids[i]], phi[i]));
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t n = 0; n < fp.perEdge && n < m_edges[edges[k]].geometry.size(); ++n)
                    point = add3(point, scl3(m_edges[edges[k]].geometry[n], phi[nf + k * fp.perEdge + n]));
            for (std::size_t m = 0; m < fp.interiorCount; ++m)
                point = add3(point, scl3(face.geometry[m], phi[firstInterior + m]));
            trace.points.push_back(point);
        }
        trace.weights = Matrix(fp.interiorCount, fp.points.size());
        for (std::size_t m = 0; m < fp.interiorCount; ++m)
        {
            std::vector<double> onGrids(nf, 0.0);
            std::vector<double> onEdges(nf * fp.perEdge, 0.0);
            for (std::size_t p = 0; p < fp.points.size(); ++p)
            {
                const double b = fp.weights(m, p);
                const std::vector<double>& phi = fp.functionValues[p];
                trace.weights(m, p) = b;
                for (std::size_t i = 0; i < nf; ++i)
                    onGrids[i] -= b * phi[i];
                for (std::size_t j = 0; j < nf * fp.perEdge; ++j)
                    onEdges[j] -= b * phi[nf + j];
            }
            std::vector<TieTerm> terms;
            for (std::size_t i = 0; i < nf; ++i)
                terms.push_back({gridUnknown(face.grids[i], 0), onGrids[i]});
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t n = 0; n < fp.perEdge; ++n)
                    terms.push_back(
                        {entityUnknown(*m_edgeEntities[edges[k]], COMPONENTS * n), onEdges[k * fp.perEdge + n]});
            trace.terms.push_back(std::move(terms));
        }
        out.push_back(std::move(trace));
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
