// Checks of what a glued contact reads of the shells of the mixed hybrid formulation (model/ShellPart.hpp): the
// unknowns of an edge as moments of the motion of its points (edgeMoments), the interior unknowns of a shell as the
// projection of the motion of its points (interiorMoments), and the motion of a point that may lie on an edge,
// where the rotation about the edge is that of the edge (pointRowsOnEdges).

#include "CheckRunner.hpp"
#include "ShellPartCheckSupport.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace oos::checks
{

namespace
{

using namespace oos::checks::shellpart;

// A flat plate of 2 × 2 elements that are no parallelograms, and a cylinder panel (axis x, radius 1, 0.8 rad).
MappedMesh skewedPlate()
{
    return mappedMesh(
        2, 2, [](double s, double t) { return mh::SpaceVector{s + 0.2 * s * t, t + 0.1 * s * s, 0.0}; });
}

MappedMesh curvedPanel()
{
    return mappedMesh(2, 2,
                      [](double s, double t)
                      {
                          const double a = 0.8 * (t - 0.5);
                          return mh::SpaceVector{2.0 * s, std::sin(a), std::cos(a)};
                      });
}

mh::ShellPart partOf(const MappedMesh& mesh, bool triangles)
{
    mh::ShellPartInput input = shellInput(mesh, isotropicSection(7e10, 0.3, 0.01), triangles);
    input.checkElements = false;
    return mh::ShellPart(input);
}

double valueOf(const std::vector<mh::ShellPartTerm>& row, const mh::Vector& u)
{
    double value = 0.0;
    for (const mh::ShellPartTerm& term : row)
        value += term.coefficient * u[term.unknown];
    return value;
}

// Points of the reference element of a shell of `cornerCount` corners: inside it, on its first edge, at its first
// corner, and on its second edge.
std::array<mh::ReferencePoint, 4> probePoints(std::size_t cornerCount)
{
    if (cornerCount == 3)
        return {mh::ReferencePoint{-0.5, -0.3}, mh::ReferencePoint{0.2, -1.0}, mh::ReferencePoint{-1.0, -1.0},
                mh::ReferencePoint{0.2, -0.2}};
    return {mh::ReferencePoint{0.3, -0.4}, mh::ReferencePoint{0.3, -1.0}, mh::ReferencePoint{-1.0, -1.0},
            mh::ReferencePoint{1.0, 0.2}};
}

} // namespace

CHECK_CASE("shellpart: the unknowns of an edge are the moments of the motion of its points")
{
    // u = b + A x with strains and a rotation; its rotation is the axial vector of the skew part of A
    const std::array<mh::SpaceVector, 3> a{{{1e-3, 2e-4, 3e-4}, {-3e-4, -5e-4, 1e-4}, {2e-4, -4e-4, 6e-4}}};
    const mh::SpaceVector b{1e-4, -2e-4, 3e-4};
    const mh::SpaceVector omega{0.5 * (a[2][1] - a[1][2]), 0.5 * (a[0][2] - a[2][0]), 0.5 * (a[1][0] - a[0][1])};
    for (const bool isCurved : {false, true})
        for (const bool triangles : {false, true})
        {
            const mh::ShellPart part = partOf(isCurved ? curvedPanel() : skewedPlate(), triangles);
            const mh::Vector exact = affineVector(part, a, b);
            const double scale = maxAbsOf(exact);
            const std::string what = std::string(isCurved ? "cylinder panel" : "flat plate") +
                                     (triangles ? ", triangles" : ", quadrilaterals");
            double displacementError = 0.0;
            double rotationError = 0.0;
            for (std::size_t e = 0; e < part.topology().edges.size(); ++e)
            {
                const mh::ShellPart::EdgeMoments moments = part.edgeMoments(e);
                for (std::size_t i = 0; i < moments.displacementUnknowns.size(); ++i)
                {
                    const std::size_t c = i % 3;
                    double value = 0.0;
                    for (std::size_t g = 0; g < moments.points.size(); ++g)
                    {
                        const mh::SpaceVector& x = moments.points[g];
                        value += moments.displacementWeights[i / 3][g] *
                                 (b[c] + a[c][0] * x[0] + a[c][1] * x[1] + a[c][2] * x[2]);
                    }
                    displacementError =
                        std::max(displacementError, std::fabs(value - exact[moments.displacementUnknowns[i]]));
                }
                for (std::size_t l = 0; l < moments.rotationUnknowns.size(); ++l)
                {
                    double value = 0.0;
                    for (const mh::SpaceVector& weight : moments.rotationWeights[l])
                        value += mh::dot3(weight, omega);
                    rotationError = std::max(rotationError, std::fabs(value - exact[moments.rotationUnknowns[l]]));
                }
            }
            runner.expect(displacementError <= 1e-10 * scale,
                          what + ": the displacement unknowns of the edges, error " + sci(displacementError / scale));
            runner.expect(rotationError <= 1e-10 * scale,
                          what + ": the rotation unknowns of the edges, error " + sci(rotationError / scale));
        }
}

CHECK_CASE("shellpart: the interior unknowns of a shell are the projection of the motion of its points")
{
    const std::array<mh::SpaceVector, 3> a{{{1e-3, 2e-4, 3e-4}, {-3e-4, -5e-4, 1e-4}, {2e-4, -4e-4, 6e-4}}};
    const mh::SpaceVector b{1e-4, -2e-4, 3e-4};
    for (const bool isCurved : {false, true})
        for (const bool triangles : {false, true})
        {
            const mh::ShellPart part = partOf(isCurved ? curvedPanel() : skewedPlate(), triangles);
            const mh::Vector exact = affineVector(part, a, b);
            const double scale = maxAbsOf(exact);
            const std::string what = std::string(isCurved ? "cylinder panel" : "flat plate") +
                                     (triangles ? ", triangles" : ", quadrilaterals");
            double error = 0.0;
            std::size_t count = 0;
            for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
            {
                const mh::ShellPart::InteriorMoments moments = part.interiorMoments(ie);
                count += moments.unknowns.size();
                for (std::size_t i = 0; i < moments.unknowns.size(); ++i)
                {
                    const std::size_t c = i % 3;
                    double value = valueOf(moments.terms[i], exact);
                    for (std::size_t q = 0; q < moments.points.size(); ++q)
                    {
                        const mh::SpaceVector& x = moments.points[q];
                        value += moments.weights[i][q] * (b[c] + a[c][0] * x[0] + a[c][1] * x[1] + a[c][2] * x[2]);
                    }
                    error = std::max(error, std::fabs(value - exact[moments.unknowns[i]]));
                }
            }
            runner.expect(count > 0, what + ": interior unknowns");
            runner.expect(error <= 1e-10 * scale, what + ": the interior unknowns, error " + sci(error / scale));
        }
}

CHECK_CASE("shellpart: a point on an edge follows a rigid motion, and turns about the edge as the edge does")
{
    const mh::SpaceVector omega{0.3, -0.5, 0.7};
    const mh::SpaceVector b{0.1, 0.2, 0.3};
    for (const bool isCurved : {false, true})
        for (const bool triangles : {false, true})
        {
            const mh::ShellPart part = partOf(isCurved ? curvedPanel() : skewedPlate(), triangles);
            const mh::Vector rigid = affineVector(part, rotationGradient(omega), b);
            const std::string what = std::string(isCurved ? "cylinder panel" : "flat plate") +
                                     (triangles ? ", triangles" : ", quadrilaterals");
            const std::size_t cornerCount = triangles ? 3 : 4;
            double error = 0.0;
            for (const mh::ReferencePoint& at : probePoints(cornerCount))
            {
                const mh::ShellPart::PointRows rows = part.pointRowsOnEdges(0, at[0], at[1]);
                const mh::SpaceVector expected = mh::add3(b, mh::cross3(omega, rows.point));
                for (std::size_t c = 0; c < 3; ++c)
                {
                    error = std::max(error, std::fabs(valueOf(rows.rows[c], rigid) - expected[c]));
                    error = std::max(error, std::fabs(valueOf(rows.rows[3 + c], rigid) - omega[c]));
                }
            }
            runner.expect(error <= 1e-10, what + ": the rigid motion at points inside, on an edge and at a corner, "
                                                 "error " + sci(error));

            // On the first edge of the first shell, between its corners 0 and 1: with the rotation unknowns of that
            // edge at zero (a clamped edge), the rotation about the edge is zero whatever the other unknowns are.
            const mh::ShellCorners& corners = part.input().shells[0].corners;
            const std::size_t edge = part.topology().edgeIndex.at(mh::EdgeKey::of(corners.at(0), corners.at(1)));
            const mh::ShellPart::EdgeMoments moments = part.edgeMoments(edge);
            mh::Vector u(part.unknownCount(), 0.0);
            for (std::size_t i = 0; i < u.size(); ++i)
                u[i] = std::sin(1.0 + 0.7 * static_cast<double>(i));
            for (const std::size_t unknown : moments.rotationUnknowns)
                u[unknown] = 0.0;
            const mh::ReferencePoint onEdge = probePoints(cornerCount)[1];
            const mh::ShellPart::PointRows rows = part.pointRowsOnEdges(0, onEdge[0], onEdge[1]);
            // the tangent of the edge at the point: the shell runs its first edge from the lower grid of the edge, so
            // the parameter of the edge is that of the shell, (ξ + 1) / 2
            const mh::SpaceVector tangent = part.edgeRotationFrame(0, 0, 0.5 * (onEdge[0] + 1.0)).tangent;
            double about = 0.0;
            double size = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
            {
                const double theta = valueOf(rows.rows[3 + c], u);
                about += tangent[c] * theta;
                size = std::max(size, std::fabs(theta));
            }
            runner.expect(std::fabs(about) <= 1e-10 * std::max(size, 1.0),
                          what + ": the rotation about an edge whose rotation unknowns are zero, " + sci(about) +
                              " of " + sci(size));
        }
}

} // namespace oos::checks
