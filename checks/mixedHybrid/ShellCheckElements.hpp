#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// Single shell elements of the mixed hybrid formulation built by hand for the checks: flat, warped and curved (a
// rational patch), quadrilaterals and triangles, with the geometry of the reference implementation (the map from the
// corners, the edge moments of X W and the projection of the interior; ShellElementData.hpp).
namespace oos::checks::shell
{

namespace mh = oos::elements::mixedHybrid;

// The surface of a test element and its weight polynomial.
struct TestSurface
{
    std::size_t cornerCount = 4;
    std::function<mh::SpaceVector(double, double)> position;
    mh::Polynomial2 weight = mh::Polynomial2::constant(1.0);
};

// The bilinear quadrilateral of four corners (warped when they are not coplanar).
inline TestSurface bilinearQuadrilateral(const std::array<mh::SpaceVector, 4>& p)
{
    TestSurface s;
    s.cornerCount = 4;
    s.position = [p](double xi, double eta)
    {
        const double n[4] = {0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta), 0.25 * (1 + xi) * (1 + eta),
                             0.25 * (1 - xi) * (1 + eta)};
        mh::SpaceVector x{};
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t c = 0; c < 3; ++c)
                x[c] += n[k] * p[k][c];
        return x;
    };
    return s;
}

// The affine triangle of three corners.
inline TestSurface affineTriangle(const std::array<mh::SpaceVector, 3>& p)
{
    TestSurface s;
    s.cornerCount = 3;
    s.position = [p](double xi, double eta)
    {
        const std::array<double, 3> l = mh::barycentric(xi, eta);
        mh::SpaceVector x{};
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t c = 0; c < 3; ++c)
                x[c] += l[k] * p[k][c];
        return x;
    };
    return s;
}

// A 90° cylindrical quadrilateral of radius r about the y axis and length `length`: the arc from (r, 0, 0) to
// (0, 0, r) is the rational quadratic of control points (r, 0), (r, r), (0, r) and weights 1, √2/2, 1 in s = (1 + ξ)/2,
// the axis linear in t = (1 + η)/2. X = N / W exactly (the surface is the cylinder), W = (1 − s)² + √2 s (1 − s) + s².
inline TestSurface cylinderQuadrant(double r, double length)
{
    const double w1 = std::numbers::sqrt2 / 2.0;
    TestSurface s;
    s.cornerCount = 4;
    s.position = [r, length, w1](double xi, double eta)
    {
        const double u = 0.5 * (1.0 + xi);
        const double t = 0.5 * (1.0 + eta);
        const double b0 = (1 - u) * (1 - u);
        const double b1 = 2 * u * (1 - u) * w1;
        const double b2 = u * u;
        const double w = b0 + b1 + b2;
        return mh::SpaceVector{(b0 * r + b1 * r) / w, length * t, (b1 * r + b2 * r) / w};
    };
    const mh::Polynomial2 su = mh::Polynomial2::linearInXi(0.5, 0.5);
    const mh::Polynomial2 one = mh::Polynomial2::constant(1.0);
    s.weight = (one - su) * (one - su) + (su * (one - su)).scaled(2.0 * w1) + su * su;
    return s;
}

// The rational quadratic triangle (Bézier form) of corners p_k, edge control points c_k and edge weights w_k:
//   W = Σ L_k² + 2 Σ w_k L_k L_{k+1},  N = Σ L_k² p_k + 2 Σ w_k L_k L_{k+1} c_k,  X = N / W.
// Edge k is the conic of (p_k, c_k, p_{k+1}, w_k).
inline TestSurface rationalTriangle(const std::array<mh::SpaceVector, 3>& p, const std::array<mh::SpaceVector, 3>& c,
                                    const std::array<double, 3>& w)
{
    TestSurface s;
    s.cornerCount = 3;
    s.position = [p, c, w](double xi, double eta)
    {
        const std::array<double, 3> l = mh::barycentric(xi, eta);
        double weight = 0.0;
        mh::SpaceVector n{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double corner = l[k] * l[k];
            const double edge = 2.0 * w[k] * l[k] * l[(k + 1) % 3];
            weight += corner + edge;
            for (std::size_t d = 0; d < 3; ++d)
                n[d] += corner * p[k][d] + edge * c[k][d];
        }
        return mh::SpaceVector{n[0] / weight, n[1] / weight, n[2] / weight};
    };
    const std::array<mh::Polynomial2, 3> l{
        mh::Polynomial2::linearInXi(0.0, -0.5) + mh::Polynomial2::linearInEta(0.0, -0.5),
        mh::Polynomial2::linearInXi(0.5, 0.5), mh::Polynomial2::linearInEta(0.5, 0.5)};
    mh::Polynomial2 weight;
    for (std::size_t k = 0; k < 3; ++k)
        weight = weight + l[k] * l[k] + (l[k] * l[(k + 1) % 3]).scaled(2.0 * w[k]);
    s.weight = weight;
    return s;
}

// The element axes of the reference (x along corner 0 → 1, z the normal of the diagonals, y = z × x).
inline mh::SurfaceFrame elementAxes(const std::vector<mh::SpaceVector>& corners)
{
    const mh::SpaceVector x = mh::unit3(mh::sub3(corners[1], corners[0]));
    mh::SpaceVector z{};
    if (corners.size() == 4)
        z = mh::unit3(mh::cross3(mh::sub3(corners[2], corners[0]), mh::sub3(corners[3], corners[1])));
    else
        z = mh::unit3(mh::cross3(mh::sub3(corners[1], corners[0]), mh::sub3(corners[2], corners[0])));
    return {x, mh::cross3(z, x)};
}

// The coefficients of a field f(ξ, η) (a displacement, or the surface itself) on the displacement space of an
// element: the dual functionals applied to f W (corner values, edge moments in the element's direction, interior
// projection).
inline std::vector<mh::SpaceVector> interpolate(const mh::ShellReferenceElement& re, const TestSurface& surface,
                                                const std::vector<double>& edgeSigns,
                                                const std::function<mh::SpaceVector(double, double)>& field)
{
    const std::size_t nv = re.cornerCount;
    std::vector<mh::SpaceVector> corners;
    for (std::size_t k = 0; k < nv; ++k)
    {
        const mh::ReferencePoint x = mh::referenceCorner(nv, k);
        corners.push_back(field(x[0], x[1]));
    }
    std::vector<std::vector<mh::SpaceVector>> moments;
    for (std::size_t k = 0; k < nv; ++k)
        moments.push_back(mh::edgeMapMoments(nv, re.degree, k, edgeSigns[k] > 0.0, field, surface.weight));
    const mh::InteriorProjection projection = mh::InteriorProjection::build(re);
    return mh::mapCoefficients(re, projection, corners, moments, edgeSigns, field, surface.weight);
}

// A test element: its reference element and its data (matrices built).
struct TestElement
{
    const mh::ShellReferenceElement* re = nullptr;
    TestSurface surface;
    mh::ShellElementData data;
    std::vector<mh::SpaceVector> corners;
};

// Each edge defaults to the degree of the interior.
inline TestElement makeElement(const TestSurface& surface, std::size_t degree, const mh::ShellSection& section,
                               double offset, const std::vector<double>& edgeSigns, bool withMass,
                               bool kirchhoff = false, const std::vector<std::size_t>& edgeDegrees = {})
{
    TestElement element;
    element.re = &mh::shellReferenceElement(surface.cornerCount, degree, mh::shellPointsPerDirection(degree),
                                            edgeDegrees);
    element.surface = surface;
    const mh::ShellReferenceElement& re = *element.re;
    for (std::size_t k = 0; k < surface.cornerCount; ++k)
    {
        const mh::ReferencePoint x = mh::referenceCorner(surface.cornerCount, k);
        element.corners.push_back(surface.position(x[0], x[1]));
    }
    mh::ShellElementData& d = element.data;
    d.cornerCount = surface.cornerCount;
    d.degree = degree;
    d.edgeDegrees = re.edgeDegrees;
    d.mapCoefficients = interpolate(re, surface, edgeSigns, surface.position);
    d.section = section;
    d.offset = offset;
    d.axes = elementAxes(element.corners);
    d.edgeSigns = edgeSigns;
    d.kirchhoff = kirchhoff;
    d.weight = surface.weight;
    mh::ElementJets jets = mh::elementJets(re, surface.weight);
    d.volumeJets = std::move(jets.volume);
    d.edgeJets = std::move(jets.edge);
    d.edgeWeights = std::move(jets.edgeWeight);
    auto built = mh::buildShellElement(re, d.definition(), withMass);
    if (std::holds_alternative<std::string>(built))
        throw std::runtime_error("test element refused: " + std::get<std::string>(built));
    d.matrices = std::move(std::get<mh::ShellElementMatrices>(built));
    return element;
}

// The local unknowns of a displacement field u(X) with the rotation field θ(X) (no transverse shear): the
// displacement interpolated, the edge rotations â = s_k W² θ · X_τ in Legendre coefficients of the canonical
// parameter.
using FieldOfPosition = std::function<mh::SpaceVector(const mh::SpaceVector&)>;

inline mh::Vector unknownsOfField(const TestElement& element, const FieldOfPosition& u, const FieldOfPosition& theta)
{
    const mh::ShellReferenceElement& re = *element.re;
    const mh::ShellElementDefinition def = element.data.definition();
    const std::size_t n1 = re.h1Count;
    mh::Vector x(mh::shellUnknownCount(re), 0.0);
    const std::vector<mh::SpaceVector> coefficients =
        interpolate(re, element.surface, element.data.edgeSigns,
                    [&](double xi, double eta) { return u(element.surface.position(xi, eta)); });
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            x[3 * i + c] = coefficients[i][c];
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        const double sk = element.data.edgeSigns[k];
        const std::size_t rotations = re.rotationCount(k);
        std::vector<double> moments(rotations, 0.0);
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const mh::PointGeometry g = mh::edgeGeometry(def, n1, k, gi);
            const auto tv = mh::referenceEdgePointAndDirection(re.cornerCount, k, re.edgeParameters[gi]).second;
            const mh::SpaceVector xt = mh::scl3(mh::add3(mh::scl3(g.a[0], tv[0]), mh::scl3(g.a[1], tv[1])), sk);
            const double te = sk > 0.0 ? re.edgeParameters[gi] : 1.0 - re.edgeParameters[gi];
            const double we = element.data.edgeWeights[k][gi];
            for (std::size_t l = 0; l < rotations; ++l)
                moments[l] +=
                    re.edgeWeights[gi] * we * we * mh::dot3(theta(g.x), xt) * mh::legendreValue(l, 2.0 * te - 1.0);
        }
        for (std::size_t l = 0; l < rotations; ++l)
            x[mh::shellRotationUnknown(re, k, l)] = static_cast<double>(2 * l + 1) * moments[l];
    }
    return x;
}

// The isotropic section of the checks.
inline mh::ShellSection isotropicSection(double thickness, const mh::SpaceVector& direction = {1.0, 0.0, 0.0})
{
    return mh::ShellSection::uniform(mh::Section::isotropic(70000.0, 0.3, thickness, 5.0 / 6.0, 2.7e-9, 2.3e-5),
                                     direction);
}

// An unsymmetric two-ply laminate (B ≠ 0).
inline mh::ShellSection laminateSection(const mh::SpaceVector& direction = {1.0, 0.0, 0.0})
{
    const double pi = std::numbers::pi;
    const mh::LaminateLayer layers[] = {
        {140000.0, 10000.0, 0.3, 5000.0, 5000.0, 3500.0, 1.6e-9, -0.5e-6, 30e-6, 0.03, 0.0},
        {140000.0, 10000.0, 0.3, 5000.0, 5000.0, 3500.0, 1.6e-9, -0.5e-6, 30e-6, 0.02, pi / 4.0}};
    return mh::ShellSection::uniform(mh::Section::laminate(layers), direction);
}

// max |a_ij − a_ji| / max |a_ij|.
inline double asymmetry(const mh::Matrix& a)
{
    return mh::maxAbs(a - a.transposed()) / mh::maxAbs(a);
}

// The eigenvalues of a symmetric matrix, increasing.
inline std::vector<double> sortedEigenvalues(const mh::Matrix& a)
{
    std::vector<double> values = mh::jacobiEigen(mh::symmetricPart(a)).values;
    std::sort(values.begin(), values.end());
    return values;
}

// A deterministic pseudo-random sequence in [−0.5, 0.5) (xorshift, the generator of the reference's checks).
class Xorshift
{
public:
    double next()
    {
        m_seed ^= m_seed << 13;
        m_seed ^= m_seed >> 7;
        m_seed ^= m_seed << 17;
        return static_cast<double>(m_seed >> 11) / static_cast<double>(1ULL << 53) - 0.5;
    }

private:
    unsigned long long m_seed = 0x2545F4914F6CDD1DULL;
};

} // namespace oos::checks::shell
