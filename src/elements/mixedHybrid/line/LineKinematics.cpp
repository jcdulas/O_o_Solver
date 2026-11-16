#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"

#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

#include <cmath>

namespace oos::elements::mixedHybrid
{

namespace
{

// A vector is taken as along the line when its part across the line is below this fraction of its length.
constexpr double ACROSS_TOLERANCE = 1e-8;

// Column `column` of a 3-row matrix as a vector.
Vec3Of<double> columnOf(const Matrix& rows, std::size_t column)
{
    return {rows(0, column), rows(1, column), rows(2, column)};
}

void setColumn(Matrix& rows, std::size_t column, const Vec3Of<double>& v)
{
    for (std::size_t r = 0; r < 3; ++r)
        rows(r, column) = v[r];
}

} // namespace

std::pair<double, double> legendreWithDerivative(std::size_t n, double x)
{
    if (n == 0)
        return {1.0, 0.0};
    double p0 = 1.0;
    double p1 = x;
    double d0 = 0.0;
    double d1 = 1.0;
    for (std::size_t k = 2; k <= n; ++k)
    {
        const double kf = static_cast<double>(k);
        const double p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        const double d2 = ((2.0 * kf - 1.0) * (p1 + x * d1) - (kf - 1.0) * d0) / kf;
        p0 = p1;
        p1 = p2;
        d0 = d1;
        d1 = d2;
    }
    return {p1, d1};
}

std::variant<SectionAxes, std::string> sectionAxes(const LineElement& line, const LineSection& section,
                                                   const Vec3Of<double>& tangent,
                                                   const std::optional<Vec3Of<double>>& normal)
{
    // The unit part of v across the line, or nothing when v is along it.
    const auto across = [&](const Vec3Of<double>& v) -> std::optional<Vec3Of<double>>
    {
        const Vec3Of<double> w = sub3(v, scl3(tangent, dot3(v, tangent)));
        if (norm3(w) > ACROSS_TOLERANCE * norm3(v))
            return unit3(w);
        return std::nullopt;
    };
    Vec3Of<double> y{};
    if (line.orientation)
    {
        const std::optional<Vec3Of<double>> v = across(*line.orientation);
        if (!v)
            return std::string("the orientation vector is along the line");
        y = *v;
    }
    else if (normal)
    {
        const std::optional<Vec3Of<double>> v = across(*normal);
        if (!v)
            return std::string("the normal of the shell is along the line");
        y = *v;
    }
    else
    {
        if (section.bending)
            return std::string("a bar or a beam on an edge that no shell has needs its orientation vector");
        // A rod has no bending: any axis across the line will do; the global axis least aligned with the line (the
        // first of equals).
        std::size_t least = 0;
        for (std::size_t c = 1; c < 3; ++c)
            if (std::fabs(tangent[c]) < std::fabs(tangent[least]))
                least = c;
        Vec3Of<double> e{};
        e[least] = 1.0;
        y = *across(e);
    }
    return SectionAxes{tangent, y, cross3(tangent, y)};
}

AloneEdge AloneEdge::make(const std::array<std::size_t, 2>& grids, const Vec3Of<double>& lowerPoint,
                          const Vec3Of<double>& upperPoint, const SectionAxes& frame, std::size_t degree,
                          std::size_t firstEdgeUnknown)
{
    AloneEdge edge;
    edge.grids = grids;
    edge.lowerPoint = lowerPoint;
    edge.upperPoint = upperPoint;
    edge.frame = frame;
    edge.degree = degree;
    const std::size_t k = degree;
    for (std::size_t g = 0; g < 2; ++g)
        for (std::size_t c = 0; c < 3; ++c)
            edge.dofs.push_back(6 * grids[g] + c);
    for (std::size_t i = 0; i < 3 * (k - 1); ++i)
        edge.dofs.push_back(firstEdgeUnknown + i);
    // The k unknowns of the shear along the edge come between the moments and the twist; a line does not use them.
    for (std::size_t l = 0; l < k; ++l)
        edge.dofs.push_back(firstEdgeUnknown + 3 * (k - 1) + k + l);
    return edge;
}

double AloneEdge::length() const
{
    return norm3(sub3(upperPoint, lowerPoint));
}

EdgeFunctions AloneEdge::functions(double s) const
{
    EdgeFunctions out;
    out.values = {1.0 - s, s};
    out.derivatives = {-1.0, 1.0};
    for (std::size_t l = 0; l + 1 < degree; ++l)
    {
        const auto [v, dv] = legendreWithDerivative(l, 2.0 * s - 1.0);
        out.values.push_back(s * (1.0 - s) * v);
        // d/ds [s (1 − s) P_l(2s − 1)] = (1 − 2s) P_l + 2 s (1 − s) P'_l
        out.derivatives.push_back((1.0 - 2.0 * s) * v + 2.0 * s * (1.0 - s) * dv);
    }
    return out;
}

Vec3Of<double> AloneEdge::point(double s) const
{
    return add3(scl3(lowerPoint, 1.0 - s), scl3(upperPoint, s));
}

LineKinematics AloneEdge::kinematics(double s, bool forward) const
{
    const std::size_t k = degree;
    const std::size_t n = unknownCount();
    const double len = length();
    const Vec3Of<double> tangent = scl3(sub3(upperPoint, lowerPoint), 1.0 / len);
    const EdgeFunctions f = functions(s);
    Matrix u(3, n);
    Matrix du(3, n);
    // The grids: (1 − s) and s on each component.
    for (std::size_t c = 0; c < 3; ++c)
    {
        u(c, c) = f.values[0];
        u(c, 3 + c) = f.values[1];
        du(c, c) = f.derivatives[0] / len;
        du(c, 3 + c) = f.derivatives[1] / len;
    }
    // The functions of the edge along the axes of its frame.
    for (std::size_t l = 0; l + 1 < k; ++l)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t column = 6 + 3 * l + axis;
            const Vec3Of<double>& e = frame[axis];
            for (std::size_t r = 0; r < 3; ++r)
            {
                u(r, column) = f.values[2 + l] * e[r];
                du(r, column) = f.derivatives[2 + l] / len * e[r];
            }
        }
    }
    // θ = t × du/ds on the displacement, ψ t on the twist.
    Matrix theta(3, n);
    for (std::size_t column = 0; column < 6 + 3 * (k - 1); ++column)
        setColumn(theta, column, cross3(tangent, columnOf(du, column)));
    for (std::size_t l = 0; l < k; ++l)
    {
        const double v = legendreValue(l, 2.0 * s - 1.0) / len;
        for (std::size_t r = 0; r < 3; ++r)
            theta(r, 6 + 3 * (k - 1) + l) = v * tangent[r];
    }
    // A line that runs from the upper grid: its tangent and du/ds change sign, θ = t × du/ds does not.
    const double sign = forward ? 1.0 : -1.0;
    LineKinematics kin;
    kin.point = point(s);
    kin.fraction = forward ? s : 1.0 - s;
    kin.tangent = scl3(tangent, sign);
    kin.fractionRate = 1.0 / len;
    kin.normal = std::nullopt;
    kin.displacement = std::move(u);
    kin.displacementRate = std::move(du) * sign;
    kin.rotation = std::move(theta);
    return kin;
}

std::vector<LineUnknown> aloneLineUnknowns(std::size_t degree)
{
    const std::size_t k = degree;
    std::vector<LineUnknown> out;
    for (std::size_t end = 0; end < 2; ++end)
        for (std::size_t c = 0; c < 3; ++c)
            out.push_back({LineUnknownKind::GridTranslation, end, 0, c, 0});
    for (std::size_t l = 0; l + 1 < k; ++l)
        for (std::size_t c = 0; c < 3; ++c)
            out.push_back({LineUnknownKind::EdgeFunction, 0, l, c, 3 * l + c});
    for (std::size_t l = 0; l < k; ++l)
        out.push_back({LineUnknownKind::EdgeTwist, 0, l, 0, 3 * (k - 1) + k + l});
    for (std::size_t end = 0; end < 2; ++end)
        for (std::size_t c = 0; c < 3; ++c)
            out.push_back({LineUnknownKind::GridRotation, end, 0, c, 0});
    return out;
}

} // namespace oos::elements::mixedHybrid
