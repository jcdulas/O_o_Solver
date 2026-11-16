#pragma once

#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/line/LineCore.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Helpers shared by the checks of the line elements of the mixed hybrid formulation (checks/mixedHybrid/Line*.cpp).
namespace lineChecks
{

namespace mh = oos::elements::mixedHybrid;
using Vec3 = mh::Vec3Of<double>;

// A line alone on its edge, built as the analysis of the reference builds it.
struct AloneLine
{
    mh::AloneEdge edge;
    mh::LineData data;
    mh::LineElement line;
    mh::LineSection section;
    bool forward = true;
};

// The first unknown of the edge in the reference numbering of a model of two grids (six slots per grid).
inline constexpr std::size_t FIRST_EDGE_UNKNOWN = 12;

// The line `line` (its grids 0 and 1, in either order) on the edge from grid 0 at `x0` to grid 1 at `x1`, of degree k.
inline std::optional<AloneLine> buildAloneLine(oos::checks::CheckRunner& runner, const Vec3& x0, const Vec3& x1,
                                               const mh::LineElement& line, const mh::LineSection& section,
                                               std::size_t degree, bool withMass = true)
{
    auto edge = mh::aloneEdgeOfLine(line, section, {0, 1}, x0, x1, degree, FIRST_EDGE_UNKNOWN);
    if (const std::string* problem = std::get_if<std::string>(&edge))
    {
        runner.expect(false, "edge of the line: " + *problem);
        return std::nullopt;
    }
    const mh::AloneEdge& e = std::get<mh::AloneEdge>(edge);
    auto data = mh::aloneLineData(e, 0, line, section, {6 * line.grids[0] + 3, 6 * line.grids[1] + 3},
                                  mh::linePointCount(degree), withMass);
    if (const std::string* problem = std::get_if<std::string>(&data))
    {
        runner.expect(false, "line data: " + *problem);
        return std::nullopt;
    }
    return AloneLine{e, std::get<mh::LineData>(data), line, section, e.grids[0] == line.grids[0]};
}

// |v − e| ≤ tolerance · |e| (|v| ≤ tolerance when e is zero).
inline void expectRelative(oos::checks::CheckRunner& runner, double value, double expected, double tolerance,
                           const std::string& what)
{
    const double scale = expected == 0.0 ? 1.0 : std::fabs(expected);
    const bool close = std::fabs(value - expected) <= tolerance * scale;
    char buffer[160];
    std::snprintf(buffer, sizeof buffer, ": %.17g expected %.17g (relative tolerance %.1e)", value, expected,
                  tolerance);
    runner.expect(close, what + buffer);
}

// A diagonal term below this fraction of the largest is round-off (an unknown without stiffness whose rows hold the
// round-off of a projection, as the functions across a rod): it is scaled as the largest, not to one.
inline constexpr double ROUND_OFF_DIAGONAL = 1e-14;

// The absolute eigenvalues of a symmetric positive semi-definite matrix, increasing, over the largest, after the
// symmetric scaling to a unit diagonal (s_i = 1 / √k_ii). The scaling is a congruence, which keeps the number of zero
// eigenvalues (Sylvester's law of inertia), and it brings the stiffnesses of very different scales (axial, shear,
// bending) to one.
inline std::vector<double> scaledSpectrum(const mh::Matrix& k)
{
    const std::size_t n = k.rows();
    double largestDiagonal = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        largestDiagonal = std::max(largestDiagonal, k(i, i));
    std::vector<double> scales(n, 1.0 / std::sqrt(largestDiagonal));
    for (std::size_t i = 0; i < n; ++i)
        if (k(i, i) > ROUND_OFF_DIAGONAL * largestDiagonal)
            scales[i] = 1.0 / std::sqrt(k(i, i));
    mh::Matrix scaled(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            scaled(i, j) = scales[i] * k(i, j) * scales[j];
    std::vector<double> values = mh::jacobiEigen(scaled).values;
    for (double& v : values)
        v = std::fabs(v);
    std::sort(values.begin(), values.end());
    const double largest = values.back();
    for (double& v : values)
        v /= largest;
    return values;
}

// The number of zero eigenvalues: those of scaledSpectrum below `tolerance`.
inline std::size_t zeroEigenvalueCount(const mh::Matrix& k, double tolerance)
{
    const std::vector<double> spectrum = scaledSpectrum(k);
    return static_cast<std::size_t>(
        std::count_if(spectrum.begin(), spectrum.end(), [tolerance](double v) { return v <= tolerance; }));
}

// K x = f with the unknowns `held` at zero, by LU on the others; empty when the reduced matrix is singular.
inline std::optional<mh::Vector> solveHeld(const mh::Matrix& k, const mh::Vector& f,
                                           const std::vector<std::size_t>& held)
{
    const std::size_t n = k.rows();
    std::vector<std::size_t> freeUnknowns;
    for (std::size_t i = 0; i < n; ++i)
        if (std::find(held.begin(), held.end(), i) == held.end())
            freeUnknowns.push_back(i);
    const std::size_t m = freeUnknowns.size();
    mh::Matrix reduced(m, m);
    mh::Vector rhs(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        rhs[i] = f[freeUnknowns[i]];
        for (std::size_t j = 0; j < m; ++j)
            reduced(i, j) = k(freeUnknowns[i], freeUnknowns[j]);
    }
    const mh::LuDecomposition lu(reduced);
    const std::optional<mh::Vector> solution = lu.solve(rhs);
    if (!solution)
        return std::nullopt;
    mh::Vector x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        x[freeUnknowns[i]] = (*solution)[i];
    return x;
}

// The section law at the edge on (Γ_x, Γ_y, Γ_z, K_x, K_y, K_z) for an axis offset by o (section axes, constant):
// TᵀDT with T the identity but for the axial strain of the axis, Γ_x + (K × o) · x = Γ_x + o_z K_y − o_y K_z
// (document 34 of the reference crate, section 8.1: the shear and the torsion stay those of the edge).
inline mh::Matrix lawAtEdge(const mh::LineSectionPoint& point, const Vec3& offsetInAxes)
{
    mh::Matrix d(6, 6);
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            d(r, c) = point.stiffness[r][c];
    mh::Matrix t = mh::Matrix::identity(6);
    t(0, 4) = offsetInAxes[2];
    t(0, 5) = -offsetInAxes[1];
    return mh::transposeTimes(t, d * t);
}

// The tip flexibility of a uniform cantilever of length L on the edge, clamped at s = 0, for a load (F, M) at the tip
// in section axes: by the unit load method, q = ∫_0^L Pᵀ C P ds Q, with C = D⁻¹ the section compliance and P(s) the
// section forces of a unit tip load, (F, M + (L − s) x × F) (the Timoshenko beam: S. P. Timoshenko, J. M. Gere,
// Mechanics of Materials, 1972). With P = P0 + (L − s) P1,
//   ∫ Pᵀ C P ds = L P0ᵀCP0 + L²/2 (P0ᵀCP1 + P1ᵀCP0) + L³/3 P1ᵀCP1.
inline std::optional<mh::Matrix> cantileverFlexibility(const mh::Matrix& law, double length)
{
    const std::optional<mh::Matrix> compliance = mh::inverse(law);
    if (!compliance)
        return std::nullopt;
    const mh::Matrix p0 = mh::Matrix::identity(6);
    // x × F: (0, −F_z, F_y) on the moments.
    mh::Matrix p1(6, 6);
    p1(4, 2) = -1.0;
    p1(5, 1) = 1.0;
    const mh::Matrix& c = *compliance;
    const mh::Matrix cross = mh::transposeTimes(p0, c * p1);
    return c * length + (cross + cross.transposed()) * (0.5 * length * length) +
           mh::transposeTimes(p1, c * p1) * (length * length * length / 3.0);
}

// R v: the components of v in the axes (rows of R).
inline Vec3 toAxes(const mh::SectionAxes& axes, const Vec3& v)
{
    return {mh::dot3(axes[0], v), mh::dot3(axes[1], v), mh::dot3(axes[2], v)};
}

// Rᵀ v: the global components of v given in the axes.
inline Vec3 fromAxes(const mh::SectionAxes& axes, const Vec3& v)
{
    return mh::add3(mh::add3(mh::scl3(axes[0], v[0]), mh::scl3(axes[1], v[1])), mh::scl3(axes[2], v[2]));
}

} // namespace lineChecks
