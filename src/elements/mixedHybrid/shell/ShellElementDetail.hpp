#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// The pieces of the shell element that its linear, load, stress and nonlinear parts share (private to this folder).
//
// Thread-safety: pure functions.
namespace oos::elements::mixedHybrid::shelldetail
{

// The covariant pairs (α, β) of the Voigt components 11, 22, 12.
inline constexpr std::size_t VOIGT_PAIRS[3][2] = {{0, 0}, {1, 1}, {0, 1}};

// The geometry at an edge point with the tangent of the map along the edge and the reference edge vector:
// X_τ = a_α t̂^α, t̂ = corner k + 1 − corner k.
struct EdgePointGeometry
{
    PointGeometry g;
    SpaceVector xt{};
    std::array<double, 2> tv{};
};

// The geometry at every point of every edge [k][g].
[[nodiscard]] std::vector<std::vector<EdgePointGeometry>> edgeGeometries(const ShellReferenceElement& re,
                                                                         const ShellElementDefinition& def);

// The geometry at every surface point.
[[nodiscard]] std::vector<PointGeometry> volumeGeometries(const ShellReferenceElement& re,
                                                          const ShellElementDefinition& def);

// The normal moment m_μμ of every moment function of the element at an edge point, μ the unit conormal: the reference
// functions by the Piola map, S^αβ (μ · a_α)(μ · a_β) / j², the three constants of the frame of the constant states by
// their local components (unless `mapReferenceConstants`).
[[nodiscard]] std::vector<double> normalMoments(const ShellReferenceElement& re, std::size_t k, std::size_t g,
                                                const PointGeometry& geometry, const SpaceVector& conormal,
                                                bool mapReferenceConstants);

// target += s aᵀ m a, a of 3 rows.
void addCongruence(Matrix& target, const Matrix& a, const Matrix33& m, double s);
// target += s aᵀ m a, a of 2 rows.
void addCongruence2(Matrix& target, const Matrix& a, const Matrix22& m, double s);
// The same for an `a` whose columns outside [firstColumn, firstColumn + columnCount) are zero: only that block of the
// target is formed (the other terms of the product are zero), with the same bits there.
void addCongruence2(Matrix& target, const Matrix& a, const Matrix22& m, double s, std::size_t firstColumn,
                    std::size_t columnCount);
// target += s aᵀ b.
void addTransposeProduct(Matrix& target, const Matrix& a, const Matrix& b, double s);
// The same for a `b` whose columns from `columnCount` on are zero: those columns of the target are left as they are.
void addTransposeProduct(Matrix& target, const Matrix& a, const Matrix& b, double s, std::size_t columnCount);
// target += s aᵀ m b, a and b of 3 rows.
void addTransposeProduct33(Matrix& target, const Matrix& a, const Matrix33& m, const Matrix& b, double s);
// (A + Aᵀ) / 2 in place.
void symmetrise(Matrix& a);

// The column of the 3 × n matrix a times v: a · v.
[[nodiscard]] std::array<double, 3> times3(const Matrix& a, std::span<const double> v);

// The Legendre polynomial P_l at 2τ − 1 (the edge moments use the parameter τ ∈ [0, 1]).
[[nodiscard]] double edgeLegendre(std::size_t l, double tau);

} // namespace oos::elements::mixedHybrid::shelldetail
