#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The kinematics of a line element of the mixed hybrid formulation (document 34 of the reference crate, section 8.1):
// at a point of its edge, the displacement u of the edge, its derivative du/ds along the line and the rotation θ, each
// as three rows (global components) on the unknowns that carry them.
//
// - A line on an edge of shells has no displacement of its own: u is the trace of the first shell of the edge, its
//   twist the rotation of the shell about the edge (the hybrid unknown a_T of the edge), and it bends with the slope
//   of the trace, θ = a_T t + t × du/ds. That kinematics is built from the shell element (the reference's shell_kin,
//   in the shell folder); the line kernel takes it as a list of LineKinematics.
// - A line alone on its edge (no shell has the edge) carries the unknowns of the edge itself (AloneEdge): u is the
//   polynomial of degree k, linear between the translations of the two grids plus k − 1 functions
//   s (1 − s) P_l(2s − 1) that vanish at the grids (three components each, in the axes of the edge); its twist ψ is
//   the polynomial of degree p = k − 1 of the twist unknowns of the edge, and θ = ψ t + t × du/ds. The edge is
//   straight.
//
// The local unknowns of a line alone, in this order (n_x = 6 + 3 (k − 1) + k of the edge, then 6 of the line):
//   0 … 2              translations x, y, z (global) of the lower grid of the edge (AloneEdge::grids[0]);
//   3 … 5              translations x, y, z of the upper grid of the edge (AloneEdge::grids[1]);
//   6 + 3 l + c        function l (l < k − 1) of the displacement along axis c of AloneEdge::frame;
//   6 + 3 (k − 1) + l  twist coefficient l (l < k): θ · t = Σ_l a_l P_l(2s − 1) / L, s from the lower grid;
//   n_x + 0 … 2        rotation Λ (global) of the lines at grid A of the line (the rotation slots of the grid);
//   n_x + 3 … 5        rotation Λ at grid B of the line.
// The translations follow the edge (lower grid first), the rotations Λ follow the line (A first): a line that runs
// from its upper grid has its A at grids[1]. aloneLineUnknowns() describes each unknown.
//
// Thread-safety: plain value types; the functions are pure; an AloneEdge is immutable once built.

// The section axes (x, y, z) at a point of a line, each a unit vector in global components.
using SectionAxes = std::array<Vec3Of<double>, 3>;

// The kinematics of a line at one point: rows on the unknowns that carry it.
// Reference: src/tdnns/line.rs::LineKin
struct LineKinematics
{
    // The point of the edge (the line of the grids).
    Vec3Of<double> point{};
    // The fraction λ of the line from its grid A (reference field: lam).
    double fraction = 0.0;
    // The unit tangent, from A to B.
    Vec3Of<double> tangent{};
    // dλ/ds (reference field: dlam).
    double fractionRate = 0.0;
    // The default first section axis: the normal of the shell, where there is one.
    std::optional<Vec3Of<double>> normal;
    // The derivatives along the line of the tangent and of the normal (the offset of the axis follows them: the metric
    // of the axis). Zero on a straight edge without a shell.
    Vec3Of<double> tangentRate{};
    Vec3Of<double> normalRate{};
    // Displacement u, its derivative du/ds along the tangent, rotation θ: 3 rows (global components) × the unknowns.
    Matrix displacement;
    Matrix displacementRate;
    Matrix rotation;
};

// A kinematics and the length ds of the axis of the edge it stands for (the weight of its quadrature point times
// ds/dλ).
struct WeightedLineKinematics
{
    LineKinematics kinematics;
    double weight = 0.0;
};

// The Legendre polynomial P_n and its derivative at x, by the recurrences
//   n P_n = (2n − 1) x P_{n−1} − (n − 1) P_{n−2},   n P'_n = (2n − 1) (P_{n−1} + x P'_{n−1}) − (n − 1) P'_{n−2}
// (M. Abramowitz, I. A. Stegun, Handbook of Mathematical Functions, 1964, 22.7.10; the second is its derivative).
// Reference: src/tdnns/line.rs::legendre_d
[[nodiscard]] std::pair<double, double> legendreWithDerivative(std::size_t n, double x);

// The section axes (x along the line, y, z) of a line at a point of tangent `tangent`: y is the part of the
// orientation vector across the line, else that of the normal of the shell, else (a rod alone) the part across the
// line of the global axis least aligned with it. Returns the message of a problem instead: the orientation vector or
// the normal along the line, a bar or a beam alone without its orientation vector.
// Reference: src/tdnns/line.rs::section_axes
[[nodiscard]] std::variant<SectionAxes, std::string> sectionAxes(const LineElement& line, const LineSection& section,
                                                                 const Vec3Of<double>& tangent,
                                                                 const std::optional<Vec3Of<double>>& normal);

// Values and derivatives (d/ds of the parameter s ∈ [0, 1]) of the k + 1 displacement functions of an edge alone:
// 1 − s and s at its grids, then s (1 − s) P_l(2s − 1) for l < k − 1.
struct EdgeFunctions
{
    std::vector<double> values;
    std::vector<double> derivatives;
};

// An edge that no shell has, with line elements on it: its straight geometry and the functions of degree k of its
// displacement (the parameter s runs from the lower grid).
// Reference: src/tdnns/line.rs::AloneEdge
struct AloneEdge
{
    // Its grids, lower first, in the caller's numbering.
    std::array<std::size_t, 2> grids{};
    // The points of the lower and upper grids.
    Vec3Of<double> lowerPoint{};
    Vec3Of<double> upperPoint{};
    // The axes of the functions of the displacement along the edge: the tangent from the lower grid and the section
    // axes of the first line of the edge.
    SectionAxes frame{};
    // The degree k of the displacement.
    std::size_t degree = 0;
    // The numbers of its n_x unknowns in the reference numbering (six slots per grid; the unknowns of the edge from
    // `firstEdgeUnknown`, laid out as moments 3 (k − 1), shear along the edge k, twist k; the shear is not used by a
    // line): translations of the two grids, the functions of the edge, the twist.
    std::vector<std::size_t> dofs;

    // `firstEdgeUnknown`: the first unknown of the edge in the reference numbering.
    // Reference: src/tdnns/line.rs::AloneEdge::new
    [[nodiscard]] static AloneEdge make(const std::array<std::size_t, 2>& grids, const Vec3Of<double>& lowerPoint,
                                        const Vec3Of<double>& upperPoint, const SectionAxes& frame, std::size_t degree,
                                        std::size_t firstEdgeUnknown);

    // n_x = 6 + 3 (k − 1) + k. Reference: AloneEdge::nx
    [[nodiscard]] std::size_t unknownCount() const { return dofs.size(); }
    // Reference: AloneEdge::len
    [[nodiscard]] double length() const;
    // Reference: AloneEdge::functions
    [[nodiscard]] EdgeFunctions functions(double s) const;
    // (1 − s) X_lower + s X_upper. Reference: AloneEdge::point
    [[nodiscard]] Vec3Of<double> point(double s) const;
    // The kinematics at the parameter s of the edge (columns: the n_x unknowns of the edge); `forward`: the line runs
    // from the lower grid.
    // Reference: AloneEdge::kin
    [[nodiscard]] LineKinematics kinematics(double s, bool forward) const;
};

// Where a line element takes its kinematics.
// Reference: src/tdnns/line.rs::LineSource
struct LineSource
{
    enum class Kind
    {
        // Edge `edge` (0 … corners − 1) of shell element `element`; `forward`: along the element's direction.
        ShellEdge,
        // An edge without shells (an AloneEdge, `edge` its number in the caller's topology); `forward`: from its
        // lower grid.
        Alone,
    };

    Kind kind = Kind::Alone;
    std::size_t element = 0;
    std::size_t edge = 0;
    bool forward = true;

    [[nodiscard]] static LineSource onShellEdge(std::size_t element, std::size_t edgeOfElement, bool forward)
    {
        return {Kind::ShellEdge, element, edgeOfElement, forward};
    }
    [[nodiscard]] static LineSource alone(std::size_t edgeNumber, bool forward)
    {
        return {Kind::Alone, 0, edgeNumber, forward};
    }
};

// What a local unknown of a line alone is (the ordering of the header comment).
enum class LineUnknownKind
{
    // A translation of a grid of the edge: `end` 0 the lower grid of the edge, 1 the upper; `component` the global
    // axis.
    GridTranslation,
    // Function `index` of the displacement of the edge, along axis `component` of AloneEdge::frame.
    EdgeFunction,
    // Twist coefficient `index` of the edge.
    EdgeTwist,
    // A rotation Λ of the lines at a grid: `end` 0 grid A of the line, 1 grid B; `component` the global axis.
    GridRotation,
};

struct LineUnknown
{
    LineUnknownKind kind = LineUnknownKind::GridTranslation;
    std::size_t end = 0;
    std::size_t index = 0;
    std::size_t component = 0;
    // For an unknown of the edge: its place among the unknowns of the edge in the reference layout (moments
    // 3 (k − 1), shear k, twist k): 3 l + c for a function, 3 (k − 1) + k + l for a twist coefficient.
    std::size_t edgeSlot = 0;
};

// The n_x + 6 local unknowns of a line alone of degree k, in the order of the kernel.
[[nodiscard]] std::vector<LineUnknown> aloneLineUnknowns(std::size_t degree);

} // namespace oos::elements::mixedHybrid
