#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/line/LineCore.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The computations of the analysis of the mixed hybrid formulation that concern one line element (document 34 of the
// reference crate, section 8.1), as free functions on the data of that line and its local unknown vector: building a
// line alone on its edge, its kinematics and state at a fraction of its chord, its loads (the inertia of an
// acceleration, the free expansion of a temperature rise, forces and couples per length on an edge of lines alone),
// its geometric stiffness, and a rigid motion on its unknowns. The assembly into the model (the signs, the numbering
// of the unknowns, the rotation ties at the grids) is the caller's.
//
// A line on an edge of shells takes its kinematics from the shell element (the reference's shell_kin); its
// LineData is built by lineCore from those kinematics, and every function here that takes a LineData and a
// LineKinematics applies to it unchanged.
//
// Thread-safety: pure functions.

// The number of points of the rule of a line (and of the loads on an edge alone), 2k + 4: the integrands of a uniform
// line alone are polynomials of degree at most 2k, integrated exactly; those of a tapered section (the compliance
// 1 / E I(s)) or of a curved shell edge are rational, on which the rule converges fast.
[[nodiscard]] constexpr std::size_t linePointCount(std::size_t degree)
{
    return 2 * degree + 4;
}

// The temperature rise of a load case at a point of the axis of a line (the thermal field of the case for that line;
// the reference asks it with the index number of shells + number of the line). An empty function: no thermal field.
using LineTemperatureField = std::function<double(const Vec3Of<double>&)>;

// Forces and couples per length on an edge, (f_x, f_y, f_z, m_x, m_y, m_z) in global components, as a function of the
// fraction of the edge from the grid the load is given from.
using EdgeTractionField = std::function<std::array<double, 6>(double)>;

// The edge of a line that no shell has: its frame is that of the section axes of `line` along it (the first line of
// the edge), its grids `edgeGrids` (lower first) at `lowerPoint` and `upperPoint`. Returns the message of a problem
// instead: the two grids at the same point, the section axes.
// Reference: src/tdnns/analysis.rs::Analysis::new, the loop of the edges that no shell has
[[nodiscard]] std::variant<AloneEdge, std::string> aloneEdgeOfLine(const LineElement& line, const LineSection& section,
                                                                   const std::array<std::size_t, 2>& edgeGrids,
                                                                   const Vec3Of<double>& lowerPoint,
                                                                   const Vec3Of<double>& upperPoint, std::size_t degree,
                                                                   std::size_t firstEdgeUnknown);

// A line element alone on `edge` (number `edgeNumber` in the caller's topology): the unknowns of the edge, moments
// of the degree of the edge, shear flexibility, `pointCount` points (linePointCount). `line.grids` and `edge.grids`
// are in the same numbering; the line runs forward when its grid A is the lower grid of the edge. `rotationDofs`:
// the first of the three rotation unknowns of its grids A and B. Returns the message of a problem instead.
// Reference: src/tdnns/analysis.rs::Analysis::new, the line elements (a line alone)
[[nodiscard]] std::variant<LineData, std::string> aloneLineData(const AloneEdge& edge, std::size_t edgeNumber,
                                                                const LineElement& line, const LineSection& section,
                                                                const std::array<std::size_t, 2>& rotationDofs,
                                                                std::size_t pointCount, bool withMass);

// The kinematics of a line alone at the fraction `fraction` of its chord from its grid A; `forward`: its grid A is
// the lower grid of the edge.
// Reference: src/tdnns/analysis.rs::Analysis::line_kin (LineSource::Alone)
[[nodiscard]] LineKinematics aloneLineKinematics(const AloneEdge& edge, bool forward, double fraction);

// The free axial strains α ΔT of a line at the points of its rule (at its axis points) for a thermal field; empty
// when there is no field or the section has no expansion.
// Reference: src/tdnns/analysis.rs::Analysis::line_free_strains
[[nodiscard]] std::vector<double> lineFreeStrains(const LineData& data, const LineSection& section,
                                                  const LineTemperatureField& temperature);

// The state of a line at the point of `kinematics` for the local unknowns `x` of a load case of thermal field
// `temperature` (possibly empty): forces of the section about its axis and strains at the axis.
// Reference: src/tdnns/analysis.rs::Analysis::line_state
[[nodiscard]] std::variant<LineState, std::string> lineStateOfCase(const LineData& data,
                                                                   const LineKinematics& kinematics,
                                                                   const LineElement& line, const LineSection& section,
                                                                   std::span<const double> x,
                                                                   const LineTemperatureField& temperature);

// The axial stress at the point (y, z) of the section from the reference axis: E (Γ_x + z K_y − y K_z − e0).
// Reference: src/tdnns/analysis.rs::Analysis::line_stress
[[nodiscard]] double lineStress(const LineSection& section, const LineState& state, double y, double z);

// The load of a uniform acceleration a on the local unknowns: ∫ (u + θ × o)ᵀ m a μ ds.
// Reference: src/tdnns/analysis.rs::Analysis::new, the loads of the lines (inertia)
[[nodiscard]] Vector lineInertiaLoad(const LineData& data, const Vec3Of<double>& acceleration);

// The load of the free expansion of a temperature rise on the local unknowns (zero without a field or an expansion).
// Reference: src/tdnns/analysis.rs::Analysis::new, the loads of the lines (thermal)
[[nodiscard]] Vector lineThermalLoad(const LineData& data, const LineSection& section,
                                     const LineTemperatureField& temperature);

// The load of forces and couples per length on an edge of lines alone, on the n_x unknowns of the edge: the force
// works on the displacement u, the couple on the rotation θ, ∫ (uᵀ f + θᵀ m) ds by the rule of `pointCount` points.
// `fromLowerGrid`: the field is given from the lower grid of the edge.
// Reference: src/tdnns/analysis.rs::Analysis::edge_traction (an edge of lines alone)
[[nodiscard]] Vector aloneEdgeTraction(const AloneEdge& edge, bool fromLowerGrid, const EdgeTractionField& traction,
                                       std::size_t pointCount);

// The geometric stiffness of a line for the local unknowns `x` of a solution with the free strains `freeStrains` at
// the points of its rule (empty without): ∫ N du⊥ᵀ du⊥ ds, N = Ā (Γ_x − e0) + (D⁻¹ Bᵀ)ᵀ M_s the axial force with
// its free strain, the moments less their free part.
// Reference: src/tdnns/analysis.rs::Analysis::geometric_stiffness (the lines)
[[nodiscard]] Matrix lineGeometricStiffness(const LineData& data, std::span<const double> x,
                                            std::span<const double> freeStrains);

// The axial force N at each point of the rule of a line (the factor of its geometric stiffness).
[[nodiscard]] Vector lineAxialForces(const LineData& data, std::span<const double> x,
                                     std::span<const double> freeStrains);

// The local unknowns of a line alone (n_x + 6) in the rigid motion u = c + ω × X, θ = ω: the grid translations, no
// function of the edge, the twist coefficient 0 equal to ω · (X_upper − X_lower), Λ = ω at both grids.
// Reference: src/tdnns/analysis.rs::Analysis::rigid_modes (the grids and the edges of lines alone)
[[nodiscard]] Vector aloneLineRigidMotion(const AloneEdge& edge, const Vec3Of<double>& translation,
                                          const Vec3Of<double>& rotation);

} // namespace oos::elements::mixedHybrid
