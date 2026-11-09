#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/aids/LinearityGeometry.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/ResultArrays.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace oos::design
{

// The relative error of the linear answer at which the two thresholds below are set: a
// linear analysis whose answer is more than 5 % off the geometrically non-linear one is not trusted.
inline constexpr double LINEARITY_RELATIVE_ERROR = 0.05;

// The deflection over thickness above which a plate leaves the linear domain.
// Source: Timoshenko and Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., 1959, §97, the clamped circular
// plate under a uniform pressure q with large deflections (ν = 0.3; within a few percent of S. Way's series solution,
// §98):
//     w0/h + 0.488·(w0/h)³ = q·a⁴/(64·D·h),
// the right-hand side being the linear answer w_lin/h. The linear answer exceeds the non-linear deflection w0 by
// 0.488·(w0/h)², which reaches LINEARITY_RELATIVE_ERROR at w0/h = √(0.05/0.488) = 0.3201, where the linear answer
// reads w_lin/h = 0.3201·1.05 = 0.3361. A plate with edges held in its plane is the case where the membrane action
// stiffens it soonest; a cantilever plate is stiffened later (its rotation decides, below), so the ratio is a
// conservative indicator for it.
inline constexpr double LINEARITY_DEFLECTION_RATIO = 0.336;

// The rotation, in radians, above which a structure leaves the linear domain.
// Source: a strip of length L bent by an end moment M; the exact (geometrically non-linear) deformed shape is a
// circular arc of curvature κ = M/(EI) (the elastica of a constant moment: Timoshenko and Gere, Theory of Elastic
// Stability, 2nd ed., 1961, ch. 2), whose end rotates by θ = κL, as in the linear answer, and moves by
//     v = (1 − cos θ)/κ across and u = L − sin θ/κ along the strip,
// where the linear answer gives v_lin = θL/2 and u_lin = 0. The relative error of the end displacement,
// |(v_lin − v, u)| / |(v, u)| ≈ θ/3, reaches LINEARITY_RELATIVE_ERROR at θ = 0.1500 rad (8.6°) (computed by
// bisection).
inline constexpr double LINEARITY_ROTATION = 0.150;

// One shell element as the indicator sees it: its corners, unit normal, area and mean thickness.
struct ShellElementGeometry
{
    std::string card;
    std::int32_t elementId = 0;
    std::vector<std::int32_t> cornerRows; // node table rows of its corners (3 or 4)
    base::Vec3 normal;                    // unit, in the basic system
    double area = 0.0;
    double thickness = 0.0; // its volume over its area: the property's T, the corner thicknesses, a laminate's Σ t_k
};

// One grid point of a shell (a corner of at least one shell element): the normal of the shell there and the thickness
// of the elements around it.
struct ShellPointGeometry
{
    std::int32_t grid = 0;
    std::int32_t row = 0;              // its node table row
    base::Vec3 normal;                 // unit, the mean of the normals of its elements (oriented as the first one)
    double thickness = 0.0;            // the mean thickness of its elements
    base::Mat3 displacementFrame;      // CD to basic: u_basic = R·u_CD
    std::vector<std::size_t> elements; // indices into LinearityIndicator::shellElements()
};

// The indicators of one subcase: per shell grid point the deflection over thickness, per grid point the rotation, and
// the largest of each with where it is.
struct LinearitySubcase
{
    std::int32_t subcase = 0;
    std::vector<std::int32_t> shellGrids;
    std::vector<double> deflectionRatios;
    std::vector<std::int32_t> grids;
    std::vector<double> rotations;
    double largestDeflectionRatio = 0.0;
    std::int32_t deflectionGrid = 0; // 0: no shell grid point
    double largestRotation = 0.0;
    std::int32_t rotationGrid = 0; // 0: no grid point
};

// Whether the geometry of a linearity report was kept from an earlier design point, and its bytes.
struct LinearityGeometryUse
{
    bool isReused = false;
    std::uint64_t bytes = 0;
};

// The linearity report of an analysis (the validity of the linear analysis): every subcase, and a warning
// for each subcase whose largest value is above its threshold.
struct LinearityReport
{
    std::vector<LinearitySubcase> subcases;
    std::vector<std::string> warnings;
    double deflectionLimit = LINEARITY_DEFLECTION_RATIO;
    double rotationLimit = LINEARITY_ROTATION;
    LinearityGeometryUse geometry;

    [[nodiscard]] bool isWithinLimits() const noexcept { return warnings.empty(); }
};

// The validity of the linear analysis (an optimisation aid beyond gradients): from
// the displacements alone, no solve,
//   - per shell grid point the normal displacement over the thickness of the elements around it, |n·u|/t̄, with n the
//     mean unit normal of the shell elements that have the grid point as a corner (each oriented as the first one, so
//     that a mesh numbered both ways keeps its normal) and t̄ the mean of their thicknesses;
//   - per grid point the magnitude of its rotation vector |θ| = (θ1² + θ2² + θ3²)^½, invariant of the frame;
// and a warning above LINEARITY_DEFLECTION_RATIO and LINEARITY_ROTATION. The same quantities are the response kinds
// deflection_to_thickness and rotation (DesignResp1Response), with their gradients (StaticSensitivity).
//
// Shells: those of LinearityGeometry, which gives the normals and areas; the thickness of an element is its volume
// (assembly::ElementVolumes: the mass with every density 1, so T·A, ∫t dA with corner thicknesses, Σt_k·A for a
// laminate, computed on the pool) over its area. An element whose area or volume is not positive is left out.
//
// Thread-safety: immutable once built. Ownership: owns its lists and shares its geometry (they grow with the shell
// elements and their grid points: their memory is taken from the budget given to the constructor, which must outlive
// the indicator); keeps no reference to the snapshot.
class LinearityIndicator
{
public:
    // The shell geometry of `snapshot` (an overlaid one: the designed thicknesses and positions are read), the
    // volumes on `pool`. `keepAlive` owns the snapshot. Throws MemoryBudgetError.
    LinearityIndicator(const input::ModelSnapshot& snapshot, const std::shared_ptr<const void>& keepAlive,
                       base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    // The same on a geometry built from `snapshot` or from a snapshot of which `snapshot` is an overlay that moves no
    // grid point (a design point without shape variables: the geometry a DesignProblem keeps); the
    // thicknesses are those of `snapshot`. The same bits as the constructor above.
    LinearityIndicator(std::shared_ptr<const LinearityGeometry> geometry, const input::ModelSnapshot& snapshot,
                       const std::shared_ptr<const void>& keepAlive, base::MemoryBudget& budget,
                       base::ThreadPool* pool = nullptr);

    // Whether a card is one of the shells the indicator reads.
    [[nodiscard]] static bool isShellCard(std::string_view card) noexcept;
    // The IDs of the grid points that are corners of a shell element of the snapshot, increasing (no volume needed:
    // what a response declaration is checked against).
    [[nodiscard]] static std::vector<std::int32_t> shellGridIds(const input::ModelSnapshot& snapshot);

    [[nodiscard]] const std::vector<ShellElementGeometry>& shellElements() const noexcept { return m_elements; }
    // The shell grid points, by increasing ID.
    [[nodiscard]] const std::vector<ShellPointGeometry>& shellPoints() const noexcept { return m_points; }
    // The shell grid point of an ID, or nullptr.
    [[nodiscard]] const ShellPointGeometry* findShellPoint(std::int32_t grid) const;

    // |n·R·t| / t̄ for the translations t of the grid point in its displacement system CD.
    [[nodiscard]] static double deflectionRatio(const ShellPointGeometry& point, std::array<double, 3> translationInCd);
    // ∂(|n·R·t| / t̄)/∂t: sign(n·R·t)·Rᵀn / t̄ (the right derivative at 0, as ABS of DesignEquation).
    [[nodiscard]] static std::array<double, 3> deflectionWeights(const ShellPointGeometry& point,
                                                                 std::array<double, 3> translationInCd);
    // |θ| for the rotations θ of a grid point (any frame).
    [[nodiscard]] static double rotationMagnitude(std::array<double, 3> rotation) noexcept;
    // ∂|θ|/∂θ = θ/|θ|; zero at θ = 0 (the smallest subgradient).
    [[nodiscard]] static std::array<double, 3> rotationWeights(std::array<double, 3> rotation) noexcept;

    // The report of the displacement tables of `arrays` (kind "displacements") of `subcases`: every grid point the
    // tables hold. A subcase without a displacement table is left out.
    [[nodiscard]] LinearityReport report(const output::ResultArrays& arrays,
                                         std::span<const std::int32_t> subcases) const;

private:
    base::MemoryReservation m_reservation; // the memory of the lists below, estimated before they are built
    std::shared_ptr<const LinearityGeometry> m_geometry;
    std::vector<ShellElementGeometry> m_elements;
    std::vector<ShellPointGeometry> m_points;
    std::unordered_map<std::int32_t, std::size_t> m_pointOfGrid;
};

} // namespace oos::design
