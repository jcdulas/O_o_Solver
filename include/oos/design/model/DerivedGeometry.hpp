#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/design/model/ShapeRelations.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// The tolerance of the check of each derived quantity against O_o_Reader's value at the base positions: O_o_Reader
// evaluates the same formula of the format in double precision when it reads the model, so the two agree to a few units
// of round-off (1e-15 relative); 1e-9 leaves room for another order of the operations and refuses any other rule (a
// PointForceLoad2 whose G3 and G4 O_o_Reader completes, a frame resolved otherwise), whose derivation would be wrong.
inline constexpr double DERIVED_CHECK_TOLERANCE = 1.0e-9;

// The directions and frames that O_o_Reader computes from grid point positions when it reads the model, computed again
// at a design point from the positions the shape variables move (from the grid points G1 to G4 of loadsTable and G1 to
// G3 of coordsTable, O_o_Reader data contract 44), so that a shape may move the grid points that define them. Each is
// written into the overlay of the design point beside the moved positions (FieldTargets::overlayOf), so that the
// analysis, the perturbations of the sensitivities and the distortion checks of the point all read it.
//
// What is derived again (each checked against O_o_Reader's value at the base positions, DERIVED_CHECK_TOLERANCE):
// - PointForceLoad1 and PointMomentLoad1 (the direction from G1 to G2): v = m·(G2 − G1)/|G2 − G1|;
// - PointForceLoad2 and PointMomentLoad2 (the cross product of the vectors from G1 to G2 and from G3 to G4): v =
// m·n/|n|,
//   n = (G2 − G1) × (G4 − G3);
//   m being the signed magnitude of the row (MAG times the LOAD factors), m = v₀·d₀ at the base positions;
// - RectFrame1, CylindricalFrame1, SphericalFrame1 (A = G1 the origin, B = G2 on the z axis, C = G3 in the x-z plane):
//   k = (B − A)/|B − A|, j = k × (C − A)/|k × (C − A)|, i = j × k;
// - the systems defined in a moved system (a CORD2x whose RID is moved, at any depth): their definition points are
//   written in RID, so they move rigidly with it: with A the matrix of the axes (columns i, j, k in the basic system),
//   o' = o_R' + A_R'·A_Rᵀ·(o − o_R) and A' = A_R'·A_Rᵀ·A;
// - the vectors given in a moved rectangular system, turned with it, v' = A'·Aᵀ·v: the force or moment of a FORCE or
//   MOMENT with CID (loadsTable), the acceleration of a GravityLoad (loadsTable ax, ay, az) and the direction of a
//   GravityLoad, ACCEL, AccelNodeLoad or RotationalForceLoad (inertialLoadsTable).
// The quantities that read coordsTable at analysis time (the CID, OCID and MCID of the elements, the MCID and CORDM of
// their properties, a FacePressureLoad's CID, the locations of an ACCEL, the systems of the section and cut loads) read
// the overlaid frames as they are.
//
// Refused (design_code::UNSUPPORTED_TARGET), because O_o_Reader resolves them from the systems when it reads the model
// and does not give them at a design point: a grid point located (CP) in a moved system (its position in the basic
// system follows the system), a grid point whose displacement system CD is a moved system (displacementRotations), a
// DesignNodeMove whose CID is a moved system (N taken along CID at read time), a FORCE or MOMENT in a curvilinear
// system at a grid point a shape moves or in a moved curvilinear system (its local directions turn with the point), a
// GravityLoad, ACCEL, AccelNodeLoad or RotationalForceLoad in a moved curvilinear system, the aerodynamic model when a
// system moves (its boxes are derived from the CAERO cards in their CP), and every quantity whose formula above does
// not give O_o_Reader's base value (the check). The fastener piercing points, the glued contact and the curvilinear
// displacement systems of moved grid points stay refused by FieldTargets.
//
// Thread-safety: immutable once built; addCells() may run concurrently. Ownership: owns its data (bounded by the
// grid points, systems and loads the shapes reach); keeps no reference to the snapshot.
class DerivedGeometry
{
public:
    // A grid point whose position a derived quantity reads: its base position and its move per unit of each basis
    // that moves it.
    struct Point
    {
        std::int32_t gridId = 0;
        base::Vec3 base;
        std::vector<std::pair<std::size_t, base::Vec3>> moves; // (basis, move), bases increasing
    };

    // A coordinate system the shapes move: a CORD1x from its three points, or a system defined in a moved one.
    struct Frame
    {
        std::int32_t frameId = 0;
        std::int64_t row = 0;                // in coordsTable
        bool isFromPoints = true;            // a CORD1x; otherwise defined in `parent`
        std::array<std::size_t, 3> points{}; // A, B, C: indices into the points
        std::size_t parent = 0;              // an index into the frames, below this frame's index
        bool isRectangular = true;
        base::Vec3 origin;                        // at the base positions, in the basic system
        base::Mat3 axes = base::Mat3::identity(); // columns i, j, k in the basic system, at the base positions
        std::vector<std::size_t> bases;           // the bases that move it, increasing
        double size = 0.0;                        // min(|B − A|, |C − A|) of a CORD1x at the base positions
    };

    enum class LoadForm : std::uint8_t
    {
        TwoPoints,  // PointForceLoad1, PointMomentLoad1
        FourPoints, // PointForceLoad2, PointMomentLoad2
        InFrame     // a vector given in a moved rectangular system
    };

    // A vector of a load table derived again.
    struct Load
    {
        LoadForm form = LoadForm::TwoPoints;
        std::string table; // loadsTable or inertialLoadsTable
        std::int64_t row = 0;
        std::array<std::string, 3> columns;  // its three components
        std::array<std::size_t, 4> points{}; // TwoPoints, FourPoints: indices into the points
        std::size_t frame = 0;               // InFrame: an index into the frames
        double magnitude = 0.0;              // TwoPoints, FourPoints: m
        base::Vec3 vector;                   // InFrame: the vector at the base positions
        std::int32_t gridId = 0;             // the grid point it acts on; 0 for a load on the whole model
        bool isConcentrated = true;          // a row of loadsTable on a grid point
        std::vector<std::size_t> bases;      // the bases that change it, increasing
        double size = 0.0;                   // the smallest distance between its points (TwoPoints, FourPoints)
    };

    DerivedGeometry() = default;
    // Binds the quantities the shape bases of `relations` change in `snapshot`; the refusals are appended to
    // `problems` (the quantities refused are not derived).
    DerivedGeometry(const ShapeRelations& relations, const input::ModelSnapshot& snapshot,
                    std::vector<base::Diagnostic>& problems);

    // Whether no quantity is derived again (no shape moves a defining grid point or a system).
    [[nodiscard]] bool isEmpty() const noexcept { return m_frames.empty() && m_loads.empty(); }
    // The CIDs of the systems shape basis `basis` moves, increasing.
    [[nodiscard]] std::vector<std::int32_t> systemsMovedBy(std::size_t basis) const;
    // Whether a basis changes a concentrated load (loadsTable: PointForceLoad1, PointForceLoad2, PointMomentLoad1,
    // PointMomentLoad2, a FORCE or MOMENT with CID), whose load vectors the sensitivities must then difference.
    [[nodiscard]] bool changesConcentratedLoads(std::size_t basis) const;
    // Whether a basis changes something every unit reads: a system (any element may name it) or an inertial load.
    [[nodiscard]] bool changesEveryUnit(std::size_t basis) const;
    // The grid points of the concentrated loads a basis changes (for the units they change), increasing.
    [[nodiscard]] std::vector<std::int32_t> loadGridsOf(std::size_t basis) const;
    // The smallest distance between two grid points that define a quantity the basis changes (G1 to G2, G3 to G4, A
    // to B, A to C), +∞ when it changes none: the scale of the step of its derivative (ShapeElements).
    [[nodiscard]] double smallestSize(std::size_t basis) const;

    // Writes the derived quantities at the shape values `shapes` (one per basis of the relations it was built with)
    // into `overlay`: every one, even at the base positions, so that every overlay of a model has the same columns.
    void addCells(std::span<const double> shapes, input::ColumnOverlay& overlay) const;

    [[nodiscard]] const std::vector<Frame>& frames() const noexcept { return m_frames; }
    [[nodiscard]] const std::vector<Load>& loads() const noexcept { return m_loads; }

private:
    std::vector<Point> m_points;
    std::vector<Frame> m_frames; // a frame's parent comes before it
    std::vector<Load> m_loads;
};

} // namespace oos::design
