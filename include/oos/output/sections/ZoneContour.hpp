#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/sections/ContourReport.hpp"
#include "oos/output/sections/SparseRows.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::output
{

// The rows of the equilibrium check of a zone: the resultant force and moment of its contour forces (rows 0 to 5),
// then of the loads and reactions of its interior grid points (rows 6 to 11), in the system of the zone about its
// origin. Their sum is zero to round-off (ZoneContour, "Equilibrium").
inline constexpr std::size_t EQUILIBRIUM_ROWS = 12;

// Two frames are the same when their rotation differs from the identity by less than this: the axes of the systems
// come from the CORDij cards through O_o_Reader in double precision, so equal systems give the identity to a few
// units of round-off, and any real difference of direction is far above it.
inline constexpr double SAME_FRAME_TOLERANCE = 1.0e-12;

// The contours of the zones of external local models and their contour data.
//
// The contour. A grid point of a zone (one its members touch) is a contour point when something outside the zone acts
// on it: O_o_Reader's boundary of the zone as a group of elements (zoneBoundaries: the elements, rigid elements, MPC
// equations and connectors outside, the grid points of the element a fastener pierces when its other side is outside),
// with the components through which they act; the enclosed elements kept outside (enclosed="warn" or "exclude": the
// components their stiffness uses there, by kind, the exact ones for a spring or a CBUSH); and the grid points
// constrained in some of the subcases only (the components constrained in some but not in every subcase), so that the
// contour does not change from one subcase to the next. The other grid points of the zone are interior; the
// components constrained in every subcase are held constraints of the zone (its local model is held there too).
//
// The contour data of one case, in the system of the zone (`frameId`; ZONE_COMPONENTS):
//  - the contour displacements, d = T·u at the contour DOFs, T = Aᵀ·R_cd the rotation from the displacement system of
//    the grid point to the zone's (A its axes in basic);
//  - the contour forces (the grid point forces of the elements of the zone at that point):
//    F_g = Σ_{e∈zone} (K_e·u_e − f_e,T)|_g, f_e,T the thermal load of the element (the element rows of the grid point
//    forces, as for the cut loads), minus the force q_g the zone's constraint rows
//    exert on the point (the MPC forces there) when only rows of the zone act on it: an RBE2 or RBE3 of the zone with a
//    point on the contour, the reference point of an absorbed spider. A point on which constraint rows of the zone and
//    of the outside both act is refused (their forces cannot be told apart from the MPC forces);
//  - the loads of the zone: the applied loads (OLOAD) on its interior grid points, forces and moments (which hold the
//    pressure, line and inertial loads of its elements there);
//  - the equilibrium: the zone is a free body made of its elements, its constraint rows and its interior grid points,
//    on which act the contour forces and the loads and reactions of the interior grid points, so
//        Σ_g W_g·F_g + Σ_i W_i·(P_i + R_i) = 0,   W = [Aᵀ·R_cd, 0; Aᵀ·[x − o]×·R_cd, Aᵀ·R_cd]
//    (every component of F_g, not only the contour ones: a component nothing outside uses carries the loads applied to
//    it), as the OLOAD and SPC force resultants of a subcase balance: the thermal loads of the elements are
//    self-equilibrated and so are the forces of rigid elements (an MPC equation that is not self-equilibrated breaks
//    it). EQUILIBRIUM_ROWS rows: the contour resultant, then the interior one.
// All of them are linear in the g-set vectors of the case (affine in u with a thermal load): rows() holds them as
// SparseRows over u, the MPC forces q, the applied loads P and the SPC forces R, so that the data of the k vectors of
// a load basis are products and the rows are the exact weights of the sensitivities; the thermal part of a case is the
// element part at zero displacements with its temperatures (thermalPart).
//
// Problems are diagnostics of the sink, naming the zone: a member the model does not have, a boundary point the model
// does not have, a CID the model does not have or that is not rectangular, a contour point on which rows of the zone
// and of the outside act, a zone without members; an enclosed element kept outside with enclosed="warn" is a warning.
// A zone with an error has no rows.
//
// Thread-safety: immutable once built, but rows() builds the rows of a zone at its first call (not thread-safe then).
// Ownership: borrows the element set and the DOF map, which must outlive it; owns its definitions, reports and rows.
class ZoneContour
{
public:
    // An element of a zone in the element set.
    struct Member
    {
        std::size_t batch = 0;
        std::int64_t row = 0;
        std::int32_t elementId = 0;
    };
    // A DOF of the contour data: a grid point and a component in the zone's system (1 to 6).
    struct ZoneDof
    {
        std::int32_t grid = 0;
        std::int32_t gridRow = 0;
        int component = 1;
    };
    // A grid point of the contour or of the interior, with its rotation T = Aᵀ·R_cd and its position relative to the
    // zone's origin, in basic.
    struct ZonePoint
    {
        std::int32_t grid = 0;
        std::int32_t gridRow = 0;
        ComponentMask components = ALL_COMPONENTS; // a contour point: the components the outside acts through (CD)
        bool takesConstraintForces = false;        // a contour point on which only constraint rows of the zone act
        base::Mat3 rotation = base::Mat3::identity();
        base::Vec3 arm; // x − o in basic
    };
    struct Zone
    {
        std::size_t definition = 0;
        ContourReport report;
        std::vector<Member> members; // the zone's elements with a stiffness, in the order of the batches
        std::vector<ZonePoint> contour;
        std::vector<ZonePoint> interior; // grid points only
        std::vector<ZoneDof> contourDofs;
        std::vector<ZoneDof> interiorDofs; // 6 per interior grid point
        base::Mat3 axes = base::Mat3::identity();
        base::Vec3 origin;
        bool isValid = false;
    };
    // The rows of a zone (the class comment), each over a g-set vector.
    struct ZoneRows
    {
        SparseRows displacements;            // contour DOFs, over u
        SparseRows forces;                   // contour DOFs, over u (the element part)
        SparseRows forceConstraints;         // contour DOFs, over the MPC forces (−q at the points that take them)
        SparseRows loads;                    // interior DOFs, over the applied loads
        SparseRows equilibriumDisplacements; // EQUILIBRIUM_ROWS, over u (the contour resultant)
        SparseRows equilibriumConstraints;   // EQUILIBRIUM_ROWS, over the MPC forces (the contour resultant)
        SparseRows equilibriumLoads;         // EQUILIBRIUM_ROWS, over the applied loads (the interior resultant)
        SparseRows equilibriumReactions;     // EQUILIBRIUM_ROWS, over the SPC forces (the interior resultant)
    };

    // `subcases`: the subcases whose constraints decide the partial ones (empty: none is compared, every constraint is
    // held).
    ZoneContour(std::vector<ZoneDefinition> definitions, std::span<const ZoneSubcaseConstraint> subcases,
                const input::ModelSnapshot& snapshot, const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] const std::vector<ZoneDefinition>& definitions() const noexcept { return m_definitions; }
    [[nodiscard]] const std::vector<Zone>& zones() const noexcept { return m_zones; }
    // The index of the zone of a definition ID, or nothing.
    [[nodiscard]] std::optional<std::size_t> findZone(std::int32_t id) const noexcept;

    // The rows of zone z (made at the first call, reserved in `budget`).
    [[nodiscard]] const ZoneRows& rows(std::size_t z, base::MemoryBudget& budget);
    [[nodiscard]] bool hasRows(std::size_t z) const noexcept { return m_rows[z] != nullptr; }
    [[nodiscard]] const ZoneRows& builtRows(std::size_t z) const { return *m_rows[z]; }

    // The element part of the forces of zone z at its contour points, six components per point in its displacement
    // system: Σ_{e∈zone} (K_e·u_e − f_e,T)|_g, the elements computed by `set` (that of the constructor, or of an
    // overlay of its snapshot with the same batches: the explicit parts of the sensitivities). Throws
    // std::invalid_argument for a vector not of the g-set size.
    [[nodiscard]] std::vector<double> elementForces(std::size_t z, const assembly::ElementSet& set,
                                                    std::span<const double> displacementsG,
                                                    const elements::ElementTemperatures* temperatures) const;
    // The contour DOF values of six components per contour point in CD (elementForces), and their resultant (rows 0 to
    // 5 of the equilibrium).
    [[nodiscard]] std::vector<double> toContourDofs(std::size_t z, std::span<const double> perPoint) const;
    [[nodiscard]] std::array<double, 6> contourResultant(std::size_t z, std::span<const double> perPoint) const;
    // The thermal part of a case: the contour forces and the equilibrium rows at zero displacements with
    // `temperatures` (the forces at the contour DOFs, then the EQUILIBRIUM_ROWS rows), on the constructor's set.
    struct ThermalPart
    {
        std::vector<double> forces;
        std::array<double, EQUILIBRIUM_ROWS> equilibrium{};
    };
    [[nodiscard]] ThermalPart thermalPart(std::size_t z, const elements::ElementTemperatures& temperatures) const;

private:
    void resolve(std::size_t d, std::span<const ZoneSubcaseConstraint> subcases, const input::ModelSnapshot& snapshot,
                 base::MemoryBudget& budget, base::DiagnosticSink& sink);
    [[nodiscard]] std::unique_ptr<ZoneRows> buildRows(const Zone& zone, base::MemoryBudget& budget) const;

    std::vector<ZoneDefinition> m_definitions;
    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    std::vector<Zone> m_zones;
    std::vector<std::unique_ptr<ZoneRows>> m_rows; // per zone, made at the first rows() call
    base::BudgetedBuffer<double> m_zeros;          // g-set zeros, for the thermal parts
};

} // namespace oos::output
