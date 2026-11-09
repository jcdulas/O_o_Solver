#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::aero
{

// The motion of a grid point: its translation and its rotation, in the basic system.
struct PointMotion
{
    base::Vec3 translation;
    base::Vec3 rotation;
};

// The plunge (along its normal) and slope (∂h/∂x, x the flow in the plane of the box) of each box at its aerodynamic
// grid point, the box center: the displacements of the aerodynamic k set (its T3 and −R2).
struct BoxMotion
{
    std::vector<double> plunge; // one per box
    std::vector<double> slope;  // one per box
};

// The interpolation between the structure and the boxes of a static aeroelastic model: the splines of the model
// (splinesTable, splineNodes, the AeroBeamSpline2, AeroBeamSpline5 and AeroBoxList cards), the matrix GKA:
// AeroSurfaceSpline1 and AeroSurfaceSpline4 (method InfinitePlateSpline or ThinPlateSpline: SurfaceSpline) in the plane
// of their panel, AeroBeamSpline2 and AeroBeamSpline5 (BeamSpline) along the y axis of their CID projected in that
// plane. Each spline moves its boxes (usage Displacement or Both) from the translation normal to the plane (and, for a
// beam, the rotations about the in-plane axes) of its structural points, and gives the aerodynamic loads of its boxes
// (usage Force or Both) to them by the transpose: a force F along the normal of a box at its force point is the load
// (F, F (x_f − x_c)) on its plunge and slope at its center, so that the virtual work of the box loads equals that of
// the structural loads. As the splines reproduce the rigid motions exactly, the structural loads have the resultant of
// the box loads.
//
// Refused (UnsupportedCardError): METH FPS (the finite plate spline) and RIS, AeroSplineEquation3,
// AeroFiniteSurfaceSpline6, AeroFiniteBeamSpline7, AeroUserSpline. A box moved by two splines, or loaded by two, is a
// SolverError (a box is splined once).
//
// Thread-safety: immutable. Ownership: owns its matrices (budgeted: each spline's boxes × its unknowns); borrows the
// boxes.
class SplineMatrix
{
public:
    // One spline: its boxes (indices into AeroBoxes) and its structural unknowns, each a grid point, a unit direction
    // in the aerodynamic frame and whether it is a translation or a rotation; `matrix` gives the plunge (row 2b) and
    // slope (row 2b + 1) of its b-th box for unit unknowns.
    struct Spline
    {
        std::int32_t elementId = 0;
        std::string card;
        bool movesBoxes = true;               // usage Displacement or Both
        bool loadsStructure = true;           // usage Force or Both
        std::vector<std::size_t> boxes;       // bounded by the boxes of the spline
        std::vector<std::int32_t> nodes;      // the grid point of each unknown
        std::vector<base::Vec3> directions;   // in the aerodynamic frame
        std::vector<std::uint8_t> isRotation; // per unknown
        linalg::DenseMatrix matrix;           // 2 · boxes × unknowns
    };

    // Throws SolverError for a spline without boxes or structural points, a point that is not a grid point, a box
    // splined twice; UnsupportedCardError for the methods and cards above.
    SplineMatrix(const input::ModelSnapshot& snapshot, const AeroReference& reference, const AeroBoxes& boxes,
                 base::MemoryBudget& budget);

    [[nodiscard]] const std::vector<Spline>& splines() const noexcept { return m_splines; }

    // The plunge and slope of every box for the motions of the grid points (`motionOf(nodeId)`; the boxes no spline
    // moves stay at 0).
    [[nodiscard]] BoxMotion boxMotion(const std::function<PointMotion(std::int32_t)>& motionOf) const;

    // The forces and moments at the grid points (basic system) of forces along the normals of the boxes at their force
    // points (`boxForces`: one per box), by the splines that load the structure; (grid point, force, moment) in the
    // order of the splines and their unknowns, a grid point repeated when several splines load it.
    struct PointLoad
    {
        std::int32_t nodeId = 0;
        base::Vec3 force;
        base::Vec3 moment;
    };
    [[nodiscard]] std::vector<PointLoad> structuralLoads(std::span<const double> boxForces) const;

private:
    const AeroBoxes* m_boxes;
    AeroReference m_reference;
    std::vector<Spline> m_splines; // bounded by the SPLINE cards
};

} // namespace oos::aero
