#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/sections/CutLoads.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace oos::output
{

// A MONPNT3 as the caller gives it: O_o_Reader gives the card's fields as columns, and the caller expands GRIDSET
// and ELEMSET (SET1 lists) to their IDs.
struct MonitorPoint3Definition
{
    std::string name;
    std::string label;
    std::int32_t axes = 123456;           // AXES: the components (1 to 6) the point reports
    std::vector<std::int32_t> grids;      // GRIDSET
    std::vector<std::int32_t> elements;   // ELEMSET; empty when blank (no element contributes)
    std::int32_t positionFrameId = 0;     // CID, the system of the point
    base::Vec3 pointInPositionFrame;      // X, Y, Z
    std::int32_t displacementFrameId = 0; // the system of the components of the loads
    std::string excluded;                 // XFLAG: the letters S, M, A, L, P, D (upper case)
};

// The g-set loads at the grid points that a MONPNT3 adds unless XFLAG excludes them: the rows appliedLoad,
// singlePointConstrForce and multiPointConstrForce of the grid point forces (StiffnessRecovery), each in the
// displacement systems of the grid points. An empty span is a contribution the analysis does not have.
struct MonitorNodalLoads
{
    std::span<const double> applied;
    std::span<const double> spc;
    std::span<const double> mpc;
};

// The integrated loads of a MONPNT3 (an integrated loads monitor point): at the grid points of GRIDSET,
// the grid point forces of the elements of ELEMSET (output::CutLoads: q_e,g = −(K_e·u_e)|g) and the loads XFLAG does
// not exclude (S the singlePointConstrForce rows, M the multiPointConstrForce rows, A, L or P the appliedLoad rows; D
// the other kinds, which the linear solutions do not have), summed about the point (X, Y, Z in CID) and given along the
// axes of CD:
//     F = Σ_g R_g·q_g^t,    M = Σ_g (x_g − p) × R_g·q_g^t + R_g·q_g^r,    loads = (A_CDᵀ·F, A_CDᵀ·M),
// with q_g the kept rows at grid point g, R_g its displacement system in basic. With ELEMSET the elements on one side
// of a cut and XFLAG blank this is the internal load the cut transmits. O_o_Reader's
// monitor_point_loads computes the same sum from the grid point forces of a result set.
//
// Thread-safety: immutable once built. Ownership: borrows the DOF map, which must outlive it; owns its definition and
// its cut.
class MonitorPoint3
{
public:
    // The monitor point of `definition` on the element set and the DOF map of the analysis. Throws SolverError naming
    // the MONPNT3 for a grid point or an element the model does not have, an empty GRIDSET, a CID or CD the snapshot
    // does not have or that is not rectangular.
    MonitorPoint3(MonitorPoint3Definition definition, const input::ModelSnapshot& snapshot,
                  const assembly::ElementSet& set, const assembly::DofMap& dofMap);

    [[nodiscard]] const MonitorPoint3Definition& definition() const noexcept { return m_definition; }
    // Whether component c (0 to 5: fx, fy, fz, mx, my, mz) is among AXES.
    [[nodiscard]] bool reports(std::size_t component) const noexcept { return m_isReported[component]; }

    // The six loads for the g-set displacements of the analysis on `set` (that of the constructor), the nodal loads and
    // the temperatures of a thermal load (null: none; the element rows are then −(K_e·u_e − f_e,T), CutLoads).
    [[nodiscard]] std::array<double, 6> loadsOf(const assembly::ElementSet& set, std::span<const double> displacementsG,
                                                const MonitorNodalLoads& nodal,
                                                const elements::ElementTemperatures* temperatures = nullptr) const;

private:
    MonitorPoint3Definition m_definition;
    const assembly::DofMap* m_dofMap;
    std::unique_ptr<CutLoads> m_cut;  // the elements of ELEMSET; null when it is blank
    std::vector<std::int32_t> m_rows; // the grid rows of GRIDSET, sorted
    base::Vec3 m_point;               // in the basic system
    base::Mat3 m_cdAxes;              // the axes of CD in the basic system (columns)
    std::array<bool, 6> m_isReported{};
    bool m_addsApplied = true;
    bool m_addsSpc = true;
    bool m_addsMpc = true;
};

} // namespace oos::output
