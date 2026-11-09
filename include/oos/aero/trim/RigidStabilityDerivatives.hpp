#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <array>
#include <string>
#include <vector>

namespace oos::aero
{

// The six non-dimensional coefficients of one trim variable, in the reference frame RCSID: CX, CY, CZ (force / (q
// REFS)), CMX, CMY, CMZ (moment about the RCSID origin / (q REFS REFB), (q REFS REFC), (q REFS REFB)).
struct StabilityCoefficients
{
    std::string label;
    std::array<double, 6> values{};
};

// The rigid, unsplined stability and control derivatives of a static aeroelastic model at a Mach number (SOL 144):
// the aerodynamic forces of a unit value of each trim variable on the rigid configuration, before any spline. For each
// variable: its normalwash (TrimVariables), the pressures Δcp = D⁻¹ w (DoubletLatticeSteady), the force q Δcp_j A_j
// along the normal of each box at its force point, summed and divided by the reference values (AeroReference). The
// intercept (DownwashMatrix, AeroPressureMatrix: TrimCorrections) is the trim's reference row, not computed here. A
// half model (AEROS SYMXZ or SYMXY) gives half of the forces and moments of the whole configuration (the boxes and
// their images): the components its symmetry doubles are those of the half model, those it cancels are zero (for SYMXZ
// = +1, CY, CMX, CMZ = 0).
//
// Thread-safety: stateless.
class RigidStabilityDerivatives
{
public:
    // The coefficients of every variable of TrimVariables at `mach`, in its order. Throws SolverError for a model
    // without AeroStaticParams or boxes, UnsupportedCardError for panels other than AeroLiftingPanel1,
    // MemoryBudgetError when the influence matrix (8 n² bytes for n boxes) does not fit `budget`.
    [[nodiscard]] static std::vector<StabilityCoefficients> compute(const input::ModelSnapshot& snapshot, double mach,
                                                                    base::MemoryBudget& budget);
};

} // namespace oos::aero
