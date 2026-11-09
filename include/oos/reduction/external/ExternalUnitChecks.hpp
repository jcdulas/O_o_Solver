#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/external/ExternalUnit.hpp"
#include "oos/reduction/external/ExternalZone.hpp"

#include <cstddef>

namespace oos::reduction
{

// The tolerances of the checks of an external unit, measured on the static condensation of a
// 9 × 9 × 9 lattice of bars (rigid-body invariant, axial stiffnesses spread over four decades) onto its surface: 1 029
// interior DOFs onto 1 158 contour DOFs, by numpy's LU. A plugin may condense in double precision (asymmetry 7e-17,
// rigid-body residual 1e-15, rigid-body pivots 2e-15 of the largest term), stop an iterative solver at 1e-10 (2.5e-11,
// 9e-11, a remainder of 1.2e-10) or work in single precision (3.7e-8, 8.3e-8, a remainder of 2e-7); every one of them
// passes a tolerance of 1e-6. A wrong matrix is off by the size of its error: a grounded spring of 1e-4 of the largest
// term leaves a rigid-body residual of 1e-4, a negative stiffness of 1e-3 a pivot of −3.3e-3, a transposed or dropped
// coupling an asymmetry of its own size. An error below 1e-6 of the largest term cannot change a result.

// The asymmetry a K* or M* may have, max |a_ij − a_ji| / max |a_ij|, before it is refused ("the symmetry
// (within EXTERNAL_UNIT_SYMMETRY_TOLERANCE, then symmetrised)"); an accepted matrix is used as ½·(A + Aᵀ).
inline constexpr double EXTERNAL_UNIT_SYMMETRY_TOLERANCE = 1.0e-6;

// The force a rigid-body motion r of the contour may leave, max_i |(K*·r)_i| / (max_ij |K*_ij| · max_j |r_j|) (the
// stiffness the motion meets against the largest of K*: a grounded spring k leaves k / max |K*|), before K* is refused
// for missing that motion ("the six rigid-body motions of the contour (K*·r ≈ 0, unless the zone holds
// constraints)"). Not against each row's own terms: a motion that barely moves the labels (a rotation about the axis
// of a line of points) leaves terms of rounding size whose ratio means nothing.
inline constexpr double EXTERNAL_UNIT_RIGID_BODY_TOLERANCE = 1.0e-6;

// The tolerance of the semi-definiteness test of K* and M* (testSemiDefinite), relative to the largest term: a pivot
// below it ends the factorisation, a remainder below −it (or a coupling above it) is a negative eigenvalue.
inline constexpr double EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE = 1.0e-6;

// What the checks of an external unit measured (each figure as defined with its tolerance above).
struct ExternalUnitCheckReport
{
    bool areLabelsAccepted = false;
    double asymmetry = 0.0;
    double massAsymmetry = 0.0;       // 0 without M*
    double derivativeAsymmetry = 0.0; // the largest over the derivatives of K* and M*
    bool isRigidBodyChecked = false;  // false when the zone holds constraints (or the labels were refused)
    double rigidBodyResidual = 0.0;   // the largest over the six motions
    std::size_t rank = 0;             // of K* to the tolerance
    double smallestPivot = 0.0;       // of K*, relative (testSemiDefinite)
    double smallestMassPivot = 0.0;   // of M*, relative; 0 without M*
    bool isAccepted = false;
};

// The checks of an external unit against its zone before use: O_o_Solver
// checks the labels and the sizes, the symmetry (within EXTERNAL_UNIT_SYMMETRY_TOLERANCE, then symmetrised), the six
// rigid-body motions of the contour (K*·r ≈ 0, unless the zone holds constraints) and the positive semi-definiteness
// (by its factorisation): a matrix that fails is refused, naming the check and the zone"):
// 1. labels and sizes: the unit's labels are exactly the contour DOFs of the zone (any order); a missing or extra
//    label is named;
// 2. symmetry of K*, of M* and of every derivative (EXTERNAL_UNIT_SYMMETRY_TOLERANCE), the worst pair named;
// 3. rigid-body motions, unless the zone holds constraints: the translations along and the rotations about the basic
//    axes through the centroid of the contour points, in the displacement system of each point
//    (u_cd = Rᵀ·(t + θ × (x − x̄)), rotation Rᵀ·θ) on the unit's labels; the worst motion and label named
//    (EXTERNAL_UNIT_RIGID_BODY_TOLERANCE);
// 4. semi-definiteness of ½·(K* + K*ᵀ) and of ½·(M* + M*ᵀ) by testSemiDefinite (EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE),
//    the negative pivot and its label named.
// Every failure is an error diagnostic of `sink` (card "ZONE", the zone's ID, code "external-unit-<check>"); the checks
// after a refused label set are not made.
//
// Thread-safety: immutable once built. Ownership: owns its report; borrows nothing after construction.
class ExternalUnitChecks
{
public:
    ExternalUnitChecks(const ExternalUnit& unit, const ExternalZone& zone, const assembly::DofMap& dofMap,
                       base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] const ExternalUnitCheckReport& report() const noexcept { return m_report; }
    [[nodiscard]] bool isAccepted() const noexcept { return m_report.isAccepted; }

private:
    ExternalUnitCheckReport m_report;
};

} // namespace oos::reduction
