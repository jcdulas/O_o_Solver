#pragma once

#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <span>
#include <string>

namespace oos::aero
{

// The direct matrix input of a static aeroelastic trim (the correction factors of the format): the matrices of the
// fixed names DownwashMatrix and AeroPressureMatrix that O_o_Reader's directMatricesTable gives, read on the boxes of
// the doublet lattice.
//
// - DownwashMatrix is the initial downwash w_j^g (incidence, camber, twist): radians of downwash at each box's control
// point.
//   A downwash is plus the normal component of the onset flow, O_o_Solver's normalwash (TrimVariables) minus it, so
//   the box's normalwash is w_0 = −DownwashMatrix: a positive DownwashMatrix is a positive incidence (a DownwashMatrix
//   of 0.17453 on every box gives 0.17453 times the derivatives of AngleOfAttack).
// - AeroPressureMatrix is f_j^e / q, experimental pressure coefficients added to the theoretical ones, Δcp_e =
// AeroPressureMatrix (a positive
//   pressure coefficient lifts its box along its normal).
// - Each is a single vector that applies to every subcase: one column, whatever the Mach number.
// - A DirectMatrix is in the j set: its row i (from 1) is the box at position i of AeroBoxes (the order of the j set),
// its row
//   count the number of boxes. A DirectMatrixAeroJ is in the js set: each term's row is a box ID with the component 3
//   (a AeroLiftingPanel1 box).
// - AeroWeightingMatrix and AeroWeightingFactor, the weighting of the theoretical forces in the k set (two rows a box),
// are accepted only as the
//   identity, which changes nothing; a true weighting is not applied by this version.
//
// Only the matrices O_o_Reader marks in effect are read (a DirectMatrixAeroJ, DirectMatrixAeroJInterference or
// DirectMatrixAeroK of a name a DirectMatrix also has is not). What cannot be read unambiguously is refused, naming the
// matrix: several columns, a complex matrix, a DirectMatrix whose rows are not the boxes, a DirectMatrixAeroJ row on a
// box the model lacks or on another component or given twice, a DirectMatrixAeroJInterference (bodies), a
// DirectMatrixAeroK AeroPressureMatrix (a k-set AeroPressureMatrix is ambiguous, the pressure correction acting on the
// j set), a AeroWeightingMatrix that is not the identity (SolverError for a matrix that contradicts the model,
// UnsupportedCardError for a form this version does not take).
//
// Thread-safety: immutable. Ownership: its own copies of the values.
class TrimCorrections
{
public:
    // The corrections of the snapshot's direct matrix tables (none when it has no directMatrixShapes).
    TrimCorrections(const input::ModelSnapshot& snapshot, const AeroBoxes& boxes, base::MemoryBudget& budget);

    [[nodiscard]] bool hasNormalwash() const noexcept { return !m_normalwashSource.empty(); }
    [[nodiscard]] bool hasPressures() const noexcept { return !m_pressureSource.empty(); }
    [[nodiscard]] bool isEmpty() const noexcept { return !hasNormalwash() && !hasPressures(); }

    // w_0 of each box (O_o_Solver's sign, −DownwashMatrix), in the order of AeroBoxes; zeros without a DownwashMatrix.
    [[nodiscard]] std::span<const double> normalwash() const noexcept { return m_normalwash.span(); }
    // Δcp_e of each box (AeroPressureMatrix); zeros without an AeroPressureMatrix.
    [[nodiscard]] std::span<const double> pressures() const noexcept { return m_pressures.span(); }

    // The matrices read, for messages: "DirectMatrix DownwashMatrix", "DirectMatrixAeroJ AeroPressureMatrix"; empty
    // when none.
    [[nodiscard]] const std::string& normalwashSource() const noexcept { return m_normalwashSource; }
    [[nodiscard]] const std::string& pressureSource() const noexcept { return m_pressureSource; }

private:
    base::BudgetedBuffer<double> m_normalwash;
    base::BudgetedBuffer<double> m_pressures;
    std::string m_normalwashSource;
    std::string m_pressureSource;
};

} // namespace oos::aero
