#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/reduction/interface/InterfaceProblem.hpp"
#include "oos/reduction/patterns/BasisCoefficients.hpp"
#include "oos/reduction/recovery/UnitRecovery.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// The condensed patterns of one unit as right-hand sides of the interface problem: f* on the rows of its K*
// (InterfaceProblem::unitLocalDofs) and their coefficients for the cases.
struct BoundaryUnitPatterns
{
    linalg::ConstDenseMatrixView patterns;           // boundaryDofCount() × k
    const BasisCoefficients* coefficients = nullptr; // k × cases
};

// Loads of the interface's own points as right-hand sides: the loads on its local DOFs before its relations are
// eliminated, the enforced values of its held DOFs (no column: none), and their coefficients for the cases (the
// interface's load basis, or the inertia relief loads of the units and of the interface).
struct BoundaryLocalLoads
{
    linalg::ConstDenseMatrixView loads;              // InterfaceProblem::localDofCount() × v
    linalg::ConstDenseMatrixView enforced;           // the same, or no column
    const BasisCoefficients* coefficients = nullptr; // v × cases
};

// What a column of the boundary basis is: a pattern of a condensed unit, or a vector of a block of local loads.
struct BoundaryColumn
{
    std::int64_t unit = -1;  // index into the units given, or -1
    std::int64_t block = -1; // index into the local load blocks given, or -1
    std::int64_t vector = 0; // the pattern of the unit, or the vector of the block
};

// Where the solutions of the boundary basis come from (the phases of the condensed path may be driven by the caller).
enum class BoundarySolveMode : std::uint8_t
{
    Solve,              // the interface problem solves the columns (its factor, or streamed)
    RightHandSidesOnly, // nothing solved: the right-hand sides, the reduced loads and the enforced values only (the
                        // interface handed to the caller, Solver.interface); displacements() and residuals() empty
    GivenEquations, // the caller's solutions of the equations (equationCount() × the columns), expanded as a solve's
    GivenLocal      // the caller's local displacements (localDofCount() × the columns), taken as they are (no
                    // expansion: a unit's boundary given by another solver, Solver.recover)
};

// How the boundary basis is solved: by the interface problem (the default), or not, or from given values; whether the
// columns are the cases themselves whatever Q (values given per case); whether the solutions of the equations are kept
// (equationSolutions(): the interface's own solution handed to the caller).
struct BoundarySolveOptions
{
    BoundarySolveMode mode = BoundarySolveMode::Solve;
    bool solvesCases = false;
    linalg::ConstDenseMatrixView given; // GivenEquations, GivenLocal
    bool keepsEquationSolutions = false;
};

// The rows of the boundary basis on one condensed unit's boundary, as UnitRecovery takes them (UnitBoundaryBasis).
struct UnitBasisRows
{
    linalg::DenseMatrix values;                // boundaryDofCount() × columns
    std::vector<std::int64_t> patternOfColumn; // per column: the unit's pattern it is, or -1 (bounded by the basis)
};

// The boundary basis of a condensed static analysis: the
// interface problem is solved for the Q distinct right-hand sides of its cases, the condensed patterns of the units
// and the loads of its own points, by blocks of columns, and the boundary displacements of every case are
// combinations of these Q solutions, u_case = Σ_q β_q,case·V_q (β_q,case the coefficient of the pattern or vector of
// column q in the case: exact, as each case's loads are those combinations, UnitLoadBasis). A pattern or vector that
// no case takes is no column. When Q exceeds the number of cases the cases themselves are solved instead, their
// right-hand sides the combinations of the columns; β is then the identity.
//
// Each column's right-hand side: its loads on the local DOFs (a unit's f* added onto the rows of its K*), the
// interface's relations folded (P_n = P̄_n + Gmᵀ·P_m), then on the equations less K_ls·u_s of its enforced values;
// its solution expanded to the local DOFs (the held DOFs at their values, u_m = Gm·u_n). Kept per column: the local
// displacements V, the reduced local loads P_n (the reactions of the held DOFs), the equation right-hand sides F and
// the residuals R = F − K·u (EPSILON and the external work of every case are their combinations).
//
// Step 10.29 (BoundarySolveOptions): the solve may be skipped (the right-hand sides handed to the caller) or replaced
// by the caller's values: solutions of the equations, expanded as the interface's own (V, P_n, F and R then those of
// the solve that gave them: the same bits for the same solutions), or local displacements taken as they are (R the
// residual of their equation rows). Given per case (solvesCases), the columns are the cases whatever Q.
//
// Thread-safety: immutable once built. Ownership: owns V, P_n, F, R and β (budgeted); borrows the interface problem,
// which must outlive it.
class BoundaryBasis
{
public:
    // caseCount: the cases of every coefficient matrix; kernels: those of the combinations of the case basis. Throws
    // std::invalid_argument on a size mismatch (given values of other rows or columns than the basis's, named);
    // MemoryBudgetError; OperationCancelledError.
    BoundaryBasis(const InterfaceProblem& interface, std::span<const BoundaryUnitPatterns> units,
                  std::span<const BoundaryLocalLoads> blocks, std::size_t caseCount,
                  const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                  const base::CancellationToken* cancellation = nullptr, const BoundarySolveOptions& solve = {});

    // Whether the cases were solved one by one (Q above their number).
    [[nodiscard]] bool isCaseBasis() const noexcept { return m_isCaseBasis; }
    // The distinct right-hand sides Q (whatever was solved).
    [[nodiscard]] std::size_t distinctRightHandSides() const noexcept { return m_distinctCount; }
    // The columns solved: Q, or the cases.
    [[nodiscard]] std::size_t columnCount() const noexcept { return m_displacements.columns(); }
    [[nodiscard]] std::size_t caseCount() const noexcept { return m_coefficients.caseCount(); }
    // What each column is (empty in the case basis).
    [[nodiscard]] std::span<const BoundaryColumn> columns() const noexcept { return m_columns; }
    // V (local DOFs × columns), P_n (local × columns), F and R (equations × columns), β (columns × cases).
    [[nodiscard]] const linalg::DenseMatrix& displacements() const noexcept { return m_displacements; }
    [[nodiscard]] const linalg::DenseMatrix& reducedLoads() const noexcept { return m_reducedLoads; }
    [[nodiscard]] const linalg::DenseMatrix& rightHandSides() const noexcept { return m_rightHandSides; }
    [[nodiscard]] const linalg::DenseMatrix& residuals() const noexcept { return m_residuals; }
    [[nodiscard]] const BasisCoefficients& coefficients() const noexcept { return m_coefficients; }
    // The enforced values of the held DOFs per column (local × columns; no column when none enforces a value), and
    // with BoundarySolveOptions::keepsEquationSolutions the solutions of the equations (equations × columns; empty
    // otherwise).
    [[nodiscard]] const linalg::DenseMatrix& enforcedValues() const noexcept { return m_enforced; }
    [[nodiscard]] const linalg::DenseMatrix& equationSolutions() const noexcept { return m_equationSolutions; }

    // The rows of V on the boundary of unit `unit` (its K* rows) and the columns that are its own patterns, for its
    // recovery (UnitRecovery with the coefficients()).
    [[nodiscard]] UnitBasisRows unitRows(std::size_t unit, base::MemoryBudget& budget) const;

private:
    const InterfaceProblem* m_interface;
    bool m_isCaseBasis = false;
    std::size_t m_distinctCount = 0;
    std::vector<BoundaryColumn> m_columns; // bounded by the patterns of the units and the interface's vectors
    linalg::DenseMatrix m_displacements;
    linalg::DenseMatrix m_reducedLoads;
    linalg::DenseMatrix m_rightHandSides;
    linalg::DenseMatrix m_residuals;
    linalg::DenseMatrix m_enforced;
    linalg::DenseMatrix m_equationSolutions;
    BasisCoefficients m_coefficients;
};

} // namespace oos::reduction
