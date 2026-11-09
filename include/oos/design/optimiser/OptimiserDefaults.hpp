#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

// The default values of the optimisation controls (the design cycles, the approximate optimisation, the move limits)
// and the public source of each. They come from the two optimisation programs G. N. Vanderplaats wrote for NASA, whose
// manuals list every control with its default:
//
//   [ADS]    G. N. Vanderplaats, "ADS - A Fortran program for automated design synthesis, Version 1.10", NASA
//            CR-177985, 1985. §4.3 "Over-riding ADS default parameters": Table 6 (the real parameters and their
//            defaults), Table 7 (their definitions), Table 8 (the integer parameters), Table 9 (their definitions).
//            ADS works on three levels: the strategy (a sequence of approximate problems, as the design cycles of
//            SOL 200: ISTRAT = 6 sequential linear programming, 9 sequential convex programming), the optimizer that
//            solves each of them (IOPT = 4 feasible directions, 5 modified feasible directions) and the
//            one-dimensional search.
//   [CONMIN] G. N. Vanderplaats, "CONMIN - A Fortran program for constrained function minimization, user's manual",
//            NASA TM X-62282, 1973: the parameters defined in the main program (§ "Parameters defined in the main
//            program" and the summary of appendix A), and the CONMIN Fortran source published with it (subroutine
//            CONMIN: the convergence tests; subroutine CNMN05: the direction-finding problem).
//
// What a DesignOptimParams keyword plays the role of is given in OptimisationParameters.cpp; a value with no
// counterpart in these sources is a choice of O_o_Solver and says so.
namespace oos::design::optimiser_defaults
{

// ---- The design cycles (the strategy level of ADS) ----

// DELSTR: the largest relative change of the objective between two consecutive iterations of the strategy that
// indicates convergence [ADS, Table 6: 0.001]. DesignOptimParams CONV1.
inline constexpr double CYCLE_RELATIVE_OBJECTIVE_CHANGE = 1.0e-3;
// DABSTR: the absolute counterpart, ABS(F0)/1000 with F0 the objective of the initial design [ADS, Table 6].
// DesignOptimParams CONV2. The same fraction gives DABOBJ at the optimizer level.
inline constexpr double ABSOLUTE_CHANGE_FRACTION_OF_F0 = 1.0e-3;
// The floor of an absolute change taken from F0, so that an initial objective of zero still gives a test:
// CONMIN sets DABFUN = 0.001·|OBJ| of the initial design and raises it to 1.0E-10 when smaller [CONMIN source,
// subroutine CONMIN].
inline constexpr double ABSOLUTE_CHANGE_FLOOR = 1.0e-10;
// The default absolute change of the objective (DABSTR, DABOBJ) for an initial objective f0.
[[nodiscard]] inline double absoluteChangeOf(double f0) noexcept
{
    return std::max(ABSOLUTE_CHANGE_FRACTION_OF_F0 * std::abs(f0), ABSOLUTE_CHANGE_FLOOR);
}
// ITRMST: the number of consecutive strategy iterations for which the absolute or relative criterion must hold
// [ADS, Table 8: 2].
inline constexpr std::size_t CONSECUTIVE_CYCLES = 2;
// An infeasible design: "if the current design is infeasible, five iterations are required to terminate and this
// situation indicates that a feasible design may not exist" [CONMIN, DELFUN].
inline constexpr std::size_t INFEASIBLE_CONSECUTIVE_CYCLES = 5;
// JTMAX: the maximum number of iterations at the strategy level [ADS, Table 8: 20]. DesignOptimParams DESMAX.
inline constexpr std::int64_t MAXIMUM_CYCLES = 20;

// ---- The approximate optimisation (the optimizer level of ADS) ----

// CT: a constraint more positive than CT is active in the feasible directions methods [ADS, Table 6: −0.03].
inline constexpr double ACTIVE_CONSTRAINT_THRESHOLD = -0.03;
// CTMIN: a constraint more positive than CTMIN is violated [ADS, Table 6: 0.01]. DesignOptimParams CTMIN, and GMAX for
// the exact analysis of a design cycle (the same tolerance on the same normalised constraints).
inline constexpr double VIOLATED_CONSTRAINT_THRESHOLD = 1.0e-2;
// DELOBJ: the largest relative change of the objective between two consecutive iterations that indicates convergence
// in the optimizer [ADS, Table 6: 0.001].
inline constexpr double ITERATION_RELATIVE_OBJECTIVE_CHANGE = 1.0e-3;
// ITMAX: the maximum number of iterations of the optimizer [ADS, Table 8: 40].
inline constexpr std::size_t ITERATION_LIMIT = 40;
// ITRMOP: the consecutive iterations for which DABOBJ or DELOBJ must hold [ADS, Table 8: 3].
inline constexpr std::size_t CONVERGED_ITERATIONS = 3;
// STOL: the tolerance on the components of the search direction that indicates that the Kuhn-Tucker conditions are
// satisfied [ADS, Table 6: 0.001].
inline constexpr double KUHN_TUCKER_TOLERANCE = 1.0e-3;

// ---- The move limits of a design cycle ----

// RMVLMZ: the initial relative move limit of the sequential linear programming strategy, "a fraction of the value of
// X(I)" [ADS, Table 6: 0.2]; DesignOptimParams DELX and DELP under the direct linearisation (APRCOD = 1).
inline constexpr double DIRECT_MOVE_LIMIT = 0.2;
// RMVLMZ of the sequential convex programming strategy [ADS, Table 6, note 4: 0.4 when ISTRAT = 9]: its reciprocal
// variables follow the structural responses better, so that "move limits were included, but they are less stringent
// than for sequential linear programming" [ADS, §1]. DesignOptimParams DELX and DELP under the other approximations.
inline constexpr double CONVEX_MOVE_LIMIT = 0.4;
// SCLMIN: the smallest scale factor of a design variable [ADS, Table 6: 0.001]. O_o_Solver's choice for the floors
// that keep a quantity near zero from vanishing: the smallest move of a variable or a property (DOPTPRM DXMIN, DPMIN)
// and the normalisation of a constraint bound near zero (DOPTPRM GSCAL).
inline constexpr double MINIMUM_SCALE = 1.0e-3;

// ---- The method of feasible directions ----

// THETA: the mean value of the push-off factor θ_j = THETA·(1 − g_j/CT)², which goes from 0 at g_j = CT to
// 4·THETA at g_j = |CT| [CONMIN, THETA: 1.0], each factor at most 50 [CONMIN source, subroutine CNMN05].
inline constexpr double PUSH_OFF_THETA = 1.0;
inline constexpr double MAXIMUM_PUSH_OFF_FACTOR = 50.0;
// PHI: the participation coefficient of the violated constraints [CONMIN, PHI: 5.0], multiplied by 10 each time no
// direction reduces the violation, up to 1000 [CONMIN source, subroutine CONMIN].
inline constexpr double INITIAL_PARTICIPATION = 5.0;
inline constexpr double PARTICIPATION_GROWTH = 10.0;
inline constexpr double MAXIMUM_PARTICIPATION = 1000.0;
// DLOBJ1: the relative change of the objective attempted on the first step of a one-dimensional search [ADS, Table 6:
// 0.1; CONMIN ABOBJ1: 0.1].
inline constexpr double FIRST_STEP_OBJECTIVE_FRACTION = 0.1;
// DX2: the largest change of a design variable attempted on the first step of a one-dimensional search [ADS, Table 6:
// 0.2], here in the scaled variables (a variable of magnitude 1).
inline constexpr double FIRST_STEP_VARIABLE_CHANGE = 0.2;

} // namespace oos::design::optimiser_defaults
