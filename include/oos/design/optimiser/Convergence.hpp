#pragma once

#include "oos/design/optimiser/OptimisationParameters.hpp"
#include "oos/design/optimiser/OptimiserDefaults.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace oos::design
{

// One design cycle of a history: the exact analysis of its design (cycle 0 is the initial analysis).
struct DesignCycle
{
    double objective = 0.0;
    double maxConstraint = 0.0;     // the largest normalised constraint of the analysis
    std::vector<double> variables;  // the design variables (every one, the dependent ones included)
    std::vector<double> properties; // the designed properties (empty: not given)
};

// The criteria of the convergence of the design cycles, the strategy level of ADS (NASA CR-177985, Tables 6 to 9:
// DELSTR, DABSTR, ITRMST, JTMAX) with the feasibility condition of CONMIN (NASA TM X-62282, DELFUN); the defaults are
// those of OptimiserDefaults.hpp. DesignOptimParams gives CONV1, CONV2, GMAX and DESMAX; the numbers of consecutive
// cycles have no keyword.
struct ConvergenceCriteria
{
    // CONV1 (ADS DELSTR): the largest relative change |ΔF|/|F_prev| of a steady objective.
    double relativeObjectiveChange = optimiser_defaults::CYCLE_RELATIVE_OBJECTIVE_CHANGE;
    // CONV2 (ADS DABSTR): the largest absolute change |ΔF| of a steady objective; nothing: |F0|/1000 with F0 the
    // objective of cycle 0 (at least 1.0E-10).
    std::optional<double> absoluteObjectiveChange;
    // GMAX (the role of ADS CTMIN on the exact analysis): a design is feasible when its largest normalised constraint
    // is at most this.
    double feasibilityTolerance = optimiser_defaults::VIOLATED_CONSTRAINT_THRESHOLD;
    // ITRMST: the consecutive cycles with a steady objective at a feasible design that end the run.
    std::size_t consecutiveCycles = optimiser_defaults::CONSECUTIVE_CYCLES;
    // CONMIN: the consecutive cycles with a steady objective at an infeasible design that end the run (a feasible
    // design may not exist).
    std::size_t infeasibleConsecutiveCycles = optimiser_defaults::INFEASIBLE_CONSECUTIVE_CYCLES;
    // DESMAX (ADS JTMAX): the run ends after the analysis of this cycle.
    std::int64_t maximumCycles = optimiser_defaults::MAXIMUM_CYCLES;

    // CONV1, CONV2 (when given), GMAX and DESMAX of a DesignOptimParams.
    [[nodiscard]] static ConvergenceCriteria fromParameters(const OptimisationParameters& parameters);
    // The absolute change for a run whose initial objective is `initialObjective`.
    [[nodiscard]] double absoluteChange(double initialObjective) const noexcept;
};

// The tests of one cycle against the previous one. A relative change with a zero previous value is the absolute
// change. The changes of the variables and properties are reported, not tested (DOPTPRM CONVDV and CONVPR have no
// effect: the public criteria rest on the objective and the feasibility).
struct ConvergenceCheck
{
    std::size_t cycle = 0;
    double relativeObjectiveChange = 0.0;
    double absoluteObjectiveChange = 0.0;
    double maxConstraint = 0.0;
    double maxRelativeVariableChange = 0.0;
    double maxRelativePropertyChange = 0.0;
    bool isObjectiveSteady = false; // |ΔF| ≤ the absolute change, or |ΔF|/|F_prev| ≤ the relative one
    bool isFeasible = false;        // max g ≤ the feasibility tolerance
    // The consecutive cycles up to this one with a steady objective at a feasible design (resp. an infeasible one).
    std::size_t steadyFeasibleCycles = 0;
    std::size_t steadyInfeasibleCycles = 0;
};

enum class ConvergenceKind : std::uint8_t
{
    NotConverged,
    Feasible,        // the objective steady at a feasible design for `consecutiveCycles` cycles
    Infeasible,      // the objective steady at an infeasible design for `infeasibleConsecutiveCycles` cycles
    DesignCycleLimit // the analysis of cycle DESMAX made without convergence
};

[[nodiscard]] std::string_view convergenceKindName(ConvergenceKind kind) noexcept;

// The decision over a history: the checks of every cycle from 1, and the first cycle where the run ends.
struct ConvergenceDecision
{
    ConvergenceKind kind = ConvergenceKind::NotConverged;
    std::size_t cycle = 0; // the cycle where it ends (0 when not converged)
    std::vector<ConvergenceCheck> checks;
};

// The convergence of the design cycles of a SOL 200 run, on the exact analysis of each cycle (one test, no decision
// from the approximate optimisation alone). Each cycle k ≥ 1 is tested against cycle k − 1:
//   - the objective is steady when |F_k − F_k−1| ≤ CONV2 or |F_k − F_k−1|/|F_k−1| ≤ CONV1 (ADS, the strategy
//     level's DABSTR and DELSTR);
//   - the run converges to a feasible design when the objective has been steady with max g ≤ GMAX for ITRMST = 2
//     consecutive cycles (ADS ITRMST; CONMIN counts the objective test only while the design is feasible);
//   - it ends at an infeasible design when the objective has been steady with max g > GMAX for 5 consecutive cycles
//     (CONMIN, DELFUN: "five iterations are required to terminate and this situation indicates that a feasible
//     design may not exist");
//   - it ends after the analysis of cycle DESMAX otherwise (ADS JTMAX).
//
// Thread-safety: stateless.
class Convergence
{
public:
    // The tests of `current` (cycle `cycle`) against `previous`, for a run of initial objective `initialObjective`
    // (the consecutive counts left at 0: review() fills them). Throws OptimisationError when the variables or the
    // properties of the two cycles differ in number.
    [[nodiscard]] static ConvergenceCheck check(const DesignCycle& previous, const DesignCycle& current,
                                                std::size_t cycle, double initialObjective,
                                                const ConvergenceCriteria& criteria);
    // The decision over a history (cycle 0 first).
    [[nodiscard]] static ConvergenceDecision review(std::span<const DesignCycle> history,
                                                    const ConvergenceCriteria& criteria);
};

} // namespace oos::design
