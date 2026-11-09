#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/aids/ApproximationHints.hpp"
#include "oos/design/approximation/MoveLimits.hpp"
#include "oos/design/approximation/ResponseApproximation.hpp"
#include "oos/design/approximation/RetainedResponses.hpp"
#include "oos/design/constraints/ConstraintNormalisation.hpp"
#include "oos/design/optimiser/OptimisationProblem.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/sensitivity/DesignGradients.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// An independent design variable of the approximate problem: its value at x0, its bounds XLB, XUB and its DELXV.
struct ApproximationVariable
{
    std::string label;
    double value = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    std::optional<double> moveLimit;
};

// A dependent design variable (DLINK): x_d = value + gradient·(x − x0), linear, with its own bounds and DELXV.
struct ApproximationLinkedVariable
{
    std::string label;
    double value = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    std::optional<double> moveLimit;
    std::vector<double> gradient; // d(x_d)/dx, one per independent variable
};

// A designed property (a field of a DVPREL, DVMREL or DVCREL): its value at x0, d(p)/dx, PMIN, PMAX, and whether the
// relation is linear in x (DVxREL1).
struct ApproximationProperty
{
    std::string label;
    double value = 0.0;
    std::vector<double> gradient;
    std::optional<double> lower;
    std::optional<double> upper;
    bool isLinear = true;
};

// How an eigenvalue response is approximated (the ATTB field of DesignResp1): by the Rayleigh quotient of its strain
// and kinetic energies (EIGN, FREQ: ATTB blank, the default form), or by a kind of its own (ATTB 1 direct, 2
// inverse under APRCOD = 1). The Rayleigh quotient approximation (Canfield, AIAA J. 28(6), 1990):
//     λ̃(x) = (λ0 + ∇U·(x − x0)) / (1 + ∇T·(x − x0)),
// U = φᵀ·K·φ and T = φᵀ·M·φ linearised at fixed φ (M-normalised: T0 = 1), exact at x0 with dλ/dx = ∇U − λ0·∇T; a FREQ
// takes f̃ = √λ̃ / 2π.
struct EigenApproximation
{
    bool usesRayleighQuotient = false;
    bool isFrequency = false;
    double root = 0.0;               // λ0
    std::vector<double> numerator;   // ∇U at x0
    std::vector<double> denominator; // ∇T at x0
    std::optional<ApproximationKind> kind;
};

// A retained response item: its value and gradient at x0 and its DesignConstrScreen type (which decides the mixed
// method).
struct ApproximationItem
{
    std::int32_t responseId = 0;
    std::size_t item = 0; // its index in the response's items
    ResponseItem label;
    std::string responseType;
    double value = 0.0;
    std::vector<double> gradient;
    std::vector<int> powers; // ApproximationKind::Hints: the power of each variable (ApproximationHints)
    std::optional<EigenApproximation> eigen{}; // EIGN, FREQ, LAMA
};

// A retained constraint on an item.
struct ApproximationConstraint
{
    std::size_t item = 0; // index into ApproximationInputs::items
    BoundSide side = BoundSide::Upper;
    double bound = 0.0;
    std::int32_t setId = 0;
    std::size_t evaluationIndex = 0; // its index in the evaluation's constraints
};

struct ApproximationObjective
{
    bool isPresent = false;
    std::int32_t responseId = 0;
    std::string responseType;
    bool isMaximised = false;
    double value = 0.0;
    std::vector<double> gradient;
    std::vector<int> powers;                   // ApproximationKind::Hints
    std::optional<EigenApproximation> eigen{}; // an eigenvalue objective
};

// What an approximate problem is made of (built by ApproximateProblem::fromDesign, or by hand in tests).
struct ApproximationInputs
{
    std::vector<ApproximationVariable> variables;
    std::vector<ApproximationLinkedVariable> linkedVariables;
    std::vector<ApproximationProperty> properties;
    std::vector<ApproximationItem> items;
    std::vector<ApproximationConstraint> constraints;
    ApproximationObjective objective;
};

struct ApproximationSettings
{
    ApproximationKind kind = ApproximationKind::Mixed; // APRCOD = 2, the default of DesignOptimParams
    MoveLimitParameters moveLimits;
    double constraintScale = ConstraintNormalisation::DEFAULT_CONSTRAINT_SCALE;
};

// What a constraint of the approximate problem constrains.
enum class ApproximateConstraintKind : std::uint8_t
{
    Response,       // a retained constraint on a response item
    LinkedVariable, // a bound (move-limited) of a dependent design variable
    Property        // a limit (move limit, PMIN, PMAX) of a designed property not reduced to a variable bound
};

struct ApproximateConstraintInfo
{
    ApproximateConstraintKind kind = ApproximateConstraintKind::Response;
    std::size_t source = 0; // the item, linked variable or property index
    BoundSide side = BoundSide::Upper;
    double bound = 0.0;
};

// The approximate problem of a design cycle, exposed to the caller: built
// once from the values and gradients of the retained responses at x0, it evaluates, with no analysis,
//   - the bounds of x: [XLB, XUB] cut by the move limits (MoveLimits) and by the property limits of the properties a
//     single variable drives linearly;
//   - the retained response items and their gradients, in the intermediate variables of the APRCOD
//     (ResponseApproximation; mixed: reciprocal for the structural responses, direct for the mass ones and the
//     equations);
//   - the constraints, the normalised g (ConstraintNormalisation): the retained response constraints, then the
//     bounds of the dependent variables (exact, DesignVarLink being linear), then the limits of the other designed
//     properties (linearised); under the convex linearisation each constraint takes 1/x where its own normalised
//     gradient is negative;
//   - the objective (minimised as is, or its opposite when maximised) approximated as a response of its type.
// It is the OptimisationProblem the optimisers solve, and the one the caller gets from problem.approximate(x0).
//
// Invariants: x0 inside the bounds; the item gradients are finite (built from gradients that covered every retained
// item). Thread-safety: immutable; every query may run concurrently. Ownership: owns its data; the budget must
// outlive it.
class ApproximateProblem final : public OptimisationProblem
{
public:
    // Throws OptimisationError for inconsistent sizes, non-finite gradients, or an x0 outside its own bounds.
    ApproximateProblem(ApproximationInputs inputs, ApproximationSettings settings, base::MemoryBudget& budget);

    // The approximate problem of a design point: the independent variables, links and relations of the problem's
    // design model at the evaluation's point, the retained constraints of `retained` with the gradients of their items
    // (from DesignProblem::gradients with retained.gradientOptions()), the objective. With ApproximationKind::Hints the
    // powers of every item and of the objective come from `hints` (ApproximationHints::compute of the same point).
    // Throws OptimisationError when a retained item has no gradient, or the hints are asked for and not given.
    [[nodiscard]] static ApproximateProblem fromDesign(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                                       const DesignGradients& gradients,
                                                       const RetainedResponses& retained,
                                                       const ApproximationSettings& settings,
                                                       base::MemoryBudget& budget,
                                                       const ApproximationHintsResult* hints = nullptr);

    // OptimisationProblem.
    [[nodiscard]] std::size_t variableCount() const override { return m_x0.size(); }
    [[nodiscard]] std::size_t constraintCount() const override { return m_constraints.size(); }
    [[nodiscard]] std::span<const double> lowerBounds() const override { return m_lower; }
    [[nodiscard]] std::span<const double> upperBounds() const override { return m_upper; }
    [[nodiscard]] double evaluate(std::span<const double> x, std::span<double> constraints) const override;
    void gradients(std::span<const double> x, std::span<double> objectiveGradient, std::span<const std::size_t> rows,
                   std::span<double> constraintGradients) const override;

    [[nodiscard]] const std::vector<double>& x0() const noexcept { return m_x0; }
    [[nodiscard]] const ApproximationSettings& settings() const noexcept { return m_settings; }
    [[nodiscard]] const ApproximationInputs& inputs() const noexcept { return m_inputs; }
    [[nodiscard]] const std::vector<ApproximateConstraintInfo>& constraintInfo() const noexcept
    {
        return m_constraints;
    }

    // The objective (in its own sense, not negated) and its gradient; 0 and zeros without an objective.
    [[nodiscard]] double objective(std::span<const double> x) const;
    void objectiveGradient(std::span<const double> x, std::span<double> gradient) const;
    // The approximated retained items (ApproximationInputs::items order) and their gradients (items × n, row-major).
    [[nodiscard]] std::vector<double> itemValues(std::span<const double> x) const;
    [[nodiscard]] std::vector<double> itemGradients(std::span<const double> x) const;
    // Every constraint: normalised g, or the approximated value it bounds (the response, the dependent variable or the
    // property).
    [[nodiscard]] std::vector<double> constraintValues(std::span<const double> x, bool isNormalised) const;
    // The largest normalised constraint (GMAX's g); −∞ without constraints.
    [[nodiscard]] double maxViolation(std::span<const double> x) const;

private:
    void checkPoint(std::span<const double> x) const;
    [[nodiscard]] ApproximationKind itemKind(const ApproximationItem& item) const noexcept;
    // The approximated value and gradient of an item or of the objective, its eigenvalue form included.
    [[nodiscard]] double approximatedValue(const ApproximationItem& item, double value0,
                                           std::span<const double> gradient0, std::span<const double> x) const;
    void approximatedGradient(const ApproximationItem& item, std::span<const double> gradient0,
                              std::span<const double> x, std::span<double> gradient) const;
    [[nodiscard]] std::span<const double> itemGradient(std::size_t item) const noexcept;
    [[nodiscard]] double constraintValue(std::size_t index, std::span<const double> x) const;
    void constraintGradient(std::size_t index, std::span<const double> x, std::span<double> gradient) const;
    [[nodiscard]] double linearValue(double value0, std::span<const double> gradient0,
                                     std::span<const double> x) const noexcept;

    ApproximationInputs m_inputs;
    ApproximationSettings m_settings;
    ConstraintNormalisation m_normalisation;
    std::vector<double> m_x0;
    std::vector<double> m_lower;
    std::vector<double> m_upper;
    ResponseApproximation m_approximation;
    base::BudgetedBuffer<double> m_itemGradients; // items × n, the gradients at x0
    std::vector<ApproximateConstraintInfo> m_constraints;
};

} // namespace oos::design
