#pragma once

#include "oos/design/constraints/ConstraintDefinitions.hpp"
#include "oos/design/constraints/ConstraintNormalisation.hpp"
#include "oos/design/responses/DesignResponses.hpp"
#include "oos/design/responses/ResponseValues.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace oos::design
{

// One constraint on one response item in one subcase: g ≤ 0 means satisfied. `normalised` is the normalised form
// (ConstraintNormalisation, DesignOptimParams GSCAL):
//   lower bound L: g = (L − r) / GNORM;   upper bound U: g = (r − U) / GNORM;   GNORM = |bound|, or GSCAL near zero.
struct ConstraintValue
{
    std::int32_t setId = 0;      // DCID; 0 for a caller's constraint without a set
    std::int32_t responseId = 0; // RID
    std::size_t item = 0;        // the index of the item in the response's values
    ResponseItem label;
    BoundSide side = BoundSide::Upper;
    double bound = 0.0;
    double value = 0.0;
    double normalised = 0.0;
};

// The constraints, constraint sets, Case Control selection, objective and screening rules of a design problem
// (constraints may be declared apart from the responses; DesignConstr, DesignConstrUnion, DESSUB, DESGLB, DESOBJ,
// DesignConstrScreen).
//
// Which constraints act where: in subcase s (s > 0), the constraints of the set its DESSUB selects (its own, or the
// global DESSUB it inherits), with DesignConstrUnion expanded (a union of sets, recursively); in the global section (s
// = 0), those of DESGLB. A constraint of set 0 (a caller's constraint declared without set_id) acts wherever its
// response is evaluated. A constraint restricted by subcases= acts only there. A constraint on a global response
// (WEIGHT, VOLUME, FRMASS, a global equation) acts once, at subcase 0, whichever selection names it. A response may
// have several constraints, with other bounds in other subcases or sets.
//
// The subcases of each response (subcasesOfResponses): those where a constraint on it acts, the subcase of the
// objective (DESOBJ inside a subcase, or subcase 0 for a global objective), its own subcases= when it has them; a
// response used by none of these (a monitored response) is evaluated in every analysed subcase.
//
// Thread-safety: immutable once built. Ownership: owns its parts.
class DesignConstraints
{
public:
    // DesignOptimParams GSCAL: the constraint normalisation factor for bounds near zero (default 0.001).
    static constexpr double DEFAULT_CONSTRAINT_SCALE = ConstraintNormalisation::DEFAULT_CONSTRAINT_SCALE;

    DesignConstraints() = default;
    // Throws OptimisationError (throwDesignDiagnostics) for a constraint on an unknown response, a constraint without
    // bounds, LOWFQ/HIGHFQ, a DesignConstrUnion of an unknown set or a cycle of unions, a DESSUB, DESGLB or DESOBJ that
    // names nothing defined, an objective response that is not one item.
    DesignConstraints(ConstraintParts parts, const DesignResponses& responses);

    [[nodiscard]] const ConstraintParts& parts() const noexcept { return m_parts; }
    [[nodiscard]] std::optional<std::int32_t> objective() const noexcept { return m_parts.selection.objective; }
    [[nodiscard]] bool isMaximised() const noexcept { return m_parts.selection.isMaximised; }
    [[nodiscard]] const std::vector<ScreeningRule>& screening() const noexcept { return m_parts.screening; }

    // The sets a set ID stands for (itself and, for a DesignConstrUnion, every member, recursively).
    [[nodiscard]] std::set<std::int32_t> expandSet(std::int32_t setId) const;

    // The subcases each response is evaluated in (before DesignResponses::closeSubcases).
    [[nodiscard]] ResponseSubcases subcasesOfResponses(std::span<const std::int32_t> analysed,
                                                       const DesignResponses& responses) const;

    // The value of every acting constraint on every item, in the order of the constraints, then of the subcases
    // (global first), then of the items; lower before upper. `values` are the responses' values (any order).
    [[nodiscard]] std::vector<ConstraintValue> evaluate(const std::vector<ResponseValues>& values,
                                                        std::span<const std::int32_t> analysed,
                                                        const DesignResponses& responses,
                                                        double constraintScale = DEFAULT_CONSTRAINT_SCALE) const;

    // The value of the objective (its single item), or nothing without one. Throws OptimisationError when its
    // response has not one item.
    [[nodiscard]] std::optional<double> objectiveValue(const std::vector<ResponseValues>& values) const;

private:
    // Whether constraint `c` acts in subcase `subcase` (0: global).
    [[nodiscard]] bool actsIn(const ConstraintDefinition& constraint, std::int32_t subcase,
                              const DesignResponses& responses) const;
    [[nodiscard]] std::optional<std::int32_t> selectedSet(std::int32_t subcase) const;

    ConstraintParts m_parts;
};

} // namespace oos::design
