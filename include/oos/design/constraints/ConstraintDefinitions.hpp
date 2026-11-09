#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oos::design
{

// One DesignConstr entry, or a constraint declared by the caller (add_constraint, the shorthand bounds of
// add_response):
//
//   | DesignConstr | DCID | RID | LALLOW | UALLOW | LOWFQ | HIGHFQ |
//
// A blank LALLOW or UALLOW is nothing here (defaults −1.0E20 and 1.0E20: no bound on that side). LOWFQ and HIGHFQ
// select frequencies of dynamic responses and are refused on static responses. `subcases` restricts the constraint to
// some subcases (the subcases= of a caller's declaration); empty, the DESSUB of each subcase decides.
struct ConstraintDefinition
{
    std::int32_t setId = 0;      // DCID
    std::int32_t responseId = 0; // RID: a DesignResp1, DesignResp2 or DesignResp3 ID
    std::optional<double> lower;
    std::optional<double> upper;
    std::vector<std::int32_t> subcases;
};

// One DesignConstrUnion entry: the constraint set `setId` is the union of the sets `members` (DCONSTR sets or
// other DesignConstrUnion sets).
struct ConstraintSetUnion
{
    std::int32_t setId = 0;
    std::vector<std::int32_t> members;
};

// The design commands of the Case Control (DESSUB, DESGLB, DESOBJ), effective per subcase. Subcase 0 is the
// global section: DESGLB (the constraints of the responses that do not depend on a subcase) and a DESOBJ given above
// the subcases. A subcase without its own DESSUB inherits the global one (in the Case Control, a command above the
// subcases applies to every subcase that does not repeat it).
struct DesignSelection
{
    struct Subcase
    {
        std::int32_t subcase = 0;
        std::optional<std::int32_t> dessub;
    };
    std::vector<Subcase> subcases;      // the DESSUB of each subcase that gives one (subcase 0: the global DESSUB)
    std::optional<std::int32_t> desglb; // DESGLB
    std::optional<std::int32_t> objective;
    bool isMaximised = false;                     // DESOBJ(MAX)
    std::optional<std::int32_t> objectiveSubcase; // the subcase of a DESOBJ given inside a subcase; nothing: global
};

// One DesignConstrScreen entry: constraints of the response type `responseType` ("Stress", "Displacement", "Equation"
// for the equation responses...) whose normalised value is below `threshold` are screened out, and at most
// `maximumRetained` are kept per region and subcase. Stored here for the constraint screening.
struct ScreeningRule
{
    std::string responseType;
    double threshold = -0.5;           // TRS, default −0.5
    std::int32_t maximumRetained = 20; // NSTR, default 20
};

// The constraints, sets, Case Control selection and screening rules of a design problem, as read (ConstraintTables) or
// declared; checked when DesignConstraints is made of them.
struct ConstraintParts
{
    std::vector<ConstraintDefinition> constraints;
    std::vector<ConstraintSetUnion> unions;
    DesignSelection selection;
    std::vector<ScreeningRule> screening;
};

} // namespace oos::design
