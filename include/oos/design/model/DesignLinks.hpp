#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/design/model/DesignVariables.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

// One term of a DesignVarLink: an independent variable and its coefficient Ci.
struct LinkTerm
{
    std::int32_t variableId = 0; // IDVi
    double coefficient = 0.0;    // Ci
};

// One DesignVarLink: DDVID = C0 + CMULT · Σ Ci · IDVi.
struct DesignLink
{
    std::int32_t id = 0;          // the DesignVarLink ID (ID 0 for a link the caller declares without one)
    std::int32_t dependentId = 0; // DDVID
    double constant = 0.0;        // C0
    double multiplier = 1.0;      // CMULT
    std::vector<LinkTerm> terms;
};

// The variable values of a design point: every variable of the model, the dependent ones computed from the
// independent ones, and the derivative of each variable with respect to the independent ones (the columns of x).
struct VariableValues
{
    std::vector<double> values; // one per variable, in the model's order
    // Per variable, the non-zero derivatives d(variable)/d(x_k) as (k, derivative): (k, 1) for the independent
    // variable of column k; (k, CMULT·Ci) summed over the terms of the same column for a dependent one.
    std::vector<std::vector<std::pair<std::size_t, double>>> derivatives;
};

// The DLINKs of a design model and the split of the variables into independent ones (the columns of x, the vector the
// optimiser moves) and dependent ones (computed by their link). The optimisers move the independent variables only;
// the links derive the others (variable linking).
//
// Checks (check()): every IDVi and DDVID names a DESVAR; a variable is dependent in one DesignVarLink at most; a
// dependent variable is never an independent variable of a link (a DDVID cannot be an IDVi), so one pass computes every
// dependent variable; a link has at least one term.
//
// Thread-safety: immutable once built. Ownership: owns its links.
class DesignLinks
{
public:
    DesignLinks() = default;
    // Precondition for independentIndices() and expand(): check() gave no error.
    DesignLinks(std::vector<DesignLink> links, const DesignVariables& variables);

    [[nodiscard]] const std::vector<DesignLink>& all() const noexcept { return m_links; }
    // The indices (in the model's variable order) of the independent variables: the columns of x, in that order.
    [[nodiscard]] const std::vector<std::size_t>& independentIndices() const noexcept { return m_independent; }
    // The link of each variable that is dependent, or -1, by variable index.
    [[nodiscard]] std::int64_t linkOf(std::size_t variableIndex) const { return m_linkOfVariable[variableIndex]; }

    void check(const DesignVariables& variables, std::vector<base::Diagnostic>& diagnostics) const;

    // Every variable's value and its derivatives with respect to x (the independent values, in the order of
    // independentIndices()). std::invalid_argument when x does not have one value per independent variable.
    [[nodiscard]] VariableValues expand(std::span<const double> x, const DesignVariables& variables) const;

private:
    std::vector<DesignLink> m_links;
    std::vector<std::int64_t> m_linkOfVariable;
    std::vector<std::size_t> m_independent;
};

} // namespace oos::design
