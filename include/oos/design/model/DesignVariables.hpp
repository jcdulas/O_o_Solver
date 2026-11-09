#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

// One design variable (DESVAR): ID, LABEL, XINIT, XLB, XUB, DELXV, DesignDiscreteValues. `discreteValues` is the
// DesignDiscreteValues ID of a discrete variable, which is refused (discrete values are not supported yet).
struct DesignVariable
{
    std::int32_t id = 0;
    std::string label;
    double initialValue = 0.0;       // XINIT
    double lower = 0.0;              // XLB
    double upper = 0.0;              // XUB
    std::optional<double> moveLimit; // DELXV, the fractional move limit of the variable (DOPTPRM DELX if blank)
    std::optional<std::int32_t> discreteValues; // DesignDiscreteValues
};

// The design variables of a design model, in the order given (a deck's by ID, then those the caller declares), with
// their lookup by ID.
//
// Checks (check()): IDs positive and unique; XLB ≤ XINIT ≤ XUB; DesignDiscreteValues refused, naming it.
//
// Thread-safety: immutable once built. Ownership: owns its variables.
class DesignVariables
{
public:
    DesignVariables() = default;
    explicit DesignVariables(std::vector<DesignVariable> variables);

    [[nodiscard]] std::size_t size() const noexcept { return m_variables.size(); }
    [[nodiscard]] const std::vector<DesignVariable>& all() const noexcept { return m_variables; }
    [[nodiscard]] const DesignVariable& operator[](std::size_t index) const { return m_variables[index]; }
    // The index of the variable of an ID, or nothing.
    [[nodiscard]] std::optional<std::size_t> findIndex(std::int32_t id) const noexcept;

    // The problems of the variables, as diagnostics of the card DESVAR.
    void check(std::vector<base::Diagnostic>& diagnostics) const;

private:
    std::vector<DesignVariable> m_variables;
    std::vector<std::pair<std::int32_t, std::size_t>> m_indexById; // sorted by ID
};

} // namespace oos::design
