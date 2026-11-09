#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"

#include <cstdint>
#include <string>
#include <utility>

namespace oos::design
{

// The codes of the diagnostics of a design model: stable identifiers that the reports and the tests select on.
namespace design_code
{
inline constexpr const char* UNSUPPORTED_CARD = "unsupported-design-card";     // DesignDiscreteValues
inline constexpr const char* UNSUPPORTED_TARGET = "unsupported-design-target"; // a field the overlay cannot design
inline constexpr const char* INVALID_VARIABLE = "invalid-design-variable";     // bounds, IDs
inline constexpr const char* INVALID_LINK = "invalid-design-link";             // DesignVarLink
inline constexpr const char* INVALID_RELATION = "invalid-design-relation";     // DVxRELi
inline constexpr const char* MISSING_REFERENCE =
    "missing-design-reference"; // a DESVAR, DesignEquation or DesignConstants not defined
inline constexpr const char* MISSING_TARGET = "missing-design-target";         // a card, ID or field the model lacks
inline constexpr const char* FIELD_DRIVEN_TWICE = "field-driven-twice";        // two relations on one field
inline constexpr const char* INVALID_RESPONSE = "invalid-design-response";     // DesignResp1/2/3 fields, cycles
inline constexpr const char* UNSUPPORTED_RESPONSE = "unsupported-design-response"; // a type or option not supported
inline constexpr const char* MISSING_RESPONSE_TARGET = "missing-response-target";  // a grid, element, property absent
inline constexpr const char* INVALID_CONSTRAINT =
    "invalid-design-constraint";                                     // DesignConstr, DesignConstrUnion, DESSUB, DESOBJ
inline constexpr const char* INVALID_SHAPE = "invalid-design-shape"; // DesignNodeMove: its vector, its DESVAR
inline constexpr const char* SHAPE_DISTORTION = "shape-distortion"; // a design point that degenerates an element
} // namespace design_code

// An error diagnostic of a design card.
[[nodiscard]] inline base::Diagnostic designProblem(const char* code, std::string card, std::int64_t id,
                                                    std::string field, std::string message)
{
    return {base::Severity::Error, code, std::move(message), std::move(card), id, std::move(field)};
}

} // namespace oos::design
