#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/elements/core/ElementBatchView.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace oos::elements
{

// The stable codes of the element diagnostics.
namespace problem_code
{
inline constexpr const char* MISSING_POINT = "missing-point";         // a grid or scalar point ID not in the model
inline constexpr const char* INVALID_COMPONENT = "invalid-component"; // a component outside 1-6 (0 for a scalar)
inline constexpr const char* DUPLICATE_DOF = "duplicate-dof";         // the two ends of a scalar element coincide
inline constexpr const char* MISSING_PROPERTY = "missing-property";   // a PID not in the model
inline constexpr const char* MISSING_MATERIAL = "missing-material";   // a MID not in the model
inline constexpr const char* UNSUPPORTED_MATERIAL = "unsupported-material"; // a material type the element cannot use
inline constexpr const char* DEGENERATE_GEOMETRY = "degenerate-geometry";   // zero length, parallel orientation vector
inline constexpr const char* MISSING_COORDINATE_SYSTEM = "missing-coordinate-system"; // a CID not in the model
inline constexpr const char* UNSUPPORTED_FIELD = "unsupported-field";                 // a field value not supported yet
inline constexpr const char* INVALID_FIELD = "invalid-field"; // a field value the input format does not allow
inline constexpr const char* IGNORED_FIELD = "ignored-field"; // a field value the solver does not use (information)
} // namespace problem_code

// The diagnostic of a problem of the element of batch row `row`: its card, its ID, the field as the card names it,
// a stable code (problem_code) and the message (what was expected, what was found, what to do).
[[nodiscard]] base::Diagnostic elementProblem(const ElementBatchView& view, std::int64_t row, std::string code,
                                              std::string field, std::string message);

// Throws std::invalid_argument for a facet called on an element that check() refuses: the caller skipped the check.
[[noreturn]] void throwUncheckedElement(const base::Diagnostic& problem);

// The resolved data of an element, or the problem that keeps it from being resolved. The resolve functions of the
// families return one; check() reports the problems, the facets throw on them (throwUncheckedElement).
template <typename T>
struct Resolution
{
    std::optional<T> value;
    base::Diagnostic problem;

    [[nodiscard]] static Resolution failure(base::Diagnostic diagnostic)
    {
        return {std::nullopt, std::move(diagnostic)};
    }
    [[nodiscard]] static Resolution success(T data) { return {std::move(data), {}}; }
    [[nodiscard]] explicit operator bool() const noexcept { return value.has_value(); }
    // The value, or throwUncheckedElement. On a temporary the value is moved out, so that
    // `const T& data = resolve(...).required();` does not dangle.
    [[nodiscard]] const T& required() const&
    {
        if (!value)
            throwUncheckedElement(problem);
        return *value;
    }
    [[nodiscard]] T required() &&
    {
        if (!value)
            throwUncheckedElement(problem);
        return std::move(*value);
    }
};

// Reports the problem of every row of the batch that `resolve(row)` (returning a Resolution) cannot resolve.
template <typename Resolve>
void checkEveryRow(const ElementBatchView& view, base::DiagnosticSink& sink, Resolve&& resolve)
{
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        auto resolution = resolve(row);
        if (!resolution)
            sink.report(std::move(resolution.problem));
    }
}

} // namespace oos::elements
