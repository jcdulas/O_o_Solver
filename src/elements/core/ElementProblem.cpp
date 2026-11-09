#include "oos/elements/core/ElementProblem.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>

namespace oos::elements
{

base::Diagnostic elementProblem(const ElementBatchView& view, std::int64_t row, std::string code, std::string field,
                                std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = view.card();
    diagnostic.id = view.elementId(row);
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

void throwUncheckedElement(const base::Diagnostic& problem)
{
    throw std::invalid_argument(base::concatenate(
        {base::formatDiagnostic(problem),
         " (an element facet was called on an element its family's check() refuses: run check() on the batch and "
         "stop on its errors first)"}));
}

} // namespace oos::elements
