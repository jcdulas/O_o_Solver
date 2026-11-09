#include "oos/base/diagnostics/Diagnostic.hpp"

#include <string>

namespace oos::base
{

const char* severityName(Severity severity) noexcept
{
    switch (severity)
    {
    case Severity::Information:
        return "information";
    case Severity::Warning:
        return "warning";
    case Severity::Error:
        return "error";
    }
    return "error";
}

std::string formatDiagnostic(const Diagnostic& diagnostic)
{
    // Appended piece by piece: GCC 12 reports a false -Wrestrict on `"literal" + std::string&&` (GCC bug 105329).
    std::string text = severityName(diagnostic.severity);
    text.append(" [").append(diagnostic.code).append("]");
    if (!diagnostic.card.empty())
    {
        text.append(" ").append(diagnostic.card);
        if (diagnostic.id != 0)
            text.append(" ").append(std::to_string(diagnostic.id));
        if (!diagnostic.field.empty())
            text.append(", field ").append(diagnostic.field);
    }
    text.append(": ").append(diagnostic.message);
    return text;
}

} // namespace oos::base
