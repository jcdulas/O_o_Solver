#pragma once

#include <cstdint>
#include <string>

namespace oos::base
{

// How serious a diagnostic is. Same meaning as O_o_Reader's severities (own code).
enum class Severity
{
    Information,
    Warning,
    Error
};

// One problem found in the model or in a run, with where it is: the card, its ID and the field, and what to do about
// it. A diagnostic is data: it never stops a computation by itself. The solver writes the card and the field by their
// neutral names, exactly as model files and the public interfaces do.
struct Diagnostic
{
    Severity severity = Severity::Error;
    std::string code;    // a stable identifier, e.g. "unsupported-card"
    std::string message; // what was expected, what was found, and what to do
    std::string card;    // "ShellQuad4", "Parameter AutoSingularityConstr"; empty when not tied to a card
    std::int64_t id = 0; // the card's ID, 0 when it has none
    std::string field;   // the neutral name of the field of the card ("offset"), empty when not tied to a field
};

// The name of a severity, as messages and reports print it.
[[nodiscard]] const char* severityName(Severity severity) noexcept;

// A one-line text of a diagnostic: "error [unsupported-card] ShellQuad4 12, field offset: message", in keywords once
// labelled ("error [unsupported-card] CQUAD4 12, field ZOFFS: message").
[[nodiscard]] std::string formatDiagnostic(const Diagnostic& diagnostic);

} // namespace oos::base
