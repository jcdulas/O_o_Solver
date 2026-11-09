#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"

#include <cstddef>
#include <mutex>
#include <vector>

namespace oos::base
{

// Collects the diagnostics of a run. Thread-safe: element batches running in parallel report into one sink. The order
// of the diagnostics returned is made independent of the thread timing by sorting them (severity, card, ID, field,
// code), so that two runs give the same report; a diagnostic reported twice with the same fields is listed (and
// counted) once.
class DiagnosticSink
{
public:
    void report(Diagnostic diagnostic);

    // The diagnostics collected so far, in their stable order.
    [[nodiscard]] std::vector<Diagnostic> diagnostics() const;

    [[nodiscard]] std::size_t count(Severity severity) const;
    [[nodiscard]] bool hasErrors() const { return count(Severity::Error) > 0; }

private:
    mutable std::mutex m_mutex;
    std::vector<Diagnostic> m_diagnostics;
};

} // namespace oos::base
