#include "oos/base/diagnostics/DiagnosticSink.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

namespace oos::base
{

void DiagnosticSink::report(Diagnostic diagnostic)
{
    std::lock_guard lock(m_mutex);
    m_diagnostics.push_back(std::move(diagnostic));
}

std::vector<Diagnostic> DiagnosticSink::diagnostics() const
{
    std::vector<Diagnostic> sorted;
    {
        std::lock_guard lock(m_mutex);
        sorted = m_diagnostics;
    }
    // Errors first, then by card, ID, field and code: the order does not depend on which thread reported first.
    const auto keyOf = [](const Diagnostic& diagnostic)
    {
        return std::tie(diagnostic.severity, diagnostic.card, diagnostic.id, diagnostic.field, diagnostic.code,
                        diagnostic.message);
    };
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const Diagnostic& left, const Diagnostic& right)
                     {
                         return std::tuple(-static_cast<int>(left.severity), left.card, left.id, left.field, left.code,
                                           left.message) < std::tuple(-static_cast<int>(right.severity), right.card,
                                                                      right.id, right.field, right.code, right.message);
                     });
    // A problem reported twice the same way (two passes over one card: the load checks read a LOAD combination's sets
    // once, which the basis vectors made of them check again) is listed once.
    sorted.erase(std::unique(sorted.begin(), sorted.end(), [&](const Diagnostic& left, const Diagnostic& right)
                             { return keyOf(left) == keyOf(right); }),
                 sorted.end());
    return sorted;
}

std::size_t DiagnosticSink::count(Severity severity) const
{
    const std::vector<Diagnostic> listed = diagnostics();
    return static_cast<std::size_t>(std::count_if(listed.begin(), listed.end(),
                                                  [severity](const Diagnostic& d) { return d.severity == severity; }));
}

} // namespace oos::base
