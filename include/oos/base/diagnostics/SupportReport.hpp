#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"

#include <string>
#include <vector>

namespace oos::base
{

// What one module supports for the model at hand: the items it accepts and the items it refuses, each refusal as a
// diagnostic naming the card, the ID and the field. The caller gathers the reports of every module into one check
// report, so that every unsupported item is listed in one pass.
struct SupportReport
{
    std::string module;                 // the module that wrote the report, e.g. "elements"
    std::vector<std::string> supported; // the card types or options accepted, e.g. "ShellQuad4 (1200)"
    std::vector<Diagnostic> refusals;   // one per unsupported item, with Severity::Error
    std::vector<Diagnostic> ignored;    // cards with no effect on the analysis, with Severity::Warning

    [[nodiscard]] bool isComplete() const { return refusals.empty(); }
};

} // namespace oos::base
