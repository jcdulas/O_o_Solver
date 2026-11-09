#pragma once

#include <cstdint>
#include <string_view>

namespace oos::solutions
{

// The two paths of every analysis: the full path factorises the whole model; the condensed path condenses some units
// onto their boundaries, solves the interface problem, and recovers the units whose results are asked for. The path
// changes the cost and the memory, never what is computed.
enum class AnalysisPath : std::uint8_t
{
    Full,
    Condensed,
    // The full path streamed: the full path's equations solved without the factor, no matrix of the model assembled.
    Streamed,
    // The full path's matrices solved by multilevel substructuring: approximate to its cut-off.
    Substructuring,
    // The full path with its factor out of core.
    OutOfCore
};

[[nodiscard]] constexpr std::string_view nameOf(AnalysisPath path) noexcept
{
    switch (path)
    {
    case AnalysisPath::Full:
        return "full";
    case AnalysisPath::Condensed:
        return "condensed";
    case AnalysisPath::Streamed:
        return "streamed";
    case AnalysisPath::Substructuring:
        return "substructuring";
    case AnalysisPath::OutOfCore:
        return "out_of_core";
    }
    return "full";
}

} // namespace oos::solutions
