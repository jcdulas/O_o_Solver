#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/EigenvalueTable.hpp"
#include "oos/output/arrays/ResultArrays.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace oos::api
{

// The analyses a model can be run with.
enum class Analysis : std::uint8_t
{
    Statics,  // linear statics (SOL 101)
    Modes,    // real eigenvalues, normal modes (SOL 103)
    Buckling, // linear buckling (SOL 105): a static preload, then the load factors
};

// What an analysis is asked to do. Every field has a default that runs the deck as it is written.
struct AnalysisSettings
{
    // The subcases to solve, in this order; empty: every subcase of the Case Control.
    std::vector<std::int32_t> subcases;
    std::size_t threads = 1;
    // The memory the analysis may use, in bytes; 0: what is left in the module's memory.
    std::uint64_t memoryBytes = 0;
};

// The outcome of an analysis: the result tables, the eigenvalues and the warnings.
struct AnalysisOutput
{
    // The memory budget of the run: the result buffers hold their reservations in it, so it is declared first and
    // destroyed last.
    std::shared_ptr<base::MemoryBudget> budget;
    // The nodal and element tables, in the layout of O_o_Reader's results ({kind, subcase, step, ids, values}).
    output::ResultArrays tables;
    // The eigenvalue table of each modal or buckling subcase (empty for statics).
    std::vector<output::EigenvalueTable> eigenvalues;
    // The warnings and information of the run (an error stops the run with an exception instead).
    std::vector<base::Diagnostic> diagnostics;
};

// The cards and fields of the model the analysis cannot take (refusals: errors) or will not use (ignored: warnings),
// all found in one pass before anything is computed.
// Supported cards and diagnostics use the neutral model card types ("ShellQuad4") and field names ("offset").
struct SupportCheck
{
    std::vector<std::string> supported;
    std::vector<base::Diagnostic> refusals;
    std::vector<base::Diagnostic> ignored;
};

// The formulations of one element card, the default first (elements/core/ElementFormulation.hpp).
struct ElementFormulations
{
    std::string card;
    std::vector<std::string> formulations;
};

// Every element card the solver computes, with its formulations, in the order of the element library.
[[nodiscard]] std::vector<ElementFormulations> elementFormulations();

[[nodiscard]] SupportCheck checkSupport(const input::ModelSnapshot& snapshot, Analysis analysis);

// Runs the analysis on the model. Throws UnsupportedCardError when a card is refused (checkSupport lists them all),
// SolverError for a problem of the model (a mechanism, a singular matrix, a missing card), MemoryBudgetError when the
// memory does not suffice.
[[nodiscard]] AnalysisOutput runAnalysis(const input::ModelSnapshot& snapshot, Analysis analysis,
                                         const AnalysisSettings& settings);

// The name and version of the solver: "O_o_Solver 0.2.0".
[[nodiscard]] std::string programName();

} // namespace oos::api
