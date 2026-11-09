#pragma once

#include "oos/base/diagnostics/SupportReport.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// A field value a supported card accepts only at its default: any other value is refused (an offset on a shell before
// offsets are implemented, a coupling material of a ShellProp). Numeric columns only (int32 or float64); a null value
// counts as the default. Card names, fields and refusal messages use neutral model names.
struct FieldRule
{
    std::string card;     // the neutral card name: "ShellQuad4"
    std::string field;    // the column, as the data contract names it: "offset"
    double acceptedValue; // the one value accepted (compared exactly: defaults are exact numbers)
    std::string reason;   // why other values are refused, for the message
};

// The cards and field values the solver supports, and the check of a snapshot against them: every
// card of the model that is not supported, and every field value a rule refuses, becomes a diagnostic of the
// SupportReport, so that the user sees all of them in one pass before any analysis starts. The cards are known by
// their neutral names (addCard("ShellQuad4")); the diagnostics, and the list of supported cards of the report, name
// them by their deck keywords (CQUAD4).
//
// current() is empty: the modules that implement cards add them (the element registry, solutions::RunCatalog), and
// every element, property, material, load and constraint card none of them adds is refused with its name and IDs.
//
// Thread-safety: immutable once built; check() may run concurrently. Ownership: plain data.
class SupportCatalog
{
public:
    // The catalog of this build of O_o_Solver.
    [[nodiscard]] static SupportCatalog current();

    void addCard(std::string card);
    void addFieldRule(FieldRule rule);

    [[nodiscard]] bool isSupported(std::string_view card) const;

    // One refusal per unsupported card type present (with its first ID, its count and up to MAX_LISTED_IDS of its IDs
    // in the message), and one per element or property row whose field breaks a rule; sorted by card keyword, ID and
    // field.
    [[nodiscard]] base::SupportReport check(const ModelSnapshot& snapshot) const;

    // The number of IDs a refusal message lists before "and N more".
    static constexpr std::size_t MAX_LISTED_IDS = 10;

private:
    std::set<std::string, std::less<>> m_cards;
    std::vector<FieldRule> m_fieldRules;
};

} // namespace oos::input
