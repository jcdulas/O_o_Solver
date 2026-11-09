#include "oos/assembly/assemble/NonStructuralMassShares.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace oos::assembly
{

namespace
{

// How an element takes a non-structural mass (NSM: per unit length on line elements, per unit area on shells).
enum class MassMeasure : std::uint8_t
{
    None,
    PerLength,
    PerArea
};

MassMeasure measureOf(std::string_view card) noexcept
{
    constexpr std::array<std::string_view, 5> LINE_CARDS{"Rod2", "Rod2InlineSection", "Tube2", "Bar2", "Beam2"};
    constexpr std::array<std::string_view, 7> AREA_CARDS{"ShellQuad4",         "ShellTria3", "ShellQuad4Drilling",
                                                         "ShellTria3Drilling", "ShellQuad8", "ShellTria6",
                                                         "ShearPanel4"};
    for (const std::string_view line : LINE_CARDS)
        if (card == line)
            return MassMeasure::PerLength;
    for (const std::string_view area : AREA_CARDS)
        if (card == area)
            return MassMeasure::PerArea;
    return MassMeasure::None;
}

base::Diagnostic nsmProblem(std::string code, std::string_view card, std::int32_t setId, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = setId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

NonStructuralMassShares::NonStructuralMassShares(const ElementSet& set) : m_shares(set.batches().size()) {}

NonStructuralMassShares::NonStructuralMassShares(const ElementSet& set, std::optional<base::EntityId> selected,
                                                 base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : NonStructuralMassShares(set)
{
    if (!selected)
        return;
    if (*selected <= 0)
    {
        sink.report(nsmProblem("nsm-invalid-selection", "NonStructMassSelection", *selected,
                               "expected a positive non-structural mass set ID"));
        return;
    }
    const auto& snapshot = set.index().snapshot();
    *this = NonStructuralMassShares(
        set, snapshot.hasTable(input::TableRole::NonStructuralMass) ? &snapshot.masses() : nullptr, *selected, budget,
        sink);
}

NonStructuralMassShares::NonStructuralMassShares(const ElementSet& set, const input::MassTable* table,
                                                 base::EntityId setId, base::MemoryBudget& budget,
                                                 base::DiagnosticSink& sink)
    : m_shares(set.batches().size())
{
    if (table == nullptr)
    {
        sink.report(nsmProblem("nsm-unknown-set", "NonStructMassSelection", setId,
                               "the selected non-structural mass set has no exported rows"));
        return;
    }
    const base::ContractReader& rows = table->reader();
    const std::span<const std::int32_t> setIds = rows.values<std::int32_t>("setId");
    const base::DictionaryView cards = rows.dictionaryTexts("card");
    const base::DictionaryView targetTypes = rows.dictionaryTexts("targetType");
    const std::span<const std::int32_t> targetIds = rows.values<std::int32_t>("targetId", base::NullPolicy::Allow);
    const base::ValidityView validTargets = rows.validity("targetId");
    const base::DictionaryView kinds = rows.dictionaryTexts("kind");
    const std::span<const double> values = rows.values<double>("value", base::NullPolicy::Allow);

    const auto shareOf = [&](std::size_t b) -> base::BudgetedBuffer<double>&
    {
        if (m_shares[b].empty())
            m_shares[b] = base::BudgetedBuffer<double>(
                budget, static_cast<std::size_t>(set.batches()[b].view.rowCount()), "non-structural mass shares");
        return m_shares[b];
    };

    bool hasSelection = false;
    for (std::int64_t row = 0; row < rows.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        if (setIds[index] != setId)
            continue;
        hasSelection = true;
        const std::string_view card = cards[row];
        const std::string_view kind = kinds[row];
        if (!rows.validity("value").isValid(row) || !std::isfinite(values[index]) || values[index] < 0.0)
        {
            sink.report(
                nsmProblem("nsm-invalid-value", card, setId, "expected a finite, nonnegative non-structural mass"));
            continue;
        }
        if (kind == "lumped")
        {
            sink.report(nsmProblem(
                "nsm-lumped-unsupported", card, setId,
                base::concatenate({"expected NonStructMass or NonStructMass1 (a mass per unit length or area), found ",
                                   card,
                                   " (a lumped mass shared between elements), which this version does not distribute; "
                                   "write the mass as NonStructMass or NonStructMass1 per unit length or area"})));
            continue;
        }
        if (kind != "perArea" && kind != "perLength")
        {
            sink.report(
                nsmProblem("nsm-unknown-kind", card, setId, "expected a perArea or perLength non-structural mass"));
            continue;
        }
        const MassMeasure wanted = kind == "perArea" ? MassMeasure::PerArea : MassMeasure::PerLength;
        const bool isElement = targetTypes[row] == "element";
        if (!validTargets.isValid(row) || (!isElement && targetTypes[row] != "property"))
        {
            sink.report(nsmProblem("nsm-unknown-target", card, setId,
                                   base::concatenate({"expected a property or an element as target, found the target "
                                                      "type '",
                                                      targetTypes[row], "'; name the properties or elements"})));
            continue;
        }
        const std::int32_t target = targetIds[index];
        bool isFound = false;
        bool isMismatched = false;
        for (std::size_t b = 0; b < set.batches().size(); ++b)
        {
            const input::ElementBatch& batch = set.batches()[b].view.batch();
            const MassMeasure measure = measureOf(batch.card());
            const std::span<const std::int32_t> keys =
                isElement ? batch.ids() : (batch.hasProperty() ? batch.propertyIds() : std::span<const std::int32_t>());
            for (std::size_t r = 0; r < keys.size(); ++r)
            {
                if (keys[r] != target)
                    continue;
                isFound = true;
                if (measure != wanted)
                {
                    isMismatched = true;
                    continue;
                }
                shareOf(b)[r] += values[index];
            }
        }
        if (!isFound)
            sink.report(nsmProblem(
                "nsm-unknown-target", card, setId,
                base::concatenate({"expected an existing ", isElement ? "element" : "property", ", found ID ",
                                   std::to_string(target), " which no element of the model has; correct the ID"})));
        else if (isMismatched)
            sink.report(nsmProblem("nsm-kind-mismatch", card, setId,
                                   base::concatenate({"the ", std::string(kind), " mass of target ",
                                                      std::to_string(target), " reaches elements that take ",
                                                      wanted == MassMeasure::PerArea ? "a mass per unit length or none"
                                                                                     : "a mass per unit area or none",
                                                      "; name only line elements (per length) or shells (per area)"})));
    }
    if (!hasSelection)
        sink.report(nsmProblem("nsm-unknown-set", "NonStructMassSelection", setId,
                               "the selected non-structural mass set has no exported rows"));
}

} // namespace oos::assembly
