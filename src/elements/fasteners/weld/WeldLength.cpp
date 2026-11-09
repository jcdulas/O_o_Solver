#include "oos/elements/fasteners/weld/WeldElement.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

namespace oos::elements
{

namespace
{

// The shell cards whose elements a Weld ELPAT or ELEMID names (SHIDA, SHIDB).
constexpr std::array<std::string_view, 6> SHELL_CARDS{"ShellQuad4",        "ShellTria3", "ShellQuad4Drilling",
                                                      "ShellTria3Drilling", "ShellQuad8", "ShellTria6"};

// The row of `id` in a column of IDs: by bisection when the column is sorted there (O_o_Reader sorts its card tables
// by ID), else by a scan.
std::optional<std::size_t> findIdRow(std::span<const std::int32_t> ids, std::int32_t id)
{
    const auto found = std::lower_bound(ids.begin(), ids.end(), id);
    if (found != ids.end() && *found == id)
        return static_cast<std::size_t>(found - ids.begin());
    const auto scanned = std::find(ids.begin(), ids.end(), id);
    if (scanned != ids.end())
        return static_cast<std::size_t>(scanned - ids.begin());
    return std::nullopt;
}

// The thickness of a shell property: the T of a ShellProp (when given), else the total thickness of a CompProp or
// CompPropGlobal (laminateSummary).
std::optional<double> propertyThickness(const input::ModelSnapshot& snapshot, std::int32_t propId)
{
    if (const input::CardTable* shellProp = snapshot.findPropertyTable("ShellProp"))
    {
        if (const std::optional<std::size_t> row = findIdRow(shellProp->ids(), propId))
        {
            const base::ContractReader& reader = shellProp->reader();
            if (!reader.validity("thickness").isValid(static_cast<std::int64_t>(*row)))
                return std::nullopt;
            return reader.values<double>("thickness", base::NullPolicy::Allow)[*row];
        }
    }
    if (snapshot.hasTable(input::TableRole::LaminateSummary))
    {
        const base::ContractReader& summary = snapshot.laminateSummary();
        if (const std::optional<std::size_t> row = findIdRow(summary.values<std::int32_t>("propId"), propId))
            return summary.values<double>("thickness")[*row];
    }
    return std::nullopt;
}

// The property of the shell element `elementId`.
std::optional<std::int32_t> elementProperty(const input::ModelSnapshot& snapshot, std::int32_t elementId)
{
    for (const std::string_view card : SHELL_CARDS)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card);
        if (batch == nullptr || !batch->hasProperty())
            continue;
        if (const std::optional<std::size_t> row = findIdRow(batch->ids(), elementId))
            return batch->propertyIds()[*row];
    }
    return std::nullopt;
}

} // namespace

std::optional<std::int32_t> weldSideProperty(const input::ModelSnapshot& snapshot, std::string_view type,
                                             std::int32_t id)
{
    if (id <= 0)
        return std::nullopt;
    if (type == "PartPatch")
        return id;
    if (type == "ElemPatch" || type == "ElemIdPatch")
        return elementProperty(snapshot, id);
    return std::nullopt;
}

std::optional<double> weldSideThickness(const input::ModelSnapshot& snapshot, std::string_view type, std::int32_t id)
{
    const std::optional<std::int32_t> propId = weldSideProperty(snapshot, type, id);
    if (!propId)
        return std::nullopt;
    return propertyThickness(snapshot, *propId);
}

double weldBeamLength(double distance, std::optional<double> thicknessA, std::optional<double> thicknessB,
                      double diameter, double ldmin, double ldmax)
{
    double length = distance;
    if (thicknessA && thicknessB)
        length = std::max(distance, 0.5 * (*thicknessA + *thicknessB));
    return std::min(std::max(length, ldmin * diameter), ldmax * diameter);
}

} // namespace oos::elements
