#include "oos/elements/shells/ShellTemperature.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/tables/NodeTable.hpp"

#include <optional>
#include <string>

namespace oos::elements
{

Resolution<ShellTemperatureChange> resolveShellTemperature(const ElementBatchView& view, std::int64_t row,
                                                           const ElementTemperatures& temperatures,
                                                           std::span<const std::int32_t> cornerRows,
                                                           const ShellSection& section)
{
    using Result = Resolution<ShellTemperatureChange>;
    const std::int32_t elementId = view.elementId(row);
    ShellTemperatureChange change;
    // A subcase with element deformations and no temperature load: no thermal strain (DEFORM loads no shell).
    if (!temperatures.hasTemperatureLoad)
        return Result::success(change);
    std::array<double, 8> loadAtCorners{};
    bool isLoadAtCorners = false;
    double loadMean = 0.0;
    double loadGradient = 0.0;
    if (const ElementTemperature* element = temperatures.load.ofElement(elementId))
    {
        loadMean = element->mean;
        loadGradient = element->gradient;
    }
    else
    {
        isLoadAtCorners = true;
        std::size_t k = 0;
        for (const std::int32_t corner : cornerRows)
        {
            const std::optional<double> value = temperatures.load.atGridRow(corner);
            if (!value)
                return Result::failure(elementProblem(
                    view, row, problem_code::INVALID_FIELD, "TemperatureSelection(LOAD)",
                    base::concatenate(
                        {"expected a temperature at grid point ",
                         std::to_string(view.snapshot().nodes().ids()[static_cast<std::size_t>(corner)]),
                         " of the element in the thermal load set, found none (no NodeTemperatureLoad, no "
                         "DefaultTemperatureLoad); give the grid point a NodeTemperatureLoad or the set a "
                         "DefaultTemperatureLoad"})));
            loadAtCorners[k++] = *value;
            loadMean += *value;
        }
        loadMean /= static_cast<double>(cornerRows.size());
    }

    std::array<double, 8> initialAtCorners{};
    initialAtCorners.fill(section.referenceTemperature);
    bool isInitialAtCorners = false;
    double initialMean = section.referenceTemperature;
    double initialGradient = 0.0;
    if (!temperatures.initial.isEmpty())
    {
        if (const ElementTemperature* element = temperatures.initial.ofElement(elementId))
        {
            initialMean = element->mean;
            initialGradient = element->gradient;
            initialAtCorners.fill(element->mean);
        }
        else
        {
            isInitialAtCorners = true;
            double sum = 0.0;
            std::size_t k = 0;
            for (const std::int32_t corner : cornerRows)
            {
                initialAtCorners[k] = temperatures.initial.atGridRow(corner).value_or(section.referenceTemperature);
                sum += initialAtCorners[k++];
            }
            initialMean = sum / static_cast<double>(cornerRows.size());
        }
    }
    change.deltaT = loadMean - initialMean;
    change.gradient = loadGradient - initialGradient;
    change.hasCornerValues = isLoadAtCorners || isInitialAtCorners;
    if (change.hasCornerValues)
        for (std::size_t k = 0; k < cornerRows.size(); ++k)
            change.atCorners[k] = (isLoadAtCorners ? loadAtCorners[k] : loadMean) - initialAtCorners[k];
    return Result::success(change);
}

} // namespace oos::elements
