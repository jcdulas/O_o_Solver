#include "oos/design/sensitivity/MethodChoice.hpp"

namespace oos::design
{

std::string_view sensitivityMethodName(SensitivityMethod method) noexcept
{
    switch (method)
    {
    case SensitivityMethod::Automatic:
        return "automatic";
    case SensitivityMethod::Direct:
        return "direct";
    case SensitivityMethod::Adjoint:
        return "adjoint";
    case SensitivityMethod::Chained:
        return "chained";
    }
    return "automatic";
}

SubcaseSensitivity MethodChoice::choose(std::int32_t subcase, std::int64_t variables, std::int64_t argumentItems,
                                        std::int64_t requestedItems, SensitivityMethod forced) noexcept
{
    SubcaseSensitivity choice;
    choice.subcase = subcase;
    choice.variables = variables;
    choice.argumentItems = argumentItems;
    choice.requestedItems = requestedItems;
    choice.directSolves = variables;
    choice.adjointSolves = argumentItems;
    choice.chainedSolves = requestedItems;
    if (forced != SensitivityMethod::Automatic)
    {
        choice.method = forced;
        return choice;
    }
    choice.method = SensitivityMethod::Direct;
    std::int64_t fewest = choice.directSolves;
    if (choice.adjointSolves < fewest)
    {
        choice.method = SensitivityMethod::Adjoint;
        fewest = choice.adjointSolves;
    }
    if (choice.chainedSolves < fewest)
        choice.method = SensitivityMethod::Chained;
    return choice;
}

} // namespace oos::design
