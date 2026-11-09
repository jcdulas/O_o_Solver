#include "oos/design/sensitivity/ChainedAdjoint.hpp"

#include "oos/design/sensitivity/AdjointSensitivity.hpp"

namespace oos::design
{

void ChainedAdjoint::chain(const ItemKey& key, LinearisedItems& linearised)
{
    ItemLinearisation& item = linearised.at(key);
    item.explicitPart = item.ownExplicit;
    item.weights.clear();
    for (const auto& [argument, weight] : item.arguments)
    {
        const ItemLinearisation& source = linearised.at(argument);
        // The sparse rows add as the dense ones did, column by column in the order of the arguments.
        item.explicitPart = addScaledRow(item.explicitPart, weight, source.explicitPart);
        if (source.readsDisplacements())
            item.weights = addScaledSparse(item.weights, weight, source.weights);
    }
}

std::int64_t ChainedAdjoint::solve(const SubcaseSolve& context, std::span<const ItemKey> equationItems,
                                   LinearisedItems& linearised)
{
    return AdjointSensitivity::solve(context, equationItems, linearised);
}

} // namespace oos::design
