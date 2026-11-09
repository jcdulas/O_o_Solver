#include "oos/design/responses/BatchCallable.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace oos::design
{

namespace
{

// The arguments of item `item`, as "name=(v1, v2)" pieces, for the messages.
std::string describeArguments(std::span<const BatchArgument> arguments, std::size_t item)
{
    std::ostringstream text;
    text.precision(17);
    for (std::size_t a = 0; a < arguments.size(); ++a)
    {
        const BatchArgument& argument = arguments[a];
        text << (a == 0 ? "" : ", ") << argument.name << "=";
        if (argument.components != 1)
            text << "(";
        for (std::size_t c = 0; c < argument.components; ++c)
            text << (c == 0 ? "" : ", ") << argument.values[item * argument.components + c];
        if (argument.components != 1)
            text << ")";
    }
    return text.str();
}

// The block [first, first + items) of every argument.
std::vector<BatchArgument> blockOf(std::span<const BatchArgument> arguments, std::size_t first, std::size_t items)
{
    std::vector<BatchArgument> block; // bounded by the arguments of one callable
    block.reserve(arguments.size());
    for (const BatchArgument& argument : arguments)
    {
        BatchArgument part{argument.name, argument.components, {}, argument.isResponse};
        const auto begin = argument.values.begin() + static_cast<std::ptrdiff_t>(first * argument.components);
        part.values.assign(begin, begin + static_cast<std::ptrdiff_t>(items * argument.components));
        block.push_back(std::move(part));
    }
    return block;
}

} // namespace

std::vector<std::vector<double>> BatchCallable::hessianVector(std::span<const BatchArgument>, std::size_t, std::size_t,
                                                              std::span<const double>, std::span<const BatchArgument>)
{
    throw base::OptimisationError(
        "BatchCallable: expected a hessian_vector= callable for the second derivatives, found "
        "none; give hessian_vector= to add_equation_response");
}

BatchCaller::BatchCaller(BatchCallable& callable, std::string what, std::function<std::string(std::size_t)> itemLabel,
                         std::size_t blockItems)
    : m_callable(&callable), m_what(std::move(what)), m_itemLabel(std::move(itemLabel)), m_blockItems(blockItems)
{
    if (m_blockItems == 0)
        throw std::invalid_argument("BatchCaller: expected at least one item per block, found 0");
}

BatchResult BatchCaller::run(std::span<const BatchArgument> arguments, std::size_t items, bool withPartials)
{
    for (const BatchArgument& argument : arguments)
        if (argument.components == 0 || argument.values.size() != items * argument.components)
            throw std::invalid_argument(
                base::concatenate({"BatchCaller: expected ", std::to_string(items), " items of ",
                                   std::to_string(argument.components), " components for argument '", argument.name,
                                   "' of ", m_what, ", found ", std::to_string(argument.values.size()), " values"}));
    if (withPartials && !m_callable->hasGradient())
        throw base::OptimisationError(base::concatenate(
            {m_what, ": expected a gradient= callable for the derivatives, found none; give gradient="}));
    BatchResult all;
    all.outputs = 0;
    if (withPartials)
        all.partials.resize(arguments.size());
    // The blocks run in item order, one call each; the results are appended in the same order.
    for (std::size_t first = 0; first < items; first += m_blockItems)
    {
        const std::size_t count = std::min(m_blockItems, items - first);
        const std::vector<BatchArgument> block = blockOf(arguments, first, count);
        BatchResult result =
            withPartials ? m_callable->evaluateWithPartials(block, count) : m_callable->evaluate(block, count);
        ++m_callCount;
        checkBlock(result, block, first, count, all.outputs, withPartials);
        if (all.outputs == 0)
            all.outputs = result.outputs;
        all.values.insert(all.values.end(), result.values.begin(), result.values.end());
        for (std::size_t a = 0; withPartials && a < arguments.size(); ++a)
            all.partials[a].insert(all.partials[a].end(), result.partials[a].begin(), result.partials[a].end());
    }
    if (all.outputs == 0)
        all.outputs = 1; // no item: the shape of an (n,) result
    return all;
}

std::vector<std::vector<double>> BatchCaller::runHessianVector(std::span<const BatchArgument> arguments,
                                                               std::size_t items, std::size_t outputs,
                                                               std::span<const double> weights,
                                                               std::span<const BatchArgument> directions)
{
    if (!m_callable->hasHessianVector())
        throw base::OptimisationError(base::concatenate(
            {m_what,
             ": expected a hessian_vector= callable for its second derivatives, found none; give "
             "hessian_vector= to add_equation_response"}));
    if (directions.size() != arguments.size() || weights.size() != items * outputs)
        throw std::invalid_argument(
            base::concatenate({"BatchCaller::runHessianVector: expected a direction per argument and ",
                               std::to_string(items * outputs), " weights for ", m_what}));
    std::vector<std::vector<double>> all(arguments.size());
    for (std::size_t first = 0; first < items; first += m_blockItems)
    {
        const std::size_t count = std::min(m_blockItems, items - first);
        const std::vector<BatchArgument> block = blockOf(arguments, first, count);
        const std::vector<BatchArgument> moves = blockOf(directions, first, count);
        const std::vector<std::vector<double>> result =
            m_callable->hessianVector(block, count, outputs, weights.subspan(first * outputs, count * outputs), moves);
        ++m_callCount;
        if (result.size() != block.size())
            throw base::OptimisationError(
                base::concatenate({m_what, ": expected hessian_vector= to give a product for each of the ",
                                   std::to_string(block.size()), " arguments, found ", std::to_string(result.size()),
                                   "; make it return (w_x, w_L) with an entry per variable and response argument"}));
        for (std::size_t a = 0; a < block.size(); ++a)
        {
            if (result[a].size() != count * block[a].components)
                throw base::OptimisationError(base::concatenate(
                    {m_what, ": expected hessian_vector= to give ", std::to_string(count * block[a].components),
                     " values for argument '", block[a].name, "' (its shape), found ", std::to_string(result[a].size()),
                     "; make it return arrays of the arguments' shapes"}));
            for (std::size_t k = 0; k < result[a].size(); ++k)
                if (!std::isfinite(result[a][k]))
                {
                    const std::size_t item = k / block[a].components;
                    throw base::OptimisationError(base::concatenate(
                        {m_what, ", item ", std::to_string(first + item), " (", m_itemLabel(first + item),
                         "): expected finite Hessian-vector products from hessian_vector=, found a value that is not "
                         "for argument '",
                         block[a].name, "'; the arguments of the item were ", describeArguments(block, item),
                         "; make it return finite values"}));
                }
            all[a].insert(all[a].end(), result[a].begin(), result[a].end());
        }
    }
    return all;
}

void BatchCaller::checkBlock(const BatchResult& result, std::span<const BatchArgument> block, std::size_t firstItem,
                             std::size_t items, std::size_t expectedOutputs, bool withPartials) const
{
    const auto refuse = [&](std::size_t item, const std::string& what)
    {
        throw base::OptimisationError(
            base::concatenate({m_what, ", item ", std::to_string(firstItem + item), " (", m_itemLabel(firstItem + item),
                               "): ", what, "; the arguments of the item were ", describeArguments(block, item),
                               ". A callable returns (n,) or (n, m) for n items (it is called by batch): make "
                               "it return finite values of that shape"}));
    };
    if (result.outputs == 0 || result.values.size() != items * result.outputs)
        refuse(0, base::concatenate({"expected (", std::to_string(items), ",) or (", std::to_string(items),
                                     ", m) values for the block of ", std::to_string(items), " items, found ",
                                     std::to_string(result.values.size()), " values"}));
    if (expectedOutputs != 0 && result.outputs != expectedOutputs)
        refuse(0,
               base::concatenate({"expected ", std::to_string(expectedOutputs),
                                  " outputs per item as in the first block, found ", std::to_string(result.outputs)}));
    for (std::size_t i = 0; i < items; ++i)
        for (std::size_t m = 0; m < result.outputs; ++m)
        {
            const double value = result.values[i * result.outputs + m];
            if (!std::isfinite(value))
            {
                std::ostringstream text;
                text << "expected finite values, found " << value << " for output " << m;
                refuse(i, text.str());
            }
        }
    if (!withPartials)
        return;
    if (result.partials.size() != block.size())
        refuse(0,
               base::concatenate({"expected the derivatives with respect to each of the ", std::to_string(block.size()),
                                  " arguments, found ", std::to_string(result.partials.size())}));
    for (std::size_t a = 0; a < block.size(); ++a)
    {
        const std::size_t expected = items * result.outputs * block[a].components;
        if (result.partials[a].size() != expected)
            refuse(0, base::concatenate({"expected derivatives of shape (", std::to_string(items), ", ",
                                         std::to_string(result.outputs), ", ", std::to_string(block[a].components),
                                         ") for argument '", block[a].name, "', found ",
                                         std::to_string(result.partials[a].size()), " values"}));
        for (std::size_t k = 0; k < expected; ++k)
            if (!std::isfinite(result.partials[a][k]))
                refuse(k / (result.outputs * block[a].components),
                       base::concatenate({"expected finite derivatives with respect to '", block[a].name,
                                          "', found a value that is not"}));
    }
}

} // namespace oos::design
