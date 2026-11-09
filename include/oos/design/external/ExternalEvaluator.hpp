#pragma once

#include "oos/design/external/ItemStatus.hpp"
#include "oos/design/responses/BatchCallable.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The subcases to evaluate per zone (select=): zone ID → its selected subcases. A zone that is not in it is evaluated
// in every subcase.
using ItemSelection = std::map<std::int32_t, std::vector<std::int32_t>>;

// What the rows of the arguments of an external call are: the load cases of a label (a plugin that is
// not linear gets the combined data of each case), the vectors of the load basis of one constraint set (a linear
// plugin, "called with the data of the k basis vectors and the coefficients α"), or none (supplied values: the program
// computed them from problem.arguments).
enum class ExternalRows : std::uint8_t
{
    Cases,
    Basis,
    Supplied
};

// One call of an external response: one label (a zone) and one batch of its rows (plugins are called per label: the
// arrays' first axis runs over the items of that label, their second has the size of that label, never padded).
// `cases` are the subcases of the batch (Cases, Supplied) or those the basis serves (Basis: the cases of the
// constraint set to combine for); `arguments` hold `rows` rows each (none for Supplied).
struct ExternalCall
{
    const ExternalResponseDefinition* response = nullptr;
    std::int32_t zone = 0;
    ExternalRows kind = ExternalRows::Cases;
    std::vector<std::int32_t> cases;
    std::size_t rows = 0;
    std::int32_t constraintSet = 0; // Basis
    std::vector<BatchArgument> arguments;
};

// What one call gives (`outputs` per row; `values` rows × outputs). `statuses` (rows × outputs, or empty: every value
// ok): a non-finite value is a failed item. With partials:
//  - for rows of cases or supplied values, `partials[a]` holds rows × outputs × components(a), the derivatives of each
//    output with respect to each component of argument a of the same row;
//  - for a linear call on the basis, `partials[a]` holds, for an argument of zone data, the linear operator of the
//    plugin, outputs × components(a) (the same for every case: its gradient gives its linear operator once per
//    zone), and for a design variable rows × outputs × 1 (its derivatives per basis vector).
struct ExternalResult
{
    std::size_t outputs = 0;
    std::vector<double> values;
    std::vector<ItemStatus> statuses;
    std::vector<std::vector<double>> partials;
};

// One call of post= of a linear response: the combined linear outputs of `cases` of one zone (cases × outputs); its
// result has the final outputs per case and, with partials, partials[0] cases × final outputs × linear outputs.
struct ExternalPostCall
{
    const ExternalResponseDefinition* response = nullptr;
    std::int32_t zone = 0;
    std::vector<std::int32_t> cases;
    std::size_t outputs = 0;
    std::vector<double> values;
};

// The plug-in point of the external responses: the caller's runtime of the external responses implements it
// (the plugin's evaluate, gradient and post, the select= table or callable, the supplied values of problem.supply),
// with the checks of the shapes and the messages; the C++ side gathers the arguments, combines the linear outputs on
// the basis and chains the partials into the sensitivities.
//
// Thread-safety: called on the calling thread only (it calls into the caller). Ownership: none.
class ExternalEvaluator
{
public:
    ExternalEvaluator() = default;
    ExternalEvaluator(const ExternalEvaluator&) = delete;
    ExternalEvaluator& operator=(const ExternalEvaluator&) = delete;
    virtual ~ExternalEvaluator() = default;

    // select=: per zone of `response`, the subcases to evaluate among `cases`, given the responses evaluated so far.
    [[nodiscard]] virtual ItemSelection select(const ExternalResponseDefinition& response,
                                               std::span<const std::int32_t> cases,
                                               const std::map<std::int32_t, const ResponseValues*>& known) = 0;
    [[nodiscard]] virtual ExternalResult evaluate(const ExternalCall& call, bool withPartials) = 0;
    [[nodiscard]] virtual ExternalResult post(const ExternalPostCall& call, bool withPartials) = 0;
};

// The evaluators of an evaluation by name (ExternalResponseDefinition::evaluator). Borrowed.
using ExternalRegistry = std::map<std::string, ExternalEvaluator*>;

} // namespace oos::design
