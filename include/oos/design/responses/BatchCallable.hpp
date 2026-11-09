#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace oos::design
{

// The items of a batch passed to every user callable of the design API, by blocks (the caller's callables are called
// by batch): N items never cost N calls. Large enough that the cost of a call (the arrays of its arguments and the
// call into the caller, a few tens of µs) vanishes beside the work on the items, small enough that the memory of a
// call stays bounded (8 192 items × 16 arguments of 6 components are 6 MB). Measured on the reference PC (a by-label
// equation response of three arguments, a vectorised callable, over 10⁵ items and 100 load cases, the cost of
// the equation beyond the analysis, best of 5): 256 items per block 66 ms, 1 024 59 ms, 4 096 58 ms, 8 192 44 ms,
// 16 384 56 ms, 65 536 56 ms; the call cost is below the noise from about 4 096 items (the rest is the gathering of
// the arguments by label), so the constant takes 8 192, the best measured.
inline constexpr std::size_t EQUATION_BATCH_ITEMS = 8192;

// One argument of a batch call: `components` values per item, item after item (`values` holds items × components).
struct BatchArgument
{
    std::string name;
    std::size_t components = 1;
    std::vector<double> values;
    // Whether the argument's values are items of responses (the loads of a reserve factor), not variables, fields,
    // constants or coordinates: the second-order callable takes the two kinds apart.
    bool isResponse = false;
};

// What a callable returns for a block of `items` items: `outputs` values per item (an (n,) result is one output), item
// after item; with partials, the derivatives of each output with respect to each component of each argument, as
// `partials[a]` holding items × outputs × components(a).
struct BatchResult
{
    std::size_t outputs = 1;
    std::vector<double> values;
    std::vector<std::vector<double>> partials;
};

// A user function called by batch: the caller's callables of a DesignResp3 (by GROUP and TYPE), of
// add_equation_response (function=, gradient=) and of add_relation(function=). The binding implements it with the
// caller's objects; the C++ code owns the convention: blocks of EQUATION_BATCH_ITEMS in a fixed order, the shape and
// finiteness checks (BatchCaller).
//
// Thread-safety: called on the calling thread only; never from a worker of the thread pool.
class BatchCallable
{
public:
    virtual ~BatchCallable() = default;

    // The values of a block: `arguments` hold `items` items each. The result's `outputs` is the m of an (n, m)
    // return, 1 for (n,).
    [[nodiscard]] virtual BatchResult evaluate(std::span<const BatchArgument> arguments, std::size_t items) = 0;
    // The values and the partials of a block (gradient=). A callable without gradient throws.
    [[nodiscard]] virtual BatchResult evaluateWithPartials(std::span<const BatchArgument> arguments,
                                                           std::size_t items) = 0;
    // Whether it has a gradient.
    [[nodiscard]] virtual bool hasGradient() const noexcept = 0;

    // The Hessian-vector products of a block (add_equation_response(hessian_vector=)): with the
    // weights y of the outputs (`weights`: items × `outputs`) and a direction of every argument (`directions`, the
    // shapes of `arguments`), per argument items × components of Σ_o y_o·∇²f_o·v. The default throws (no such
    // callable).
    [[nodiscard]] virtual std::vector<std::vector<double>> hessianVector(std::span<const BatchArgument> arguments,
                                                                         std::size_t items, std::size_t outputs,
                                                                         std::span<const double> weights,
                                                                         std::span<const BatchArgument> directions);
    // Whether it has a Hessian-vector product.
    [[nodiscard]] virtual bool hasHessianVector() const noexcept { return false; }
};

// The convention of every batch call: the items split into blocks of `blockItems` in their order, one
// call per block; each result checked (its value count a multiple of the block's items, the same number of outputs in
// every block, every value finite, the partials of the shape (n, m, c)); a result that breaks the rule is refused by an
// OptimisationError naming `what` (the response or relation), the item (by `itemLabel`) and its argument values.
//
// Thread-safety: run on the calling thread. Ownership: borrows the callable.
class BatchCaller
{
public:
    // `itemLabel(i)`: how item i is named in the messages ("ShellQuad4 12, subcase 3").
    BatchCaller(BatchCallable& callable, std::string what, std::function<std::string(std::size_t)> itemLabel,
                std::size_t blockItems = EQUATION_BATCH_ITEMS);

    // The values of all `items` items (items × outputs, and `outputs` set). `arguments` hold every item. Throws
    // OptimisationError as above; std::invalid_argument for a block size of 0 or arguments of the wrong size.
    [[nodiscard]] BatchResult run(std::span<const BatchArgument> arguments, std::size_t items, bool withPartials);
    // The Hessian-vector products of all `items` items (BatchCallable::hessianVector), by the same blocks: per
    // argument items × components. Throws OptimisationError for a callable without them, a result of the wrong shape
    // or not finite.
    [[nodiscard]] std::vector<std::vector<double>> runHessianVector(std::span<const BatchArgument> arguments,
                                                                    std::size_t items, std::size_t outputs,
                                                                    std::span<const double> weights,
                                                                    std::span<const BatchArgument> directions);
    // The calls made so far.
    [[nodiscard]] std::size_t callCount() const noexcept { return m_callCount; }

private:
    void checkBlock(const BatchResult& result, std::span<const BatchArgument> block, std::size_t firstItem,
                    std::size_t items, std::size_t expectedOutputs, bool withPartials) const;

    BatchCallable* m_callable;
    std::string m_what;
    std::function<std::string(std::size_t)> m_itemLabel;
    std::size_t m_blockItems;
    std::size_t m_callCount = 0;
};

} // namespace oos::design
