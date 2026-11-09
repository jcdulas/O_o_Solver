#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ProgressCounter.hpp"

#include <cstdint>

namespace oos::base
{

// The cancellation token and the progress counter of a long operation, handed down together to the steps that may
// last more than a fraction of a second (orderings, symbolic analysis, conversions, graph builds).
// Either may be null: the default control neither stops nor reports. Each operation states the unit it advances the
// counter by (nodes, columns), so that the caller sizes the counter's total.
//
// Thread-safety: as its members (the token and the counter are thread-safe). Ownership: references both; they outlive
// the operation.
struct OperationControl
{
    const CancellationToken* cancellation = nullptr;
    ProgressCounter* progress = nullptr;

    // Throws OperationCancelledError when the token is cancelled.
    void throwIfCancelled() const
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
    }
    void advance(std::uint64_t amount) const noexcept
    {
        if (progress != nullptr)
            progress->advance(amount);
    }
};

} // namespace oos::base
