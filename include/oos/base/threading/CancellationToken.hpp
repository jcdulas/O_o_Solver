#pragma once

#include <atomic>
#include <memory>

namespace oos::base
{

// Lets a long operation be stopped from outside (a user interruption, a progress callback that asks to stop). Copies
// share the same state. The operation calls throwIfCancelled() at points where stopping leaves nothing half done; it
// checks often enough that an interruption takes effect within a fraction of a second (0.1 s of work).
// Thread-safe.
class CancellationToken
{
public:
    CancellationToken() : m_cancelled(std::make_shared<std::atomic<bool>>(false)) {}

    void cancel() noexcept { m_cancelled->store(true, std::memory_order_relaxed); }
    [[nodiscard]] bool isCancelled() const noexcept { return m_cancelled->load(std::memory_order_relaxed); }

    // Throws OperationCancelledError when cancel() has been called.
    void throwIfCancelled() const;

private:
    std::shared_ptr<std::atomic<bool>> m_cancelled;
};

} // namespace oos::base
