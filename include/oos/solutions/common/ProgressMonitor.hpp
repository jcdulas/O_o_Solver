#pragma once

#include "oos/base/threading/ProgressCounter.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace oos::solutions
{

// What a progress poll reads: the stage under way and its counts.
struct ProgressSnapshot
{
    std::string stage; // empty before the first stage
    std::uint64_t done = 0;
    std::uint64_t total = 0;
};

// The progress of an analysis as a sequence of stages ("assembly", "factorisation", "solve", "recovery"), each a
// base::ProgressCounter advanced by the worker threads. The calling thread polls it (the caller's callbacks are only
// called from the calling thread) while the analysis runs on another thread.
//
// Thread-safety: beginStage() and the counters from the analysis threads, poll() from the calling thread; all methods
// are safe concurrently, setStageListener() excepted (before the analysis starts). Ownership: owns the counter of the
// current stage (shared with poll() while it reads).
class ProgressMonitor
{
public:
    // Called on the analysis thread at the start of each stage, with its name, before any work of the stage; it must
    // not throw. The robustness tests cancel an analysis there, to stop it at a stage of their choice without depending
    // on the timing of a poll.
    using StageListener = std::function<void(const std::string& stage)>;

    // Starts a stage of `total` units and returns its counter, valid until the next beginStage().
    base::ProgressCounter& beginStage(std::string stage, std::uint64_t total);

    [[nodiscard]] ProgressSnapshot poll() const;

    // Set before the analysis starts; an empty listener removes it.
    void setStageListener(StageListener listener) { m_stageListener = std::move(listener); }

private:
    StageListener m_stageListener;
    mutable std::mutex m_mutex;
    std::shared_ptr<base::ProgressCounter> m_current;
    std::shared_ptr<base::ProgressCounter> m_previous; // kept one stage longer: a worker may still advance it
};

} // namespace oos::solutions
