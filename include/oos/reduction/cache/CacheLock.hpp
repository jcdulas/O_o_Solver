#pragma once

#include <chrono>
#include <filesystem>

namespace oos::reduction
{

// The lock several processes sharing one cache folder take around its cleanup and the writing of an entry (several
// processes may use one cache: a lock around the index and the cleanup; every entry written through a temporary file
// and renamed). Reading an entry needs no lock: an entry file is replaced in one
// step (CacheFile), and a file another process removes while it is read is either kept (Windows refuses to remove an
// open file: the remover skips it) or read to its end (POSIX keeps its content for the reader).
//
// Method (standard library only; the operating-system locks belong to base/platform): the lock is the
// folder "<cache>/lock", whose creation is atomic on Windows and POSIX (std::filesystem::create_directory says whether
// this call made it), with a file "owner" inside whose time the holder refreshes while it works (refresh()). A lock
// whose time is older than STALE_AGE is the leftover of a process that ended without releasing it (a crash, a kill):
// a waiter renames it away (a rename of the same folder succeeds for one waiter only), removes it and tries again.
// Waiters poll with a growing pause (POLL_FIRST to POLL_LONGEST). The time the folder exists without its owner file
// (between its creation and the owner's) is covered by the folder's own time.
//
// Thread-safety: one object per thread; two threads of one process wait for each other like two processes.
// Ownership: owns the lock folder while it lives; movable, not copyable.
class CacheLock
{
public:
    // A lock untouched for this long is abandoned: the longest a holder works between two refreshes (writing one
    // column of an entry, or one cleanup), with a wide margin for a slow disk.
    static constexpr std::chrono::seconds STALE_AGE{300};
    static constexpr std::chrono::milliseconds POLL_FIRST{1};
    static constexpr std::chrono::milliseconds POLL_LONGEST{50};

    // Takes the lock of `folder` (which must exist), waiting at most `wait`. Throws base::CacheError when the lock is
    // still held after `wait` (naming the lock and what to do).
    CacheLock(const std::filesystem::path& folder, std::chrono::milliseconds wait);
    CacheLock(const CacheLock&) = delete;
    CacheLock& operator=(const CacheLock&) = delete;
    CacheLock(CacheLock&& other) noexcept;
    CacheLock& operator=(CacheLock&& other) noexcept;
    ~CacheLock();

    // Marks the lock as alive (its time now), so that waiters do not take it for abandoned during a long write.
    void refresh() noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

private:
    void release() noexcept;

    std::filesystem::path m_path; // empty once released or moved from
};

} // namespace oos::reduction
