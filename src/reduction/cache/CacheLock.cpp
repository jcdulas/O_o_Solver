#include "oos/reduction/cache/CacheLock.hpp"

#include "oos/base/errors/CacheError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace oos::reduction
{

namespace
{

constexpr const char* LOCK_FOLDER = "lock";
constexpr const char* OWNER_FILE = "owner";

// A name no other breaker of a stale lock uses at the same time (a per-process random number and a counter).
std::string uniqueSuffix()
{
    static const std::uint64_t PROCESS_PART = []
    {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32U) ^ device();
    }();
    static std::atomic<std::uint64_t> counter{0};
    return base::concatenate({".stale-", std::to_string(PROCESS_PART), "-", std::to_string(counter.fetch_add(1))});
}

// The time the lock was last known alive: its owner file's, else the folder's; nothing when neither can be read (the
// lock was just released).
std::optional<std::filesystem::file_time_type> lockTime(const std::filesystem::path& lock)
{
    std::error_code error;
    const std::filesystem::file_time_type owner = std::filesystem::last_write_time(lock / OWNER_FILE, error);
    if (!error)
        return owner;
    error.clear();
    const std::filesystem::file_time_type folder = std::filesystem::last_write_time(lock, error);
    if (!error)
        return folder;
    return std::nullopt;
}

// Removes an abandoned lock: renamed first (one waiter wins the rename), then removed.
void breakStaleLock(const std::filesystem::path& lock)
{
    std::filesystem::path aside = lock;
    aside += uniqueSuffix();
    std::error_code error;
    std::filesystem::rename(lock, aside, error);
    if (error)
        return; // another waiter broke it, or its holder released it
    std::filesystem::remove_all(aside, error);
}

} // namespace

CacheLock::CacheLock(const std::filesystem::path& folder, std::chrono::milliseconds wait)
{
    const std::filesystem::path lock = folder / LOCK_FOLDER;
    const auto start = std::chrono::steady_clock::now();
    std::chrono::milliseconds pause = POLL_FIRST;
    for (;;)
    {
        std::error_code error;
        if (std::filesystem::create_directory(lock, error))
        {
            m_path = lock;
            std::ofstream owner(lock / OWNER_FILE, std::ios::trunc);
            owner << "O_o_Solver condensation cache lock\n";
            return;
        }
        // Held (or the folder cannot hold it: then the wait below ends in the error, naming the folder).
        if (const std::optional<std::filesystem::file_time_type> time = lockTime(lock);
            time && std::filesystem::file_time_type::clock::now() - *time > STALE_AGE)
        {
            breakStaleLock(lock);
            continue;
        }
        if (std::chrono::steady_clock::now() - start >= wait)
            throw base::CacheError(base::concatenate(
                {"the condensation cache in '", folder.string(), "' is locked by another process (the lock '",
                 lock.string(), "' is held for longer than ", std::to_string(wait.count()),
                 " ms): wait for the other analysis to end, or use another cache_dir; a lock left by a process that "
                 "ended is broken after ",
                 std::to_string(STALE_AGE.count()), " s"}));
        std::this_thread::sleep_for(pause);
        pause = std::min(POLL_LONGEST, pause * 2);
    }
}

CacheLock::CacheLock(CacheLock&& other) noexcept : m_path(std::exchange(other.m_path, {})) {}

CacheLock& CacheLock::operator=(CacheLock&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_path = std::exchange(other.m_path, {});
    }
    return *this;
}

CacheLock::~CacheLock()
{
    release();
}

void CacheLock::refresh() noexcept
{
    if (m_path.empty())
        return;
    std::error_code error;
    std::filesystem::last_write_time(m_path / OWNER_FILE, std::filesystem::file_time_type::clock::now(), error);
}

void CacheLock::release() noexcept
{
    if (m_path.empty())
        return;
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
    m_path.clear();
}

} // namespace oos::reduction
