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
// The first line of the owner file (for a person who opens it); its second line is the holder's token.
constexpr const char* OWNER_TITLE = "O_o_Solver condensation cache lock";

// A text no other lock object and no other breaker of a stale lock uses (a per-process random number and a counter).
std::string uniqueToken()
{
    static const std::uint64_t PROCESS_PART = []
    {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32U) ^ device();
    }();
    static std::atomic<std::uint64_t> counter{0};
    return base::concatenate({std::to_string(PROCESS_PART), "-", std::to_string(counter.fetch_add(1))});
}

// A name no other breaker of a stale lock uses at the same time.
std::string uniqueSuffix()
{
    return base::concatenate({".stale-", uniqueToken()});
}

// The token of the holder of the lock folder `lock` (the second line of its owner file); empty when the file does not
// read (no lock there, or one whose owner file is not written yet).
std::string ownerTokenOf(const std::filesystem::path& lock)
{
    std::ifstream owner(lock / OWNER_FILE);
    std::string title;
    std::string token;
    if (!std::getline(owner, title) || !std::getline(owner, token))
        return {};
    return token;
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
            m_token = uniqueToken();
            std::ofstream owner(lock / OWNER_FILE, std::ios::trunc);
            owner << OWNER_TITLE << '\n' << m_token << '\n';
            owner.close();
            // Without its owner file the lock has no token to be told by: it is released as a lock without one.
            if (!owner)
                m_token.clear();
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

CacheLock::CacheLock(CacheLock&& other) noexcept
    : m_path(std::exchange(other.m_path, {})), m_token(std::exchange(other.m_token, {}))
{
}

CacheLock& CacheLock::operator=(CacheLock&& other) noexcept
{
    if (this != &other)
    {
        release();
        m_path = std::exchange(other.m_path, {});
        m_token = std::exchange(other.m_token, {});
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

bool CacheLock::isStillOwned() const noexcept
{
    // A lock whose owner file could not be written has no token: nothing tells it from another, it is taken for ours.
    if (m_token.empty())
        return true;
    try
    {
        return ownerTokenOf(m_path) == m_token;
    }
    catch (...) // no memory for the texts: the folder is left, and broken as an abandoned lock after STALE_AGE
    {
        return false;
    }
}

void CacheLock::release() noexcept
{
    if (m_path.empty())
        return;
    // A lock this holder kept for longer than STALE_AGE without a refresh may have been broken and taken by a waiter:
    // the folder at this path is then the waiter's lock, which must stay (removing it would let a third process in
    // beside the waiter).
    if (isStillOwned())
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }
    m_path.clear();
    m_token.clear();
}

} // namespace oos::reduction
