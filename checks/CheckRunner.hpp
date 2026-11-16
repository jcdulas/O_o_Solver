#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace oos::checks
{

// A minimal runner for the native checks of the element kernels (no third-party test library: LEGAL.md). A check is a
// named function; it reports a failure through expect(), and the runner prints one line per failure and a summary.
//
// Thread-safety: none; the checks run one after the other on the calling thread.
class CheckRunner
{
public:
    using CheckFunction = std::function<void(CheckRunner&)>;

    // The checks registered by the CHECK_CASE macro of every translation unit.
    static std::vector<std::pair<std::string, CheckFunction>>& registry()
    {
        static std::vector<std::pair<std::string, CheckFunction>> checks;
        return checks;
    }

    // Records a failure when `condition` is false.
    void expect(bool condition, const std::string& message)
    {
        if (condition)
            return;
        ++m_failures;
        std::printf("  FAILED %s: %s\n", m_current.c_str(), message.c_str());
    }

    // Records a failure when |value - expected| > tolerance · max(1, |expected|).
    void expectNear(double value, double expected, double tolerance, const std::string& what)
    {
        const double scale = std::fmax(1.0, std::fabs(expected));
        const bool close = std::fabs(value - expected) <= tolerance * scale;
        expect(close, what + ": " + std::to_string(value) + " expected " + std::to_string(expected) + " (tolerance " +
                          std::to_string(tolerance) + ")");
    }

    // Runs the checks whose name contains `filter` (all of them when it is empty); returns the number of failures.
    int run(const std::string& filter)
    {
        int ran = 0;
        for (const auto& [name, check] : registry())
        {
            if (!filter.empty() && name.find(filter) == std::string::npos)
                continue;
            m_current = name;
            const int before = m_failures;
            check(*this);
            ++ran;
            std::printf("%s %s\n", m_failures == before ? "ok    " : "FAILED", name.c_str());
        }
        std::printf("%d checks, %d failures\n", ran, m_failures);
        return m_failures;
    }

private:
    std::string m_current;
    int m_failures = 0;
};

// Registers a check at static initialisation.
struct CheckRegistration
{
    CheckRegistration(const char* name, CheckRunner::CheckFunction function)
    {
        CheckRunner::registry().emplace_back(name, std::move(function));
    }
};

} // namespace oos::checks

#define OOS_CHECK_CONCAT_INNER(a, b) a##b
#define OOS_CHECK_CONCAT(a, b) OOS_CHECK_CONCAT_INNER(a, b)

// CHECK_CASE("name") { ... runner.expect(...); }
#define CHECK_CASE(name)                                                                       \
    static void OOS_CHECK_CONCAT(checkFunction, __LINE__)(oos::checks::CheckRunner & runner);  \
    static const oos::checks::CheckRegistration OOS_CHECK_CONCAT(checkRegistration, __LINE__){ \
        name, OOS_CHECK_CONCAT(checkFunction, __LINE__)};                                      \
    static void OOS_CHECK_CONCAT(checkFunction, __LINE__)([[maybe_unused]] oos::checks::CheckRunner & runner)
