#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oos::base
{

// The requested outputs would not fit their destination (memory or disk); carries the estimate and the
// remedies ("request envelopes instead of every load case", "restrict the SET of STRESS"), which the message also
// lists.
class OutputVolumeError : public SolverError
{
public:
    OutputVolumeError(const std::string& message, std::uint64_t estimatedBytes, std::vector<std::string> remedies)
        : SolverError(message), m_estimatedBytes(estimatedBytes), m_remedies(std::move(remedies))
    {
    }

    [[nodiscard]] std::uint64_t estimatedBytes() const noexcept { return m_estimatedBytes; }
    [[nodiscard]] const std::vector<std::string>& remedies() const noexcept { return m_remedies; }

private:
    std::uint64_t m_estimatedBytes;
    std::vector<std::string> m_remedies;
};

} // namespace oos::base
