#pragma once

#include "oos/reduction/plan/CondensationPlan.hpp"

#include <string>
#include <vector>

namespace oos::reduction
{

// The plan report (per unit, the boundary size before and after absorption, the size of K*, the estimated
// condensation time and memory, and a recommendation to keep full a unit whose K* would cost more than its interior):
// a text table, one row per unit, then what each unit
// absorbed, the spiders (AbsorptionReport: absorbed per unit, every spider kept on its legs with the reason), the
// connection system, the interface problem and the full path. Sizes in MB of 2^20 bytes (8-byte
// values), work in GFLOP, times on one thread at ESTIMATE_FLOP_RATE. Cards and fields use the model's neutral names.
//
// Thread-safety: immutable. Ownership: owns its text.
class PlanReport
{
public:
    explicit PlanReport(const CondensationPlan& plan);

    [[nodiscard]] const std::string& text() const noexcept { return m_text; }
    // The units the report recommends keeping full although the plan condenses them.
    [[nodiscard]] const std::vector<std::int64_t>& keptFullRecommendations() const noexcept
    {
        return m_keptFullRecommendations;
    }

private:
    std::string m_text;
    std::vector<std::int64_t> m_keptFullRecommendations;
};

} // namespace oos::reduction
