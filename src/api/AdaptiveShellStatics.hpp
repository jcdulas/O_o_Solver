#pragma once

#include "oos/solutions/sol101/StaticSolution.hpp"

namespace oos::api
{

[[nodiscard]] solutions::StaticResults runStaticSolution(const input::ModelSnapshot& snapshot,
                                                         solutions::StaticOptions options, base::MemoryBudget& budget,
                                                         std::optional<std::vector<std::int32_t>>& selected,
                                                         std::shared_ptr<const elements::SnapshotIndex>* finalIndex = nullptr);

} // namespace oos::api
