#pragma once

#include "oos/api/Optimization.hpp"

#include <memory>
#include <string>

namespace oor::api { class Deck; }
namespace oos::input { class ModelSnapshot; }

namespace oos::api
{
struct DeckOptions
{
    std::string elementFormulation; // Empty uses the deck's selection.
    bool condensation = false;      // Include the component and interface tables.
};

// Build an independent analysis snapshot. Arrow release callbacks retain the exported columns;
// the caller may edit or release its deck once the snapshot has been built.
[[nodiscard]] std::shared_ptr<const input::ModelSnapshot> modelFromDeck(
    const std::shared_ptr<oor::api::Deck>& deck, const DeckOptions& options = {});
// Prepare once and retain each evaluated point for subsequent derivative requests.
[[nodiscard]] std::shared_ptr<OptimizationProblem> prepareOptimization(const std::shared_ptr<oor::api::Deck>& deck,
                                                                       OptimizationSettings settings = {},
                                                                       const DeckOptions& options = {},
                                                                       OptimizationInput declarations = {});
} // namespace oos::api
