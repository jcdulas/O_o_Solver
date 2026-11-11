#pragma once

#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oos::api
{
struct AnalysisCapability
{
    std::string analysis;
    std::int32_t solution = 0;
    bool sizingFirstOrder = true;
    bool shapeFirstOrder = true;
    bool secondOrder = false;
    std::vector<std::string> restrictions;
};

// Real columns bindable by a native design relation. List fields require a zero-based fieldItem, checked against
// the selected target row when the problem is prepared. This is a candidate catalogue, not a promise that changing
// every field has a physical effect in every element formulation.
struct DesignFieldCapability
{
    std::string targetKind;
    std::string targetCard;
    std::string field;
    std::string type;
    bool requiresItem = false;
};

struct OptimizationCapabilities
{
    std::int32_t version = 1;
    std::vector<AnalysisCapability> analyses;
    std::vector<std::string> responses;
    std::vector<DesignFieldCapability> fields;
};

[[nodiscard]] OptimizationCapabilities optimizationCapabilities(const input::ModelSnapshot* snapshot = nullptr);
} // namespace oos::api
