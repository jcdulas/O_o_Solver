#pragma once

#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/OperationControl.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/ApproximateMinimumDegree.hpp"
#include "oos/linalg/ordering/GeometricDissection.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

enum class OrderingMethod : std::uint8_t
{
    Automatic,        // the rule of ConstrainedOrdering::chooseMethod
    MinimumDegree,    // ApproximateMinimumDegree
    NestedDissection, // GeometricDissection (needs coordinates)
    Natural           // the node index order (tests and comparisons)
};

// The ordering of a unit: the interior nodes by a fill-reducing method, then a given trailing set (the
// boundary of the unit, kept last for a partial factorisation, in the order given). The method is chosen by a fixed
// rule on the size and the dimension of the unit, set by measurement:
//   - no coordinates, or fewer than DISSECTION_MIN_WEIGHT_SURFACE DOFs (surface meshes) or
//     DISSECTION_MIN_WEIGHT_VOLUME DOFs (volume meshes): approximate minimum degree;
//   - otherwise: geometric nested dissection.
// The dimension is read from the graph: a mean node degree above VOLUME_MEAN_DEGREE is a volume mesh (a hexahedral
// mesh has 26 neighbours per node, a quadrilateral shell mesh 8, a tetrahedral mesh about 14).
//
// Thread-safety: order() is const; it uses the pool given (the dissection). The result depends on the graph, the
// weights, the coordinates and the trailing set only.
class ConstrainedOrdering
{
public:
    static constexpr std::int64_t DISSECTION_MIN_WEIGHT_SURFACE = 10'000;
    static constexpr std::int64_t DISSECTION_MIN_WEIGHT_VOLUME = 2'000;
    static constexpr double VOLUME_MEAN_DEGREE = 12.0;

    explicit ConstrainedOrdering(DissectionOptions dissection = {}, MinimumDegreeOptions minimumDegree = {})
        : m_dissection(dissection), m_minimumDegree(minimumDegree)
    {
    }

    // The method the rule chooses for a unit.
    [[nodiscard]] static OrderingMethod chooseMethod(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                                     std::span<const base::Vec3> coordinates);

    // The order of the nodes of `graph`, the trailing nodes last in their order. nodeWeights: empty for 1 each;
    // coordinates: empty when unknown. Throws std::invalid_argument as the methods do, and when NestedDissection is
    // asked without coordinates. control: passed to the method (cancellation; progress by the nodes ordered, weighted
    // for the minimum degree).
    [[nodiscard]] NodePermutation order(const NodeGraph& graph, std::span<const std::int32_t> nodeWeights,
                                        std::span<const base::Vec3> coordinates,
                                        std::span<const std::int32_t> trailingNodes, OrderingMethod method,
                                        base::MemoryBudget& budget, base::ThreadPool* pool = nullptr,
                                        const base::OperationControl& control = {}) const;

private:
    DissectionOptions m_dissection;
    MinimumDegreeOptions m_minimumDegree;
};

} // namespace oos::linalg
