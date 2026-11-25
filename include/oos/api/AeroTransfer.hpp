#pragma once

#include "oos/aero/splines/SplineMatrix.hpp"

#include <span>
#include <unordered_map>
#include <vector>

namespace oos::api
{

// Prepared spline transfer on a fixed geometry. Construction reads the snapshot once.
// Node fields use six consecutive basic-frame components per node in nodeIds() order.
// Box fields follow boxIds(). The object owns its geometry and supports concurrent const calls.
class AeroTransfer
{
public:
    explicit AeroTransfer(const input::ModelSnapshot& snapshot);
    [[nodiscard]] std::span<const std::int32_t> nodeIds() const noexcept { return m_nodeIds; }
    [[nodiscard]] std::span<const std::int32_t> boxIds() const noexcept { return m_boxIds; }

    // Translation XYZ and small rotation XYZ (radians) -> box-centre plunge and chordwise slope.
    [[nodiscard]] aero::BoxMotion boxMotion(std::span<const double> nodalMotion) const;
    // Signed normal force at each box force point -> nodal forces XYZ and moments XYZ.
    [[nodiscard]] std::vector<double> structuralLoads(std::span<const double> boxForces) const;

private:
    base::MemoryBudget m_budget{base::MemoryBudget::UNLIMITED};
    aero::AeroReference m_reference;
    aero::AeroBoxes m_boxes;
    aero::SplineMatrix m_splines;
    std::vector<std::int32_t> m_nodeIds, m_boxIds;
    std::unordered_map<std::int32_t, std::size_t> m_nodeRows;
};

} // namespace oos::api
