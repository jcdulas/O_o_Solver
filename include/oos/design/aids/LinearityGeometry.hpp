#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace oos::design
{

// The geometry of the shells that the linearity indicator reads (LinearityIndicator), the part that does not depend on
// the designed fields: per shell element of positive area its corners, unit normal and area, per grid point that is a
// corner of one the elements around it and its displacement frame. A design point that moves no grid point (no shape
// variable) leaves it as it is, so a DesignProblem keeps it between its points; the thicknesses, which
// the designed fields move, are the indicator's.
//
// Shells: ShellQuad4, ShellQuad4Drilling, ShellQuad8 (4 corners, the first grid points of the card), ShellTria3,
// ShellTria3Drilling, ShellTria6 (3 corners). The normal of an element is (p3 − p1) × (p4 − p2) for 4 corners, (p2 − p1) ×
// (p3 − p1) for 3, its area half the length of that vector (exact for a flat element, the mean plane's for a warped
// one).
//
// Thread-safety: immutable once built. Ownership: owns its geometry (its memory taken from the budget given to the
// constructor, which must outlive it); keeps no reference to the snapshot.
class LinearityGeometry
{
public:
    // One shell element of positive area.
    struct Element
    {
        std::string card;
        std::int32_t elementId = 0;
        std::size_t batch = 0; // its batch in ModelSnapshot::elementBatches()
        std::int64_t row = 0;  // its row in that batch
        std::vector<std::int32_t> cornerRows;
        base::Vec3 normal; // unit, in the basic system
        double area = 0.0;
    };

    // One grid point that is a corner of an element of positive area.
    struct Point
    {
        std::int32_t grid = 0;
        std::int32_t row = 0;              // its node table row
        base::Mat3 displacementFrame;      // CD to basic: u_basic = R·u_CD
        std::vector<std::size_t> elements; // indices into elements(), in the order of the batches and rows
    };

    // The geometry of `snapshot`. Throws MemoryBudgetError when its estimate does not fit `budget`.
    LinearityGeometry(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget);

    // The shell cards and the number of their corners; 0 for another card.
    [[nodiscard]] static std::size_t cornerCountOf(std::string_view card) noexcept;
    // The estimate of the memory of the geometry of `snapshot` (and of an indicator's lists built on it), made before
    // either is built: per shell element its geometry and corner list, per grid point (at most one per corner) its
    // geometry and the list of the elements around it.
    [[nodiscard]] static std::uint64_t estimateBytes(const input::ModelSnapshot& snapshot);
    // The node rows of the corners of a shell row, or an empty list when its grid points are not all known.
    [[nodiscard]] static std::vector<std::int32_t> cornerRowsOf(const input::ElementBatch& batch, std::int64_t row,
                                                                std::size_t corners, std::int64_t nodeCount);

    [[nodiscard]] const std::vector<Element>& elements() const noexcept { return m_elements; }
    // The grid points, by increasing ID.
    [[nodiscard]] const std::vector<Point>& points() const noexcept { return m_points; }

private:
    base::MemoryReservation m_reservation; // the estimate of the lists below, reserved before they are built
    std::vector<Element> m_elements;
    std::vector<Point> m_points;
};

} // namespace oos::design
