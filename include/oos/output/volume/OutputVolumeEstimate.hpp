#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/output/arrays/OutputPlan.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace oos::output
{

// The volume of the requested outputs, estimated before the analysis (rows × components × load cases × 8 bytes,
// with the labels of the rows), so that a run whose results would not fit is refused before it starts, with
// the estimate and the remedies (OutputVolumeError).
//
// The rows of each element type are counted by recovering its first element once with zero displacements (the rows
// and columns of each kind: a CQUAD4 has its centre and four corners at two fibres, a laminate its plies), times the
// elements of the request; the nodal tables have one row per point of the request; the grid point forces one row per
// element and grid point plus the applied load, constraint and total rows of each grid point; the strain energy one row
// per element.
//
// Thread-safety: none (filled case by case by one thread). Ownership: borrows the element set and the DOF map.
class OutputVolumeEstimate
{
public:
    OutputVolumeEstimate(const assembly::ElementSet& set, const assembly::DofMap& dofMap);

    // Adds the tables one case gives under `plan` (the m set: whether MPC forces are written). `isKindKept(kind)`
    // false leaves a kind out (an envelope replaces its per-case tables).
    template <typename IsKindKept>
    void addCase(const OutputPlan& plan, bool hasMset, IsKindKept&& isKindKept)
    {
        for (std::size_t k = 0; k < OUTPUT_KIND_COUNT; ++k)
        {
            const auto kind = static_cast<OutputKind>(k);
            if (!plan.of(kind).isRequested || !isKindKept(std::string(tableKindOf(kind))))
                continue;
            if (kind == OutputKind::MpcForces && !hasMset)
                continue;
            m_bytes[k] += cachedBytesOf(kind, plan.of(kind));
        }
        ++m_caseCount;
    }

    [[nodiscard]] std::uint64_t totalBytes() const noexcept;
    [[nodiscard]] std::uint64_t bytesOf(OutputKind kind) const noexcept
    {
        return m_bytes[static_cast<std::size_t>(kind)];
    }
    [[nodiscard]] std::int64_t caseCount() const noexcept { return m_caseCount; }

    // Throws OutputVolumeError when the total exceeds `limitBytes`: the message gives the estimate, the largest kinds
    // and the remedies (restrict the outputs by a SET, ask for envelopes, write to a file).
    void requireWithin(std::uint64_t limitBytes) const;
    // The same for tables written to files as they come: `limitBytes` is the free space of their disk (the estimate
    // counts 8 bytes per value; the files take 4 in single precision, or 8).
    void requireWithinDestination(std::uint64_t limitBytes) const;

private:
    // Throws the OutputVolumeError of a volume above `limitBytes` of `destination`, `lastRemedy` after the SET and
    // envelope remedies.
    [[noreturn]] void refuse(std::uint64_t limitBytes, std::string_view destination, std::string_view lastRemedy) const;
    // The bytes of one case's table of `kind` under its request, the last request of each kind remembered (the cases
    // of a deck mostly share their requests: the element counts are not taken again).
    [[nodiscard]] std::uint64_t cachedBytesOf(OutputKind kind, const OutputRequestPlan& request);
    [[nodiscard]] std::uint64_t bytesOf(OutputKind kind, const OutputRequestPlan& request);
    // The rows and columns of the first element of a batch for an element kind (cached).
    struct RowShape
    {
        std::uint64_t rows = 0;
        std::uint64_t columns = 0;
        std::uint64_t grids = 0; // the grid points of the element (from its stiffness DOFs)
        bool isKnown = false;
    };
    [[nodiscard]] RowShape shapeOf(std::size_t batch, OutputKind kind, bool isFiber);

    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    std::array<std::uint64_t, OUTPUT_KIND_COUNT> m_bytes{};
    std::int64_t m_caseCount = 0;
    std::vector<std::array<RowShape, 5>> m_shapes; // per batch: stress, strain (STRCUR), strain (FIBER), force, grid
    std::array<OutputRequestPlan, OUTPUT_KIND_COUNT> m_lastRequests;
    std::array<std::uint64_t, OUTPUT_KIND_COUNT> m_lastBytes{};
    std::array<bool, OUTPUT_KIND_COUNT> m_hasLast{};
};

} // namespace oos::output
