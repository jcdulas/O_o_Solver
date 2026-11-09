#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::assembly
{

// The volume of each element of an element set: the total translational mass of its lumped mass matrix in the
// snapshot with every density 1 and every non-structural mass 0 (unitDensitySnapshot), so length × area for the rods,
// bars and beams, area × thickness for the shells (∫t dA with corner thicknesses, area × Σ t_k for a laminate), the
// volume for the solids. The element strain energy density is the strain energy divided by this volume (the ESE
// output, and the item 4 of a DesignResp1 ESE). Only the structural elements have one, those whose family has both a
// stiffness and a mass: not the concentrated masses (CONM1, PointMassInertia, CMASSi) nor the springs, bushes and dampers.
//
// The mass of an element is Σ_ij d_i·M_ij·d_j with d the x direction of the basic system at each translational DOF
// (the (1, 1) term of its rigid-body mass Dᵀ·M·D, which the rotation of the displacement systems leaves the total
// mass); computed range by range (bounded blocks), the ranges on the pool (each element's volume alone: the same bits
// whatever the threads).
//
// Thread-safety: immutable. Ownership: owns its per-batch arrays (budgeted); the budget must outlive it.
class ElementVolumes
{
public:
    // The volumes of the elements of the batches of `set` that `isBatchWanted` flags (one flag per batch of `set`;
    // every batch when empty). The arrays follow the batch order of `set`.
    ElementVolumes(const ElementSet& set, std::span<const std::uint8_t> isBatchWanted, base::MemoryBudget& budget,
                   base::ThreadPool* pool = nullptr);

    // The volume of each row of batch `batchIndex` (an index in ElementSet::batches()), or an empty span when the
    // batch was not wanted or is not of structural elements.
    [[nodiscard]] std::span<const double> volumesOf(std::size_t batchIndex) const noexcept
    {
        return m_volumes[batchIndex].span();
    }

    // The snapshot `snapshot` with every density (rho) 1 and every non-structural mass (nsm) 0, in every table of
    // materials, properties and elements that has the column (input::ColumnOverlay). `keepAlive` as for
    // ColumnOverlay::applyTo.
    [[nodiscard]] static input::ModelSnapshot unitDensitySnapshot(const input::ModelSnapshot& snapshot,
                                                                  const std::shared_ptr<const void>& keepAlive);

private:
    std::vector<base::BudgetedBuffer<double>> m_volumes; // one per batch of the set
};

} // namespace oos::assembly
