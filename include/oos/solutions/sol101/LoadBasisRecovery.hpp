#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/QuantityMap.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/sections/SectionLoadRun.hpp"
#include "oos/output/sections/SectionLoadTable.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace oos::solutions
{

// The results of the units of a kept load basis, recovered on demand (unit blocks, lazily): the units are numbered 1 to
// K over the constraint sets of the basis, in their order, and each is recovered as a subcase whose ID is its number,
// from its g-set displacements, as the static solution recovers a case (output::NodalResultBuilder,
// output::ElementResultBuilder: the layouts of the data contract, the element frames of the result tables), or its
// section and cut loads (output::SectionLoadRun::offerCase). The units carry no line load (the static solution refuses
// them in a kept basis); a thermal unit has its element tables recovered with the temperatures of its pair of sets, the
// thermal strain subtracted as the static solution recovers a thermal case, so that the tables of a case are the sums
// of the tables of its units with its coefficients. The grid point forces, the declared quantities and the section and
// cut loads of a thermal unit are refused (SolverError): its elements' thermal loads are not in them.
//
// The element set is made again on the run's index (with the standard element registry): the run's own set is gone
// with its analysis model; no external unit replaced elements in a kept basis.
//
// On the condensed path (KeptLoadBasisSet::condensed) a call recovers, for the units it asks of a set together, the
// interiors of the condensed units its points, elements, sections or cuts read (CondensedBasisColumns: the regime of
// fewest interior solves for those units), nothing else; the interiors are not kept between calls. In a memory mode
// that keeps no factor each unit read has its interior factorised again (or recovered by regions) and its equations
// built again when released, then released once its interiors are recovered: nothing is kept between calls either.
//
// Thread-safety: the const methods may run concurrently (each fills its own tables; the pool given is used); their
// recoveries of condensed interiors, which make and release the units' factors, run one at a time. Ownership: shares
// the kept basis; owns its element set; borrows the snapshot, which must outlive it.
class LoadBasisRecovery
{
public:
    // Throws std::invalid_argument when the basis is null or lacks its index or DOF map.
    LoadBasisRecovery(const input::ModelSnapshot& snapshot, std::shared_ptr<const KeptLoadBasis> basis);

    [[nodiscard]] const KeptLoadBasis& basis() const noexcept { return *m_basis; }
    // K: the units of every constraint set.
    [[nodiscard]] std::int32_t unitCount() const noexcept { return m_unitCount; }

    // The tables of the units `units` (numbers 1..K, any order, one subcase each in that order): the displacements
    // of `points` when `withDisplacements` (sorted; empty with isAllPoints for every point), the element tables the
    // selection asks for and, with `gridPointForces` requested, the grid point forces of its points: the rows of the
    // elements there (−K_e·u_e, the GPFORCE rows, only the elements with a grid point asked computed, a block of
    // units at a time) and, at the points where the unit's set keeps them (KeptLoadBasisSet::fixedRows), its
    // appliedLoad, singlePointConstrForce, multiPointConstrForce and total rows as statics writes them
    // (BasisFixedRows); elsewhere the element rows alone. Throws std::invalid_argument for a unit outside 1..K;
    // MemoryBudgetError.
    [[nodiscard]] output::ResultArrays recover(std::span<const std::int32_t> units, bool withDisplacements,
                                               bool isAllPoints, std::span<const std::int32_t> points,
                                               const output::ElementOutputSelection& selection,
                                               base::MemoryBudget& budget, base::ThreadPool* pool,
                                               const output::OutputRequestPlan* gridPointForces = nullptr) const;

    // The declared quantities of the units `units`: the force tables of the map's elements alone recovered (the
    // describers of `selection`; on the condensed path the interiors of the units holding them), a block of units at a
    // time, each unit's reduced by `map` (QuantityMap::reduce) and its tables then dropped. Returns units ×
    // map.quantityCount() values, row-major. Throws std::invalid_argument for a unit outside 1..K; MemoryBudgetError.
    [[nodiscard]] base::BudgetedBuffer<double> quantities(std::span<const std::int32_t> units,
                                                          const output::QuantityMap& map,
                                                          const output::ElementOutputSelection& selection,
                                                          base::MemoryBudget& budget, base::ThreadPool* pool) const;

    // The section or cut loads (`request`, its definitions resolved on the model: their problems go to `sink`, and
    // none is made when it has errors) of the units `units`, one case each, its subcase the unit's number.
    [[nodiscard]] std::optional<output::SectionLoadTable> sectionLoads(const output::SectionLoadRequest& request,
                                                                       bool isCut, std::span<const std::int32_t> units,
                                                                       base::MemoryBudget& budget,
                                                                       base::DiagnosticSink& sink) const;

    // Whether the basis is the full path streamed's (KeptLoadBasisSet::streamed): its declared quantities and the grid
    // point forces of its declared points alone (recover() and quantities() give those; anything else is refused).
    [[nodiscard]] bool isStreamed() const noexcept;
    // Whether one of `units` is a thermal unit (LoadBasisUnit::temperatureLoad).
    [[nodiscard]] bool hasThermalUnit(std::span<const std::int32_t> units) const;

private:
    class UnitDisplacements;
    [[nodiscard]] output::ResultArrays recoverStreamed(std::span<const std::int32_t> units, bool withDisplacements,
                                                       const output::ElementOutputSelection& selection,
                                                       const output::OutputRequestPlan* gridPointForces,
                                                       base::MemoryBudget& budget) const;
    // Where unit `unit` (1..K) is: its constraint set and its position among the set's units.
    struct UnitPlace
    {
        std::size_t set = 0;
        std::size_t column = 0;
    };
    [[nodiscard]] UnitPlace placeOf(std::int32_t unit) const;
    // The condensed units of the kept sets (0 on the full path), and those whose interiors a block reads: every one,
    // or those holding an interior point of `points`, an element of `elements`, or any point of `touchedPoints` (the
    // grid point forces there read the elements of the units around).
    [[nodiscard]] std::size_t condensedUnitCount() const noexcept;
    [[nodiscard]] std::vector<std::uint8_t> condensedUnitsRead(bool isEveryUnit, std::span<const std::int32_t> points,
                                                               std::span<const std::int32_t> elements,
                                                               std::span<const std::int32_t> touchedPoints = {}) const;

    // The elements of each node of the DOF map (made on the first grid point force request: ElementConnectivity),
    // and the rows of each batch whose elements have a node `isNodeTaken` flags, in increasing ranges.
    struct ElementsOfNodes
    {
        base::BudgetedBuffer<std::int64_t> offsets;  // nodes + 1
        base::BudgetedBuffer<std::int64_t> elements; // batchFirst[b] + row
        std::vector<std::int64_t> batchFirst;        // batches + 1 (bounded by the element types)
    };
    [[nodiscard]] const ElementsOfNodes& elementsOfNodes(base::MemoryBudget& budget, base::ThreadPool* pool) const;
    [[nodiscard]] std::vector<std::vector<elements::ElementRange>> rowsAtNodes(
        std::span<const std::uint8_t> isNodeTaken, base::MemoryBudget& budget, base::ThreadPool* pool,
        std::uint64_t& pairs) const;

    const input::ModelSnapshot* m_snapshot;
    std::shared_ptr<const KeptLoadBasis> m_basis;
    elements::ElementRegistry m_registry;
    std::unique_ptr<assembly::ElementSet> m_set; // borrows m_registry and the basis's index
    std::int32_t m_unitCount = 0;
    // The recoveries of condensed interiors one at a time (they make and release the kept units' factors).
    mutable std::mutex m_condensedRecovery;
    mutable std::mutex m_elementsOfNodesMade;
    mutable std::unique_ptr<const ElementsOfNodes> m_elementsOfNodes;
};

} // namespace oos::solutions
