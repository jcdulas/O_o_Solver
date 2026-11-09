#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/output/arrays/ElementResultTable.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::output
{

// What a subcase asks of one element table kind (OutputPlan: STRESS, STRAIN, FORCE): whether it is on, its elements
// (all, or the sorted members of its SET), whether the shells keep their corner rows (CORNER, BILIN) or their centre
// row only (CENTER, the default of the Case Control).
struct ElementKindSelection
{
    bool isRequested = true;
    bool isAll = true;
    std::span<const std::int32_t> members; // sorted, when !isAll; borrowed
    bool isCentreOnly = false;

    [[nodiscard]] bool takes(std::int32_t elementId) const;
    // Whether the two take the same elements at the same places.
    [[nodiscard]] bool selectsSameAs(const ElementKindSelection& other) const;
};

// The element tables a subcase asks for: one selection per kind, the strain form (FIBER: the strains
// at the fibres; otherwise STRCUR, the membrane strains and curvatures, for the shells that give both) and the stress
// column of the shells (MAXS, the default: the maximum shear (σ1 − σ2)/2 in place of the von Mises stress the families
// give; VONMISES keeps it).
struct ElementOutputSelection
{
    ElementKindSelection stress;
    ElementKindSelection strain;
    ElementKindSelection forces;
    bool isStrainAtFibres = false;
    bool isMaximumShear = false;

    // Whether the two make the same tables (the same kinds, elements, places and forms).
    [[nodiscard]] bool selectsSameAs(const ElementOutputSelection& other) const;
};

// The load-dependent parts of one case of a block (ElementResultBuilder::buildCases): its line loads per batch of the
// set (LoadVectorBuilder::lineLoadsOf; empty: none) and its temperatures (null: none). The cases whose `temperatures`
// is the same object share their rows at rest. Borrowed for the call.
struct CaseLoads
{
    std::span<const std::vector<elements::ElementLineLoad>> lineLoads;
    const elements::ElementTemperatures* temperatures = nullptr;
};

// The elements a build recovers and where it reads their displacements (the streamed path: the elements of
// a region, with the displacements of its points only): per batch of the set the ranges of its rows (sorted,
// disjoint; none: the batch is left out), and the displacements' rows, six per grid slot then one per scalar slot
// (elements::DisplacementSlots).
struct RecoveryWindow
{
    std::span<const std::vector<elements::ElementRange>> rowsOfBatch;
    elements::DisplacementSlots slots;
    std::size_t gridSlotCount = 0;
    std::size_t scalarSlotCount = 0;
};

// Makes the element result tables of the data contract from the displacements of a subcase: every batch of the
// element set whose family offers the Recovery facet gives, for each output kind it names columns for, a table of its
// rows in the layout of O_o_Reader's result tables:
//   - "stress" from ElementOutput::Stresses;
//   - "strain" from ElementOutput::StrainCurvatures when the family gives it (the shells: STRCUR, the membrane
//     strains and curvatures, is the default form of STRAIN), else ElementOutput::Strains;
//   - "forces" from ElementOutput::Forces;
//   - for the families with layers (the laminated shells), "stress", "strain" and "failureIndices" from
//     recoverLayers() (LayerOutput), under ElementFamily::layerResultElementType() ("ShellQuad4Composite").
// The element type of a table is ElementFamily::resultElementType(). The elements of a subcase loaded by line loads
// (PLOAD1 on a CBAR or CBEAM) are recovered by recoverUnderLineLoads(), in the station layout, which can show the
// forces between the ends (a CBAR under PLOAD1: its own element type); when that layout is the element's
// ordinary one (CBEAM), its rows join the ordinary table in the element order.
//
// The selection (from the Case Control requests and SETs: OutputPlan) limits each kind to its elements and,
// for the shells, to their centre rows (an element whose rows hold a centre row, grid point 0, and corner rows keeps
// the centre row only under CENTER: the data decide, so no family needs to say it has corners); without a selection
// every element of every family is recovered for every kind, at every place the family gives.
//
// A block of cases (buildCases) gives each case the tables build() gives it: the kinds of a family that completes its
// rows (ElementFamily::vectorisedRecoveryMinCases, vectorisedLayerRecoveryMinCases) are made from the recovery slopes
// S_e of each element and its rows at rest (recover() or recoverLayers() at zero displacements under the case's
// temperatures): the Linear columns of a row are rest + Σ_d S_e(row, column, d)·u_d, d over the element's DOFs in their
// order, for every case of the block (the element's slopes made once for the block, its rows at rest once per set of
// temperatures), the other columns completed by the family with the functions of its recovery (a ply's failure indices
// from the rows of its ply states: ElementFamily::vectorisedLayerSource; a CWELD's stresses from its forces:
// ElementFamily::vectorisedSource); equal to build()'s values to round-off. The elements of a case under line loads are
// recovered under their loads, as build() recovers them, in place of their vectorised rows. Every other kind is
// recovered case by case, with build()'s bits.
//
// Determinism: each element's rows depend on its own data only and the tables keep the batch order, whatever the
// pool; a vectorised row depends on its element and its case only, not on the other cases of the block. Memory: the
// rows are computed range by range (at most RANGE_ELEMENTS elements, a bounded block each); the blocks held until
// their table is made are charged to the budget, and each table is a budgeted buffer.
//
// Thread-safety: build() and buildCases() are const; they use the pool given. Ownership: borrows the element set and
// the DOF map.
class ElementResultBuilder
{
public:
    // The most elements recovered by one task: large enough to amortise the virtual call and the column reads, small
    // enough to bound the block of a task (a CQUAD4 has 10 stress rows of 8 values: 2 048 elements make a block of
    // about 1.3 MB).
    static constexpr std::int64_t RANGE_ELEMENTS = 2048;
    // With a pool, a batch is cut into about RANGES_PER_THREAD ranges per thread, so that a mid-size batch is shared
    // between the threads too (2 464 CQUAD4 made two ranges of 2 048: two threads), but none shorter than
    // MIN_RANGE_ELEMENTS, whose virtual calls and column reads would weigh. The rows do not depend on the ranges.
    static constexpr std::int64_t RANGES_PER_THREAD = 4;
    static constexpr std::int64_t MIN_RANGE_ELEMENTS = 64;
    // The most elements of one task of the vectorised recovery: a task holds the rows of its range for every case of
    // the block (a CQUAD4's 25 rows of 8 values, 256 elements and 32 cases: about 13 MB).
    static constexpr std::int64_t VECTORISED_RANGE_ELEMENTS = 256;

    // rangeElements: the elements of one task, RANGE_ELEMENTS but in the tests of the threaded recovery, which lower it
    // so that a small model has several ranges. Throws std::invalid_argument when it is not positive.
    ElementResultBuilder(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                         std::int64_t rangeElements = RANGE_ELEMENTS);

    // Appends to `tables` the element tables of one subcase. displacementsG: ug, one value per g-set DOF of the DOF
    // map. lineLoads: the line loads of the subcase per batch of the set (LoadVectorBuilder::lineLoadsOf), or empty
    // when it has none. temperatures: those of the subcase's thermal load, or null (the recovery then subtracts no
    // thermal strain). Throws std::invalid_argument when displacementsG is not of the g-set size or lineLoads neither
    // empty nor of one entry per batch; MemoryBudgetError; OperationCancelledError between batches.
    void build(std::int32_t subcase, std::span<const double> displacementsG,
               std::span<const std::vector<elements::ElementLineLoad>> lineLoads, base::MemoryBudget& budget,
               std::vector<ElementResultTable>& tables, base::ThreadPool* pool = nullptr,
               const base::CancellationToken* cancellation = nullptr,
               const elements::ElementTemperatures* temperatures = nullptr,
               const ElementOutputSelection* selection = nullptr) const;

    // Appends to tables[c] the element tables of case c of a block of cases, those build() gives it, in its order
    // (class comment): `subcases` the labels of the cases, `displacementsG` their g-set displacements (one column per
    // case, the g-set rows of the DOF map), `loads` their line loads and temperatures (one entry per case; empty:
    // none). The cases share `selection`. `vectorisedCases`: how many cases the caller recovers with buildCases in all
    // (every case of the run it could, not only those of the block); a kind of a family whose
    // vectorisedRecoveryMinCases is reached is vectorised, so that the bits of a case do not depend on the block it
    // comes in. Throws std::invalid_argument when the shapes disagree; MemoryBudgetError; OperationCancelledError
    // between batches.
    void buildCases(std::span<const std::int32_t> subcases, linalg::ConstDenseMatrixView displacementsG,
                    std::size_t vectorisedCases, base::MemoryBudget& budget,
                    std::span<std::vector<ElementResultTable>> tables, base::ThreadPool* pool = nullptr,
                    const base::CancellationToken* cancellation = nullptr,
                    const ElementOutputSelection* selection = nullptr, std::span<const CaseLoads> loads = {}) const;

    // buildCases() on the elements of a window only (RecoveryWindow), `displacements` on its slots: each case's tables
    // hold the rows of those elements, the rows buildCases() gives them (the same values to the bit). No line load and
    // no temperature (the streamed path refuses them). Throws as buildCases().
    void buildWindow(std::span<const std::int32_t> subcases, linalg::ConstDenseMatrixView displacements,
                     std::size_t vectorisedCases, const RecoveryWindow& window, base::MemoryBudget& budget,
                     std::span<std::vector<ElementResultTable>> tables, base::ThreadPool* pool = nullptr,
                     const base::CancellationToken* cancellation = nullptr,
                     const ElementOutputSelection* selection = nullptr) const;

private:
    void buildCasesIn(std::span<const std::int32_t> subcases, linalg::ConstDenseMatrixView displacements,
                      std::size_t vectorisedCases, base::MemoryBudget& budget,
                      std::span<std::vector<ElementResultTable>> tables, base::ThreadPool* pool,
                      const base::CancellationToken* cancellation, const ElementOutputSelection* selection,
                      std::span<const CaseLoads> loads, const RecoveryWindow* window) const;
    // The elements of one task for a batch of `rows` rows (RANGES_PER_THREAD).
    [[nodiscard]] std::int64_t rangeLengthOf(std::int64_t rows, const base::ThreadPool* pool) const noexcept;

    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    std::int64_t m_rangeElements;
};

} // namespace oos::output
