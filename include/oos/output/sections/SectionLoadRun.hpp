#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/output/arrays/ResultCombination.hpp"
#include "oos/output/sections/CutLoads.hpp"
#include "oos/output/sections/SectionLoadAccumulator.hpp"
#include "oos/output/sections/SectionLoads.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::output
{

// The section and cut loads a static analysis computes (StaticOptions::sectionLoads): the definitions, whether
// the values of every case are kept, and the sink of the batches of load cases (called on the analysis thread, sections
// and cuts each with their own table).
struct SectionLoadRequest
{
    std::vector<SectionDefinition> sections;
    std::vector<CutDefinition> cuts;
    bool keepsValues = true;
    std::function<void(const SectionLoadBatch&)> sink;
    // The most cases of one batch: 0 (the default) for the batches of the static solution's combination
    // (ResultCombination::batchCaseCount, within BATCH_BYTES for the g-set quantities, so that a batch of rows ×
    // cases stays within that bound too); the tests lower it so that a few hundred cases come in several batches.
    std::size_t maximumBatchCases = 0;

    [[nodiscard]] bool isEmpty() const noexcept { return sections.empty() && cuts.empty(); }
};

// The section and cut loads of one static analysis, on the load basis (both are linear in the displacements:
// they are computed on the vectors of the load basis and combined for every load case):
//  1. at the start, the definitions resolved and checked (SectionLoads, CutLoads: every problem a diagnostic, before
//     any factorisation) and their displacement rows made;
//  2. per constraint set, the rows applied to the displacements of the k basis vectors, B = S·U (rows × k, a sparse
//     product: SparseRows::apply);
//  3. per batch of load cases, the values Y = B·A(:, batch) (ResultCombination: one product, the same bits whatever
//     the threads), the thermal part of each case with a thermal load added (the recovery at zero displacements with
//     its temperatures: the loads are affine in the displacements), then offered to the accumulators (envelopes, sink,
//     kept values);
//  4. a case known by its displacements only (a SUBCOM, the condensed path) gives its values from the rows directly.
// A bar or beam of a section under a line load (PLOAD1) is refused for that subcase (checkLineLoads): its forces are
// then given at the stations of the element, a layout the sections do not read.
//
// Thread-safety: none (driven by the analysis, in case order). Ownership: owns its sections, cuts and accumulators;
// borrows the element set and the DOF map, which must outlive it.
class SectionLoadRun
{
public:
    // Resolves the definitions (their problems reported to `sink`: the caller stops on errors) and, when they have
    // none, makes the displacement rows and the accumulators (the kept values of `caseCount` cases reserved now).
    SectionLoadRun(const SectionLoadRequest& request, const input::ModelSnapshot& snapshot,
                   const assembly::ElementSet& set, const assembly::DofMap& dofMap, std::size_t caseCount,
                   base::MemoryBudget& budget, base::DiagnosticSink& sink, base::ThreadPool* pool = nullptr);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_sectionRows + m_cutRows; }

    // Reports an error for each bar or beam of a section that a line load of the subcase loads (lineLoads: per batch
    // of the element set, as LoadVectorBuilder::lineLoadsOf gives them).
    void checkLineLoads(std::int32_t subcase, std::span<const std::vector<elements::ElementLineLoad>> lineLoads,
                        base::DiagnosticSink& sink) const;

    // B = S·U, the rows of the basis vectors (rows × k) from their g-set displacements (g × k).
    [[nodiscard]] linalg::DenseMatrix basisValues(linalg::ConstDenseMatrixView displacements,
                                                  base::MemoryBudget& budget, base::ThreadPool* pool) const;
    // The temperatures of case j of a batch (an index into its cases), or null without a thermal load; made one case at
    // a time, so that a batch holds one temperature field.
    using TemperaturesOf = std::function<std::unique_ptr<assembly::TemperatureField>(std::size_t)>;
    // The cases first ... first + cases.size() of `combination` (their subcases `cases`), from the rows of their basis.
    void offerBatch(const linalg::DenseMatrix& basis, const ResultCombination& combination, std::size_t first,
                    std::span<const std::int32_t> cases, const TemperaturesOf& temperaturesOf,
                    base::MemoryBudget& budget);
    // One case from its g-set displacements (and temperatures).
    void offerCase(std::int32_t subcase, std::span<const double> displacements,
                   const elements::ElementTemperatures* temperatures, base::MemoryBudget& budget);

    // The tables once every case was offered (nothing for a kind without definitions).
    [[nodiscard]] std::optional<SectionLoadTable> takeSectionTable();
    [[nodiscard]] std::optional<SectionLoadTable> takeCutTable();

private:
    // Adds the thermal part of one case (the rows at zero displacements) to column j of `values`.
    void addThermalPart(const elements::ElementTemperatures& temperatures, linalg::DenseMatrixView values,
                        std::size_t j) const;
    void offerRows(std::span<const std::int32_t> cases, linalg::ConstDenseMatrixView values);

    const assembly::ElementSet* m_set;
    std::unique_ptr<SectionLoads> m_sections;
    std::unique_ptr<CutLoads> m_cuts;
    std::size_t m_sectionRows = 0;
    std::size_t m_cutRows = 0;
    std::unique_ptr<SectionLoadAccumulator> m_sectionAccumulator;
    std::unique_ptr<SectionLoadAccumulator> m_cutAccumulator;
    base::BudgetedBuffer<double> m_zeros; // g-set zeros, for the thermal parts
    std::size_t m_maximumBatchCases = 0;
};

} // namespace oos::output
