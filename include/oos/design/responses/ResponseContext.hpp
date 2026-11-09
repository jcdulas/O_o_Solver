#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/aids/LinearityIndicator.hpp"
#include "oos/design/aids/SelfAdjointResponses.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/responses/ResultLookup.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/EigenvalueTable.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/diagnostics/SubcaseChecks.hpp"
#include "oos/output/sections/SectionLoadTable.hpp"
#include "oos/output/sections/ZoneTable.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace oos::design
{

// What the responses of one design point read: the overlaid snapshot the analysis ran on (and what owns it), the
// snapshot of the initial design (FRMASS), the result tables, the values of the design point, the mass formulation of
// the analysis (PARAM COUPMASS), the budget. Built by DesignProblem for one evaluation.
//
// The response kinds of the optimisation aids read more: the checks of the analysis (the external work of
// the compliance), its result arrays (the applied loads of the mean displacement), the subcases with enforced
// displacements (where the self-adjoint kinds are refused), and the shell geometry of the linearity indicator.
//
// Thread-safety: used on one thread (the lookup and the index are built on first use). Ownership: borrows everything;
// owns the snapshot index and the linearity indicator it builds.
struct ResponseContext
{
    const input::ModelSnapshot* snapshot = nullptr;
    std::shared_ptr<const void> snapshotOwner;
    const input::ModelSnapshot* initialSnapshot = nullptr; // nullptr: FRMASS refused
    std::shared_ptr<const void> initialSnapshotOwner;
    ResultLookup* results = nullptr; // nullptr: only the global responses can be evaluated
    const DesignModel* model = nullptr;
    const DesignValues* design = nullptr;
    elements::MassFormulation massFormulation = elements::MassFormulation::Lumped;
    base::MemoryBudget* budget = nullptr;
    std::unique_ptr<elements::SnapshotIndex> index;               // of `snapshot`, for the element sections (CSTRAT)
    std::optional<std::map<std::int32_t, double>> elementVolumes; // of `snapshot`, for the ESE density (item 4)
    // The checks (external work per subcase) and result arrays of the analysis: nullptr when not given (the
    // compliance and mean displacement are then refused).
    std::span<const output::SubcaseCheck> checks;
    const output::ResultArrays* arrays = nullptr;
    // The subcases whose constraints enforce a displacement (the self-adjoint kinds are refused there), among the
    // analysed subcases; computed on first use (hasEnforcedDisplacements).
    std::optional<std::set<std::int32_t>> enforcedSubcases;
    std::span<const std::int32_t> analysedSubcases;
    std::unique_ptr<LinearityIndicator> linearity; // of `snapshot`, built on first use
    // The section and cut loads of the analysis (SECTLOAD, CUTLOAD): nullptr when it computed none.
    const output::SectionLoadTable* sectionLoads = nullptr;
    const output::SectionLoadTable* cutLoads = nullptr;
    // The contour data of the zones of the analysis (CONTDISP, CONTFORC and the external responses).
    std::span<const output::ZoneTable> zones;
    // The eigenvalue tables of the MODES and BUCK subcases of the analysis (EIGN, FREQ, LAMA; empty when it
    // has none), and the mode an eigenvalue response takes in a subcase when it is not its ATTA (mode tracking:
    // DesignProblem::evaluateResponses), by (response ID, subcase).
    std::vector<const output::EigenvalueTable*> eigenvalueTables;
    std::map<std::pair<std::int32_t, std::int32_t>, std::int32_t> trackedModes;
    // The trim of the SAERO subcases (TRIM, STABDER): null without them.
    const solutions::TrimResults* trim = nullptr;

    // The first row of each section or cut of a section or cut loads table, by the group that declares it (the
    // response ID) and by its ID: built on first use for each table (DesignResp1Response::sectionValues reads it for
    // each response and subcase).
    using FirstSectionRows = std::map<std::int32_t, std::size_t>;
    std::map<const output::SectionLoadTable*, std::map<std::int32_t, FirstSectionRows>> sectionFirstRows;

    // The snapshot index, built on first use.
    [[nodiscard]] const elements::SnapshotIndex& snapshotIndex()
    {
        if (!index)
            index = std::make_unique<elements::SnapshotIndex>(*snapshot, *budget);
        return *index;
    }
    // The shell geometry of the linearity indicator, built on first use.
    [[nodiscard]] const LinearityIndicator& linearityIndicator()
    {
        if (!linearity)
            linearity = std::make_unique<LinearityIndicator>(*snapshot, snapshotOwner, *budget);
        return *linearity;
    }
    // The first rows of the sections or cuts of `group` in `table` (empty when it has none), the index of the table
    // built on its first use.
    // PERFORMANCE: each response searched the whole table for its rows in every subcase it reads (a table of 30 000
    // rows on a stiffened barrel panel model); the table is indexed once.
    // Measured: the response time of its design points 0.11 → 0.015 s (single thread).
    [[nodiscard]] const FirstSectionRows& firstSectionRows(const output::SectionLoadTable& table, std::int32_t group)
    {
        const auto [indexed, isNew] = sectionFirstRows.try_emplace(&table);
        if (isNew)
            for (std::size_t r = table.rowCount(); r-- > 0;)
            {
                const auto item = static_cast<std::size_t>(table.rowItems[r]);
                indexed->second[table.groups[item]][table.ids[item]] = r;
            }
        static const FirstSectionRows NO_ROWS;
        const auto rows = indexed->second.find(group);
        return rows == indexed->second.end() ? NO_ROWS : rows->second;
    }
    // Whether a subcase enforces a displacement (computed on first use over the analysed subcases).
    [[nodiscard]] bool hasEnforcedDisplacements(std::int32_t subcase)
    {
        if (!enforcedSubcases)
            enforcedSubcases =
                SelfAdjointResponses::subcasesWithEnforcedDisplacements(snapshotIndex(), analysedSubcases, *budget);
        return enforcedSubcases->count(subcase) != 0;
    }
};

} // namespace oos::design
