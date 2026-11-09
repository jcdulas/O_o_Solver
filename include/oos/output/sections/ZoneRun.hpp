#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/output/arrays/ResultCombination.hpp"
#include "oos/output/sections/ZoneContour.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"
#include "oos/output/sections/ZoneTable.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace oos::output
{

// The zones of external local models whose contour data a static analysis computes (StaticOptions::zones): their
// definitions, the subcases whose constraints decide their contours (empty: those of the analysis), whether the data of
// every case are kept, and whether the data on the load basis are kept (a linear plugin works on them).
struct ZoneRequest
{
    std::vector<ZoneDefinition> zones;
    std::vector<ZoneSubcaseConstraint> contourSubcases;
    bool keepsValues = true;
    bool keepsBasis = false;
    // The most cases of one batch (0: the static solution's batches); the tests lower it.
    std::size_t maximumBatchCases = 0;

    [[nodiscard]] bool isEmpty() const noexcept { return zones.empty(); }
};

// The g-set vectors of one case, or of the k vectors of a load basis (g × k each): the displacements u, the applied
// loads P (OLOAD), the SPC forces R and the MPC forces q (no columns when the model has no m set).
struct ZoneVectors
{
    linalg::ConstDenseMatrixView displacements;
    linalg::ConstDenseMatrixView applied;
    linalg::ConstDenseMatrixView spcForces;
    linalg::ConstDenseMatrixView mpcForces;
};

// The forces of a condensed unit on its boundary DOFs (g set) in one case, standing for the contour forces of a zone
// that is the unit (on the condensed path, the contour data of such a zone need no recovery: they are the unit's u_a
// and its K*·u_a − f*): the forces of its elements at its boundary, without the thermal loads of its elements there
// (they are the interface's loads), which ZoneRun adds as for any zone.
struct ZoneGivenForces
{
    std::int32_t zoneId = 0;
    std::span<const base::DofIndex> dofs;
    std::span<const double> forces;
    // A constraint inside the unit takes a reaction, known as a total only: the interior resultant is then the unit's
    // by its equilibrium; otherwise it is that of the interior loads, as for any zone.
    bool hasInteriorReactions = false;
};

// The contour data of the zones of one static analysis on the load basis (the contour data are linear in the
// displacements: they are computed on the load basis and combined), as SectionLoadRun does for the section loads:
//  1. at the start, the zones resolved (ZoneContour: every problem a diagnostic, before any factorisation) and their
//     rows made;
//  2. per constraint set, the data of the k basis vectors, B = S·[u; q; P; R] (rows × k, sparse products), kept per
//     set when the basis is kept (with α and the cases of the set);
//  3. per batch of load cases, Y = B·A(:, batch) (ResultCombination, the same bits whatever the threads), the thermal
//     part of each case with a thermal load added (the element part at zero displacements with its temperatures);
//  4. a case known by its vectors only (a SUBCOM, the condensed path) from the rows directly.
//
// Thread-safety: none (driven by the analysis, in case order). Ownership: owns its contours and tables; borrows the
// element set and the DOF map, which must outlive it.
class ZoneRun
{
public:
    // Resolves the zones (their problems reported to `sink`: the caller stops on errors) and, when they have none,
    // makes their rows and the tables (the kept values of `caseCount` cases reserved now). `analysisSubcases`: the
    // subcases of the analysis, for a request without contourSubcases.
    ZoneRun(const ZoneRequest& request, std::span<const ZoneSubcaseConstraint> analysisSubcases,
            const input::ModelSnapshot& snapshot, const assembly::ElementSet& set, const assembly::DofMap& dofMap,
            std::size_t caseCount, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] const ZoneContour& contour() const noexcept { return *m_contour; }
    [[nodiscard]] bool keepsBasis() const noexcept { return m_keepsBasis; }

    // The data of the basis vectors, per zone: the four matrices of its rows (contour displacements, contour forces,
    // interior loads, equilibrium) × k.
    struct BasisValues
    {
        std::vector<std::array<linalg::DenseMatrix, 4>> zones;
    };
    [[nodiscard]] BasisValues basisValues(const ZoneVectors& vectors, base::MemoryBudget& budget,
                                          base::ThreadPool* pool) const;
    // Keeps the basis of a constraint set (keepsBasis): its cases, α (k × cases) and whether a case has a thermal load.
    void keepBasis(std::int32_t constraintSet, std::span<const std::int32_t> cases, const BasisValues& basis,
                   const linalg::DenseMatrix& coefficients, bool hasThermalCases, base::MemoryBudget& budget);
    using TemperaturesOf = std::function<std::unique_ptr<assembly::TemperatureField>(std::size_t)>;
    // The cases first ... first + cases.size() of `combination` (their subcases `cases`), from the data of their basis.
    void offerBatch(const BasisValues& basis, const ResultCombination& combination, std::size_t first,
                    std::span<const std::int32_t> cases, const TemperaturesOf& temperaturesOf,
                    base::MemoryBudget& budget);
    // One case from its vectors (one column each) and temperatures. `given`: the zones whose contour forces are given
    // by the analysis (a zone that is a condensed unit not recovered: its interior displacements are not known): their
    // contour forces and the contour resultant come from them; with interior reactions (known as totals only, as in the
    // checks of the condensed path), the interior resultant of their equilibrium is the unit's by its equilibrium.
    void offerCase(std::int32_t subcase, const ZoneVectors& vectors, const elements::ElementTemperatures* temperatures,
                   base::MemoryBudget& budget, std::span<const ZoneGivenForces> given = {});

    // The tables once every case was offered.
    [[nodiscard]] std::vector<ZoneTable> takeTables();

private:
    // Appends the values of one case to the tables: `values` per zone, the four row blocks.
    void store(std::int32_t subcase, std::size_t z, std::span<const double> displacements,
               std::span<const double> forces, std::span<const double> loads, std::span<const double> equilibrium);

    const assembly::DofMap* m_dofMap = nullptr;
    std::unique_ptr<ZoneContour> m_contour;
    std::vector<ZoneTable> m_tables;
    std::vector<std::size_t> m_offered; // per zone, the cases stored
    std::size_t m_caseCount = 0;
    bool m_keepsValues = true;
    bool m_keepsBasis = false;
    std::size_t m_maximumBatchCases = 0;
};

} // namespace oos::output
