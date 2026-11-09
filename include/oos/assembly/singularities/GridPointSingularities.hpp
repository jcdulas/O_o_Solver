#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/ParamTable.hpp"

#include <cstdint>
#include <optional>
#include <ostream>
#include <span>

namespace oos::assembly
{

// The parameters of the grid point singularity processor (PARAM AUTOSPC, EPZERO, PRGPST), with the defaults of SOL
// 101, 103 and 105.
struct SingularityOptions
{
    static constexpr double DEFAULT_PIVOT_ZERO_THRESHOLD = 1.0e-8; // default of PARAM EPZERO

    bool isAutoSpcOn = true; // AUTOSPC YES (the default in the structural solutions)
    double pivotZeroThreshold = DEFAULT_PIVOT_ZERO_THRESHOLD;
    bool isTablePrinted = true; // PRGPST YES

    // The options of a subcase from the PARAM table (a Case Control PARAM overrides the Bulk Data one); `params` may be
    // null (no table: the defaults). A value of the wrong kind or outside YES/NO is reported ("bad-param") and the
    // default kept.
    [[nodiscard]] static SingularityOptions fromParams(const input::ParamTable* params,
                                                       std::optional<std::int32_t> subcase, base::DiagnosticSink& sink);
};

// The stiffness the singularity processor looks at, through a neutral interface: for every grid point (by row) its
// two 3 × 3 diagonal blocks — translations 1-3 then rotations 4-6, each row-major, 18 values per grid point — and for
// every scalar point its diagonal term, of the n-set stiffness Knn (the g-set stiffness after the elimination of the m
// set, before AUTOSPC), in the displacement systems CD. The rows and columns of the m-set components are not read.
// Step 3.3 fills it from the assembled matrix; the tests from a dense one.
struct NodeDiagonalBlocks
{
    std::span<const double> gridBlocks;      // 18 per grid point
    std::span<const double> scalarDiagonals; // 1 per scalar point
};

// One line of the table of the singular components.
struct SingularityRow
{
    base::DofIndex dof = 0;
    base::EntityId pointId = 0;
    bool isScalarPoint = false;
    int component = 1;           // 1 to 6, 0 for a scalar point: the FAILED DIRECTION
    double stiffnessRatio = 0.0; // λ/λmax of the singular direction it stands for
    ExclusiveSet oldExclusive = ExclusiveSet::B;
    bool isConstrained = false; // constrained by AUTOSPC (the "*" of the table, NEW USET SB)
    bool isAligned = true;      // the singular direction lies along the component axes (see below)
};

// The grid point singularity processor (PARAM AUTOSPC and EPZERO), on the 3 × 3 translational and rotational blocks
// of each grid point and the 1 × 1 block of each scalar point, after the m set is eliminated.
//
// For each block, on its free components F (those in neither the m nor the s set: the constrained ones are already
// held and are not listed):
// 1. the eigenvalues λ of the |F| × |F| sub-block (Jacobi, SmallSymmetricEigen); λmax the largest;
// 2. a direction is singular when λ/λmax < EPZERO, every direction when λmax ≤ 0 (no stiffness at all; ratio 0); a
//    scalar point is singular when its diagonal is at most EPZERO times the largest diagonal term of the whole matrix
//    (a 1 × 1 block has no ratio of its own);
// 3. the s singular directions V (|F| × s) are removed by constraining s components C of F, chosen to maximise
//    |det V_C| (V restricted to the rows C): constraining C leaves the rest of the block nonsingular, and the choice is
//    the components best aligned with the singular directions (ties: the lowest components);
// 4. the singularity is "aligned" (the strong case) when |det V_C| ≥ ALIGNMENT_COSINE: the singular directions lie
//    along the components constrained (the usual case: a rotation of a solid or rod grid point, the drilling rotation
//    of a flat shell in its plane's system); otherwise it is a combination of components (the weak case, e.g. the
//    drilling rotation of a shell in a skewed system), still removed by constraining C, and reported with each
//    component's ratio;
// 5. each component of C is a row of the table, its ratio that of the singular direction it pairs with (the largest
//    |V(c, j)|); with AUTOSPC YES it is constrained (NEW USET SB, "*"), with AUTOSPC NO it is reported as a Warning
//    "grid-point-singularity" (the factorisation then meets the mechanism).
//
// The rules of steps 3 to 5 are this implementation's own design.
//
// Thread-safety: immutable once built. Ownership: owns its rows and DOFs (budgeted); the budget must outlive it.
class GridPointSingularities
{
public:
    // How close to the component axes a singular direction must be to be "aligned": cos 1°.
    static constexpr double ALIGNMENT_COSINE = 0.9998476951563913;

    // Throws std::invalid_argument when the sizes of `blocks` do not match the DOF map.
    GridPointSingularities(const DofMap& dofMap, const DofSets& sets, const NodeDiagonalBlocks& blocks,
                           const SingularityOptions& options, base::MemoryBudget& budget, base::DiagnosticSink& sink);

    // The table, by point ID then component.
    [[nodiscard]] std::span<const SingularityRow> rows() const noexcept { return {m_rows.data(), m_rowCount}; }
    // The DOFs AUTOSPC constrains (empty with AUTOSPC NO), increasing: DofSets::withAutoSpc() takes them.
    [[nodiscard]] std::span<const base::DofIndex> autoSpcDofs() const noexcept
    {
        return {m_autoSpcDofs.data(), m_autoSpcCount};
    }
    [[nodiscard]] const SingularityOptions& options() const noexcept { return m_options; }

    // The table of the singular components of the text report (nothing when there is no singularity, or with
    // PARAM PRGPST NO: the constraints of AUTOSPC are then applied but not printed).
    void writeTable(std::ostream& out) const;

private:
    SingularityOptions m_options;
    base::BudgetedBuffer<SingularityRow> m_rows; // sized for every free DOF; the first m_rowCount are used
    std::size_t m_rowCount = 0;
    base::BudgetedBuffer<base::DofIndex> m_autoSpcDofs;
    std::size_t m_autoSpcCount = 0;
};

// The table of the singular components of `rows` (sorted by point ID then component) in the text report, nothing
// when there is no row or with PARAM PRGPST NO: GridPointSingularities::writeTable, for rows gathered elsewhere (the
// condensed path gathers those of its units and of its interface problem).
void writeSingularityTable(std::ostream& out, std::span<const SingularityRow> rows, const SingularityOptions& options);

} // namespace oos::assembly
