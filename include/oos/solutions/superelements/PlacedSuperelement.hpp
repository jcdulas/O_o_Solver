#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/solutions/superelements/SuperelementPart.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::solutions
{

// What a reduced superelement brings to the residual structure, placed on its g set (the residual structure is the
// interface problem): its matrices on the residual DOFs its boundary points are (SECONCT: a boundary point of the
// superelement is a point of the residual structure), its rows in this order:
// - its boundary DOFs (the rows of K*: a part's reduced boundary, or the a set of an external superelement), each the
//   DOF of the same component of the residual point its point is;
// - the generalised coordinates of its q set (Craig-Bampton), each the residual scalar point its q set point is.
// Its stiffness [K* 0; 0 K_qq], its mass [M_bb M_bq; M_qb M_qq] (a modal analysis), its loads f* per subcase of the
// residual structure (zero for a subcase that does not apply to it).
//
// The displacement systems: a boundary grid point must have the same displacement system (CD) in the superelement and
// in the residual structure (to 1e-12 of each term of the rotations), so that a component is the same direction on
// both sides; another one is refused (a superelement moved by SuperelementLocation or SuperelementMirrorPlane, or a CD
// that differs, is not supported by this version).
//
// Determinism: the sums into the residual's matrices in a fixed order (rows, then columns).
//
// Thread-safety: immutable once built. Ownership: owns its matrices (budgeted: the budget must outlive it).
class PlacedSuperelement
{
public:
    // The rows: `labels` (the superelement's own IDs: boundary DOFs then, for a q set, its scalar points, component
    // 0) and the boundary map. A label whose point the map does not join to the residual structure, or whose residual
    // point has no such DOF, is an error of the sink (the row is then left out: no DOF). The superelement's DOF map
    // (null for an external superelement) checks the displacement systems.
    PlacedSuperelement(std::int32_t superelementId, std::span<const reduction::UnitDofLabel> labels,
                       std::span<const SuperelementBoundaryPoint> boundary, const assembly::DofMap& residual,
                       const assembly::DofMap* superelement, base::DiagnosticSink& sink);

    [[nodiscard]] std::int32_t superelementId() const noexcept { return m_superelementId; }
    [[nodiscard]] std::size_t size() const noexcept { return m_dofs.size(); }
    // The residual g-set DOF of each row (-1: none, after an error).
    [[nodiscard]] std::span<const base::DofIndex> dofs() const noexcept { return m_dofs; }
    // The residual nodes of its rows, sorted (a clique of its matrices: one group of the node graph).
    [[nodiscard]] std::span<const std::int32_t> nodes() const noexcept { return m_nodes; }

    // Its matrices (size() square) and loads (size() × the residual's subcases), set once.
    void setStiffness(linalg::DenseMatrix stiffness) { m_stiffness = std::move(stiffness); }
    void setMass(linalg::DenseMatrix mass) { m_mass = std::move(mass); }
    void setLoads(linalg::DenseMatrix loads) { m_loads = std::move(loads); }
    [[nodiscard]] const linalg::DenseMatrix& stiffness() const noexcept { return m_stiffness; }
    [[nodiscard]] bool hasMass() const noexcept { return m_mass.has_value(); }
    [[nodiscard]] const linalg::DenseMatrix& loads() const noexcept { return m_loads; }

    // Adds the stiffness (mass) onto `target`, a g-set matrix of the residual whose pattern couples nodes(). Throws
    // std::logic_error when two of its nodes are not coupled.
    void addStiffnessTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& residual) const;
    void addMassTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& residual) const;
    // Adds column `column` of its loads onto a residual g-set vector.
    void addLoadsTo(std::size_t column, std::span<double> loadsG) const;
    // The values of its rows (size() × columns) from residual g-set vectors (g × columns).
    [[nodiscard]] linalg::DenseMatrix gather(linalg::ConstDenseMatrixView valuesG, base::MemoryBudget& budget) const;

private:
    void addMatrixTo(const linalg::DenseMatrix& matrix, linalg::BlockSparseMatrix& target,
                     const assembly::DofMap& residual) const;

    std::int32_t m_superelementId = 0;
    std::vector<base::DofIndex> m_dofs; // bounded by the superelement's boundary
    std::vector<std::int32_t> m_nodes;  // bounded by the superelement's boundary
    linalg::DenseMatrix m_stiffness;
    std::optional<linalg::DenseMatrix> m_mass;
    linalg::DenseMatrix m_loads;
};

} // namespace oos::solutions
