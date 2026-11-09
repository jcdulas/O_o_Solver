#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::reduction
{

// A term of a row of Gm on a local numbering: the independent DOF and its coefficient.
struct LocalTerm
{
    std::int64_t dof = 0;
    double coefficient = 0.0;
};

// The rows of a Gm on the node-major DOFs of a local numbering (a unit's, UnitSystem; the interface problem's,
// InterfaceProblem): u_m = Gm·u_n, every term on an independent DOF (assembly::RigidElimination resolves the chains),
// with the linear elimination of assembly::MsetElimination on the g set:
//     Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm,   P_n = P̄_n + Gmᵀ·P_m,   u_m = Gm·u_n.
//
// Thread-safety: immutable once filled; concurrent reads are safe. Ownership: owns its rows (budgeted).
class LocalRelations
{
public:
    LocalRelations() = default;
    // Room for rowCount rows and termCount terms, filled by appendRow and appendTerm in increasing order of dependent
    // DOF. Throws MemoryBudgetError.
    LocalRelations(std::size_t rowCount, std::size_t termCount, base::MemoryBudget& budget, const std::string& label);

    // Starts the next row (its terms follow). Throws std::logic_error past the rows given, or when the dependent DOFs
    // do not increase.
    void appendRow(std::int64_t dependent);
    // Adds a term to the last row. Throws std::logic_error past the terms given or before the first row.
    void appendTerm(LocalTerm term);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_rowCount; }
    [[nodiscard]] std::size_t termCount() const noexcept { return m_termCount; }
    [[nodiscard]] std::int64_t dependent(std::size_t row) const noexcept { return m_dependents[row]; }
    [[nodiscard]] std::span<const LocalTerm> terms(std::size_t row) const noexcept
    {
        const auto first = static_cast<std::size_t>(m_offsets[row]);
        const auto last = row + 1 < m_rowCount ? static_cast<std::size_t>(m_offsets[row + 1]) : m_termCount;
        return {m_terms.data() + first, last - first};
    }

    // Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm in place on `values`, a symmetric block matrix on `pattern` whose
    // nodes own the DOFs [firstDof[n], firstDof[n + 1]): every read is a term with a dependent index (K(i, d)), every
    // write a term between two independent DOFs (the rows are resolved onto the n set), so no read sees a write. One
    // serial pass in a fixed order (the dependent DOFs increasing, the DOFs coupled to each by node and component), so
    // the bits do not depend on the threads. `graph` must hold the couplings of the rows (each group of the matrix
    // extended with the nodes of its dependents' terms, as assembly::MsetCouplings): std::logic_error otherwise. The
    // rows and columns of the dependent DOFs are left as they were (stale); MemoryBudgetError for its index.
    void eliminate(const linalg::SparsePattern& pattern, std::span<const std::int64_t> firstDof,
                   const linalg::NodeGraph& graph, std::span<double> values, base::MemoryBudget& budget) const;

    // P_n += Gmᵀ·P_m, then P_m = 0, on one vector of the local DOFs.
    void reduce(std::span<double> loads) const noexcept;
    // u_m = Gm·u_n on one vector of the local DOFs; the rows flagged in isSkipped (one flag per row; empty: none) keep
    // their values.
    void expand(std::span<double> values, std::span<const std::uint8_t> isSkipped = {}) const noexcept;

private:
    std::size_t m_rowCount = 0;
    std::size_t m_termCount = 0;
    base::BudgetedBuffer<std::int64_t> m_dependents;
    base::BudgetedBuffer<std::int64_t> m_offsets; // the first term of each row
    base::BudgetedBuffer<LocalTerm> m_terms;
};

} // namespace oos::reduction
