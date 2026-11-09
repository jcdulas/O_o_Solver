#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace oos::output
{

// The rows of a linear map of the g-set displacements (compressed sparse rows): row r is Σ_k c_k·u[d_k] over its
// entries (d_k, c_k), the g-DOFs in increasing order. The section and cut loads are such maps (their displacement
// part: SectionLoads, CutLoads), so that their values for the k vectors of a load basis are one product, and the rows
// are the weights ∂r/∂u of their sensitivities.
//
// Determinism: each value is one sum over the entries of its row in their order, whatever the threads.
//
// Thread-safety: immutable once built. Ownership: owns its arrays (budgeted).
class SparseRows
{
public:
    SparseRows() = default;

    // The rows of `rows` (each a list of (DOF, coefficient), DOFs sorted and distinct), copied into budgeted arrays.
    SparseRows(const std::vector<std::vector<std::pair<base::DofIndex, double>>>& rows, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_starts.empty() ? 0 : m_starts.size() - 1; }
    [[nodiscard]] std::size_t entryCount() const noexcept { return m_dofs.size(); }
    [[nodiscard]] std::span<const base::DofIndex> dofsOf(std::size_t row) const;
    [[nodiscard]] std::span<const double> coefficientsOf(std::size_t row) const;

    // out(r, j) = row r · displacements(:, j), for the k columns of `displacements` (g × k); `out` is rows × k. The
    // rows are shared between the threads of `pool` by blocks (null: one thread). Throws std::invalid_argument for
    // sizes that disagree.
    void apply(linalg::ConstDenseMatrixView displacements, linalg::DenseMatrixView out,
               base::ThreadPool* pool = nullptr) const;
    // values[r] = row r · displacements (one g-set vector).
    void apply(std::span<const double> displacements, std::span<double> values) const;

private:
    base::BudgetedBuffer<std::size_t> m_starts; // rowCount() + 1
    base::BudgetedBuffer<base::DofIndex> m_dofs;
    base::BudgetedBuffer<double> m_coefficients;
};

} // namespace oos::output
