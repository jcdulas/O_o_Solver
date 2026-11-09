#pragma once

#include <cstddef>

namespace oos::linalg
{

// A non-owning view of a column-major matrix: element (i, j) is data[i + j·leadingDimension], with
// leadingDimension >= rows (and >= 1). Indices are 64-bit: a dense K* above 46 340 DOF has more than 2³¹ terms.
// A view never owns its memory; the owner (DenseMatrix, a front of the factorisation, a caller's array)
// must outlive it.
struct DenseMatrixView
{
    double* data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t leadingDimension = 1;

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) const noexcept
    {
        return data[i + j * leadingDimension];
    }
    // The rowCount × columnCount sub-matrix whose first element is (firstRow, firstColumn).
    [[nodiscard]] DenseMatrixView block(std::size_t firstRow, std::size_t firstColumn, std::size_t rowCount,
                                        std::size_t columnCount) const noexcept
    {
        return {data + firstRow + firstColumn * leadingDimension, rowCount, columnCount, leadingDimension};
    }
};

// The read-only counterpart of DenseMatrixView.
struct ConstDenseMatrixView
{
    const double* data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t leadingDimension = 1;

    ConstDenseMatrixView() = default;
    ConstDenseMatrixView(const double* values, std::size_t rowCount, std::size_t columnCount,
                         std::size_t leading) noexcept
        : data(values), rows(rowCount), columns(columnCount), leadingDimension(leading)
    {
    }
    // A writable view is readable (implicit on purpose, as T* converts to const T*).
    // NOLINTNEXTLINE(google-explicit-constructor)
    ConstDenseMatrixView(const DenseMatrixView& view) noexcept
        : data(view.data), rows(view.rows), columns(view.columns), leadingDimension(view.leadingDimension)
    {
    }

    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept
    {
        return data[i + j * leadingDimension];
    }
    // The address of element (i, j).
    [[nodiscard]] const double* pointer(std::size_t i, std::size_t j) const noexcept
    {
        return data + i + j * leadingDimension;
    }
    [[nodiscard]] ConstDenseMatrixView block(std::size_t firstRow, std::size_t firstColumn, std::size_t rowCount,
                                             std::size_t columnCount) const noexcept
    {
        return {data + firstRow + firstColumn * leadingDimension, rowCount, columnCount, leadingDimension};
    }
};

} // namespace oos::linalg
