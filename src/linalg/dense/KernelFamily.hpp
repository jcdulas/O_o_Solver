#pragma once

#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"

namespace oos::linalg
{

// The entry points of one dense kernel family (kernels/DenseKernelEntryPoints.hpp), so that the blocked algorithms
// and the tiling of DenseKernels are written once for every family. Internal to linalg/dense.
struct KernelFamily
{
    void (*subtractProductTile)(const ProductTileOperands&, const PackingWorkspace&) noexcept;
    void (*subtractColumnProducts)(const ColumnProductOperands&) noexcept;
    void (*solvePanel)(const PanelSolveOperands&) noexcept;
    void (*solveUnitLowerBlock)(const TriangleSolveOperands&) noexcept;
    void (*solveUnitLowerTransposedBlock)(const TriangleSolveOperands&) noexcept;
    void (*subtractRowProducts)(const RowProductOperands&) noexcept;
    void (*solveRowTriangle)(const RowTriangleOperands&) noexcept;
    // The products on panels packed once and shared by the tasks; null for the reference family.
    void (*packProductColumns)(const ProductTileOperands&, std::size_t, std::size_t, double*) noexcept;
    void (*subtractPackedProduct)(const ProductTileOperands&, std::size_t, std::size_t, const double*,
                                  double*) noexcept;
    // The rows packed alone and the product of panels packed beforehand (the front's tasks); null for the
    // reference family.
    void (*packProductRows)(const ProductTileOperands&, std::size_t, std::size_t, double*) noexcept;
    void (*subtractPrepackedProduct)(const ProductTileOperands&, std::size_t, const double*, const double*) noexcept;
    // False for the reference family, which needs no packing workspace and runs the unblocked factorisations.
    bool isBlocked;
};

// The entry points of `kind`. Taking them executes nothing of the family.
[[nodiscard]] const KernelFamily& kernelFamilyOf(KernelKind kind) noexcept;

} // namespace oos::linalg
