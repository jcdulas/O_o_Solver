#include "oos/reduction/condense/CraigBampton.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/eigen/BlockLanczos.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oos::reduction
{

namespace
{

// The interior DOFs with mass: the positive diagonal terms of M_oo (each column of a SymmetricCsc starts with its
// diagonal). For a lumped mass they are its rank, the number of finite roots of (K_oo, M_oo); a coupled mass has at
// most as many finite roots as its rank, never more than the DOFs.
std::size_t massiveDofCount(const linalg::SymmetricCsc& mass)
{
    const std::span<const std::int64_t> offsets = mass.columnOffsets();
    const std::span<const std::int32_t> rows = mass.rowIndices();
    const std::span<const double> values = mass.values();
    std::size_t count = 0;
    for (std::int32_t j = 0; j < mass.dimension(); ++j)
    {
        const auto first = static_cast<std::size_t>(offsets[static_cast<std::size_t>(j)]);
        if (first < static_cast<std::size_t>(offsets[static_cast<std::size_t>(j) + 1]) && rows[first] == j &&
            values[first] > 0.0)
            ++count;
    }
    return count;
}

// A ← (A + Aᵀ)/2, term by term in a fixed order: the projections are symmetric to round-off, made so to the bit.
void symmetrize(linalg::DenseMatrixView matrix)
{
    for (std::size_t j = 0; j < matrix.columns; ++j)
        for (std::size_t i = j + 1; i < matrix.rows; ++i)
        {
            const double mean = 0.5 * (matrix(i, j) + matrix(j, i));
            matrix(i, j) = mean;
            matrix(j, i) = mean;
        }
}

// Φ̃ = [Φ; 0] on the unit's equations (the interior equations lead).
linalg::DenseMatrix extendedModes(const UnitSystem& unit, const linalg::DenseMatrix& modes, base::MemoryBudget& budget)
{
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    linalg::DenseMatrix extended(budget, equations, modes.columns(), "Craig-Bampton extended modes");
    for (std::size_t j = 0; j < modes.columns(); ++j)
        for (std::size_t i = 0; i < modes.rows(); ++i)
            extended(i, j) = modes(i, j);
    return extended;
}

// Φᵀ·Y_o (N × c) for Y on the unit's equations, by the dense kernels (each term summed over the interior in
// increasing order): C ← 0 − Φᵀ·Y_o, then negated.
linalg::DenseMatrix modalProduct(const linalg::DenseMatrix& modes, linalg::ConstDenseMatrixView values,
                                 const linalg::DenseKernels& kernels, base::MemoryBudget& budget)
{
    const std::size_t interior = modes.rows();
    linalg::DenseMatrix product(budget, modes.columns(), values.columns, "Craig-Bampton modal product");
    if (interior > 0 && modes.columns() > 0 && values.columns > 0)
        kernels.subtractTransposedProduct(product.view(), modes.view(), values.block(0, 0, interior, values.columns));
    for (std::size_t j = 0; j < product.columns(); ++j)
        for (std::size_t i = 0; i < product.rows(); ++i)
            product(i, j) = -product(i, j);
    return product;
}

void requireMatrixOfUnit(const UnitSystem& unit, const linalg::SymmetricCsc& matrix, const char* operation)
{
    if (matrix.dimension() != unit.equationCount())
        throw std::invalid_argument(base::concatenate(
            {"CraigBampton::", operation, " (unit ", std::to_string(unit.unitId()), "): a matrix of ",
             std::to_string(matrix.dimension()), " equations for a unit of ", std::to_string(unit.equationCount()),
             "; give a matrix on the unit's equations (UnitSystem::matrix(), mass(), differentialStiffness())"}));
}

} // namespace

FixedInterfaceModes CraigBampton::computeModes(const UnitSystem& system, const ModeRequest& request,
                                               const linalg::DenseKernels& kernels,
                                               const linalg::FactorOptions& options, base::MemoryBudget& budget)
{
    const linalg::SymmetricCsc& mass = system.mass();
    const std::int32_t interior = system.interiorDofCount();
    const auto rows = static_cast<std::size_t>(interior);
    FixedInterfaceModes result;
    const auto none = [&]
    {
        result.modes = linalg::DenseMatrix(budget, rows, 0, "fixed-interface modes");
        result.eigenvalues = linalg::DenseMatrix(budget, 0, 1, "fixed-interface eigenvalues");
        return std::move(result);
    };
    const bool isAnyAsked = request.isCount() || request.cutoffEigenvalue > 0.0;
    if (interior == 0 || !isAnyAsked)
        return none();
    const linalg::SymmetricCsc stiffnessInterior =
        linalg::SymmetricCsc::leadingBlock(system.matrix(), interior, budget);
    const linalg::SymmetricCsc massInterior = linalg::SymmetricCsc::leadingBlock(mass, interior, budget);
    const std::size_t massive = massiveDofCount(massInterior);
    if (massive == 0)
        return none();

    // From the lowest root (K_oo is positive definite: no rigid-body mode, the Sturm count at the rigid shift is 0).
    linalg::EigenRequest eigenRequest;
    eigenRequest.lowerBound = -std::numeric_limits<double>::infinity();
    if (request.isCount())
        eigenRequest.modeCount = std::min(static_cast<std::size_t>(request.count), massive);
    else
        eigenRequest.upperBound = request.cutoffEigenvalue;
    const linalg::SymbolicFactor symbolic = system.interiorSymbolic(budget);
    linalg::SparsePencil pencil(symbolic, stiffnessInterior, massInterior, kernels, options, budget);
    linalg::LanczosOptions lanczos;
    lanczos.control.cancellation = options.cancellation;
    const linalg::EigenSolution solution = linalg::BlockLanczos(lanczos).solve(pencil, eigenRequest, kernels, budget);
    const std::size_t count = solution.eigenvalues.size();
    result.modes = linalg::DenseMatrix(budget, rows, count, "fixed-interface modes");
    result.eigenvalues = linalg::DenseMatrix(budget, count, 1, "fixed-interface eigenvalues");
    for (std::size_t j = 0; j < count; ++j)
    {
        result.eigenvalues(j, 0) = solution.eigenvalues[j];
        for (std::size_t i = 0; i < rows; ++i)
            result.modes(i, j) = solution.eigenvectors(i, j);
    }
    result.factorisationCount = solution.factorisationCount;
    result.isComplete = solution.isComplete;
    return result;
}

linalg::DenseMatrix CraigBampton::projectOnBoundary(const StaticCondensation& condensation,
                                                    const linalg::SymmetricCsc& matrix,
                                                    const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                                    const base::CancellationToken* cancellation)
{
    const UnitSystem& unit = condensation.system();
    requireMatrixOfUnit(unit, matrix, "projectOnBoundary");
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto boundary = static_cast<std::size_t>(unit.boundaryDofCount());
    linalg::DenseMatrix result(budget, boundary, boundary, "Craig-Bampton boundary projection");
    const std::size_t block = std::min(CONSTRAINT_MODE_BLOCK, std::max<std::size_t>(boundary, 1));
    linalg::DenseMatrix modes(budget, equations, block, "Craig-Bampton constraint modes");
    linalg::DenseMatrix values(budget, boundary, block, "Craig-Bampton constraint mode values");
    linalg::DenseMatrix products(budget, equations, block, "Craig-Bampton constraint mode products");
    for (std::size_t first = 0; first < boundary; first += block)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const std::size_t count = std::min(block, boundary - first);
        // Ψ̃ of boundary DOFs first, ..., first + count − 1: the recovery of the interior from a unit motion of each
        // with no interior load, u_o = −K_oo⁻¹·K_oa·e_a.
        const linalg::DenseMatrixView columns = modes.view().block(0, 0, equations, count);
        const linalg::DenseMatrixView unitValues = values.view().block(0, 0, boundary, count);
        for (std::size_t j = 0; j < count; ++j)
        {
            for (std::size_t i = 0; i < equations; ++i)
                columns(i, j) = 0.0;
            for (std::size_t i = 0; i < boundary; ++i)
                unitValues(i, j) = i == first + j ? 1.0 : 0.0;
        }
        condensation.recover(columns, unitValues, kernels, cancellation);
        const linalg::DenseMatrixView product = products.view().block(0, 0, equations, count);
        matrix.multiply(columns, product, kernels.options().pool, cancellation);
        // Ψ̃ᵀ·(A·Ψ̃): the condensation of the products as loads.
        const linalg::DenseMatrix condensed = condensation.condenseLoads(product, kernels, budget, cancellation);
        for (std::size_t j = 0; j < count; ++j)
            for (std::size_t i = 0; i < boundary; ++i)
                result(i, first + j) = condensed(i, j);
    }
    symmetrize(result.view());
    return result;
}

ModalReduction CraigBampton::reduceModes(const StaticCondensation& condensation, const linalg::DenseMatrix& modes,
                                         const linalg::DenseMatrix& modeEigenvalues,
                                         const linalg::DenseKernels& kernels, base::MemoryBudget& budget,
                                         const base::CancellationToken* cancellation)
{
    const UnitSystem& unit = condensation.system();
    const std::size_t count = modes.columns();
    const auto boundary = static_cast<std::size_t>(unit.boundaryDofCount());
    if (modes.rows() != static_cast<std::size_t>(unit.interiorDofCount()) || modeEigenvalues.rows() != count)
        throw std::invalid_argument(
            base::concatenate({"CraigBampton::reduceModes (unit ", std::to_string(unit.unitId()), "): modes of ",
                               std::to_string(modes.rows()), " rows and ", std::to_string(modeEigenvalues.rows()),
                               " eigenvalues for ", std::to_string(unit.interiorDofCount()), " interior equations and ",
                               std::to_string(count), " modes; give the unit's own modes"}));
    linalg::DenseMatrix eigenvalues(budget, count, 1, "Craig-Bampton eigenvalues");
    for (std::size_t k = 0; k < count; ++k)
        eigenvalues(k, 0) = modeEigenvalues(k, 0);
    if (count == 0)
        return ModalReduction(std::move(eigenvalues), linalg::DenseMatrix(budget, 0, 0, "Craig-Bampton K_qq"),
                              linalg::DenseMatrix(budget, boundary, 0, "Craig-Bampton M_aq"),
                              linalg::DenseMatrix(budget, 0, 0, "Craig-Bampton M_qq"));
    const linalg::DenseMatrix extended = extendedModes(unit, modes, budget);
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    base::ThreadPool* pool = kernels.options().pool;
    // M·Φ̃: its condensation is M_aq = Ψ̃ᵀ·M·Φ̃, its interior rows give M_qq = Φᵀ·M_oo·Φ.
    linalg::DenseMatrix massTimes(budget, equations, count, "Craig-Bampton M times modes");
    unit.mass().multiply(extended.view(), massTimes.view(), pool, cancellation);
    linalg::DenseMatrix coupling = condensation.condenseLoads(massTimes.view(), kernels, budget, cancellation);
    linalg::DenseMatrix modalMass = modalProduct(modes, massTimes.view(), kernels, budget);
    symmetrize(modalMass.view());
    // K·Φ̃: its interior rows give K_qq = Φᵀ·K_oo·Φ (K_aq is zero: Ψ̃ᵀ·K·Φ̃ = (K_ao − K_ao·K_oo⁻¹·K_oo)·Φ).
    linalg::DenseMatrix stiffnessTimes(budget, equations, count, "Craig-Bampton K times modes");
    unit.matrix().multiply(extended.view(), stiffnessTimes.view(), pool, cancellation);
    linalg::DenseMatrix modalStiffness = modalProduct(modes, stiffnessTimes.view(), kernels, budget);
    symmetrize(modalStiffness.view());
    return ModalReduction(std::move(eigenvalues), std::move(modalStiffness), std::move(coupling), std::move(modalMass));
}

linalg::DenseMatrix CraigBampton::project(const StaticCondensation& condensation, const linalg::SymmetricCsc& matrix,
                                          const linalg::DenseMatrix& modes, const linalg::DenseKernels& kernels,
                                          base::MemoryBudget& budget, const base::CancellationToken* cancellation)
{
    const UnitSystem& unit = condensation.system();
    requireMatrixOfUnit(unit, matrix, "project");
    const auto boundary = static_cast<std::size_t>(unit.boundaryDofCount());
    const std::size_t count = modes.columns();
    if (count > 0 && modes.rows() != static_cast<std::size_t>(unit.interiorDofCount()))
        throw std::invalid_argument(
            base::concatenate({"CraigBampton::project (unit ", std::to_string(unit.unitId()), "): modes of ",
                               std::to_string(modes.rows()), " rows for ", std::to_string(unit.interiorDofCount()),
                               " interior equations; give the unit's own modes"}));
    const linalg::DenseMatrix onBoundary = projectOnBoundary(condensation, matrix, kernels, budget, cancellation);
    linalg::DenseMatrix result(budget, boundary + count, boundary + count, "Craig-Bampton projection");
    for (std::size_t j = 0; j < boundary; ++j)
        for (std::size_t i = 0; i < boundary; ++i)
            result(i, j) = onBoundary(i, j);
    if (count == 0)
        return result;
    const linalg::DenseMatrix extended = extendedModes(unit, modes, budget);
    linalg::DenseMatrix times(budget, static_cast<std::size_t>(unit.equationCount()), count,
                              "Craig-Bampton matrix times modes");
    matrix.multiply(extended.view(), times.view(), kernels.options().pool, cancellation);
    const linalg::DenseMatrix coupling = condensation.condenseLoads(times.view(), kernels, budget, cancellation);
    const linalg::DenseMatrix modal = modalProduct(modes, times.view(), kernels, budget);
    for (std::size_t q = 0; q < count; ++q)
    {
        for (std::size_t i = 0; i < boundary; ++i)
        {
            result(i, boundary + q) = coupling(i, q);
            result(boundary + q, i) = coupling(i, q);
        }
        for (std::size_t p = 0; p < count; ++p)
            result(boundary + p, boundary + q) = modal(p, q);
    }
    symmetrize(result.view().block(boundary, boundary, count, count));
    return result;
}

void CraigBampton::recover(const StaticCondensation& condensation, const linalg::DenseMatrix& modes,
                           linalg::ConstDenseMatrixView boundaryValues, linalg::ConstDenseMatrixView modalValues,
                           linalg::DenseMatrixView equationValues, const linalg::DenseKernels& kernels,
                           const base::CancellationToken* cancellation)
{
    const UnitSystem& unit = condensation.system();
    const auto equations = static_cast<std::size_t>(unit.equationCount());
    const auto interior = static_cast<std::size_t>(unit.interiorDofCount());
    const std::size_t columns = equationValues.columns;
    const std::size_t count = modes.columns();
    if (equationValues.rows != equations || boundaryValues.columns != columns ||
        (count > 0 && (modalValues.rows != count || modalValues.columns != columns)) || modes.rows() != interior)
        throw std::invalid_argument(base::concatenate(
            {"CraigBampton::recover (unit ", std::to_string(unit.unitId()), "): ", std::to_string(equationValues.rows),
             " x ", std::to_string(columns), " equation values, ", std::to_string(modalValues.rows), " modal rows for ",
             std::to_string(equations), " equations and ", std::to_string(count),
             " modes; give as many columns everywhere"}));
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t i = 0; i < interior; ++i)
            equationValues(i, j) = 0.0;
    condensation.recover(equationValues, boundaryValues, kernels, cancellation);
    if (count == 0 || columns == 0)
        return;
    // u_o += Φ·q, by the kernels' C ← C − A·B with B = −q (each term summed over the modes in increasing order).
    std::vector<double> negated(count * columns); // bounded by the modes times the columns
    for (std::size_t j = 0; j < columns; ++j)
        for (std::size_t q = 0; q < count; ++q)
            negated[q + j * count] = -modalValues(q, j);
    kernels.subtractProduct(equationValues.block(0, 0, interior, columns), modes.view(),
                            linalg::ConstDenseMatrixView(negated.data(), count, columns, count));
}

} // namespace oos::reduction
