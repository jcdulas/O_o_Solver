#include "oos/assembly/relief/RigidBodyModes.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <stdexcept>
#include <string>

namespace oos::assembly
{

RigidBodyModes::RigidBodyModes(const DofMap& dofMap, base::Vec3 referencePoint, base::MemoryBudget& budget)
    : m_reference(referencePoint),
      m_modes(budget, static_cast<std::size_t>(dofMap.gSetSize()), MODE_COUNT, "rigid-body modes")
{
    for (std::int32_t grid = 0; grid < dofMap.gridCount(); ++grid)
    {
        const base::Vec3 r = dofMap.positionInBasic(grid) - referencePoint;
        const base::Mat3 toCd = dofMap.cdToBasic(grid).transposed();
        for (std::size_t mode = 0; mode < MODE_COUNT; ++mode)
        {
            base::Vec3 translation{};
            base::Vec3 rotation{};
            if (mode < 3)
                translation[mode] = 1.0;
            else
            {
                rotation[mode - 3] = 1.0;
                translation = base::cross(rotation, r); // u = θ × r
            }
            const base::Vec3 u = toCd * translation;
            const base::Vec3 theta = toCd * rotation;
            for (int k = 0; k < 3; ++k)
            {
                m_modes(static_cast<std::size_t>(dofMap.gridDof(grid, k + 1)), mode) = u[static_cast<std::size_t>(k)];
                m_modes(static_cast<std::size_t>(dofMap.gridDof(grid, k + 4)), mode) =
                    theta[static_cast<std::size_t>(k)];
            }
        }
    }
}

std::array<double, RigidBodyModes::MODE_COUNT> RigidBodyModes::resultantOf(std::span<const double> valuesG) const
{
    if (valuesG.size() != m_modes.rows())
        throw std::invalid_argument(
            base::concatenate({"RigidBodyModes::resultantOf: expected ", std::to_string(m_modes.rows()),
                               " values (the g set), found ", std::to_string(valuesG.size())}));
    std::array<double, MODE_COUNT> resultant{};
    for (std::size_t mode = 0; mode < MODE_COUNT; ++mode)
    {
        double sum = 0.0;
        for (std::size_t dof = 0; dof < valuesG.size(); ++dof)
            sum += m_modes(dof, mode) * valuesG[dof];
        resultant[mode] = sum;
    }
    return resultant;
}

base::MatN<6> RigidBodyModes::rigidBodyMass(const linalg::BlockSparseMatrix& mass, base::MemoryBudget& budget) const
{
    const std::size_t size = m_modes.rows();
    base::BudgetedBuffer<double> column(budget, size, "rigid-body mode");
    base::BudgetedBuffer<double> product(budget, size, "mass times rigid-body mode");
    base::MatN<6> result;
    for (std::size_t j = 0; j < MODE_COUNT; ++j)
    {
        for (std::size_t dof = 0; dof < size; ++dof)
            column[dof] = m_modes(dof, j);
        mass.multiply(column.span(), product.span());
        for (std::size_t i = 0; i < MODE_COUNT; ++i)
        {
            double sum = 0.0;
            for (std::size_t dof = 0; dof < size; ++dof)
                sum += m_modes(dof, i) * product[dof];
            result(i, j) = sum;
        }
    }
    // Exactly symmetric: the two triangles are rounded differently.
    for (std::size_t i = 0; i < MODE_COUNT; ++i)
        for (std::size_t j = i + 1; j < MODE_COUNT; ++j)
            result(i, j) = result(j, i) = 0.5 * (result(i, j) + result(j, i));
    return result;
}

} // namespace oos::assembly
