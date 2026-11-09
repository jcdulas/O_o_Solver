#include "oos/reduction/condense/ReducedUnit.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

ReducedUnit::ReducedUnit(std::int64_t unitId, base::BudgetedBuffer<UnitDofLabel> labels, linalg::DenseMatrix stiffness,
                         linalg::DenseMatrix patterns, std::int64_t patternStamp, linalg::DenseMatrix inertialPatterns,
                         const base::MatN<6>& rigidBodyMass)
    : m_unitId(unitId),
      m_labels(std::move(labels)),
      m_stiffness(std::move(stiffness)),
      m_patterns(std::move(patterns)),
      m_patternStamp(patternStamp),
      m_inertialPatterns(std::move(inertialPatterns)),
      m_rigidBodyMass(rigidBodyMass)
{
    const std::size_t n = m_labels.size();
    if (m_stiffness.rows() != n || m_stiffness.columns() != n || m_patterns.rows() != n ||
        m_inertialPatterns.rows() != n)
        throw std::invalid_argument(base::concatenate(
            {"ReducedUnit (unit ", std::to_string(unitId), "): ", std::to_string(n), " labels for a ",
             std::to_string(m_stiffness.rows()), " x ", std::to_string(m_stiffness.columns()), " K*, ",
             std::to_string(m_patterns.rows()), " pattern rows and ", std::to_string(m_inertialPatterns.rows()),
             " inertial rows; give every matrix one row per boundary DOF"}));
}

std::uint64_t ReducedUnit::byteCount() const noexcept
{
    const auto values = static_cast<std::uint64_t>(m_stiffness.rows() * m_stiffness.columns() +
                                                   m_patterns.rows() * m_patterns.columns() +
                                                   m_inertialPatterns.rows() * m_inertialPatterns.columns());
    return values * sizeof(double) + static_cast<std::uint64_t>(m_labels.size() * sizeof(UnitDofLabel)) +
           static_cast<std::uint64_t>(sizeof(m_rigidBodyMass));
}

void ReducedUnit::boundaryForces(linalg::ConstDenseMatrixView boundaryDisplacements,
                                 linalg::ConstDenseMatrixView coefficients, linalg::DenseMatrixView forces,
                                 const linalg::DenseKernels& kernels) const
{
    const std::size_t n = boundaryDofCount();
    if (boundaryDisplacements.rows != n || coefficients.rows != m_patterns.columns() || forces.rows != n ||
        boundaryDisplacements.columns != forces.columns || coefficients.columns != forces.columns)
        throw std::invalid_argument(base::concatenate(
            {"ReducedUnit::boundaryForces (unit ", std::to_string(m_unitId), "): displacements ",
             std::to_string(boundaryDisplacements.rows), " x ", std::to_string(boundaryDisplacements.columns),
             ", coefficients ", std::to_string(coefficients.rows), " x ", std::to_string(coefficients.columns),
             ", forces ", std::to_string(forces.rows), " x ", std::to_string(forces.columns), "; expected ",
             std::to_string(n), " boundary rows, ", std::to_string(m_patterns.columns()),
             " pattern rows and one column per case"}));
    // 0 − f*·α, negated to f*·α, less K*·u_a, negated to K*·u_a − f*·α: every element one chain of terms in a fixed
    // order (DenseKernels::subtractProduct), the sign changes exact.
    for (std::size_t j = 0; j < forces.columns; ++j)
        for (std::size_t i = 0; i < n; ++i)
            forces(i, j) = 0.0;
    if (n == 0 || forces.columns == 0)
        return;
    if (m_patterns.columns() > 0)
        kernels.subtractProduct(forces, m_patterns.view(), coefficients);
    for (std::size_t j = 0; j < forces.columns; ++j)
        for (std::size_t i = 0; i < n; ++i)
            forces(i, j) = -forces(i, j);                                       // f*·α
    kernels.subtractProduct(forces, m_stiffness.view(), boundaryDisplacements); // f*·α − K*·u_a
    for (std::size_t j = 0; j < forces.columns; ++j)
        for (std::size_t i = 0; i < n; ++i)
            forces(i, j) = -forces(i, j); // K*·u_a − f*·α
}

} // namespace oos::reduction
