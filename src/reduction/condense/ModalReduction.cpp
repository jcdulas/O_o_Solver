#include "oos/reduction/condense/ModalReduction.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

ModalReduction::ModalReduction(linalg::DenseMatrix eigenvalues, linalg::DenseMatrix modalStiffness,
                               linalg::DenseMatrix couplingMass, linalg::DenseMatrix modalMass)
    : m_eigenvalues(std::move(eigenvalues)),
      m_modalStiffness(std::move(modalStiffness)),
      m_couplingMass(std::move(couplingMass)),
      m_modalMass(std::move(modalMass))
{
    const std::size_t modes = m_eigenvalues.rows();
    const bool isShaped = (m_eigenvalues.columns() == 1 || modes == 0) && m_modalStiffness.rows() == modes &&
                          m_modalStiffness.columns() == modes && m_modalMass.rows() == modes &&
                          m_modalMass.columns() == modes && m_couplingMass.columns() == modes;
    if (!isShaped)
        throw std::invalid_argument(base::concatenate(
            {"ModalReduction: ", std::to_string(modes), " eigenvalues with a modal stiffness of ",
             std::to_string(m_modalStiffness.rows()), " x ", std::to_string(m_modalStiffness.columns()),
             ", a coupling mass of ", std::to_string(m_couplingMass.columns()), " columns and a modal mass of ",
             std::to_string(m_modalMass.rows()), " x ", std::to_string(m_modalMass.columns()),
             "; give N eigenvalues, N x N matrices and N coupling columns"}));
}

std::uint64_t ModalReduction::byteCount() const noexcept
{
    const auto values = [](const linalg::DenseMatrix& matrix)
    { return static_cast<std::uint64_t>(matrix.rows() * matrix.columns()); };
    return (values(m_eigenvalues) + values(m_modalStiffness) + values(m_couplingMass) + values(m_modalMass)) *
           sizeof(double);
}

} // namespace oos::reduction
