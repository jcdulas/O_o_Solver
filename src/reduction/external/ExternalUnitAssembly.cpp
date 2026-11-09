#include "oos/reduction/external/ExternalUnitAssembly.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

// ½·(A + Aᵀ) of a unit matrix with its rows and columns taken in the contour order (rows[c]: the unit row of contour
// DOF c).
linalg::DenseMatrix placed(const linalg::DenseMatrix& a, std::span<const std::size_t> rows, base::MemoryBudget& budget,
                           const char* label)
{
    const std::size_t n = rows.size();
    linalg::DenseMatrix result(budget, n, n, label);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            result(i, j) = 0.5 * (a(rows[i], rows[j]) + a(rows[j], rows[i]));
    return result;
}

} // namespace

ExternalUnitAssembly::ExternalUnitAssembly(std::shared_ptr<const ExternalUnit> unit, ExternalZone zone,
                                           base::MemoryBudget& budget)
    : m_unit(std::move(unit)), m_zone(std::move(zone))
{
    if (!m_unit)
        throw std::invalid_argument("ExternalUnitAssembly: expected an external unit, found none");
    const std::size_t n = m_zone.contourLabels.size();
    if (n != m_unit->size() || m_zone.contourDofs.size() != n ||
        m_zone.interiorDofs.size() != m_zone.interiorLabels.size())
        throw std::invalid_argument(base::concatenate(
            {"ExternalUnitAssembly (", m_zone.displayName(), "): ", std::to_string(n), " contour labels, ",
             std::to_string(m_zone.contourDofs.size()), " contour DOFs and ", std::to_string(m_unit->size()),
             " unit labels; ExternalUnitChecks accepts a unit with exactly the contour DOFs"}));
    for (const UnitDofLabel& label : m_zone.contourLabels)
    {
        const std::optional<std::size_t> row = m_unit->rowOf(label);
        if (!row)
            throw std::invalid_argument(
                base::concatenate({"ExternalUnitAssembly (", m_zone.displayName(), "): the contour DOF (grid ",
                                   std::to_string(label.id), ", component ", std::to_string(label.component),
                                   ") is not a label of the unit; run ExternalUnitChecks first"}));
        m_unitRow.push_back(*row);
    }
    m_stiffness = placed(m_unit->stiffness(), m_unitRow, budget, "external unit K*");
    if (m_unit->hasMass())
        m_mass = placed(m_unit->mass(), m_unitRow, budget, "external unit M*");
    for (const ExternalUnitDerivative& derivative : m_unit->derivatives())
    {
        m_derivatives.push_back(placed(derivative.stiffness, m_unitRow, budget, "external unit dK*/dx"));
        if (derivative.mass)
            m_massDerivatives.emplace_back(placed(*derivative.mass, m_unitRow, budget, "external unit dM*/dx"));
        else
            m_massDerivatives.emplace_back(std::nullopt);
    }
}

std::vector<std::int32_t> ExternalUnitAssembly::contourNodes(const assembly::DofMap& dofMap) const
{
    std::vector<std::int32_t> nodes; // bounded by the contour
    for (const base::DofIndex dof : m_zone.contourDofs)
        nodes.push_back(dofMap.nodeDofOf(dof).node);
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

void ExternalUnitAssembly::addStiffnessTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& dofMap) const
{
    addMatrixTo(m_stiffness, target, dofMap);
}

void ExternalUnitAssembly::addMassTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& dofMap) const
{
    if (m_mass)
        addMatrixTo(*m_mass, target, dofMap);
}

void ExternalUnitAssembly::addMatrixTo(const linalg::DenseMatrix& matrix, linalg::BlockSparseMatrix& target,
                                       const assembly::DofMap& dofMap) const
{
    // The pattern holds the lower node blocks (row node ≥ column node), each column-major; a diagonal block is held
    // whole (linalg::SparsePattern). Every term (a, b) whose row node is at least its column node is added once, in
    // the contour order: the same sums whatever the threads.
    const linalg::SparsePattern& pattern = target.pattern();
    const std::size_t n = m_zone.contourDofs.size();
    for (std::size_t b = 0; b < n; ++b)
    {
        const linalg::NodeDof column = dofMap.nodeDofOf(m_zone.contourDofs[b]);
        for (std::size_t a = 0; a < n; ++a)
        {
            const linalg::NodeDof row = dofMap.nodeDofOf(m_zone.contourDofs[a]);
            if (row.node < column.node)
                continue;
            const std::int64_t block = pattern.findBlock(row.node, column.node);
            if (block < 0)
                throw std::logic_error(base::concatenate(
                    {"ExternalUnitAssembly (", m_zone.displayName(), "): the pattern does not couple nodes ",
                     std::to_string(row.node), " and ", std::to_string(column.node),
                     " of the contour; build the node graph with contourNodes() as one group"}));
            const std::span<double> values = target.blockValues(block);
            const auto rowDofs = static_cast<std::size_t>(pattern.nodeDofCount(row.node));
            values[static_cast<std::size_t>(column.component) * rowDofs + static_cast<std::size_t>(row.component)] +=
                matrix(a, b);
        }
    }
}

void ExternalUnitAssembly::gatherContour(std::span<const double> valuesG, std::span<double> contour) const
{
    for (std::size_t a = 0; a < m_zone.contourDofs.size(); ++a)
        contour[a] = valuesG[static_cast<std::size_t>(m_zone.contourDofs[a])];
}

void ExternalUnitAssembly::gatherInterior(std::span<const double> valuesG, std::span<double> interior) const
{
    for (std::size_t i = 0; i < m_zone.interiorDofs.size(); ++i)
        interior[i] = valuesG[static_cast<std::size_t>(m_zone.interiorDofs[i])];
}

void ExternalUnitAssembly::multiply(const linalg::DenseMatrix& a, std::span<const double> contour,
                                    std::span<double> product)
{
    const std::size_t n = a.rows();
    std::fill(product.begin(), product.begin() + static_cast<std::ptrdiff_t>(n), 0.0);
    for (std::size_t j = 0; j < n; ++j)
    {
        const double value = contour[j];
        if (value == 0.0)
            continue;
        for (std::size_t i = 0; i < n; ++i)
            product[i] += a(i, j) * value;
    }
}

std::array<double, 6> ExternalUnitAssembly::resultantOf(std::span<const double> contour,
                                                        const assembly::DofMap& dofMap) const
{
    // The motion k of a DOF about the basic origin: a grid point at x moves by t + θ × x and turns by θ, in its CD
    // (assembly::RigidBodyModes); the resultant is Σ_a r_k(a)·v_a.
    std::array<double, 6> resultant{};
    for (std::size_t a = 0; a < m_zone.contourDofs.size(); ++a)
    {
        const assembly::DofLabel label = dofMap.labelOf(m_zone.contourDofs[a]);
        const base::Vec3 position = dofMap.positionInBasic(label.row);
        const base::Mat3 toBasic = dofMap.cdToBasic(label.row);
        // The direction of component c in basic is column c of R (CD → basic).
        const auto c = static_cast<std::size_t>((label.component - 1) % 3);
        const base::Vec3 direction{toBasic(0, c), toBasic(1, c), toBasic(2, c)};
        if (label.component <= 3)
        {
            // A force along `direction` at x: its force, and its moment x × F about the origin.
            const base::Vec3 moment = base::cross(position, direction);
            for (std::size_t k = 0; k < 3; ++k)
            {
                resultant[k] += direction[k] * contour[a];
                resultant[3 + k] += moment[k] * contour[a];
            }
        }
        else
        {
            for (std::size_t k = 0; k < 3; ++k)
                resultant[3 + k] += direction[k] * contour[a];
        }
    }
    return resultant;
}

} // namespace oos::reduction
