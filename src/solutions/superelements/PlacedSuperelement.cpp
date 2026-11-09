#include "oos/solutions/superelements/PlacedSuperelement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::solutions
{

namespace
{

// Two displacement systems are the same when every term of their rotations (CD to basic) agrees to this: the rotations
// are evaluated by O_o_Reader from the same CORD cards on both sides (a superelement and the residual structure share
// the coordinate systems a deck defines for both, an external superelement repeating its RectFrame2 cards), so they
// agree to the bit or differ by a rotation.
constexpr double SAME_SYSTEM_TOLERANCE = 1.0e-12;

base::Diagnostic placementProblem(std::int32_t superelementId, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = "SuperelementBulkOptions";
    diagnostic.id = superelementId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

PlacedSuperelement::PlacedSuperelement(std::int32_t superelementId, std::span<const reduction::UnitDofLabel> labels,
                                       std::span<const SuperelementBoundaryPoint> boundary,
                                       const assembly::DofMap& residual, const assembly::DofMap* superelement,
                                       base::DiagnosticSink& sink)
    : m_superelementId(superelementId)
{
    const auto residualPointOf = [&](base::EntityId nodeId) -> std::optional<base::EntityId>
    {
        const auto found = std::lower_bound(boundary.begin(), boundary.end(), nodeId,
                                            [](const SuperelementBoundaryPoint& point, base::EntityId id)
                                            { return point.nodeId < id; });
        if (found == boundary.end() || found->nodeId != nodeId)
            return std::nullopt;
        return found->residualNodeId;
    };
    std::vector<base::EntityId> checkedSystems; // the boundary grid points whose CD was checked
    for (const reduction::UnitDofLabel& label : labels)
    {
        const std::optional<base::EntityId> residualNodeId = residualPointOf(label.id);
        const bool isScalar = label.kind == reduction::UnitPointKind::Scalar;
        const std::optional<base::DofIndex> dof =
            residualNodeId ? residual.findDof(*residualNodeId, isScalar ? 0 : label.component) : std::nullopt;
        if (!dof)
        {
            // Joined to a point the residual lacks (or to a scalar point for a grid component), or not joined at all.
            const std::string found =
                residualNodeId ? base::concatenate({"is joined to point ", std::to_string(*residualNodeId),
                                                    ", which has no such DOF in the residual structure"})
                               : std::string(
                                     "is not joined to a DOF of the residual structure (SuperelementConnection, or the "
                                     "boundary search of SuperelementBulkOptions)");
            sink.report(placementProblem(
                superelementId, "superelement-boundary",
                base::concatenate({"superelement ", std::to_string(superelementId), ": its boundary DOF (point ",
                                   std::to_string(label.id), ", component ", std::to_string(label.component), ") ",
                                   found, "; join its point to a grid or scalar point of the residual structure"})));
            m_dofs.push_back(-1);
            continue;
        }
        m_dofs.push_back(*dof);
        // The same displacement system on both sides (the class comment).
        if (isScalar || superelement == nullptr ||
            std::find(checkedSystems.begin(), checkedSystems.end(), label.id) != checkedSystems.end())
            continue;
        checkedSystems.push_back(label.id);
        const std::optional<std::int32_t> ownRow = superelement->points().findGrid(label.id);
        const std::optional<std::int32_t> residualRow = residual.points().findGrid(*residualNodeId);
        if (!ownRow || !residualRow)
            continue;
        const base::Mat3 own = superelement->cdToBasic(*ownRow);
        const base::Mat3 theirs = residual.cdToBasic(*residualRow);
        double difference = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                difference = std::max(difference, std::abs(own(i, j) - theirs(i, j)));
        if (difference > SAME_SYSTEM_TOLERANCE)
            sink.report(placementProblem(
                superelementId, "superelement-displacement-system",
                base::concatenate({"superelement ", std::to_string(superelementId), ": boundary point ",
                                   std::to_string(label.id),
                                   " has another displacement system (Node.displacementFrameId) than residual point ",
                                   std::to_string(*residualNodeId),
                                   ": this version joins boundary points in the same system only (no "
                                   "SuperelementLocation, SuperelementMirrorPlane or "
                                   "differing Node.displacementFrameId); give both points the same "
                                   "Node.displacementFrameId"})));
    }
    for (const base::DofIndex dof : m_dofs)
        if (dof >= 0)
            m_nodes.push_back(residual.nodeDofOf(dof).node);
    std::sort(m_nodes.begin(), m_nodes.end());
    m_nodes.erase(std::unique(m_nodes.begin(), m_nodes.end()), m_nodes.end());
}

void PlacedSuperelement::addMatrixTo(const linalg::DenseMatrix& matrix, linalg::BlockSparseMatrix& target,
                                     const assembly::DofMap& residual) const
{
    const linalg::SparsePattern& pattern = target.pattern();
    const std::size_t n = m_dofs.size();
    for (std::size_t b = 0; b < n; ++b)
    {
        if (m_dofs[b] < 0)
            continue;
        const linalg::NodeDof column = residual.nodeDofOf(m_dofs[b]);
        for (std::size_t a = 0; a < n; ++a)
        {
            if (m_dofs[a] < 0)
                continue;
            const linalg::NodeDof row = residual.nodeDofOf(m_dofs[a]);
            // The pattern holds the lower blocks (row node >= column node), both triangles of a diagonal block.
            if (row.node < column.node)
                continue;
            const std::int64_t block = pattern.findBlock(row.node, column.node);
            if (block < 0)
                throw std::logic_error(base::concatenate(
                    {"PlacedSuperelement (superelement ", std::to_string(m_superelementId),
                     "): the residual pattern does not couple nodes ", std::to_string(row.node), " and ",
                     std::to_string(column.node), "; build the node graph with nodes() as one group"}));
            const std::span<double> values = target.blockValues(block);
            const auto rowDofs = static_cast<std::size_t>(pattern.nodeDofCount(row.node));
            values[static_cast<std::size_t>(column.component) * rowDofs + static_cast<std::size_t>(row.component)] +=
                matrix(a, b);
        }
    }
}

void PlacedSuperelement::addStiffnessTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& residual) const
{
    addMatrixTo(m_stiffness, target, residual);
}

void PlacedSuperelement::addMassTo(linalg::BlockSparseMatrix& target, const assembly::DofMap& residual) const
{
    if (m_mass)
        addMatrixTo(*m_mass, target, residual);
}

void PlacedSuperelement::addLoadsTo(std::size_t column, std::span<double> loadsG) const
{
    if (column >= m_loads.columns())
        return;
    for (std::size_t a = 0; a < m_dofs.size(); ++a)
        if (m_dofs[a] >= 0)
            loadsG[static_cast<std::size_t>(m_dofs[a])] += m_loads(a, column);
}

linalg::DenseMatrix PlacedSuperelement::gather(linalg::ConstDenseMatrixView valuesG, base::MemoryBudget& budget) const
{
    linalg::DenseMatrix values(budget, m_dofs.size(), valuesG.columns, "superelement boundary values");
    for (std::size_t c = 0; c < valuesG.columns; ++c)
        for (std::size_t a = 0; a < m_dofs.size(); ++a)
            values(a, c) = m_dofs[a] >= 0 ? valuesG(static_cast<std::size_t>(m_dofs[a]), c) : 0.0;
    return values;
}

} // namespace oos::solutions
