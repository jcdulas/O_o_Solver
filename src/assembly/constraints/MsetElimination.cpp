#include "oos/assembly/constraints/MsetElimination.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::assembly
{

namespace
{

// The terms of a symmetric matrix on a block pattern read and written as those of the full matrix: (i, j) and (j, i)
// are the same stored value when the nodes differ (the block of the larger node row holds it), and two values of the
// complete diagonal block when they are one node.
class FullMatrixAccess
{
public:
    FullMatrixAccess(const linalg::SparsePattern& pattern, const DofMap& dofMap)
        : m_pattern(&pattern), m_dofMap(&dofMap)
    {
    }

    // The position of term (i, j) in the values, or -1 when the nodes are not coupled; `isStored` is false for a term
    // of the upper triangle of two different nodes (its value is that of (j, i)).
    [[nodiscard]] std::int64_t positionOf(base::DofIndex i, base::DofIndex j, bool& isStored) const
    {
        const linalg::NodeDof a = m_dofMap->nodeDofOf(i);
        const linalg::NodeDof b = m_dofMap->nodeDofOf(j);
        isStored = a.node >= b.node;
        const linalg::NodeDof row = isStored ? a : b;
        const linalg::NodeDof column = isStored ? b : a;
        const std::int64_t block = m_pattern->findBlock(row.node, column.node);
        if (block < 0)
            return -1;
        return m_pattern->blockValueOffset(block) + row.component +
               static_cast<std::int64_t>(m_pattern->nodeDofCount(row.node)) * column.component;
    }

    [[nodiscard]] double read(std::span<const double> values, base::DofIndex i, base::DofIndex j) const
    {
        bool isStored = false;
        const std::int64_t position = positionOf(i, j, isStored);
        return position < 0 ? 0.0 : values[static_cast<std::size_t>(position)];
    }

    // Adds `value` to term (i, j) of the full matrix: the caller adds (j, i) too, so a term of the upper triangle of
    // two nodes is skipped (its partner writes the stored value).
    void add(std::span<double> values, base::DofIndex i, base::DofIndex j, double value) const
    {
        bool isStored = false;
        const std::int64_t position = positionOf(i, j, isStored);
        if (position < 0)
            throw std::invalid_argument(base::concatenate(
                {"MsetElimination: the term (", std::to_string(i), ", ", std::to_string(j),
                 ") of Knn is outside the pattern of the stiffness; build the node graph with MsetCouplings"}));
        if (isStored)
            values[static_cast<std::size_t>(position)] += value;
    }

private:
    const linalg::SparsePattern* m_pattern;
    const DofMap* m_dofMap;
};

// The g DOFs of a node.
void appendNodeDofs(const DofMap& dofMap, std::int32_t node, std::vector<base::DofIndex>& dofs)
{
    if (node < dofMap.gridCount())
        for (int c = 1; c <= 6; ++c)
            dofs.push_back(dofMap.gridDof(node, c));
    else
        dofs.push_back(dofMap.scalarDof(node - dofMap.gridCount()));
}

} // namespace

void MsetElimination::reduce(const linalg::BlockSparseMatrix& kgg, const linalg::NodeGraph& graph,
                             const RigidElimination& gm, const DofMap& dofMap, linalg::BlockSparseMatrix& knn)
{
    if (&kgg.pattern() != &knn.pattern())
        throw std::invalid_argument("MsetElimination: Kgg and Knn must share one pattern; make Knn on Kgg's pattern");
    const std::span<const double> k = kgg.values();
    const std::span<double> out = knn.values();
    std::copy(k.begin(), k.end(), out.begin());
    const FullMatrixAccess access(kgg.pattern(), dofMap);

    std::vector<base::DofIndex> coupled; // the DOFs of the nodes coupled to one node: bounded by a node's degree
    for (std::size_t r = 0; r < gm.rowCount(); ++r)
    {
        const base::DofIndex d = gm.mDofs()[r];
        const std::span<const ConstraintTerm> rowD = gm.row(r);
        const std::int32_t nodeD = dofMap.nodeDofOf(d).node;
        coupled.clear();
        appendNodeDofs(dofMap, nodeD, coupled);
        for (const std::int32_t neighbour : graph.neighbours(nodeD))
            appendNodeDofs(dofMap, neighbour, coupled);
        std::sort(coupled.begin(), coupled.end());
        for (const base::DofIndex i : coupled)
        {
            const double kid = access.read(k, i, d);
            if (kid == 0.0)
                continue;
            if (const std::optional<std::size_t> rowE = gm.findRow(i))
            {
                // Gmᵀ·K̄mm·Gm: d and e = i both dependent.
                for (const ConstraintTerm& termJ : rowD)
                    for (const ConstraintTerm& termK : gm.row(*rowE))
                        access.add(out, termJ.dof, termK.dof, termJ.coefficient * kid * termK.coefficient);
                continue;
            }
            // K̄nm·Gm and its transpose: i independent.
            for (const ConstraintTerm& term : rowD)
            {
                const double value = kid * term.coefficient;
                access.add(out, i, term.dof, value);
                access.add(out, term.dof, i, value);
            }
        }
    }
}

} // namespace oos::assembly
