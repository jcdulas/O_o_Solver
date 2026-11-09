#include "oos/reduction/absorption/SpiderWeights.hpp"

#include "oos/base/math/Vec3.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace oos::reduction
{

namespace
{

// A spider whose legs all sit on its reference point has no length to scale rotations by: 1 (the RigidInterpolation's
// Lc rule).
constexpr double UNIT_LENGTH = 1.0;

} // namespace

SpiderDofKind dofKindOf(const assembly::DofMap& dofMap, base::DofIndex dof) noexcept
{
    const assembly::DofLabel label = dofMap.labelOf(dof);
    return label.kind == input::PointKind::Grid && label.component >= 4 ? SpiderDofKind::Rotation
                                                                        : SpiderDofKind::Translation;
}

SpiderWeights::SpiderWeights(const assembly::RigidElimination& gm, const assembly::DofMap& dofMap,
                             base::MemoryBudget& budget)
    : m_gm(&gm), m_dofMap(&dofMap)
{
    m_index = base::BudgetedBuffer<RowKey>(budget, gm.rowCount(), "spider relations: rows of Gm by element");
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
    {
        if (gm.sourceOf(i) == assembly::ConstraintSource::Mpc)
            continue;
        m_index[m_indexSize++] =
            RowKey{static_cast<std::uint8_t>(gm.sourceOf(i)), gm.sourceIdOf(i), static_cast<std::int64_t>(i)};
    }
    const auto key = [](const RowKey& entry) { return std::make_tuple(entry.source, entry.elementId, entry.row); };
    std::sort(m_index.data(), m_index.data() + m_indexSize,
              [&](const RowKey& a, const RowKey& b) { return key(a) < key(b); });
}

std::vector<std::size_t> SpiderWeights::rowsOf(base::EntityId elementId, assembly::ConstraintSource source) const
{
    const auto code = static_cast<std::uint8_t>(source);
    const RowKey* first =
        std::lower_bound(m_index.data(), m_index.data() + m_indexSize, std::make_pair(code, elementId),
                         [](const RowKey& entry, const std::pair<std::uint8_t, base::EntityId>& k)
                         { return std::make_pair(entry.source, entry.elementId) < k; });
    const RowKey* const end = m_index.data() + m_indexSize;
    std::vector<std::size_t> rows; // bounded by one element's dependent DOFs
    for (const RowKey* entry = first; entry != end && entry->source == code && entry->elementId == elementId; ++entry)
        rows.push_back(static_cast<std::size_t>(entry->row));
    return rows;
}

std::optional<SpiderRelation> SpiderWeights::rigidInterpolationRelation(base::EntityId elementId) const
{
    const std::vector<std::size_t> rows = rowsOf(elementId, assembly::ConstraintSource::RigidInterpolation);
    if (rows.empty())
        return std::nullopt;
    return relationOfRows(rows);
}

std::optional<SpiderRelation> SpiderWeights::fastenerPointRelation(base::EntityId elementId,
                                                                   assembly::ConstraintSource source,
                                                                   base::EntityId pointNodeId) const
{
    std::vector<std::size_t> rows = rowsOf(elementId, source);
    std::erase_if(rows, [&](std::size_t i) { return m_dofMap->pointIdOf(m_gm->mDofs()[i]) != pointNodeId; });
    if (rows.empty())
        return std::nullopt;
    return relationOfRows(rows);
}

SpiderRelation SpiderWeights::relationOfRows(const std::vector<std::size_t>& rows) const
{
    SpiderRelation relation;
    for (const std::size_t i : rows)
    {
        relation.referenceDofs.push_back(m_gm->mDofs()[i]);
        for (const assembly::ConstraintTerm& term : m_gm->row(i))
            relation.legDofs.push_back(term.dof);
    }
    std::sort(relation.legDofs.begin(), relation.legDofs.end());
    relation.legDofs.erase(std::unique(relation.legDofs.begin(), relation.legDofs.end()), relation.legDofs.end());

    SpiderWeightMatrix& weights = relation.weights;
    weights.referenceDofCount = rows.size();
    weights.legDofCount = relation.legDofs.size();
    weights.values.assign(weights.referenceDofCount * weights.legDofCount, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (const assembly::ConstraintTerm& term : m_gm->row(rows[r]))
        {
            const auto column =
                static_cast<std::size_t>(std::lower_bound(relation.legDofs.begin(), relation.legDofs.end(), term.dof) -
                                         relation.legDofs.begin());
            weights.values[r * weights.legDofCount + column] += term.coefficient;
        }
    for (const base::DofIndex dof : relation.referenceDofs)
        weights.referenceKinds.push_back(dofKindOf(*m_dofMap, dof));
    for (const base::DofIndex dof : relation.legDofs)
        weights.legKinds.push_back(dofKindOf(*m_dofMap, dof));

    // Lc: the mean distance of the leg grid points to the reference grid point (the RigidInterpolation rule).
    const assembly::DofLabel reference = m_dofMap->labelOf(relation.referenceDofs.front());
    if (reference.kind == input::PointKind::Grid)
    {
        const base::Vec3 origin = m_dofMap->positionInBasic(reference.row);
        double total = 0.0;
        std::int64_t count = 0;
        std::int32_t previousRow = -1;
        for (const base::DofIndex dof : relation.legDofs)
        {
            const assembly::DofLabel leg = m_dofMap->labelOf(dof);
            if (leg.kind != input::PointKind::Grid || leg.row == previousRow)
                continue;
            previousRow = leg.row;
            const base::Vec3 arm = m_dofMap->positionInBasic(leg.row) - origin;
            const double length = std::sqrt(arm[0] * arm[0] + arm[1] * arm[1] + arm[2] * arm[2]);
            // A leg on the reference point within the model's coincidence distance has no arm (RigidElementEquations'
            // RigidInterpolation).
            total += length <= m_dofMap->coincidenceDistance() ? 0.0 : length;
            ++count;
        }
        weights.characteristicLength = count > 0 && total > 0.0 ? total / static_cast<double>(count) : UNIT_LENGTH;
    }
    return relation;
}

std::vector<base::DofIndex> SpiderWeights::dependentDofs(base::EntityId elementId,
                                                         assembly::ConstraintSource source) const
{
    std::vector<base::DofIndex> dofs;
    for (const std::size_t i : rowsOf(elementId, source))
        dofs.push_back(m_gm->mDofs()[i]);
    std::sort(dofs.begin(), dofs.end());
    return dofs;
}

} // namespace oos::reduction
