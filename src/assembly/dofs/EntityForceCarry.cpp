#include "oos/assembly/dofs/EntityForceCarry.hpp"

#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/FormulationModel.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oos::assembly
{

namespace
{

constexpr std::size_t MOTIONS = RigidBodyModes::MODE_COUNT;

// Solves the symmetric positive definite 6 × 6 system a·x = b by Gaussian elimination with partial pivoting (a and b
// are overwritten). A pivot that vanishes (an entity whose grid points cannot carry a resultant: none here, the
// moments at the grid points carry any) leaves its unknown at zero.
std::array<double, MOTIONS> solve(std::array<std::array<double, MOTIONS>, MOTIONS>& a, std::array<double, MOTIONS>& b)
{
    for (std::size_t k = 0; k < MOTIONS; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < MOTIONS; ++i)
            if (std::fabs(a[i][k]) > std::fabs(a[pivot][k]))
                pivot = i;
        std::swap(a[k], a[pivot]);
        std::swap(b[k], b[pivot]);
        if (a[k][k] == 0.0)
            continue;
        for (std::size_t i = k + 1; i < MOTIONS; ++i)
        {
            const double factor = a[i][k] / a[k][k];
            for (std::size_t j = k; j < MOTIONS; ++j)
                a[i][j] -= factor * a[k][j];
            b[i] -= factor * b[k];
        }
    }
    std::array<double, MOTIONS> x{};
    for (std::size_t k = MOTIONS; k-- > 0;)
    {
        if (a[k][k] == 0.0)
            continue;
        double sum = b[k];
        for (std::size_t j = k + 1; j < MOTIONS; ++j)
            sum -= a[k][j] * x[j];
        x[k] = sum / a[k][k];
    }
    return x;
}

} // namespace

EntityForceCarry::EntityForceCarry(const DofMap& dofMap) : m_dofMap(&dofMap)
{
    if (!dofMap.hasEntities())
        return;
    m_modes = entityRigidModes(dofMap);
    m_entityDofCount = static_cast<std::size_t>(dofMap.gSetSize() - dofMap.firstEntityDof());
    const elements::FormulationModel* formulation = dofMap.formulation();
    const std::int32_t entityCount = dofMap.entities().count();
    m_gridOffsets.reserve(static_cast<std::size_t>(entityCount) + 1);
    m_gridOffsets.push_back(0);
    for (std::int32_t entity = 0; entity < entityCount; ++entity)
    {
        if (formulation != nullptr)
        {
            const std::vector<std::int32_t> rows = formulation->entityGridRows(dofMap.layoutKind(), entity);
            m_gridRows.insert(m_gridRows.end(), rows.begin(), rows.end());
        }
        m_gridOffsets.push_back(m_gridRows.size());
    }
}

void EntityForceCarry::carryResultant(std::int32_t entity, const std::array<double, 6>& resultant,
                                      const Add& add) const
{
    const std::span<const std::int32_t> grids(m_gridRows.data() + m_gridOffsets[static_cast<std::size_t>(entity)],
                                              m_gridOffsets[static_cast<std::size_t>(entity) + 1] -
                                                  m_gridOffsets[static_cast<std::size_t>(entity)]);
    if (grids.empty())
        return;
    // The centre of the grid points and the size of the entity about it.
    base::Vec3 centre{};
    for (const std::int32_t row : grids)
        centre = centre + m_dofMap->positionInBasic(row);
    centre = centre * (1.0 / static_cast<double>(grids.size()));
    double size = 0.0;
    for (const std::int32_t row : grids)
        size = std::max(size, base::norm(m_dofMap->positionInBasic(row) - centre));
    // The weight of a moment against a force: w·L² (1 at a single grid point, which takes the resultant as it is).
    const double weight = size > 0.0 ? ROTATION_WEIGHT * size * size : 1.0;

    // The resultant about the centre: the force, and the moment less centre × force.
    const base::Vec3 force{resultant[0], resultant[1], resultant[2]};
    const base::Vec3 moment = base::Vec3{resultant[3], resultant[4], resultant[5]} - base::cross(centre, force);
    std::array<double, MOTIONS> rhs{force[0], force[1], force[2], moment[0], moment[1], moment[2]};

    // The motion of grid point i in the rigid motion (t, ω) about the centre is u_i = t + ω × d_i, θ_i = ω, that is
    // G_i·(t, ω) with G_i = [I, −[d_i]×; 0, I]. The forces of least weighted norm with the resultant are
    // F = W·G·y, (Gᵀ·W·G)·y = resultant, W = diag(1, 1, 1, w, w, w) per grid point.
    std::array<std::array<double, MOTIONS>, MOTIONS> normal{};
    for (const std::int32_t row : grids)
    {
        const base::Vec3 d = m_dofMap->positionInBasic(row) - centre;
        // −[d]×, the block of the translations on ω.
        const double arm[3][3] = {{0.0, d[2], -d[1]}, {-d[2], 0.0, d[0]}, {d[1], -d[0], 0.0}};
        for (std::size_t i = 0; i < 3; ++i)
        {
            normal[i][i] += 1.0;
            normal[3 + i][3 + i] += weight;
            for (std::size_t j = 0; j < 3; ++j)
            {
                normal[i][3 + j] += arm[i][j];
                normal[3 + j][i] += arm[i][j];
                for (std::size_t k = 0; k < 3; ++k)
                    normal[3 + i][3 + j] += arm[k][i] * arm[k][j];
            }
        }
    }
    const std::array<double, MOTIONS> y = solve(normal, rhs);
    const base::Vec3 translation{y[0], y[1], y[2]};
    const base::Vec3 rotation{y[3], y[4], y[5]};
    for (const std::int32_t row : grids)
    {
        const base::Vec3 d = m_dofMap->positionInBasic(row) - centre;
        const base::Mat3 toCd = m_dofMap->cdToBasic(row).transposed();
        const base::Vec3 pointForce = toCd * (translation + base::cross(rotation, d));
        const base::Vec3 pointMoment = toCd * (rotation * weight);
        for (int c = 0; c < 3; ++c)
        {
            add(m_dofMap->gridDof(row, c + 1), pointForce[static_cast<std::size_t>(c)]);
            add(m_dofMap->gridDof(row, c + 4), pointMoment[static_cast<std::size_t>(c)]);
        }
    }
}

void EntityForceCarry::carry(std::span<const base::DofIndex> dofs, std::span<const double> forces,
                             const Add& add) const
{
    if (!hasEntities())
        return;
    // The resultant of the forces on the DOFs of each entity among `dofs` (an element has a few).
    std::vector<std::pair<std::int32_t, std::array<double, 6>>> resultants; // bounded by the entities of an element
    const base::DofIndex first = m_dofMap->firstEntityDof();
    for (std::size_t k = 0; k < dofs.size(); ++k)
    {
        if (dofs[k] < first || forces[k] == 0.0)
            continue;
        const std::int32_t entity = m_dofMap->labelOf(dofs[k]).row;
        auto found = std::find_if(resultants.begin(), resultants.end(),
                                  [&](const auto& entry) { return entry.first == entity; });
        if (found == resultants.end())
        {
            resultants.emplace_back(entity, std::array<double, 6>{});
            found = resultants.end() - 1;
        }
        const auto row = static_cast<std::size_t>(dofs[k] - first);
        for (std::size_t a = 0; a < MOTIONS; ++a)
            found->second[a] += m_modes[a * m_entityDofCount + row] * forces[k];
    }
    for (const auto& [entity, resultant] : resultants)
        carryResultant(entity, resultant, add);
}

void EntityForceCarry::carry(std::span<double> forcesG) const
{
    if (!hasEntities())
        return;
    const Add add = [&](base::DofIndex gridDof, double value) { forcesG[static_cast<std::size_t>(gridDof)] += value; };
    const base::DofIndex first = m_dofMap->firstEntityDof();
    const elements::EntityLayout& entities = m_dofMap->entities();
    for (std::int32_t entity = 0; entity < entities.count(); ++entity)
    {
        std::array<double, 6> resultant{};
        bool hasForce = false;
        for (int index = 0; index < entities.dofCount(entity); ++index)
        {
            const base::DofIndex dof = m_dofMap->entityDof(entity, index);
            const double force = forcesG[static_cast<std::size_t>(dof)];
            if (force == 0.0)
                continue;
            forcesG[static_cast<std::size_t>(dof)] = 0.0;
            const auto row = static_cast<std::size_t>(dof - first);
            for (std::size_t a = 0; a < MOTIONS; ++a)
                resultant[a] += m_modes[a * m_entityDofCount + row] * force;
            hasForce = true;
        }
        if (hasForce)
            carryResultant(entity, resultant, add);
    }
}

std::vector<ConstraintTerm> EntityForceCarry::termsOf(base::DofIndex entityDof) const
{
    std::vector<ConstraintTerm> terms; // the six components of the grid points of one entity
    if (!hasEntities() || entityDof < m_dofMap->firstEntityDof())
        return terms;
    std::array<double, 6> resultant{};
    const auto row = static_cast<std::size_t>(entityDof - m_dofMap->firstEntityDof());
    for (std::size_t a = 0; a < MOTIONS; ++a)
        resultant[a] = m_modes[a * m_entityDofCount + row];
    carryResultant(m_dofMap->labelOf(entityDof).row, resultant,
                   [&](base::DofIndex gridDof, double value)
                   {
                       if (value != 0.0)
                           terms.push_back({gridDof, value});
                   });
    return terms;
}

} // namespace oos::assembly
