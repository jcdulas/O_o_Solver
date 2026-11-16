#include "oos/elements/mixedHybrid/families/LocalMap.hpp"

namespace oos::elements::mixedHybrid
{

LocalMap localMapOf(const ElementBatchView& view, const MixedHybridModel::LocalDofs& local)
{
    const std::size_t n = local.dofs.size();
    LocalMap map;
    map.dofs = local.dofs;
    map.toLocal = Matrix(n, n);
    const auto signOf = [&](std::size_t i) { return i < local.signs.size() ? local.signs[i] : 1.0; };
    for (std::size_t i = 0; i < n; ++i)
        if (local.basicGridRows[i] < 0)
            map.toLocal(i, i) = signOf(i);
    for (std::size_t first = 0; first < n; ++first)
    {
        const ElementDof& dof = local.dofs[first];
        if (local.basicGridRows[first] < 0 || (dof.component != 1 && dof.component != 4))
            continue;
        const base::Mat3 rotation = view.displacementFrame(local.basicGridRows[first]);
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t b = 0; b < 3; ++b)
                map.toLocal(first + a, first + b) = signOf(first + a) * rotation(a, b);
    }
    return map;
}

std::vector<double> congruent(const Matrix& local, const LocalMap& map)
{
    const Matrix product = transposeTimes(map.toLocal, local * map.toLocal);
    const std::size_t m = map.dofs.size();
    std::vector<double> values(m * m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            values[i * m + j] = 0.5 * (product(i, j) + product(j, i));
    return values;
}

std::vector<double> onDofs(const Vector& local, const LocalMap& map)
{
    std::vector<double> values(map.dofs.size(), 0.0);
    for (std::size_t i = 0; i < map.toLocal.rows(); ++i)
        for (std::size_t j = 0; j < map.dofs.size(); ++j)
            values[j] += map.toLocal(i, j) * local[i];
    return values;
}

Vector localUnknowns(const LocalMap& map, const NodalDisplacements& displacements)
{
    Vector u(map.dofs.size());
    for (std::size_t j = 0; j < map.dofs.size(); ++j)
        u[j] = displacements.valueAt(map.dofs[j]);
    return map.toLocal * std::span<const double>(u);
}

} // namespace oos::elements::mixedHybrid
