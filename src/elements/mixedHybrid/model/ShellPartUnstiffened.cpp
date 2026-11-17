#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>

namespace oos::elements::mixedHybrid
{

namespace
{

// A direction of a displacement function is singular when its stiffness is below this fraction of the largest of the
// function: the default threshold of the singularity processor of the grid points (PivotZeroThreshold).
constexpr double UNSTIFFENED_DIRECTION_RATIO = 1e-8;
// The singular directions lie along the components that hold them when |det V_C| reaches cos 1° (the rule of the
// singularity processor of the grid points).
constexpr double ALIGNMENT_COSINE = 0.9998476951563913;
// A rotation about an edge that only lines turn is unstiffened when their diagonal terms on it are at most this
// fraction of their largest diagonal term (a rod without torsion constant), as the edges of lines alone.
constexpr double UNSTIFFENED_DIAGONAL_RATIO = 1e-14;

// The stiffness of the three components of one displacement function.
using FunctionBlock = std::array<std::array<double, 3>, 3>;

// The determinant of the rows `rows` (as many as the columns `columns`, at most three) of v.
double minorOf(const Matrix& v, const std::vector<std::size_t>& rows, const std::vector<std::size_t>& columns)
{
    const auto at = [&](std::size_t i, std::size_t j) { return v(rows[i], columns[j]); };
    if (rows.size() == 1)
        return at(0, 0);
    if (rows.size() == 2)
        return at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);
    return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
           at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
           at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

// The components C that hold the singular directions `singular` (columns of the eigenvectors v of a 3 × 3 block): of
// the subsets of {0, 1, 2} with as many components as directions, the one that maximises |det V_C| (the lowest
// components on a tie, the subsets taken in increasing order), and that value.
std::pair<std::vector<std::size_t>, double> heldComponents(const Matrix& v, const std::vector<std::size_t>& singular)
{
    std::vector<std::size_t> best;
    double bestValue = -1.0;
    // the subsets of the three components as bit masks, in increasing order of their members
    for (const unsigned mask : {1U, 2U, 4U, 3U, 5U, 6U, 7U})
    {
        if (static_cast<std::size_t>(std::popcount(mask)) != singular.size())
            continue;
        std::vector<std::size_t> components;
        for (std::size_t c = 0; c < 3; ++c)
            if ((mask & (1U << c)) != 0)
                components.push_back(c);
        const double value = std::fabs(minorOf(v, components, singular));
        if (value > bestValue)
        {
            bestValue = value;
            best = std::move(components);
        }
    }
    return {std::move(best), bestValue};
}

} // namespace

ShellUnstiffened ShellPart::unstiffenedUnknowns(const std::function<bool(std::size_t)>& followed) const
{
    ShellUnstiffened out;
    const auto partial = [&](std::size_t shell)
    { return m_input.shells[shell].section.parts != SectionParts::MembraneAndBending; };
    if (!m_usable || m_shells.empty())
        return out;
    bool anyPartial = false;
    for (std::size_t shell = 0; shell < m_shells.size(); ++shell)
        anyPartial = anyPartial || partial(shell);
    if (!anyPartial)
        return out;
    const std::size_t k = degree();
    const std::size_t p = k - 1;
    std::set<std::size_t> held;

    // the lines on each edge
    std::vector<std::vector<std::size_t>> linesOfEdge(m_topology.edges.size());
    for (std::size_t line = 0; line < m_lineEdges.size(); ++line)
        if (m_lineEdges[line])
            linesOfEdge[m_lineEdges[line]->edge].push_back(line);

    // ---- the entities whose shells are all partial; the shear and the rotations no element stiffens
    std::vector<bool> candidate(m_entities.size(), false);
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
    {
        const std::vector<std::size_t>& shells = m_topology.edgeElements[e];
        if (shells.empty() || !std::all_of(shells.begin(), shells.end(), partial))
            continue;
        candidate[edgeEntity(e)] = true;
        const std::size_t first = m_entities[edgeEntity(e)].firstUnknown;
        const bool sheared = std::any_of(shells.begin(), shells.end(), [&](std::size_t shell)
                                         { return carriesMoments(shell) && !m_shells[shell].kirchhoff; });
        const bool turned =
            std::any_of(shells.begin(), shells.end(), [&](std::size_t shell) { return carriesMoments(shell); });
        for (std::size_t l = 0; l <= p; ++l)
        {
            if (!sheared)
                held.insert(first + edgeShearDof(l));
            if (!turned && linesOfEdge[e].empty())
                held.insert(first + edgeRotationDof(l));
        }
        // the rotation that only lines turn: held where their diagonal terms on it vanish (a rod without torsion)
        if (turned || linesOfEdge[e].empty())
            continue;
        std::vector<double> diagonal(p + 1, 0.0);
        double largest = 0.0;
        for (const std::size_t line : linesOfEdge[e])
        {
            std::variant<LineData, std::string> built = lineData(line);
            const LineData* data = std::get_if<LineData>(&built);
            if (data == nullptr)
                continue; // a refused line: the problems of the part say so
            for (std::size_t i = 0; i < data->dofs.size(); ++i)
            {
                const double term = std::fabs(data->stiffness(i, i));
                largest = std::max(largest, term);
                for (std::size_t l = 0; l <= p; ++l)
                    if (data->dofs[i] == first + edgeRotationDof(l))
                        diagonal[l] += term;
            }
        }
        for (std::size_t l = 0; l <= p; ++l)
            if (diagonal[l] <= UNSTIFFENED_DIAGONAL_RATIO * largest)
                held.insert(first + edgeRotationDof(l));
    }
    for (std::size_t shell = 0; shell < m_shells.size(); ++shell)
    {
        if (!partial(shell))
            continue;
        const ShellPartEntity& interior = m_entities[interiorEntity(shell)];
        candidate[interiorEntity(shell)] = true;
        if (carriesMoments(shell))
            continue;
        // the interior shear of a shell without moments
        const ShellReferenceElement& re = referenceElement(m_shells[shell].cornerCount);
        for (std::size_t i = 0; i < re.hcurlCount; ++i)
        {
            const std::size_t unknown = m_shells[shell].dofs[3 * re.h1Count + i];
            if (unknown >= interior.firstUnknown && unknown < interior.firstUnknown + interior.dofCount)
                held.insert(unknown);
        }
    }

    // ---- the displacement functions of those entities that no solid follows: their first unknown, and their entity
    std::map<std::size_t, std::size_t> slotOfFunction;
    std::vector<std::size_t> functionEntity;
    std::set<std::size_t> shellsToBuild;
    for (std::size_t entity = 0; entity < m_entities.size(); ++entity)
    {
        if (!candidate[entity] || (followed && followed(entity)))
            continue;
        const ShellPartEntity& e = m_entities[entity];
        std::size_t functions = 0;
        if (e.kind == ShellPartEntity::Kind::Edge)
        {
            functions = k - 1;
            for (const std::size_t shell : m_topology.edgeElements[e.edge])
                shellsToBuild.insert(shell);
        }
        else
        {
            const ShellReferenceElement& re = referenceElement(m_shells[e.shell].cornerCount);
            functions = static_cast<std::size_t>(std::count_if(re.h1Kinds.begin(), re.h1Kinds.end(),
                                                               [](const FunctionKind& f) { return f.isInterior(); }));
            shellsToBuild.insert(e.shell);
        }
        for (std::size_t f = 0; f < functions; ++f)
        {
            slotOfFunction[e.firstUnknown + 3 * f] = functionEntity.size();
            functionEntity.push_back(entity);
        }
    }
    if (functionEntity.empty())
    {
        out.unknowns.assign(held.begin(), held.end());
        return out;
    }

    // ---- the 3 × 3 blocks of the stiffness of the shells and of the lines on the three components of each function
    std::vector<FunctionBlock> blocks(functionEntity.size(), FunctionBlock{});
    // adds the block terms of an element of local stiffness `stiffness`, part unknowns `dofs` and signs `signs`, its
    // displacement unknowns the first `count` local ones
    const auto accumulate = [&](const Matrix& stiffness, std::span<const std::size_t> dofs,
                                std::span<const double> signs, std::size_t count)
    {
        // the slot of each local unknown (or none), and its component
        std::vector<std::pair<std::size_t, std::size_t>> slots;
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t unknown = dofs[i];
            if (unknown < firstEntityUnknown())
                continue;
            const ShellPartUnknown described = describe(unknown);
            const std::size_t component = described.component % 3;
            const auto found = slotOfFunction.find(unknown - component);
            if (found != slotOfFunction.end())
                slots.emplace_back(i, found->second * 3 + component);
        }
        for (const auto& [i, si] : slots)
            for (const auto& [j, sj] : slots)
                if (si / 3 == sj / 3)
                    blocks[si / 3][si % 3][sj % 3] += signs[i] * signs[j] * stiffness(i, j);
    };
    for (const std::size_t shell : shellsToBuild)
    {
        std::variant<ShellElementData, std::string> built = elementWithMatrices(shell, false);
        const ShellElementData* data = std::get_if<ShellElementData>(&built);
        if (data == nullptr)
            continue; // a refused element: the problems of the part say so
        const ShellReferenceElement& re = referenceElement(data->cornerCount);
        accumulate(data->matrices.stiffness, data->dofs, data->signs, 3 * re.h1Count);
    }
    for (std::size_t e = 0; e < linesOfEdge.size(); ++e)
    {
        if (!candidate[edgeEntity(e)])
            continue;
        for (const std::size_t line : linesOfEdge[e])
        {
            std::variant<LineData, std::string> built = lineData(line);
            const LineData* data = std::get_if<LineData>(&built);
            if (data != nullptr)
                accumulate(data->stiffness, data->dofs, data->signs, data->dofs.size());
        }
    }

    // ---- the singular directions of each block, held by the components closest to them
    for (const auto& [firstUnknown, slot] : slotOfFunction)
    {
        Matrix block(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                block(i, j) = 0.5 * (blocks[slot][i][j] + blocks[slot][j][i]);
        const SymmetricEigen eigen = jacobiEigen(block);
        double largest = 0.0;
        for (const double value : eigen.values)
            largest = std::max(largest, value);
        std::vector<std::size_t> singular;
        for (std::size_t d = 0; d < 3; ++d)
            if (!(eigen.values[d] > UNSTIFFENED_DIRECTION_RATIO * largest))
                singular.push_back(d);
        if (singular.empty())
            continue;
        const auto [components, alignment] = heldComponents(eigen.vectors, singular);
        for (const std::size_t c : components)
            held.insert(firstUnknown + c);
        out.displacementDirections += singular.size();
        if (alignment < ALIGNMENT_COSINE)
        {
            if (out.skewedDirections == 0)
                out.firstSkewedEntity = functionEntity[slot];
            out.skewedDirections += singular.size();
        }
    }
    out.unknowns.assign(held.begin(), held.end());
    return out;
}

} // namespace oos::elements::mixedHybrid
