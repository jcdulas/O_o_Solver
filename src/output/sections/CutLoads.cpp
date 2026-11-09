#include "oos/output/sections/CutLoads.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace oos::output
{

namespace
{

// The cuts of one task of the rows (displacementRows).
constexpr std::size_t CUTS_PER_PIECE = 8;

base::Diagnostic cutProblem(const CutDefinition& definition, std::string code, std::string message, std::int64_t id = 0)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = "CUT";
    diagnostic.id = id;
    diagnostic.message = base::concatenate(
        {"cut ", definition.name.empty() ? std::to_string(definition.id) : definition.name, ": ", message});
    return diagnostic;
}

// The coordinate system `frameId` of the snapshot (0: basic), or nothing.
std::optional<input::CoordinateFrame> frameOf(const input::ModelSnapshot& snapshot, std::int32_t frameId)
{
    if (frameId == 0)
        return input::CoordinateFrame{};
    if (!snapshot.hasTable(input::TableRole::CoordinateSystems))
        return std::nullopt;
    const input::CoordinateSystemTable& systems = snapshot.coordinateSystems();
    const std::span<const std::int32_t> ids = systems.ids();
    for (std::size_t row = 0; row < ids.size(); ++row)
        if (ids[row] == frameId)
            return systems.frameAt(static_cast<std::int64_t>(row));
    return std::nullopt;
}

} // namespace

CutLoads::CutLoads(std::vector<CutDefinition> definitions, const input::ModelSnapshot& snapshot,
                   const assembly::ElementSet& set, const assembly::DofMap& dofMap, base::DiagnosticSink& sink)
    : m_definitions(std::move(definitions)), m_set(&set), m_dofMap(&dofMap)
{
    // The elements named by the definitions, found by one pass over the batches (no index of every element).
    std::vector<std::int32_t> wanted;
    for (const CutDefinition& definition : m_definitions)
        wanted.insert(wanted.end(), definition.elements.begin(), definition.elements.end());
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    std::vector<std::pair<std::size_t, std::int64_t>> where(wanted.size(), {0, -1});
    const std::span<const assembly::AssemblyBatch> batches = set.batches();
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const elements::ElementBatchView& view = batches[b].view;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
        {
            const auto found = std::lower_bound(wanted.begin(), wanted.end(), view.elementId(row));
            if (found == wanted.end() || *found != view.elementId(row))
                continue;
            const auto index = static_cast<std::size_t>(found - wanted.begin());
            if (where[index].second < 0)
                where[index] = {b, row};
        }
    }

    for (std::size_t d = 0; d < m_definitions.size(); ++d)
    {
        const CutDefinition& definition = m_definitions[d];
        Cut cut;
        cut.definition = d;
        bool isValid = true;
        const auto refuse = [&](base::Diagnostic diagnostic)
        {
            sink.report(std::move(diagnostic));
            isValid = false;
        };
        if (definition.grids.empty() || definition.elements.empty())
            refuse(cutProblem(
                definition, "cut-empty",
                base::concatenate({"expected grid points and elements, found ", std::to_string(definition.grids.size()),
                                   " grid points and ", std::to_string(definition.elements.size()),
                                   " elements; give the grid points of the cut and the elements on one "
                                   "side of it"})));
        for (const std::int32_t grid : definition.grids)
        {
            if (const std::optional<std::int32_t> row = dofMap.points().findGrid(grid))
                cut.gridRows.push_back(*row);
            else
                refuse(cutProblem(definition, "cut-unknown-grid",
                                  base::concatenate({"names grid point ", std::to_string(grid),
                                                     ", which the model does not have; correct its grid points"}),
                                  grid));
        }
        std::sort(cut.gridRows.begin(), cut.gridRows.end());
        cut.gridRows.erase(std::unique(cut.gridRows.begin(), cut.gridRows.end()), cut.gridRows.end());
        std::vector<std::int32_t> elementIds = definition.elements;
        std::sort(elementIds.begin(), elementIds.end());
        elementIds.erase(std::unique(elementIds.begin(), elementIds.end()), elementIds.end());
        for (const std::int32_t elementId : elementIds)
        {
            const auto index =
                static_cast<std::size_t>(std::lower_bound(wanted.begin(), wanted.end(), elementId) - wanted.begin());
            if (where[index].second < 0)
            {
                refuse(cutProblem(definition, "cut-unknown-element",
                                  base::concatenate({"names element ", std::to_string(elementId),
                                                     ", which the model does not have; correct its elements"}),
                                  elementId));
                continue;
            }
            const auto [batch, row] = where[index];
            if (batches[batch].family->facets().has(elements::Facet::Stiffness))
                cut.elements.push_back({batch, row, elementId});
        }
        // The element order of the batches: the sums of every cut in the order of the assembly.
        std::sort(cut.elements.begin(), cut.elements.end(), [](const Element& a, const Element& b)
                  { return std::make_pair(a.batch, a.row) < std::make_pair(b.batch, b.row); });
        const std::optional<input::CoordinateFrame> positionFrame = frameOf(snapshot, definition.positionFrameId);
        const std::optional<input::CoordinateFrame> displacementFrame =
            frameOf(snapshot, definition.displacementFrameId);
        for (const auto& [frame, field, frameId] :
             {std::make_tuple(&positionFrame, "CP", definition.positionFrameId),
              std::make_tuple(&displacementFrame, "CD", definition.displacementFrameId)})
        {
            if (!*frame)
                refuse(cutProblem(
                    definition, "cut-frame",
                    base::concatenate({"its ", field, " names the coordinate system ", std::to_string(frameId),
                                       ", which the model does not have; add its coordinate system card or "
                                       "correct the ID"}),
                    frameId));
            else if ((*frame)->kind != input::CoordinateFrame::Kind::Rectangular)
                refuse(cutProblem(
                    definition, "cut-frame",
                    base::concatenate({"expected a rectangular ", field,
                                       " (RectFrame1 or RectFrame2), found the cylindrical or spherical system ",
                                       std::to_string(frameId),
                                       "; give the point and the axes in a "
                                       "rectangular system"}),
                    frameId));
        }
        if (!isValid)
            continue;
        cut.point = positionFrame->origin + positionFrame->axes * definition.point;
        cut.cdAxes = displacementFrame->axes;
        cut.firstRow = CUT_COMPONENTS.size() * m_cuts.size();
        m_cuts.push_back(std::move(cut));
    }
}

bool CutLoads::isCutGrid(const Cut& cut, const elements::ElementDof& dof) const
{
    return dof.kind == input::PointKind::Grid && std::binary_search(cut.gridRows.begin(), cut.gridRows.end(), dof.row);
}

std::array<double, 6> CutLoads::unitLoad(const Cut& cut, const elements::ElementBatchView& view, std::int32_t gridRow,
                                         int component) const
{
    const base::Mat3 toBasic = view.displacementFrame(gridRow);
    const auto axis = static_cast<std::size_t>((component - 1) % 3);
    const base::Vec3 direction{toBasic(0, axis), toBasic(1, axis), toBasic(2, axis)};
    base::Vec3 force;
    base::Vec3 moment;
    if (component <= 3)
    {
        force = direction;
        moment = base::cross(view.positionInBasic(gridRow) - cut.point, direction);
    }
    else
    {
        moment = direction;
    }
    const base::Mat3 toCd = cut.cdAxes.transposed();
    const base::Vec3 f = toCd * force;
    const base::Vec3 m = toCd * moment;
    return {f[0], f[1], f[2], m[0], m[1], m[2]};
}

std::array<double, 6> CutLoads::evaluateCut(std::size_t c, const assembly::ElementSet& set,
                                            std::span<const double> displacementsG,
                                            const elements::ElementTemperatures* temperatures) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize)
        throw std::invalid_argument(
            base::concatenate({"CutLoads::evaluateCut: expected ", std::to_string(gSize), " displacements, found ",
                               std::to_string(displacementsG.size())}));
    const Cut& cut = m_cuts[c];
    std::array<double, 6> loads{};
    elements::ElementMatrixBlock stiffness;
    elements::ElementVectorBlock thermal;
    std::vector<double> force; // bounded by the DOFs of one element
    for (const Element& element : cut.elements)
    {
        const assembly::AssemblyBatch& batch = set.batches()[element.batch];
        const elements::ElementRange range{element.row, element.row + 1};
        stiffness.clear();
        batch.family->computeStiffness(batch.view, range, stiffness);
        if (stiffness.empty())
            continue;
        const std::span<const elements::ElementDof> dofs = stiffness.dofs(0);
        const std::span<const double> k = stiffness.matrix(0);
        const std::size_t n = dofs.size();
        bool touchesCut = false;
        for (const elements::ElementDof& dof : dofs)
            touchesCut = touchesCut || isCutGrid(cut, dof);
        if (!touchesCut)
            continue;
        // f = K_e·u_e − f_e,T on the DOFs of the element (StiffnessRecovery: the element force on its grid points is
        // its opposite).
        force.assign(n, 0.0);
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += k[r * n + j] * displacementsG[static_cast<std::size_t>(m_dofMap->dofOf(dofs[j]))];
            force[r] = sum;
        }
        if (temperatures != nullptr && !batch.family->isThermallyInert())
        {
            thermal.clear();
            batch.family->computeThermalLoads(batch.view, range, *temperatures, thermal);
            for (std::size_t t = 0; t < thermal.size(); ++t)
            {
                const std::span<const elements::ElementDof> thermalDofs = thermal.dofs(t);
                const std::span<const double> values = thermal.values(t);
                for (std::size_t i = 0; i < thermalDofs.size(); ++i)
                    for (std::size_t r = 0; r < n; ++r)
                        if (dofs[r] == thermalDofs[i])
                        {
                            force[r] -= values[i];
                            break;
                        }
            }
        }
        for (std::size_t r = 0; r < n; ++r)
        {
            if (!isCutGrid(cut, dofs[r]) || force[r] == 0.0)
                continue;
            const std::array<double, 6> unit = unitLoad(cut, batch.view, dofs[r].row, dofs[r].component);
            for (std::size_t q = 0; q < 6; ++q)
                loads[q] -= force[r] * unit[q];
        }
    }
    return loads;
}

bool CutLoads::cutFieldRates(std::size_t c, const assembly::ElementSet& set, std::span<const double> displacementsG,
                             const elements::ElementTemperatures* temperatures, const elements::DesignedField& field,
                             const std::function<bool(std::size_t, std::int64_t)>& isReached,
                             std::array<double, 6>& out) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize)
        throw std::invalid_argument(
            base::concatenate({"CutLoads::cutFieldRates: expected ", std::to_string(gSize), " displacements, found ",
                               std::to_string(displacementsG.size())}));
    out = {};
    const Cut& cut = m_cuts[c];
    elements::ElementMatrixBlock rate;
    std::vector<double> force; // bounded by the DOFs of one element
    for (const Element& element : cut.elements)
    {
        if (!isReached(element.batch, element.row))
            continue;
        const assembly::AssemblyBatch& batch = set.batches()[element.batch];
        if (temperatures != nullptr && !batch.family->isThermallyInert())
            return false;
        rate.clear();
        if (!batch.family->stiffnessFieldDerivative(batch.view, element.row, field, rate))
            return false;
        if (rate.empty())
            continue;
        // evaluateCut's sum at dK_e: df = dK_e·u_e on the element's DOFs, −df·(the unit loads) on the cut's grids.
        const std::span<const elements::ElementDof> dofs = rate.dofs(0);
        const std::span<const double> k = rate.matrix(0);
        const std::size_t n = dofs.size();
        force.assign(n, 0.0);
        for (std::size_t r = 0; r < n; ++r)
        {
            if (!isCutGrid(cut, dofs[r]))
                continue;
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += k[r * n + j] * displacementsG[static_cast<std::size_t>(m_dofMap->dofOf(dofs[j]))];
            force[r] = sum;
        }
        for (std::size_t r = 0; r < n; ++r)
        {
            if (force[r] == 0.0)
                continue;
            const std::array<double, 6> unit = unitLoad(cut, batch.view, dofs[r].row, dofs[r].component);
            for (std::size_t q = 0; q < 6; ++q)
                out[q] -= force[r] * unit[q];
        }
    }
    return true;
}

void CutLoads::evaluate(const assembly::ElementSet& set, std::span<const double> displacementsG,
                        const elements::ElementTemperatures* temperatures, std::span<double> out) const
{
    if (out.size() != rowCount())
        throw std::invalid_argument(base::concatenate({"CutLoads::evaluate: expected ", std::to_string(rowCount()),
                                                       " values, found ", std::to_string(out.size())}));
    for (std::size_t c = 0; c < m_cuts.size(); ++c)
    {
        const std::array<double, 6> loads = evaluateCut(c, set, displacementsG, temperatures);
        std::copy(loads.begin(), loads.end(), out.begin() + static_cast<std::ptrdiff_t>(m_cuts[c].firstRow));
    }
}

const SparseRows& CutLoads::displacementRows(base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (m_rowsBuilt)
        return m_rows;
    std::vector<std::vector<std::pair<base::DofIndex, double>>> rows(rowCount());
    elements::ElementMatrixBlock stiffness;
    // The entries before they are merged: at most the DOFs of every element per row, reserved first.
    std::uint64_t entries = 0;
    for (const Cut& cut : m_cuts)
        for (const Element& element : cut.elements)
        {
            stiffness.clear();
            const assembly::AssemblyBatch& batch = m_set->batches()[element.batch];
            batch.family->computeStiffness(batch.view, {element.row, element.row + 1}, stiffness);
            if (!stiffness.empty())
                entries += static_cast<std::uint64_t>(stiffness.dofs(0).size());
        }
    const base::MemoryReservation charge = budget.reserve(
        entries * CUT_COMPONENTS.size() * sizeof(std::pair<base::DofIndex, double>), "cut rows before merging");
    // Cut by cut on the pool: each writes its own rows only, its elements in order.
    base::parallelFor(
        pool, m_cuts.size(), CUTS_PER_PIECE,
        [&](std::size_t begin, std::size_t end)
        {
            elements::ElementMatrixBlock stiffness;
            for (std::size_t c = begin; c < end; ++c)
            {
                const Cut& cut = m_cuts[c];
                for (const Element& element : cut.elements)
                {
                    stiffness.clear();
                    const assembly::AssemblyBatch& batch = m_set->batches()[element.batch];
                    batch.family->computeStiffness(batch.view, {element.row, element.row + 1}, stiffness);
                    if (stiffness.empty())
                        continue;
                    const std::span<const elements::ElementDof> dofs = stiffness.dofs(0);
                    const std::span<const double> k = stiffness.matrix(0);
                    const std::size_t n = dofs.size();
                    // Row q: −Σ_r a_q(r)·K_e(r, :) over the DOFs r of the element at the grid points of the cut.
                    std::vector<std::array<double, 6>> coefficients(n,
                                                                    std::array<double, 6>{}); // bounded by the element
                    bool touchesCut = false;
                    for (std::size_t r = 0; r < n; ++r)
                    {
                        if (!isCutGrid(cut, dofs[r]))
                            continue;
                        touchesCut = true;
                        const std::array<double, 6> unit = unitLoad(cut, batch.view, dofs[r].row, dofs[r].component);
                        for (std::size_t j = 0; j < n; ++j)
                            for (std::size_t q = 0; q < 6; ++q)
                                coefficients[j][q] -= unit[q] * k[r * n + j];
                    }
                    if (!touchesCut)
                        continue;
                    for (std::size_t j = 0; j < n; ++j)
                        for (std::size_t q = 0; q < 6; ++q)
                            if (coefficients[j][q] != 0.0)
                                rows[cut.firstRow + q].emplace_back(m_dofMap->dofOf(dofs[j]), coefficients[j][q]);
                }
            }
        });
    // Merged per DOF, the elements' contributions summed in element order (a stable sort).
    for (std::vector<std::pair<base::DofIndex, double>>& row : rows)
    {
        std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::size_t next = 0;
        for (std::size_t i = 0; i < row.size(); ++i)
        {
            if (next > 0 && row[next - 1].first == row[i].first)
                row[next - 1].second += row[i].second;
            else
                row[next++] = row[i];
        }
        row.resize(next);
    }
    m_rows = SparseRows(rows, budget);
    m_rowsBuilt = true;
    return m_rows;
}

} // namespace oos::output
