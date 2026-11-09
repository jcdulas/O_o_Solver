#include "oos/solutions/sol101/StreamedFront.hpp"

#include "../common/ElementPasses.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace oos::solutions
{
namespace
{

// The rigid-body motions of one g-set DOF about the basic origin (RigidBodyModes' convention: u_cd = Rᵀ·(t + θ × x),
// θ_cd = Rᵀ·θ; the columns the translations along the basic axes, then the rotations about them); zero for a scalar
// point.
std::array<double, 6> rigidRowOf(const assembly::DofMap& dofMap, const elements::ElementDof& dof)
{
    std::array<double, 6> row{};
    if (dof.kind == input::PointKind::Scalar)
        return row;
    const base::Mat3 toCd = dofMap.cdToBasic(dof.row).transposed();
    const base::Vec3 x = dofMap.positionInBasic(dof.row);
    const auto c = static_cast<std::size_t>(dof.component - 1);
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        base::Vec3 translation{};
        base::Vec3 rotation{};
        if (mode < 3)
            translation[mode] = 1.0;
        else
        {
            rotation[mode - 3] = 1.0;
            translation = base::cross(rotation, x);
        }
        row[mode] = c < 3 ? (toCd * translation)[c] : (toCd * rotation)[c - 3];
    }
    return row;
}

// The row of the support motions of inertia relief at one g-set DOF before C (the full path's D = Φ·C): Φ on a free
// independent DOF, zero on a held one (the s set of the relief's sets), Σ coefficient·row of its Gm terms on a
// dependent one (u_m = Gm·u_n).
std::array<double, 6> reliefRowAt(const StreamedFront::Inputs& inputs, const elements::ElementDof& dof)
{
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    std::array<double, 6> row{};
    const base::DofIndex g = dofMap.dofOf(dof);
    const auto addIndependent = [&](base::DofIndex independent, double coefficient)
    {
        if (inputs.reliefSets->contains(assembly::DofSet::S, independent))
            return;
        const linalg::NodeDof nodeDof = dofMap.nodeDofOf(independent);
        if (nodeDof.node >= dofMap.gridCount())
            return;
        const std::array<double, 6> phi =
            rigidRowOf(dofMap, elements::ElementDof::grid(nodeDof.node, nodeDof.component + 1));
        for (std::size_t mode = 0; mode < 6; ++mode)
            row[mode] += coefficient * phi[mode];
    };
    if (const std::optional<std::size_t> dependent = inputs.gm->findRow(g))
        for (const assembly::ConstraintTerm& term : inputs.gm->row(*dependent))
            addIndependent(term.dof, term.coefficient);
    else
        addIndependent(g, 1.0);
    return row;
}

// The work of one run of consecutive rows of a batch: its stiffness, its mass, its pressures and their loads.
struct RunWork
{
    elements::ElementMatrixBlock stiffness;
    elements::ElementMatrixBlock mass;
    std::vector<std::int64_t> pressureRows;
    std::vector<std::int32_t> pressureConsumers; // per pressure: the consumers of its set
    std::vector<assembly::LoadVectorBuilder::ElementLocation> locations;
    std::vector<elements::ElementPressure> pressures;
    elements::ElementVectorBlock pressureLoads;
};

void computeRun(const StreamedFront::Inputs& inputs, std::size_t batchIndex, elements::ElementRange range,
                bool needsStiffness, bool needsLoads, bool needsMass, RunWork& work)
{
    const assembly::AssemblyBatch& batch = inputs.model->elementSet().batches()[batchIndex];
    const elements::FacetSet facets = batch.family->facets();
    work.stiffness.clear();
    work.mass.clear();
    work.pressureLoads.clear();
    work.pressureRows.clear();
    work.pressureConsumers.clear();
    if (needsStiffness && facets.has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, range, work.stiffness);
    if (!needsLoads)
        return;
    if (needsMass && facets.has(elements::Facet::Mass))
    {
        elements::MassOptions options;
        options.formulation = inputs.massOptions.formulation;
        options.addedNonStructuralMass = inputs.shares->sharesOf(batchIndex);
        batch.family->computeMass(batch.view, range, options, work.mass);
    }
    if (inputs.loads->hasPressures())
    {
        const std::int64_t first = inputs.elements->batchFirst(batchIndex);
        for (std::int64_t row = range.first; row < range.last; ++row)
            for (const StreamedLoads::ElementPressureRow& pressure :
                 inputs.loads->pressuresOf(static_cast<std::int32_t>(first + row)))
            {
                work.pressureRows.push_back(pressure.row);
                work.pressureConsumers.push_back(pressure.consumers);
            }
        if (!work.pressureRows.empty())
        {
            inputs.loadBuilder->pressuresOfRows(work.pressureRows, work.locations, work.pressures);
            batch.family->computePressureLoads(batch.view, work.pressures, work.pressureLoads);
            if (work.pressureLoads.size() != work.pressures.size())
                throw base::InternalError(base::concatenate(
                    {"the full path streamed: ", std::string(batch.view.card()), " gave ",
                     std::to_string(work.pressureLoads.size()), " pressure loads for ",
                     std::to_string(work.pressures.size()), " pressures (a bug of O_o_Solver: report it)"}));
        }
    }
}

// The loads of one element's mass for every vector (WTMASS applied): f_b = W·M_e·field_b on the mass's DOFs, the
// field of a vector at a DOF its GravityLoad acceleration (a uniform translation, in the CD system) less its rigid
// relief D·ü = row·w (reliefRowOf).
void massLoads(const StreamedFront::Inputs& inputs, std::span<const elements::ElementDof> dofs,
               std::span<const double> matrix, std::size_t columns, std::vector<double>& loads)
{
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    const std::size_t n = dofs.size();
    loads.assign(n * columns, 0.0);
    const std::span<const std::array<double, 3>> gravity = inputs.loads->accelerations();
    const std::span<const std::array<double, 6>> relief = inputs.loads->reliefAmplitudes();
    // Per DOF: the CD components of the basic axes (zero for a rotation or a scalar point) and the relief's row.
    std::vector<std::array<double, 3>> axes(n); // bounded by the element
    std::vector<std::array<double, 6>> rows(n); // bounded by the element
    for (std::size_t j = 0; j < n; ++j)
    {
        const elements::ElementDof& dof = dofs[j];
        if (dof.kind == input::PointKind::Scalar)
            continue;
        const auto c = static_cast<std::size_t>(dof.component - 1);
        if (c < 3)
        {
            const base::Mat3 toCd = dofMap.cdToBasic(dof.row).transposed();
            for (std::size_t a = 0; a < 3; ++a)
                axes[j][a] = toCd(c, a);
        }
        if (!relief.empty())
            rows[j] = reliefRowAt(inputs, dof);
    }
    std::vector<double> field(n); // bounded by the element
    const double weight = inputs.massOptions.weightToMass;
    for (std::size_t b = 0; b < columns; ++b)
    {
        bool isZero = true;
        for (std::size_t j = 0; j < n; ++j)
        {
            double value = 0.0;
            for (std::size_t a = 0; a < 3; ++a)
                value += axes[j][a] * gravity[b][a];
            if (!relief.empty())
                for (std::size_t mode = 0; mode < 6; ++mode)
                    value -= rows[j][mode] * relief[b][mode];
            field[j] = value;
            isZero = isZero && value == 0.0;
        }
        if (isZero)
            continue;
        for (std::size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += matrix[i * n + j] * field[j];
            loads[i + b * n] = weight * sum;
        }
    }
}

} // namespace

StreamedFront::StreamedFront(const Inputs& inputs) : m_inputs(inputs) {}

linalg::DenseMatrix StreamedFront::loadsAt(std::span<const base::DofIndex> dofs, base::MemoryBudget& budget) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const std::size_t columns = m_inputs.loads->columnCount();
    linalg::DenseMatrix out(budget, dofs.size(), columns, "streamed loads at kept DOFs");
    if (dofs.empty())
        return out;
    const auto placeOf = [&](base::DofIndex dof) -> std::optional<std::size_t>
    {
        const auto found = std::lower_bound(dofs.begin(), dofs.end(), dof);
        if (found == dofs.end() || *found != dof)
            return std::nullopt;
        return static_cast<std::size_t>(found - dofs.begin());
    };
    base::BudgetedBuffer<std::uint8_t> isNode(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                              "streamed loads: kept nodes");
    for (const base::DofIndex dof : dofs)
        isNode[static_cast<std::size_t>(dofMap.nodeDofOf(dof).node)] = 1;
    // The elements with a node there, in their order (batches, rows), in ranges of consecutive rows.
    const StreamedElements& elements = *m_inputs.elements;
    const auto batches = m_inputs.model->elementSet().batches();
    RunWork work;
    std::vector<double> loads; // bounded by an element's DOFs times the vectors
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        std::vector<elements::ElementRange> ranges; // bounded by the batch
        for (std::int64_t row = 0; row < batches[b].view.rowCount(); ++row)
        {
            const auto element = static_cast<std::int32_t>(elements.batchFirst(b) + row);
            const std::span<const std::int32_t> nodes = elements.nodesOf(element);
            if (std::none_of(nodes.begin(), nodes.end(),
                             [&](std::int32_t node) { return isNode[static_cast<std::size_t>(node)] != 0; }))
                continue;
            if (!ranges.empty() && ranges.back().last == row)
                ranges.back().last = row + 1;
            else
                ranges.push_back({row, row + 1});
        }
        for (const elements::ElementRange& range : ranges)
        {
            computeRun(m_inputs, b, range, false, true, true, work);
            for (std::size_t m = 0; m < work.mass.size(); ++m)
            {
                const std::span<const elements::ElementDof> elementDofs = work.mass.dofs(m);
                massLoads(m_inputs, elementDofs, work.mass.matrix(m), columns, loads);
                for (std::size_t i = 0; i < elementDofs.size(); ++i)
                    if (const std::optional<std::size_t> place = placeOf(dofMap.dofOf(elementDofs[i])))
                        for (std::size_t column = 0; column < columns; ++column)
                            out(*place, column) += loads[i + column * elementDofs.size()];
            }
            for (std::size_t k = 0; k < work.pressureLoads.size(); ++k)
            {
                const std::span<const elements::ElementDof> elementDofs = work.pressureLoads.dofs(k);
                const std::span<const double> values = work.pressureLoads.values(k);
                for (const StreamedLoads::Consumer& taker : m_inputs.loads->consumers(work.pressureConsumers[k]))
                    for (std::size_t i = 0; i < elementDofs.size(); ++i)
                        if (const std::optional<std::size_t> place = placeOf(dofMap.dofOf(elementDofs[i])))
                            out(*place, static_cast<std::size_t>(taker.column)) += taker.factor * values[i];
            }
        }
    }
    for (const StreamedLoads::DofLoad& load : m_inputs.loads->concentratedLoads())
        if (const std::optional<std::size_t> place = placeOf(load.dof))
            out(*place, static_cast<std::size_t>(load.column)) += load.value;
    return out;
}

std::array<double, 6> StreamedFront::reliefRowOf(base::DofIndex dof) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dof);
    if (nodeDof.node >= dofMap.gridCount())
        return {};
    return reliefRowAt(m_inputs, elements::ElementDof::grid(nodeDof.node, nodeDof.component + 1));
}

void StreamedFront::addTerms(linalg::FrontTerms& front)
{
    const StreamedElements& elements = *m_inputs.elements;
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const std::size_t columns = front.rightHandColumns.columns;
    const std::span<const std::int32_t> placed = elements.elementsAt(front.supernode);
    RunWork work;
    StreamedReduction reduction;
    std::vector<base::DofIndex> gDofs;
    std::vector<std::int32_t> positions;
    std::vector<double> reduced;
    std::vector<double> loads;
    // The terms of a matrix on DOFs (an element's stiffness) added to the front: K_red = Tᵀ·K·T.
    const auto addMatrix = [&](std::span<const elements::ElementDof> dofs, std::span<const double> matrix)
    {
        passes::gDofsOf(dofMap, dofs, gDofs);
        elements.reduce(gDofs, reduction);
        passes::positionsOf(front, reduction, positions);
        const std::size_t n = dofs.size();
        const std::size_t m = reduction.equations.size();
        reduced.assign(m * m, 0.0);
        for (const StreamedReduction::Term& a : reduction.terms)
            for (const StreamedReduction::Term& b : reduction.terms)
                reduced[static_cast<std::size_t>(a.equation) * m + static_cast<std::size_t>(b.equation)] +=
                    a.coefficient * b.coefficient *
                    matrix[static_cast<std::size_t>(a.dof) * n + static_cast<std::size_t>(b.dof)];
        for (std::size_t p = 0; p < m; ++p)
            for (std::size_t q = 0; q <= p; ++q)
                front.at(positions[p], positions[q]) += reduced[p * m + q];
    };
    // The loads of the columns on DOFs (n × columns) added to the front's right-hand sides: f_red = Tᵀ·f, scaled.
    const auto addLoads = [&](std::span<const elements::ElementDof> dofs, std::span<const double> values,
                              std::span<const StreamedLoads::Consumer> takers)
    {
        passes::gDofsOf(dofMap, dofs, gDofs);
        elements.reduce(gDofs, reduction);
        passes::positionsOf(front, reduction, positions);
        const std::size_t n = dofs.size();
        if (takers.empty())
        {
            for (std::size_t b = 0; b < columns; ++b)
                for (const StreamedReduction::Term& term : reduction.terms)
                    front.rightHandSide(positions[static_cast<std::size_t>(term.equation)], b) +=
                        term.coefficient * values[static_cast<std::size_t>(term.dof) + b * n];
            return;
        }
        for (const StreamedLoads::Consumer& taker : takers)
            for (const StreamedReduction::Term& term : reduction.terms)
                front.rightHandSide(positions[static_cast<std::size_t>(term.equation)],
                                    static_cast<std::size_t>(taker.column)) +=
                    taker.factor * term.coefficient * values[static_cast<std::size_t>(term.dof)];
    };

    for (std::size_t k = 0; k < placed.size();)
    {
        // A run of consecutive rows of one batch.
        const StreamedElements::Place place = elements.placeOf(placed[k]);
        std::size_t end = k + 1;
        const std::int64_t batchEnd = elements.batchFirst(place.batch + 1);
        while (end < placed.size() && placed[end] == placed[end - 1] + 1 && placed[end] < batchEnd)
            ++end;
        const elements::ElementRange range{place.row, place.row + static_cast<std::int64_t>(end - k)};
        computeRun(m_inputs, place.batch, range, true, columns > 0, m_inputs.loads->usesMass(), work);
        // In element order: each element's stiffness, then its mass's loads, then its pressures' loads.
        std::size_t stiffness = 0;
        std::size_t mass = 0;
        std::size_t pressure = 0;
        for (std::int64_t row = range.first; row < range.last; ++row)
        {
            for (; stiffness < work.stiffness.size() && work.stiffness.elementRow(stiffness) == row; ++stiffness)
                addMatrix(work.stiffness.dofs(stiffness), work.stiffness.matrix(stiffness));
            for (; mass < work.mass.size() && work.mass.elementRow(mass) == row; ++mass)
            {
                massLoads(m_inputs, work.mass.dofs(mass), work.mass.matrix(mass), columns, loads);
                addLoads(work.mass.dofs(mass), loads, {});
            }
            for (; pressure < work.pressureLoads.size() && work.pressureLoads.elementRow(pressure) == row; ++pressure)
                addLoads(work.pressureLoads.dofs(pressure), work.pressureLoads.values(pressure),
                         m_inputs.loads->consumers(work.pressureConsumers[pressure]));
        }
        k = end;
    }
    for (const StreamedLoads::EquationLoad& load : m_inputs.loads->concentratedAt(front.supernode))
        front.rightHandSide(front.positionOf(load.equation), static_cast<std::size_t>(load.column)) += load.value;
    // The refinement's residual: K·û subtracted at the equations' own supernode, once each.
    if (m_inputs.subtractedLoads.columns > 0)
        for (std::int32_t p = 0; p < front.columnCount; ++p)
            for (std::size_t b = 0; b < columns; ++b)
                front.rightHandSide(p, b) -=
                    m_inputs.subtractedLoads(static_cast<std::size_t>(front.firstColumn + p), b);
}

void StreamedFront::diagonal(std::int32_t supernode, std::span<double> values) const
{
    const linalg::SymbolicFactor& symbolic = m_inputs.elements->symbolic();
    const linalg::DofSelection& selection = m_inputs.elements->selection();
    const std::int32_t first = symbolic.firstColumn(supernode);
    for (std::size_t j = 0; j < values.size(); ++j)
        values[j] =
            m_inputs
                .diagonalOfDof[static_cast<std::size_t>(selection.dofOfEquation(first + static_cast<std::int32_t>(j)))];
}

StreamedFront::ReliefSums StreamedFront::reliefSums(base::ThreadPool* pool, base::MemoryBudget& budget,
                                                    const base::CancellationToken* cancellation) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const std::size_t columns = m_inputs.loads->columnCount();
    ReliefSums sums;
    sums.resultants.assign(columns, {});
    constexpr std::int64_t RUN_ROWS = 2048;
    // The runs (batch, rows), and each run's sums: DᵀMD's 36 values, then 6 a vector.
    struct Run
    {
        std::size_t batch;
        elements::ElementRange range;
    };
    std::vector<Run> runs; // bounded by the elements / RUN_ROWS
    const auto batches = m_inputs.model->elementSet().batches();
    for (std::size_t b = 0; b < batches.size(); ++b)
        for (std::int64_t first = 0; first < batches[b].view.rowCount(); first += RUN_ROWS)
            runs.push_back({b, {first, std::min(first + RUN_ROWS, batches[b].view.rowCount())}});
    const std::size_t stride = 36 + 6 * columns;
    base::BudgetedBuffer<double> partial(budget, runs.size() * stride, "streamed relief: sums of the runs");
    const auto sumRun = [&](std::size_t r)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        double* const into = partial.data() + r * stride;
        RunWork work;
        std::vector<double> loads;              // bounded by an element's DOFs times the vectors
        std::vector<std::array<double, 6>> phi; // bounded by an element's DOFs
        computeRun(m_inputs, runs[r].batch, runs[r].range, false, true, true, work);
        std::size_t mass = 0;
        std::size_t pressure = 0;
        for (std::int64_t row = runs[r].range.first; row < runs[r].range.last; ++row)
        {
            for (; mass < work.mass.size() && work.mass.elementRow(mass) == row; ++mass)
            {
                const std::span<const elements::ElementDof> dofs = work.mass.dofs(mass);
                const std::span<const double> matrix = work.mass.matrix(mass);
                const std::size_t n = dofs.size();
                phi.resize(n);
                for (std::size_t i = 0; i < n; ++i)
                    phi[i] = reliefRowAt(m_inputs, dofs[i]);
                for (std::size_t a = 0; a < 6; ++a)
                    for (std::size_t c = 0; c < 6; ++c)
                    {
                        double sum = 0.0;
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t j = 0; j < n; ++j)
                                sum += phi[i][a] * matrix[i * n + j] * phi[j][c];
                        into[6 * a + c] += m_inputs.massOptions.weightToMass * sum;
                    }
                massLoads(m_inputs, dofs, matrix, columns, loads);
                for (std::size_t column = 0; column < columns; ++column)
                    for (std::size_t i = 0; i < n; ++i)
                        if (const double value = loads[i + column * n]; value != 0.0)
                            for (std::size_t mode = 0; mode < 6; ++mode)
                                into[36 + 6 * column + mode] += phi[i][mode] * value;
            }
            for (; pressure < work.pressureLoads.size() && work.pressureLoads.elementRow(pressure) == row; ++pressure)
            {
                const std::span<const elements::ElementDof> dofs = work.pressureLoads.dofs(pressure);
                const std::span<const double> values = work.pressureLoads.values(pressure);
                std::array<double, 6> resultant{}; // the element's pressure load on the support motions
                for (std::size_t i = 0; i < dofs.size(); ++i)
                {
                    const std::array<double, 6> row6 = reliefRowAt(m_inputs, dofs[i]);
                    for (std::size_t mode = 0; mode < 6; ++mode)
                        resultant[mode] += row6[mode] * values[i];
                }
                for (const StreamedLoads::Consumer& taker : m_inputs.loads->consumers(work.pressureConsumers[pressure]))
                    for (std::size_t mode = 0; mode < 6; ++mode)
                        into[36 + 6 * static_cast<std::size_t>(taker.column) + mode] += taker.factor * resultant[mode];
            }
        }
    };
    if (pool != nullptr && pool->threadCount() > 1)
        pool->runBatch(runs.size(), sumRun);
    else
        for (std::size_t r = 0; r < runs.size(); ++r)
            sumRun(r);
    for (std::size_t r = 0; r < runs.size(); ++r)
    {
        const double* const from = partial.data() + r * stride;
        for (std::size_t a = 0; a < 6; ++a)
            for (std::size_t c = 0; c < 6; ++c)
                sums.rigidBodyMass(a, c) += from[6 * a + c];
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t mode = 0; mode < 6; ++mode)
                sums.resultants[column][mode] += from[36 + 6 * column + mode];
    }
    const auto addResultant = [&](const elements::ElementDof& dof, std::size_t column, double value)
    {
        const std::array<double, 6> row6 = reliefRowAt(m_inputs, dof);
        for (std::size_t mode = 0; mode < 6; ++mode)
            sums.resultants[column][mode] += row6[mode] * value;
    };
    for (const StreamedLoads::DofLoad& load : m_inputs.loads->concentratedLoads())
    {
        const assembly::DofLabel label = dofMap.labelOf(load.dof);
        if (label.kind == input::PointKind::Scalar)
            continue;
        const linalg::NodeDof nodeDof = dofMap.nodeDofOf(load.dof);
        addResultant(elements::ElementDof::grid(nodeDof.node, nodeDof.component + 1),
                     static_cast<std::size_t>(load.column), load.value);
    }
    return sums;
}

namespace
{

// The pieces of the element set for the refinement's passes: runs of consecutive rows of one batch.
constexpr std::int64_t REFINEMENT_PIECE_ROWS = 1024;
// The most equation terms an element carries in those passes, for their memory bound (a CHEXA20's 60 DOFs and the
// relations of its points).
constexpr std::uint64_t REFINEMENT_TERMS_PER_ELEMENT = 128;

} // namespace

std::uint64_t StreamedFront::refinementWorkBytes(std::size_t threads, std::size_t columns) noexcept
{
    const std::uint64_t wave = 4 * std::max<std::uint64_t>(threads, 1);
    return wave * static_cast<std::uint64_t>(REFINEMENT_PIECE_ROWS) * REFINEMENT_TERMS_PER_ELEMENT *
           (static_cast<std::uint64_t>(columns) * sizeof(double) + sizeof(std::int32_t));
}

std::vector<std::array<double, 6>> StreamedFront::rigidMotionSums(linalg::ConstDenseMatrixView solution,
                                                                  base::ThreadPool* pool,
                                                                  const base::CancellationToken* cancellation) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const std::size_t columns = solution.columns;
    std::vector<std::array<double, 6>> sums(columns); // per column
    const elements::Facet massFacet[] = {elements::Facet::Mass};
    const std::vector<passes::ElementPiece> pieces =
        passes::elementPieces(*m_inputs.model, massFacet, REFINEMENT_PIECE_ROWS);
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    std::vector<std::vector<double>> partial(std::max<std::size_t>(4 * threads, 1)); // per slot of a wave: 6 × columns
    const auto work = [&](std::size_t p, std::size_t slot)
    {
        std::vector<double>& into = partial[slot];
        into.assign(6 * columns, 0.0);
        const assembly::AssemblyBatch& batch = m_inputs.model->elementSet().batches()[pieces[p].batch];
        elements::ElementMatrixBlock mass;
        elements::MassOptions options;
        options.formulation = m_inputs.massOptions.formulation;
        options.addedNonStructuralMass = m_inputs.shares->sharesOf(pieces[p].batch);
        batch.family->computeMass(batch.view, pieces[p].range, options, mass);
        StreamedReduction reduction;
        std::vector<base::DofIndex> gDofs;
        std::vector<double> values;
        std::vector<std::array<double, 6>> phi; // bounded by an element's DOFs
        for (std::size_t e = 0; e < mass.size(); ++e)
        {
            const std::span<const elements::ElementDof> dofs = mass.dofs(e);
            const std::span<const double> matrix = mass.matrix(e);
            const std::size_t n = dofs.size();
            passes::gDofsOf(dofMap, dofs, gDofs);
            m_inputs.elements->reduce(gDofs, reduction);
            passes::elementValues(reduction, n, solution, values);
            phi.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                phi[i] = reliefRowAt(m_inputs, dofs[i]);
            for (std::size_t b = 0; b < columns; ++b)
                for (std::size_t i = 0; i < n; ++i)
                {
                    double force = 0.0; // (M_e·u_e)_i
                    for (std::size_t j = 0; j < n; ++j)
                        force += matrix[i * n + j] * values[j + b * n];
                    if (force == 0.0)
                        continue;
                    for (std::size_t mode = 0; mode < 6; ++mode)
                        into[6 * b + mode] += m_inputs.massOptions.weightToMass * phi[i][mode] * force;
                }
        }
    };
    const auto merge = [&](std::size_t, std::size_t slot)
    {
        for (std::size_t b = 0; b < columns; ++b)
            for (std::size_t mode = 0; mode < 6; ++mode)
                sums[b][mode] += partial[slot][6 * b + mode];
    };
    passes::inWaves(pieces.size(), pool, cancellation, work, merge);
    return sums;
}

void StreamedFront::multiplyStiffness(linalg::ConstDenseMatrixView solution,
                                      std::span<const std::array<double, 6>> amplitudes,
                                      linalg::DenseMatrixView products, base::ThreadPool* pool,
                                      const base::CancellationToken* cancellation) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const std::size_t columns = solution.columns;
    for (std::size_t b = 0; b < columns; ++b)
        std::fill_n(&products(0, b), products.rows, 0.0);
    const elements::Facet stiffnessFacet[] = {elements::Facet::Stiffness};
    const std::vector<passes::ElementPiece> pieces =
        passes::elementPieces(*m_inputs.model, stiffnessFacet, REFINEMENT_PIECE_ROWS);
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    // Per slot of a wave: the piece's terms (an equation and its value per column), in element order.
    struct Terms
    {
        std::vector<std::int32_t> equations;
        std::vector<double> values; // equations × columns, row by row
    };
    std::vector<Terms> partial(std::max<std::size_t>(4 * threads, 1));
    const auto work = [&](std::size_t p, std::size_t slot)
    {
        Terms& into = partial[slot];
        into.equations.clear();
        into.values.clear();
        const assembly::AssemblyBatch& batch = m_inputs.model->elementSet().batches()[pieces[p].batch];
        elements::ElementMatrixBlock stiffness;
        batch.family->computeStiffness(batch.view, pieces[p].range, stiffness);
        StreamedReduction reduction;
        std::vector<base::DofIndex> gDofs;
        std::vector<double> values;
        std::vector<double> forces;
        for (std::size_t e = 0; e < stiffness.size(); ++e)
        {
            const std::span<const elements::ElementDof> dofs = stiffness.dofs(e);
            const std::span<const double> matrix = stiffness.matrix(e);
            const std::size_t n = dofs.size();
            passes::gDofsOf(dofMap, dofs, gDofs);
            m_inputs.elements->reduce(gDofs, reduction);
            if (reduction.equations.empty())
                continue;
            passes::elementValues(reduction, n, solution, values);
            // û_e = T·u − Φ_e·a: the gauge of the residual, M-orthogonal to the rigid motions.
            for (std::size_t i = 0; i < n && !amplitudes.empty(); ++i)
            {
                const std::array<double, 6> row6 = reliefRowAt(m_inputs, dofs[i]);
                for (std::size_t b = 0; b < columns; ++b)
                    for (std::size_t mode = 0; mode < 6; ++mode)
                        values[i + b * n] -= row6[mode] * amplitudes[b][mode];
            }
            forces.assign(n * columns, 0.0);
            for (std::size_t b = 0; b < columns; ++b)
                for (std::size_t i = 0; i < n; ++i)
                {
                    double sum = 0.0;
                    for (std::size_t j = 0; j < n; ++j)
                        sum += matrix[i * n + j] * values[j + b * n];
                    forces[i + b * n] = sum;
                }
            const std::size_t m = reduction.equations.size();
            const std::size_t base = into.equations.size();
            into.equations.insert(into.equations.end(), reduction.equations.begin(), reduction.equations.end());
            into.values.resize((base + m) * columns, 0.0);
            for (const StreamedReduction::Term& term : reduction.terms)
                for (std::size_t b = 0; b < columns; ++b)
                    into.values[(base + static_cast<std::size_t>(term.equation)) * columns + b] +=
                        term.coefficient * forces[static_cast<std::size_t>(term.dof) + b * n];
        }
    };
    const auto merge = [&](std::size_t, std::size_t slot)
    {
        const Terms& from = partial[slot];
        for (std::size_t t = 0; t < from.equations.size(); ++t)
            for (std::size_t b = 0; b < columns; ++b)
                products(static_cast<std::size_t>(from.equations[t]), b) += from.values[t * columns + b];
    };
    passes::inWaves(pieces.size(), pool, cancellation, work, merge);
}

StreamedFront::NodeDiagonals StreamedFront::nodeDiagonals(const AnalysisModel& model,
                                                          const assembly::RigidElimination& gm,
                                                          base::MemoryBudget& budget, base::ThreadPool* pool,
                                                          const base::CancellationToken* cancellation)
{
    const assembly::DofMap& dofMap = model.dofMap();
    const auto grids = static_cast<std::size_t>(dofMap.gridCount());
    base::BudgetedBuffer<double> gridBlocks(budget, 18 * grids, "streamed node diagonal blocks");
    base::BudgetedBuffer<double> scalarDiagonals(budget, static_cast<std::size_t>(dofMap.scalarCount()),
                                                 "streamed node diagonal blocks");
    // One element's contributions to the blocks: (the block's first value, 3 × 3 or 1 values).
    struct Contribution
    {
        std::size_t offset = 0;
        std::array<double, 9> values{};
        bool isScalar = false;
    };
    // PERFORMANCE: the elements' matrices made in pieces of a fixed size on the pool, their contributions summed in
    // element order after each wave: the same sums whatever the threads, the matrices off the calling thread.
    constexpr std::int64_t PIECE_ROWS = 1024;
    struct Piece
    {
        std::size_t batch = 0;
        elements::ElementRange range;
        std::vector<Contribution> contributions; // bounded by the piece
        double rigidForce = 0.0;
    };
    const auto batches = model.elementSet().batches();
    std::vector<Piece> pieces; // bounded by the elements over PIECE_ROWS
    for (std::size_t b = 0; b < batches.size(); ++b)
        if (batches[b].family->facets().has(elements::Facet::Stiffness))
            for (std::int64_t first = 0; first < batches[b].view.rowCount(); first += PIECE_ROWS)
                pieces.push_back({b, {first, std::min(first + PIECE_ROWS, batches[b].view.rowCount())}, {}});
    const std::size_t threads = pool != nullptr ? pool->threadCount() : 1;
    const std::size_t wave = std::max<std::size_t>(4 * threads, 1);
    double largestRigidForce = 0.0;
    for (std::size_t begin = 0; begin < pieces.size(); begin += wave)
    {
        const std::size_t end = std::min(begin + wave, pieces.size());
        base::parallelFor(
            pool, end - begin, 1,
            [&](std::size_t first, std::size_t last)
            {
                elements::ElementMatrixBlock block;
                std::vector<base::DofIndex> nDofs;                              // the n-set DOFs reached, sorted
                std::vector<double> reduced;                                    // the element's Knn terms on them
                std::vector<std::vector<std::pair<std::size_t, double>>> terms; // per element DOF: (n DOF, coef)
                std::vector<std::array<double, 6>> phiRows;                     // per n DOF: its rigid motions
                for (std::size_t p = begin + first; p < begin + last; ++p)
                {
                    Piece& piece = pieces[p];
                    const assembly::AssemblyBatch& batch = batches[piece.batch];
                    block.clear();
                    batch.family->computeStiffness(batch.view, piece.range, block);
                    for (std::size_t e = 0; e < block.size(); ++e)
                    {
                        const std::span<const elements::ElementDof> dofs = block.dofs(e);
                        const std::span<const double> matrix = block.matrix(e);
                        const std::size_t n = dofs.size();
                        nDofs.clear();
                        terms.assign(n, {});
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            const base::DofIndex dof = dofMap.dofOf(dofs[i]);
                            if (const std::optional<std::size_t> row = gm.findRow(dof))
                                for (const assembly::ConstraintTerm& term : gm.row(*row))
                                {
                                    terms[i].push_back({static_cast<std::size_t>(term.dof), term.coefficient});
                                    nDofs.push_back(term.dof);
                                }
                            else
                            {
                                terms[i].push_back({static_cast<std::size_t>(dof), 1.0});
                                nDofs.push_back(dof);
                            }
                        }
                        std::sort(nDofs.begin(), nDofs.end());
                        nDofs.erase(std::unique(nDofs.begin(), nDofs.end()), nDofs.end());
                        const std::size_t m = nDofs.size();
                        const auto local = [&](std::size_t dof)
                        {
                            return static_cast<std::size_t>(
                                std::lower_bound(nDofs.begin(), nDofs.end(), static_cast<base::DofIndex>(dof)) -
                                nDofs.begin());
                        };
                        reduced.assign(m * m, 0.0);
                        for (std::size_t i = 0; i < n; ++i)
                            for (const auto& [di, ci] : terms[i])
                                for (std::size_t j = 0; j < n; ++j)
                                    for (const auto& [dj, cj] : terms[j])
                                        reduced[local(di) * m + local(dj)] += ci * cj * matrix[i * n + j];
                        // K_e·Φ on the n set: zero but round-off for an element that does not resist a rigid motion.
                        phiRows.assign(m, {});
                        for (std::size_t c = 0; c < m; ++c)
                        {
                            const linalg::NodeDof nodeDof = dofMap.nodeDofOf(nDofs[c]);
                            if (nodeDof.node < dofMap.gridCount())
                                phiRows[c] =
                                    rigidRowOf(dofMap, elements::ElementDof::grid(nodeDof.node, nodeDof.component + 1));
                        }
                        for (std::size_t mode = 0; mode < 6; ++mode)
                            for (std::size_t a = 0; a < m; ++a)
                            {
                                double force = 0.0;
                                for (std::size_t c = 0; c < m; ++c)
                                    force += reduced[a * m + c] * phiRows[c][mode];
                                piece.rigidForce = std::max(piece.rigidForce, std::abs(force));
                            }
                        // The diagonal blocks of each node reached: the translations' and the rotations' 3 × 3.
                        for (std::size_t p0 = 0; p0 < m;)
                        {
                            const linalg::NodeDof first = dofMap.nodeDofOf(nDofs[p0]);
                            std::size_t p1 = p0;
                            while (p1 < m && dofMap.nodeDofOf(nDofs[p1]).node == first.node)
                                ++p1;
                            if (first.node >= dofMap.gridCount())
                            {
                                Contribution scalar;
                                scalar.isScalar = true;
                                scalar.offset = static_cast<std::size_t>(first.node - dofMap.gridCount());
                                scalar.values[0] = reduced[p0 * m + p0];
                                piece.contributions.push_back(scalar);
                            }
                            else
                                for (std::size_t half = 0; half < 2; ++half)
                                {
                                    Contribution block3;
                                    block3.offset = 18 * static_cast<std::size_t>(first.node) + 9 * half;
                                    bool hasTerm = false;
                                    for (std::size_t a = p0; a < p1; ++a)
                                        for (std::size_t c = p0; c < p1; ++c)
                                        {
                                            const auto ia =
                                                static_cast<std::size_t>(dofMap.nodeDofOf(nDofs[a]).component);
                                            const auto ic =
                                                static_cast<std::size_t>(dofMap.nodeDofOf(nDofs[c]).component);
                                            if (ia / 3 != half || ic / 3 != half)
                                                continue;
                                            block3.values[3 * (ia % 3) + ic % 3] += reduced[a * m + c];
                                            hasTerm = true;
                                        }
                                    if (hasTerm)
                                        piece.contributions.push_back(block3);
                                }
                            p0 = p1;
                        }
                    }
                }
            },
            cancellation);
        for (std::size_t p = begin; p < end; ++p)
        {
            largestRigidForce = std::max(largestRigidForce, pieces[p].rigidForce);
            for (const Contribution& contribution : pieces[p].contributions)
            {
                if (contribution.isScalar)
                    scalarDiagonals[contribution.offset] += contribution.values[0];
                else
                    for (std::size_t v = 0; v < 9; ++v)
                        gridBlocks[contribution.offset + v] += contribution.values[v];
            }
            pieces[p].contributions = {};
        }
    }
    base::BudgetedBuffer<float> diagonalOfDof(budget, static_cast<std::size_t>(dofMap.gSetSize()),
                                              "streamed matrix diagonal");
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
        for (int c = 0; c < 6; ++c)
        {
            const std::size_t half = static_cast<std::size_t>(c) / 3;
            const std::size_t k = static_cast<std::size_t>(c) % 3;
            diagonalOfDof[static_cast<std::size_t>(dofMap.gridDof(row, c + 1))] =
                static_cast<float>(gridBlocks[18 * static_cast<std::size_t>(row) + 9 * half + 4 * k]);
        }
    for (std::int32_t s = 0; s < dofMap.scalarCount(); ++s)
        diagonalOfDof[static_cast<std::size_t>(dofMap.scalarDof(s))] =
            static_cast<float>(scalarDiagonals[static_cast<std::size_t>(s)]);
    NodeDiagonals diagonals{assembly::NodeDiagonalBlockSet(std::move(gridBlocks), std::move(scalarDiagonals), dofMap),
                            std::move(diagonalOfDof), largestRigidForce};
    return diagonals;
}

} // namespace oos::solutions
