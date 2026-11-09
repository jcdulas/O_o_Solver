#include "oos/output/volume/OutputVolumeEstimate.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OutputVolumeError.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"

#include <algorithm>
#include <vector>

namespace oos::output
{

namespace
{

// The bytes of the labels of a row: element or point ID, grid point, layer (int32 each) and a position (float64).
constexpr std::uint64_t ELEMENT_LABEL_BYTES = 3 * sizeof(std::int32_t) + sizeof(double);
constexpr std::uint64_t NODAL_LABEL_BYTES = sizeof(std::int32_t) + sizeof(std::int8_t);
constexpr std::uint64_t NODAL_COMPONENTS = 6;
// The fixed rows of a grid point in the grid point forces: appliedLoad, singlePointConstrForce or
// multiPointConstrForce, total.
constexpr std::uint64_t GRID_POINT_FORCE_FIXED_ROWS = 3;
constexpr std::uint64_t STRAIN_ENERGY_COMPONENTS = 3;
constexpr std::size_t STRESS_SHAPE = 0;
constexpr std::size_t STRAIN_SHAPE = 1;
constexpr std::size_t FIBER_STRAIN_SHAPE = 2;
constexpr std::size_t FORCE_SHAPE = 3;
constexpr std::size_t GRID_SHAPE = 4;

std::string megabytes(std::uint64_t bytes)
{
    if (bytes < (1ULL << 20))
        return std::to_string(bytes) + " bytes";
    const std::uint64_t tenths = (bytes * 10 + (1ULL << 19)) >> 20;
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " MB";
}

} // namespace

OutputVolumeEstimate::OutputVolumeEstimate(const assembly::ElementSet& set, const assembly::DofMap& dofMap)
    : m_set(&set), m_dofMap(&dofMap), m_shapes(set.batches().size())
{
}

std::uint64_t OutputVolumeEstimate::totalBytes() const noexcept
{
    std::uint64_t total = 0;
    for (const std::uint64_t bytes : m_bytes)
        total += bytes;
    return total;
}

OutputVolumeEstimate::RowShape OutputVolumeEstimate::shapeOf(std::size_t batch, OutputKind kind, bool isFiber)
{
    const std::size_t slot = kind == OutputKind::Stress   ? STRESS_SHAPE
                             : kind == OutputKind::Strain ? (isFiber ? FIBER_STRAIN_SHAPE : STRAIN_SHAPE)
                             : kind == OutputKind::Force  ? FORCE_SHAPE
                                                          : GRID_SHAPE;
    RowShape& shape = m_shapes[batch][slot];
    if (shape.isKnown)
        return shape;
    shape.isKnown = true;
    const assembly::AssemblyBatch& entry = m_set->batches()[batch];
    const elements::ElementFamily& family = *entry.family;
    if (entry.view.rowCount() == 0)
        return shape;
    const elements::ElementRange first{0, 1};
    if (slot == GRID_SHAPE)
    {
        if (family.facets().has(elements::Facet::Stiffness))
        {
            elements::ElementMatrixBlock block;
            family.computeStiffness(entry.view, first, block);
            std::vector<std::int32_t> grids; // the grid points of one element
            for (std::size_t i = 0; i < block.size(); ++i)
                for (const elements::ElementDof& dof : block.dofs(i))
                    if (std::find(grids.begin(), grids.end(), dof.row) == grids.end())
                        grids.push_back(dof.row);
            shape.grids = grids.size();
        }
        return shape;
    }
    if (!family.facets().has(elements::Facet::Recovery))
        return shape;
    elements::ElementOutput output = kind == OutputKind::Stress   ? elements::ElementOutput::Stresses
                                     : kind == OutputKind::Strain ? elements::ElementOutput::Strains
                                                                  : elements::ElementOutput::Forces;
    if (output == elements::ElementOutput::Strains && !isFiber &&
        !family.outputColumns(elements::ElementOutput::StrainCurvatures).empty())
        output = elements::ElementOutput::StrainCurvatures;
    const std::vector<double> zeros(static_cast<std::size_t>(m_dofMap->gSetSize()), 0.0);
    const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    const elements::NodalDisplacements displacements{std::span<const double>(zeros).first(gridValues),
                                                     std::span<const double>(zeros).subspan(gridValues), nullptr};
    if (!family.outputColumns(output).empty())
    {
        elements::ElementResultBlock block;
        family.recover(entry.view, first, output, displacements, block);
        shape.rows += block.size();
        shape.columns = std::max<std::uint64_t>(shape.columns, block.columns().size());
    }
    const elements::LayerOutput layerOutput =
        kind == OutputKind::Strain ? elements::LayerOutput::Strains : elements::LayerOutput::Stresses;
    if (kind != OutputKind::Force && !family.layerOutputColumns(layerOutput).empty())
    {
        elements::ElementResultBlock block;
        family.recoverLayers(entry.view, first, layerOutput, displacements, block);
        shape.rows += block.size();
        shape.columns = std::max<std::uint64_t>(shape.columns, block.columns().size());
    }
    return shape;
}

std::uint64_t OutputVolumeEstimate::cachedBytesOf(OutputKind kind, const OutputRequestPlan& request)
{
    const auto k = static_cast<std::size_t>(kind);
    const OutputRequestPlan& last = m_lastRequests[k];
    if (m_hasLast[k] && last.isAll == request.isAll && last.isFiber == request.isFiber &&
        last.members == request.members)
        return m_lastBytes[k];
    m_lastBytes[k] = bytesOf(kind, request);
    m_lastRequests[k] = request;
    m_hasLast[k] = true;
    return m_lastBytes[k];
}

std::uint64_t OutputVolumeEstimate::bytesOf(OutputKind kind, const OutputRequestPlan& request)
{
    if (kind == OutputKind::Displacement || kind == OutputKind::SpcForces || kind == OutputKind::MpcForces ||
        kind == OutputKind::AppliedLoads || kind == OutputKind::GridPointForces)
    {
        std::uint64_t points = 0;
        for (std::int32_t grid = 0; grid < m_dofMap->gridCount(); ++grid)
            points += request.takes(m_dofMap->gridId(grid)) ? 1 : 0;
        for (std::int32_t scalar = 0; scalar < m_dofMap->scalarCount(); ++scalar)
            points += request.takes(m_dofMap->scalarId(scalar)) ? 1 : 0;
        if (kind != OutputKind::GridPointForces)
            return points * (NODAL_COMPONENTS * sizeof(double) + NODAL_LABEL_BYTES);
        // One row per element and grid point, the elements' grid points counted on all elements (an upper bound when
        // the request is a SET of grid points).
        std::uint64_t elementRows = 0;
        for (std::size_t b = 0; b < m_set->batches().size(); ++b)
            elementRows +=
                shapeOf(b, kind, false).grids * static_cast<std::uint64_t>(m_set->batches()[b].view.rowCount());
        if (!request.isAll)
            elementRows = std::min<std::uint64_t>(elementRows, points * GRID_POINT_FORCE_FIXED_ROWS * 4);
        return (elementRows + points * GRID_POINT_FORCE_FIXED_ROWS) *
               (NODAL_COMPONENTS * sizeof(double) + ELEMENT_LABEL_BYTES);
    }
    std::uint64_t bytes = 0;
    for (std::size_t b = 0; b < m_set->batches().size(); ++b)
    {
        const elements::ElementBatchView& view = m_set->batches()[b].view;
        std::uint64_t elements = 0;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
            elements += request.takes(view.elementId(row)) ? 1 : 0;
        if (elements == 0)
            continue;
        if (kind == OutputKind::StrainEnergy)
        {
            if (m_set->batches()[b].family->facets().has(elements::Facet::Stiffness))
                bytes += elements * (STRAIN_ENERGY_COMPONENTS * sizeof(double) + ELEMENT_LABEL_BYTES);
            continue;
        }
        const RowShape shape = shapeOf(b, kind, request.isFiber);
        bytes += elements * shape.rows * (shape.columns * sizeof(double) + ELEMENT_LABEL_BYTES);
    }
    return bytes;
}

void OutputVolumeEstimate::requireWithin(std::uint64_t limitBytes) const
{
    if (totalBytes() <= limitBytes)
        return;
    refuse(limitBytes, " the results may take in memory",
           "write the results to files batch by batch, or give the analysis more memory");
}

void OutputVolumeEstimate::requireWithinDestination(std::uint64_t limitBytes) const
{
    if (totalBytes() <= limitBytes)
        return;
    refuse(limitBytes, " of free space on the disk of the result files",
           "free some disk space, or write the files on another disk");
}

void OutputVolumeEstimate::refuse(std::uint64_t limitBytes, std::string_view destination,
                                  std::string_view lastRemedy) const
{
    const std::uint64_t total = totalBytes();
    std::vector<std::size_t> order(OUTPUT_KIND_COUNT);
    for (std::size_t k = 0; k < order.size(); ++k)
        order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return m_bytes[a] > m_bytes[b]; });
    std::string largest;
    for (std::size_t i = 0; i < order.size() && m_bytes[order[i]] > 0 && i < 3; ++i)
        largest.append(largest.empty() ? "" : ", ")
            .append(requestNameOf(static_cast<OutputKind>(order[i])))
            .append(" ")
            .append(megabytes(m_bytes[order[i]]));
    const std::string topRequest(requestNameOf(static_cast<OutputKind>(order[0])));
    std::vector<std::string> remedies{
        base::concatenate({"restrict the output by a SetDefinition (", topRequest,
                           " = 10 with SetDefinition 10 = the points or elements "
                           "you need) or switch requests off (",
                           topRequest, " = NONE)"}),
        "ask for envelopes instead of every load case (for example the largest von Mises stress and the largest "
        "displacement magnitude), without the tables of each case",
        std::string(lastRemedy)};
    std::string message =
        base::concatenate({"the requested outputs need about ", megabytes(total), " for ", std::to_string(m_caseCount),
                           " load case", m_caseCount == 1 ? "" : "s", " (", largest, "), more than the ",
                           megabytes(limitBytes), destination, "; nothing was computed. What to do:"});
    for (const std::string& remedy : remedies)
        message.append("\n  - ").append(remedy);
    throw base::OutputVolumeError(message, total, remedies);
}

} // namespace oos::output
