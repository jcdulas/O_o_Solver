#include "oos/design/sensitivity/ShapeElements.hpp"

#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <utility>

namespace oos::design
{

namespace
{

// The elements of a batch formed per facet call: bounds the matrices held at once.
constexpr std::int64_t ELEMENTS_PER_CALL = 256;
// The columns of the orientation point of the line elements and connectors (CBAR, Beam2: G0; Bushing, Gap2: GO).
constexpr std::array<const char*, 1> ORIENTATION_COLUMNS{"orientationNode"};
// The columns of the coordinate systems an element reads: its own (CBUSH CID and OCID, Gap2 and Bushing1D CID,
// PointMassMatrix and PointMassInertia CID, the MCID of the shells and of Weld) and its property's (the MCID of a
// FastenerProp, the CORDM of a SolidProp).
constexpr std::array<const char*, 3> ELEMENT_FRAME_COLUMNS{"frameId", "offsetFrameId", "matFrameId"};
constexpr std::array<const char*, 2> PROPERTY_FRAME_COLUMNS{"frameId", "matFrameId"};

using BasisSet = std::set<std::size_t>;

// The shape bases that move each system they move (DerivedGeometry), by CID.
[[nodiscard]] std::map<std::int32_t, BasisSet> basesOfSystems(const DerivedGeometry& derived)
{
    std::map<std::int32_t, BasisSet> bases;
    for (const DerivedGeometry::Frame& frame : derived.frames())
        bases[frame.frameId].insert(frame.bases.begin(), frame.bases.end());
    return bases;
}

// The bases of the systems the int32 column `name` of `reader` names, per row (empty sets when it names none).
void addFrameBases(const base::ContractReader& reader, const char* name,
                   const std::map<std::int32_t, BasisSet>& basesOfSystem, std::vector<BasisSet>& bases)
{
    if (!reader.hasColumn(name) || reader.columnType(name) != "int32")
        return;
    const std::span<const std::int32_t> frameIds = reader.values<std::int32_t>(name, base::NullPolicy::Allow);
    const base::ValidityView valid = reader.validity(name);
    for (std::int64_t row = 0; row < reader.rowCount(); ++row)
    {
        if (!valid.isValid(row))
            continue;
        if (const auto found = basesOfSystem.find(frameIds[static_cast<std::size_t>(row)]);
            found != basesOfSystem.end())
            bases[static_cast<std::size_t>(row)].insert(found->second.begin(), found->second.end());
    }
}

// The bases of the moved systems that the properties name, by PID (PIDs are unique across property types).
[[nodiscard]] std::map<std::int32_t, BasisSet> basesOfProperties(const input::ModelSnapshot& snapshot,
                                                                 const std::map<std::int32_t, BasisSet>& basesOfSystem)
{
    std::map<std::int32_t, BasisSet> bases;
    if (basesOfSystem.empty())
        return bases;
    for (const input::CardTable& table : snapshot.propertyTables())
    {
        std::vector<BasisSet> rows(static_cast<std::size_t>(table.reader().rowCount()));
        for (const char* name : PROPERTY_FRAME_COLUMNS)
            addFrameBases(table.reader(), name, basesOfSystem, rows);
        const std::span<const std::int32_t> propIds = table.ids();
        for (std::size_t row = 0; row < rows.size(); ++row)
            if (!rows[row].empty())
                bases[propIds[row]].insert(rows[row].begin(), rows[row].end());
    }
    return bases;
}

// The bases whose moved systems each element of a batch reads, per row.
[[nodiscard]] std::vector<BasisSet> frameBasesOf(const assembly::AssemblyBatch& batch,
                                                 const std::map<std::int32_t, BasisSet>& basesOfSystem,
                                                 const std::map<std::int32_t, BasisSet>& basesOfProperty)
{
    const base::ContractReader& columns = batch.view.columns();
    std::vector<BasisSet> bases(static_cast<std::size_t>(batch.view.rowCount()));
    if (basesOfSystem.empty())
        return bases;
    for (const char* name : ELEMENT_FRAME_COLUMNS)
        addFrameBases(columns, name, basesOfSystem, bases);
    if (!basesOfProperty.empty() && columns.hasColumn("propId") && columns.columnType("propId") == "int32")
    {
        const std::span<const std::int32_t> propIds = columns.values<std::int32_t>("propId", base::NullPolicy::Allow);
        const base::ValidityView valid = columns.validity("propId");
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            if (valid.isValid(row))
                if (const auto found = basesOfProperty.find(propIds[static_cast<std::size_t>(row)]);
                    found != basesOfProperty.end())
                    bases[static_cast<std::size_t>(row)].insert(found->second.begin(), found->second.end());
    }
    return bases;
}

[[nodiscard]] base::Vec3 positionOf(const input::NodeTable& nodes, std::int64_t row)
{
    const std::span<const double> xyz = nodes.positionsInBasic()[row];
    return {xyz[0], xyz[1], xyz[2]};
}

[[nodiscard]] double lengthOf(const std::array<double, 3>& vector)
{
    return std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}

// The node table rows of the grid points whose positions each element of a batch reads (ShapeElements.hpp), sorted
// and distinct, for the elements of rows [first, last).
void collectGeometryRows(const assembly::AssemblyBatch& batch, std::int64_t first, std::int64_t last,
                         std::vector<std::vector<std::int32_t>>& rows)
{
    const elements::FacetSet facets = batch.family->facets();
    elements::ElementMatrixBlock block;
    if (facets.has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, {first, last}, block);
    else if (facets.has(elements::Facet::Mass))
        batch.family->computeMass(batch.view, {first, last}, elements::MassOptions{}, block);
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        std::vector<std::int32_t>& element = rows[static_cast<std::size_t>(block.elementRow(i))];
        for (const elements::ElementDof& dof : block.dofs(i))
            if (dof.kind == input::PointKind::Grid)
                element.push_back(dof.row);
    }
    const base::ContractReader& columns = batch.view.columns();
    for (const char* column : ORIENTATION_COLUMNS)
    {
        if (!columns.hasColumn(column) || columns.columnType(column) != "int32")
            continue;
        const std::span<const std::int32_t> grids = columns.values<std::int32_t>(column, base::NullPolicy::Allow);
        const base::ValidityView valid = columns.validity(column);
        for (std::int64_t row = first; row < last; ++row)
        {
            if (!valid.isValid(row))
                continue;
            if (const std::optional<std::int32_t> gridRow =
                    batch.view.index().points().findGrid(grids[static_cast<std::size_t>(row)]))
                rows[static_cast<std::size_t>(row)].push_back(*gridRow);
        }
    }
    for (std::int64_t row = first; row < last; ++row)
    {
        std::vector<std::int32_t>& element = rows[static_cast<std::size_t>(row)];
        std::sort(element.begin(), element.end());
        element.erase(std::unique(element.begin(), element.end()), element.end());
    }
}

} // namespace

ShapeElements::ShapeElements(const DesignModel& model, const FieldTargets& targets, const assembly::ElementSet& set)
    : m_model(&model), m_snapshot(&set.index().snapshot())
{
    const std::vector<ShapeBasis>& bases = model.shapes().bases();
    const std::vector<ShapeTarget>& shapeTargets = targets.shapeTargets();
    m_elements.resize(bases.size());
    m_steps.assign(bases.size(), 0.0);
    if (bases.empty())
        return;
    // The bases that really move each moved row (a non-zero move).
    const std::vector<std::int64_t>& movedRows = targets.movedRows();
    std::vector<std::vector<std::size_t>> basesOfPoint(movedRows.size());
    std::vector<double> largestMove(bases.size(), 0.0);
    for (std::size_t b = 0; b < bases.size(); ++b)
        for (std::size_t m = 0; m < shapeTargets[b].points.size(); ++m)
        {
            const double move = lengthOf(bases[b].moves[m].move);
            if (move == 0.0)
                continue;
            basesOfPoint[shapeTargets[b].points[m]].push_back(b);
            largestMove[b] = std::max(largestMove[b], move);
        }

    const input::NodeTable& nodes = m_snapshot->nodes();
    constexpr double NO_SIZE = std::numeric_limits<double>::infinity();
    std::vector<double> smallest(bases.size(), NO_SIZE);
    // The systems the bases move: the elements that read one are reached too, and the smallest distance
    // between the grid points that define what a basis changes bounds its step.
    const DerivedGeometry& derived = targets.derived();
    const std::map<std::int32_t, BasisSet> basesOfSystem = basesOfSystems(derived);
    const std::map<std::int32_t, BasisSet> basesOfProperty = basesOfProperties(*m_snapshot, basesOfSystem);
    for (std::size_t b = 0; b < bases.size(); ++b)
        smallest[b] = std::min(smallest[b], derived.smallestSize(b));
    std::vector<std::vector<std::int32_t>> rows; // per element of a batch: bounded by the batch's elements
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        const assembly::AssemblyBatch& batch = set.batches()[b];
        const std::int64_t count = batch.view.rowCount();
        rows.assign(static_cast<std::size_t>(count), {});
        for (std::int64_t first = 0; first < count; first += ELEMENTS_PER_CALL)
            collectGeometryRows(batch, first, std::min(count, first + ELEMENTS_PER_CALL), rows);
        const std::vector<BasisSet> framed = frameBasesOf(batch, basesOfSystem, basesOfProperty);
        for (std::int64_t row = 0; row < count; ++row)
        {
            const std::vector<std::int32_t>& element = rows[static_cast<std::size_t>(row)];
            std::set<std::size_t> reaching = framed[static_cast<std::size_t>(row)];
            for (const std::int32_t gridRow : element)
            {
                const auto found = std::lower_bound(movedRows.begin(), movedRows.end(), gridRow);
                if (found != movedRows.end() && *found == gridRow)
                    for (const std::size_t basis : basesOfPoint[static_cast<std::size_t>(found - movedRows.begin())])
                        reaching.insert(basis);
            }
            if (reaching.empty())
                continue;
            // The size of the element: the largest distance between two of its grid points.
            double size = 0.0;
            for (std::size_t i = 0; i < element.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    size = std::max(size, base::norm(positionOf(nodes, element[i]) - positionOf(nodes, element[j])));
            for (const std::size_t basis : reaching)
            {
                m_elements[basis].push_back({b, row});
                if (size > 0.0)
                    smallest[basis] = std::min(smallest[basis], size);
            }
        }
    }
    // The fallback size: the box of every grid point of the model.
    double modelSize = 0.0;
    if (std::find(smallest.begin(), smallest.end(), NO_SIZE) != smallest.end() && nodes.rowCount() > 0)
    {
        base::Vec3 low = positionOf(nodes, 0);
        base::Vec3 high = low;
        for (std::int64_t row = 1; row < nodes.rowCount(); ++row)
        {
            const base::Vec3 point = positionOf(nodes, row);
            for (std::size_t k = 0; k < 3; ++k)
            {
                low[k] = std::min(low[k], point[k]);
                high[k] = std::max(high[k], point[k]);
            }
        }
        modelSize = base::norm(high - low);
    }
    for (std::size_t b = 0; b < bases.size(); ++b)
    {
        if (largestMove[b] == 0.0)
            continue;
        if (smallest[b] != NO_SIZE)
            m_steps[b] = SHAPE_STEP_RELATIVE * smallest[b] / largestMove[b];
        else if (modelSize > 0.0)
            m_steps[b] = SHAPE_STEP_RELATIVE * modelSize / largestMove[b];
        else
            m_steps[b] = SHAPE_STEP_AT_ZERO_SIZE / largestMove[b];
    }
}

std::vector<DeformedRigidElement> ShapeElements::deformedRigidElements(std::size_t basis) const
{
    std::vector<DeformedRigidElement> deformed;
    if (!m_snapshot->hasTable(input::TableRole::RigidElements))
        return deformed;
    // The move of each grid point of the basis.
    std::map<std::int32_t, std::array<double, 3>> moveOf;
    for (const ShapeMove& move : m_model->shapes().bases()[basis].moves)
        moveOf[move.gridId] = move.move;
    const input::RigidTable& rigid = m_snapshot->rigidElements();
    const std::span<const std::int32_t> elementIds = rigid.ids();
    const base::DictionaryView types = rigid.types();
    const std::span<const std::int32_t> nodeIds = rigid.nodeIds();
    const auto moveAt = [&](std::int64_t row)
    {
        const auto found = moveOf.find(nodeIds[static_cast<std::size_t>(row)]);
        return found == moveOf.end() ? std::array<double, 3>{} : found->second;
    };
    // The rows of the rigid elements that hold a grid point of the basis; an element is deformed when two of its grid
    // points move by different vectors (a grid point the basis does not name moves by zero).
    std::map<std::pair<std::string, std::int32_t>, std::vector<std::int64_t>> touched;
    for (std::int64_t row = 0; row < rigid.rowCount(); ++row)
        if (moveOf.count(nodeIds[static_cast<std::size_t>(row)]) != 0)
            touched[{std::string(types[row]), elementIds[static_cast<std::size_t>(row)]}];
    for (std::int64_t row = 0; row < rigid.rowCount() && !touched.empty(); ++row)
        if (const auto found = touched.find({std::string(types[row]), elementIds[static_cast<std::size_t>(row)]});
            found != touched.end())
            found->second.push_back(row);
    for (const auto& [element, rows] : touched)
    {
        const std::array<double, 3> reference = moveAt(rows.front());
        for (const std::int64_t row : rows)
            if (moveAt(row) != reference)
            {
                const bool isFirstMoving = reference != std::array<double, 3>{};
                deformed.push_back({element.first, element.second,
                                    nodeIds[static_cast<std::size_t>(isFirstMoving ? rows.front() : row)]});
                break;
            }
    }
    return deformed;
}

} // namespace oos::design
