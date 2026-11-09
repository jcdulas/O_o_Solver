#include "oos/assembly/assemble/ElementVolumes.hpp"

#include "oos/base/threading/ParallelFor.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

// The elements whose mass matrices are computed at once: a bounded block.
constexpr std::int64_t RANGE_ELEMENTS = 2048;

// The x component of the basic system at one DOF of an element: column c of the displacement frame R
// (u_basic = R·u_CD), zero at a rotation or a scalar point.
double basicXOf(const elements::ElementBatchView& view, const elements::ElementDof& dof)
{
    if (dof.kind != input::PointKind::Grid || dof.component < 1 || dof.component > 3)
        return 0.0;
    return view.displacementFrame(dof.row)(0, static_cast<std::size_t>(dof.component - 1));
}

// The batch of `snapshot` that is the `occurrence`-th batch of card `card`, or null.
const input::ElementBatch* batchOfCard(const input::ModelSnapshot& snapshot, const std::string& card,
                                       std::size_t occurrence)
{
    for (const input::ElementBatch& batch : snapshot.elementBatches())
    {
        if (batch.card() != card)
            continue;
        if (occurrence == 0)
            return &batch;
        --occurrence;
    }
    return nullptr;
}

} // namespace

ElementVolumes::ElementVolumes(const ElementSet& set, std::span<const std::uint8_t> isBatchWanted,
                               base::MemoryBudget& budget, base::ThreadPool* pool)
    : m_volumes(set.batches().size())
{
    // A volume is that of a structural element, which has a stiffness and a mass: not of a concentrated mass (its
    // mass is not a density times a volume) nor of a spring.
    const auto isWanted = [&](std::size_t b)
    {
        const elements::FacetSet facets = set.batches()[b].family->facets();
        return facets.has(elements::Facet::Mass) && facets.has(elements::Facet::Stiffness) &&
               (isBatchWanted.empty() || isBatchWanted[b] != 0);
    };
    bool hasWanted = false;
    for (std::size_t b = 0; b < set.batches().size(); ++b)
        hasWanted = hasWanted || isWanted(b);
    if (!hasWanted)
        return;
    // The base snapshot outlives the unit-density one, which lives only in this constructor: no keepAlive needed.
    const input::ModelSnapshot unitDensity = unitDensitySnapshot(set.index().snapshot(), {});
    const elements::SnapshotIndex index(unitDensity, budget);
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        if (!isWanted(b))
            continue;
        const AssemblyBatch& base = set.batches()[b];
        std::size_t occurrence = 0;
        for (std::size_t previous = 0; previous < b; ++previous)
            occurrence += set.batches()[previous].view.card() == base.view.card() ? 1 : 0;
        const input::ElementBatch* unitBatch = batchOfCard(unitDensity, base.view.card(), occurrence);
        if (unitBatch == nullptr || unitBatch->rowCount() != base.view.rowCount())
            continue; // an overlay keeps every table: unreachable
        const elements::ElementBatchView view(*unitBatch, index);
        m_volumes[b] =
            base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(view.rowCount()), "element volumes");
        elements::MassOptions options;
        options.formulation = elements::MassFormulation::Lumped;
        const auto ranges = static_cast<std::size_t>((view.rowCount() + RANGE_ELEMENTS - 1) / RANGE_ELEMENTS);
        base::parallelFor(pool, ranges, 1,
                          [&](std::size_t firstRange, std::size_t endRange)
                          {
                              elements::ElementMatrixBlock block;
                              std::vector<double> direction; // bounded by the DOFs of one element
                              for (std::size_t r = firstRange; r < endRange; ++r)
                              {
                                  const auto first = static_cast<std::int64_t>(r) * RANGE_ELEMENTS;
                                  block.clear();
                                  base.family->computeMass(
                                      view, {first, std::min(view.rowCount(), first + RANGE_ELEMENTS)}, options, block);
                                  for (std::size_t e = 0; e < block.size(); ++e)
                                  {
                                      const std::span<const elements::ElementDof> dofs = block.dofs(e);
                                      const std::span<const double> matrix = block.matrix(e);
                                      const std::size_t n = dofs.size();
                                      direction.resize(n);
                                      for (std::size_t i = 0; i < n; ++i)
                                          direction[i] = basicXOf(view, dofs[i]);
                                      double mass = 0.0;
                                      for (std::size_t i = 0; i < n; ++i)
                                      {
                                          if (direction[i] == 0.0)
                                              continue;
                                          for (std::size_t j = 0; j < n; ++j)
                                              mass += direction[i] * matrix[i * n + j] * direction[j];
                                      }
                                      m_volumes[b][static_cast<std::size_t>(block.elementRow(e))] = mass;
                                  }
                              }
                          });
    }
}

input::ModelSnapshot ElementVolumes::unitDensitySnapshot(const input::ModelSnapshot& snapshot,
                                                         const std::shared_ptr<const void>& keepAlive)
{
    input::ColumnOverlay overlay;
    for (const std::string& name : snapshot.tableNames())
    {
        const bool isMaterial = name == "materialsTable" || name.rfind("materials/", 0) == 0;
        const bool isPropertyOrElement = name.rfind("properties/", 0) == 0 || name.rfind("elements/", 0) == 0;
        if (!isMaterial && !isPropertyOrElement)
            continue;
        const base::ContractReader* table = snapshot.findTable(name);
        if (table == nullptr)
            continue;
        for (const auto& [column, value] :
             std::array<std::pair<const char*, double>, 2>{{{"density", 1.0}, {"nonStructMass", 0.0}}})
        {
            if (!table->hasColumn(column))
                continue;
            const std::string type = table->columnType(column);
            const std::string fixedList = "fixed_list<float64>[";
            std::int32_t items = 0;
            if (type.rfind(fixedList, 0) == 0)
                items = std::stoi(type.substr(fixedList.size()));
            else if (type != "float64")
                continue;
            for (std::int64_t row = 0; row < table->rowCount(); ++row)
            {
                if (items == 0)
                    overlay.setValue(name, column, {row, input::CellValue::SCALAR_ITEM, value});
                for (std::int32_t item = 0; item < items; ++item)
                    overlay.setValue(name, column, {row, item, value});
            }
        }
    }
    return overlay.applyTo(snapshot, keepAlive);
}

} // namespace oos::assembly
