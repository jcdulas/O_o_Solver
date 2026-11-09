#include "oos/design/responses/weight_volume/MassProperties.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/ElementVolumes.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

// The elements whose mass matrices are computed at once: a bounded block (the element matrices of a range live in an
// ElementMatrixBlock sized by the range).
constexpr std::int64_t RANGE_ELEMENTS = 2048;

constexpr std::array<std::string_view, 6> CONCENTRATED_MASS_CARDS{"PointMassMatrix", "PointMassInertia", "MassDofProperty",
                                                                  "MassDofInline",  "MassScalarProperty",  "MassScalarInline"};

// The six rows of the rigid-body modes at one DOF (MassProperties class comment).
std::array<double, 6> rigidRow(const elements::ElementBatchView& view, const elements::ElementDof& dof,
                               base::Vec3 reference)
{
    std::array<double, 6> row{};
    if (dof.kind != input::PointKind::Grid || dof.component < 1 || dof.component > 6)
        return row;
    const base::Mat3 rotation = view.displacementFrame(dof.row);
    const base::Vec3 position = view.positionInBasic(dof.row);
    const base::Vec3 arm{position[0] - reference[0], position[1] - reference[1], position[2] - reference[2]};
    // The direction of the component in the basic system: column c of R (u_basic = R·u_CD, so u_CD,c = R(:, c)·u).
    const std::size_t c = static_cast<std::size_t>(dof.component - 1) % 3;
    const base::Vec3 direction{rotation(0, c), rotation(1, c), rotation(2, c)};
    if (dof.component <= 3)
    {
        // u = t + θ × arm: the component is direction·t + direction·(θ × arm) = direction·t + θ·(arm × direction).
        row[0] = direction[0];
        row[1] = direction[1];
        row[2] = direction[2];
        row[3] = arm[1] * direction[2] - arm[2] * direction[1];
        row[4] = arm[2] * direction[0] - arm[0] * direction[2];
        row[5] = arm[0] * direction[1] - arm[1] * direction[0];
    }
    else
    {
        row[3] = direction[0];
        row[4] = direction[1];
        row[5] = direction[2];
    }
    return row;
}

std::int32_t propertyOf(const input::ElementBatch& batch, std::int64_t row)
{
    if (!batch.hasProperty())
        return 0;
    return batch.propertyIds()[static_cast<std::size_t>(row)];
}

// MO_e(p, q) = Σ_ij D(i, p)·M(i, j)·D(j, q) of element `e` of `block`, in a fixed order.
base::MatN<6> rigidBodyMassOfElement(const elements::ElementBatchView& view, const elements::ElementMatrixBlock& block,
                                     std::size_t e, base::Vec3 reference, std::vector<std::array<double, 6>>& rows)
{
    const std::span<const elements::ElementDof> dofs = block.dofs(e);
    const std::span<const double> matrix = block.matrix(e);
    const std::size_t n = dofs.size();
    rows.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        rows[i] = rigidRow(view, dofs[i], reference);
    base::MatN<6> element;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            const double m = matrix[i * n + j];
            if (m == 0.0)
                continue;
            for (std::size_t p = 0; p < 6; ++p)
            {
                const double left = rows[i][p] * m;
                if (left == 0.0)
                    continue;
                for (std::size_t q = 0; q < 6; ++q)
                    element(p, q) += left * rows[j][q];
            }
        }
    return element;
}

void addTo(base::MatN<6>& total, const base::MatN<6>& element)
{
    for (std::size_t p = 0; p < 6; ++p)
        for (std::size_t q = 0; q < 6; ++q)
            total(p, q) += element(p, q);
}

// Σ D_eᵀ·M_e·D_e over the listed elements of `set` (increasing (batch, row)), each computed alone with the family of
// `registry` for its card: the terms of sumRigidBodyMass for these elements, in its order.
base::MatN<6> sumListedRigidBodyMass(const assembly::ElementSet& set,
                                     std::span<const MassProperties::SetElement> elements,
                                     const elements::ElementRegistry& registry, base::Vec3 reference,
                                     elements::MassFormulation formulation, bool skipsConcentratedMasses)
{
    base::MatN<6> total;
    elements::ElementMatrixBlock block;
    std::vector<std::array<double, 6>> rows; // bounded by the DOFs of one element
    elements::MassOptions options;
    options.formulation = formulation;
    const elements::ElementFamily* family = nullptr;
    std::size_t familyBatch = set.batches().size();
    for (const MassProperties::SetElement& element : elements)
    {
        const assembly::AssemblyBatch& batch = set.batches()[element.batch];
        if (element.batch != familyBatch)
        {
            familyBatch = element.batch;
            family = registry.find(batch.view.card());
            if (family != nullptr &&
                (!family->facets().has(elements::Facet::Mass) ||
                 (skipsConcentratedMasses && MassProperties::isConcentratedMass(batch.view.card()))))
                family = nullptr;
        }
        if (family == nullptr)
            continue;
        block.clear();
        family->computeMass(batch.view, {element.row, element.row + 1}, options, block);
        for (std::size_t e = 0; e < block.size(); ++e)
            addTo(total, rigidBodyMassOfElement(batch.view, block, e, reference, rows));
    }
    return total;
}

// Σ D_eᵀ·M_e·D_e over the elements `filter` takes; the entry `perElementEntry` of each element's matrix (the x
// translational mass by default) is added to `perElement` when it is given.
base::MatN<6> sumRigidBodyMass(const input::ModelSnapshot& snapshot, base::Vec3 reference,
                               elements::MassFormulation formulation, const ElementFilter& filter,
                               bool skipsConcentratedMasses, base::MemoryBudget& budget,
                               std::map<std::int32_t, double>* perElement = nullptr, const CardFilter& takesCard = {},
                               std::pair<std::size_t, std::size_t> perElementEntry = {0, 0})
{
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const elements::SnapshotIndex index(snapshot, budget);
    const assembly::ElementSet set(index, registry);
    base::MatN<6> total;
    elements::ElementMatrixBlock block;
    std::vector<std::array<double, 6>> rows; // bounded by the DOFs of one element
    for (const std::size_t b : set.batchesWith(elements::Facet::Mass))
    {
        const assembly::AssemblyBatch& batch = set.batches()[b];
        if (skipsConcentratedMasses && MassProperties::isConcentratedMass(batch.view.card()))
            continue;
        if (takesCard && !takesCard(batch.view.card()))
            continue;
        const input::ElementBatch& elementBatch = batch.view.batch();
        elements::MassOptions options;
        options.formulation = formulation;
        for (std::int64_t first = 0; first < batch.view.rowCount(); first += RANGE_ELEMENTS)
        {
            const std::int64_t last = std::min(batch.view.rowCount(), first + RANGE_ELEMENTS);
            block.clear();
            batch.family->computeMass(batch.view, {first, last}, options, block);
            for (std::size_t e = 0; e < block.size(); ++e)
            {
                const std::int64_t row = block.elementRow(e);
                if (filter && !filter(batch.view.card(), batch.view.elementId(row), propertyOf(elementBatch, row)))
                    continue;
                const base::MatN<6> element = rigidBodyMassOfElement(batch.view, block, e, reference, rows);
                addTo(total, element);
                if (perElement != nullptr)
                    (*perElement)[batch.view.elementId(row)] += element(perElementEntry.first, perElementEntry.second);
            }
        }
    }
    return total;
}

} // namespace

bool MassProperties::isConcentratedMass(std::string_view card) noexcept
{
    return std::find(CONCENTRATED_MASS_CARDS.begin(), CONCENTRATED_MASS_CARDS.end(), card) !=
           CONCENTRATED_MASS_CARDS.end();
}

base::MatN<6> MassProperties::rigidBodyMass(const input::ModelSnapshot& snapshot, base::Vec3 reference,
                                            elements::MassFormulation formulation, const ElementFilter& filter,
                                            base::MemoryBudget& budget)
{
    return sumRigidBodyMass(snapshot, reference, formulation, filter, false, budget);
}

double MassProperties::volume(const input::ModelSnapshot& snapshot, const std::shared_ptr<const void>& keepAlive,
                              const ElementFilter& filter, base::MemoryBudget& budget)
{
    const input::ModelSnapshot unitDensity = assembly::ElementVolumes::unitDensitySnapshot(snapshot, keepAlive);
    // The x translational mass of the rigid-body mass matrix is the total mass, here the volume.
    return sumRigidBodyMass(unitDensity, {0.0, 0.0, 0.0}, elements::MassFormulation::Lumped, filter, true, budget)(0,
                                                                                                                   0);
}

std::map<std::int32_t, double> MassProperties::elementVolumes(const input::ModelSnapshot& snapshot,
                                                              const std::shared_ptr<const void>& keepAlive,
                                                              const ElementFilter& filter, base::MemoryBudget& budget)
{
    const input::ModelSnapshot unitDensity = assembly::ElementVolumes::unitDensitySnapshot(snapshot, keepAlive);
    std::map<std::int32_t, double> volumes;
    (void)sumRigidBodyMass(unitDensity, {0.0, 0.0, 0.0}, elements::MassFormulation::Lumped, filter, true, budget,
                           &volumes);
    return volumes;
}

std::map<std::int32_t, double> MassProperties::elementRigidBodyMass(const input::ModelSnapshot& snapshot,
                                                                    base::Vec3 reference,
                                                                    elements::MassFormulation formulation,
                                                                    const ElementFilter& filter, std::size_t row,
                                                                    std::size_t column, base::MemoryBudget& budget)
{
    std::map<std::int32_t, double> entries;
    (void)sumRigidBodyMass(snapshot, reference, formulation, filter, false, budget, &entries, {}, {row, column});
    return entries;
}

base::MatN<6> MassProperties::rigidBodyMassOf(const assembly::ElementSet& set, std::span<const SetElement> elements,
                                              const elements::ElementRegistry& registry, base::Vec3 reference,
                                              elements::MassFormulation formulation)
{
    return sumListedRigidBodyMass(set, elements, registry, reference, formulation, false);
}

double MassProperties::volumeOf(const assembly::ElementSet& set, const std::shared_ptr<const void>& keepAlive,
                                std::span<const SetElement> elements, const elements::ElementRegistry& registry)
{
    // The unit densities change no ID: the index of `set` serves the overlay, and its batches are the same.
    const auto unitDensity = std::make_shared<const input::ModelSnapshot>(
        assembly::ElementVolumes::unitDensitySnapshot(set.index().snapshot(), keepAlive));
    const elements::SnapshotIndex index(*unitDensity, set.index());
    const assembly::ElementSet unitSet(index, registry);
    return sumListedRigidBodyMass(unitSet, elements, registry, {0.0, 0.0, 0.0}, elements::MassFormulation::Lumped,
                                  true)(0, 0);
}

std::map<std::int32_t, double> MassProperties::elementVolumesOfCards(const input::ModelSnapshot& snapshot,
                                                                     const std::shared_ptr<const void>& keepAlive,
                                                                     const CardFilter& takesCard,
                                                                     base::MemoryBudget& budget)
{
    const input::ModelSnapshot unitDensity = assembly::ElementVolumes::unitDensitySnapshot(snapshot, keepAlive);
    std::map<std::int32_t, double> volumes;
    (void)sumRigidBodyMass(unitDensity, {0.0, 0.0, 0.0}, elements::MassFormulation::Lumped, {}, true, budget, &volumes,
                           takesCard);
    return volumes;
}

} // namespace oos::design
