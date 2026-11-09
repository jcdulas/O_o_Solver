#include "oos/design/sensitivity/ElementItemFunction.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/responses/composite/StrengthRatio.hpp"
#include "oos/design/responses/weight_volume/MassProperties.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>

namespace oos::design
{

namespace
{

constexpr std::size_t NO_ROW = std::numeric_limits<std::size_t>::max();
// The column of an item its output does not have (a strain energy, which no recovered output holds).
constexpr std::size_t NO_COLUMN = std::numeric_limits<std::size_t>::max();
// Item 4 of an ESE output is the element strain energy density.
constexpr std::int32_t STRAIN_ENERGY_DENSITY_ITEM = 4;
// The positions of the ends of a bar or beam in its station rows (ResultLookup: END_A_POSITION, END_B_POSITION).
constexpr double END_A_POSITION = 0.0;
constexpr double END_B_POSITION = 1.0;

// The row of a block at a place and layer, as ResultLookup::elementRow finds it in a table: the centre is grid point
// 0 of a block without stations, an end the station at position 0 or 1; layer 0 takes the first row of the place.
std::size_t findRow(const elements::ElementResultBlock& block, ItemPlace place, std::int32_t layer)
{
    bool hasPositions = false;
    for (std::size_t i = 0; i < block.size(); ++i)
        hasPositions = hasPositions || block.place(i).position != 0.0;
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        const elements::ResultPlace where = block.place(i);
        const bool isPlace =
            place == ItemPlace::Centre
                ? !hasPositions && where.nodeId == 0
                : hasPositions && where.position == (place == ItemPlace::EndA ? END_A_POSITION : END_B_POSITION);
        if (isPlace && (layer == 0 || where.layer == layer))
            return i;
    }
    return NO_ROW;
}

// The output kind of the family a table kind reads (ElementResultBuilder: the strains as the membrane strains and
// curvatures when the family gives them).
elements::ElementOutput outputOf(const elements::ElementFamily& family, const std::string& tableKind)
{
    if (tableKind == "forces")
        return elements::ElementOutput::Forces;
    if (tableKind == "strain")
        return family.outputColumns(elements::ElementOutput::StrainCurvatures).empty()
                   ? elements::ElementOutput::Strains
                   : elements::ElementOutput::StrainCurvatures;
    return elements::ElementOutput::Stresses;
}

// The table kind of the ply states (LayerOutput::PlyStates: σ1, σ2, τ12, the mechanical ε1, ε2, γ12, τ1z, τ2z) and
// the state's columns the indices read.
const std::string PLY_STATES = "plyStates";
constexpr std::size_t PLY_STATE_WEIGHTS = 8;

elements::LayerOutput layerOutputOf(const std::string& tableKind)
{
    if (tableKind == "strain")
        return elements::LayerOutput::Strains;
    if (tableKind == "failureIndices")
        return elements::LayerOutput::FailureIndices;
    if (tableKind == PLY_STATES)
        return elements::LayerOutput::PlyStates;
    return elements::LayerOutput::Stresses;
}

// A ply's failure index or bond index (CFAILURE items of a laminate's ply): a function of the ply's state (its
// stresses, mechanical strains and transverse shears), whose slopes in the displacements are those of the state
// chained with the index's gradient.
[[nodiscard]] bool isPlyIndexItem(const ElementItemSpec& item)
{
    return item.location.isComposite && item.location.tableKind == "failureIndices" && !item.location.isStrengthRatio &&
           (item.location.column == FAILURE_INDEX_COLUMN || item.location.column == BOND_FAILURE_INDEX_COLUMN);
}

[[noreturn]] void refuse(const elements::ElementBatchView& view, std::int64_t row, const ElementItemSpec& item,
                         const std::string& what)
{
    throw base::OptimisationError(base::concatenate(
        {"the sensitivity of item ", std::to_string(item.code), " of ", responseTypeName(item.type), " on ",
         view.card(), " ", std::to_string(view.elementId(row)),
         item.lamina != 0 ? std::string(", lamina ").append(std::to_string(item.lamina)) : std::string(), ": ", what,
         "; correct the element (its geometry, property or laminate) or leave it out of the response"}));
}

// The in-plane components of a shell's stress and strain rows (ShellRecovery's columns): the centre rows (fibre
// distance or curvature first) and the ply rows.
struct PlaneColumns
{
    std::array<const char*, 3> names;
    double shearFactor;
    double scale;
};
// The equivalent (von Mises) strain: ⅔·√(e1² − e1·e2 + e2²) of the principal strains of (εx, εy, γxy/2).
constexpr double STRAIN_VON_MISES_SCALE = 2.0 / 3.0;
constexpr std::array<PlaneColumns, 4> PLANE_COLUMNS{{{{"oxx", "oyy", "txy"}, 1.0, 1.0},
                                                     {{"exx", "eyy", "exy"}, 0.5, STRAIN_VON_MISES_SCALE},
                                                     {{"o11", "o22", "t12"}, 1.0, 1.0},
                                                     {{"e11", "e22", "e12"}, 0.5, STRAIN_VON_MISES_SCALE}}};
// The columns linear in u (affine under a thermal strain) of the shells' and plies' stress and strain rows (with the
// in-plane components above) and of the rods' (RodFormulation: their margins are not).
constexpr std::array<const char*, 9> SHELL_AFFINE_COLUMNS{
    "fiberDistance", "fiberCurvature", "t1z", "t2z", "e1z", "e2z", "oxx", "oyy", "txy"};
constexpr std::array<const char*, 2> ROD_AFFINE_COLUMNS{"axial", "torsion"};

[[nodiscard]] bool isNamed(const std::string& name, std::span<const char* const> names)
{
    return std::any_of(names.begin(), names.end(), [&](const char* candidate) { return name == candidate; });
}

// The form of the item at `column` of row `index` (ItemForm), from the columns of its block: a shell's or ply's rows
// hold the in-plane components of one of PLANE_COLUMNS and a "vonMises" column, a rod's are those of RodFormulation.
ItemForm formOf(const elements::ElementResultBlock& block, std::size_t index, std::size_t column)
{
    const std::vector<std::string>& names = block.columns();
    if (column >= names.size())
        return {};
    const std::string& name = names[column];
    const bool isRod = (names.size() == 2 || names.size() == 4) && names[0] == "axial";
    if (isRod)
        return isNamed(name, ROD_AFFINE_COLUMNS) ? ItemForm{ItemForm::Kind::Affine} : ItemForm{};
    if (std::find(names.begin(), names.end(), "vonMises") == names.end())
        return {};
    for (const PlaneColumns& plane : PLANE_COLUMNS)
    {
        std::array<std::size_t, 3> at{};
        bool hasAll = true;
        for (std::size_t k = 0; k < 3 && hasAll; ++k)
        {
            const auto found = std::find(names.begin(), names.end(), plane.names[k]);
            hasAll = found != names.end();
            at[k] = static_cast<std::size_t>(found - names.begin());
        }
        if (!hasAll)
            continue;
        if (name == "vonMises")
        {
            const std::span<const double> row = block.values(index);
            return {
                ItemForm::Kind::PlaneVonMises, {row[at[0]], row[at[1]], row[at[2]]}, plane.shearFactor, plane.scale};
        }
        const bool isComponent = std::any_of(plane.names.begin(), plane.names.end(),
                                             [&](const char* component) { return name == component; });
        return isComponent || isNamed(name, SHELL_AFFINE_COLUMNS) ? ItemForm{ItemForm::Kind::Affine} : ItemForm{};
    }
    return {};
}

// The element's stiffness applied to its displacements: f = K_e·u_e on the DOFs of K_e and uᵀ·K_e·u
// (StiffnessRecovery::strainEnergy: the element force dotted with u_e), twice the strain energy.
struct EnergyProducts
{
    std::vector<elements::ElementDof> dofs;
    std::vector<double> forces;
    double energyTwice = 0.0;
};

EnergyProducts energyProducts(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                              std::int64_t row, const elements::NodalDisplacements& displacements)
{
    EnergyProducts result;
    elements::ElementMatrixBlock stiffness;
    family.computeStiffness(view, {row, row + 1}, stiffness);
    if (stiffness.empty())
        return result;
    const std::span<const elements::ElementDof> dofs = stiffness.dofs(0);
    const std::span<const double> matrix = stiffness.matrix(0);
    result.dofs.assign(dofs.begin(), dofs.end());
    result.forces.resize(dofs.size());
    double sum = 0.0;
    for (std::size_t r = 0; r < dofs.size(); ++r)
    {
        double force = 0.0;
        for (std::size_t c = 0; c < dofs.size(); ++c)
            force += matrix[r * dofs.size() + c] * displacements.valueAt(dofs[c]);
        result.forces[r] = force;
        sum += displacements.valueAt(dofs[r]) * force;
    }
    result.energyTwice = sum;
    return result;
}

} // namespace

std::vector<elements::ElementDof> ElementItemFunction::dofs(const elements::ElementBatchView& view) const
{
    elements::ElementMatrixBlock block;
    m_family->computeStiffness(view, {m_row, m_row + 1}, block);
    if (block.empty())
        return {};
    return {block.dofs(0).begin(), block.dofs(0).end()};
}

double ElementItemFunction::forceProducts(const elements::ElementBatchView& view,
                                          const elements::NodalDisplacements& displacements) const
{
    elements::ElementMatrixBlock block;
    m_family->computeStiffness(view, {m_row, m_row + 1}, block);
    if (block.empty())
        return 0.0;
    const std::span<const elements::ElementDof> dofs = block.dofs(0);
    const std::span<const double> matrix = block.matrix(0);
    double largest = 0.0;
    for (std::size_t r = 0; r < dofs.size(); ++r)
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < dofs.size(); ++c)
            sum += std::abs(matrix[r * dofs.size() + c] * displacements.valueAt(dofs[c]));
        largest = std::max(largest, sum);
    }
    return largest;
}

double ElementItemFunction::volumeOf(const elements::ElementBatchView& view,
                                     const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                     base::MemoryBudget& budget) const
{
    if (m_volumeSnapshot.lock() == snapshot)
        return m_volume;
    const std::int32_t elementId = view.elementId(m_row);
    const std::string card = view.card();
    const std::map<std::int32_t, double> volumes = MassProperties::elementVolumes(
        *snapshot, snapshot, [&](std::string_view elementCard, std::int32_t candidateId, std::int32_t)
        { return candidateId == elementId && elementCard == card; }, budget);
    const auto found = volumes.find(elementId);
    m_volume = found == volumes.end() ? 0.0 : found->second;
    m_volumeSnapshot = snapshot;
    return m_volume;
}

std::vector<std::size_t> ElementItemFunction::columnsOf(const elements::ElementFamily& family,
                                                        const std::vector<ElementItemSpec>& items)
{
    std::vector<std::size_t> columns;
    columns.reserve(items.size());
    for (const ElementItemSpec& item : items)
    {
        if (item.type == ResponseType::StrainEnergy)
        {
            columns.push_back(NO_COLUMN);
            continue;
        }
        const std::vector<std::string> names = item.location.isComposite
                                                   ? family.layerOutputColumns(layerOutputOf(item.location.tableKind))
                                                   : family.outputColumns(outputOf(family, item.location.tableKind));
        columns.push_back(item.location.columnIn(names).value_or(NO_COLUMN));
    }
    return columns;
}

const elements::ElementResultBlock& ElementItemFunction::recoveredOutput(
    const elements::ElementBatchView& view, const ElementItemSpec& item,
    const elements::NodalDisplacements& displacements, Outputs& outputs) const
{
    const OutputKey key{item.location.isComposite, item.location.tableKind};
    auto [found, isNew] = outputs.try_emplace(key);
    if (isNew)
    {
        if (item.location.isComposite)
            m_family->recoverLayers(view, {m_row, m_row + 1}, layerOutputOf(item.location.tableKind), displacements,
                                    found->second);
        else
            m_family->recover(view, {m_row, m_row + 1}, outputOf(*m_family, item.location.tableKind), displacements,
                              found->second);
    }
    return found->second;
}

double ElementItemFunction::itemValue(const elements::ElementBatchView& view, const ElementItemSpec& item,
                                      std::size_t column, const elements::NodalDisplacements& displacements,
                                      Outputs& outputs, ItemForm* form) const
{
    const elements::ElementResultBlock& block = recoveredOutput(view, item, displacements, outputs);
    const std::int32_t layer = item.location.layer + item.lamina;
    const std::size_t index = findRow(block, item.location.place, layer);
    if (index == NO_ROW || column >= block.columns().size())
        refuse(view, m_row, item, "expected the recovered row of the item, found none");
    const double value = block.values(index)[column];
    if (form != nullptr && !item.location.isStrengthRatio)
        *form = formOf(block, index, column);
    if (!item.location.isStrengthRatio)
        return value;
    // CSTRAT (DesignResp1Response): the strength ratio of the failure index column, from the ply stresses and the
    // allowables of the laminate of the element's property.
    if (item.location.column == BOND_FAILURE_INDEX_COLUMN)
        return StrengthRatio::ofBond(value);
    ElementItemSpec stressItem = item;
    stressItem.location.tableKind = "stress";
    const elements::ElementResultBlock& stresses = recoveredOutput(view, stressItem, displacements, outputs);
    const std::size_t stressRow = findRow(stresses, ItemPlace::Centre, item.lamina);
    if (stressRow == NO_ROW || stresses.values(stressRow).size() < 3)
        refuse(view, m_row, item, "expected the ply stresses of the strength ratio, found none");
    const std::span<const double> stress = stresses.values(stressRow);
    const elements::Resolution<elements::ShellSection> section =
        elements::resolveShellSection(view, elements::ShellSectionColumns(view), m_row);
    if (!section || !section.value->isLaminate())
        refuse(view, m_row, item, "expected the laminate of the element's CompProp, found none");
    const elements::Laminate& laminate = *section.value->laminate;
    const auto ply =
        std::find_if(laminate.plies.begin(), laminate.plies.end(),
                     [&](const elements::LaminatePly& candidate) { return candidate.number == item.lamina; });
    if (ply == laminate.plies.end())
        refuse(view, m_row, item, "expected the lamina in the laminate, found none");
    return StrengthRatio::ofDirectStresses(laminate.failureTheory, ply->material, {stress[0], stress[1], stress[2]},
                                           value);
}

std::vector<double> ElementItemFunction::values(const elements::ElementBatchView& view,
                                                const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                                const elements::NodalDisplacements& displacements,
                                                base::MemoryBudget& budget, std::vector<ItemForm>* forms) const
{
    if (forms != nullptr)
        forms->assign(m_items.size(), ItemForm{});
    Outputs outputs;
    std::optional<double> energy;
    std::vector<double> values;
    values.reserve(m_items.size());
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const ElementItemSpec& item = m_items[k];
        if (item.type == ResponseType::StrainEnergy)
        {
            if (!energy)
                energy = 0.5 * energyProducts(*m_family, view, m_row, displacements).energyTwice;
            if (item.code != STRAIN_ENERGY_DENSITY_ITEM)
            {
                values.push_back(*energy);
                continue;
            }
            const double volume = volumeOf(view, snapshot, budget);
            if (!(volume > 0.0))
                refuse(view, m_row, item, "expected a positive volume of the element, found none");
            values.push_back(*energy / volume);
            continue;
        }
        ItemForm* form = forms != nullptr ? &(*forms)[values.size()] : nullptr;
        values.push_back(itemValue(view, item, m_columns[k], displacements, outputs, form));
    }
    return values;
}

std::optional<std::vector<double>> ElementItemFunction::fieldRates(const elements::ElementBatchView& view,
                                                                   const elements::NodalDisplacements& displacements,
                                                                   const elements::DesignedField& field) const
{
    std::map<OutputKey, elements::ElementResultBlock> ratesOf; // by layer flag and table kind
    std::vector<double> result;
    result.reserve(m_items.size());
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const ElementItemSpec& item = m_items[k];
        if (item.type == ResponseType::StrainEnergy || item.location.isStrengthRatio)
            return std::nullopt;
        auto [found, isNew] = ratesOf.try_emplace(OutputKey{item.location.isComposite, item.location.tableKind});
        const bool isGiven =
            !isNew ||
            (item.location.isComposite
                 ? m_family->layerRecoveryFieldRates(view, m_row, field, layerOutputOf(item.location.tableKind),
                                                     displacements, found->second)
                 : m_family->recoveryFieldRates(view, m_row, field, outputOf(*m_family, item.location.tableKind),
                                                displacements, found->second));
        if (!isGiven)
            return std::nullopt;
        const elements::ElementResultBlock& rates = found->second;
        const std::size_t index = findRow(rates, item.location.place, item.location.layer + item.lamina);
        if (index == NO_ROW || m_columns[k] >= rates.columns().size())
            return std::nullopt;
        const double rate = rates.values(index)[m_columns[k]];
        if (std::isnan(rate))
            return std::nullopt;
        result.push_back(rate);
    }
    return result;
}

std::vector<std::optional<std::vector<double>>> ElementItemFunction::fieldRatesOfFields(
    const elements::ElementBatchView& view, const elements::NodalDisplacements& displacements,
    std::span<const elements::DesignedField> fields) const
{
    std::vector<std::optional<std::vector<double>>> result(fields.size());
    for (const ElementItemSpec& item : m_items)
        if (item.type == ResponseType::StrainEnergy || item.location.isStrengthRatio)
            return result;
    // The rates of every field by table, the family asked once per table (fieldRates' table loop); a ply table's field
    // by field (layerRecoveryFieldRates).
    std::map<OutputKey, std::pair<std::vector<elements::ElementResultBlock>, std::vector<std::uint8_t>>> ratesOf;
    for (const ElementItemSpec& item : m_items)
    {
        auto [found, isNew] = ratesOf.try_emplace(OutputKey{item.location.isComposite, item.location.tableKind});
        if (!isNew)
            continue;
        found->second.first.resize(fields.size());
        found->second.second.assign(fields.size(), 0);
        if (item.location.isComposite)
        {
            m_family->layerRecoveryFieldRatesOfFields(view, m_row, fields, layerOutputOf(item.location.tableKind),
                                                      displacements, found->second.first, found->second.second);
            continue;
        }
        m_family->recoveryFieldRatesOfFields(view, m_row, fields, outputOf(*m_family, item.location.tableKind),
                                             displacements, found->second.first, found->second.second);
    }
    for (std::size_t k = 0; k < fields.size(); ++k)
    {
        std::vector<double> rates;
        rates.reserve(m_items.size());
        bool isGiven = true;
        for (std::size_t i = 0; i < m_items.size(); ++i)
        {
            const ElementItemSpec& item = m_items[i];
            const auto& [blocks, isClosed] = ratesOf.at(OutputKey{item.location.isComposite, item.location.tableKind});
            if (isClosed[k] == 0)
            {
                isGiven = false;
                break;
            }
            const elements::ElementResultBlock& block = blocks[k];
            const std::size_t index = findRow(block, item.location.place, item.location.layer + item.lamina);
            if (index == NO_ROW || m_columns[i] >= block.columns().size())
            {
                isGiven = false;
                break;
            }
            const double rate = block.values(index)[m_columns[i]];
            if (std::isnan(rate))
            {
                isGiven = false;
                break;
            }
            rates.push_back(rate);
        }
        if (isGiven)
            result[k] = std::move(rates);
    }
    return result;
}

std::optional<ItemSlopes> ElementItemFunction::slopes(const elements::ElementBatchView& view,
                                                      const std::shared_ptr<const input::ModelSnapshot>& snapshot,
                                                      const elements::NodalDisplacements& displacements,
                                                      const std::function<ScratchDisplacements()>& scratch,
                                                      base::MemoryBudget& budget) const
{
    // The form of every item first: a job any of whose items the slopes do not give is differenced whole. A ply's
    // failure or bond index takes the slopes of the ply states.
    std::map<OutputKey, std::vector<elements::ColumnForm>> formsOf;
    const OutputKey plyStatesKey{true, PLY_STATES};
    bool needsPlyStates = false;
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const ElementItemSpec& item = m_items[k];
        if (item.type == ResponseType::StrainEnergy)
            continue;
        if (item.location.isStrengthRatio)
            return std::nullopt;
        if (isPlyIndexItem(item))
        {
            needsPlyStates = true;
            continue;
        }
        const OutputKey key{item.location.isComposite, item.location.tableKind};
        auto [found, isNew] = formsOf.try_emplace(key);
        if (isNew)
            found->second = item.location.isComposite
                                ? m_family->layerOutputColumnForms(layerOutputOf(item.location.tableKind))
                                : m_family->outputColumnForms(outputOf(*m_family, item.location.tableKind));
        const std::vector<elements::ColumnForm>& forms = found->second;
        if (m_columns[k] >= forms.size())
            return std::nullopt;
        const elements::ColumnForm& form = forms[m_columns[k]];
        if (form.kind == elements::ColumnForm::Kind::Other)
            return std::nullopt;
        if (form.kind == elements::ColumnForm::Kind::PlaneVonMises)
            for (const std::size_t component : form.components)
                if (component >= forms.size() || forms[component].kind != elements::ColumnForm::Kind::Linear)
                    return std::nullopt;
    }

    if (needsPlyStates)
        formsOf[plyStatesKey] = m_family->layerOutputColumnForms(elements::LayerOutput::PlyStates);

    ItemSlopes result;
    result.values.reserve(m_items.size());
    // The values, as values() computes them (the same outputs, the same bits).
    Outputs outputs;
    std::optional<EnergyProducts> energy;
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const ElementItemSpec& item = m_items[k];
        if (item.type == ResponseType::StrainEnergy)
        {
            if (!energy)
                energy = energyProducts(*m_family, view, m_row, displacements);
            const double value = 0.5 * energy->energyTwice;
            if (item.code != STRAIN_ENERGY_DENSITY_ITEM)
            {
                result.values.push_back(value);
                continue;
            }
            const double volume = volumeOf(view, snapshot, budget);
            if (!(volume > 0.0))
                refuse(view, m_row, item, "expected a positive volume of the element, found none");
            result.values.push_back(value / volume);
            continue;
        }
        result.values.push_back(itemValue(view, item, m_columns[k], displacements, outputs, nullptr));
    }
    // The ply states the indices are functions of, at the point.
    std::optional<elements::ShellSection> section;
    if (needsPlyStates)
    {
        ElementItemSpec states;
        states.location.isComposite = true;
        states.location.tableKind = PLY_STATES;
        static_cast<void>(recoveredOutput(view, states, displacements, outputs));
        const elements::Resolution<elements::ShellSection> resolved =
            elements::resolveShellSection(view, elements::ShellSectionColumns(view), m_row);
        if (!resolved || !resolved.value->isLaminate())
            return std::nullopt;
        section = *resolved.value;
    }

    // The slopes of every output the items read; the element's DOFs those of the family's slopes, of its stiffness
    // (a strain energy), or asked of its stiffness facet for the recoveries at unit displacements.
    // PERFORMANCE: a family's slopes carry the DOFs of its stiffness, which the differences took from a stiffness
    // computation per job (a ShellQuad4's plate built once more).
    if (energy)
        result.dofs = energy->dofs;
    std::map<OutputKey, elements::RecoverySlopes> slopesOf;
    for (const auto& [key, forms] : formsOf)
    {
        elements::RecoverySlopes& slopes = slopesOf[key];
        const bool isGiven = key.first ? m_family->layerRecoverySlopes(view, m_row, layerOutputOf(key.second), slopes)
                                       : m_family->recoverySlopes(view, m_row, outputOf(*m_family, key.second), slopes);
        if (isGiven)
        {
            ++result.familySlopes;
            if (result.dofs.empty())
                result.dofs = slopes.dofs();
            continue;
        }
        if (result.dofs.empty())
            result.dofs = dofs(view);
        const std::size_t n = result.dofs.size();
        // rows(e_d) − rows(0) at unit displacements of the element's DOFs, the others at zero, no temperatures.
        const ScratchDisplacements copy = scratch();
        const std::span<double> grid = copy.grid;
        const std::span<double> scalar = copy.scalar;
        std::vector<double> saved(n);
        const auto at = [&](const elements::ElementDof& dof) -> double&
        {
            return dof.kind == input::PointKind::Scalar
                       ? scalar[static_cast<std::size_t>(dof.row)]
                       : grid[static_cast<std::size_t>(dof.row) * 6 + dof.component - 1];
        };
        for (std::size_t d = 0; d < n; ++d)
        {
            saved[d] = at(result.dofs[d]);
            at(result.dofs[d]) = 0.0;
        }
        const elements::NodalDisplacements unit{grid, scalar, nullptr};
        const auto recoverAt = [&](elements::ElementResultBlock& block)
        {
            if (key.first)
                m_family->recoverLayers(view, {m_row, m_row + 1}, layerOutputOf(key.second), unit, block);
            else
                m_family->recover(view, {m_row, m_row + 1}, outputOf(*m_family, key.second), unit, block);
            ++result.unitRecoveries;
        };
        elements::ElementResultBlock atRest;
        recoverAt(atRest);
        slopes.reset(result.dofs, atRest.columns().size());
        for (std::size_t r = 0; r < atRest.size(); ++r)
            slopes.appendRow(atRest.place(r));
        elements::ElementResultBlock moved;
        bool isSameLayout = true;
        for (std::size_t d = 0; d < n && isSameLayout; ++d)
        {
            at(result.dofs[d]) = 1.0;
            recoverAt(moved);
            at(result.dofs[d]) = 0.0;
            isSameLayout = moved.size() == atRest.size();
            for (std::size_t r = 0; r < atRest.size() && isSameLayout; ++r)
                for (std::size_t c = 0; c < forms.size() && c < atRest.columns().size(); ++c)
                    if (forms[c].kind == elements::ColumnForm::Kind::Linear)
                        slopes.slopes(r, c)[d] = moved.values(r)[c] - atRest.values(r)[c];
        }
        for (std::size_t d = 0; d < n; ++d)
            at(result.dofs[d]) = saved[d];
        // A recovery whose rows depend on the displacements has no slopes: the job is differenced.
        if (!isSameLayout)
            return std::nullopt;
    }

    // The weights of each item over the element's DOFs.
    const std::size_t n = result.dofs.size();
    result.weights.assign(m_items.size() * n, 0.0);
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const ElementItemSpec& item = m_items[k];
        const std::span<double> weights = std::span<double>(result.weights).subspan(k * n, n);
        if (item.type == ResponseType::StrainEnergy)
        {
            // ∂(½·uᵀK u)/∂u = K·u (K symmetric); the density over the volume.
            const double scale = item.code == STRAIN_ENERGY_DENSITY_ITEM ? 1.0 / volumeOf(view, snapshot, budget) : 1.0;
            for (std::size_t d = 0; d < n; ++d)
            {
                const auto found = std::find(energy->dofs.begin(), energy->dofs.end(), result.dofs[d]);
                if (found != energy->dofs.end())
                    weights[d] = scale * energy->forces[static_cast<std::size_t>(found - energy->dofs.begin())];
            }
            continue;
        }
        const bool isPlyIndex = isPlyIndexItem(item);
        const OutputKey key = isPlyIndex ? plyStatesKey : OutputKey{item.location.isComposite, item.location.tableKind};
        const elements::ElementResultBlock& block = outputs.at(key);
        const elements::RecoverySlopes& slopes = slopesOf.at(key);
        const std::size_t index = findRow(block, item.location.place, item.location.layer + item.lamina);
        if (slopes.size() != block.size() || index == NO_ROW)
            return std::nullopt;
        // The DOFs of the slopes, in the order of result.dofs (the family's are those of its stiffness).
        std::vector<std::size_t> position(slopes.dofs().size());
        for (std::size_t d = 0; d < slopes.dofs().size(); ++d)
        {
            const auto found = std::find(result.dofs.begin(), result.dofs.end(), slopes.dofs()[d]);
            if (found == result.dofs.end())
                return std::nullopt;
            position[d] = static_cast<std::size_t>(found - result.dofs.begin());
        }
        if (isPlyIndex)
        {
            // ∂index/∂u = ∇index(σ, ε) · ∂(σ, ε)/∂u (plyFailureIndexGradient), the bond index max(|τ1z|, |τ2z|)/SB
            // along its larger shear; differenced where the index has no derivative (a tie) or no value.
            const elements::Laminate& laminate = *section->laminate;
            const auto ply =
                std::find_if(laminate.plies.begin(), laminate.plies.end(),
                             [&](const elements::LaminatePly& candidate) { return candidate.number == item.lamina; });
            if (ply == laminate.plies.end())
                return std::nullopt;
            const std::span<const double> state = block.values(index);
            std::array<double, PLY_STATE_WEIGHTS> along{};
            if (item.location.column == FAILURE_INDEX_COLUMN)
            {
                const elements::FailureIndexGradient gradient =
                    elements::plyFailureIndexGradient(laminate.failureTheory, ply->material.allowables, ply->material,
                                                      {state[0], state[1], state[2]}, {state[3], state[4], state[5]});
                if (!gradient.isDefined)
                    return std::nullopt;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    along[i] = gradient.stress[i];
                    along[3 + i] = gradient.strain[i];
                }
            }
            else
            {
                const bool isLast = ply + 1 == laminate.plies.end();
                const double first = std::abs(state[6]);
                const double second = std::abs(state[7]);
                if (isLast || !(laminate.bondAllowable > 0.0) || first == second)
                    return std::nullopt;
                const std::size_t larger = first > second ? 6 : 7;
                along[larger] = (state[larger] >= 0.0 ? 1.0 : -1.0) / laminate.bondAllowable;
            }
            for (std::size_t d = 0; d < slopes.dofs().size(); ++d)
            {
                double weight = 0.0;
                for (std::size_t c = 0; c < PLY_STATE_WEIGHTS; ++c)
                    if (along[c] != 0.0)
                        weight += along[c] * slopes.slopes(index, c)[d];
                weights[position[d]] = weight;
            }
            continue;
        }
        const elements::ColumnForm& form = formsOf.at(key)[m_columns[k]];
        if (form.kind == elements::ColumnForm::Kind::Linear)
        {
            const std::span<const double> slope = slopes.slopes(index, m_columns[k]);
            for (std::size_t d = 0; d < slope.size(); ++d)
                weights[position[d]] = slope[d];
            continue;
        }
        // A plane von Mises, chained from its three components at their recovered values.
        const std::span<const double> row = block.values(index);
        const std::array<double, 3> components{row[form.components[0]], row[form.components[1]],
                                               row[form.components[2]]};
        for (std::size_t d = 0; d < slopes.dofs().size(); ++d)
        {
            const std::array<double, 3> rates{slopes.slopes(index, form.components[0])[d],
                                              slopes.slopes(index, form.components[1])[d],
                                              slopes.slopes(index, form.components[2])[d]};
            weights[position[d]] = elements::planeVonMisesRate(form, components, rates);
        }
    }
    return result;
}

namespace
{

// The slopes of a function of the displacements linear in its Linear columns (a recovery, or the rates of a recovery
// at fixed displacements) over `dofs`, at unit displacements: S(·, d) = f(e_d) − f(0), exact, on a writable copy of
// the model's displacements whose element DOFs are moved and restored; false when the layout of the rows changes with
// the displacements.
template <typename RowsAt>
bool unitSlopes(std::span<const elements::ElementDof> dofs, std::span<const elements::ColumnForm> forms,
                const ScratchDisplacements& copy, RowsAt&& rowsAt, elements::RecoverySlopes& slopes)
{
    const std::size_t n = dofs.size();
    const auto at = [&](const elements::ElementDof& dof) -> double&
    {
        return dof.kind == input::PointKind::Scalar
                   ? copy.scalar[static_cast<std::size_t>(dof.row)]
                   : copy.grid[static_cast<std::size_t>(dof.row) * 6 + dof.component - 1];
    };
    std::vector<double> saved(n); // bounded by the element's DOFs
    for (std::size_t d = 0; d < n; ++d)
    {
        saved[d] = at(dofs[d]);
        at(dofs[d]) = 0.0;
    }
    const elements::NodalDisplacements unit{copy.grid, copy.scalar, nullptr};
    elements::ElementResultBlock atRest;
    bool isSameLayout = rowsAt(unit, atRest);
    slopes.reset(dofs, atRest.columns().size());
    for (std::size_t r = 0; r < atRest.size(); ++r)
        slopes.appendRow(atRest.place(r));
    elements::ElementResultBlock moved;
    for (std::size_t d = 0; d < n && isSameLayout; ++d)
    {
        at(dofs[d]) = 1.0;
        isSameLayout = rowsAt(unit, moved) && moved.size() == atRest.size();
        at(dofs[d]) = 0.0;
        for (std::size_t r = 0; r < atRest.size() && isSameLayout; ++r)
            for (std::size_t c = 0; c < forms.size() && c < atRest.columns().size(); ++c)
                if (forms[c].kind == elements::ColumnForm::Kind::Linear)
                    slopes.slopes(r, c)[d] = moved.values(r)[c] - atRest.values(r)[c];
    }
    for (std::size_t d = 0; d < n; ++d)
        at(dofs[d]) = saved[d];
    return isSameLayout;
}

// The position in `dofs` of each DOF of `slopes`, or nothing when one is not there.
std::optional<std::vector<std::size_t>> positionsIn(const std::vector<elements::ElementDof>& dofs,
                                                    const elements::RecoverySlopes& slopes)
{
    std::vector<std::size_t> position(slopes.dofs().size()); // bounded by the element's DOFs
    for (std::size_t d = 0; d < slopes.dofs().size(); ++d)
    {
        const auto found = std::find(dofs.begin(), dofs.end(), slopes.dofs()[d]);
        if (found == dofs.end())
            return std::nullopt;
        position[d] = static_cast<std::size_t>(found - dofs.begin());
    }
    return position;
}

// What the second derivatives read of one output of the element: its forms, its rows at the displacements, their
// slopes, and with a field their rates, the slopes of those rates and their second rates.
struct SecondOrderOutput
{
    elements::ElementOutput output = elements::ElementOutput::Stresses;
    std::vector<elements::ColumnForm> forms;
    elements::ElementResultBlock values;
    elements::RecoverySlopes slopes;
    std::vector<std::size_t> slopePositions;
    elements::ElementResultBlock rates;
    elements::RecoverySlopes slopeRates;
    std::vector<std::size_t> slopeRatePositions;
    elements::ElementResultBlock secondRates;
};

} // namespace

std::optional<ItemSecondOrder> ElementItemFunction::secondOrder(const elements::ElementBatchView& view,
                                                                const elements::NodalDisplacements& displacements,
                                                                std::span<const double> weights,
                                                                std::span<const elements::DesignedField> fields,
                                                                const std::function<ScratchDisplacements()>& scratch,
                                                                ItemSlopeCache* cache) const
{
    if (fields.size() > 1 || weights.size() != m_items.size())
        return std::nullopt;
    // The outputs the weighted items read, and their forms: an item none of them gives leaves the element to the
    // differences.
    std::map<std::string, SecondOrderOutput> outputs;
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        if (weights[k] == 0.0)
            continue;
        const ElementItemSpec& item = m_items[k];
        const bool isRecovered =
            item.type == ResponseType::Stress || item.type == ResponseType::Strain || item.type == ResponseType::Force;
        if (!isRecovered || item.location.isComposite || item.location.isStrengthRatio)
            return std::nullopt;
        auto [found, isNew] = outputs.try_emplace(item.location.tableKind);
        if (isNew)
        {
            found->second.output = outputOf(*m_family, item.location.tableKind);
            found->second.forms = m_family->outputColumnForms(found->second.output);
        }
        const std::vector<elements::ColumnForm>& forms = found->second.forms;
        if (m_columns[k] >= forms.size())
            return std::nullopt;
        const elements::ColumnForm& form = forms[m_columns[k]];
        if (form.kind == elements::ColumnForm::Kind::Other)
            return std::nullopt;
        if (form.kind == elements::ColumnForm::Kind::PlaneVonMises)
            for (const std::size_t component : form.components)
                if (component >= forms.size() || forms[component].kind != elements::ColumnForm::Kind::Linear)
                    return std::nullopt;
    }

    ItemSecondOrder result;
    for (auto& [kind, part] : outputs)
    {
        // PERFORMANCE: the slopes (and below the slopes of their rates) of an earlier subcase of the element (the
        // caller's cache): they do not depend on the displacements, and they are the most of the cost (a ShellQuad4's
        // plate and place states built again for each output).
        // Measured: the threaded Hessian-vector products of a design point.
        ItemSlopeCache::Output* cached = nullptr;
        if (cache != nullptr)
        {
            const auto found = cache->byKind.find(kind);
            if (found != cache->byKind.end())
                cached = &found->second;
        }
        if (cached != nullptr)
        {
            part.slopes = cached->slopes;
            if (result.dofs.empty() && part.slopes.dofs().empty())
                result.dofs = dofs(view);
        }
        else if (!m_family->recoverySlopes(view, m_row, part.output, part.slopes))
        {
            if (result.dofs.empty())
                result.dofs = dofs(view);
            const auto recoverAt = [&](const elements::NodalDisplacements& unit, elements::ElementResultBlock& block)
            {
                m_family->recover(view, {m_row, m_row + 1}, part.output, unit, block);
                return true;
            };
            if (!unitSlopes(result.dofs, part.forms, scratch(), recoverAt, part.slopes))
                return std::nullopt;
        }
        if (cache != nullptr && cached == nullptr)
            cached = &(cache->byKind[kind] = ItemSlopeCache::Output{part.slopes, {}, false});
        if (result.dofs.empty())
            result.dofs = part.slopes.dofs();
        m_family->recover(view, {m_row, m_row + 1}, part.output, displacements, part.values);
        if (fields.empty())
            continue;
        const elements::DesignedField& field = fields.front();
        if (!m_family->recoveryFieldRates(view, m_row, field, part.output, displacements, part.rates) ||
            !m_family->recoveryFieldSecondRates(view, m_row, field, part.output, displacements, part.secondRates))
            return std::nullopt;
        if (cached != nullptr && cached->hasSlopeRates)
            part.slopeRates = cached->slopeRates;
        else
        {
            if (!m_family->recoverySlopeRates(view, m_row, field, part.output, part.slopeRates))
            {
                const auto ratesAt = [&](const elements::NodalDisplacements& unit, elements::ElementResultBlock& block)
                { return m_family->recoveryFieldRates(view, m_row, field, part.output, unit, block); };
                if (!unitSlopes(result.dofs, part.forms, scratch(), ratesAt, part.slopeRates))
                    return std::nullopt;
            }
            if (cached != nullptr)
            {
                cached->slopeRates = part.slopeRates;
                cached->hasSlopeRates = true;
            }
        }
    }
    const std::size_t n = result.dofs.size();
    const std::size_t fieldCount = fields.size();
    for (auto& [kind, part] : outputs)
    {
        std::optional<std::vector<std::size_t>> positions = positionsIn(result.dofs, part.slopes);
        if (!positions || part.slopes.size() != part.values.size())
            return std::nullopt;
        part.slopePositions = std::move(*positions);
        if (fieldCount == 0)
            continue;
        positions = positionsIn(result.dofs, part.slopeRates);
        if (!positions || part.slopeRates.size() != part.values.size() || part.rates.size() != part.values.size() ||
            part.secondRates.size() != part.values.size())
            return std::nullopt;
        part.slopeRatePositions = std::move(*positions);
    }

    result.mixedBlock.assign(n * fieldCount, 0.0);
    result.fieldBlock.assign(fieldCount * fieldCount, 0.0);
    std::vector<double> slope(n);     // bounded by the element's DOFs
    std::vector<double> slopeRate(n); // bounded by the element's DOFs
    std::array<std::vector<double>, 3> componentSlopes;
    std::array<std::vector<double>, 3> componentSlopeRates;
    for (std::size_t k = 0; k < m_items.size(); ++k)
    {
        const double weight = weights[k];
        if (weight == 0.0)
            continue;
        const ElementItemSpec& item = m_items[k];
        const SecondOrderOutput& part = outputs.at(item.location.tableKind);
        const std::size_t index = findRow(part.values, item.location.place, item.location.layer + item.lamina);
        if (index == NO_ROW || m_columns[k] >= part.values.columns().size())
            return std::nullopt;
        // A column's slopes (and the slopes of its rate) in the order of result.dofs.
        const auto gather = [&](const elements::RecoverySlopes& source, const std::vector<std::size_t>& position,
                                std::size_t column, std::vector<double>& into)
        {
            into.assign(n, 0.0);
            const std::span<const double> values = source.slopes(index, column);
            for (std::size_t d = 0; d < values.size(); ++d)
                into[position[d]] = values[d];
        };
        const std::size_t column = m_columns[k];
        const elements::ColumnForm& form = part.forms[column];
        if (form.kind == elements::ColumnForm::Kind::Linear)
        {
            if (fieldCount == 0)
                continue;
            gather(part.slopeRates, part.slopeRatePositions, column, slopeRate);
            const double secondRate = part.secondRates.values(index)[column];
            if (std::isnan(secondRate))
                return std::nullopt;
            for (std::size_t d = 0; d < n; ++d)
                result.mixedBlock[d] += weight * slopeRate[d];
            result.fieldBlock[0] += weight * secondRate;
            continue;
        }
        // A plane von Mises of three Linear components (a, b, c) of its row: chained from theirs.
        const std::span<const double> row = part.values.values(index);
        const std::array<double, 3> components{row[form.components[0]], row[form.components[1]],
                                               row[form.components[2]]};
        for (std::size_t m = 0; m < 3; ++m)
            gather(part.slopes, part.slopePositions, form.components[m], componentSlopes[m]);
        if (result.displacementBlock.empty())
            result.displacementBlock.assign(n * n, 0.0);
        const std::array<double, 3> none{};
        for (std::size_t d = 0; d < n; ++d)
        {
            const std::array<double, 3> along{componentSlopes[0][d], componentSlopes[1][d], componentSlopes[2][d]};
            for (std::size_t e = 0; e < n; ++e)
            {
                const std::array<double, 3> other{componentSlopes[0][e], componentSlopes[1][e], componentSlopes[2][e]};
                result.displacementBlock[d * n + e] +=
                    weight * elements::planeVonMisesSecondRate(form, components, along, other, none);
            }
        }
        if (fieldCount == 0)
            continue;
        std::array<double, 3> rates{};
        std::array<double, 3> secondRates{};
        for (std::size_t m = 0; m < 3; ++m)
        {
            rates[m] = part.rates.values(index)[form.components[m]];
            secondRates[m] = part.secondRates.values(index)[form.components[m]];
            gather(part.slopeRates, part.slopeRatePositions, form.components[m], componentSlopeRates[m]);
        }
        if (std::isnan(rates[0] + rates[1] + rates[2] + secondRates[0] + secondRates[1] + secondRates[2]))
            return std::nullopt;
        for (std::size_t d = 0; d < n; ++d)
        {
            const std::array<double, 3> along{componentSlopes[0][d], componentSlopes[1][d], componentSlopes[2][d]};
            const std::array<double, 3> mixed{componentSlopeRates[0][d], componentSlopeRates[1][d],
                                              componentSlopeRates[2][d]};
            result.mixedBlock[d] += weight * elements::planeVonMisesSecondRate(form, components, along, rates, mixed);
        }
        result.fieldBlock[0] += weight * elements::planeVonMisesSecondRate(form, components, rates, rates, secondRates);
    }
    if (result.dofs.empty())
        result.dofs = dofs(view);
    return result;
}

} // namespace oos::design
