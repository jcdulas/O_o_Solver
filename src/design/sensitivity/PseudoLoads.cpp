#include "oos/design/sensitivity/PseudoLoads.hpp"

#include <algorithm>
#include <stdexcept>

namespace oos::design
{

PseudoLoads::PseudoLoads(const std::vector<FieldValue>& fields, std::size_t variableCount, std::size_t subcaseCount,
                         bool keepsLoadParts)
    : m_loads(fields.size(), std::vector<SparseVector>(subcaseCount)),
      m_energyRates(fields.size(), std::vector<double>(subcaseCount, 0.0)),
      m_fieldsOfVariable(variableCount),
      m_keepsLoadParts(keepsLoadParts),
      m_subcaseCount(subcaseCount)
{
    for (std::size_t r = 0; r < fields.size(); ++r)
        for (const auto& [column, derivative] : fields[r].gradient)
            if (derivative != 0.0)
                m_fieldsOfVariable[column].emplace_back(r, derivative);
    if (m_keepsLoadParts)
    {
        m_loadParts.assign(fields.size(), std::vector<SparseVector>(subcaseCount));
        m_appliedLoadParts.assign(fields.size(), std::vector<SparseVector>(subcaseCount));
    }
}

void PseudoLoads::addStiffness(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                               std::span<const double> stiffnessDerivative, std::span<const double> displacementsG)
{
    const StiffnessTerms terms = stiffnessTerms(dofs, stiffnessDerivative, displacementsG);
    addStiffnessTerms(field, subcase, dofs, terms.loads, terms.energy);
}

PseudoLoads::StiffnessTerms PseudoLoads::stiffnessTerms(std::span<const base::DofIndex> dofs,
                                                        std::span<const double> stiffnessDerivative,
                                                        std::span<const double> displacementsG)
{
    StiffnessTerms terms;
    const std::size_t n = dofs.size();
    terms.loads.resize(n);
    for (std::size_t r = 0; r < n; ++r)
    {
        double force = 0.0;
        for (std::size_t c = 0; c < n; ++c)
            force += stiffnessDerivative[r * n + c] * displacementsG[static_cast<std::size_t>(dofs[c])];
        terms.loads[r] = -force;
        terms.energy += displacementsG[static_cast<std::size_t>(dofs[r])] * force;
    }
    return terms;
}

void PseudoLoads::addStiffnessTerms(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                                    std::span<const double> loads, double energy)
{
    SparseVector& load = m_loads[field][subcase];
    for (std::size_t r = 0; r < dofs.size(); ++r)
        load.emplace_back(dofs[r], loads[r]);
    m_energyRates[field][subcase] += energy;
}

void PseudoLoads::addLoad(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                          std::span<const double> values)
{
    SparseVector& load = m_loads[field][subcase];
    for (std::size_t k = 0; k < dofs.size(); ++k)
        load.emplace_back(dofs[k], values[k]);
    if (m_keepsLoadParts)
        for (std::size_t k = 0; k < dofs.size(); ++k)
            m_loadParts[field][subcase].emplace_back(dofs[k], values[k]);
}

void PseudoLoads::addAppliedLoad(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                                 std::span<const double> values)
{
    SparseVector& load = m_loads[field][subcase];
    for (std::size_t k = 0; k < dofs.size(); ++k)
        load.emplace_back(dofs[k], values[k]);
    if (m_keepsLoadParts)
        for (std::size_t k = 0; k < dofs.size(); ++k)
        {
            m_loadParts[field][subcase].emplace_back(dofs[k], values[k]);
            m_appliedLoadParts[field][subcase].emplace_back(dofs[k], values[k]);
        }
}

void PseudoLoads::addDenseLoad(std::size_t field, std::size_t subcase, std::span<const double> valuesG)
{
    SparseVector& load = m_loads[field][subcase];
    for (std::size_t dof = 0; dof < valuesG.size(); ++dof)
        if (valuesG[dof] != 0.0)
        {
            load.emplace_back(static_cast<base::DofIndex>(dof), valuesG[dof]);
            if (m_keepsLoadParts)
            {
                m_loadParts[field][subcase].emplace_back(static_cast<base::DofIndex>(dof), valuesG[dof]);
                m_appliedLoadParts[field][subcase].emplace_back(static_cast<base::DofIndex>(dof), valuesG[dof]);
            }
        }
}

void PseudoLoads::finish(std::size_t field)
{
    for (SparseVector& load : m_loads[field])
        compressSparse(load);
    if (m_keepsLoadParts)
    {
        for (SparseVector& load : m_loadParts[field])
            compressSparse(load);
        for (SparseVector& load : m_appliedLoadParts[field])
            compressSparse(load);
    }
}

void PseudoLoads::addCombination(std::size_t field, std::size_t subcase,
                                 std::span<const std::pair<std::size_t, double>> terms)
{
    if (!m_keepsLoadParts)
        throw std::logic_error(
            "PseudoLoads::addCombination: the load parts are not kept (construct with keepsLoadParts for a "
            "SubcaseCombination; a "
            "bug of O_o_Solver: report it)");
    SparseVector& load = m_loads[field][subcase];
    SparseVector& part = m_loadParts[field][subcase];
    SparseVector& applied = m_appliedLoadParts[field][subcase];
    bool isAdded = false;
    for (const auto& [term, coefficient] : terms)
    {
        if (coefficient == 0.0)
            continue;
        for (const auto& [dof, value] : m_loadParts[field][term])
        {
            load.emplace_back(dof, coefficient * value);
            part.emplace_back(dof, coefficient * value);
            isAdded = true;
        }
        for (const auto& [dof, value] : m_appliedLoadParts[field][term])
            applied.emplace_back(dof, coefficient * value);
    }
    if (!isAdded)
        return;
    compressSparse(load);
    compressSparse(part);
    compressSparse(applied);
}

void PseudoLoads::setRelief(std::size_t subcase, const solutions::KeptRelief* relief)
{
    if (m_reliefs.empty())
    {
        m_reliefs.assign(m_subcaseCount, nullptr);
        m_reliefCoefficients.resize(m_loads.size());
        for (std::vector<std::array<double, assembly::MAX_SUPPORT_DOFS>>& field : m_reliefCoefficients)
            field.resize(m_subcaseCount);
    }
    m_reliefs[subcase] = relief;
}

void PseudoLoads::projectRelief(std::size_t field)
{
    if (m_reliefs.empty())
        return;
    if (!m_keepsLoadParts)
        throw std::logic_error(
            "PseudoLoads::projectRelief: the load parts are not kept (construct with "
            "keepsLoadParts for a relieved subcase; a bug of O_o_Solver: report it)");
    for (std::size_t s = 0; s < m_subcaseCount; ++s)
        if (m_reliefs[s] != nullptr)
            m_reliefCoefficients[field][s] = m_reliefs[s]->projectionOf(m_loadParts[field][s]);
}

std::array<double, assembly::MAX_SUPPORT_DOFS> PseudoLoads::reliefOf(std::size_t variable, std::size_t subcase) const
{
    std::array<double, assembly::MAX_SUPPORT_DOFS> combined{};
    for (const auto& [field, derivative] : m_fieldsOfVariable[variable])
        for (std::size_t k = 0; k < assembly::MAX_SUPPORT_DOFS; ++k)
            combined[k] += derivative * m_reliefCoefficients[field][subcase][k];
    return combined;
}

void PseudoLoads::addVariableLoad(std::size_t variable, std::size_t subcase, std::span<const base::DofIndex> dofs,
                                  std::span<const double> values)
{
    if (m_variableLoads.empty())
        m_variableLoads.assign(m_fieldsOfVariable.size(), std::vector<SparseVector>(m_subcaseCount));
    SparseVector& load = m_variableLoads[variable][subcase];
    for (std::size_t k = 0; k < dofs.size(); ++k)
        load.emplace_back(dofs[k], values[k]);
}

void PseudoLoads::finishVariableLoads()
{
    for (std::vector<SparseVector>& loads : m_variableLoads)
        for (SparseVector& load : loads)
            compressSparse(load);
}

const SparseVector& PseudoLoads::loadPartOf(std::size_t field, std::size_t subcase) const
{
    return m_keepsLoadParts ? m_loadParts[field][subcase] : m_empty;
}

const SparseVector& PseudoLoads::appliedLoadPartOf(std::size_t field, std::size_t subcase) const
{
    return m_keepsLoadParts ? m_appliedLoadParts[field][subcase] : m_empty;
}

bool PseudoLoads::hasLoad(std::size_t variable, std::size_t subcase) const
{
    if (!m_reliefs.empty() && m_reliefs[subcase] != nullptr)
    {
        const std::array<double, assembly::MAX_SUPPORT_DOFS> relief = reliefOf(variable, subcase);
        if (std::any_of(relief.begin(), relief.end(), [](double c) { return c != 0.0; }))
            return true;
    }
    return !variableLoadOf(variable, subcase).empty() ||
           std::any_of(m_fieldsOfVariable[variable].begin(), m_fieldsOfVariable[variable].end(),
                       [&](const auto& term) { return !m_loads[term.first][subcase].empty(); });
}

void PseudoLoads::markSupport(std::size_t subcase, std::span<std::uint8_t> mask) const
{
    for (const std::vector<SparseVector>& field : m_loads)
        for (const auto& [dof, value] : field[subcase])
            if (value != 0.0)
                mask[static_cast<std::size_t>(dof)] = 1;
    for (std::size_t variable = 0; variable < m_variableLoads.size(); ++variable)
        for (const auto& [dof, value] : variableLoadOf(variable, subcase))
            if (value != 0.0)
                mask[static_cast<std::size_t>(dof)] = 1;
    // The relief term −M·D·c is on every DOF with mass.
    if (!m_reliefs.empty() && m_reliefs[subcase] != nullptr)
    {
        const linalg::DenseMatrix& massModes = m_reliefs[subcase]->massModes;
        for (std::size_t k = 0; k < massModes.columns(); ++k)
            for (std::size_t dof = 0; dof < massModes.rows(); ++dof)
                if (massModes(dof, k) != 0.0)
                    mask[dof] = 1;
    }
}

bool PseudoLoads::markReads(std::size_t subcase, std::span<std::uint8_t> mask) const
{
    if (!m_reliefs.empty() && m_reliefs[subcase] != nullptr)
        return false;
    for (const std::vector<SparseVector>& field : m_loads)
        for (const auto& entry : field[subcase])
            mask[static_cast<std::size_t>(entry.first)] = 1;
    for (std::size_t variable = 0; variable < m_variableLoads.size(); ++variable)
        for (const auto& entry : variableLoadOf(variable, subcase))
            mask[static_cast<std::size_t>(entry.first)] = 1;
    return true;
}

void PseudoLoads::assemble(std::size_t variable, std::size_t subcase, std::span<double> valuesG) const
{
    std::fill(valuesG.begin(), valuesG.end(), 0.0);
    for (const auto& [field, derivative] : m_fieldsOfVariable[variable])
        for (const auto& [dof, value] : m_loads[field][subcase])
            valuesG[static_cast<std::size_t>(dof)] += derivative * value;
    for (const auto& [dof, value] : variableLoadOf(variable, subcase))
        valuesG[static_cast<std::size_t>(dof)] += value;
    if (!m_reliefs.empty() && m_reliefs[subcase] != nullptr)
    {
        const linalg::DenseMatrix& massModes = m_reliefs[subcase]->massModes;
        const std::array<double, assembly::MAX_SUPPORT_DOFS> relief = reliefOf(variable, subcase);
        for (std::size_t k = 0; k < massModes.columns(); ++k)
            if (relief[k] != 0.0)
                for (std::size_t dof = 0; dof < massModes.rows(); ++dof)
                    valuesG[dof] -= massModes(dof, k) * relief[k];
    }
}

void PseudoLoads::combine(std::size_t subcase, std::span<const double> v, std::span<double> valuesG) const
{
    // PERFORMANCE: one pass over the fields' sparse pseudo-loads, the variables folded into their weights first, in
    // place of a dense g-set assembly per variable.
    // Measured: a Hessian-vector product of the 7 152-bar truss after the
    // first 30 -> 6 ms (its solves and blocks left), of the 1 248-shell caisson 10 -> 5 ms.
    std::fill(valuesG.begin(), valuesG.end(), 0.0);
    std::vector<double> fieldWeights(m_loads.size(), 0.0); // bounded by the relations
    for (std::size_t j = 0; j < m_fieldsOfVariable.size(); ++j)
        if (v[j] != 0.0)
            for (const auto& [field, derivative] : m_fieldsOfVariable[j])
                fieldWeights[field] += v[j] * derivative;
    for (std::size_t field = 0; field < m_loads.size(); ++field)
        if (fieldWeights[field] != 0.0)
            for (const auto& [dof, value] : m_loads[field][subcase])
                valuesG[static_cast<std::size_t>(dof)] += fieldWeights[field] * value;
    for (std::size_t j = 0; j < m_fieldsOfVariable.size(); ++j)
        if (v[j] != 0.0)
            for (const auto& [dof, value] : variableLoadOf(j, subcase))
                valuesG[static_cast<std::size_t>(dof)] += v[j] * value;
    if (!m_reliefs.empty() && m_reliefs[subcase] != nullptr)
    {
        std::array<double, assembly::MAX_SUPPORT_DOFS> relief{};
        for (std::size_t j = 0; j < m_fieldsOfVariable.size(); ++j)
            if (v[j] != 0.0)
            {
                const std::array<double, assembly::MAX_SUPPORT_DOFS> ofVariable = reliefOf(j, subcase);
                for (std::size_t k = 0; k < relief.size(); ++k)
                    relief[k] += v[j] * ofVariable[k];
            }
        const linalg::DenseMatrix& massModes = m_reliefs[subcase]->massModes;
        for (std::size_t k = 0; k < massModes.columns(); ++k)
            if (relief[k] != 0.0)
                for (std::size_t dof = 0; dof < massModes.rows(); ++dof)
                    valuesG[dof] -= massModes(dof, k) * relief[k];
    }
}

void PseudoLoads::dotAll(std::size_t subcase, std::span<const double> vectorG, std::span<double> products) const
{
    dotRange(subcase, vectorG, 0, products.first(m_fieldsOfVariable.size()));
}

void PseudoLoads::dotRange(std::size_t subcase, std::span<const double> vectorG, std::size_t first,
                           std::span<double> products) const
{
    // The relief term: (M·D)ᵀ·v once for the call.
    const bool isRelieved = !m_reliefs.empty() && m_reliefs[subcase] != nullptr;
    std::array<double, assembly::MAX_SUPPORT_DOFS> massModeProducts{};
    if (isRelieved)
    {
        const linalg::DenseMatrix& massModes = m_reliefs[subcase]->massModes;
        for (std::size_t m = 0; m < massModes.columns(); ++m)
            for (std::size_t dof = 0; dof < massModes.rows(); ++dof)
                massModeProducts[m] += massModes(dof, m) * vectorG[dof];
    }
    for (std::size_t k = 0; k < products.size(); ++k)
    {
        double sum = 0.0;
        for (const auto& [field, derivative] : m_fieldsOfVariable[first + k])
            sum += derivative * dotSparse(m_loads[field][subcase], vectorG);
        sum += dotSparse(variableLoadOf(first + k, subcase), vectorG);
        if (isRelieved)
        {
            const std::array<double, assembly::MAX_SUPPORT_DOFS> relief = reliefOf(first + k, subcase);
            for (std::size_t m = 0; m < assembly::MAX_SUPPORT_DOFS; ++m)
                sum -= massModeProducts[m] * relief[m];
        }
        products[k] = sum;
    }
}

PseudoLoads::LoadSegments PseudoLoads::segments(std::size_t subcase, std::size_t tileDofs, std::size_t gSize,
                                                base::MemoryBudget& budget) const
{
    if (tileDofs == 0)
        throw std::invalid_argument("PseudoLoads::segments: expected tiles of at least one DOF, found 0");
    const std::size_t tiles = gSize / tileDofs + (gSize % tileDofs != 0 ? 1 : 0);
    const std::size_t sources = m_loads.size() + m_variableLoads.size();
    // The runs of each source's entries in one tile, counted per tile first (a counting sort by tile, stable in the
    // sources' order).
    const auto forEachRun = [&](auto&& f)
    {
        for (std::size_t source = 0; source < sources; ++source)
        {
            const SparseVector& entries = sourceOf(source, subcase);
            std::size_t begin = 0;
            while (begin < entries.size())
            {
                const std::size_t tile = static_cast<std::size_t>(entries[begin].first) / tileDofs;
                std::size_t end = begin + 1;
                while (end < entries.size() && static_cast<std::size_t>(entries[end].first) / tileDofs == tile)
                    ++end;
                f(tile, source, begin, end);
                begin = end;
            }
        }
    };
    LoadSegments result;
    result.tileStart = base::BudgetedBuffer<std::size_t>(budget, tiles + 1, "pseudo-load tiles");
    std::size_t runs = 0;
    forEachRun(
        [&](std::size_t tile, std::size_t, std::size_t, std::size_t)
        {
            ++result.tileStart[tile + 1];
            ++runs;
        });
    for (std::size_t t = 0; t < tiles; ++t)
        result.tileStart[t + 1] += result.tileStart[t];
    result.segments = base::BudgetedBuffer<LoadSegment>(budget, runs, "pseudo-load segments");
    std::vector<std::size_t> next(result.tileStart.data(), result.tileStart.data() + tiles); // one per tile
    forEachRun(
        [&](std::size_t tile, std::size_t source, std::size_t begin, std::size_t end)
        {
            result.segments[next[tile]++] = {static_cast<std::uint32_t>(source), static_cast<std::uint32_t>(begin),
                                             static_cast<std::uint32_t>(end)};
        });
    return result;
}

} // namespace oos::design
