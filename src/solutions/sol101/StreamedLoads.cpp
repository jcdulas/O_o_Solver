#include "oos/solutions/sol101/StreamedLoads.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace oos::solutions
{
namespace
{

using assembly::LoadFamily;

// The remedy every refusal of a load names: the paths that take it.
constexpr const char* OTHER_PATHS = "solve this analysis on the full path (not streamed) or the condensed path";

base::Diagnostic refusal(std::string card, base::EntityId id, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = "streamed-load-unsupported";
    diagnostic.card = std::move(card);
    diagnostic.id = id;
    diagnostic.message = base::concatenate({std::move(message), "; ", OTHER_PATHS});
    return diagnostic;
}

// The load set of a basis vector (nothing for a unit load of the loaded DOFs' basis).
std::optional<base::EntityId> loadSetOf(const StaticLoadCases::VectorName& name,
                                        std::span<const StaticSubcase* const> subcases)
{
    switch (name.source)
    {
    case assembly::BasisVector::Source::LoadSet:
        return name.loadSet;
    case assembly::BasisVector::Source::LoadedRow:
        return std::nullopt;
    case assembly::BasisVector::Source::Case:
        for (const StaticSubcase* subcase : subcases)
            if (subcase->label == name.subcase)
                return subcase->load;
        return std::nullopt;
    }
    return std::nullopt;
}

} // namespace

StreamedLoads::StreamedLoads(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_columnCount(inputs.cases->basisSize())
{
    const StreamedElements& elements = *inputs.elements;
    const assembly::LoadVectorBuilder& loads = *inputs.loads;
    const std::int32_t supernodeCount = elements.symbolic().supernodeCount();
    m_accelerations.assign(m_columnCount, {0.0, 0.0, 0.0});

    std::vector<std::pair<std::int32_t, EquationLoad>> concentrated; // (supernode, term), bounded by the point loads
    std::map<base::EntityId, std::vector<Consumer>> consumersOfSet;  // bounded by the load sets
    StreamedReduction reduction;
    for (std::size_t b = 0; b < m_columnCount; ++b)
    {
        const auto column = static_cast<std::int32_t>(b);
        const StaticLoadCases::VectorName name = inputs.cases->nameOf(b);
        const std::optional<base::EntityId> set = loadSetOf(name, inputs.subcases);
        switch (name.family)
        {
        case LoadFamily::Concentrated:
        {
            std::vector<assembly::LoadVectorBuilder::DofValue> values; // bounded by the set's loads
            if (name.source == assembly::BasisVector::Source::LoadedRow)
                values.push_back({name.dof, 1.0, 0.0});
            else if (set)
            {
                const base::BudgetedBuffer<assembly::LoadVectorBuilder::DofValue> entries =
                    loads.concentratedEntries(*set, budget);
                values.assign(entries.data(), entries.data() + entries.size());
            }
            for (const assembly::LoadVectorBuilder::DofValue& value : values)
            {
                const base::DofIndex dof = value.dof;
                m_dofLoads.push_back({dof, column, value.value});
                elements.reduce(std::span<const base::DofIndex>(&dof, 1), reduction);
                for (const StreamedReduction::Term& term : reduction.terms)
                {
                    const std::int32_t equation = reduction.equations[static_cast<std::size_t>(term.equation)];
                    concentrated.push_back(
                        {elements.supernodeOf(equation), {equation, column, term.coefficient * value.value}});
                }
            }
            break;
        }
        case LoadFamily::Distributed:
            if (!set)
                break;
            if (loads.hasLineLoads(*set))
                sink.report(
                    refusal("BeamDistributedLoad", *set,
                            "expected no line load (BeamDistributedLoad) on the full path streamed, whose bar and beam "
                            "results under line loads are not made by its fronts"));
            for (const input::LoadTerm& term : loads.termsOf(*set))
                consumersOfSet[term.set].push_back({column, term.factor});
            break;
        case LoadFamily::Inertial:
            if (!set)
                break;
            for (const assembly::InertialTerm& term : loads.inertialTermsOf(*set, sink))
            {
                // A GravityLoad row: the generator of the uniform translations of every grid point (its kind alone).
                if (term.generator.size() != 1 || term.generator.front() != 0.0 || term.coefficients.size() != 3)
                {
                    sink.report(refusal(
                        "AccelFieldLoad", *set,
                        "expected only GravityLoad in the inertial loads of the full path streamed (AccelFieldLoad, "
                        "AccelNodeLoad and RotationalForceLoad are not uniform translations of every grid point)"));
                    continue;
                }
                for (std::size_t k = 0; k < 3; ++k)
                    m_accelerations[b][k] += term.coefficients[k];
                m_hasInertial = true;
            }
            break;
        case LoadFamily::Thermal:
            sink.report(refusal("TemperatureSelection", name.subcase,
                                "expected no thermal or deformation load (TemperatureSelection(LOAD), "
                                "DeformationLoadSelection) on the full path "
                                "streamed (its element loads and results under initial strains are not made by its "
                                "fronts)"));
            break;
        case LoadFamily::Enforced:
            sink.report(refusal("EnforcedMotionConstr", name.subcase,
                                "expected no enforced displacement on the full path streamed (it holds every "
                                "constrained DOF at zero)"));
            break;
        }
    }

    // The concentrated terms by supernode (a stable counting sort: the vectors' order within each).
    const auto total = static_cast<std::size_t>(std::max(supernodeCount, 0));
    m_concentratedOffsets = base::BudgetedBuffer<std::int64_t>(budget, total + 1, "streamed loads: point loads");
    for (const auto& [supernode, load] : concentrated)
        ++m_concentratedOffsets[static_cast<std::size_t>(supernode) + 1];
    for (std::size_t s = 0; s < total; ++s)
        m_concentratedOffsets[s + 1] += m_concentratedOffsets[s];
    m_concentrated = base::BudgetedBuffer<EquationLoad>(budget, concentrated.size(), "streamed loads: point loads");
    {
        std::vector<std::int64_t> cursor(m_concentratedOffsets.data(), m_concentratedOffsets.data() + total);
        for (const auto& [supernode, load] : concentrated)
            m_concentrated[static_cast<std::size_t>(cursor[static_cast<std::size_t>(supernode)]++)] = load;
    }

    // The consumers of every set, then the pressure rows of each element (in table order) with the index of their
    // set's consumers.
    m_consumerOffsets.push_back(0);
    std::vector<std::pair<std::int32_t, ElementPressureRow>> rows; // (element, row), bounded by the pressure rows
    std::vector<assembly::LoadVectorBuilder::ElementLocation> locations;
    std::vector<elements::ElementPressure> pressures;
    for (const auto& [set, list] : consumersOfSet)
    {
        const auto index = static_cast<std::int32_t>(m_consumerOffsets.size() - 1);
        m_consumerList.insert(m_consumerList.end(), list.begin(), list.end());
        m_consumerOffsets.push_back(static_cast<std::int32_t>(m_consumerList.size()));
        const base::BudgetedBuffer<std::int64_t> setRows = loads.pressureRowsOf(set, budget);
        loads.pressuresOfRows(setRows.span(), locations, pressures);
        for (std::size_t k = 0; k < setRows.size(); ++k)
        {
            if (locations[k].elementId == 0)
            {
                sink.report(
                    refusal(std::string(pressures[k].card), set,
                            "expected the pressures of the full path streamed on elements (a PointsPressureLoad on the "
                            "faces of grid points, or an element no batch holds)"));
                continue;
            }
            const std::int64_t element =
                elements.batchFirst(static_cast<std::size_t>(locations[k].batchIndex)) + locations[k].row;
            rows.push_back({static_cast<std::int32_t>(element), {setRows[k], index}});
        }
    }
    if (!rows.empty())
    {
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b)
                  { return a.first != b.first ? a.first < b.first : a.second.row < b.second.row; });
        const auto elementCount = static_cast<std::size_t>(elements.elementCount());
        m_pressureOffsets = base::BudgetedBuffer<std::int64_t>(budget, elementCount + 1, "streamed loads: pressures");
        m_pressureRows = base::BudgetedBuffer<ElementPressureRow>(budget, rows.size(), "streamed loads: pressures");
        for (std::size_t k = 0; k < rows.size(); ++k)
        {
            ++m_pressureOffsets[static_cast<std::size_t>(rows[k].first) + 1];
            m_pressureRows[k] = rows[k].second;
        }
        for (std::size_t e = 0; e < elementCount; ++e)
            m_pressureOffsets[e + 1] += m_pressureOffsets[e];
    }
}

std::span<const StreamedLoads::EquationLoad> StreamedLoads::concentratedAt(std::int32_t s) const noexcept
{
    const auto at = static_cast<std::size_t>(s);
    return m_concentrated.span().subspan(
        static_cast<std::size_t>(m_concentratedOffsets[at]),
        static_cast<std::size_t>(m_concentratedOffsets[at + 1] - m_concentratedOffsets[at]));
}

std::span<const StreamedLoads::ElementPressureRow> StreamedLoads::pressuresOf(std::int32_t element) const noexcept
{
    if (m_pressureOffsets.size() == 0)
        return {};
    const auto at = static_cast<std::size_t>(element);
    return m_pressureRows.span().subspan(static_cast<std::size_t>(m_pressureOffsets[at]),
                                         static_cast<std::size_t>(m_pressureOffsets[at + 1] - m_pressureOffsets[at]));
}

std::span<const StreamedLoads::Consumer> StreamedLoads::consumers(std::int32_t index) const noexcept
{
    const auto at = static_cast<std::size_t>(index);
    return std::span<const Consumer>(m_consumerList)
        .subspan(static_cast<std::size_t>(m_consumerOffsets[at]),
                 static_cast<std::size_t>(m_consumerOffsets[at + 1] - m_consumerOffsets[at]));
}

void StreamedLoads::setRelief(std::vector<std::array<double, 6>> amplitudes)
{
    m_relief = std::move(amplitudes);
    m_hasRelief = !m_relief.empty();
}

std::uint64_t StreamedLoads::bytes() const noexcept
{
    return m_concentratedOffsets.size() * sizeof(std::int64_t) + m_concentrated.size() * sizeof(EquationLoad) +
           m_pressureOffsets.size() * sizeof(std::int64_t) + m_pressureRows.size() * sizeof(ElementPressureRow) +
           m_consumerList.size() * sizeof(Consumer) + m_dofLoads.size() * sizeof(DofLoad) +
           m_accelerations.size() * sizeof(std::array<double, 3>) + m_relief.size() * sizeof(std::array<double, 6>);
}

std::uint64_t StreamedLoads::estimateBytes(std::int64_t pressureRows, std::int64_t elements, std::int64_t supernodes,
                                           std::size_t columns) noexcept
{
    // The pressure rows of every element (and, while they are sorted, their pairs), the offsets, the point loads.
    const auto rows = static_cast<std::uint64_t>(std::max<std::int64_t>(pressureRows, 0));
    return rows * (sizeof(ElementPressureRow) + sizeof(std::pair<std::int32_t, ElementPressureRow>)) +
           static_cast<std::uint64_t>(elements + 1) * sizeof(std::int64_t) +
           static_cast<std::uint64_t>(supernodes + 1) * sizeof(std::int64_t) +
           columns * (sizeof(std::array<double, 3>) + sizeof(std::array<double, 6>));
}

} // namespace oos::solutions
