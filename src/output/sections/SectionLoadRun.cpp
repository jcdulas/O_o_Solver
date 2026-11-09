#include "oos/output/sections/SectionLoadRun.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::output
{

namespace
{

SectionLoadTable sectionLabels(const SectionLoads& sections)
{
    SectionLoadTable table;
    table.kind = "sectionLoads";
    for (std::size_t s = 0; s < sections.sections().size(); ++s)
    {
        const SectionLoads::Section& section = sections.sections()[s];
        const SectionDefinition& definition = sections.definitions()[section.definition];
        table.ids.push_back(definition.id);
        table.names.push_back(definition.name);
        table.groups.push_back(definition.group);
        table.families.emplace_back(nameOf(section.family));
        for (const std::string_view component : sectionComponents(section.family))
        {
            table.rowItems.push_back(static_cast<std::int32_t>(s));
            table.rowComponents.push_back(component);
        }
    }
    return table;
}

SectionLoadTable cutLabels(const CutLoads& cuts)
{
    SectionLoadTable table;
    table.kind = "cutLoads";
    for (std::size_t c = 0; c < cuts.cuts().size(); ++c)
    {
        const CutLoads::Cut& cut = cuts.cuts()[c];
        const CutDefinition& definition = cuts.definitions()[cut.definition];
        table.ids.push_back(definition.id);
        table.names.push_back(definition.name);
        table.groups.push_back(definition.group);
        table.families.emplace_back("cut");
        table.points.push_back(cut.point);
        table.axes.push_back(cut.cdAxes);
        for (const std::string_view component : CUT_COMPONENTS)
        {
            table.rowItems.push_back(static_cast<std::int32_t>(c));
            table.rowComponents.push_back(component);
        }
    }
    return table;
}

} // namespace

SectionLoadRun::SectionLoadRun(const SectionLoadRequest& request, const input::ModelSnapshot& snapshot,
                               const assembly::ElementSet& set, const assembly::DofMap& dofMap, std::size_t caseCount,
                               base::MemoryBudget& budget, base::DiagnosticSink& sink, base::ThreadPool* pool)
    : m_set(&set), m_maximumBatchCases(request.maximumBatchCases)
{
    base::DiagnosticSink problems;
    if (!request.sections.empty())
        m_sections = std::make_unique<SectionLoads>(request.sections, snapshot, set, dofMap, problems);
    if (!request.cuts.empty())
        m_cuts = std::make_unique<CutLoads>(request.cuts, snapshot, set, dofMap, problems);
    for (const base::Diagnostic& diagnostic : problems.diagnostics())
        sink.report(diagnostic);
    if (problems.hasErrors())
        return;
    if (m_sections)
    {
        m_sectionRows = m_sections->displacementRows(budget, pool).rowCount();
        m_sectionAccumulator = std::make_unique<SectionLoadAccumulator>(sectionLabels(*m_sections), caseCount,
                                                                        request.keepsValues, request.sink, budget);
    }
    if (m_cuts)
    {
        m_cutRows = m_cuts->displacementRows(budget, pool).rowCount();
        m_cutAccumulator = std::make_unique<SectionLoadAccumulator>(cutLabels(*m_cuts), caseCount, request.keepsValues,
                                                                    request.sink, budget);
    }
    m_zeros = base::BudgetedBuffer<double>(budget, static_cast<std::size_t>(dofMap.gSetSize()),
                                           "section load zero displacements");
}

void SectionLoadRun::checkLineLoads(std::int32_t subcase,
                                    std::span<const std::vector<elements::ElementLineLoad>> lineLoads,
                                    base::DiagnosticSink& sink) const
{
    if (!m_sections)
        return;
    for (std::size_t b = 0; b < lineLoads.size(); ++b)
        for (const elements::ElementLineLoad& load : lineLoads[b])
        {
            const std::optional<std::size_t> section = m_sections->lineSectionOf(b, load.elementRow);
            if (!section)
                continue;
            const SectionDefinition& definition =
                m_sections->definitions()[m_sections->sections()[*section].definition];
            const elements::ElementBatchView& view = m_set->batches()[b].view;
            // The message names the cards by their deck keywords.

            base::Diagnostic diagnostic;
            diagnostic.severity = base::Severity::Error;
            diagnostic.code = "section-line-load";
            diagnostic.card = std::string(view.card());
            diagnostic.id = view.elementId(load.elementRow);
            diagnostic.message = base::concatenate(
                {"section ", definition.name.empty() ? std::to_string(definition.id) : definition.name, ": ",
                 diagnostic.card, " ", std::to_string(view.elementId(load.elementRow)), " is loaded by a ",
                 std::string(load.card), " in subcase ", std::to_string(subcase),
                 ", whose forces are given at the stations of the element, which a section does not read in this "
                 "version; leave the element out of the section, or move its load to its grid points"});
            sink.report(std::move(diagnostic));
        }
}

linalg::DenseMatrix SectionLoadRun::basisValues(linalg::ConstDenseMatrixView displacements, base::MemoryBudget& budget,
                                                base::ThreadPool* pool) const
{
    linalg::DenseMatrix basis(budget, rowCount(), displacements.columns, "section and cut loads of the load basis");
    if (m_sections && m_sectionRows > 0)
        m_sections->builtDisplacementRows().apply(displacements,
                                                  basis.view().block(0, 0, m_sectionRows, displacements.columns), pool);
    if (m_cuts && m_cutRows > 0)
        m_cuts->builtDisplacementRows().apply(
            displacements, basis.view().block(m_sectionRows, 0, m_cutRows, displacements.columns), pool);
    return basis;
}

void SectionLoadRun::addThermalPart(const elements::ElementTemperatures& temperatures, linalg::DenseMatrixView values,
                                    std::size_t j) const
{
    std::vector<double> part(rowCount(), 0.0); // bounded by the rows of the definitions
    if (m_sections && m_sectionRows > 0)
        m_sections->evaluate(*m_set, m_zeros.span(), &temperatures, std::span<double>(part).first(m_sectionRows));
    if (m_cuts && m_cutRows > 0)
        m_cuts->evaluate(*m_set, m_zeros.span(), &temperatures, std::span<double>(part).subspan(m_sectionRows));
    for (std::size_t r = 0; r < rowCount(); ++r)
        values(r, j) += part[r];
}

void SectionLoadRun::offerRows(std::span<const std::int32_t> cases, linalg::ConstDenseMatrixView values)
{
    if (m_sectionAccumulator)
        m_sectionAccumulator->offer(cases, {values.data, m_sectionRows, cases.size(), values.leadingDimension});
    if (m_cutAccumulator)
        m_cutAccumulator->offer(cases, {values.data + m_sectionRows, m_cutRows, cases.size(), values.leadingDimension});
}

void SectionLoadRun::offerBatch(const linalg::DenseMatrix& basis, const ResultCombination& combination,
                                std::size_t first, std::span<const std::int32_t> cases,
                                const TemperaturesOf& temperaturesOf, base::MemoryBudget& budget)
{
    if (rowCount() == 0 || cases.empty())
        return;
    const std::size_t size = m_maximumBatchCases > 0 ? std::min(m_maximumBatchCases, cases.size()) : cases.size();
    linalg::DenseMatrix values(budget, rowCount(), size, "section and cut loads of a batch of cases");
    for (std::size_t start = 0; start < cases.size(); start += size)
    {
        const std::size_t count = std::min(size, cases.size() - start);
        combination.combine(basis.view(), first + start, count, values.view(), budget);
        if (temperaturesOf)
            for (std::size_t j = 0; j < count; ++j)
                if (const std::unique_ptr<assembly::TemperatureField> field = temperaturesOf(start + j))
                    addThermalPart(field->temperatures(), values.view(), j);
        offerRows(cases.subspan(start, count), values.view());
    }
}

void SectionLoadRun::offerCase(std::int32_t subcase, std::span<const double> displacements,
                               const elements::ElementTemperatures* temperatures, base::MemoryBudget& budget)
{
    if (rowCount() == 0)
        return;
    linalg::DenseMatrix values(budget, rowCount(), 1, "section and cut loads of a case");
    const linalg::ConstDenseMatrixView u{displacements.data(), displacements.size(), 1, displacements.size()};
    if (m_sections && m_sectionRows > 0)
        m_sections->builtDisplacementRows().apply(u, values.view().block(0, 0, m_sectionRows, 1));
    if (m_cuts && m_cutRows > 0)
        m_cuts->builtDisplacementRows().apply(u, values.view().block(m_sectionRows, 0, m_cutRows, 1));
    if (temperatures != nullptr)
        addThermalPart(*temperatures, values.view(), 0);
    const std::array<std::int32_t, 1> cases{subcase};
    offerRows(cases, values.view());
}

std::optional<SectionLoadTable> SectionLoadRun::takeSectionTable()
{
    if (!m_sectionAccumulator)
        return std::nullopt;
    SectionLoadTable table = std::move(*m_sectionAccumulator).finish();
    m_sectionAccumulator.reset();
    return table;
}

std::optional<SectionLoadTable> SectionLoadRun::takeCutTable()
{
    if (!m_cutAccumulator)
        return std::nullopt;
    SectionLoadTable table = std::move(*m_cutAccumulator).finish();
    m_cutAccumulator.reset();
    return table;
}

} // namespace oos::output
