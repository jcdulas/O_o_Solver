#include "oos/assembly/singularities/GridPointSingularities.hpp"

#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/singularities/SingularBlock.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

// Yes or No of a text parameter; nothing (and a diagnostic) for anything else.
std::optional<bool> yesNo(const input::ParamTable* params, const std::string& name, std::optional<std::int32_t> subcase,
                          base::DiagnosticSink& sink)
{
    if (params == nullptr)
        return std::nullopt;
    const std::optional<input::ParamValue> value = params->find(name, subcase);
    if (!value)
        return std::nullopt;
    if (value->kind == input::ParamValue::Kind::Text && (value->text == "Yes" || value->text == "No"))
        return value->text == "Yes";
    sink.report(badParamDiagnostic(name, "Yes or No"));
    return std::nullopt;
}

// The smallest mantissa that "%.2f" prints as 10.00: the exponent then goes up by one ("9.995E-03" is "1.00E-02").
constexpr double ROUNDED_UP_MANTISSA = 9.995;

// The E format of the text tables with two exponent digits on every platform ("0.00E+00", "-1.23E-09").
std::string formatRatio(double value)
{
    if (value == 0.0)
        return "0.00E+00";
    int exponent = static_cast<int>(std::floor(std::log10(std::abs(value))));
    double mantissa = value / std::pow(10.0, exponent);
    if (std::abs(mantissa) >= ROUNDED_UP_MANTISSA)
    {
        mantissa /= 10.0;
        ++exponent;
    }
    char text[32];
    std::snprintf(text, sizeof text, "%.2fE%c%02d", mantissa, exponent < 0 ? '-' : '+', std::abs(exponent));
    return text;
}

} // namespace

SingularityOptions SingularityOptions::fromParams(const input::ParamTable* params, std::optional<std::int32_t> subcase,
                                                  base::DiagnosticSink& sink)
{
    SingularityOptions options;
    if (const std::optional<bool> autoSpc = yesNo(params, "AutoSingularityConstr", subcase, sink))
        options.isAutoSpcOn = *autoSpc;
    if (const std::optional<bool> print = yesNo(params, "PrintSingularities", subcase, sink))
        options.isTablePrinted = *print;
    if (params != nullptr)
    {
        if (const std::optional<input::ParamValue> value = params->find("PivotZeroThreshold", subcase))
        {
            const bool isNumber =
                value->kind == input::ParamValue::Kind::Real || value->kind == input::ParamValue::Kind::Integer;
            const double pivotZeroThreshold =
                value->kind == input::ParamValue::Kind::Real ? value->real : static_cast<double>(value->integer);
            if (isNumber && pivotZeroThreshold > 0.0 && pivotZeroThreshold < 1.0)
                options.pivotZeroThreshold = pivotZeroThreshold;
            else
                sink.report(badParamDiagnostic("PivotZeroThreshold", "a real number between 0 and 1"));
        }
    }
    return options;
}

GridPointSingularities::GridPointSingularities(const DofMap& dofMap, const DofSets& sets,
                                               const NodeDiagonalBlocks& blocks, const SingularityOptions& options,
                                               base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_options(options)
{
    const auto gridCount = static_cast<std::size_t>(dofMap.gridCount());
    const auto scalarCount = static_cast<std::size_t>(dofMap.scalarCount());
    if (blocks.gridBlocks.size() != 18 * gridCount || blocks.scalarDiagonals.size() != scalarCount)
        throw std::invalid_argument(base::concatenate(
            {"GridPointSingularities: expected 18 block values per grid point and 1 diagonal per scalar point (",
             std::to_string(18 * gridCount), " and ", std::to_string(scalarCount), "), found ",
             std::to_string(blocks.gridBlocks.size()), " and ", std::to_string(blocks.scalarDiagonals.size())}));
    const auto isFree = [&](base::DofIndex dof) { return sets.contains(DofSet::F, dof); };

    // The ratio of every free DOF found singular (NOT_SINGULAR_RATIO otherwise), and the alignment of each grid point
    // block.
    base::BudgetedBuffer<double> ratios(budget, static_cast<std::size_t>(dofMap.gSetSize()), "singularity ratios");
    base::BudgetedBuffer<std::uint8_t> isAligned(budget, 2 * gridCount, "singularity alignment");
    for (std::size_t dof = 0; dof < ratios.size(); ++dof)
        ratios[dof] = NOT_SINGULAR_RATIO;
    double largestDiagonal = 0.0;
    for (std::size_t row = 0; row < gridCount; ++row)
        for (std::size_t half = 0; half < 2; ++half)
        {
            const std::span<const double> block = blocks.gridBlocks.subspan(18 * row + 9 * half, 9);
            std::array<bool, 3> isFreeComponent{};
            for (std::size_t c = 0; c < 3; ++c)
            {
                if (!isFree(dofMap.gridDof(static_cast<std::int32_t>(row), static_cast<int>(3 * half + c + 1))))
                    continue;
                isFreeComponent[c] = true;
                largestDiagonal = std::max(largestDiagonal, std::abs(block[4 * c]));
            }
            const SingularBlock analysis =
                SingularBlock::analyse(block.first<9>(), isFreeComponent, options.pivotZeroThreshold);
            isAligned[2 * row + half] = analysis.isAligned ? 1 : 0;
            for (std::size_t c = 0; c < 3; ++c)
                ratios[static_cast<std::size_t>(dofMap.gridDof(
                    static_cast<std::int32_t>(row), static_cast<int>(3 * half + c + 1)))] = analysis.ratios[c];
        }
    for (std::size_t s = 0; s < scalarCount; ++s)
        if (isFree(dofMap.scalarDof(static_cast<std::int32_t>(s))))
            largestDiagonal = std::max(largestDiagonal, std::abs(blocks.scalarDiagonals[s]));
    for (std::size_t s = 0; s < scalarCount; ++s)
    {
        const base::DofIndex dof = dofMap.scalarDof(static_cast<std::int32_t>(s));
        const double diagonal = blocks.scalarDiagonals[s];
        if (isFree(dof) && diagonal <= options.pivotZeroThreshold * largestDiagonal)
            ratios[static_cast<std::size_t>(dof)] =
                largestDiagonal > 0.0 ? std::max(diagonal / largestDiagonal, 0.0) : 0.0;
    }

    std::size_t count = 0;
    for (std::size_t dof = 0; dof < ratios.size(); ++dof)
        count += ratios[dof] != NOT_SINGULAR_RATIO ? 1 : 0;
    m_rows = base::BudgetedBuffer<SingularityRow>(budget, count, "singularity table");
    m_autoSpcDofs = base::BudgetedBuffer<base::DofIndex>(budget, options.isAutoSpcOn ? count : 0, "AUTOSPC DOFs");
    for (std::size_t dof = 0; dof < ratios.size(); ++dof)
    {
        if (ratios[dof] == NOT_SINGULAR_RATIO)
            continue;
        const auto index = static_cast<base::DofIndex>(dof);
        const DofLabel label = dofMap.labelOf(index);
        SingularityRow row;
        row.dof = index;
        row.pointId = dofMap.pointIdOf(index);
        row.isScalarPoint = label.kind == input::PointKind::Scalar;
        row.component = label.component;
        row.stiffnessRatio = ratios[dof];
        row.oldExclusive = sets.exclusiveSetOf(index);
        row.isConstrained = options.isAutoSpcOn;
        row.isAligned = row.isScalarPoint ||
                        isAligned[2 * static_cast<std::size_t>(label.row) + (label.component <= 3 ? 0 : 1)] != 0;
        m_rows[m_rowCount++] = row;
        if (options.isAutoSpcOn)
            m_autoSpcDofs[m_autoSpcCount++] = index;
    }
    std::sort(m_rows.data(), m_rows.data() + m_rowCount, [](const SingularityRow& a, const SingularityRow& b)
              { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
    if (options.isAutoSpcOn)
        return;
    // Without AUTOSPC each singular point is reported: the factorisation will meet the mechanism.
    std::size_t first = 0;
    while (first < m_rowCount)
    {
        std::size_t end = first;
        std::string components;
        while (end < m_rowCount && m_rows[end].pointId == m_rows[first].pointId)
            components.append(std::to_string(m_rows[end++].component));
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "grid-point-singularity";
        diagnostic.card = m_rows[first].isScalarPoint ? "ScalarPoint" : "Node";
        diagnostic.id = m_rows[first].pointId;
        diagnostic.field = components;
        diagnostic.message = base::concatenate(
            {"components ", components, " of point ", std::to_string(m_rows[first].pointId),
             " have no stiffness (ratio below PivotZeroThreshold = ", formatRatio(options.pivotZeroThreshold),
             ") and PARAM AutoSingularityConstr is NO: constrain them (SinglePointConstr) or connect the point, or "
             "set PARAM AutoSingularityConstr YES"});
        sink.report(std::move(diagnostic));
        first = end;
    }
}

void GridPointSingularities::writeTable(std::ostream& out) const
{
    writeSingularityTable(out, rows(), m_options);
}

void writeSingularityTable(std::ostream& out, std::span<const SingularityRow> rows, const SingularityOptions& options)
{
    if (rows.empty() || !options.isTablePrinted)
        return;
    // A title, a header, then one line per singular component, in a layout of this project's own.
    out << "Singular components of the points (the constrained ones marked *)\n"
           "   point  type  component  stiffness ratio  set before  union before  set after  union after\n";
    for (const SingularityRow& row : rows)
    {
        const std::string oldExclusive(nameOf(row.oldExclusive));
        const std::string oldUnion = isMemberOf(row.oldExclusive, DofSet::S) ? "S" : "F";
        const std::string newExclusive = row.isConstrained ? "SB" : oldExclusive;
        const std::string newUnion = row.isConstrained ? "S" : oldUnion;
        char line[160];
        std::snprintf(line, sizeof line, "%8d  %4s  %9d  %15s  %10s  %12s  %9s  %11s %s", row.pointId,
                      row.isScalarPoint ? "S" : "G", row.component, formatRatio(row.stiffnessRatio).c_str(),
                      oldExclusive.c_str(), oldUnion.c_str(), newExclusive.c_str(), newUnion.c_str(),
                      row.isConstrained ? "*" : "");
        out << line << '\n';
    }
}

} // namespace oos::assembly
