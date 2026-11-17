#include "oos/assembly/constraints/FastenerConstraints.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/fasteners/PiercingInterpolation.hpp"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace oos::assembly
{

namespace
{

// The six rows of one side whose grid point is given (sized by one fastener side: std::vector).
struct SideRows
{
    std::int32_t gridRow = 0;
    std::vector<elements::ElementDof> piercedDofs; // the DOFs of the pierced element the side reads
    std::vector<double> coefficients;              // 6 rows of piercedDofs.size()
};

base::Diagnostic sideProblem(const std::string& card, std::int32_t elementId, const char* field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = "invalid-fastener-piercing";
    diagnostic.card = card;
    diagnostic.id = elementId;
    diagnostic.field = field;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The rows of the side on table rows [first, last), or nothing (with a diagnostic when the side cannot be resolved).
std::optional<SideRows> buildSide(const input::FastenerPiercingTable& table, std::int64_t first, std::int64_t last,
                                  const DofMap& dofMap, const elements::FormulationModel* formulation,
                                  base::DiagnosticSink& sink)
{
    const auto f = static_cast<std::size_t>(first);
    const std::int32_t elementId = table.fastenerIds()[f];
    const std::string card(table.card(first));
    const bool isSideA = table.isSideA(first);
    const char* field = isSideA ? "nodeA" : "nodeB";
    const std::int32_t pointId = table.pointGrids()[f];
    const std::optional<std::int32_t> pointRow = dofMap.points().findGrid(pointId);
    if (!pointRow)
    {
        sink.report(sideProblem(card, elementId, field,
                                base::concatenate({"expected grid point ", std::to_string(pointId),
                                                   " of the fastener piercing table in the model, found none; take the "
                                                   "snapshot again"})));
        return std::nullopt;
    }
    const elements::SideGridLookup findGrid = [&dofMap](std::int32_t gridId) -> std::optional<elements::SideGrid>
    {
        const std::optional<std::int32_t> gridRow = dofMap.points().findGrid(gridId);
        if (!gridRow)
            return std::nullopt;
        return elements::SideGrid{*gridRow, dofMap.positionInBasic(*gridRow), dofMap.cdToBasic(*gridRow)};
    };
    std::variant<elements::SideMotion, std::string> resolved =
        elements::resolveSideMotion(table, {first, last}, findGrid, formulation);
    if (const auto* problem = std::get_if<std::string>(&resolved))
    {
        sink.report(sideProblem(card, elementId, isSideA ? "patchIdA" : "patchIdB",
                                base::concatenate({"pierced element of ", card, ".", field, ": ", *problem})));
        return std::nullopt;
    }
    const elements::SideMotion& motion = std::get<elements::SideMotion>(resolved);
    SideRows side;
    side.gridRow = *pointRow;
    side.piercedDofs = motion.dofs;
    const base::Vec3 point = motion.point;

    // G = diag(R_GAᵀ, R_GAᵀ)·[I, −[x_GA − x_P]×; 0, I]·M.
    const base::Mat3 toPoint = dofMap.cdToBasic(side.gridRow).transposed();
    const base::Mat3 arm = elements::crossProductMatrix(dofMap.positionInBasic(side.gridRow) - point);
    const std::size_t width = side.piercedDofs.size();
    side.coefficients.assign(6 * width, 0.0);
    const auto at = [&](std::size_t i, std::size_t j) { return motion.map[i * width + j]; };
    for (std::size_t j = 0; j < width; ++j)
    {
        base::Vec3 translation{at(0, j), at(1, j), at(2, j)};
        const base::Vec3 rotation{at(3, j), at(4, j), at(5, j)};
        translation = translation - arm * rotation; // u_P + θ_P × (x_GA − x_P) = u_P − [x_GA − x_P]×·θ_P
        const base::Vec3 localTranslation = toPoint * translation;
        const base::Vec3 localRotation = toPoint * rotation;
        for (std::size_t i = 0; i < 3; ++i)
        {
            side.coefficients[i * width + j] = localTranslation[i];
            side.coefficients[(3 + i) * width + j] = localRotation[i];
        }
    }
    return side;
}

// Calls visit(first, last) for the rows of each side whose grid point the card gives.
template <typename Visit>
void forEachGivenSide(const input::FastenerPiercingTable& table, Visit visit)
{
    const base::ValidityView isGiven = table.isPointGridGiven();
    const std::span<const std::int32_t> elementIds = table.fastenerIds();
    std::int64_t first = 0;
    while (first < table.rowCount())
    {
        std::int64_t last = first + 1;
        while (last < table.rowCount() &&
               elementIds[static_cast<std::size_t>(last)] == elementIds[static_cast<std::size_t>(first)] &&
               table.isSideA(last) == table.isSideA(first))
            ++last;
        if (isGiven.isValid(first))
            visit(first, last);
        first = last;
    }
}

ConstraintSource sourceOfCard(std::string_view card)
{
    return card == "Weld" ? ConstraintSource::Weld : ConstraintSource::Fastener;
}

} // namespace

FastenerConstraints::FastenerConstraints(const input::FastenerPiercingTable* piercings, const DofMap& dofMap,
                                         base::MemoryBudget& budget, base::DiagnosticSink& sink,
                                         const elements::FormulationModel* formulation)
{
    if (piercings == nullptr)
        return;
    // Pass 1: the sizes (diagnostics to a scratch sink; pass 2 reports them).
    // A grid point shared by two fasteners (a stack of three sheets: the GB of one fastener is the GA of the next, on
    // the same middle sheet) is the one auxiliary point of both: it is made dependent once, by the first side that
    // names it (the two sides pierce the same sheet at the same point and interpolate it alike).
    base::BudgetedBuffer<std::uint8_t> isConstrained(budget, static_cast<std::size_t>(dofMap.gridCount()),
                                                     "fastener constrained grid points");
    std::size_t rowCapacity = 0;
    std::int64_t termCapacity = 0;
    base::DiagnosticSink scratch;
    forEachGivenSide(*piercings,
                     [&](std::int64_t first, std::int64_t last)
                     {
                         const std::optional<SideRows> side =
                             buildSide(*piercings, first, last, dofMap, formulation, scratch);
                         if (!side || isConstrained[static_cast<std::size_t>(side->gridRow)] != 0)
                             return;
                         isConstrained[static_cast<std::size_t>(side->gridRow)] = 1;
                         rowCapacity += 6;
                         termCapacity += static_cast<std::int64_t>(side->coefficients.size());
                     });
    for (std::size_t g = 0; g < isConstrained.size(); ++g)
        isConstrained[g] = 0;
    m_dependents = base::BudgetedBuffer<base::DofIndex>(budget, rowCapacity, "fastener constraint rows");
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, rowCapacity + 1, "fastener constraint rows");
    m_terms = base::BudgetedBuffer<ConstraintTerm>(budget, static_cast<std::size_t>(termCapacity),
                                                   "fastener constraint terms");
    m_sources = base::BudgetedBuffer<std::uint8_t>(budget, rowCapacity, "fastener constraint rows");
    m_sourceIds = base::BudgetedBuffer<base::EntityId>(budget, rowCapacity, "fastener constraint rows");
    m_offsets[0] = 0;

    // Pass 2: the rows, six per given grid point, its components 1 to 6.
    forEachGivenSide(*piercings,
                     [&](std::int64_t first, std::int64_t last)
                     {
                         const std::optional<SideRows> side =
                             buildSide(*piercings, first, last, dofMap, formulation, sink);
                         if (!side || isConstrained[static_cast<std::size_t>(side->gridRow)] != 0)
                             return;
                         isConstrained[static_cast<std::size_t>(side->gridRow)] = 1;
                         const std::size_t width = side->piercedDofs.size();
                         const ConstraintSource source = sourceOfCard(piercings->card(first));
                         const std::int32_t elementId = piercings->fastenerIds()[static_cast<std::size_t>(first)];
                         for (int component = 1; component <= 6; ++component)
                         {
                             const auto i = static_cast<std::size_t>(component - 1);
                             m_dependents[m_rowCount] = dofMap.gridDof(side->gridRow, component);
                             m_sources[m_rowCount] = static_cast<std::uint8_t>(source);
                             m_sourceIds[m_rowCount] = elementId;
                             for (std::size_t j = 0; j < width; ++j)
                             {
                                 const double coefficient = side->coefficients[i * width + j];
                                 if (coefficient == 0.0)
                                     continue;
                                 m_terms[static_cast<std::size_t>(m_termCount++)] = {dofMap.dofOf(side->piercedDofs[j]),
                                                                                     coefficient};
                             }
                             ++m_rowCount;
                             m_offsets[m_rowCount] = m_termCount;
                         }
                     });
}

} // namespace oos::assembly
