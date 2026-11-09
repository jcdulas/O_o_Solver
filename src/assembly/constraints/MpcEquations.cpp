#include "oos/assembly/constraints/MpcEquations.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

base::Diagnostic mpcDiagnostic(std::string code, base::EntityId setId, std::int32_t equation, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = "MultiPointConstr";
    diagnostic.id = setId;
    diagnostic.field = base::concatenate({"equation ", std::to_string(equation)});
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The optional column of multiPointConstrTable that names the MPC card of each row (the MultiPointConstrUnion
// membership).
constexpr const char* SOURCE_SET_ID_COLUMN = "sourceSetId";

} // namespace

MpcEquations::MpcEquations(const input::ConstraintTable* constraints, const DofMap& dofMap,
                           std::optional<base::EntityId> mpcSetId, base::MemoryBudget& budget,
                           base::DiagnosticSink& sink)
    : m_setId(mpcSetId)
{
    if (constraints == nullptr || !mpcSetId)
        return;
    readEquations(*constraints, dofMap, nullptr, budget, sink);
}

MpcEquations::MpcEquations(const input::ConstraintTable* constraints, const DofMap& dofMap,
                           std::optional<base::EntityId> mpcSetId, std::span<const base::EntityId> keptSourceSetIds,
                           base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_setId(mpcSetId)
{
    if (!std::is_sorted(keptSourceSetIds.begin(), keptSourceSetIds.end()))
        throw std::invalid_argument(
            "MpcEquations: the MultiPointConstr card MultiPointConstr.ids to keep must be sorted; sort them "
            "first");
    if (constraints == nullptr || !mpcSetId)
        return;
    readEquations(*constraints, dofMap, &keptSourceSetIds, budget, sink);
}

bool MpcEquations::hasSourceSetIds(const input::ConstraintTable* constraints)
{
    return constraints != nullptr && constraints->mpcReader().hasColumn(SOURCE_SET_ID_COLUMN);
}

std::vector<base::EntityId> MpcEquations::sourceSetIdsOf(const input::ConstraintTable* constraints,
                                                         base::EntityId mpcSetId)
{
    if (constraints == nullptr)
        return {};
    if (!hasSourceSetIds(constraints))
        return {mpcSetId};
    const std::span<const std::int32_t> setIds = constraints->mpcSetIds();
    const std::span<const std::int32_t> sources = constraints->mpcReader().values<std::int32_t>(SOURCE_SET_ID_COLUMN);
    std::vector<base::EntityId> members; // bounded by the MPC cards of one set
    for (std::size_t row = 0; row < setIds.size(); ++row)
        if (setIds[row] == mpcSetId)
            members.push_back(sources[row]);
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());
    return members;
}

void MpcEquations::readEquations(const input::ConstraintTable& constraints, const DofMap& dofMap,
                                 const std::span<const base::EntityId>* keptSourceSetIds, base::MemoryBudget& budget,
                                 base::DiagnosticSink& sink)
{
    const std::span<const std::int32_t> setIds = constraints.mpcSetIds();
    const std::span<const std::int32_t> equations = constraints.mpcEquations();
    const std::span<const std::int32_t> nodeIds = constraints.mpcNodeIds();
    const std::span<const std::int32_t> components = constraints.mpcComponents();
    const std::span<const double> coefficients = constraints.mpcCoefficients();
    const base::EntityId setId = *m_setId;
    const bool hasSources = hasSourceSetIds(&constraints);
    const std::span<const std::int32_t> sources =
        hasSources ? constraints.mpcReader().values<std::int32_t>(SOURCE_SET_ID_COLUMN)
                   : std::span<const std::int32_t>{};
    // The MPC card of a row: its source_sid, or the set itself without the column.
    const auto sourceOf = [&](std::size_t row) { return hasSources ? sources[row] : setId; };
    const auto isKept = [&](std::size_t row)
    {
        return keptSourceSetIds == nullptr ||
               std::binary_search(keptSourceSetIds->begin(), keptSourceSetIds->end(), sourceOf(row));
    };

    std::size_t rowCount = 0;
    std::size_t equationCount = 0;
    for (std::size_t row = 0; row < setIds.size(); ++row)
    {
        if (setIds[row] != setId || !isKept(row))
            continue;
        ++rowCount;
        // multiPointConstrTable is sorted by set, equation, term: an equation starts where the equation number changes.
        if (row == 0 || setIds[row - 1] != setId || equations[row - 1] != equations[row])
            ++equationCount;
    }
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, equationCount + 1, "MPC equation offsets");
    m_terms = base::BudgetedBuffer<ConstraintTerm>(budget, rowCount, "MPC equation terms");
    m_equationNumbers = base::BudgetedBuffer<std::int32_t>(budget, equationCount, "MPC equation numbers");
    m_sourceSetIds = base::BudgetedBuffer<base::EntityId>(budget, equationCount, "MPC equation cards");

    std::int64_t termCount = 0;
    std::size_t row = 0;
    while (row < setIds.size())
    {
        if (setIds[row] != setId)
        {
            ++row;
            continue;
        }
        const std::int32_t equation = equations[row];
        std::size_t end = row;
        while (end < setIds.size() && setIds[end] == setId && equations[end] == equation)
            ++end;
        // The rows of one equation are the terms of one MPC card: its first row names the card.
        if (!isKept(row))
        {
            row = end;
            continue;
        }
        bool isAccepted = true;
        const std::int64_t first = termCount;
        for (std::size_t term = row; term < end && isAccepted; ++term)
        {
            const std::optional<input::PointRef> point = dofMap.points().find(nodeIds[term]);
            if (!point)
            {
                sink.report(mpcDiagnostic("mpc-unknown-point", setId, equation,
                                          base::concatenate({"the equation names point ", std::to_string(nodeIds[term]),
                                                             ", which the model does not define; add the Node or "
                                                             "ScalarPoint, or correct the MultiPointConstr card"})));
                isAccepted = false;
                break;
            }
            const std::optional<base::DofIndex> dof = dofMap.findDof(nodeIds[term], components[term]);
            if (!dof)
            {
                sink.report(mpcDiagnostic(
                    "mpc-bad-component", setId, equation,
                    base::concatenate({"the equation gives component ", std::to_string(components[term]), " to ",
                                       point->kind == input::PointKind::Scalar ? "scalar point " : "grid point ",
                                       std::to_string(nodeIds[term]),
                                       "; expected 1 to 6 for a grid point, 0 or blank for a scalar point"})));
                isAccepted = false;
                break;
            }
            m_terms[static_cast<std::size_t>(termCount++)] = {*dof, coefficients[term]};
        }
        if (isAccepted && m_terms[static_cast<std::size_t>(first)].coefficient == 0.0)
        {
            sink.report(mpcDiagnostic("mpc-zero-dependent-coefficient", setId, equation,
                                      "the coefficient of the dependent (first) term is zero, so the equation cannot "
                                      "be solved for it; give it a nonzero coefficient or reorder the terms"));
            isAccepted = false;
        }
        for (std::int64_t term = first + 1; isAccepted && term < termCount; ++term)
        {
            if (m_terms[static_cast<std::size_t>(term)].dof == m_terms[static_cast<std::size_t>(first)].dof)
            {
                sink.report(mpcDiagnostic("mpc-dependent-repeated", setId, equation,
                                          "the dependent DOF (first term) appears again among the other terms; "
                                          "combine the terms into the first one"));
                isAccepted = false;
            }
        }
        if (isAccepted)
        {
            m_offsets[m_count] = first;
            m_equationNumbers[m_count] = equation;
            m_sourceSetIds[m_count] = sourceOf(row);
            ++m_count;
        }
        else
        {
            termCount = first;
        }
        row = end;
    }
    m_offsets[m_count] = termCount;
}

} // namespace oos::assembly
