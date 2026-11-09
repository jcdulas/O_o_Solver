#include "oos/assembly/constraints/RigidElimination.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

// The rows of every element and equation as they come, before the chains are resolved (model-sized: budgeted).
struct DirectRows
{
    base::BudgetedBuffer<base::DofIndex> dofs;
    base::BudgetedBuffer<std::int64_t> offsets;
    base::BudgetedBuffer<ConstraintTerm> terms;
    base::BudgetedBuffer<std::uint8_t> sources;
    base::BudgetedBuffer<base::EntityId> sourceIds;
    base::BudgetedBuffer<std::int32_t> relationIds; // RigidElimination::relationOf
    std::size_t rowCount = 0;
    std::int64_t termCount = 0;

    DirectRows(base::MemoryBudget& budget, std::size_t rows, std::int64_t termCapacity)
        : dofs(budget, rows, "Gm direct rows"),
          offsets(budget, rows + 1, "Gm direct rows"),
          terms(budget, static_cast<std::size_t>(termCapacity), "Gm direct terms"),
          sources(budget, rows, "Gm direct rows"),
          sourceIds(budget, rows, "Gm direct rows"),
          relationIds(budget, rows, "Gm direct rows")
    {
    }

    void add(base::DofIndex dof, std::span<const ConstraintTerm> rowTerms, ConstraintSource source,
             base::EntityId sourceId, std::int32_t relationId)
    {
        dofs[rowCount] = dof;
        offsets[rowCount] = termCount;
        sources[rowCount] = static_cast<std::uint8_t>(source);
        sourceIds[rowCount] = sourceId;
        relationIds[rowCount] = relationId;
        for (const ConstraintTerm& term : rowTerms)
            terms[static_cast<std::size_t>(termCount++)] = term;
        ++rowCount;
        offsets[rowCount] = termCount;
    }
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t r) const
    {
        return {terms.data() + offsets[r], static_cast<std::size_t>(offsets[r + 1] - offsets[r])};
    }
};

// Calls visit(begin, end) for the rows of each rigid element of the table (one elementId, contiguous: the table is
// sorted by elementId, then the order of the card).
template <typename Visit>
void forEachRigidElement(const input::RigidTable& table, Visit visit)
{
    const std::span<const std::int32_t> elementIds = table.ids();
    const std::int64_t rowCount = table.rowCount();
    std::int64_t begin = 0;
    while (begin < rowCount)
    {
        std::int64_t end = begin + 1;
        while (end < rowCount &&
               elementIds[static_cast<std::size_t>(end)] == elementIds[static_cast<std::size_t>(begin)])
            ++end;
        visit(begin, end);
        begin = end;
    }
}

// The rows of the MPC equation i: u_m = −Σ_{j≥1} (A_j/A_0)·u_j (MPC).
void mpcRow(const MpcEquations& mpc, std::size_t i, std::vector<ConstraintTerm>& terms)
{
    const std::span<const ConstraintTerm> equation = mpc.terms(i);
    terms.clear();
    for (std::size_t j = 1; j < equation.size(); ++j)
        terms.push_back({equation[j].dof, -equation[j].coefficient / equation[0].coefficient});
}

std::string ownerText(ConstraintSource source, base::EntityId id)
{
    return source == ConstraintSource::Mpc ? base::concatenate({"MultiPointConstr set ", std::to_string(id)})
                                           : base::concatenate({cardNameOf(source), " ", std::to_string(id)});
}

std::string dofText(const DofMap& dofMap, base::DofIndex dof)
{
    const DofLabel label = dofMap.labelOf(dof);
    return base::concatenate({"component ", std::to_string(static_cast<int>(label.component)), " of ",
                              label.kind == input::PointKind::Scalar ? "scalar point " : "grid point ",
                              std::to_string(dofMap.pointIdOf(dof))});
}

constexpr std::uint8_t UNVISITED = 0;
constexpr std::uint8_t ON_STACK = 1;
constexpr std::uint8_t DONE = 2;

} // namespace

RigidElimination::RigidElimination(const input::RigidTable* rigidElements, const MpcEquations* mpc,
                                   const DofMap& dofMap, base::MemoryBudget& budget, base::DiagnosticSink& sink,
                                   const FastenerConstraints* fasteners, const GlueConstraints* glue,
                                   const ConstraintRowFilter& filter)
{
    const auto isTaken = [&filter](ConstraintSource source, base::EntityId id, base::DofIndex dependent)
    { return !filter || filter(ConstraintRowOrigin{source, id, dependent}); };
    const RigidElementEquations equations(dofMap);
    LocalConstraintRows local;
    std::vector<ConstraintTerm> terms;

    // Pass 1: the sizes (the element diagnostics go to a scratch sink; pass 2 reports them).
    std::size_t rowCapacity = 0;
    std::int64_t termCapacity = 0;
    if (rigidElements != nullptr)
    {
        base::DiagnosticSink scratch;
        forEachRigidElement(*rigidElements,
                            [&](std::int64_t begin, std::int64_t end)
                            {
                                if (equations.build(*rigidElements, begin, end, local, scratch))
                                {
                                    rowCapacity += local.size();
                                    termCapacity += static_cast<std::int64_t>(local.terms.size());
                                }
                            });
    }
    if (fasteners != nullptr)
    {
        rowCapacity += fasteners->rowCount();
        termCapacity += fasteners->termCount();
    }
    if (mpc != nullptr)
    {
        rowCapacity += mpc->size();
        for (std::size_t i = 0; i < mpc->size(); ++i)
            termCapacity += static_cast<std::int64_t>(mpc->terms(i).size()) - 1;
    }
    if (glue != nullptr)
    {
        rowCapacity += glue->rowCount();
        termCapacity += glue->termCount();
    }

    // Pass 2: the rows, rigid elements first (in table order), then the fasteners, the MPC equations and the glue.
    DirectRows direct(budget, rowCapacity, termCapacity);
    if (rigidElements != nullptr)
    {
        const std::span<const std::int32_t> elementIds = rigidElements->ids();
        const base::DictionaryView types = rigidElements->types();
        forEachRigidElement(*rigidElements,
                            [&](std::int64_t begin, std::int64_t end)
                            {
                                if (!equations.build(*rigidElements, begin, end, local, sink))
                                    return;
                                const ConstraintSource source = *RigidElementEquations::findSource(types[begin]);
                                const base::EntityId elementId = elementIds[static_cast<std::size_t>(begin)];
                                for (std::size_t r = 0; r < local.size(); ++r)
                                    if (isTaken(source, elementId, local.dependents[r]))
                                        direct.add(local.dependents[r], local.row(r), source, elementId, elementId);
                            });
    }
    if (fasteners != nullptr)
    {
        for (std::size_t i = 0; i < fasteners->rowCount(); ++i)
            if (isTaken(fasteners->sourceOf(i), fasteners->sourceIdOf(i), fasteners->dependentDof(i)))
                direct.add(fasteners->dependentDof(i), fasteners->row(i), fasteners->sourceOf(i),
                           fasteners->sourceIdOf(i), 0);
    }
    if (mpc != nullptr)
    {
        for (std::size_t i = 0; i < mpc->size(); ++i)
        {
            if (!isTaken(ConstraintSource::Mpc, mpc->sourceSetIdOf(i), mpc->dependentDof(i)))
                continue;
            mpcRow(*mpc, i, terms);
            direct.add(mpc->dependentDof(i), terms, ConstraintSource::Mpc, mpc->setId().value_or(0),
                       -mpc->sourceSetIdOf(i));
        }
    }
    if (glue != nullptr)
    {
        for (std::size_t i = 0; i < glue->rowCount(); ++i)
            if (isTaken(glue->sourceOf(i), glue->sourceIdOf(i), glue->dependentDof(i)))
                direct.add(glue->dependentDof(i), glue->row(i), glue->sourceOf(i), glue->sourceIdOf(i), 0);
    }
    const std::size_t rowCount = direct.rowCount;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());

    // Each DOF dependent once: the first row keeps it.
    base::BudgetedBuffer<std::int32_t> rowOfDof(budget, gSize, "Gm row of each DOF");
    for (std::size_t dof = 0; dof < gSize; ++dof)
        rowOfDof[dof] = -1;
    base::BudgetedBuffer<std::uint8_t> isValid(budget, rowCount, "Gm row flags");
    for (std::size_t r = 0; r < rowCount; ++r)
    {
        std::int32_t& owner = rowOfDof[static_cast<std::size_t>(direct.dofs[r])];
        if (owner < 0)
        {
            owner = static_cast<std::int32_t>(r);
            isValid[r] = 1;
            continue;
        }
        const auto first = static_cast<std::size_t>(owner);
        const auto source = static_cast<ConstraintSource>(direct.sources[r]);
        base::Diagnostic diagnostic;
        diagnostic.code = "dof-dependent-twice";
        diagnostic.card = std::string(cardNameOf(source));
        diagnostic.id = direct.sourceIds[r];
        diagnostic.field = std::to_string(static_cast<int>(dofMap.labelOf(direct.dofs[r]).component));
        diagnostic.message = base::concatenate(
            {dofText(dofMap, direct.dofs[r]), " is made dependent by ", ownerText(source, direct.sourceIds[r]),
             " and already by ",
             ownerText(static_cast<ConstraintSource>(direct.sources[first]), direct.sourceIds[first]),
             " (a DOF may be dependent, in the m set, only once); make it dependent in one of them only"});
        sink.report(std::move(diagnostic));
    }
    const auto dependentRowOf = [&](base::DofIndex dof) -> std::int64_t
    {
        const std::int32_t r = rowOfDof[static_cast<std::size_t>(dof)];
        return r >= 0 && isValid[static_cast<std::size_t>(r)] != 0 ? r : -1;
    };

    // Dependency order by an iterative depth-first search; a row met again while on the stack closes a cycle.
    base::BudgetedBuffer<std::uint8_t> state(budget, rowCount, "Gm row states");
    base::BudgetedBuffer<std::int64_t> stackRows(budget, rowCount, "Gm dependency stack");
    base::BudgetedBuffer<std::int64_t> stackNext(budget, rowCount, "Gm dependency stack");
    base::BudgetedBuffer<std::int64_t> stackPosition(budget, rowCount, "Gm dependency stack");
    base::BudgetedBuffer<std::int64_t> order(budget, rowCount, "Gm dependency order");
    std::size_t orderCount = 0;
    for (std::size_t root = 0; root < rowCount; ++root)
    {
        if (isValid[root] == 0 || state[root] != UNVISITED)
            continue;
        std::size_t depth = 0;
        stackRows[depth] = static_cast<std::int64_t>(root);
        stackNext[depth] = 0;
        stackPosition[root] = 0;
        state[root] = ON_STACK;
        ++depth;
        while (depth > 0)
        {
            const auto current = static_cast<std::size_t>(stackRows[depth - 1]);
            const std::span<const ConstraintTerm> rowTerms = direct.row(current);
            const auto next = static_cast<std::size_t>(stackNext[depth - 1]);
            if (next < rowTerms.size())
            {
                ++stackNext[depth - 1];
                const std::int64_t child = dependentRowOf(rowTerms[next].dof);
                if (child < 0)
                    continue;
                const auto c = static_cast<std::size_t>(child);
                if (state[c] == UNVISITED)
                {
                    stackRows[depth] = child;
                    stackNext[depth] = 0;
                    stackPosition[c] = static_cast<std::int64_t>(depth);
                    state[c] = ON_STACK;
                    ++depth;
                }
                else if (state[c] == ON_STACK)
                {
                    // The rows from the child up to the top of the stack form the cycle.
                    std::string members;
                    base::EntityId firstId = 0;
                    std::string firstCard;
                    for (auto position = static_cast<std::size_t>(stackPosition[c]); position < depth; ++position)
                    {
                        const auto member = static_cast<std::size_t>(stackRows[position]);
                        const auto source = static_cast<ConstraintSource>(direct.sources[member]);
                        if (members.empty())
                        {
                            firstId = direct.sourceIds[member];
                            firstCard = std::string(cardNameOf(source));
                        }
                        members.append(members.empty() ? "" : ", ")
                            .append(ownerText(source, direct.sourceIds[member]))
                            .append(" (")
                            .append(dofText(dofMap, direct.dofs[member]))
                            .append(")");
                        isValid[member] = 0;
                    }
                    base::Diagnostic diagnostic;
                    diagnostic.code = "dependency-cycle";
                    diagnostic.card = firstCard;
                    diagnostic.id = firstId;
                    diagnostic.message = base::concatenate(
                        {"the dependent DOFs of these elements depend on each other in a cycle: ", members,
                         "; a dependent DOF cannot be independent in an element its own motion depends on: make one "
                         "of them independent"});
                    sink.report(std::move(diagnostic));
                }
                continue;
            }
            --depth;
            state[current] = DONE;
            if (isValid[current] != 0)
                order[orderCount++] = static_cast<std::int64_t>(current);
        }
    }

    // Expansion in dependency order: each row is its terms with the rows of the dependent DOFs substituted, summed in
    // a scatter accumulator over the g set, then written by increasing DOF (a fixed order of every sum).
    base::BudgetedBuffer<std::int64_t> bound(budget, rowCount, "Gm row bounds");
    std::int64_t boundTotal = 0;
    for (std::size_t k = 0; k < orderCount; ++k)
    {
        const auto r = static_cast<std::size_t>(order[k]);
        std::int64_t size = 0;
        for (const ConstraintTerm& term : direct.row(r))
        {
            const std::int64_t child = dependentRowOf(term.dof);
            size += child < 0 ? 1 : bound[static_cast<std::size_t>(child)];
        }
        bound[r] = std::min<std::int64_t>(size, static_cast<std::int64_t>(gSize));
        boundTotal += bound[r];
    }
    base::BudgetedBuffer<ConstraintTerm> expanded(budget, static_cast<std::size_t>(boundTotal), "Gm expanded terms");
    base::BudgetedBuffer<std::int64_t> expandedStart(budget, rowCount, "Gm expanded rows");
    base::BudgetedBuffer<std::int64_t> expandedLength(budget, rowCount, "Gm expanded rows");
    base::BudgetedBuffer<double> accumulator(budget, gSize, "Gm accumulator");
    base::BudgetedBuffer<std::int64_t> mark(budget, gSize, "Gm accumulator");
    base::BudgetedBuffer<base::DofIndex> touched(budget, gSize, "Gm accumulator");
    for (std::size_t dof = 0; dof < gSize; ++dof)
        mark[dof] = -1;
    std::int64_t written = 0;
    for (std::size_t k = 0; k < orderCount; ++k)
    {
        const auto r = static_cast<std::size_t>(order[k]);
        const auto stamp = static_cast<std::int64_t>(k);
        std::size_t touchedCount = 0;
        const auto add = [&](base::DofIndex dof, double value)
        {
            const auto index = static_cast<std::size_t>(dof);
            if (mark[index] != stamp)
            {
                mark[index] = stamp;
                accumulator[index] = 0.0;
                touched[touchedCount++] = dof;
            }
            accumulator[index] += value;
        };
        for (const ConstraintTerm& term : direct.row(r))
        {
            const std::int64_t child = dependentRowOf(term.dof);
            if (child < 0)
            {
                add(term.dof, term.coefficient);
                continue;
            }
            const auto c = static_cast<std::size_t>(child);
            for (std::int64_t e = 0; e < expandedLength[c]; ++e)
            {
                const ConstraintTerm& childTerm = expanded[static_cast<std::size_t>(expandedStart[c] + e)];
                add(childTerm.dof, term.coefficient * childTerm.coefficient);
            }
        }
        std::sort(touched.data(), touched.data() + touchedCount);
        expandedStart[r] = written;
        for (std::size_t t = 0; t < touchedCount; ++t)
        {
            const double value = accumulator[static_cast<std::size_t>(touched[t])];
            if (value != 0.0)
                expanded[static_cast<std::size_t>(written++)] = {touched[t], value};
        }
        expandedLength[r] = written - expandedStart[r];
    }

    // The final Gm, by increasing m DOF, exactly sized.
    std::size_t finalRows = 0;
    for (std::size_t dof = 0; dof < gSize; ++dof)
        finalRows += dependentRowOf(static_cast<base::DofIndex>(dof)) >= 0 ? 1 : 0;
    m_mDofs = base::BudgetedBuffer<base::DofIndex>(budget, finalRows, "Gm m DOFs");
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, finalRows + 1, "Gm offsets");
    m_terms = base::BudgetedBuffer<ConstraintTerm>(budget, static_cast<std::size_t>(written), "Gm terms");
    m_sources = base::BudgetedBuffer<std::uint8_t>(budget, finalRows, "Gm sources");
    m_sourceIds = base::BudgetedBuffer<base::EntityId>(budget, finalRows, "Gm sources");
    m_relationIds = base::BudgetedBuffer<std::int32_t>(budget, finalRows, "Gm relations");
    m_rowOfDof = base::BudgetedBuffer<std::int32_t>(budget, gSize, "Gm row of each DOF");
    std::size_t next = 0;
    std::int64_t termCount = 0;
    for (std::size_t dof = 0; dof < gSize; ++dof)
    {
        const std::int64_t r = dependentRowOf(static_cast<base::DofIndex>(dof));
        m_rowOfDof[dof] = r < 0 ? -1 : static_cast<std::int32_t>(next);
        if (r < 0)
            continue;
        const auto source = static_cast<std::size_t>(r);
        m_mDofs[next] = static_cast<base::DofIndex>(dof);
        m_offsets[next] = termCount;
        m_sources[next] = direct.sources[source];
        m_sourceIds[next] = direct.sourceIds[source];
        m_relationIds[next] = direct.relationIds[source];
        for (std::int64_t e = 0; e < expandedLength[source]; ++e)
            m_terms[static_cast<std::size_t>(termCount++)] =
                expanded[static_cast<std::size_t>(expandedStart[source] + e)];
        ++next;
    }
    m_offsets[finalRows] = termCount;

    // The rows' own terms, kept when a chain was resolved (the relations' forces need them: forEachRelationForce).
    bool isChained = false;
    std::int64_t directTermCount = 0;
    for (std::size_t k = 0; k < orderCount; ++k)
        for (const ConstraintTerm& term : direct.row(static_cast<std::size_t>(order[k])))
        {
            ++directTermCount;
            isChained = isChained || dependentRowOf(term.dof) >= 0;
        }
    if (!isChained)
        return;
    m_directOffsets = base::BudgetedBuffer<std::int64_t>(budget, finalRows + 1, "Gm own rows");
    m_directTerms =
        base::BudgetedBuffer<ConstraintTerm>(budget, static_cast<std::size_t>(directTermCount), "Gm own terms");
    std::int64_t directNext = 0;
    for (std::size_t i = 0; i < finalRows; ++i)
    {
        const auto source = static_cast<std::size_t>(dependentRowOf(m_mDofs[i]));
        m_directOffsets[i] = directNext;
        for (const ConstraintTerm& term : direct.row(source))
            m_directTerms[static_cast<std::size_t>(directNext++)] = term;
    }
    m_directOffsets[finalRows] = directNext;
    // The dependency order puts a row after the rows its terms depend on (children first): reversed, the parents come
    // first, as the multipliers of the chains need them.
    std::size_t chained = 0;
    for (std::size_t k = 0; k < orderCount; ++k)
        chained += m_rowOfDof[static_cast<std::size_t>(direct.dofs[static_cast<std::size_t>(order[k])])] >= 0 ? 1 : 0;
    m_chainOrder = base::BudgetedBuffer<std::int64_t>(budget, chained, "Gm chain order");
    std::size_t nextChained = 0;
    for (std::size_t k = orderCount; k-- > 0;)
        if (const std::int32_t row =
                m_rowOfDof[static_cast<std::size_t>(direct.dofs[static_cast<std::size_t>(order[k])])];
            row >= 0)
            m_chainOrder[nextChained++] = row;
}

void RigidElimination::expandToG(std::span<double> displacementsG) const noexcept
{
    for (std::size_t i = 0; i < m_mDofs.size(); ++i)
    {
        double sum = 0.0;
        for (const ConstraintTerm& term : row(i))
            sum += term.coefficient * displacementsG[static_cast<std::size_t>(term.dof)];
        displacementsG[static_cast<std::size_t>(m_mDofs[i])] = sum;
    }
}

void RigidElimination::reduceToN(std::span<double> loadsG) const noexcept
{
    for (std::size_t i = 0; i < m_mDofs.size(); ++i)
    {
        double& dependentLoad = loadsG[static_cast<std::size_t>(m_mDofs[i])];
        for (const ConstraintTerm& term : row(i))
            loadsG[static_cast<std::size_t>(term.dof)] += term.coefficient * dependentLoad;
        dependentLoad = 0.0;
    }
}

base::BudgetedBuffer<RelationForce> RigidElimination::relationForces(std::span<const double> mpcForcesG,
                                                                     base::MemoryBudget& budget) const
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < m_mDofs.size(); ++i)
        count += 1 + relationTerms(i).size();
    base::BudgetedBuffer<RelationForce> forces(budget, count, "MPC forces by relation");
    std::size_t next = 0;
    forEachRelationForce(mpcForcesG, budget, [&](std::size_t row, base::DofIndex dof, double force)
                         { forces[next++] = RelationForce{dof, m_relationIds[row], force}; });
    return forces;
}

void RigidElimination::computeMpcForces(std::span<const double> elementForcesG, std::span<const double> loadsG,
                                        std::span<double> mpcForcesG) const noexcept
{
    for (double& value : mpcForcesG)
        value = 0.0;
    for (std::size_t i = 0; i < m_mDofs.size(); ++i)
    {
        const auto dependent = static_cast<std::size_t>(m_mDofs[i]);
        const double force = elementForcesG[dependent] - loadsG[dependent];
        mpcForcesG[dependent] = force;
        for (const ConstraintTerm& term : row(i))
            mpcForcesG[static_cast<std::size_t>(term.dof)] -= term.coefficient * force;
    }
}

} // namespace oos::assembly
