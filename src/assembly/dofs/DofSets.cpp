#include "oos/assembly/dofs/DofSets.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

constexpr std::uint8_t UNASSIGNED = 0xFF;

[[nodiscard]] std::uint8_t code(ExclusiveSet set) noexcept
{
    return static_cast<std::uint8_t>(set);
}

[[nodiscard]] std::string componentText(const DofLabel& label)
{
    return std::to_string(static_cast<int>(label.component));
}

base::Diagnostic overlapDiagnostic(const DofMap& dofMap, base::DofIndex dof, ExclusiveSet kept, ExclusiveSet refused)
{
    const DofLabel label = dofMap.labelOf(dof);
    base::Diagnostic diagnostic;
    diagnostic.code = "dof-in-exclusive-sets";
    diagnostic.card = label.kind == input::PointKind::Scalar ? "ScalarPoint" : "Node";
    diagnostic.id = dofMap.pointIdOf(dof);
    diagnostic.field = componentText(label);
    diagnostic.message = base::concatenate(
        {"component ", componentText(label), " of point ", std::to_string(diagnostic.id), " is in the set ",
         nameOf(kept), " and in the set ", nameOf(refused),
         ", which are mutually exclusive (degree-of-freedom sets); remove it from one of the cards that put it "
         "there (a dependent DOF of an MultiPointConstr or rigid element cannot be constrained, nor a FreeBodySupport "
         "DOF)"});
    return diagnostic;
}

} // namespace

std::string_view nameOf(DofSet set) noexcept
{
    switch (set)
    {
    case DofSet::G:
        return "G";
    case DofSet::M:
        return "M";
    case DofSet::N:
        return "N";
    case DofSet::S:
        return "S";
    case DofSet::Sb:
        return "SB";
    case DofSet::Sg:
        return "SG";
    case DofSet::F:
        return "F";
    case DofSet::O:
        return "O";
    case DofSet::A:
        return "A";
    case DofSet::R:
        return "R";
    case DofSet::L:
        return "L";
    }
    return "?";
}

std::string_view nameOf(ExclusiveSet set) noexcept
{
    switch (set)
    {
    case ExclusiveSet::M:
        return "M";
    case ExclusiveSet::Sb:
    case ExclusiveSet::AutoSpc:
        return "SB";
    case ExclusiveSet::Sg:
        return "SG";
    case ExclusiveSet::O:
        return "O";
    case ExclusiveSet::R:
        return "R";
    case ExclusiveSet::B:
        return "B";
    }
    return "?";
}

bool isMemberOf(ExclusiveSet exclusive, DofSet set) noexcept
{
    const bool isS =
        exclusive == ExclusiveSet::Sb || exclusive == ExclusiveSet::Sg || exclusive == ExclusiveSet::AutoSpc;
    const bool isA = exclusive == ExclusiveSet::R || exclusive == ExclusiveSet::B;
    const bool isF = isA || exclusive == ExclusiveSet::O;
    switch (set)
    {
    case DofSet::G:
        return true;
    case DofSet::M:
        return exclusive == ExclusiveSet::M;
    case DofSet::N:
        return exclusive != ExclusiveSet::M;
    case DofSet::S:
        return isS;
    case DofSet::Sb:
        return exclusive == ExclusiveSet::Sb || exclusive == ExclusiveSet::AutoSpc;
    case DofSet::Sg:
        return exclusive == ExclusiveSet::Sg;
    case DofSet::F:
        return isF;
    case DofSet::O:
        return exclusive == ExclusiveSet::O;
    case DofSet::A:
        return isA;
    case DofSet::R:
        return exclusive == ExclusiveSet::R;
    case DofSet::L:
        return exclusive == ExclusiveSet::B;
    }
    return false;
}

DofSets::DofSets(const DofMap& dofMap, const DofSetInputs& inputs, base::MemoryBudget& budget,
                 base::DiagnosticSink& sink)
    : m_budget(&budget), m_exclusive(budget, static_cast<std::size_t>(dofMap.gSetSize()), "DOF sets")
{
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
        m_exclusive[dof] = UNASSIGNED;
    // The order of the passes is the priority of the sets when they overlap (class comment).
    for (const base::DofIndex dof : inputs.m)
        m_exclusive[static_cast<std::size_t>(dof)] = code(ExclusiveSet::M);
    const auto assign = [&](std::span<const base::DofIndex> dofs, ExclusiveSet set)
    {
        for (const base::DofIndex dof : dofs)
        {
            std::uint8_t& current = m_exclusive[static_cast<std::size_t>(dof)];
            if (current == UNASSIGNED || current == code(set))
                current = code(set);
            else if (current == code(ExclusiveSet::Sg) && set == ExclusiveSet::Sb)
                continue; // sb and sg are both parts of s; the permanent constraint is kept
            else
                sink.report(overlapDiagnostic(dofMap, dof, static_cast<ExclusiveSet>(current), set));
        }
    };
    assign(inputs.sg, ExclusiveSet::Sg);
    assign(inputs.sb, ExclusiveSet::Sb);
    assign(inputs.r, ExclusiveSet::R);
    assign(inputs.o, ExclusiveSet::O);
    if (!inputs.o.empty())
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "unsupported-omitted-dofs";
        diagnostic.card = "OmittedDofSet";
        diagnostic.id = dofMap.pointIdOf(inputs.o.front());
        diagnostic.message = base::concatenate(
            {"the model omits ", std::to_string(inputs.o.size()),
             " degrees of freedom (OmittedDofSet, OmittedDofSet1, or AnalysisDofSet, AnalysisDofSet1 which omit the "
             "others); the o-set reduction is not "
             "supported by this version of O_o_Solver: remove these cards (the condensation of components replaces "
             "them)"});
        sink.report(std::move(diagnostic));
    }
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
    {
        if (m_exclusive[dof] == UNASSIGNED)
            m_exclusive[dof] = code(ExclusiveSet::B);
    }
    countMembers();
}

DofSets::DofSets(base::BudgetedBuffer<std::uint8_t> exclusive, base::MemoryBudget& budget)
    : m_budget(&budget), m_exclusive(std::move(exclusive))
{
    countMembers();
}

void DofSets::countMembers() noexcept
{
    m_counts.fill(0);
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
        ++m_counts[m_exclusive[dof]];
}

DofSets DofSets::withAutoSpc(std::span<const base::DofIndex> autoSpcDofs) const
{
    base::BudgetedBuffer<std::uint8_t> exclusive(*m_budget, m_exclusive.size(), "DOF sets");
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
        exclusive[dof] = m_exclusive[dof];
    for (const base::DofIndex dof : autoSpcDofs)
    {
        std::uint8_t& current = exclusive[static_cast<std::size_t>(dof)];
        if (current != code(ExclusiveSet::B))
            throw std::invalid_argument(
                base::concatenate({"DofSets::withAutoSpc: expected DOFs of the a set (exclusive set B), found DOF ",
                                   std::to_string(dof), " in the set ", nameOf(static_cast<ExclusiveSet>(current)),
                                   "; pass the DOFs GridPointSingularities chose on these sets"}));
        current = code(ExclusiveSet::AutoSpc);
    }
    return DofSets(std::move(exclusive), *m_budget);
}

DofSets DofSets::withSupports(std::span<const base::DofIndex> supports) const
{
    base::BudgetedBuffer<std::uint8_t> exclusive(*m_budget, m_exclusive.size(), "DOF sets");
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
        exclusive[dof] = m_exclusive[dof] == code(ExclusiveSet::R) ? code(ExclusiveSet::B) : m_exclusive[dof];
    for (const base::DofIndex dof : supports)
    {
        std::uint8_t& current = exclusive[static_cast<std::size_t>(dof)];
        if (current != code(ExclusiveSet::B))
            throw std::invalid_argument(
                base::concatenate({"DofSets::withSupports: expected DOFs of the a set (exclusive set B or R), found "
                                   "DOF ",
                                   std::to_string(dof), " in the set ", nameOf(static_cast<ExclusiveSet>(current)),
                                   "; choose the supports among the free DOFs of these sets"}));
        current = code(ExclusiveSet::R);
    }
    return DofSets(std::move(exclusive), *m_budget);
}

std::int64_t DofSets::count(DofSet set) const noexcept
{
    std::int64_t total = 0;
    for (std::size_t exclusive = 0; exclusive < m_counts.size(); ++exclusive)
    {
        if (isMemberOf(static_cast<ExclusiveSet>(exclusive), set))
            total += m_counts[exclusive];
    }
    return total;
}

base::BudgetedBuffer<base::DofIndex> DofSets::members(DofSet set, base::MemoryBudget& budget) const
{
    base::BudgetedBuffer<base::DofIndex> dofs(budget, static_cast<std::size_t>(count(set)),
                                              base::concatenate({"members of the set ", nameOf(set)}));
    std::size_t next = 0;
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
    {
        if (isMemberOf(static_cast<ExclusiveSet>(m_exclusive[dof]), set))
            dofs[next++] = static_cast<base::DofIndex>(dof);
    }
    return dofs;
}

base::BudgetedBuffer<std::uint8_t> DofSets::eliminationFlags(DofSet kept, base::MemoryBudget& budget) const
{
    base::BudgetedBuffer<std::uint8_t> flags(budget, m_exclusive.size(), "eliminated DOF flags");
    for (std::size_t dof = 0; dof < m_exclusive.size(); ++dof)
        flags[dof] = isMemberOf(static_cast<ExclusiveSet>(m_exclusive[dof]), kept) ? 0 : 1;
    return flags;
}

} // namespace oos::assembly
