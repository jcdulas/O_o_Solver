#include "oos/assembly/dofs/Uset.hpp"

#include <cstdio>
#include <string>

namespace oos::assembly
{

namespace
{

constexpr int MEMBERS_PER_LINE = 10;
constexpr std::size_t ENTRY_WIDTH = 11;

void writeHeader(std::ostream& out, DofSet set)
{
    out << "Set " << nameOf(set) << '\n';
    out << "                ";
    for (int column = 1; column <= MEMBERS_PER_LINE; ++column)
    {
        const std::string label = std::to_string(column);
        out << std::string(ENTRY_WIDTH - label.size(), ' ') << label;
    }
    out << '\n';
}

} // namespace

void Uset::write(std::ostream& out, const DofSets& dofSets, const DofMap& dofMap, std::span<const DofSet> sets)
{
    // A layout of this project's own: a title, then each set.
    out << "Degrees of freedom of the sets, in the order of the g-set\n";
    for (const DofSet set : sets)
    {
        writeHeader(out, set);
        std::int64_t position = 0;
        for (base::DofIndex dof = 0; dof < dofSets.gSetSize(); ++dof)
        {
            if (!dofSets.contains(set, dof))
                continue;
            if (position % MEMBERS_PER_LINE == 0)
            {
                if (position > 0)
                    out << '\n';
                char prefix[24];
                std::snprintf(prefix, sizeof prefix, "%15lld=", static_cast<long long>(position + 1));
                out << prefix;
            }
            const DofLabel label = dofMap.labelOf(dof);
            std::string entry = std::to_string(dofMap.pointIdOf(dof));
            entry.append(1, '-').append(std::to_string(static_cast<int>(label.component)));
            out << std::string(entry.size() < ENTRY_WIDTH ? ENTRY_WIDTH - entry.size() : 1, ' ') << entry;
            ++position;
        }
        out << '\n';
    }
}

} // namespace oos::assembly
