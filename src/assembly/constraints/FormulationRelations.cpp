#include "oos/assembly/constraints/FormulationRelations.hpp"

namespace oos::assembly
{

void FormulationRelations::add(const elements::ElementDof& dependent, std::span<const FormulationTerm> terms)
{
    m_dependents.push_back(m_dofMap->dofOf(dependent));
    for (const FormulationTerm& term : terms)
    {
        if (term.coefficient != 0.0)
            m_terms.push_back(ConstraintTerm{m_dofMap->dofOf(term.dof), term.coefficient});
    }
    m_offsets.push_back(m_terms.size());
}

} // namespace oos::assembly
