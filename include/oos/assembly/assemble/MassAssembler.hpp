#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/tables/ParamTable.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparseAssembler.hpp"

#include <cstdint>
#include <optional>

namespace oos::assembly
{

// The mass parameters of a subcase (PARAM COUPMASS, PARAM WTMASS).
struct MassAssemblyOptions
{
    static constexpr double DEFAULT_WEIGHT_TO_MASS = 1.0; // default of PARAM WTMASS

    // COUPMASS: a value > 0 asks for the coupled mass of the elements that have one; the default -1 the lumped.
    elements::MassFormulation formulation = elements::MassFormulation::Lumped;
    // WTMASS: the factor of every term of the structural mass matrix (mass units of the deck to consistent units).
    double weightToMass = DEFAULT_WEIGHT_TO_MASS;

    // The options of a subcase from the PARAM table (a Case Control PARAM overrides the Bulk Data one); `params` may be
    // null (the defaults). A value of the wrong kind is reported ("bad-param") and the default kept.
    [[nodiscard]] static MassAssemblyOptions fromParams(const input::ParamTable* params,
                                                        std::optional<std::int32_t> subcase,
                                                        base::DiagnosticSink& sink);
};

// The mass matrix Mgg of the g set: the element mass matrices (lumped or coupled) of
// every batch whose family has the mass facet, with the non-structural mass of the NSM set, added by
// linalg::SparseAssembler on the node pattern of the model, then multiplied by WTMASS (PARAM WTMASS applies to the
// assembled mass, the elements give their mass in the units of the deck). The order of the sums is that of
// StiffnessAssembler.
//
// Thread-safety: assemble() is const. Ownership: borrows the set, the DOF map and the shares, which must outlive it.
class MassAssembler
{
public:
    MassAssembler(const ElementSet& set, const DofMap& dofMap, const NonStructuralMassShares& shares,
                  MassAssemblyOptions options, linalg::AssemblyOptions assembly = {})
        : m_set(&set), m_dofMap(&dofMap), m_shares(&shares), m_options(options), m_assembly(assembly)
    {
    }

    // Adds Mgg to `target`, which must be zero when called (the WTMASS factor scales every value of the target).
    // Precondition: as StiffnessAssembler::assemble.
    void assemble(linalg::BlockSparseMatrix& target, base::ThreadPool* pool = nullptr,
                  const base::CancellationToken* cancellation = nullptr) const;

    [[nodiscard]] const MassAssemblyOptions& options() const noexcept { return m_options; }

private:
    const ElementSet* m_set;
    const DofMap* m_dofMap;
    const NonStructuralMassShares* m_shares;
    MassAssemblyOptions m_options;
    linalg::AssemblyOptions m_assembly;
};

} // namespace oos::assembly
