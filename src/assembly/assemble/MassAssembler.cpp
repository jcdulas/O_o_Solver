#include "oos/assembly/assemble/MassAssembler.hpp"

#include "ElementMatrixSource.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"

namespace oos::assembly
{

MassAssemblyOptions MassAssemblyOptions::fromParams(const input::ParamTable* params,
                                                    std::optional<std::int32_t> subcase, base::DiagnosticSink& sink)
{
    MassAssemblyOptions options;
    if (const std::optional<std::int32_t> coupled = findIntegerParam(params, "CoupledMass", subcase, sink))
        options.formulation = *coupled > 0 ? elements::MassFormulation::Coupled : elements::MassFormulation::Lumped;
    if (const std::optional<double> factor = findRealParam(params, "WeightToMass", subcase, sink))
    {
        if (*factor > 0.0)
            options.weightToMass = *factor;
        else
            sink.report(badParamDiagnostic("WeightToMass", "a positive real number"));
    }
    return options;
}

void MassAssembler::assemble(linalg::BlockSparseMatrix& target, base::ThreadPool* pool,
                             const base::CancellationToken* cancellation) const
{
    const MassAssemblyOptions options = m_options;
    const NonStructuralMassShares* shares = m_shares;
    const ElementMatrixSource source(
        *m_set, m_set->batchesWith(elements::Facet::Mass), *m_dofMap,
        [options, shares](std::size_t batchIndex, const AssemblyBatch& batch, elements::ElementRange range,
                          elements::ElementMatrixBlock& block)
        {
            elements::MassOptions massOptions;
            massOptions.formulation = options.formulation;
            massOptions.addedNonStructuralMass = shares->sharesOf(batchIndex);
            batch.family->computeMass(batch.view, range, massOptions, block);
        });
    linalg::SparseAssembler(m_assembly)
        .assemble(
            source.elementCount(), [&source](std::size_t begin, std::size_t end, linalg::ContributionBlock& block)
            { source.writeContributions(begin, end, block); }, target, pool, cancellation);
    // PARAM WTMASS multiplies the structural mass matrix once assembled.
    if (m_options.weightToMass != 1.0)
        for (double& value : target.values())
            value *= m_options.weightToMass;
}

} // namespace oos::assembly
