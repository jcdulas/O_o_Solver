#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/fasteners/fastener/FastenerElement.hpp"

#include <array>

namespace oos::elements
{

// forces.fastener (fx ... mz) and stress.fastener (force_x ... moment_z) of the O_o_Reader data contract.
std::vector<std::string> FastenerElement::outputColumns(ElementOutput output) const
{
    if (output == ElementOutput::Forces)
        return {"fx", "fy", "fz", "mx", "my", "mz"};
    if (output == ElementOutput::Stresses)
        return {"forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"};
    return {};
}

// f = diag(KT, KR)·H·w, w = M·u the motions of A and B; the stress table holds the same values.
void FastenerElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                  const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Fastener fastener = resolve(view, columns, row).required();
        const std::array<double, 12> motions = fastener.connection.pointMotions(displacements);
        const std::array<double, 6> relative = fastener.relativeMotion * motions;
        std::array<double, 6> forces{};
        for (std::size_t c = 0; c < 6; ++c)
            forces[c] = fastener.stiffness[c] * relative[c];
        block.append(row, view.elementId(row), forces);
    }
}

std::vector<ColumnForm> FastenerElement::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool FastenerElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                         RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The fastener resolved once for every output (its piercing points interpolated once): f = diag(KT, KR)·H·M·u, the
// forces and the stresses the same values, so the slopes are K_c·(H·M)(c, d) over the DOFs of the pierced elements'
// grid points (FastenerConnection::dofs, those of the stiffness).
// PERFORMANCE: each recovery of a case resolved the fastener again, its piercing points' interpolation most of it; the
// slopes take one resolution for every case of a block.
bool FastenerElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                  std::span<const RecoveryOutput> outputs,
                                                  std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Fastener& fastener, ElementOutput /*output*/, RecoverySlopes& rowSlopes)
        {
            const FastenerConnection& connection = fastener.connection;
            const std::size_t n = connection.dofCount();
            // H·M (6 × n), summed over the 12 motions in their order.
            std::vector<double> relative(6 * n, 0.0); // bounded by the grid points of two pierced elements
            for (std::size_t c = 0; c < 6; ++c)
                for (std::size_t d = 0; d < n; ++d)
                {
                    double sum = 0.0;
                    for (std::size_t i = 0; i < 12; ++i)
                        sum += fastener.relativeMotion(c, i) * connection.map[i * n + d];
                    relative[c * n + d] = sum;
                }
            const std::vector<ElementDof> dofs = connection.dofs();
            setLinearRowSlopes(
                dofs, 6, [&](std::size_t c, std::size_t d) { return fastener.stiffness[c] * relative[c * n + d]; },
                rowSlopes);
        });
}

std::size_t FastenerElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : FASTENER_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void FastenerElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                                std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
