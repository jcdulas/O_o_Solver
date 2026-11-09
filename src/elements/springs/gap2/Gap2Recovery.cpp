#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/springs/gap2/Gap2Element.hpp"

#include <array>

namespace oos::elements
{

// forces.gap2 of the O_o_Reader data contract.
std::vector<std::string> Gap2Element::outputColumns(ElementOutput output) const
{
    if (output == ElementOutput::Forces)
        return {"fx", "sfy", "sfz", "u", "v", "w", "sv", "sw"};
    return {};
}

// Δ = H·u; fx = −k_x·Δx (compression positive), sfy = k_t·Δy, sfz = k_t·Δz, u, v, w = Δ, no slip (Gap2Element.hpp).
void Gap2Element::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                             const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Gap& gap = resolve(view, columns, row).required();
        std::array<double, 6> u{};
        for (int c = 0; c < 3; ++c)
        {
            u[static_cast<std::size_t>(c)] = displacements.valueAt(ElementDof::grid(gap.rowA, c + 1));
            u[static_cast<std::size_t>(3 + c)] = displacements.valueAt(ElementDof::grid(gap.rowB, c + 1));
        }
        const std::array<double, 3> relative = gap.relativeMotion * u;
        const std::array<double, 8> values{-gap.stiffness[0] * relative[0],
                                           gap.stiffness[1] * relative[1],
                                           gap.stiffness[2] * relative[2],
                                           relative[0],
                                           relative[1],
                                           relative[2],
                                           0.0,
                                           0.0};
        block.append(row, view.elementId(row), values);
    }
}

std::vector<ColumnForm> Gap2Element::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool Gap2Element::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                    RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The gap resolved once: Δ = H·u over the translations of GA and GB (the DOFs of the stiffness), the row (−k_x·Δx,
// k_t·Δy, k_t·Δz, Δ, 0, 0) of its initial state, its slopes those coefficients times H(i, d).
bool Gap2Element::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                             std::span<const RecoveryOutput> outputs,
                                             std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Gap& gap, ElementOutput /*output*/, RecoverySlopes& rowSlopes)
        {
            const std::array<ElementDof, 6> dofs{ElementDof::grid(gap.rowA, 1), ElementDof::grid(gap.rowA, 2),
                                                 ElementDof::grid(gap.rowA, 3), ElementDof::grid(gap.rowB, 1),
                                                 ElementDof::grid(gap.rowB, 2), ElementDof::grid(gap.rowB, 3)};
            setLinearRowSlopes(
                dofs, 8,
                [&](std::size_t c, std::size_t d)
                {
                    switch (c)
                    {
                    case 0:
                        return -gap.stiffness[0] * gap.relativeMotion(0, d);
                    case 1:
                    case 2:
                        return gap.stiffness[c] * gap.relativeMotion(c, d);
                    case 3:
                    case 4:
                    case 5:
                        return gap.relativeMotion(c - 3, d);
                    default: // no slip in the initial state
                        return 0.0;
                    }
                },
                rowSlopes);
        });
}

std::size_t Gap2Element::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SPRING_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void Gap2Element::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                           std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
