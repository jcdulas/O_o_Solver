#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"

#include <array>

namespace oos::elements
{

// Output columns of the O_o_Reader data contract (forces.springScalarInline: spring_force). No stress: the card has no stress
// coefficient (CELAS4).
std::vector<std::string> SpringScalarInlineElement::outputColumns(ElementOutput output) const
{
    if (output == ElementOutput::Forces)
        return {"springForce"};
    return {};
}

// SpringScalarInline: force f = K (u1 − u2).
void SpringScalarInlineElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                 const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    if (output != ElementOutput::Forces)
        refuseFacet("stress recovery");
    block.reset(outputColumns(output));
    const Columns columns(view.columns());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Spring& spring = resolve(view, columns, row).required();
        const std::array<double, 1> value{spring.stiffness * spring.connection.relativeMotion(displacements)};
        block.append(row, view.elementId(row), value);
    }
}

std::vector<ColumnForm> SpringScalarInlineElement::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool SpringScalarInlineElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                        RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The spring resolved once: f = K·Σ_d s_d·u_d, its slopes K·s_d.
bool SpringScalarInlineElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const RecoveryOutput> outputs,
                                                 std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view.columns());
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Spring& spring, ElementOutput /*output*/, RecoverySlopes& rowSlopes)
        {
            setLinearRowSlopes(
                spring.connection.presentDofs(), 1, [&](std::size_t /*column*/, std::size_t d)
                { return spring.stiffness * spring.connection.signs[d]; }, rowSlopes);
        });
}

std::size_t SpringScalarInlineElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SPRING_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void SpringScalarInlineElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                               std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
