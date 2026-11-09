#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/springs/springScalarProperty/SpringScalarPropertyElement.hpp"

#include <array>

namespace oos::elements
{

// Output columns of the O_o_Reader data contract (forces.springScalarProperty: spring_force; stress.springScalarProperty: spring_stress).
std::vector<std::string> SpringScalarPropertyElement::outputColumns(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"springForce"};
    case ElementOutput::Stresses:
        return {"springStress"};
    case ElementOutput::Strains:
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

// SpringScalarProperty, SpringProp: force f = K (u1 − u2); stress = S · f.
void SpringScalarPropertyElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                 const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Spring& spring = resolve(view, columns, row).required();
        const double force = spring.stiffness * spring.connection.relativeMotion(displacements);
        const std::array<double, 1> value{output == ElementOutput::Forces ? force : spring.stressCoefficient * force};
        block.append(row, view.elementId(row), value);
    }
}

std::vector<ColumnForm> SpringScalarPropertyElement::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool SpringScalarPropertyElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                        RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The spring resolved once for every output: f = K·Σ_d s_d·u_d and the stress S·f, their slopes K·s_d and S·(K·s_d).
bool SpringScalarPropertyElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                 std::span<const RecoveryOutput> outputs,
                                                 std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Spring& spring, ElementOutput output, RecoverySlopes& rowSlopes)
        {
            setLinearRowSlopes(
                spring.connection.presentDofs(), 1,
                [&](std::size_t /*column*/, std::size_t d)
                {
                    const double force = spring.stiffness * spring.connection.signs[d];
                    return output == ElementOutput::Forces ? force : spring.stressCoefficient * force;
                },
                rowSlopes);
        });
}

std::size_t SpringScalarPropertyElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SPRING_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void SpringScalarPropertyElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                               std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
