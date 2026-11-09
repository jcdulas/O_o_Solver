#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"

#include <array>

namespace oos::elements
{

// stress.bushing1D and strain.bushing1D of the O_o_Reader data contract (the same record).
std::vector<std::string> Bushing1DElement::outputColumns(ElementOutput output) const
{
    if (output == ElementOutput::Stresses || output == ElementOutput::Strains)
        return {"elementForce", "axialDisplacement", "axialVelocity", "axialStress",
                "axialStrain",  "plasticStrain",     "isFailed"};
    return {};
}

// e = h·u, f = K·e; BushProp1D: stress SA·f, strain SE·e. A linear static analysis has no velocity, no plastic strain
// and no failure.
void Bushing1DElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        std::array<ElementDof, 6> storage;
        const std::span<const ElementDof> dofs = translationDofs(bush, storage);
        double axial = 0.0;
        for (std::size_t i = 0; i < dofs.size(); ++i)
            axial += bush.axialMap[i] * displacements.valueAt(dofs[i]);
        const double force = bush.stiffness * axial;
        const std::array<double, 7> values{
            force, axial, 0.0, bush.stressCoefficient * force, bush.strainCoefficient * axial, 0.0, 0.0};
        block.append(row, view.elementId(row), values);
    }
}

std::vector<ColumnForm> Bushing1DElement::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool Bushing1DElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                       RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The bush resolved once for every output: e = h·u, the row (K·e, e, 0, SA·K·e, SE·e, 0, 0), its slopes over the
// translations of the stiffness (translationDofs) h_d times those coefficients.
bool Bushing1DElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                                std::span<const RecoveryOutput> outputs,
                                                std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Bush& bush, ElementOutput /*output*/, RecoverySlopes& rowSlopes)
        {
            std::array<ElementDof, 6> storage;
            const std::span<const ElementDof> dofs = translationDofs(bush, storage);
            setLinearRowSlopes(
                dofs, 7,
                [&](std::size_t c, std::size_t d)
                {
                    const double axial = bush.axialMap[d];
                    switch (c)
                    {
                    case 0:
                        return bush.stiffness * axial;
                    case 1:
                        return axial;
                    case 3:
                        return bush.stressCoefficient * (bush.stiffness * axial);
                    case 4:
                        return bush.strainCoefficient * axial;
                    default: // no velocity, plastic strain or failure in a linear static analysis
                        return 0.0;
                    }
                },
                rowSlopes);
        });
}

std::size_t Bushing1DElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SPRING_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void Bushing1DElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                              std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
