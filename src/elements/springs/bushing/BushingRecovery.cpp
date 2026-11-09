#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/springs/bushing/BushingElement.hpp"

#include <array>

namespace oos::elements
{

// Output columns of the O_o_Reader data contract: forces.bushing (fx, fy, fz, mx, my, mz, in the element system),
// stress.bushing and strain.bushing (tx, ty, tz, rx, ry, rz).
std::vector<std::string> BushingElement::outputColumns(ElementOutput output) const
{
    switch (output)
    {
    case ElementOutput::Forces:
        return {"fx", "fy", "fz", "mx", "my", "mz"};
    case ElementOutput::Stresses:
    case ElementOutput::Strains:
        return {"tx", "ty", "tz", "rx", "ry", "rz"};
    case ElementOutput::StrainCurvatures:
        break;
    }
    return {};
}

// Bushing: e = H·u the element displacements (Δu, Δθ) at the spring point and f = diag(K1..K6)·e the spring forces
// and moments, in the element system; BushProp RCV: stresses SA·f (1-3), ST·f (4-6), strains EA·e (1-3), ET·e (4-6).
void BushingElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                              const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Bush& bush = resolve(view, columns, row).required();
        std::array<double, 12> u{};
        for (int k = 0; k < 6; ++k)
        {
            u[static_cast<std::size_t>(k)] = displacements.valueAt(ElementDof::grid(bush.rowA, k + 1));
            if (!bush.isGrounded)
                u[static_cast<std::size_t>(6 + k)] = displacements.valueAt(ElementDof::grid(bush.rowB, k + 1));
        }
        const std::array<double, 6> motion = bush.relativeMotion * u;
        std::array<double, 6> values{};
        for (std::size_t c = 0; c < 6; ++c)
        {
            const bool isTranslation = c < 3;
            switch (output)
            {
            case ElementOutput::Forces:
                values[c] = bush.stiffness[c] * motion[c];
                break;
            case ElementOutput::Stresses:
                values[c] = (isTranslation ? bush.recovery[0] : bush.recovery[1]) * bush.stiffness[c] * motion[c];
                break;
            default: // Strains
                values[c] = (isTranslation ? bush.recovery[2] : bush.recovery[3]) * motion[c];
                break;
            }
        }
        block.append(row, view.elementId(row), values);
    }
}

std::vector<ColumnForm> BushingElement::outputColumnForms(ElementOutput output) const
{
    return linearColumnForms(outputColumns(output).size());
}

bool BushingElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                     RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The bush resolved once for every output: the rows of recover() are diag(c)·H·u, c = K (forces), SA·K or ST·K (the
// stresses of the translations and rotations), EA or ET (the strains), so their slopes are c_i·H(i, d), over the DOFs
// of the stiffness (GA 1-6, then GB 1-6 unless grounded).
bool BushingElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                              std::span<const RecoveryOutput> outputs,
                                              std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Bush& bush, ElementOutput output, RecoverySlopes& rowSlopes)
        {
            const std::array<ElementDof, 6> dofsA = gridPointDofs(bush.rowA);
            const std::array<ElementDof, 6> dofsB = gridPointDofs(bush.rowB);
            std::array<ElementDof, 12> dofs;
            for (std::size_t k = 0; k < 6; ++k)
            {
                dofs[k] = dofsA[k];
                dofs[6 + k] = dofsB[k];
            }
            const std::size_t dofCount = bush.isGrounded ? 6 : 12;
            setLinearRowSlopes(
                std::span<const ElementDof>(dofs.data(), dofCount), 6,
                [&](std::size_t c, std::size_t d)
                {
                    const bool isTranslation = c < 3;
                    switch (output)
                    {
                    case ElementOutput::Forces:
                        return bush.stiffness[c] * bush.relativeMotion(c, d);
                    case ElementOutput::Stresses:
                        return (isTranslation ? bush.recovery[0] : bush.recovery[1]) * bush.stiffness[c] *
                               bush.relativeMotion(c, d);
                    default: // Strains
                        return (isTranslation ? bush.recovery[2] : bush.recovery[3]) * bush.relativeMotion(c, d);
                    }
                },
                rowSlopes);
        });
}

std::size_t BushingElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : SPRING_VECTORISED_RECOVERY_MIN_CASES;
}

// Every column Linear: nothing to complete.
void BushingElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                            std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
}

} // namespace oos::elements
