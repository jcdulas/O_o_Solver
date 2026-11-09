#pragma once

#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/RecoverySlopes.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements
{

// The vectorised recovery of the families whose rows are linear maps of their displacements: the springs (CELAS1 to
// SpringScalarInline, Bushing, Bushing1D, Gap2) and the fasteners (CFAST, Weld's forces) give one row per element
// whose every column is a fixed matrix times the element's DOFs, value(0) zero (a static analysis' spring has no
// thermal or initial force here). Their slopes are that matrix, in closed form from one resolution of the element
// (ElementFamily::recoverySlopes, recoverySlopesOfOutputs), and their rows need no completion.

// The fewest cases of a block from which the springs' rows (CELAS1 to SpringScalarInline, Bushing, Bushing1D, Gap2) are
// made from their slopes (ElementFamily::vectorisedRecoveryMinCases). Calibrated on a spring network (81 000 springs of
// every kind, six threads, every output): case by case 19 ms a case, by blocks 39 ms once and 5.4 ms a case, even
// from 2.8 cases; 4 leaves a margin for a block asking for fewer outputs (the slopes then shared by fewer rows).
inline constexpr std::size_t SPRING_VECTORISED_RECOVERY_MIN_CASES = 4;
// The same for the fasteners (CFAST, Weld): their resolution (the piercing points' interpolation) is most of a
// recovery, which a case recovered alone pays for each output (two) and the recovery by blocks three times once (the
// rows at rest of each output, the slopes): even from about 1.5 cases. Measured on an assembly of 3 000
// RigidInterpolation-Bushing-RigidInterpolation, Fastener and Weld joints: their tables 0.26 s for 32 cases
// case by case, 0.03 s by blocks.
inline constexpr std::size_t FASTENER_VECTORISED_RECOVERY_MIN_CASES = 2;

// Every one of `columnCount` columns Linear (ColumnForm::linear).
[[nodiscard]] inline std::vector<ColumnForm> linearColumnForms(std::size_t columnCount)
{
    return std::vector<ColumnForm>(columnCount, ColumnForm::linear());
}

// Sets `slopes` to one row (at the default place: ResultPlace{}, as the families append it) of `columnCount` columns
// over `dofs`, S(0, c, d) = slopeOf(c, d).
template <typename SlopeOf>
void setLinearRowSlopes(std::span<const ElementDof> dofs, std::size_t columnCount, SlopeOf&& slopeOf,
                        RecoverySlopes& slopes)
{
    slopes.reset(dofs, columnCount);
    slopes.appendRow(ResultPlace{});
    for (std::size_t c = 0; c < columnCount; ++c)
    {
        const std::span<double> row = slopes.slopes(0, c);
        for (std::size_t d = 0; d < dofs.size(); ++d)
            row[d] = slopeOf(c, d);
    }
}

// recoverySlopesOfOutputs of a family whose element outputs come from one resolved element: `resolveElement()` once,
// then `slopesOf(element, output, slopes[k])` for each output; false when an output is a layer output (the family has
// none).
template <typename Resolve, typename SlopesOf>
bool linearRowSlopesOfOutputs(std::span<const RecoveryOutput> outputs, std::span<RecoverySlopes> slopes,
                              Resolve&& resolveElement, SlopesOf&& slopesOf)
{
    for (const RecoveryOutput& output : outputs)
        if (output.isLayer)
            return false;
    const auto element = resolveElement();
    for (std::size_t k = 0; k < outputs.size(); ++k)
        slopesOf(element, outputs[k].output, slopes[k]);
    return true;
}

} // namespace oos::elements
