#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/solids/SolidResponse.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::elements
{

// The principal values of a symmetric 3 × 3 tensor and their directions, in the order and signs of the result tables
// of the solids (the stress and strain tables of SolidHexa, SolidPenta, SolidTetra, SolidPyramid: columns o1, o2, o3 and
// the direction cosines o1x ... o3z):
// - the values come as (largest, smallest, middle), the order of the trigonometric roots of the characteristic cubic
//   2√(J2/3)·cos(θ/3 + 2πk/3), k = 0, 1, 2;
// - the direction of the first and the second has its largest component positive, that of the third its largest
//   component negative (when two components tie, the last decides).
struct PrincipalState
{
    std::array<double, 3> values{};
    std::array<base::Vec3, 3> directions{};
};

// The principal state of a symmetric tensor, by the cyclic Jacobi method (Golub & Van Loan, Matrix Computations,
// §8.5; a fixed number of sweeps at most, the same bits for the same tensor).
[[nodiscard]] PrincipalState principalStateOf(const base::Mat3& tensor);
// The principal states of several tensors (the rows of a block's cases at once), each with the bits of
// principalStateOf alone. Throws std::invalid_argument when the spans differ in size.
void principalStatesOf(std::span<const base::Mat3> tensors, std::span<PrincipalState> states);

// The columns of the solids' stress and strain tables in the O_o_Reader data contract (types 39, 67, 68, 255): oxx
// oyy ozz txy tyz txz o1 o2 o3 von_mises pressure o1x o1y o1z o2x o2y o2z o3x o3y o3z, and for the strains exx ... exz
// e1 e2 e3 von_mises pressure and the same direction cosines; empty for the forces and the curvatures (the solids
// have no force output).
[[nodiscard]] std::vector<std::string> solidOutputColumns(ElementOutput output);

// The form of each column of solidOutputColumns(output) in the displacements (ColumnForm): the six components Linear
// (σ = D·B·u − σ_T, ε = B·u), the principal values, their directions, the von Mises value and the mean pressure Other
// (made from the components by solidStressRow and solidStrainRow).
[[nodiscard]] std::vector<ColumnForm> solidOutputColumnForms(ElementOutput output);

// The vectorised recovery of the solids (ElementFamily::vectorisedRecoveryMinCases): the fewest cases of a block from
// which their rows are made from their recovery matrices (SolidElementKernel::recoverySlopesOfOutputs: the operators
// of the formulation built once). Calibrated on solid blocks (19 200 to 54 000 SolidHexa, SolidPenta, SolidTetra, six
// threads): the per-case recovery takes 0.15 to 0.39 s per case, the recovery by blocks 0.24 to 0.44 s once and 0.04 to
// 0.10 s per case (the principal values of every row): even from 1.3 to 1.9 cases; 4 leaves a margin.
inline constexpr std::size_t SOLID_VECTORISED_RECOVERY_MIN_CASES = 4;

// The columns of a stress or strain row that are not Linear (solidOutputColumnForms), set from its components in every
// row of every block (ElementFamily::completeRecoveredRows) by solidStressRow or solidStrainRow, as appendSolidRows
// makes them: the bits of its rows for the same components.
void completeSolidRows(ElementOutput output, std::span<ElementResultBlock> blocks);

// The 20 values of a stress row from a stress (Voigt order): the components, the principal values and directions,
// the von Mises stress √(½[(σx − σy)² + (σy − σz)² + (σz − σx)²] + 3(τxy² + τyz² + τzx²)) and the mean pressure
// −(σx + σy + σz)/3.
[[nodiscard]] std::array<double, 20> solidStressRow(const Voigt6& stress);
// The 20 values of a strain row from an engineering strain: the components, the principal strains of the tensor
// (γ/2 off the diagonal) and their directions, the equivalent strain (√2/3)·√[(e1 − e2)² + (e2 − e3)² + (e3 − e1)²]
// and −(εx + εy + εz)/3.
[[nodiscard]] std::array<double, 20> solidStrainRow(const Voigt6& strain);

// Appends the rows of one element to `block`: the centre (grid point 0) then each corner (its grid point ID), from
// the operators of `response` and the element's displacements in its material system. `thermalStresses`, when not
// empty (one per place), is the stress subtracted at each place: D·ε_T of the thermal strain, less D·ε_α of the strain
// ε_α the internal modes take from the thermal load (SolidResponse::thermalStrain), so that σ = D·(ε − ε_T) with ε the
// total strain; `thermalModeStrains`, when not empty (one per place), is that ε_α, added to the strains. The strains
// are the total strains, as the shells give them (ShellRecovery).
void appendSolidRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                     std::span<const std::int32_t> cornerIds, const SolidResponse& response,
                     std::span<const double> displacements, ElementOutput output,
                     std::span<const Voigt6> thermalStresses = {}, std::span<const Voigt6> thermalModeStrains = {});

} // namespace oos::elements
