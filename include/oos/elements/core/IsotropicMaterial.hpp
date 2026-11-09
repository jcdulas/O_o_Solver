#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The constants of an isotropic material (MAT1) that the line and shell elements read, as O_o_Reader's materialsTable
// gives them: E, G and NU completed from the two given (G = E / 2(1 + NU) when G is blank...), a constant that stays
// blank after that read as 0 (when only E or only G is given, the other and NU are 0).
// The stress limits ST, SC, SS are 0 when not given; they serve only the margins of safety.
struct IsotropicMaterial
{
    double youngModulus = 0.0;         // E
    double shearModulus = 0.0;         // G
    double density = 0.0;              // RHO, mass per unit volume
    double tensionLimit = 0.0;         // ST
    double compressionLimit = 0.0;     // SC
    double shearLimit = 0.0;           // SS
    double poissonRatio = 0.0;         // NU (the shells' plane-stress matrix)
    double expansion = 0.0;            // A, the thermal expansion coefficient (0 when blank)
    double referenceTemperature = 0.0; // TREF (default 0.0)
};

// The strain of a uniaxial stress state (bars, beams, rods): ε = σ/E (Hooke's law), 0 when E is 0 (the strain is not
// defined). The strain tables give this strain at the stress recovery points of a linear analysis, with the margins of
// the stresses.
[[nodiscard]] inline double uniaxialStrain(double stress, const IsotropicMaterial& material) noexcept
{
    return material.youngModulus > 0.0 ? stress / material.youngModulus : 0.0;
}

// The columns of the materials table that IsotropicMaterial reads, once per facet call (all empty when the snapshot
// has no materials table: every MID is then missing).
struct IsotropicMaterialColumns
{
    base::DictionaryView types;
    std::array<std::span<const double>, 9> constants; // e, g, rho, st, sc, ss, nu, a, tref
    std::array<base::ValidityView, 9> isGiven;        // their validity

    explicit IsotropicMaterialColumns(const ElementBatchView& view);
};

// The IsoMat of ID `matId` for the element of batch row `row`; a missing material or a type other than IsoMat is a
// problem on the property field `matIdField` ("MID").
[[nodiscard]] Resolution<IsotropicMaterial> resolveIsotropicMaterial(const ElementBatchView& view,
                                                                     const IsotropicMaterialColumns& columns,
                                                                     std::int64_t row, std::int32_t matId,
                                                                     const char* matIdField);

} // namespace oos::elements
