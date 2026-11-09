#pragma once

#include "oos/elements/solids/SolidKinematics.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace oos::elements
{

// The places where a solid gives its stresses and strains: the centre, then the corners in the order of the card (the
// rows of SolidHexa, SolidPenta, SolidTetra and SolidPyramid with or without mid-side nodes: 9, 7, 5 and 6 rows).
inline constexpr std::size_t MAX_SOLID_RECOVERY_POINTS = 9;

// Which parts of a SolidResponse a facet needs.
enum class SolidResponseParts : std::uint8_t
{
    Stiffness,
    Recovery
};

// The linear response of one solid element in its material system (PSOLID CORDM), over the translations of its nodes
// in that system (u1, v1, w1, u2, ...): its stiffness matrix, and for each recovery place the operators that give the
// stress and the engineering strain from the nodal displacements (σ = stress[p]·u, ε = strain[p]·u). Each formulation
// (the isoparametric solids, the incompatible-modes SolidHexa) fills one.
//
// Internal modes under a thermal strain. A formulation with internal modes condensed at element level (the default
// 8-node SolidHexa's incompatible modes) also takes a share of an initial strain: with ε_T = a·ΔT, the internal modes α
// carry the load f_α = ∫ Gᵀ·D·a·ΔT dV (G their strain operator), and condensing them adds −K_uα·K_αα⁻¹·f_α to the
// nodal loads and G·K_αα⁻¹·f_α to the strains (Zienkiewicz & Taylor vol. 1, initial strains with static
// condensation). Both are linear in the temperature changes ΔT_b of the nodes: `thermalLoad` (3n × n, row-major, the
// Stiffness part) and `thermalStrain[p]` (6 × n, the Recovery part) give them per unit change of each node.
// `hasThermalModes` is false for a formulation without internal modes (the kernel clears it before each element).
//
// A large value (about 100 kB): a facet allocates one per call and reuses it for every element of its range.
struct SolidResponse
{
    SolidMatrix stiffness;
    std::size_t pointCount = 0;
    std::array<SolidOperator, MAX_SOLID_RECOVERY_POINTS> stress{};
    std::array<SolidOperator, MAX_SOLID_RECOVERY_POINTS> strain{};

    bool hasThermalModes = false;
    std::array<double, MAX_SOLID_DOFS * MAX_SOLID_NODES> thermalLoad{}; // (i, b) at i·n + b
    std::array<SolidOperator, MAX_SOLID_RECOVERY_POINTS> thermalStrain{};
};

} // namespace oos::elements
