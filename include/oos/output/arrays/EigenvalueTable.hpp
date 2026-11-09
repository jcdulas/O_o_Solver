#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// One row of an eigenvalue table: the real eigenvalues (O_o_Reader's result kind "eigenvalues" with the components
// extraction_order, eigenvalue, radians, cycles, generalized_mass, generalized_stiffness).
//
// For a vibration mode: λ = ω², radians = √|λ|, cycles = radians / 2π, the generalised mass φᵀ·M·φ and stiffness
// φᵀ·K·φ = λ·φᵀ·M·φ of the normalised vector. For a buckling mode (SOL 105): λ the load factor, radians = √|λ| and
// cycles = radians / 2π (kept so that both kinds of table have the same columns), the generalised stiffness φᵀ·K·φ and
// the "generalised mass" φᵀ·(−K_d)·φ = φᵀ·K·φ / λ.
struct EigenvalueRow
{
    std::int32_t mode = 0;            // 1, 2, ... in the order of the table
    std::int32_t extractionOrder = 0; // the order in which the solver gives the root (the mode number here)
    double eigenvalue = 0.0;
    double radians = 0.0;
    double cycles = 0.0;
    double generalizedMass = 0.0;
    double generalizedStiffness = 0.0;
};

// The eigenvalue table of one subcase of a modal (SOL 103) or buckling (SOL 105) solution.
//
// Thread-safety: an immutable value. Ownership: plain data (one row per mode: bounded by the modes requested).
struct EigenvalueTable
{
    std::int32_t subcase = 1;
    std::string analysis = "modes"; // "modes" or "buckling"
    std::vector<EigenvalueRow> rows;
};

// The modal effective mass of one mode (the Case Control request MEFFMASS), for the six rigid-body motions D of the
// analysis set about a reference point: the participation factor Γ_ij = φ_iᵀ·M·D_j / m_i, the effective mass
// M_ij = (φ_iᵀ·M·D_j)² / m_i (m_i = φ_iᵀ·M·φ_i), and its fraction of the rigid-body mass (Dᵀ·M·D)_jj.
struct ModalEffectiveMassRow
{
    std::int32_t mode = 0;
    double cycles = 0.0;
    std::array<double, 6> participation{};
    std::array<double, 6> effectiveMass{};
    std::array<double, 6> fraction{};
    std::array<double, 6> fractionSum{}; // the fractions summed over this mode and the ones before it
};

// The modal effective mass table of one subcase (Case Control MEFFMASS).
//
// Thread-safety: an immutable value. Ownership: plain data (one row per mode).
struct ModalEffectiveMassTable
{
    std::int32_t subcase = 1;
    std::int32_t referencePoint = 0; // ReferenceNode, 0 for the origin of the basic system
    // The rigid-body mass of the analysis set, Dᵀ·M·D (6 × 6, row-major), in the units of the mass matrix (WTMASS
    // applied).
    std::array<double, 36> rigidBodyMass{};
    std::vector<ModalEffectiveMassRow> rows;
    std::array<double, 6> totalEffectiveMass{}; // the effective masses summed over the modes
};

} // namespace oos::output
