#pragma once

#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/problem/DesignProblem.hpp"
#include "oos/solutions/sol101/StaticSolution.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::design
{

// The powers of a field a stiffness or a mass is a polynomial of: a thickness enters the membrane and the transverse
// shear as t, the coupling as t², the bending as t³ (the ShellProp membrane, coupling and bending terms); an area, a
// modulus, a density, a spring rate as their first power; a mass as the first power of a thickness, an area, a density.
inline constexpr std::size_t HINT_POWER_COUNT = 4; // the powers 0 to 3
// The scales of the field at which the energies are sampled: four values fit the cubic exactly.
inline constexpr std::array<double, HINT_POWER_COUNT> HINT_SCALES{0.5, 1.0, 1.5, 2.0};
// The largest power an intermediate variable takes (y = x^±1, ±2, ±3).
inline constexpr int HINT_LARGEST_POWER = 3;

// How one designed field (a relation) enters the stiffness and the mass of the elements it reaches, at the design
// point: the share of the strain energy (of the analysed subcases, at the displacements of the analysis) and of the
// mass carried by each power of the field.
struct RelationHint
{
    std::size_t relation = 0;
    std::string label;
    std::array<double, HINT_POWER_COUNT> stiffnessShares{}; // Σ = 1 when the field enters the stiffness at all
    double stiffnessExponent = 0.0;                         // v·(dE/dv)/E: 1 membrane, 3 bending, between for a mix
    std::array<double, HINT_POWER_COUNT> massShares{};
    double massExponent = 0.0; // v·(dm/dv)/m
    double energy = 0.0;       // E at the design point, the weight of the relation in its variable's exponent
    double mass = 0.0;
};

// How one independent variable enters the stiffness and the mass: the exponents of its relations, weighted by their
// energies (masses) and by the elasticity x·(dv/dx)/v of each field (1 for v = c·x).
struct VariableHint
{
    std::size_t column = 0;
    std::string label;
    double stiffnessExponent = 0.0;
    double massExponent = 0.0;
};

// The intermediate variable proposed for each response along each variable: y = x^p, p in ±1, ±2, ±3.
struct ResponseHint
{
    std::int32_t responseId = 0;
    std::string type;               // its DesignConstrScreen type
    std::vector<int> exponents;     // one per independent variable
    std::vector<std::string> forms; // "x", "1/x", "x^3", "1/x^3" ...
};

struct ApproximationHintsResult
{
    std::vector<RelationHint> relations;
    std::vector<VariableHint> variables;
    std::vector<ResponseHint> responses;

    // The exponents of a response (nullptr when it has none).
    [[nodiscard]] const ResponseHint* find(std::int32_t responseId) const noexcept;
};

// The approximation hints of a design point (an optimisation aid beyond gradients): how each relation enters the
// stiffness and the mass, and the intermediate variable in which each response is
// closest to linear. No solve: the strain energies Σ_s u_sᵀ·K_e(s·v)·u_s of the elements a field reaches (and their
// masses) are computed with the field scaled by HINT_SCALES and fitted exactly by a cubic in the scale, whose
// coefficients are the shares of the powers of the field.
//
// The intermediate variable of a response along variable j, y = x_j^p (Vanderplaats, Numerical Optimization
// Techniques for Engineering Design, §7.3, generalised from the reciprocal variables to a power):
// - the displacements of the structure (DISP, COMPLIANCE, MEANDISP, DEFLTHK, ROTATION): p = −round(stiffness exponent
//   of x_j): a displacement under fixed loads goes as K⁻¹, so one variable entering K as x³ (a plate in bending) gives
//   u ∝ x⁻³, exactly linear in y = x⁻³;
// - the forces and stresses (STRESS, STRAIN, FORCE, ESE, SPCFORCE, the composite ones): p = −1 where x_j enters the
//   stiffness (reciprocal variables: exact for a statically determinate structure), else direct;
// - WEIGHT, VOLUME, FRMASS: p = round(mass exponent) (1: linear in a thickness, an area, a density);
// - the equations (DRESP2, DesignResp3): direct (their form is unknown);
// a variable with no stiffness or mass effect, or a power of 0, gives p = 1 (direct). The internal optimiser keeps
// the approximation of DesignOptimParams APRCOD unless asked for the hints (ApproximationKind::Hints).
//
// Thread-safety: run on the calling thread. Ownership: none kept.
class ApproximationHints
{
public:
    // The hints of a kept evaluation (the displacements of every point). Throws as the element families and
    // MassProperties.
    [[nodiscard]] static ApproximationHintsResult compute(const DesignProblem& problem,
                                                          const DesignEvaluation& evaluation,
                                                          base::MemoryBudget& budget,
                                                          const solutions::StaticOptions& analysisOptions = {});

    // "x", "1/x", "x^2" ... for a power.
    [[nodiscard]] static std::string formOf(int power);
};

} // namespace oos::design
