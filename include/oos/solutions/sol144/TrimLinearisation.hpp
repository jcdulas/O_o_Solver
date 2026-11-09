#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions
{

// The trimmed state of the subcases of a static aeroelastic run as a function of the structure, kept after the run for
// the design derivatives of SOL 200 SAERO (TrimOptions::keepsLinearisation: the derivative of the trim system). The
// structure K, M of a design point enters the trim of TrimSolution; the aerodynamic matrices (the doublet lattice D,
// the splines T and S, the normalwash of the variables) and the support modes D_r do not depend on a design of the
// structure. With the trim equations of TrimSolution,
//     K·u = q·T·Δcp − M·D_r·ü_r,    D·Δcp = W·x + AEQR·S·u,    D_rᵀ·(q·T·Δcp − M·D_r·ü_r) = 0,
// differentiating along a design direction p (∂K, ∂M the derivatives of the structure's matrices) gives the same
// equations for (du, dΔcp, dx, dü_r) with the pseudo-load Q = −∂K·u − ∂M·D_r·ü_r on the structure and the term
// −D_rᵀ·∂M·D_r·ü_r in the equilibrium of the supports, the fixed variables held (dx = 0):
//     dΔcp = Σ_x Δcp_x·dx_x + δΔcp(Q),    δΔcp(Q) = (D − AEQR·q·E)⁻¹·AEQR·S·K⁻¹·Q,    E = S·K⁻¹·T,
//     Σ_x A_x·dx_x + q·D_rᵀ·T·δΔcp(Q) − D_rᵀ·∂M·D_r·ü_r = 0     (A_x the column of variable x of the trim equations),
//     dü_r = Σ_x ü_r(x)·dx_x,    du = K⁻¹·(q·T·dΔcp − M·D_r·dü_r + Q).
// The elastic coefficients of the stability derivatives of a variable x are linear in its pressures Δcp_x, which
// solve the same equations with x = 1 and the supports held (restrained) or free (unrestrained), so that
//     restrained:    dΔcp_x = δΔcp(−∂K·u_x − ∂M·D_r·ü_r(x)),    u_x = K⁻¹·(q·T·Δcp_x − M·D_r·ü_r(x)),
//     unrestrained:  dΔcp_x = (D − AEQR·q·E_R)⁻¹·AEQR·S·K⁻¹·(−∂K·u_x − R·∂M·D_r·a_x),
//                    u_x = K⁻¹·R·q·T·Δcp_x,    a_x = m_rr⁻¹·D_rᵀ·q·T·Δcp_x,    R = I − M·D_r·m_rr⁻¹·D_rᵀ,
// (E_R = S·K⁻¹·R·T, m_rr = D_rᵀ·M·D_r: R carries the derivative of the relief, dR·P = −R·∂M·D_r·m_rr⁻¹·D_rᵀ·P).
// The caller forms −∂K·u and −∂M·w for the g-set vectors this class gives (Directions): the element derivatives are
// the design layer's; each derivative then costs one structural solve on the factor of the run (two with du).
//
// Memory: the run's parts stay reserved in its memory budget (the l-set factor, the flexibility of the boxes, the
// factors of the pressures, the trimmed displacements) until the last owner drops the linearisation; the budget must
// outlive it. The solves are serial (the run's thread pool may be gone): the same bits as the run for a given kernel.
//
// Thread-safety: none (a derivative solves on the kept factor). Ownership: owns the run's parts; borrows the snapshot
// of the run, which must outlive it.
class TrimLinearisation
{
public:
    struct State; // the parts of the run it keeps (TrimSolution)

    explicit TrimLinearisation(std::unique_ptr<State> state);
    TrimLinearisation(const TrimLinearisation&) = delete;
    TrimLinearisation& operator=(const TrimLinearisation&) = delete;
    TrimLinearisation(TrimLinearisation&&) = delete;
    TrimLinearisation& operator=(TrimLinearisation&&) = delete;
    ~TrimLinearisation();

    // The g-set vectors a derivative is formed along: the caller gives −∂K·displacements and −∂M·motions.
    struct Directions
    {
        std::span<const double> displacements;
        std::span<const double> motions;
    };
    // The derivative of the trimmed state along a design direction: every variable of the trim case in its order (the
    // fixed ones zero), the support accelerations ü_r, the pressures Δcp of the boxes and, when asked, the g-set
    // displacements (relative to the supports, as the trimmed ones).
    struct StateDerivative
    {
        std::vector<double> variables;
        std::vector<double> supportAccelerations;
        std::vector<double> pressures;
        std::vector<double> displacements;
    };

    // The subcases, in the order of TrimResults::subcases.
    [[nodiscard]] std::vector<base::EntityId> subcases() const;
    [[nodiscard]] std::size_t gSetSize() const noexcept;
    // The labels of the trim case of a subcase (index into subcases()), in the order of TrimSubcaseResult::variables.
    [[nodiscard]] std::vector<std::string> variableLabels(std::size_t subcase) const;
    // Whether the subcase has FreeBodySupport DOFs (without, its unrestrained coefficients are zero).
    [[nodiscard]] bool hasSupports(std::size_t subcase) const;
    // The condensed unit that holds element `elementId`, 0 on the full path and for an element of a full unit or of the
    // connection system. The derivatives solve on the interface problem, which takes no load on the interior points of
    // a condensed unit: the pseudo-loads of a design variable must lie on full units (or on boundaries).
    [[nodiscard]] std::int64_t condensedUnitOf(base::EntityId elementId) const;

    // The trimmed state of a subcase: u and D_r·ü_r.
    [[nodiscard]] Directions trimmedState(std::size_t subcase) const;
    // The derivative of the trimmed state for −∂K·u (`stiffnessLoad`) and −∂M·D_r·ü_r (`inertialLoad`), g-set vectors
    // of trimmedState(subcase). Throws std::invalid_argument for vectors not of the g set.
    [[nodiscard]] StateDerivative stateDerivative(std::size_t subcase, std::span<const double> stiffnessLoad,
                                                  std::span<const double> inertialLoad, bool withDisplacements,
                                                  base::MemoryBudget& budget);

    // The vectors of the elastic coefficients of variable `variable` (index into variableLabels(subcase)): u_x and
    // D_r·ü_r(x) restrained, u_x and D_r·a_x unrestrained (the class comment); computed at the first call, kept.
    [[nodiscard]] Directions coefficientState(std::size_t subcase, std::size_t variable, bool isUnrestrained,
                                              base::MemoryBudget& budget);
    // The derivative of the six elastic coefficients (CX, CY, CZ, CMX, CMY, CMZ of StabilityDerivativeRow) of a
    // variable for −∂K·u_x (`stiffnessLoad`) and −∂M·w_x (`inertialLoad`) of coefficientState().
    [[nodiscard]] std::array<double, 6> coefficientDerivative(std::size_t subcase, std::size_t variable,
                                                              bool isUnrestrained,
                                                              std::span<const double> stiffnessLoad,
                                                              std::span<const double> inertialLoad,
                                                              base::MemoryBudget& budget);

private:
    std::unique_ptr<State> m_state;
};

} // namespace oos::solutions
