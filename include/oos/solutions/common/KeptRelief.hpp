#pragma once

#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace oos::solutions
{

// The inertia relief of a constraint set (PARAM INREL = −1 on its FreeBodySupport or SUPORT1 DOFs), kept with the set
// for the sensitivities of a design point: the support motions D (g × r), M·D, (Dᵀ·M·D)⁻¹ and the accelerations ü_r of
// each subcase solved on the set (assembly::InertiaRelief). The relieved loads of a subcase are P' = Q·P, Q = I −
// M·D·(Dᵀ·M·D)⁻¹·Dᵀ; with support motions that are rigid-body motions (D does not depend on the stiffness nor on the
// mass), their derivative with respect to a designed field is
//     dP' = Q·(dP − dM·D·ü_r),
// which the sensitivities form as the pseudo-load of the unrelieved loads and of the mass, corrected by the low-rank
// term −M·D·c, c = (Dᵀ·M·D)⁻¹·Dᵀ·(dP − dM·D·ü_r) (design/sensitivity/PseudoLoads).
//
// `rigidResidual` measures how far D is from rigid-body motions: the largest support reaction of D, max |K·D| at the
// constrained DOFs (zero for the rigid motions of a body free apart from its supports; the stiffness of a constraint
// or a grounded element otherwise), over the largest |K·e| of a support DOF. The sensitivities refuse a set whose
// residual exceeds RIGID_SUPPORT_TOLERANCE (its D would depend on the design).
//
// Thread-safety: immutable once built. Ownership: owns its matrices (budgeted in the run's budget, which must outlive
// it).
struct KeptRelief
{
    // The sensitivities take D as design-independent (rigid support motions): the error of that, against the motions of
    // a body that is not quite free, is of the order of `rigidResidual` relative to the responses' derivatives, so the
    // sensitivities accept a residual well under their own accuracy (1e-6 of an entry, the element derivatives'
    // differences), looser than the path choice's assembly::RIGID_SUPPORT_TOLERANCE (which decides whether the analysis
    // takes the geometric motions; it is not changed): a small barrel of mixed elements, free on a FreeBodySupport of a
    // frame point, measures 2.4e-8 (warped shells and the drilling stiffness), an SPC or a stiff grounded element 1e-3
    // to 1.
    static constexpr double RIGID_SUPPORT_TOLERANCE = 1.0e-7;

    linalg::DenseMatrix supportModes;                                                                 // D
    linalg::DenseMatrix massModes;                                                                    // M·D
    std::array<double, assembly::MAX_SUPPORT_DOFS * assembly::MAX_SUPPORT_DOFS> reducedMassInverse{}; // r × r
    std::vector<std::array<double, assembly::MAX_SUPPORT_DOFS>> accelerations; // per subcase of the set, in order
    double rigidResidual = 0.0;

    [[nodiscard]] std::size_t supportCount() const noexcept { return supportModes.columns(); }
    // (Dᵀ·M·D)⁻¹·Dᵀ·v for v given by its non-zero entries (DOF, value).
    [[nodiscard]] std::array<double, assembly::MAX_SUPPORT_DOFS> projectionOf(
        std::span<const std::pair<base::DofIndex, double>> entries) const
    {
        const std::size_t r = supportCount();
        std::array<double, assembly::MAX_SUPPORT_DOFS> generalised{};
        for (const auto& [dof, value] : entries)
            for (std::size_t k = 0; k < r; ++k)
                generalised[k] += supportModes(static_cast<std::size_t>(dof), k) * value;
        std::array<double, assembly::MAX_SUPPORT_DOFS> result{};
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = 0; k < r; ++k)
                result[i] += reducedMassInverse[i * r + k] * generalised[k];
        return result;
    }
};

} // namespace oos::solutions
