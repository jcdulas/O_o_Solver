#pragma once

#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/input/tables/RigidTable.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace oos::assembly
{

// The card a constraint comes from.
enum class ConstraintSource : std::uint8_t
{
    RigidBodyMulti,
    RigidBodyStar,
    RigidInterpolation,
    RigidBar,
    RigidRod,
    Mpc,
    Fastener, // the internal constraints of a Fastener whose GA or GB is a grid point (FastenerConstraints)
    Weld,     // the same for a Weld
    ContactTable, // the glue of a glued contact pair of a ContactTable (GlueConstraints; the source ID is the table's)
    ContactPair   // the same for a ContactPair
};

[[nodiscard]] std::string_view cardNameOf(ConstraintSource source) noexcept;

// The dependent DOFs of one rigid element or MPC equation, each as an explicit linear combination of the element's
// other DOFs: u_m = Σ_j c_j·u_j (a row of Gm before the chains are resolved; a term may still be on a dependent DOF of
// another element). Sized by one element: std::vector is the right container.
struct LocalConstraintRows
{
    std::vector<base::DofIndex> dependents;
    std::vector<std::size_t> offsets{0}; // dependents.size() + 1
    std::vector<ConstraintTerm> terms;

    void clear()
    {
        dependents.clear();
        offsets.assign(1, 0);
        terms.clear();
    }
    [[nodiscard]] std::size_t size() const noexcept { return dependents.size(); }
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t i) const noexcept
    {
        return {terms.data() + offsets[i], offsets[i + 1] - offsets[i]};
    }
    // Appends a row; the terms with a zero coefficient are left out.
    void addRow(base::DofIndex dependent, std::span<const ConstraintTerm> rowTerms);
};

// The equations of the rigid elements, eliminated by default (linear elimination, dependents in the
// m set), one element at a time, from the rows of rigidElementsTable (O_o_Reader: one row per grid point, the
// independent rows first then the dependent ones, in the order of the card; RigidBar CMA/CMB defaults already
// resolved). All the geometry is in the basic system; a DOF is in the displacement system CD of its grid point.
//
// - RigidBodyStar, RigidBodyMulti, RigidBar: the element moves rigidly. Its six independent DOFs (RBE2: GN with all its
// components; RigidBodyMulti:
//   the GNi with their CNi; RigidBar: CNA at GA and CNB at GB) fix the rigid motion q (translation and rotation at the
//   first independent grid point); each DOF is a row of the small-rotation rigid transfer
//       t_p = R_pᵀ·(t − [r_p×]·θ),   θ_p = R_pᵀ·θ,   r_p = x_p − x_0,
//   so u_ind = A·q, u_dep = B·q and u_dep = B·A⁻¹·u_ind (RBE1, RigidBodyStar, RigidBar: a rigid body, whose six
//   independent DOFs must be able to describe any rigid motion). A is solved after scaling the rotations by the element
//   size, and an A that cannot describe every rigid motion (pivot below RIGID_PIVOT_TOLERANCE) is refused.
// - RigidRod: rigid in extension only: e·(t_B − t_A) = 0 with e the unit axis from GA to GB, solved for the one
//   dependent translation (CMA or CMB).
// - RigidInterpolation: the reference motion is the weighted least-squares rigid motion of the independent DOFs (the
// interpolation
//   element):
//       u_ref = (Sᵀ·W·S)⁻¹·Sᵀ·W·u_ind,
//   S the rows (components Ci) of the rigid transfer from the reference point to each independent point, W the
//   diagonal weights: WTi for a translation, WTi·Lc² for a rotation, Lc the characteristic length, the mean distance
//   of the independent grid points to the reference point (the rotational weights
//   are scaled by Lc² so that the result does not depend on the length unit; Lc = 1 when every point is on the
//   reference point). Only the REFC components are kept; they must be determined by the independent DOFs (the null
//   space of Sᵀ·W·S must not touch them). The equations u_ref,c − G_c·u_ind = 0 are solved for the REFC DOFs, or, with
//   the UM option, for the DOFs listed after "UM" (as many as REFC, each a DOF of the element).
//   The load distribution follows: the forces on the independent DOFs are Gᵀ·F_ref (f_i =
//   W_i·S_i·(Sᵀ·W·S)⁻¹·F), which keep the force and moment of F_ref.
//
// RigidBar1, RigidJoint, RigidSpline, RigidTriaPlate, RigidTriaPlate1, RigidShellSolidConnection are refused
// ("unsupported-rigid-element"): the supported rigid elements are RigidBodyMulti, RigidBodyStar, RigidInterpolation,
// RigidBar and RigidRod. The thermal fields (ALPHA, TREF) are not read.
//
// Thread-safety: immutable; concurrent calls are safe. Ownership: borrows the DOF map.
class RigidElementEquations
{
public:
    // The smallest pivot of the scaled independent matrix A of a rigid element (rows normalised to 1, rotations scaled
    // by the element size, so the pivots are cosines of angles): below it the independent DOFs cannot describe every
    // rigid motion (an RigidBar with CNA = CNB = 123 along an axis has no torsion DOF and a zero pivot).
    static constexpr double RIGID_PIVOT_TOLERANCE = 1.0e-8;
    // The smallest |e·d| of the RigidRod axis e and the direction d of the dependent component: a dependent translation
    // nearly perpendicular to the axis takes the whole extension with a huge coefficient (a valid RigidRod has its
    // dependent component the one most nearly aligned with the element).
    static constexpr double RIGID_ROD_MIN_AXIS_COSINE = 1.0e-6;
    // The relative eigenvalue of the scaled RigidInterpolation normal matrix Sᵀ·W·S below which a rigid motion is not
    // seen by the independent DOFs (the round-off of a 6 × 6 symmetric eigenproblem is about 1e-15 of the largest
    // eigenvalue).
    static constexpr double RIGID_INTERPOLATION_RANK_TOLERANCE = 1.0e-10;
    // The part of a null vector on a REFC component above which that component is not determined.
    static constexpr double RIGID_INTERPOLATION_NULL_COMPONENT_TOLERANCE = 1.0e-8;
    // The smallest pivot, relative to the largest term, of the UM equations of an RigidInterpolation.
    static constexpr double RIGID_INTERPOLATION_DEPENDENT_PIVOT_TOLERANCE = 1.0e-10;

    explicit RigidElementEquations(const DofMap& dofMap) : m_dofMap(&dofMap) {}

    // The rows of the rigid element on rows [beginRow, endRow) of the table (one elementId). Returns false, with the
    // rows cleared and Diagnostics reported (card, elementId), when the element cannot be eliminated.
    bool build(const input::RigidTable& table, std::int64_t beginRow, std::int64_t endRow, LocalConstraintRows& rows,
               base::DiagnosticSink& sink) const;

    // The source of a card type name of the table; nothing for a card that this class does not eliminate.
    [[nodiscard]] static std::optional<ConstraintSource> findSource(std::string_view type) noexcept;

private:
    const DofMap* m_dofMap;
};

} // namespace oos::assembly
