#pragma once

// The structure of a static aeroelastic trim in the space of the boxes, shared by TrimSolution (the trim) and
// TrimLinearisation (its design derivatives): the l-set map, the slopes and resultants of the boxes, the
// support modes, the full and condensed structures, the flexibility in the space of the boxes and the factors of the
// pressures. An internal header of solutions/sol144 (included by its sources only).

#include "oos/aero/dlm/DoubletLatticeSteady.hpp"
#include "oos/aero/geometry/AeroBoxes.hpp"
#include "oos/aero/geometry/AeroReference.hpp"
#include "oos/aero/outputs/HingeMoments.hpp"
#include "oos/aero/splines/SplineMatrix.hpp"
#include "oos/aero/trim/ReferenceCoefficients.hpp"
#include "oos/aero/trim/TrimSystem.hpp"
#include "oos/aero/trim/TrimVariables.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/assembly/relief/RigidBodyModes.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OperationCancelledError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseLu.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/output/monitors/MonitorPoint1.hpp"
#include "oos/output/volume/OutputVolumeEstimate.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/RunCatalog.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"
#include "oos/solutions/sol144/TrimLinearisation.hpp"
#include "oos/solutions/sol144/TrimSolution.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::solutions::trim
{

// The boxes whose structural solves are made together: the right-hand sides of one call of FsetSolver::solve (the
// l set × this many doubles, and as many g-set vectors).
constexpr std::size_t FLEXIBILITY_BLOCK = 32;
// The rigid-body variables of AeroRigidTrimVariable with an aerodynamic effect, and the accelerations (in the reference
// frame RCSID: three translations, then three rotations).
constexpr const auto& AERODYNAMIC_LABELS = aero::AERODYNAMIC_RIGID_BODY_LABELS;
constexpr const auto& ACCELERATION_LABELS = aero::RIGID_ACCELERATION_LABELS;
// The flow, the x axis of the aerodynamic frame.
const base::Vec3 FLOW_AXIS{1.0, 0.0, 0.0};
// The share of its own length below which a rigid-body direction, seen at the supports, is taken as dependent on
// those chosen before it (geometricSupportModes): the columns are normalised, and a rounding of their sums is near
// 1e-15. Two directions whose shares differ by less are a tie, which goes to the lower direction (TX before RX), not
// to the rounding of the normalisation.
constexpr double SUPPORT_INDEPENDENCE = 1.0e-8;
// The motion at the supports, in units of the supports' largest lever arm, below which a rigid-body direction is not
// seen by them at all (geometricSupportModes): the rounding of the grid coordinates, not a lever. A grid point
// on the plane of symmetry may lie 1e-13 off it after the changes of frame that placed it (on a half
// aircraft: 2.4e-13 mm at 8 m, 2e-17 of the lever arms); the coordinates and the frames round near 1e-15 of the
// model's size, and a support whose lever arm is 1e-10 of the others holds nothing either.
constexpr double SUPPORT_NEGLIGIBLE_MOTION = 1.0e-10;
// The share of the memory budget left at the start that the result tables may take when the caller gives no limit,
// as in the static solution: the other half for the factorisation and the solves.
constexpr double OUTPUT_BUDGET_FRACTION = 0.5;

class Stopwatch
{
public:
    double lap()
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - m_start).count();
        m_start = now;
        return seconds;
    }

private:
    std::chrono::steady_clock::time_point m_start = std::chrono::steady_clock::now();
};

inline void checkCancelled(const base::CancellationToken* cancellation)
{
    if (cancellation != nullptr && cancellation->isCancelled())
        throw base::OperationCancelledError();
}

inline linalg::PivotPolicy pivotPolicyOf(const input::ParamTable* params, std::optional<std::int32_t> subcase,
                                         base::DiagnosticSink& sink)
{
    linalg::PivotPolicy policy;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", subcase, sink))
    {
        if (*ratio > 0.0)
            policy.maxRatio = *ratio;
        else
            sink.report(assembly::badParamDiagnostic("MaxDiagonalRatio", "a positive real number"));
    }
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", subcase, sink))
        policy.bailout = *bailout;
    return policy;
}

inline base::Diagnostic subcaseProblem(base::EntityId subcase, base::Severity severity, std::string code,
                                       std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.message = base::concatenate({"subcase ", std::to_string(subcase), ": ", message});
    return diagnostic;
}

inline std::optional<std::size_t> indexIn(std::span<const std::string_view> labels, std::string_view label)
{
    for (std::size_t k = 0; k < labels.size(); ++k)
        if (labels[k] == label)
            return k;
    return std::nullopt;
}

// The supported trim variables of the case; the others reported to `sink`.
inline void checkVariables(const aero::TrimCase& trimCase, const aero::TrimVariables& variables,
                           base::DiagnosticSink& sink)
{
    for (const aero::TrimSetting& setting : trimCase.settings)
    {
        bool isKnown = indexIn(AERODYNAMIC_LABELS, setting.label) || indexIn(ACCELERATION_LABELS, setting.label);
        for (const aero::TrimVariables::ControlSurface& surface : variables.controlSurfaces())
            isKnown = isKnown || surface.label == setting.label;
        if (isKnown)
            continue;
        // An AeroController, another AeroRigidTrimVariable label or an undefined one acts only through user matrices
        // (AEDW, AeroForceVector, AeroPressureVector, DirectMatrixAeroJ: not supported): no aerodynamic or inertial
        // effect here.
        if (setting.state == aero::TrimVariableState::Free || setting.state == aero::TrimVariableState::Linked)
            sink.report(subcaseProblem(
                trimCase.subcase, base::Severity::Error, "trim-user-variable",
                base::concatenate({"the trim variable ", setting.label, " of AeroTrim ", std::to_string(trimCase.setId),
                                   " acts only through user aerodynamic matrices, which this "
                                   "version does not support; fix it on the AeroTrim"})));
        else if (setting.state == aero::TrimVariableState::Fixed && setting.value != 0.0)
            sink.report(subcaseProblem(
                trimCase.subcase, base::Severity::Warning, "trim-variable-without-effect",
                base::concatenate({"the trim variable ", setting.label, " of AeroTrim ", std::to_string(trimCase.setId),
                                   " acts only through user aerodynamic matrices (not "
                                   "supported): its value has no effect"})));
    }
}

// The g-set loads of forces and moments at grid points (basic system), added to `loadsG` in the displacement
// systems.
inline void addPointLoads(const std::vector<aero::SplineMatrix::PointLoad>& loads, const assembly::DofMap& dofMap,
                          std::span<double> loadsG)
{
    for (const aero::SplineMatrix::PointLoad& load : loads)
    {
        const std::optional<std::int32_t> row = dofMap.points().findGrid(load.nodeId);
        if (!row)
            continue; // SplineMatrix checked that its points are grid points
        const base::Mat3 basicToCd = dofMap.cdToBasic(*row).transposed();
        const base::Vec3 force = basicToCd * load.force;
        const base::Vec3 moment = basicToCd * load.moment;
        for (int c = 0; c < 3; ++c)
        {
            loadsG[static_cast<std::size_t>(dofMap.gridDof(*row, 1 + c))] += force[static_cast<std::size_t>(c)];
            loadsG[static_cast<std::size_t>(dofMap.gridDof(*row, 4 + c))] += moment[static_cast<std::size_t>(c)];
        }
    }
}

// The structure of one constraint set in the space of the boxes: what every subcase of the set reuses.
struct BoxFlexibility
{
    linalg::DenseMatrix slopes;              // E = S·K_ll⁻¹·T (n × n): the slopes of unit box pressures (unit q)
    linalg::DenseMatrix structuralResultant; // 6 × n: Ψᵀ·T, the resultant (aero frame, about RCSID) of each box's loads
    linalg::DenseMatrix supportLoads;        // r × n: D_rᵀ·T
    linalg::DenseMatrix inertialSlopes;      // n × r: S·K_ll⁻¹·M·D_r
    linalg::DenseMatrix inertialResultant;   // 6 × r: Ψᵀ·M·D_r
    linalg::DenseMatrix supportMass;         // r × r: D_rᵀ·M·D_r
};

// The equations of the l set from g-set loads: Pn = P̄n + Gmᵀ·Pm, then its l-set rows (the s and r sets are held).
class LsetMap
{
public:
    LsetMap(const assembly::RigidElimination& gm, const FsetSolver& solver, std::size_t gSize,
            base::MemoryBudget& budget)
        : m_gm(&gm), m_solver(&solver), m_scratch(budget, gSize, "trim g-set scratch")
    {
    }

    void loadsToEquations(std::span<const double> loadsG, linalg::DenseMatrixView rhs, std::size_t column)
    {
        std::copy(loadsG.begin(), loadsG.end(), m_scratch.data());
        m_gm->reduceToN(m_scratch.span());
        for (std::size_t e = 0; e < rhs.rows; ++e)
            rhs(e, column) = m_scratch[static_cast<std::size_t>(m_solver->dofOfEquation(static_cast<std::int32_t>(e)))];
    }

    // u_g of the solution column (u_r = 0, the s set zero, the m set from Gm).
    void equationsToDisplacements(linalg::ConstDenseMatrixView solution, std::size_t column,
                                  std::span<double> displacementsG) const
    {
        std::fill(displacementsG.begin(), displacementsG.end(), 0.0);
        for (std::size_t e = 0; e < solution.rows; ++e)
            displacementsG[static_cast<std::size_t>(m_solver->dofOfEquation(static_cast<std::int32_t>(e)))] =
                solution(e, column);
        m_gm->expandToG(displacementsG);
    }

private:
    const assembly::RigidElimination* m_gm;
    const FsetSolver* m_solver;
    base::BudgetedBuffer<double> m_scratch;
};

// The slopes of the boxes for the displacements u_g.
inline void slopesOf(const aero::SplineMatrix& splines, const assembly::DofMap& dofMap,
                     std::span<const double> displacementsG, linalg::DenseMatrixView target, std::size_t column)
{
    const aero::BoxMotion motion = splines.boxMotion(
        [&](std::int32_t nodeId)
        {
            aero::PointMotion point;
            if (const std::optional<std::int32_t> row = dofMap.points().findGrid(nodeId))
            {
                const base::Mat3 cdToBasic = dofMap.cdToBasic(*row);
                base::Vec3 translation;
                base::Vec3 rotation;
                for (int c = 0; c < 3; ++c)
                {
                    translation[static_cast<std::size_t>(c)] =
                        displacementsG[static_cast<std::size_t>(dofMap.gridDof(*row, 1 + c))];
                    rotation[static_cast<std::size_t>(c)] =
                        displacementsG[static_cast<std::size_t>(dofMap.gridDof(*row, 4 + c))];
                }
                point.translation = cdToBasic * translation;
                point.rotation = cdToBasic * rotation;
            }
            return point;
        });
    for (std::size_t j = 0; j < target.rows; ++j)
        target(j, column) = motion.slope[j];
}

// The resultant of g-set loads about the RCSID origin, in the aerodynamic frame.
inline aero::AeroResultant resultantOf(const assembly::RigidBodyModes& modes, const aero::AeroReference& reference,
                                       std::span<const double> loadsG)
{
    const std::array<double, 6> basic = modes.resultantOf(loadsG);
    return {reference.directionInAeroFrame({basic[0], basic[1], basic[2]}),
            reference.directionInAeroFrame({basic[3], basic[4], basic[5]})};
}

inline void storeResultant(const aero::AeroResultant& resultant, linalg::DenseMatrix& target, std::size_t column)
{
    for (std::size_t k = 0; k < 3; ++k)
    {
        target(k, column) = resultant.force[k];
        target(3 + k, column) = resultant.moment[k];
    }
}

// Σ_j matrix(:, j)·weights_j as a resultant (matrix: 6 × n).
inline aero::AeroResultant combine(const linalg::DenseMatrix& matrix, std::span<const double> weights)
{
    aero::AeroResultant resultant;
    for (std::size_t j = 0; j < matrix.columns(); ++j)
        for (std::size_t k = 0; k < 3; ++k)
        {
            resultant.force[k] += matrix(k, j) * weights[j];
            resultant.moment[k] += matrix(3 + k, j) * weights[j];
        }
    return resultant;
}

inline aero::AeroResultant scaled(aero::AeroResultant resultant, double factor)
{
    return {factor * resultant.force, factor * resultant.moment};
}

// D_r (g × r): the rigid-body motions of the structure that move each support DOF by a unit and the others not,
// from the geometry: exact rigid-body motions of the structure about the RCSID origin, on the supported directions
// (not −K_ll⁻¹·K_lr, which an improper support makes elastic). Of the six motions Ψ of a unit translation
// along, and rotation about, each RCSID axis through the RCSID origin, the supported directions S are the r whose
// rows at the support DOFs, Ψ_r (r × 6), are independent: chosen one by one, the column of Ψ_r farthest from the
// span of those taken relative to its own length (scale-free: translations and rotations are not of one unit), the
// lower direction on a tie (TX before MZ). A column is a candidate only while what is left of it moves the supports
// by at least SUPPORT_NEGLIGIBLE_MOTION, measured without units (the lever arms of the translational rows over the
// largest of them): normalised to its own length, a rotation whose only lever arm is the rounding of a coordinate
// (a support 2.4e-13 off the plane of symmetry) looked like a full direction, took the place of a translation, and
// left the trim equations singular; then
//     D_r = Ψ_S·(Ψ_r,S)⁻¹,
// so that D_rᵀ·F = 0 is the equilibrium of the supported components of the resultant. As rigid-body modes of the a set,
// they are zero on the s set (SPC, AUTOSPC: a held DOF does not move, and its mass is not accelerated: an
// unconnected grid point with a mass carries none of the inertia) and Gm·u_n on the m set. `supportRows` receives
// Ψ_r. Throws SolverError when the supports do not hold r independent rigid-body motions.
struct SupportModes
{
    linalg::DenseMatrix modes; // D_r (g × r)
    // C (6 × r): the rigid-body motions, about the basic origin along and about the basic axes, whose fields are
    // D_r away from the held and dependent DOFs (D_r = Φ_r·C there, Φ_r of RigidBodyModes about the basic origin):
    // the accelerations of the condensed path, whose units know their inertia by −M·Φ_r.
    linalg::DenseMatrix basicMotions;
};

inline SupportModes geometricSupportModes(std::span<const base::DofIndex> rSet, std::span<const std::uint8_t> isHeld,
                                          const assembly::RigidElimination& gm,
                                          const assembly::RigidBodyModes& rigidModes, const base::Mat3& axes,
                                          linalg::DenseMatrix& supportRows, base::MemoryBudget& budget)
{
    const std::size_t gSize = rigidModes.modes().rows();
    const std::size_t r = rSet.size();
    // Ψ in the RCSID axes: a unit motion along RCSID axis a is Σ_i axes(i, a) times the motion along basic axis i.
    const auto referenceMode = [&](std::size_t dof, std::size_t c)
    {
        const std::size_t offset = c < 3 ? 0 : 3;
        double value = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
            value += rigidModes.modes()(dof, offset + i) * axes(i, c - offset);
        return value;
    };
    for (std::size_t k = 0; k < r; ++k)
        for (std::size_t c = 0; c < 6; ++c)
            supportRows(k, c) = referenceMode(static_cast<std::size_t>(rSet[k]), c);

    // Ψ_r without units, for the choice only: a translational row (its translation part a unit vector; a rotational
    // row's is exactly zero, RigidBodyModes) has its lever arms divided by the largest of them, so that every entry is
    // at most 1 and the rounding of a coordinate stays near 1e-16 (a unit rotation turns a rotational row by 1
    // already).
    std::vector<std::uint8_t> isTranslationRow(r, 0); // bounded by the support count
    double largestLever = 0.0;
    for (std::size_t k = 0; k < r; ++k)
    {
        for (std::size_t c = 0; c < 3; ++c)
            isTranslationRow[k] = static_cast<std::uint8_t>(isTranslationRow[k] != 0 || supportRows(k, c) != 0.0);
        if (isTranslationRow[k] != 0)
            for (std::size_t c = 3; c < 6; ++c)
                largestLever = std::max(largestLever, std::abs(supportRows(k, c)));
    }
    const double leverScale = largestLever > 0.0 ? 1.0 / largestLever : 1.0;
    const auto unitless = [&](std::size_t k, std::size_t c)
    { return c >= 3 && isTranslationRow[k] != 0 ? supportRows(k, c) * leverScale : supportRows(k, c); };

    // The supported directions: Gram-Schmidt on the normalised columns of Ψ_r; `length` keeps each column's own
    // length, so that a residual's motion at the supports is its relative length times it.
    std::array<std::vector<double>, 6> residual; // bounded by the support count
    std::array<double, 6> length{};
    for (std::size_t c = 0; c < 6; ++c)
    {
        residual[c].resize(r);
        double norm = 0.0;
        for (std::size_t k = 0; k < r; ++k)
            norm += unitless(k, c) * unitless(k, c);
        length[c] = std::sqrt(norm);
        for (std::size_t k = 0; k < r; ++k)
            residual[c][k] = length[c] > 0.0 ? unitless(k, c) / length[c] : 0.0;
    }
    std::vector<std::size_t> directions; // S, bounded by the support count
    std::array<bool, 6> isTaken{};
    for (std::size_t step = 0; step < r; ++step)
    {
        std::size_t best = 6;
        double bestRelative = 0.0;
        for (std::size_t c = 0; c < 6; ++c)
        {
            if (isTaken[c])
                continue;
            double norm = 0.0;
            for (std::size_t k = 0; k < r; ++k)
                norm += residual[c][k] * residual[c][k];
            const double relative = std::sqrt(norm);
            if (relative * length[c] < SUPPORT_NEGLIGIBLE_MOTION)
                continue; // the supports do not see this motion, or only the rounding of their coordinates
            if (relative > bestRelative + SUPPORT_INDEPENDENCE)
            {
                bestRelative = relative;
                best = c;
            }
        }
        if (best == 6 || bestRelative < SUPPORT_INDEPENDENCE)
            throw base::SolverError(base::concatenate(
                {"the ", std::to_string(r), " FreeBodySupport DOFs hold ", std::to_string(step),
                 " independent rigid-body motions of the free structure (two supports of the same motion, or a "
                 "support that no rigid motion moves); keep one FreeBodySupport DOF per rigid-body motion"}));
        isTaken[best] = true;
        directions.push_back(best);
        std::vector<double> unit(r); // bounded by the support count
        for (std::size_t k = 0; k < r; ++k)
            unit[k] = residual[best][k] / bestRelative;
        for (std::size_t c = 0; c < 6; ++c)
        {
            if (isTaken[c])
                continue;
            double projection = 0.0;
            for (std::size_t k = 0; k < r; ++k)
                projection += unit[k] * residual[c][k];
            for (std::size_t k = 0; k < r; ++k)
                residual[c][k] -= projection * unit[k];
        }
    }

    // (Ψ_r,S)⁻¹ by solving with the identity.
    linalg::DenseMatrix block(budget, r, r, "supported directions at the supports");
    linalg::DenseMatrix inverse(budget, r, r, "inverse of the supported directions");
    for (std::size_t k = 0; k < r; ++k)
    {
        inverse(k, k) = 1.0;
        for (std::size_t s = 0; s < r; ++s)
            block(k, s) = supportRows(k, directions[s]);
    }
    const linalg::DenseLu factor(std::move(block), budget);
    if (factor.isSingular())
        throw base::SolverError(
            "the FreeBodySupport DOFs do not hold independent rigid-body motions; keep one FreeBodySupport DOF per "
            "rigid-body motion of the free structure");
    factor.solve(inverse.view());
    SupportModes result{linalg::DenseMatrix(budget, gSize, r, "trim support modes"),
                        linalg::DenseMatrix(budget, 6, r, "trim support motions about the basic origin")};
    linalg::DenseMatrix& modes = result.modes;
    // C: the motion of support k about the RCSID origin, B = Σ_s Ψ's direction s·(Ψ_r,S)⁻¹(s, k) (translations
    // along and rotations about the basic axes), carried to the basic origin: a rotation θ about o is the rotation
    // θ about the origin and the translation o × θ.
    for (std::size_t k = 0; k < r; ++k)
    {
        std::array<double, 6> motion{};
        for (std::size_t s = 0; s < r; ++s)
        {
            const std::size_t offset = directions[s] < 3 ? 0 : 3;
            for (std::size_t i = 0; i < 3; ++i)
                motion[offset + i] += axes(i, directions[s] - offset) * inverse(s, k);
        }
        const base::Vec3 shift = base::cross(rigidModes.referencePoint(), base::Vec3{motion[3], motion[4], motion[5]});
        for (std::size_t i = 0; i < 3; ++i)
        {
            result.basicMotions(i, k) = motion[i] + shift[i];
            result.basicMotions(3 + i, k) = motion[3 + i];
        }
    }
    for (std::size_t dof = 0; dof < gSize; ++dof)
    {
        if (isHeld[dof] != 0 && std::find(rSet.begin(), rSet.end(), static_cast<base::DofIndex>(dof)) == rSet.end())
            continue;
        for (std::size_t s = 0; s < r; ++s)
        {
            const double value = referenceMode(dof, directions[s]);
            if (value == 0.0)
                continue;
            for (std::size_t k = 0; k < r; ++k)
                modes(dof, k) += value * inverse(s, k);
        }
    }
    base::BudgetedBuffer<double> column(budget, gSize, "trim support mode");
    for (std::size_t k = 0; k < r; ++k)
    {
        for (std::size_t dof = 0; dof < gSize; ++dof)
            column[dof] = modes(dof, k);
        gm.expandToG(column.span());
        for (std::size_t dof = 0; dof < gSize; ++dof)
            modes(dof, k) = column[dof];
    }
    return result;
}

// The structure of one constraint set as the trim reads it: its supports and their rigid-body motions D_r
// (geometricSupportModes), M·D_r on every DOF, and the solves u = K_ll⁻¹·(P − M·D_r·ü_r) with u_r = 0. The full
// path factorises the l set (FullStructure); the condensed path solves the interface problem of the condensed
// units, the spline nodes and the supports on their boundaries (CondensedStructure).
class TrimStructure
{
public:
    TrimStructure() = default;
    TrimStructure(const TrimStructure&) = delete;
    TrimStructure& operator=(const TrimStructure&) = delete;
    TrimStructure(TrimStructure&&) = delete;
    TrimStructure& operator=(TrimStructure&&) = delete;
    virtual ~TrimStructure() = default;

    [[nodiscard]] std::size_t supportCount() const noexcept { return m_supports.modes.columns(); }
    // D_r (g × r), M·D_r (g × r, zero on the held DOFs), Ψ_r (r × 6: the r-set motion of a unit motion of the RCSID
    // origin).
    [[nodiscard]] const linalg::DenseMatrix& supportModes() const noexcept { return m_supports.modes; }
    [[nodiscard]] const linalg::DenseMatrix& massModes() const noexcept { return m_massModes; }
    [[nodiscard]] const linalg::DenseMatrix& referenceMotion() const noexcept { return m_referenceMotion; }

    // u (g × k) = K_ll⁻¹·(P − M·D_r·ü_r) of loads P (g × k) and support accelerations ü_r (r × k, or no column:
    // none): u_r = 0, the s set zero, the m set from Gm. `isForOutputs`: every DOF the outputs read (the condensed
    // path recovers its units; otherwise only the interface points, the spline nodes among them).
    virtual void solve(linalg::ConstDenseMatrixView loads, linalg::ConstDenseMatrixView accelerations,
                       linalg::DenseMatrixView displacements, bool isForOutputs, base::MemoryBudget& budget) = 0;
    // K_ll⁻¹·M·D_r (g × r), at the interface points on the condensed path.
    virtual void solveMassModes(linalg::DenseMatrixView displacements, base::MemoryBudget& budget) = 0;
    // forces = K·u at every DOF the outputs read, for the support accelerations ü_r (r) of the case; the condensed
    // path adds the grid point force rows of its units not recovered to `unitRows`.
    virtual void elementForces(std::span<const double> displacements, std::span<const double> accelerations,
                               std::span<double> forces, std::vector<output::StiffnessRecovery::UnitRow>& unitRows,
                               base::MemoryBudget& budget) const = 0;
    [[nodiscard]] virtual const assembly::RigidElimination& gm() const = 0;
    [[nodiscard]] virtual std::span<const std::uint8_t> isConstrained() const = 0; // the s and r sets, per g-set DOF
    // The grid point forces of the units not recovered with the rows of a case (nothing on the full path).
    [[nodiscard]] virtual std::optional<output::StiffnessRecovery::CondensedUnits> condensedUnits(
        std::span<const output::StiffnessRecovery::UnitRow> rows) const = 0;
    // The kernels the later solves take, serial and without cancellation (a TrimLinearisation keeps the structure
    // after the run, whose thread pool and cancellation token may be gone); the same bits for a given kernel kind.
    virtual void rebind(const linalg::DenseKernels& kernels) = 0;

protected:
    SupportModes m_supports;
    linalg::DenseMatrix m_massModes;
    linalg::DenseMatrix m_referenceMotion;
};

// The full path: the l set factorised (FsetSolver), M·D_r from Mgg.
class FullStructure final : public TrimStructure
{
public:
    FullStructure(const ConstrainedSystem& system, const FsetSolver& solver, const linalg::BlockSparseMatrix& mass,
                  const assembly::RigidBodyModes& rigidModes, const base::Mat3& axes,
                  const base::CancellationToken* cancellation, base::MemoryBudget& budget)
        : m_system(&system), m_solver(&solver), m_cancellation(cancellation)
    {
        const std::size_t gSize = rigidModes.modes().rows();
        const std::size_t r = system.supportDofs().size();
        m_referenceMotion = linalg::DenseMatrix(budget, r, 6, "reference motion of the supports");
        m_massModes = linalg::DenseMatrix(budget, gSize, r, "mass support modes");
        if (r == 0)
            return;
        m_supports = geometricSupportModes(system.supportDofs(), system.isConstrained(), system.gm(), rigidModes, axes,
                                           m_referenceMotion, budget);
        base::BudgetedBuffer<double> column(budget, gSize, "trim support mode");
        base::BudgetedBuffer<double> product(budget, gSize, "trim mass times support mode");
        for (std::size_t k = 0; k < r; ++k)
        {
            for (std::size_t dof = 0; dof < gSize; ++dof)
                column[dof] = m_supports.modes(dof, k);
            mass.multiply(column.span(), product.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                m_massModes(dof, k) = product[dof];
        }
    }

    void solve(linalg::ConstDenseMatrixView loads, linalg::ConstDenseMatrixView accelerations,
               linalg::DenseMatrixView displacements, bool /*isForOutputs*/, base::MemoryBudget& budget) override
    {
        const std::size_t gSize = loads.rows;
        const std::size_t columns = loads.columns;
        LsetMap lset(m_system->gm(), *m_solver, gSize, budget);
        linalg::DenseMatrix rhs(budget, static_cast<std::size_t>(m_solver->equationCount()), columns,
                                "trim right-hand sides");
        base::BudgetedBuffer<double> column(budget, gSize, "trim loads");
        for (std::size_t c = 0; c < columns; ++c)
        {
            for (std::size_t dof = 0; dof < gSize; ++dof)
                column[dof] = loads(dof, c);
            for (std::size_t k = 0; k < accelerations.rows && accelerations.columns > 0; ++k)
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    column[dof] -= m_massModes(dof, k) * accelerations(k, c);
            lset.loadsToEquations(column.span(), rhs.view(), c);
        }
        solveEquations(rhs.view());
        for (std::size_t c = 0; c < columns; ++c)
        {
            lset.equationsToDisplacements(rhs.view(), c, column.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                displacements(dof, c) = column[dof];
        }
    }

    void solveMassModes(linalg::DenseMatrixView displacements, base::MemoryBudget& budget) override
    {
        const std::size_t gSize = m_massModes.rows();
        const std::size_t r = m_massModes.columns();
        LsetMap lset(m_system->gm(), *m_solver, gSize, budget);
        linalg::DenseMatrix rhs(budget, static_cast<std::size_t>(m_solver->equationCount()), r,
                                "trim inertial right-hand sides");
        for (std::size_t k = 0; k < r; ++k)
            lset.loadsToEquations(std::span<const double>(&m_massModes(0, k), gSize), rhs.view(), k);
        solveEquations(rhs.view());
        base::BudgetedBuffer<double> column(budget, gSize, "trim inertial displacements");
        for (std::size_t k = 0; k < r; ++k)
        {
            lset.equationsToDisplacements(rhs.view(), k, column.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                displacements(dof, k) = column[dof];
        }
    }

    void elementForces(std::span<const double> displacements, std::span<const double> /*accelerations*/,
                       std::span<double> forces, std::vector<output::StiffnessRecovery::UnitRow>& /*unitRows*/,
                       base::MemoryBudget& /*budget*/) const override
    {
        m_system->kgg().multiply(displacements, forces);
    }

    [[nodiscard]] const assembly::RigidElimination& gm() const override { return m_system->gm(); }
    [[nodiscard]] std::span<const std::uint8_t> isConstrained() const override { return m_system->isConstrained(); }
    [[nodiscard]] std::optional<output::StiffnessRecovery::CondensedUnits> condensedUnits(
        std::span<const output::StiffnessRecovery::UnitRow> /*rows*/) const override
    {
        return std::nullopt;
    }
    void rebind(const linalg::DenseKernels& kernels) override
    {
        m_kernels = &kernels;
        m_cancellation = nullptr;
    }

private:
    void solveEquations(linalg::DenseMatrixView rhs) const
    {
        if (m_kernels != nullptr)
            m_solver->solve(rhs, *m_kernels, m_cancellation);
        else
            m_solver->solve(rhs, m_cancellation);
    }

    const ConstrainedSystem* m_system;
    const FsetSolver* m_solver;
    const base::CancellationToken* m_cancellation;
    const linalg::DenseKernels* m_kernels = nullptr; // the run's kernels until rebind()
};

// The condensed path: the interface problem of the constraint set with its supports held
// (CondensedStatics::prepareLoadGroup). D_r = Φ_r·C away from the held and dependent DOFs, so the inertial loads
// −M·D_r·ü_r are those of the rigid-body accelerations a = C·ü_r, which the units know by their condensed inertial
// patterns −M·Φ_r; M·D_r on the g set is the units' and the interface's −(−M·Φ_r)·C, zero on the held DOFs as the
// full path's (exactly with lumped masses, which couple no held DOF with a free one: the units' patterns move the
// whole unit).
class CondensedStructure final : public TrimStructure
{
public:
    CondensedStructure(CondensedStatics& statics, CondensedStatics::LoadGroup group,
                       std::span<const base::DofIndex> supports, const assembly::RigidBodyModes& rigidModes,
                       const base::Mat3& axes, base::MemoryBudget& budget)
        : m_statics(&statics), m_group(std::move(group))
    {
        const std::size_t gSize = rigidModes.modes().rows();
        const std::size_t r = supports.size();
        m_referenceMotion = linalg::DenseMatrix(budget, r, 6, "reference motion of the supports");
        m_massModes = linalg::DenseMatrix(budget, gSize, r, "mass support modes");
        if (r == 0)
            return;
        m_supports = geometricSupportModes(supports, m_group.isConstrained.span(), *m_group.gm, rigidModes, axes,
                                           m_referenceMotion, budget);
        // Zero on the s set as the full path's (whose D_r is zero there); the supports themselves move (D_r = I).
        base::BudgetedBuffer<std::uint8_t> isStill(budget, gSize, "trim held DOFs but the supports");
        for (std::size_t dof = 0; dof < gSize; ++dof)
            isStill[dof] = m_group.isConstrained[dof];
        for (const base::DofIndex support : supports)
            isStill[static_cast<std::size_t>(support)] = 0;
        base::BudgetedBuffer<double> column(budget, gSize, "trim support inertial loads");
        for (std::size_t k = 0; k < r; ++k)
        {
            std::fill(column.data(), column.data() + gSize, 0.0);
            const std::array<double, 6> motion = motionOf(k);
            m_statics->addInertialLoads(m_group, motion, column.span());
            for (std::size_t dof = 0; dof < gSize; ++dof)
                m_massModes(dof, k) = isStill[dof] != 0 ? 0.0 : -column[dof];
        }
    }

    void solve(linalg::ConstDenseMatrixView loads, linalg::ConstDenseMatrixView accelerations,
               linalg::DenseMatrixView displacements, bool isForOutputs, base::MemoryBudget& budget) override
    {
        const std::size_t columns = loads.columns;
        const bool hasAccelerations = accelerations.columns > 0 && supportCount() > 0;
        linalg::DenseMatrix motions(budget, 6, hasAccelerations ? columns : 0, "trim rigid-body accelerations");
        for (std::size_t c = 0; c < motions.columns(); ++c)
            for (std::size_t k = 0; k < supportCount(); ++k)
                for (std::size_t i = 0; i < 6; ++i)
                    motions(i, c) += m_supports.basicMotions(i, k) * accelerations(k, c);
        m_statics->solveLoads(m_group, loads, motions.view(), isForOutputs, displacements, budget);
    }

    void solveMassModes(linalg::DenseMatrixView displacements, base::MemoryBudget& budget) override
    {
        // K⁻¹·M·D_r: the loads −M·Φ_r·a of the accelerations a = −C.
        const std::size_t r = supportCount();
        const linalg::DenseMatrix noLoads(budget, m_massModes.rows(), r, "no trim loads");
        linalg::DenseMatrix motions(budget, 6, r, "trim inertial accelerations");
        for (std::size_t k = 0; k < r; ++k)
            for (std::size_t i = 0; i < 6; ++i)
                motions(i, k) = -m_supports.basicMotions(i, k);
        m_statics->solveLoads(m_group, noLoads.view(), motions.view(), false, displacements, budget);
    }

    void elementForces(std::span<const double> displacements, std::span<const double> accelerations,
                       std::span<double> forces, std::vector<output::StiffnessRecovery::UnitRow>& unitRows,
                       base::MemoryBudget& budget) const override
    {
        std::fill(forces.begin(), forces.end(), 0.0);
        std::array<double, 6> motion{};
        for (std::size_t k = 0; k < accelerations.size(); ++k)
            for (std::size_t i = 0; i < 6; ++i)
                motion[i] += m_supports.basicMotions(i, k) * accelerations[k];
        m_statics->addStructureForces(m_group, displacements, motion, forces, unitRows, budget);
    }

    [[nodiscard]] const assembly::RigidElimination& gm() const override { return *m_group.gm; }
    [[nodiscard]] std::span<const std::uint8_t> isConstrained() const override { return m_group.isConstrained.span(); }
    [[nodiscard]] std::optional<output::StiffnessRecovery::CondensedUnits> condensedUnits(
        std::span<const output::StiffnessRecovery::UnitRow> rows) const override
    {
        if (!m_statics->hasUnrecoveredUnits())
            return std::nullopt;
        return m_statics->gridPointForceUnits(rows);
    }
    void rebind(const linalg::DenseKernels& kernels) override { m_statics->rebind(kernels); }
    [[nodiscard]] const CondensedStatics::LoadGroup& group() const noexcept { return m_group; }

private:
    [[nodiscard]] std::array<double, 6> motionOf(std::size_t k) const
    {
        std::array<double, 6> motion{};
        for (std::size_t i = 0; i < 6; ++i)
            motion[i] = m_supports.basicMotions(i, k);
        return motion;
    }

    CondensedStatics* m_statics;
    CondensedStatics::LoadGroup m_group;
};

// The structure in the space of the boxes (BoxFlexibility) for one constraint set.
inline BoxFlexibility flexibilityOf(TrimStructure& structure, const assembly::DofMap& dofMap,
                                    const aero::AeroReference& reference, const aero::AeroBoxes& boxes,
                                    const aero::SplineMatrix& splines, const assembly::RigidBodyModes& rigidModes,
                                    const base::CancellationToken* cancellation, base::MemoryBudget& budget)
{
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const std::size_t n = boxes.size();
    const std::size_t r = structure.supportCount();
    BoxFlexibility flexibility{linalg::DenseMatrix(budget, n, n, "aeroelastic slopes E"),
                               linalg::DenseMatrix(budget, 6, n, "box structural resultants"),
                               linalg::DenseMatrix(budget, r, n, "box support loads"),
                               linalg::DenseMatrix(budget, n, r, "inertial slopes"),
                               linalg::DenseMatrix(budget, 6, r, "inertial resultants"),
                               linalg::DenseMatrix(budget, r, r, "support mass")};
    const linalg::DenseMatrix& modes = structure.supportModes();
    const linalg::DenseMatrix noAccelerations(budget, r, 0, "no support accelerations");

    // E, Ψᵀ·T and D_rᵀ·T: the structural loads of a unit pressure on each box (its force A_j along the normal at
    // its force point, through the splines), FLEXIBILITY_BLOCK boxes per solve.
    std::vector<double> boxForces(n, 0.0); // bounded by the boxes, as the matrices above
    for (std::size_t first = 0; first < n; first += FLEXIBILITY_BLOCK)
    {
        checkCancelled(cancellation);
        const std::size_t width = std::min(FLEXIBILITY_BLOCK, n - first);
        linalg::DenseMatrix loads(budget, gSize, width, "trim box loads");
        linalg::DenseMatrix displacements(budget, gSize, width, "trim box displacements");
        for (std::size_t b = 0; b < width; ++b)
        {
            const std::size_t j = first + b;
            const std::span<double> column(&loads(0, b), gSize);
            boxForces[j] = boxes[j].area;
            addPointLoads(splines.structuralLoads(boxForces), dofMap, column);
            boxForces[j] = 0.0;
            storeResultant(resultantOf(rigidModes, reference, column), flexibility.structuralResultant, j);
            for (std::size_t k = 0; k < r; ++k)
            {
                double work = 0.0;
                for (std::size_t dof = 0; dof < gSize; ++dof)
                    work += modes(dof, k) * column[dof];
                flexibility.supportLoads(k, j) = work;
            }
        }
        structure.solve(loads.view(), noAccelerations.view(), displacements.view(), false, budget);
        for (std::size_t b = 0; b < width; ++b)
            slopesOf(splines, dofMap, std::span<const double>(&displacements(0, b), gSize), flexibility.slopes.view(),
                     first + b);
    }
    if (r == 0)
        return flexibility;

    // D_rᵀ·M·D_r, Ψᵀ·M·D_r and S·K_ll⁻¹·M·D_r.
    const linalg::DenseMatrix& massModes = structure.massModes();
    for (std::size_t k = 0; k < r; ++k)
    {
        const linalg::ConstDenseMatrixView columns = massModes.view();
        const std::span<const double> loads(columns.data + k * columns.leadingDimension, gSize);
        for (std::size_t l = 0; l < r; ++l)
        {
            double work = 0.0;
            for (std::size_t dof = 0; dof < gSize; ++dof)
                work += modes(dof, l) * loads[dof];
            flexibility.supportMass(l, k) = work;
        }
        storeResultant(resultantOf(rigidModes, reference, loads), flexibility.inertialResultant, k);
    }
    linalg::DenseMatrix displacements(budget, gSize, r, "trim inertial displacements");
    structure.solveMassModes(displacements.view(), budget);
    for (std::size_t k = 0; k < r; ++k)
        slopesOf(splines, dofMap, std::span<const double>(&displacements(0, k), gSize),
                 flexibility.inertialSlopes.view(), k);

    return flexibility;
}

// The LU factor of a matrix of the pressures: a singular one at a positive dynamic pressure is a divergence of the
// structure (the aerodynamic stiffness q·E cancels the structural one).
inline std::unique_ptr<linalg::DenseLu> pressureFactor(linalg::DenseMatrix matrix, const aero::TrimCase& trimCase,
                                                       const char* structure, base::MemoryBudget& budget)
{
    auto factor = std::make_unique<linalg::DenseLu>(std::move(matrix), budget);
    if (factor->isSingular())
        throw base::SolverError(
            base::concatenate({"AeroTrim ", std::to_string(trimCase.setId), " (subcase ",
                               std::to_string(trimCase.subcase), "): the aeroelastic system of the ", structure,
                               " structure is singular at Q = ", std::to_string(trimCase.dynamicPressure),
                               ", a static divergence; lower the dynamic pressure or stiffen the structure"}));
    return factor;
}

inline linalg::DenseMatrix copyOf(const linalg::DenseMatrix& source, base::MemoryBudget& budget, const char* label)
{
    linalg::DenseMatrix copy(budget, source.rows(), source.columns(), label);
    for (std::size_t j = 0; j < source.columns(); ++j)
        for (std::size_t i = 0; i < source.rows(); ++i)
            copy(i, j) = source(i, j);
    return copy;
}

// A dense copy of `source` minus factor·`other` (same shape).
inline linalg::DenseMatrix minusScaled(const linalg::DenseMatrix& source, double factor,
                                       const linalg::DenseMatrix& other, base::MemoryBudget& budget, const char* label)
{
    linalg::DenseMatrix result(budget, source.rows(), source.columns(), label);
    for (std::size_t j = 0; j < source.columns(); ++j)
        for (std::size_t i = 0; i < source.rows(); ++i)
            result(i, j) = source(i, j) - factor * other(i, j);
    return result;
}

// One constraint set of a trim run: its structure (the full path's system, mass and l-set factor, or the condensed
// path's interface problem) and its flexibility in the space of the boxes; a TrimLinearisation keeps it.
struct TrimGroup
{
    std::optional<assembly::SpcSet> spc;
    std::optional<ConstrainedSystem> system;
    std::optional<linalg::BlockSparseMatrix> mass;
    std::optional<FsetSolver> solver;
    std::unique_ptr<TrimStructure> structure;
    BoxFlexibility flexibility;
    std::unique_ptr<linalg::DenseLu> supportMass;    // m_rr = D_rᵀ·M·D_r (null without supports)
    linalg::DenseMatrix supportAccelerationsOfBoxes; // m_rr⁻¹·D_rᵀ·T (r × n)
};

// One subcase of a trim run as its derivatives read it (TrimLinearisation).
struct TrimSubcaseState
{
    base::EntityId label = 0;
    std::size_t group = 0;
    double dynamicPressure = 0.0;
    double feedback = 0.0; // AEQR
    aero::TrimCase trimCase;
    linalg::DenseMatrix equations;                     // r × m: q·D_rᵀ·T·Δcp_x − m_rr·ü_r(x)
    linalg::DenseMatrix restrained;                    // n × m: Δcp_x, supports held
    linalg::DenseMatrix unrestrained;                  // n × m: Δcp_x, free body (empty without supports)
    linalg::DenseMatrix accelerations;                 // r × m: ü_r(x)
    std::unique_ptr<linalg::DenseLu> restrainedFactor; // D − AEQR·q·E
    std::unique_ptr<linalg::DenseLu> freeFactor;       // D − AEQR·q·E_R (null without supports)
    std::vector<double> supportAccelerations;          // ü_r of the trimmed state
    std::vector<double> displacements;                 // u_g of the trimmed state
    std::vector<double> supportMotion;                 // D_r·ü_r
    // The vectors of the elastic coefficients of a variable (TrimLinearisation::coefficientState), by
    // (variable, unrestrained), made at the first call.
    std::map<std::pair<std::size_t, bool>, std::pair<std::vector<double>, std::vector<double>>> coefficientStates;
};

} // namespace oos::solutions::trim

namespace oos::solutions
{

// The parts of a trim run a TrimLinearisation keeps: the model, the aerodynamic model, the condensed path's parts, the
// constraint sets and the subcases, with the serial kernels of the later solves. In destruction order (reverse of the
// declarations) the users go before what they borrow.
struct TrimLinearisation::State
{
    std::unique_ptr<const AnalysisModel> model;
    std::unique_ptr<const aero::AeroBoxes> boxes;
    std::unique_ptr<const aero::SplineMatrix> splines;
    aero::AeroReference reference;
    std::unique_ptr<const assembly::RigidBodyModes> rigidModes;
    std::unique_ptr<const linalg::DenseKernels> kernels;
    // The condensed path: what CondensedStatics borrows.
    std::shared_ptr<const reduction::CondensationPlan> plan;
    std::vector<StaticSubcase> staticSubcases;
    std::vector<output::OutputPlan> outputPlans;
    std::vector<std::size_t> solved;
    std::vector<assembly::SpcSet> spcSets;
    std::unique_ptr<assembly::LoadVectorBuilder> loadBuilder;
    std::unique_ptr<CondensedStatics> condensed;
    std::vector<std::unique_ptr<trim::TrimGroup>> groups;
    std::vector<trim::TrimSubcaseState> subcases;
};

} // namespace oos::solutions
