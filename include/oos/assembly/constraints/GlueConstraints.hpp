#pragma once

#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/GluedContactPairs.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElementEquations.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/RigidTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::assembly
{

// The other constraints of the subcase, which a glued grid point must not meet (GlueConstraints, "points left out").
struct GlueContext
{
    const input::RigidTable* rigidElements = nullptr; // null: no rigidElementsTable
    const MpcEquations* mpc = nullptr;
    const FastenerConstraints* fasteners = nullptr;
    std::span<const base::DofIndex> spcDofs;       // the sb set of the subcase (SpcSet::sbDofs)
    std::span<const base::DofIndex> permanentDofs; // the sg set (SpcSet::sgDofs: GRID PS)
};

// What the glue did with one glued pair: its tolerances, the grid points it ties and those it does not.
struct GluedPairReport
{
    GluedPair pair;
    double typicalElementSize = 0.0;      // the mean edge length of the primary faces
    double positionTolerance = 0.0;       // ERROR when given, else the default tolerance
    double inPlaneTolerance = 0.0;        // the default tolerance, DEFAULT_TOLERANCE_FRACTION of the typical size
    std::int64_t candidatePointCount = 0; // the secondary grid points of contactNodePairs
    std::int64_t primaryFaceCount = 0;    // the primary faces of contactNodePairs
    std::int64_t tiedPointCount = 0;
    std::int64_t outsidePointCount = 0; // no face within the tolerances: not tied
    std::int64_t leftOutPointCount = 0; // already dependent or held, glued twice, a loop of glue
    std::int64_t faceCornerCount = 0;   // the distinct face corners the tied points depend on (the mesh ratio)
};

// The glue equations of the glued contact of a subcase: the node-to-surface tie of the open-source solver CalculiX
// (G. Dhondt, CalculiX CrunchiX User's Manual, version 2.21, 2023, section 7.132 *TIE, "tied contact"), eliminated
// exactly as the MPCs and the rigid elements are (the dependent DOFs into the m set through Gm: no penalty, no stiff
// spring). For each glued pair of bodies (GluedContactPairs; the secondary body is the slave surface, the primary body
// the master surface):
//
// - The surfaces. O_o_Reader's contactNodePairs, taken with the reader's own rule (rule="components"), gives the
//   grid points of the secondary surface near the primary surface and, for each, a face of the primary surface near
//   it. The glue reads those rows as the two surfaces near each other: the candidate points are the distinct grid
//   points of the rows searched from the secondary body, the primary faces the distinct faces of those rows (the rows
//   of the search from the primary body are not read: the tie is one way, slave to master). Nothing else of the rows
//   is read (not their choice of face, projection, local coordinates nor gap): the projection is made here.
// - The search (GlueSearch): every candidate point is projected on the primary faces near it; it is tied to the face
//   whose nearest point is nearest, when its orthogonal distance to that face is at most the position tolerance and
//   its projection lies within the face or no farther outside, parallel to the face, than the default tolerance (the
//   manual's rule). A point no face takes is not tied ("glue-points-outside-tolerance").
// - The tolerances. The default tolerance is DEFAULT_TOLERANCE_FRACTION (2.5 %, the manual's default) of the typical
//   element size, which the manual does not define further: here the mean length of the corner-to-corner edges of the
//   pair's primary faces (this project's choice). The position tolerance is the pair's ERROR when the deck gives one
//   (the distance within which the format pairs a grid point with a face of the other body, read here as the tie's
//   POSITION TOLERANCE: this project's choice), else the default tolerance.
// - The tie: u_s = R_sᵀ·Σ_i N_i(ξ, η)·R_i·u_i on the translations (components 1 2 3), N_i the shape functions of
//   the face (faceShapeFunctions: linear or quadratic, quadrilateral or triangular) at the tied point, R the rotation
//   from the displacement system CD of a grid point to the basic system. Σ N_i = 1, so a rigid translation gives no
//   tie force; a rigid rotation gives none where the point lies on its face (a point kept off its face by a gap feels
//   the rotation times the gap, as with the manual's ADJUST=NO).
// - Translations only, for solids and shells alike, as the tie of the manual: it ties the displacements of
//   three-dimensional elements only, shells and beams being expanded into them (section 7.132, first paragraph), and
//   "the connection between 3D elements and all other elements (1D or 2D) is always hinged" (section 6.2.11). The
//   rotations of a secondary shell grid point are left free (its own shells hold them): a shell edge tied along one
//   line is a hinge about that line, and a moment passes only through tie forces at points apart (an overlap of two
//   rows of tied points or more). IGLUE 3 and 4, which ask for a glue of the shell rotations too, get the same tie
//   and a warning ("glue-rotations-not-tied").
// - The grid points are not moved. The manual moves a tied slave node onto the master surface by default
//   (ADJUST=YES), but the model snapshot and its geometry are shared and immutable here, so the tie is that of
//   ADJUST=NO; a pair whose tied points lie off their faces is reported with the largest distance kept
//   ("glue-gap-kept"). IGLUE 1 or 3 (move) and 2 or 4 (keep) make no difference. Every tied point is within the
//   position tolerance of its face.
// - JGLUE (permanent in a linear analysis), ICOORD, ISEARCH and EulerMeshBias change nothing.
//
// Conflicts, after the manual's rule that a slave node is no dependent node of another constraint: a component of a
// secondary point that an SPC, SinglePointConstrList or GRID PS holds is not tied (the face is never constrained
// through it); a component another relation already makes dependent (rigid element, fastener, MPC) keeps that relation;
// the other components are tied ("glue-components-left"). A point none of whose translations is free is left out
// ("glue-points-left-out"); a grid point of the primary surface itself (shared by the bodies) has nothing to tie. A
// point another pair already ties keeps the first tie (pairs in the order of contactPairsTable, points by increasing
// ID; "glue-point-twice"), and an equation that would close a loop of glue equations is left out
// ("glue-dependency-cycle"); chains of glue, rigid elements and MPCs are resolved by composition in RigidElimination,
// whose own cycle check covers the mixed loops. A pair that ties one or two points only is a hinge, warned
// ("glued-pair-hinge"; the singularity processor reports the mechanism, never hidden by a stiffness). A face that
// cannot interpolate (a SlidelineSegments segment, a point the model does not define, coincident corners) is an error
// "invalid-contact-face" (the first one of each pair, with the count).
//
// Every glued pair is reported by an Information diagnostic "glued-pair" (the run report and the model check list
// them). The glue forces are the MPC forces of these rows (RigidElimination::computeMpcForces).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its rows (budgeted); the budget must outlive it.
class GlueConstraints
{
public:
    // The default tolerance of the tie as a share of the typical element size (CalculiX manual, section 7.132: "2.5% of
    // the typical element size").
    static constexpr double DEFAULT_TOLERANCE_FRACTION = 0.025;
    // The distance from a tied point to its face, relative to the typical element size, under which the point is on its
    // face (round-off of the geometry) and no gap is reported.
    static constexpr double GLUE_GAP_TOLERANCE = 1.0e-6;

    // `contactId`: the contact table the subcase selects (BCONTACT), nothing for none. Throws ContractError when the
    // contact tables lack the glue columns (requireGlueContract), MemoryBudgetError.
    GlueConstraints(const input::ModelSnapshot& snapshot, std::optional<base::EntityId> contactId,
                    base::EntityId subcase, const DofMap& dofMap, const GlueContext& context,
                    base::MemoryBudget& budget, base::DiagnosticSink& sink);

    [[nodiscard]] std::size_t rowCount() const noexcept { return m_rowCount; }
    [[nodiscard]] std::int64_t termCount() const noexcept { return m_termCount; }
    // Row i: its dependent DOF, its terms on the face grid DOFs, its contact table card and ID.
    [[nodiscard]] base::DofIndex dependentDof(std::size_t i) const noexcept { return m_dependents[i]; }
    [[nodiscard]] std::span<const ConstraintTerm> row(std::size_t i) const noexcept
    {
        return {m_terms.data() + m_offsets[i], static_cast<std::size_t>(m_offsets[i + 1] - m_offsets[i])};
    }
    [[nodiscard]] ConstraintSource sourceOf(std::size_t i) const noexcept
    {
        return static_cast<ConstraintSource>(m_sources[i]);
    }
    [[nodiscard]] base::EntityId sourceIdOf(std::size_t i) const noexcept { return m_sourceIds[i]; }

    // The glued pairs of the subcase, in the order of contactPairsTable.
    [[nodiscard]] const std::vector<GluedPairReport>& pairs() const noexcept { return m_pairs; }

private:
    std::vector<GluedPairReport> m_pairs; // bounded by the pairs of one contact table
    base::BudgetedBuffer<base::DofIndex> m_dependents;
    base::BudgetedBuffer<std::int64_t> m_offsets;
    base::BudgetedBuffer<ConstraintTerm> m_terms;
    base::BudgetedBuffer<std::uint8_t> m_sources;
    base::BudgetedBuffer<base::EntityId> m_sourceIds;
    std::size_t m_rowCount = 0;
    std::int64_t m_termCount = 0;
};

} // namespace oos::assembly
