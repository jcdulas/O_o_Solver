#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/sections/CutDefinition.hpp"
#include "oos/output/sections/SparseRows.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace oos::output
{

// The loads through the cuts the caller defines (the integrated grid point forces of a cut), and the computation of a
// MONPNT3: six rows per cut, fx, fy, fz, mx, my, mz (CUT_COMPONENTS).
//
// The force that element e exerts on its grid point g is the element row of the grid point forces (GPFORCE,
// StiffnessRecovery): q_e,g = −(K_e·u_e − f_e,T)|g, in the displacement system CD_g of the grid point (f_e,T the
// equivalent loads of a thermal load). Summed over the elements E of the cut on the grid points G of the cut, with the
// rotations R_g from CD_g to basic and the positions x_g in basic,
//     F = Σ_{e∈E} Σ_{g∈G} R_g·q_e,g^t,      M = Σ_{e∈E} Σ_{g∈G} (x_g − p) × R_g·q_e,g^t + R_g·q_e,g^r,
// (t the translations, r the rotations of q, p the point of the cut in basic), and the loads are A_CDᵀ·F and A_CDᵀ·M
// with A_CD the axes of the system CD in basic. For a cut through a structure, E the elements on one side, this is the
// resultant of the internal loads that side transmits through the cut: the shear and the bending moment of a beam at
// the cut.
//
// The loads are affine in the displacements (linear without a thermal load): displacementRows() holds the linear part
// on the g-set, −Σ a_g·K_e(g, :) over the elements and the DOFs of the cut (exact: the element stiffness matrices
// themselves), on which the loads of a load basis are one product and the sensitivities read their weights.
//
// For the MONPNT3 of a deck: a CutDefinition per card (GRIDSET and ELEMSET expanded to their IDs, CID and X, Y, Z
// as the point, CD), CutLoads built on the element set and the DOF map of the analysis, then per load case
// evaluateCut(c, set, ug, temperatures) (six values, along the axes of CD about the point), or for many cases
// displacementRows(budget).apply(U) for the displacements U of a load basis and the thermal parts added per case
// (evaluateCut at zero displacements) — SectionLoadRun does this for the sections and cuts of a static run.
//
// Definition problems are diagnostics of the sink (errors, naming the cut): a grid point or an element the model does
// not have, an empty set, a CP or CD that the model does not have or that is not rectangular. The elements of the set
// without stiffness (masses) exert no force and are left out; the elements that touch no grid point of the cut add
// nothing. A cut with a problem has no rows.
//
// Thread-safety: immutable once built; evaluate() is safe concurrently. Ownership: borrows the element set and the DOF
// map, which must outlive it; owns its definitions and rows.
class CutLoads
{
public:
    struct Element
    {
        std::size_t batch = 0;
        std::int64_t row = 0;
        std::int32_t elementId = 0;
    };
    struct Cut
    {
        std::size_t definition = 0;
        std::size_t firstRow = 0;
        std::vector<std::int32_t> gridRows; // sorted node table rows of the grid points of the cut
        std::vector<Element> elements;      // in the order of the batches
        base::Vec3 point;                   // in basic
        base::Mat3 cdAxes;                  // columns: the axes of CD in basic
    };

    CutLoads(std::vector<CutDefinition> definitions, const input::ModelSnapshot& snapshot,
             const assembly::ElementSet& set, const assembly::DofMap& dofMap, base::DiagnosticSink& sink);

    [[nodiscard]] const std::vector<CutDefinition>& definitions() const noexcept { return m_definitions; }
    [[nodiscard]] const std::vector<Cut>& cuts() const noexcept { return m_cuts; }
    [[nodiscard]] std::size_t rowCount() const noexcept { return CUT_COMPONENTS.size() * m_cuts.size(); }

    // The linear part of the rows (the class comment), made at the first call, its arrays reserved in `budget`. Not
    // thread-safe at the first call.
    // Made once, cut by cut on `pool` (each row the same bits whatever the threads).
    [[nodiscard]] const SparseRows& displacementRows(base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    [[nodiscard]] bool hasDisplacementRows() const noexcept { return m_rowsBuilt; }
    [[nodiscard]] const SparseRows& builtDisplacementRows() const noexcept { return m_rows; }

    // The six loads of cut c for the g-set displacements and the temperatures of a thermal load (or null), the element
    // matrices and the positions taken from `set` (that of the constructor, or of an overlay of its snapshot with the
    // same element batches). Throws std::invalid_argument for sizes that disagree.
    [[nodiscard]] std::array<double, 6> evaluateCut(std::size_t c, const assembly::ElementSet& set,
                                                    std::span<const double> displacementsG,
                                                    const elements::ElementTemperatures* temperatures) const;
    // The rows of every cut (out: rowCount() values).
    void evaluate(const assembly::ElementSet& set, std::span<const double> displacementsG,
                  const elements::ElementTemperatures* temperatures, std::span<double> out) const;

    // The derivatives of the six loads of cut c with respect to the designed field `field` at fixed displacements, in
    // closed form: evaluateCut is linear in each element's K_e, so its rate is evaluateCut with dK_e/dv of the elements
    // the field reaches (`isReached(batch, row)`; the others' rates are zero), the rates from their families
    // (ElementFamily::stiffnessFieldDerivative), the cut's grid points and frames being the geometry (the caller asks
    // it for a field of a property, a material or an element card, never a shape). False under a thermal load (its
    // equivalent loads move with the field too) or when a reached element's family gives no closed form: the caller
    // then differences the cut.
    [[nodiscard]] bool cutFieldRates(std::size_t c, const assembly::ElementSet& set,
                                     std::span<const double> displacementsG,
                                     const elements::ElementTemperatures* temperatures,
                                     const elements::DesignedField& field,
                                     const std::function<bool(std::size_t, std::int64_t)>& isReached,
                                     std::array<double, 6>& out) const;

private:
    // The 6 loads (along CD, about the point) of a force on DOF `dof` of grid row `gridRow` of unit value, on `set`.
    [[nodiscard]] std::array<double, 6> unitLoad(const Cut& cut, const elements::ElementBatchView& view,
                                                 std::int32_t gridRow, int component) const;
    [[nodiscard]] bool isCutGrid(const Cut& cut, const elements::ElementDof& dof) const;

    std::vector<CutDefinition> m_definitions;
    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    std::vector<Cut> m_cuts; // bounded by the definitions
    bool m_rowsBuilt = false;
    SparseRows m_rows;
};

} // namespace oos::output
