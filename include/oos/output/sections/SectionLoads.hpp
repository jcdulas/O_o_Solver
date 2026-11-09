#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/DesignFieldUse.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/sections/SectionDefinition.hpp"
#include "oos/output/sections/SparseRows.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace oos::output
{

// An axis whose projection on the plane of a shell is shorter than this fraction of its length has no direction in
// that plane (an axis along the normal of the element): the section frame is undefined there, and the section is
// refused naming the element. The same limit as an MCID projection.
inline constexpr double SECTION_AXIS_MIN_PROJECTION = 1.0e-8;

// The internal loads of the sections the caller defines (a weighted combination of the forces of their elements), as
// rows: section by section in the order of the definitions, the components of its family (sectionComponents) one row
// each.
//
// For a member e of weight w_e with element forces F_e (the FORCE output of its family at the centre of a shell, at the
// ends of a bar or beam), the rows of a section are
//     S = Σ_e w_e·T_e·F_e / Σ_e w_e   (reduce "average"),   S = Σ_e w_e·T_e·F_e   (reduce "sum"),
// T_e the rotation from the frame of the element into the section frame for a shell (identity otherwise): with
// x_s the projection of the section axis on the plane of the element, c = x_s·x_e and s = x_s·y_e (x_e, y_e the axes of
// the element frame of the shell family, the frame of the corners for CQUAD8 and CTRIA6), the tensor rotation of the
// membrane forces and of the moments (Timoshenko and Goodier, plane stress transformation; the force resultants per
// unit length are the components of symmetric tensors in the element plane)
//     N'x = c²Nx + s²Ny + 2csNxy,  N'y = s²Nx + c²Ny − 2csNxy,  N'xy = −csNx + csNy + (c² − s²)Nxy,
// and the vector rotation of the transverse shears, Q'x = cQx + sQy, Q'y = −sQx + cQy. The y_s of an element whose
// normal opposes the section's (the area-weighted mean of its shells' normals) is turned to point as the section's
// (N'xy, M'xy, Q'y change sign), as O_o_Reader's readBlock does for element groups: the same numbers for the same
// group; its moments and transverse shears stay those the element gives about its normal. The default weight of a
// shell is its area (in its plane: 4·det J0 for a quadrilateral), recomputed from the geometry it is evaluated on (the
// sensitivities to a shape move it).
//
// The forces of the elements are affine in the displacements, F_e = R_e·u_e + F_e(0, T) (F_e(0, T) the thermal part of
// a thermal load, zero otherwise), so the rows are too: displacementRows() holds Σ_e w_e·T_e·R_e/W on the g-set, made
// once from the families' own recovery of unit displacements (exact: the recovery is linear), on which the values of a
// load basis are one product and the sensitivities read their weights; evaluate() recovers the members themselves (any
// displacements and temperatures, on the element set of the snapshot or of an overlay of it with the same batches).
//
// Definition problems are diagnostics of the sink (errors, naming the section and the element): an element the model
// does not have, an element type no section takes (sectionFamilyOf), elements of two families in one section, an
// element named twice, a total weight of zero, a weight that is not finite, a shell section without its frame (an axis
// or a CID, exactly one), a CID the model does not have or that is not rectangular, an axis normal to a member. A
// section with a problem has no rows.
//
// Thread-safety: immutable once built; evaluate() is safe concurrently. Ownership: borrows the element set, the DOF map
// and the snapshot (for the CIDs), which must outlive it; owns its definitions and rows.
class SectionLoads
{
public:
    // A member resolved: its batch and row in the element set.
    struct Member
    {
        std::size_t batch = 0;
        std::int64_t row = 0;
        std::int32_t elementId = 0;
        std::optional<double> weight;
    };
    // A section resolved: its definition, family, first row, members and, for a shell section, its axis in basic.
    struct Section
    {
        std::size_t definition = 0;
        SectionFamily family = SectionFamily::Shell;
        std::size_t firstRow = 0;
        std::vector<Member> members; // bounded by the definition
        base::Vec3 axis;
    };

    SectionLoads(std::vector<SectionDefinition> definitions, const input::ModelSnapshot& snapshot,
                 const assembly::ElementSet& set, const assembly::DofMap& dofMap, base::DiagnosticSink& sink);

    [[nodiscard]] const std::vector<SectionDefinition>& definitions() const noexcept { return m_definitions; }
    [[nodiscard]] const std::vector<Section>& sections() const noexcept { return m_sections; }
    [[nodiscard]] std::size_t rowCount() const noexcept { return m_rowCount; }
    // The rows of section s: [firstRow, firstRow + componentCount).
    [[nodiscard]] std::size_t componentCount(std::size_t section) const noexcept
    {
        return sectionComponents(m_sections[section].family).size();
    }

    // The displacement part of the rows (the class comment), made at the first call from the families' recovery of unit
    // displacements of every member DOF, its arrays reserved in `budget`. Not thread-safe at the first call.
    // Made once, section by section on `pool` (each row the same bits whatever the threads: a row is one section's,
    // its members in order).
    [[nodiscard]] const SparseRows& displacementRows(base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    [[nodiscard]] bool hasDisplacementRows() const noexcept { return m_rowsBuilt; }
    // The rows once made (precondition: hasDisplacementRows()).
    [[nodiscard]] const SparseRows& builtDisplacementRows() const noexcept { return m_rows; }

    // The values of every row (out: rowCount() values) for the g-set displacements `displacementsG` and the element
    // temperatures of a thermal load (or null), the members recovered by the families of `set` (that of the
    // constructor, or of an overlay of its snapshot with the same element batches). Throws std::invalid_argument for
    // sizes that disagree.
    void evaluate(const assembly::ElementSet& set, std::span<const double> displacementsG,
                  const elements::ElementTemperatures* temperatures, std::span<double> out) const;
    // The rows of one section only (out: componentCount(section) values).
    void evaluateSection(std::size_t section, const assembly::ElementSet& set, std::span<const double> displacementsG,
                         const elements::ElementTemperatures* temperatures, std::span<double> out) const;

    // The derivatives of the rows of section s (out: componentCount(section) values) with respect to the designed field
    // `field` at fixed displacements, in closed form: the sum of evaluateSection at the rates of the
    // members' forces, C_m·dF_m/dv, the rates from their families (ElementFamily::recoveryFieldRates) and C_m from
    // memberCoefficients, which the field does not move (the weights and frames are the members' geometry: the caller
    // asks it for a field of a property, a material or an element card, never a shape). `isReached(batch, row)` says
    // whether the field reaches a member (the others' rates are zero). False when a reached member's family gives no
    // closed form, or a NaN in a column the section reads: the caller then differences the section.
    [[nodiscard]] bool sectionFieldRates(std::size_t section, const assembly::ElementSet& set,
                                         std::span<const double> displacementsG,
                                         const elements::ElementTemperatures* temperatures,
                                         const elements::DesignedField& field,
                                         const std::function<bool(std::size_t, std::int64_t)>& isReached,
                                         std::span<double> out) const;

    // The coefficients of the element forces in the rows of section s on `set`: per member, w·T/W (W the total weight
    // for "average", 1 for "sum"), components × components of its family, row-major in an array of
    // MAX_SECTION_COMPONENTS columns: row k of the section is Σ_m Σ_l C_m(k, l)·F_m(l). The design module
    // uses them as the constant coefficients of a DRESP2 of DRESP1 FORCE items.
    using Coefficients = std::array<double, MAX_SECTION_COMPONENTS * MAX_SECTION_COMPONENTS>;
    [[nodiscard]] std::vector<Coefficients> memberCoefficients(std::size_t section,
                                                               const assembly::ElementSet& set) const;

    // The section of a bar or beam member at row `row` of batch `batch`, if any: such a member under a line load
    // (PLOAD1) gives its forces at its stations, a layout the sections do not read (the caller refuses the case).
    [[nodiscard]] std::optional<std::size_t> lineSectionOf(std::size_t batch, std::int64_t row) const;

private:
    struct ColumnMap
    {
        bool isKnown = false;
        std::array<std::size_t, MAX_SECTION_COMPONENTS> columns{};
    };

    void resolve(const input::ModelSnapshot& snapshot, base::DiagnosticSink& sink);
    // The element forces of one member, in the order of its family's components; false when the family gives no row.
    void memberForces(const assembly::ElementSet& set, const Member& member, SectionFamily family,
                      const elements::NodalDisplacements& nodal, std::span<double> forces) const;
    // The element forces of one member from the rows of its force recovery (memberForces after the recovery);
    // std::logic_error when the block has no row.
    void memberForcesOf(const elements::ElementResultBlock& block, const Member& member, SectionFamily family,
                        std::span<double> forces) const;
    // The components of a member's family read from the rows of its recovery (or of their rates).
    void forcesOfBlock(const elements::ElementResultBlock& block, const Member& member, SectionFamily family,
                       std::span<double> forces) const;

    std::vector<SectionDefinition> m_definitions;
    const assembly::ElementSet* m_set;
    const assembly::DofMap* m_dofMap;
    std::vector<Section> m_sections;         // bounded by the definitions
    std::vector<ColumnMap> m_columnsOfBatch; // bounded by the element types
    std::size_t m_rowCount = 0;
    // (batch, row, section) of the bar and beam members, sorted (lineSectionOf).
    std::vector<std::tuple<std::size_t, std::int64_t, std::size_t>> m_lineMembers;
    bool m_rowsBuilt = false;
    SparseRows m_rows;
};

} // namespace oos::output
