#pragma once

#include "oos/design/sensitivity/FieldPerturbation.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::design
{

// The derivative of one element matrix with respect to a field: the element's DOFs and dM/dv over them, row-major
// (both triangles). Empty (no DOFs) when the element has no such matrix.
struct ElementMatrixDerivative
{
    std::vector<elements::ElementDof> dofs;
    std::vector<double> values;

    [[nodiscard]] bool isEmpty() const noexcept { return dofs.empty(); }
};

// The derivative of the equivalent nodal loads of one element (its thermal loads) with respect to a field.
struct ElementVectorDerivative
{
    std::vector<elements::ElementDof> dofs;
    std::vector<double> values;

    [[nodiscard]] bool isEmpty() const noexcept { return dofs.empty(); }
};

// The semi-analytic derivatives of the element computations with respect to one designed field (the
// element matrices differenced with respect to the designed field, then exact linear algebra): the family's own
// facets run on the snapshots where the field is v ± h and v ± h/2 (FieldPerturbation), every other field at its
// value of the design point, and their differences extrapolated (FieldPerturbation::derivative) are the derivative.
// Nothing in the element families is written twice: whatever an element computes from its fields (a section, a
// laminate, an offset, a thermal strain), its derivative follows.
//
// The four sides of an element must have the same layout (the same DOFs, the same rows): checked, std::logic_error
// otherwise (a facet whose output layout depends on a real field value cannot be differenced).
//
// Thread-safety: stateless besides its borrowed perturbation; concurrent calls are safe. Ownership: borrows the
// family and the perturbation.
class ElementDerivatives
{
public:
    // `batch`: the index of the element's batch in the element sets of the perturbation.
    ElementDerivatives(const elements::ElementFamily& family, const FieldPerturbation& perturbation,
                       std::size_t batch) noexcept
        : m_family(&family), m_perturbation(&perturbation), m_batch(batch)
    {
    }

    // dK_e/dv of the element of a row.
    [[nodiscard]] ElementMatrixDerivative stiffness(std::int64_t row) const;
    // dM_e/dv of the element of a row: its mass facet with `options` (the eigenvalue derivatives), in the
    // mass units of the deck (WTMASS is applied by the caller, as MassAssembler applies it to the assembled matrix).
    // Empty for an element without mass.
    [[nodiscard]] ElementMatrixDerivative mass(std::int64_t row, const elements::MassOptions& options) const;
    // dK_d/dv of the element of a row at fixed preload displacements and temperatures (the derivatives of
    // the buckling load factors: the explicit part, the preload's own derivative being taken apart).
    [[nodiscard]] ElementMatrixDerivative differentialStiffness(std::int64_t row,
                                                                const elements::NodalDisplacements& preload) const;
    // df_e/dv of the thermal loads of the element of a row under `temperatures` (empty for an element without
    // thermal load).
    [[nodiscard]] ElementVectorDerivative thermalLoads(std::int64_t row,
                                                       const elements::ElementTemperatures& temperatures) const;
    // df_e/dv of the pressures and line loads of the element of a row (its entries of one load set,
    // assembly::LoadVectorBuilder::elementDistributedLoads), empty without any.
    [[nodiscard]] ElementVectorDerivative distributedLoads(std::int64_t row,
                                                           std::span<const elements::ElementPressure> pressures,
                                                           std::span<const elements::ElementLineLoad> lineLoads) const;

private:
    // The derivative of the load vectors `compute(view, block)` gives on the four sides, summed DOF by DOF in the order
    // of the plus side.
    template <typename Compute>
    [[nodiscard]] ElementVectorDerivative vectorDerivative(std::int64_t row, const char* what, Compute&& compute) const;
    [[nodiscard]] ElementMatrixDerivative matrixDerivative(
        std::array<elements::ElementMatrixBlock, FIELD_SIDE_COUNT>& blocks, std::int64_t row, const char* what) const;
    [[nodiscard]] const elements::ElementBatchView& view(FieldSide side) const noexcept
    {
        return m_perturbation->set(side).batches()[m_batch].view;
    }

    const elements::ElementFamily* m_family;
    const FieldPerturbation* m_perturbation;
    std::size_t m_batch;
};

} // namespace oos::design
