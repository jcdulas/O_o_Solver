#pragma once

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/common/EigenMethod.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

namespace oos::design::detail
{

// The fixed component of the solver's Point/Max normalisation, or no component for Mass.
// An equal maximum has no unique derivative and is refused for field responses.
[[nodiscard]] std::optional<std::size_t> modalNormalisationPivot(
    const solutions::EigenMethod& method, std::span<const double> shape,
    const linalg::DofSelection& selection, const assembly::DofMap& dofMap);

// The original vibration pencil on the solved equations, shared by modal field gradients and eigenvalue Hessians.
// It borrows the analysis model and budget. Dependent DOFs use the same Gm reduction as the eigensolution, and the
// mass carries the non-structural mass set the subcase selects (NonStructMassSelection), as the eigensolution's does:
// K − λ·M is then the pencil the modes of the analysis satisfy.
class ModalSystem
{
public:
    ModalSystem(const solutions::AnalysisModel& model, std::int32_t subcase, base::MemoryBudget& budget,
                base::ThreadPool* pool, bool needsMass = true);

    [[nodiscard]] const solutions::ConstrainedSystem& constraints() const noexcept { return *m_constraints; }
    [[nodiscard]] const linalg::DofSelection& selection() const noexcept { return m_ordering->selection(); }
    [[nodiscard]] const linalg::SymmetricCsc& stiffness() const noexcept { return m_stiffness; }
    [[nodiscard]] const linalg::SymmetricCsc& mass() const noexcept { return m_mass; }
    [[nodiscard]] const assembly::MassAssemblyOptions& massOptions() const noexcept { return m_massOptions; }
    // The non-structural mass set the subcase selects, or nothing.
    [[nodiscard]] std::optional<base::EntityId> selectedNonStructuralMass() const noexcept { return m_selectedMass; }
    // The mass that set adds to each row of batch `batchIndex` of the model's element set (elements::MassOptions::
    // addedNonStructuralMass, for the derivatives of the element masses of this pencil); empty when the batch takes
    // none, or when the system was made without its mass.
    [[nodiscard]] std::span<const double> addedNonStructuralMass(std::size_t batchIndex) const noexcept
    {
        return m_shares ? m_shares->sharesOf(batchIndex) : std::span<const double>();
    }

private:
    std::unique_ptr<solutions::ConstrainedSystem> m_constraints;
    std::unique_ptr<solutions::FsetOrdering> m_ordering;
    linalg::SymmetricCsc m_stiffness;
    linalg::SymmetricCsc m_mass;
    assembly::MassAssemblyOptions m_massOptions;
    std::optional<base::EntityId> m_selectedMass;
    std::optional<assembly::NonStructuralMassShares> m_shares; // of the selected set, made with the mass
};

} // namespace oos::design::detail
