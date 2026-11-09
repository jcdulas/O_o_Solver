#pragma once

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/linalg/eigen/PencilSource.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::solutions
{

// The pencil of SOL 103 or SOL 105 from the elements, no matrix of the model assembled (linalg::PencilSource): K, M and
// K_d made element by element where they are needed. A front takes the elements placed at its supernode
// (StreamedElements: the placement of the full path streamed), each element's matrices made then and carried onto the
// equations (StreamedReduction: T_eᵀ·A_e·T_e, Gm's rows on the dependent DOFs), the terms of a combination α·K + β·M +
// γ·K_d added in a fixed order (the elements increasing; each element's stiffness, then its mass, then its differential
// stiffness, each in its DOFs' order). The products Y = A·X are made the same way over the element set in pieces on the
// pool, each piece's terms merged in piece order; a mass whose every element is diagonal on the equations (a lumped
// mass without relations) is multiplied by its diagonal (summed once).
//
// The bits: the same as the assembled path's to round-off, not to the bit: an entry of the assembled K_nn is the sum
// of its elements' terms in the assembly's order, then Gm's reduction (MsetElimination); here each element's terms are
// reduced first and the entries of one front are summed in the placement's order, the entries of the rows carried up
// by the update matrices (extend-add), WTMASS applied to each element's mass rather than once to M_gg. Deterministic:
// the same bits whatever the threads.
//
// B, the pencil's right-hand matrix: M for vibration, −K_d for buckling (K_d of the preload: its g-set displacements
// and temperatures, as DifferentialStiffnessAssembler); the mass source of a buckling pencil (the multilevel
// substructuring's vibration basis) M.
//
// Memory: the placement (borrowed), the mass's diagonal on the equations when it is diagonal, a wave of pieces' terms
// in the products (workBytes()). Thread-safety: the sources' addTerms() are safe concurrently (each call its own
// scratch); the products and diagonals are const, called from one thread at a time. Ownership: borrows the model, the
// placement, the shares, the preload and the pool, which outlive it.
class ElementPencil final : public linalg::PencilSource
{
public:
    struct Inputs
    {
        const AnalysisModel* model = nullptr;
        const StreamedElements* elements = nullptr;
        assembly::MassAssemblyOptions massOptions;
        const assembly::NonStructuralMassShares* shares = nullptr;
        linalg::SpectralTransform transform = linalg::SpectralTransform::Vibration;
        // Buckling: the preload's displacements (one per g-set DOF of the DOF map) and temperatures (null: none).
        std::span<const double> preloadDisplacements;
        const elements::ElementTemperatures* preloadTemperatures = nullptr;
        // Buckling: whether the mass source is asked (the multilevel substructuring's vibration basis); not given when
        // the analysis set has no mass.
        bool hasMassSource = false;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
    };

    // Throws std::invalid_argument for a buckling pencil without the preload's g-set displacements; MemoryBudgetError.
    ElementPencil(const Inputs& inputs, base::MemoryBudget& budget);

    [[nodiscard]] std::int32_t dimension() const noexcept override;
    [[nodiscard]] linalg::FrontSource& stiffnessSource() override { return m_stiffness; }
    [[nodiscard]] linalg::FrontSource& rightHandSource() override { return m_rightHand; }
    [[nodiscard]] linalg::FrontSource* massSource() override;
    [[nodiscard]] linalg::FrontSource& shiftedSource(double shift) override;
    void multiplyStiffness(linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const override;
    void multiplyRightHand(linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const override;
    void diagonals(std::span<double> stiffness, std::span<double> rightHand) const override;

    // Whether a buckling pencil gives its mass source (asked, and a mass on the analysis set).
    [[nodiscard]] bool hasMassSource() const noexcept { return m_inputs.hasMassSource; }
    // Whether the mass is held on the equations (a lumped mass: its diagonal and its few terms off it), not multiplied
    // element by element.
    [[nodiscard]] bool isMassHeld() const noexcept { return !m_massDiagonal.empty(); }
    // An upper bound of the work memory of a product of `columns` columns (a wave of pieces' terms).
    [[nodiscard]] static std::uint64_t workBytes(std::size_t threads, std::size_t columns) noexcept;

    // The coefficients of a combination α·K + β·M + γ·K_d.
    struct Combination
    {
        double stiffness = 0.0;
        double mass = 0.0;
        double differential = 0.0;
    };

private:
    // The front source of one combination.
    class Source final : public linalg::FrontSource
    {
    public:
        Source(const ElementPencil& pencil, Combination combination) : m_pencil(pencil), m_combination(combination) {}
        void addTerms(linalg::FrontTerms& front) override;
        // The MAXRATIO diagonal is read by the positive definite factorisation alone, never of a pencil's source
        // (MultifrontalDriver's indefinite mode): zero.
        void diagonal(std::int32_t supernode, std::span<double> values) const override;
        void setCombination(Combination combination) noexcept { m_combination = combination; }

    private:
        const ElementPencil& m_pencil;
        Combination m_combination;
    };

    // Y ← A·X over the element set, A the combination; or its diagonal into `diagonal` (X, Y unused).
    void multiply(Combination combination, linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const;
    // A term of the lower triangle off the diagonal (row > column, equations).
    struct OffDiagonalTerm
    {
        std::int32_t row = 0;
        std::int32_t column = 0;
        double value = 0.0;
    };
    // The diagonal of the combination on the equations and, with `offDiagonal`, its terms off it in the order they
    // come, up to `limit` of them (*isHeld false beyond: none kept).
    void sumDiagonal(Combination combination, std::span<double> diagonal,
                     std::vector<OffDiagonalTerm>* offDiagonal = nullptr, bool* isHeld = nullptr,
                     std::size_t limit = 0) const;

    Inputs m_inputs;
    Combination m_rightHandCombination;
    Source m_stiffness;
    Source m_rightHand;
    Source m_mass;
    Source m_shifted;
    elements::NodalDisplacements m_preload{};                // buckling: the preload of K_d
    base::BudgetedBuffer<double> m_massDiagonal;             // per equation, when the mass is held (vibration)
    base::BudgetedBuffer<OffDiagonalTerm> m_massOffDiagonal; // by column then row
};

} // namespace oos::solutions
