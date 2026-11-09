#pragma once

#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/linalg/eigen/ShiftedPencil.hpp"
#include "oos/linalg/symbolic/Amalgamation.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/ElementPencil.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/sol101/StreamedElements.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace oos::solutions
{

// The analysis set of SOL 103 or SOL 105 and its pencil from the elements, no matrix of the model assembled: the
// constrained system of the selection without Kgg nor Knn (ConstrainedSystemOptions::diagonalBlocks: the grid point
// singularity processor on the n-set diagonal blocks summed element by element, the full path streamed's
// StreamedFront::nodeDiagonals), the ordering of its f set, the placement of every element at the supernode that
// eliminates its first equation (StreamedElements) and the ElementPencil on it. The shapes on the analysis set are
// expanded by ModeRecovery on system() (Gm and the sets; system().kgg() does not exist).
//
// Memory: Gm, the sets, the node graph, the ordering's symbolic factor, the placement, and a vibration pencil's lumped
// mass diagonal; nothing of the size of a matrix of the model. Thread-safety: as its parts (built once, then const but
// the pencil). Ownership: owns its parts; borrows the model and the preload, which outlive it. Not movable.
class ElementEigenSystem
{
public:
    struct Inputs
    {
        const AnalysisModel* model = nullptr;
        ConstraintSelection selection;
        // The supernodes of the ordering (the multilevel substructuring's of at most 256 columns,
        // orderingThresholdsOf).
        linalg::AmalgamationThresholds thresholds;
        linalg::SpectralTransform transform = linalg::SpectralTransform::Vibration;
        // Buckling: the preload's g-set displacements and temperatures (null: none).
        std::span<const double> preloadDisplacements;
        const elements::ElementTemperatures* preloadTemperatures = nullptr;
        base::ThreadPool* pool = nullptr;
        const base::CancellationToken* cancellation = nullptr;
    };

    // Throws what ConstrainedSystem, FsetOrdering, StreamedElements and ElementPencil throw.
    ElementEigenSystem(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    ElementEigenSystem(const ElementEigenSystem&) = delete;
    ElementEigenSystem& operator=(const ElementEigenSystem&) = delete;
    ElementEigenSystem(ElementEigenSystem&&) = delete;
    ElementEigenSystem& operator=(ElementEigenSystem&&) = delete;
    ~ElementEigenSystem();

    [[nodiscard]] const ConstrainedSystem& system() const noexcept { return *m_system; }
    [[nodiscard]] const FsetOrdering& ordering() const noexcept { return *m_ordering; }
    // The pencil; null for an empty analysis set (every DOF constrained).
    [[nodiscard]] ElementPencil* pencil() noexcept { return m_pencil.get(); }
    // The work memory of the pencil's products of `columns` columns (the out-of-core plan's).
    [[nodiscard]] std::uint64_t productBytes(std::size_t columns) const noexcept;

private:
    base::ThreadPool* m_pool;
    std::unique_ptr<ConstrainedSystem> m_system;
    std::unique_ptr<FsetOrdering> m_ordering;
    base::BudgetedBuffer<std::uint8_t> m_isOutputNode; // no point placed: the shapes are expanded after the solve
    std::unique_ptr<StreamedElements> m_elements;
    std::unique_ptr<assembly::NonStructuralMassShares> m_shares;
    std::unique_ptr<ElementPencil> m_pencil;
};

} // namespace oos::solutions
