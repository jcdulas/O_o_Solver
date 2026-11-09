#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparseAssembler.hpp"

#include <span>

namespace oos::assembly
{

// The differential stiffness matrix K_d of the g set under a preload (SOL 105): the
// element differential stiffness matrices of every batch with stiffness (ElementFamily::computeDifferentialStiffness,
// from the element's internal forces under the preload displacements, and its temperatures for a thermal preload), in
// the displacement systems of the grid points, added by linalg::SparseAssembler on the node pattern of the stiffness
// (the pattern couples the nodes of every element, so K_d has the structure of K: one symbolic factor serves
// K + σ·K_d). The element types without differential stiffness (ElementFamily::isDifferentiallyInert: springs, bushes,
// gaps, fasteners) add nothing. The order of the sums is that of StiffnessAssembler, whatever the number of threads.
//
// SOL 105 solves (K + λ·K_d)·φ = 0 with K_d of the preload at scale 1.
//
// Thread-safety: const methods; assemble() may run concurrently on different targets. Ownership: borrows the set and
// the DOF map, which must outlive it.
class DifferentialStiffnessAssembler
{
public:
    DifferentialStiffnessAssembler(const ElementSet& set, const DofMap& dofMap, linalg::AssemblyOptions options = {})
        : m_set(&set), m_dofMap(&dofMap), m_options(options)
    {
    }

    // Reports every element whose differential stiffness the facets cannot compute
    // (ElementFamily::checkDifferentialStiffness of every batch with stiffness).
    void check(base::DiagnosticSink& sink) const;

    // Adds K_d to `target` (not cleared first). displacementsG: the preload displacements, one value per g-set DOF of
    // the DOF map; temperatures: those of a thermal preload, or null. Preconditions: check() reported no error, and the
    // pattern of `target` couples the nodes of every element (ElementConnectivity). Throws std::invalid_argument when
    // displacementsG is not of the g-set size, and as linalg::SparseAssembler.
    void assemble(std::span<const double> displacementsG, const elements::ElementTemperatures* temperatures,
                  linalg::BlockSparseMatrix& target, base::ThreadPool* pool = nullptr,
                  const base::CancellationToken* cancellation = nullptr) const;

private:
    const ElementSet* m_set;
    const DofMap* m_dofMap;
    linalg::AssemblyOptions m_options;
};

} // namespace oos::assembly
