#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/ResultArrays.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oos::solutions
{

// The result tables of the modes of an eigenvalue solution (SOL 103 vibration modes, SOL 105 buckling shapes), one
// mode at a time, as the Case Control requests them (OutputPlan):
//   - DISPLACEMENT (or VECTOR): the "eigenvectors" table of the mode;
//   - SPCFORCES and MPCFORCES: the forces that hold the mode in equilibrium, from the residual of the g-set equation
//     r = Kgg·φ + c·Bgg·φ (c·Bgg = −λ·Mgg for a vibration mode, the inertia forces; none for a buckling shape, whose
//     forces come from Kgg·φ alone, without λ·K_d,gg): the MPC forces from Gm
//     (RigidElimination::computeMpcForces with −c·Bgg·φ as the "load"), the SPC forces the rest on the constrained
//     DOFs, as the static solution computes them from Kgg·u − P;
//   - STRESS, STRAIN, FORCE (output::ElementResultBuilder) and ESE (output::StiffnessRecovery) of the mode shape;
//   - OLOAD and GPFORCE have no table (a mode has no applied load; the grid point force balance of a mode is not
//     produced by this version): the solution reports them once.
// Every table carries the analysis ("modes" or "buckling"), the mode number as its step and the eigenvalue.
//
// Thread-safety: recover() is const and uses the pool given. Ownership: borrows the model and the system.
class ModeRecovery
{
public:
    // `isRecoveredCaseByCase`: elementTables() gives none and recover() recovers the element tables of each mode alone
    // (the reference of the recovery of many modes at once; ModesOptions::isRecoveredCaseByCase).
    ModeRecovery(const AnalysisModel& model, const ConstrainedSystem& system, std::string analysis,
                 std::int64_t recoveryRangeElements = output::ElementResultBuilder::RANGE_ELEMENTS,
                 bool isRecoveredCaseByCase = false);
    // Without the full path's constrained system (the condensed path of an eigenvalue analysis: no Kgg is assembled):
    // the eigenvectors and the element tables only; the SPC and MPC forces are not produced (the solution says so) and
    // expand() is not available.
    ModeRecovery(const AnalysisModel& model, std::string analysis,
                 std::int64_t recoveryRangeElements = output::ElementResultBuilder::RANGE_ELEMENTS,
                 bool isRecoveredCaseByCase = false);

    // The g-set vector of column `column` of the analysis-set vectors (one row per equation of `selection`): the free
    // DOFs from the vector, the constrained DOFs zero, the m set from Gm.
    [[nodiscard]] base::BudgetedBuffer<double> expand(const linalg::DenseMatrix& vectors, std::size_t column,
                                                      const linalg::DofSelection& selection,
                                                      base::MemoryBudget& budget) const;

    // Appends the tables of one mode of `subcase` to `arrays`. second, secondFactor: c·Bgg of the residual (see the
    // class comment; null for none). `strainEnergies`: the mode's ESE tables made beforehand by strainEnergyTables()
    // (moved from), or null to make them here; `elementTables`: its STRESS, STRAIN and FORCE tables made beforehand by
    // elementTables() (moved from), or null to make them here.
    void recover(std::int32_t subcase, const output::OutputPlan& plan, std::span<const double> modeG, std::int32_t step,
                 double eigenvalue, const linalg::BlockSparseMatrix* second, double secondFactor,
                 base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation,
                 output::ResultArrays& arrays, std::vector<output::ElementResultTable>* strainEnergies = nullptr,
                 std::vector<output::ElementResultTable>* elementTables = nullptr) const;

    // The STRESS, STRAIN and FORCE tables of every mode of `modes` (g × modes, the g-set mode shapes) for `subcase`,
    // one entry per mode, for recover() to take; empty when `plan` asks for none or the modes are recovered one by one
    // (isRecoveredCaseByCase).
    // PERFORMANCE: the modes recovered by blocks of MODE_BLOCK from the recovery slopes of the element families
    // (output::ElementResultBuilder::buildCases), the count of the modes deciding the vectorised kinds, where recover()
    // made every element's rows again for each mode; the tables of a mode equal its own recovery's to round-off.
    [[nodiscard]] std::vector<std::vector<output::ElementResultTable>> elementTables(
        std::int32_t subcase, const output::OutputPlan& plan, linalg::ConstDenseMatrixView modes,
        base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation) const;

    // The most modes of one block of elementTables(): its blocks hold the rows of a range for every mode of the block.
    static constexpr std::size_t MODE_BLOCK = 32;

    // The ESE tables of every mode of `modes` (g × modes, the g-set mode shapes) for `subcase`, one entry per mode,
    // for recover() to take; empty when `plan` asks for no strain energy.
    // PERFORMANCE: the modes at once (StiffnessRecovery::strainEnergies on the pool: each element's stiffness once for
    // a group of modes) where recover() computed every element's stiffness again for each mode on one thread; the
    // energies of a mode are the chains of its own recovery, so the same tables to the bit.
    // Measured: 20 modes of a 242 406 DOF plate.
    [[nodiscard]] std::vector<std::vector<output::ElementResultTable>> strainEnergyTables(
        std::int32_t subcase, const output::OutputPlan& plan, linalg::ConstDenseMatrixView modes,
        base::MemoryBudget& budget, base::ThreadPool* pool, const base::CancellationToken* cancellation) const;

private:
    const AnalysisModel* m_model;
    const ConstrainedSystem* m_system;
    std::string m_analysis;
    output::NodalResultBuilder m_nodalBuilder;
    output::ElementResultBuilder m_elementBuilder;
    bool m_isRecoveredCaseByCase = false;
};

} // namespace oos::solutions
