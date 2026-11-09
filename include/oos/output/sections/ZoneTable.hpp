#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/output/sections/ContourReport.hpp"
#include "oos/output/sections/ZoneContour.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// The contour data of a zone on the load basis of one constraint set (a linear plugin works on the load basis: it is
// called with the data of the k basis vectors and the coefficients α, not with every load case): the data of the k
// basis vectors (rows × k, the rows those of ZoneTable) and the coefficients α (k × cases) that give each case of the
// set as Σ_i α(i, c)·basis vector i. The contour forces of the basis hold no thermal part: a set with
// a thermal case says so (`hasThermalCases`), and its cases are not the combination of the basis there.
//
// Thread-safety: immutable once built. Ownership: owns its matrices (budgeted).
struct ZoneBasisSet
{
    std::int32_t constraintSet = 0;  // its index among the constraint sets of the run, in their order
    std::vector<std::int32_t> cases; // the subcases of the set, the columns of α
    linalg::DenseMatrix displacements;
    linalg::DenseMatrix forces;
    linalg::DenseMatrix loads;
    linalg::DenseMatrix coefficients;
    bool hasThermalCases = false;

    [[nodiscard]] std::size_t basisSize() const noexcept { return coefficients.rows(); }
};

// The contour data of one zone in one static analysis (arrays labelled by zone, grid point, component and subcase):
// the contour report, the labels of the rows, the cases in the order they were computed, and, when kept, per case the
// contour displacements and forces (contour DOFs, in the zone's system), the loads of the interior grid points
// (interior DOFs) and the EQUILIBRIUM_ROWS rows of the equilibrium check; with the basis kept (a linear plugin), the
// data on the load basis of each constraint set.
//
// Thread-safety: immutable once built. Ownership: owns its labels and values (the values budgeted: rows × cases).
struct ZoneTable
{
    std::int32_t id = 0;
    std::string name;
    std::int32_t group = 0;
    std::int32_t frameId = 0;
    ContourReport report;
    // Per contour DOF and per interior DOF: the grid point and the component (1 to 6) in the zone's system.
    std::vector<std::int32_t> contourGrids;
    std::vector<std::int32_t> contourComponents;
    std::vector<std::int32_t> interiorGrids;
    std::vector<std::int32_t> interiorComponents;
    std::vector<std::int32_t> cases;
    // Case after case (value of row r in case c at c·rows + r); empty when not kept.
    base::BudgetedBuffer<double> displacements;
    base::BudgetedBuffer<double> forces;
    base::BudgetedBuffer<double> loads;
    base::BudgetedBuffer<double> equilibrium;
    bool hasValues = false;
    std::vector<ZoneBasisSet> basis;

    [[nodiscard]] std::size_t contourDofCount() const noexcept { return contourGrids.size(); }
    [[nodiscard]] std::size_t interiorDofCount() const noexcept { return interiorGrids.size(); }
    // The largest distance of a grid point of the zone from the origin of its system (the arm of the moments).
    double referenceLength = 0.0;

    // The relative residual of the equilibrium of case c (an index into `cases`), forces and moments apart: the
    // magnitude of the sum of the contour and interior resultants over the gross magnitude of what is summed (the
    // larger of the two resultants, Σ|F| of the contour forces and Σ|P| of the interior loads, a moment scale adding
    // referenceLength × those forces): a zone the load passes through has resultants near zero, whose noise is no
    // measure; 0 for an unloaded case.
    [[nodiscard]] double equilibriumResidual(std::size_t c) const;
};

} // namespace oos::output
