#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/solutions/common/EigenMethod.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::solutions
{

// Scales the eigenvectors of a solve (the columns of `vectors`, one row per equation of the analysis set, as the
// block Lanczos solver gives them: M-orthonormal for a vibration problem, K-orthonormal for a buckling problem) by
// the normalisation of the method (field NORM of LanczosEigenMethod, RealEigenMethod, BucklingEigenMethod), and returns
// the factor of each, φ ← s·φ:
//
//   MASS   s = ±1: the vectors keep their unit generalised mass; the sign makes the component of largest magnitude
//          positive (a convention of O_o_Solver: the format fixes no sign; results compare up to sign);
//   MAX    s = 1/φ_k with k the component of largest magnitude in the analysis set (the first on a tie): it becomes
//          +1;
//   POINT  s = 1/φ_k with k the component C of point G: it becomes +1. A point or component outside the analysis set,
//          or a vector without motion there, falls back to MAX with a warning.
//
// The generalised mass of a scaled vibration mode is s², its generalised stiffness s²·λ; for a buckling mode the
// generalised stiffness is s² and the "generalised mass" s²/λ.
//
// Determinism: each column is scaled by its own values in a fixed order.
[[nodiscard]] std::vector<double> scaleEigenvectors(linalg::DenseMatrix& vectors, const EigenMethod& method,
                                                    const assembly::DofMap& dofMap,
                                                    const linalg::DofSelection& selection, base::DiagnosticSink& sink);

// The same scaling for eigenvectors on the g set (one row per DOF of the DOF map), the components compared being those
// flagged in isAnalysisSet (the condensed path: the DOFs of the full path's analysis set whose values are known: the
// interface's and the recovered units' interiors). Every row is scaled.
[[nodiscard]] std::vector<double> scaleEigenvectorsOnSet(linalg::DenseMatrix& vectorsG,
                                                         std::span<const std::uint8_t> isAnalysisSet,
                                                         const EigenMethod& method, const assembly::DofMap& dofMap,
                                                         base::DiagnosticSink& sink);

} // namespace oos::solutions
