#pragma once

#include "oos/assembly/constraints/MsetCouplings.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

namespace oos::assembly
{

// The reduction of the g-set stiffness to the n set by the elimination of the dependent DOFs (linear elimination of
// the multipoint constraints): with u_m = Gm·u_n,
//     Knn = K̄nn + K̄nm·Gm + Gmᵀ·K̄mn + Gmᵀ·K̄mm·Gm,
// computed on the node-block structure: for each m DOF d with its Gm row (c_j on n DOFs j),
//     K(i, j) += K(i, d)·c_j and K(j, i) += K(i, d)·c_j for every n DOF i coupled to d,
//     K(j, k) += c_j·K(d, e)·c_k for every m DOF e coupled to d (e = d included) and every term c_k of e's row,
// the reads from Kgg and the sums into a copy of it. The rows and columns of the m DOFs keep their Kgg values: the
// DofSelection of the f set removes them.
//
// Determinism: one serial pass in a fixed order (the m DOFs increasing, their neighbours by node and component), so the
// bits do not depend on the threads.
class MsetElimination
{
public:
    // Writes Knn into `knn` (overwritten). Preconditions: both matrices on the same pattern, whose graph is `graph` and
    // holds the couplings of MsetCouplings for `gm` (std::invalid_argument otherwise: a term outside the pattern).
    static void reduce(const linalg::BlockSparseMatrix& kgg, const linalg::NodeGraph& graph, const RigidElimination& gm,
                       const DofMap& dofMap, linalg::BlockSparseMatrix& knn);
};

} // namespace oos::assembly
