# `linalg`

Linear algebra written in the project: dense kernels (scalar reference and SIMD), sparse matrices, orderings, symbolic analysis, the supernodal LDLT factorisation, Schur complements, eigenvalues.

It may use `base` only.

Folders: `dense/`, `kernels/`, `sparse/`, `ordering/`, `symbolic/`, `supernodal/`, `eigen/`. Where to start reading:
the folders in that order, which is the order in which the pieces depend on one another.

## Kernel families and their dispatch

- `kernels/reference/`, `kernels/avx2/`, `kernels/avx512/`: one object library each, compiled with its own target flags
  (`cmake/OosKernels.cmake`: `-ffp-contract=off`; `-mavx2 -mfma`; `-mavx512f -mavx512dq -mavx512vl`); the rest
  of the project is compiled for the baseline x86-64. They are the only folders with SIMD intrinsics.
- A SIMD unit defines nothing with external linkage but its entry points, and includes only headers without inline
  functions (`kernels/MatrixProduct.hpp`): an inline function compiled with AVX2 and shared with baseline code could be
  the copy the linker keeps, and fault on a processor without AVX2.
- `dense/KernelSelection` chooses the family once per analysis: the option `kernel=`, then `OOS_KERNEL`, then
  the CPU detection (`base/platform/CpuFeatures`): AVX2 when the processor and the OS support AVX2 and FMA, the
  reference otherwise. AVX-512 is selectable by force only until it is validated on a machine that has it.
- The reference family sums in a fixed order without fused multiply-add: the same bits everywhere. The SIMD families
  compute every element as the same chain of fused multiply-adds in increasing order, whatever the register blocking:
  AVX2 and AVX-512 give the same bits as each other, and as `std::fma` applied in that order.

## Dense kernels

- `dense/DenseMatrix` (column-major, columns 64-byte aligned, budgeted, 64-bit indices), `dense/DenseMatrixView`,
  `dense/SymmetricPacked` (lower triangle packed by columns), `dense/Inertia`.
- `dense/DenseKernels`, the public face of the kernels of the factorisation: the update `C ← C − A·D·Aᵀ` (and its
  rectangular form), the LDLᵀ without pivoting (with the MaxDiagonalRatio ratios, partial factorisation for Schur complements),
  the LDLᵀ with Bunch-Kaufman pivoting (inertia), the panel solve, the triangular solves with many right-hand sides
  and the block-diagonal solve. One object per kernel family and options (pool, threading threshold, tile size,
  cancellation token checked between blocks and tiles, a `dense/KernelWorkspace` keeping the packing buffers between
  calls). The large products run on panels packed once and shared by the tasks, and the LDLᵀ without pivoting
  eliminates a panel's columns recursively (k-blocking), with the bits of the tile-by-tile code.
- `dense/SmallEigenSolver`: Householder tridiagonalisation and implicit QL (EISPACK tred2/tql2) for problems of a few
  hundred.
- Internal: `dense/KernelFamily` (the entry points of a family), `dense/TileRunner` (tiles and threads, and the scratch
  buffers of an operation), `dense/BlockedLdlt` (the blocked factorisations of the SIMD families),
  `dense/LdltArithmetic` (W = L·D, the 2 × 2 pivot formula, the inertia, shared with the families).
- `kernels/DenseKernelEntryPoints.hpp`: the operands and entry points of the families (no inline function: it is
  included by SIMD units). `kernels/reference/DenseBlockKernels.cpp` holds the unblocked textbook algorithms (the
  oracle); `kernels/SimdDenseKernels.ipp` holds the packed, register-blocked algorithms, written once and included by
  `kernels/avx2/` and `kernels/avx512/` with their vector operations.
- The arithmetic contract: every result element is one chain of terms in a fixed order, so the bits do not depend on
  tiles or threads, and AVX2 and AVX-512 agree.

## Sparse matrices and assembly

- `ordering/NodeGroupList` (a view of groups of nodes: element connectivity, absorbed spiders, internal constraints) and
  `ordering/NodeGraph` (the node adjacency, compressed rows with 64-bit offsets, sorted, symmetric, built in parallel by
  pieces of nodes; `inducedSubgraph` drops the nodes whose DOFs are all eliminated).
- `sparse/SparsePattern`: the lower triangle of the node blocks (6 × 6 for a grid point, 1 × 1 for a scalar point),
  built once from the node graph and shared by every matrix on it; `sparse/BlockSparseMatrix`: the values, diagonal
  blocks complete; its product with a block of vectors shares groups of columns between the threads, each column with
  the bits of its own product.
- `sparse/ContributionBlock` and `NodeDof`: the neutral form in which the assembly module hands over element matrices
  (DOFs as node and component from 0, dense symmetric matrix, row-major); linalg includes no element header.
- `sparse/SparseAssembler`: element matrices formed in parallel by pieces of consecutive elements, then accumulated by
  ownership of block columns (chunks of 64 nodes dealt round robin to the threads): every term is summed by one thread
  in element order, so the bits are those of a serial loop whatever the number of threads, piece or wave size. No list
  of triplets, no atomics.
- `sparse/DofSelection` (the eliminated DOFs removed, the others numbered node by node in a given node order) and
  `sparse/SymmetricCsc` (the scalar lower triangle, `int32` rows and `int64` offsets, structural pattern: the stored
  zeros of a block are kept so that the pattern depends on the graph and the selection only). `fromBlocks` gathers the
  terms by the node that owns their columns on the pool (the kept DOFs of a node are consecutive equations), each
  column sorted by row: the serial conversion's result. `DenseMatrix` and `BlockSparseMatrix` take a pool to zero their
  values on it (`base::fillOnPool`). A small dense product over a long inner dimension is cut into tiles down to the
  granule (`LONG_PRODUCT_TILES`): every element one chain whatever its tile.

## Orderings and symbolic analysis

- `ordering/NodePermutation` (an elimination order and its inverse), `ordering/ApproximateMinimumDegree` (Amestoy,
  Davis and Duff 1996, ties by index through an indexed heap, a trailing set kept last),
  `ordering/GeometricDissection` (coordinate bisection, separator from the edge cut, leaves by the constrained minimum
  degree, halves in parallel), `ordering/ConstrainedOrdering` (the rule that chooses the method of a unit, the trailing
  set last), `ordering/QuotientGraph` (one vertex per unit). Internal: `ordering/TrailingSet`.
- `symbolic/EliminationTree` (Liu, postorder), `symbolic/ColumnCounts` (Gilbert, Ng and Peyton, weighted),
  `symbolic/Supernodes` (fundamental supernodes, the trailing block last), `symbolic/Amalgamation` (relaxed
  supernodes, CHOLMOD's rule in nodes of 6 DOFs), `symbolic/SymbolicFactor` (final order, supernode rows, panel sizes,
  flops, update stack, memory estimate: the storage model the numeric factorisation relies on).
- The long steps (`NodeGraph`, the orderings, `SymbolicFactor::analyse`, `SymmetricCsc::fromBlocks`) take a
  `base::OperationControl`: cancellation token and progress counter, each stating its progress unit.

## Supernodal factorisation and solves

- `supernodal/SupernodalFactor`: the numeric LDLᵀ by the multifrontal method: positive definite mode without pivoting
  (the MaxDiagonalRatio test), indefinite mode with Bunch-Kaufman inside each front and delayed pivots passed to the parent;
  inertia; panels in the shape of the symbolic factor (exactly `panelValueCount()` values without delays); independent
  subtrees in parallel and the kernels' threads in the top fronts, the same bits for any number of threads; everything
  reserved in the budget before the first column is eliminated (the positive definite mode's working storage bounded
  without front buffers). `supernodal/FactorStorage`: the panels' memory a factor gives back (`releaseStorage`) for
  the next factorisation of the same structure (`FactorOptions::storage`).
- `supernodal/FactorSolve`: forward, diagonal and backward sweeps over the panels, each column with the same bits
  whatever its tile. Tiles of 128 columns stored by rows and swept by the row kernels of `DenseKernels`;
  `supernodal/SolvePlan` (built with the factor) groups the supernodes into units whose rows are gathered once per
  sweep; the rows of the large top supernodes shared between the threads, two tiles in flight; the forward sweep of
  sparse right-hand sides limited to the reach of their nonzero rows; no −0 written. Without delayed pivots, the
  forward and backward sweeps also run the independent subtrees of the factorisation in parallel
  (`SupernodalFactor::subtrees`, the cut of `FactorSchedule`): forward, each subtree's contributions to the rows of its
  ancestors accumulated apart and added in subtree order (the same bits whatever the threads); backward, the top first,
  then the subtrees (the bits of the sweep in supernode order).
- `supernodal/PivotDiagnostics`: `PivotPolicy` (MaxDiagonalRatio, SingularityBailOut), the failing pivots with their equations, named for
  the user by an `EquationNamer` of the caller.
- Internal: `supernodal/MultifrontalDriver` (the numeric loop, the tasks, the checks, the admission of subtrees beyond
  the plan), `supernodal/WorkBuffer` (a buffer of the working storage and the `WorkspaceMeter` that measures it),
  `supernodal/WorkspacePool` (the large blocks of the working storage kept for reuse within its bound, by size
  classes), `supernodal/FrontAssembly` (the extend-add of a front, whole or in two parts, by runs of positions, a large
  top front by ranges of its columns on the threads), `supernodal/FactorSchedule` (children, the cut into subtrees and
  top supernodes, the bound of the working storage).
- Dense additions: the candidate form of Bunch-Kaufman (`DenseKernels::factorWithPivoting(a, candidates)`, delayed
  candidates), the products of the solves (`subtractProduct`, `subtractTransposedProduct`), and the LDLᵀ of a front in
  two parts (`factorFrontWithoutPivoting`: the panel and the trailing block, the bits of the whole front).

- `dense/IterativeRefinement`: the stopping rule of a fixed-precision iterative refinement (LAPACK xPORFS: a step while
  the correction is above 1e-8 of the displacement and halved, a larger one not applied, at most four), and the largest
  relative correction of a block of vectors; the relieved solution of a free body is refined with it on the full path
  and on the full path streamed.

- `dense/FullRankCertificate`: a lower bound of the smallest singular value of a tall matrix, 1/‖R⁻¹‖_F of its
  unpivoted blocked Householder QR (compact WY panels updated by the products of the dense kernels), stopping at the
  first diagonal of R below the bound; the rank test of `assembly::LoadBasis` uses it to avoid a column-pivoted QR of
  thousands of independent load cases.

## Partial factorisation

- `supernodal/PartialFactor`: the interior of a unit eliminated, the trailing set left as the exact Schur complement
  K* (a budgeted `DenseMatrix`, both triangles), the condensed loads f* = f_b − K_bi·K_ii⁻¹·f_i and the recovery
  u_i = K_ii⁻¹·(f_i − K_ib·u_b) by the sweeps of `FactorSolve`; the interior factor kept, released and recomputed
  with the same bits (never stored on disk); a recomputation refuses another kernel family, pivot policy or matrix (a
  checksum of its structure and values).
- Without the interior factor: `PartialFactor::condenseWithoutFactor` (K*, the forward-eliminated loads and the pivots
  while each panel is used once and released) and `recoverWithoutFactor` (the backward sweep by regions of the tree,
  each factorised again), through the internal `supernodal/StreamedFactorisation` (the multifrontal driver's positive
  definite fronts, each panel lent to a skeleton factor while resident), `supernodal/ForwardStream` (FactorSolve's
  forward sweep unit by unit) and `supernodal/SolveSweeps` (the sweep steps FactorSolve and they share): the bits of
  the kept factor's.
- `supernodal/StreamedSolve`: the complete solve of a positive definite matrix without its factor (the streamed
  factorisation with the forward sweep of every column, the backward sweep by regions of the tree, each factorised
  again: the bits of `FactorSolve::solve`), its estimate and the plan of its regions (`RegionSweepCost`) from the
  symbolic factor alone; the regions shared with `recoverWithoutFactor` through the internal `supernodal/RegionSweep`.

## Eigenvalues

- `eigen/ShiftedPencil` (the interface: factorise K − σM, solve, multiply by M, the spectrum scale; the caller
  provides it, linalg stays below assembly) and `eigen/SparsePencil` (two `SymmetricCsc` of one pattern, K − σM by
  `SymmetricCsc::combine`, the supernodal factor in indefinite mode, a lumped mass multiplied by its diagonal).
- `eigen/EigenRequest` (range, mode count), `eigen/SturmCheck` (Sturm points and the walk that confirms no root is
  missed), `eigen/ShiftStrategy` (where the shifts go, when the request is met), `eigen/BlockLanczos` (the driver:
  shifts, runs, locking, the M-orthonormal result).
- Internal: `eigen/LanczosRun` (block shift-invert Lanczos at one shift: thick restart, full reorthogonalisation,
  purification), `eigen/MassOrthogonalizer` (block Gram-Schmidt in the M-inner product, its inner products by fixed
  pieces of rows on the threads), `eigen/EigenvectorStore` (the pairs found, locked out of every later basis).
- Sparse additions: `SymmetricCsc::combine`, `hasSameStructure` and the block product `multiply(X, Y, pool)`.

## Buckling eigenproblem

- `eigen/ShiftedPencil::transform()` (`SpectralTransform::Vibration` or `Buckling`) and `multiplyInnerProduct` (M, or
  K for buckling): the run, the store, the orthogonaliser and the restart use them unchanged.
- `eigen/BucklingPencil`: K and K_d of one pattern, K − σB = K + σK_d (B = −K_d), Op = (K − σB)⁻¹·B in the K-inner
  product, θ = 1/(λ − σ) as for vibration.
- `BlockLanczos::solveBuckling` (`BlockLanczosBuckling.cpp`): load factors of both signs; each sign a side with its
  own Sturm walk and `ShiftStrategy` in x = |λ| (K − σB counts the roots between 0 and σ); K checked positive definite
  at σ = 0, the first run at σ = 0; the roots by increasing magnitude, K-orthonormal. Internal `SolveSupport.hpp`
  holds what both solves share.
