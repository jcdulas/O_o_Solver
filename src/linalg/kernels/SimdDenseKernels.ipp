// The algorithms of the SIMD dense kernels, written once for every vector width. This file is not compiled on its
// own: kernels/avx2/DenseBlockKernels.cpp and kernels/avx512/DenseBlockKernels.cpp include it inside their family
// namespace and an unnamed namespace (so that everything here has internal linkage and nothing compiled with SIMD
// flags is shared with baseline code), after defining `struct Simd`, the vector operations of the family:
//
//   using Vector;                                      a register of LANES doubles
//   static constexpr std::size_t LANES;                doubles per register
//   static constexpr std::size_t ROW_VECTORS;          registers of rows in a micro-tile (rows = LANES·ROW_VECTORS)
//   static constexpr std::size_t COLUMNS;              columns in a micro-tile
//   static constexpr std::size_t ROW_TARGETS;          target rows of a micro-tile of the row products
//   static constexpr std::size_t ROW_STRIP_VECTORS;    registers of columns of that micro-tile
//   static Vector load(const double*);                 unaligned load
//   static void store(double*, Vector);                unaligned store
//   static Vector broadcast(double);
//   static Vector subtractProduct(Vector a, Vector b, Vector c);   c − a·b, one fused rounding
//   static Vector multiply(Vector, Vector); subtract(Vector, Vector); divide(Vector, Vector);   each rounded
//   static double subtractProduct(double a, double b, double c);   c − a·b, one fused rounding (scalar)
//   static void prefetch(const double*);               a hint: the line brought to L1 (no value changes)
//
// Here only the Simd operations touch vectors: no intrinsic appears in this file (SIMD intrinsics only
// in kernels/avx2 and kernels/avx512). The families are compiled with -ffp-contract=off, so the only fused operations
// are the explicit subtractProduct: the scalar remainders and the vector bodies give the same bits, and the products
// of W = B·D, the divisions and the 2 × 2 pivot formula are rounded exactly as in the reference family.
//
// The arithmetic contract (kernels/DenseKernelEntryPoints.hpp): every result element is one chain
// acc = subtractProduct(x_p, y_p, acc) over its terms in the order the operands fix, starting from the element itself;
// the blocking (micro-tiles, inner blocks of KERNEL_INNER_BLOCK terms, tiles, threads) only decides when a chain is
// stored and reloaded, never its order.

constexpr std::size_t MICRO_ROWS = Simd::LANES * Simd::ROW_VECTORS;
constexpr std::size_t MICRO_COLUMNS = Simd::COLUMNS;
static_assert(KERNEL_TILE_GRANULE % MICRO_ROWS == 0 && KERNEL_TILE_GRANULE % MICRO_COLUMNS == 0,
              "a tile must hold whole micro-tiles");

using Vector = typename Simd::Vector;

std::size_t smaller(std::size_t left, std::size_t right)
{
    return left < right ? left : right;
}

// ---------------------------------------------------------------------------------------------------------------
// Packing

std::size_t termIndex(const ProductTileOperands& operands, std::size_t term)
{
    return operands.isInnerReversed ? operands.innerCount - 1 - term : term;
}

double readB(const ProductTileOperands& operands, std::size_t j, std::size_t p)
{
    return operands.isBTransposed ? operands.b[p + j * operands.ldb] : operands.b[j + p * operands.ldb];
}

// Ŵ(j, p) = (B̂·D)(j, p), the formula of ProductTileOperands (the same expressions as the reference family).
double readW(const ProductTileOperands& operands, std::size_t j, std::size_t p)
{
    const BlockDiagonalView& d = operands.d;
    if (d.diagonal == nullptr)
        return readB(operands, j, p);
    const double dp = d.diagonal[p * d.diagonalStride];
    if (d.subdiagonal != nullptr)
    {
        if (p + 1 < d.size && d.subdiagonal[p] != 0.0)
            return readB(operands, j, p) * dp + readB(operands, j, p + 1) * d.subdiagonal[p];
        if (p > 0 && d.subdiagonal[p - 1] != 0.0)
            return readB(operands, j, p - 1) * d.subdiagonal[p - 1] + readB(operands, j, p) * dp;
    }
    return readB(operands, j, p) * dp;
}

// PERFORMANCE: packing. The rows of Â for the terms [firstTerm, firstTerm + termCount) are copied into micro-panels of
// MICRO_ROWS rows, term after term (zero rows pad the last panel), so that the micro-kernel reads A with contiguous
// unit-stride loads from L2 whatever the leading dimension or transposition of A; the copy costs O(rows·terms) against
// O(rows·columns·terms) multiply-adds.
// Measured: see packW.
void packA(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount, double* packed)
{
    for (std::size_t firstRow = 0; firstRow < operands.rowCount; firstRow += MICRO_ROWS)
    {
        const std::size_t rows = smaller(MICRO_ROWS, operands.rowCount - firstRow);
        double* panel = packed + firstRow * termCount;
        for (std::size_t t = 0; t < termCount; ++t)
        {
            const std::size_t p = termIndex(operands, firstTerm + t);
            double* target = panel + t * MICRO_ROWS;
            if (!operands.isATransposed)
            {
                const double* source = operands.a + firstRow + p * operands.lda;
                for (std::size_t r = 0; r < rows; ++r)
                    target[r] = source[r];
            }
            else
            {
                for (std::size_t r = 0; r < rows; ++r)
                    target[r] = operands.a[p + (firstRow + r) * operands.lda];
            }
            for (std::size_t r = rows; r < MICRO_ROWS; ++r)
                target[r] = 0.0;
        }
    }
}

// PERFORMANCE: packing of Ŵ = B̂·D into micro-panels of MICRO_COLUMNS columns, term after term, D applied once per
// element here rather than once per multiply-add; the micro-kernel broadcasts its values from L1.
// Measured (update, i5-12400F, one thread): 47-56 GFLOP/s at n = 1 000-2 000 with packed panels,
// against 23-30 GFLOP/s for an earlier unpacked 12 × 4 kernel (n = 64-512), which fell as its operands left L1.
void packW(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount, double* packed)
{
    const bool isPlainCopy = operands.d.diagonal == nullptr && !operands.isBTransposed;
    const bool isDiagonalScaling =
        operands.d.diagonal != nullptr && operands.d.subdiagonal == nullptr && !operands.isBTransposed;
    for (std::size_t firstColumn = 0; firstColumn < operands.columnCount; firstColumn += MICRO_COLUMNS)
    {
        const std::size_t columns = smaller(MICRO_COLUMNS, operands.columnCount - firstColumn);
        double* panel = packed + firstColumn * termCount;
        for (std::size_t t = 0; t < termCount; ++t)
        {
            const std::size_t p = termIndex(operands, firstTerm + t);
            double* target = panel + t * MICRO_COLUMNS;
            if (isPlainCopy)
            {
                const double* source = operands.b + firstColumn + p * operands.ldb;
                for (std::size_t c = 0; c < columns; ++c)
                    target[c] = source[c];
            }
            else if (isDiagonalScaling)
            {
                const double* source = operands.b + firstColumn + p * operands.ldb;
                const double dp = operands.d.diagonal[p * operands.d.diagonalStride];
                for (std::size_t c = 0; c < columns; ++c)
                    target[c] = source[c] * dp;
            }
            else
            {
                for (std::size_t c = 0; c < columns; ++c)
                    target[c] = readW(operands, firstColumn + c, p);
            }
            for (std::size_t c = columns; c < MICRO_COLUMNS; ++c)
                target[c] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The micro-kernel

// PERFORMANCE: register blocking. A MICRO_ROWS × MICRO_COLUMNS block of C stays in registers while termCount terms are
// swept (AVX2: 3 × 4 = 12 accumulators, 3 loads of A and 1 broadcast of W: 16 YMM registers; AVX-512: 3 × 8 = 24 of
// the 32 ZMM registers), so that each vector of A loaded feeds MICRO_COLUMNS fused multiply-subtracts and each
// broadcast feeds ROW_VECTORS of them: more independent accumulators than the FMA latency × throughput (4 × 2 = 8 on
// Golden Cove) keep both FMA ports busy. The accumulators start from C and are stored back to C: the chain of each
// element continues across the inner blocks.
// Measured (update): 67-80 % of the one-core peak above n = 1 000.
//
// PERFORMANCE: the next micro-tile's elements of C (nextC, down the same columns or the top of the next ones) are
// prefetched while this one sweeps its terms, so that its first loads do not wait on L2 or L3 (C is a tile of a large
// front, not in L1). A hint only: no value changes.
// Measured (the updates of 2 000 and 3 000 on one thread, a scratch harness on the library): 1 to 3 %.
void subtractMicroTile(const double* packedA, const double* packedW, std::size_t termCount, double* c, std::size_t ldc,
                       const double* nextC = nullptr)
{
    Vector accumulators[Simd::ROW_VECTORS][MICRO_COLUMNS];
    for (std::size_t column = 0; column < MICRO_COLUMNS; ++column)
        for (std::size_t r = 0; r < Simd::ROW_VECTORS; ++r)
            accumulators[r][column] = Simd::load(c + column * ldc + r * Simd::LANES);
    if (nextC != nullptr)
        for (std::size_t column = 0; column < MICRO_COLUMNS; ++column)
        {
            Simd::prefetch(nextC + column * ldc);
            Simd::prefetch(nextC + column * ldc + MICRO_ROWS - 1);
        }

    for (std::size_t t = 0; t < termCount; ++t)
    {
        const double* aTerm = packedA + t * MICRO_ROWS;
        const double* wTerm = packedW + t * MICRO_COLUMNS;
        Vector aValues[Simd::ROW_VECTORS];
        for (std::size_t r = 0; r < Simd::ROW_VECTORS; ++r)
            aValues[r] = Simd::load(aTerm + r * Simd::LANES);
        for (std::size_t column = 0; column < MICRO_COLUMNS; ++column)
        {
            const Vector w = Simd::broadcast(wTerm[column]);
            for (std::size_t r = 0; r < Simd::ROW_VECTORS; ++r)
                accumulators[r][column] = Simd::subtractProduct(aValues[r], w, accumulators[r][column]);
        }
    }

    for (std::size_t column = 0; column < MICRO_COLUMNS; ++column)
        for (std::size_t r = 0; r < Simd::ROW_VECTORS; ++r)
            Simd::store(c + column * ldc + r * Simd::LANES, accumulators[r][column]);
}

// A micro-tile cut by the edge of the tile or by the diagonal: computed whole in a local copy (zeros beyond the edge),
// then only its valid elements are written back: rows < rows, columns < columns and, when isLowerOnly, r + offset >= c.
// The chain of each written element is the one of a full micro-tile.
void subtractPartialMicroTile(const double* packedA, const double* packedW, std::size_t termCount, double* c,
                              std::size_t ldc, std::size_t rows, std::size_t columns, bool isLowerOnly,
                              std::ptrdiff_t offset)
{
    double local[MICRO_ROWS * MICRO_COLUMNS];
    for (std::size_t column = 0; column < MICRO_COLUMNS; ++column)
        for (std::size_t r = 0; r < MICRO_ROWS; ++r)
            local[r + column * MICRO_ROWS] = (r < rows && column < columns) ? c[r + column * ldc] : 0.0;
    subtractMicroTile(packedA, packedW, termCount, local, MICRO_ROWS);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t r = 0; r < rows; ++r)
            if (!isLowerOnly || static_cast<std::ptrdiff_t>(r) + offset >= static_cast<std::ptrdiff_t>(column))
                c[r + column * ldc] = local[r + column * MICRO_ROWS];
}

// The terms of one block (termCount of them) of C ← C − Â·Ŵᵀ for the operands' rows and columns, from their packed
// panels. Loop order (Goto & van de Geijn 2008, "Anatomy of high-performance matrix multiplication"): for each
// micro-panel of W (it stays in L1) sweep the micro-panels of A (they stream from L2).
void subtractPackedMicroTiles(const ProductTileOperands& operands, std::size_t termCount, const double* packedA,
                              const double* packedW)
{
    for (std::size_t firstColumn = 0; firstColumn < operands.columnCount; firstColumn += MICRO_COLUMNS)
    {
        const std::size_t columns = smaller(MICRO_COLUMNS, operands.columnCount - firstColumn);
        const double* wPanel = packedW + firstColumn * termCount;
        for (std::size_t firstRow = 0; firstRow < operands.rowCount; firstRow += MICRO_ROWS)
        {
            const std::size_t rows = smaller(MICRO_ROWS, operands.rowCount - firstRow);
            // Element (r, c) of the micro-tile is in the lower triangle when r + offset >= c.
            const std::ptrdiff_t offset = operands.diagonalOffset + static_cast<std::ptrdiff_t>(firstRow) -
                                          static_cast<std::ptrdiff_t>(firstColumn);
            if (operands.isLowerOnly && static_cast<std::ptrdiff_t>(rows) - 1 + offset < 0)
                continue; // wholly above the diagonal
            const bool isWholeTriangle =
                !operands.isLowerOnly || offset >= static_cast<std::ptrdiff_t>(MICRO_COLUMNS) - 1;
            const double* aPanel = packedA + firstRow * termCount;
            double* cBlock = operands.c + firstRow + firstColumn * operands.ldc;
            // The next micro-tile's elements of C: down the same columns, or the top of the next ones.
            const double* nextC = firstRow + MICRO_ROWS < operands.rowCount ? cBlock + MICRO_ROWS
                                  : firstColumn + MICRO_COLUMNS < operands.columnCount
                                      ? operands.c + (firstColumn + MICRO_COLUMNS) * operands.ldc
                                      : nullptr;
            if (rows == MICRO_ROWS && columns == MICRO_COLUMNS && isWholeTriangle)
                subtractMicroTile(aPanel, wPanel, termCount, cBlock, operands.ldc, nextC);
            else
                subtractPartialMicroTile(aPanel, wPanel, termCount, cBlock, operands.ldc, rows, columns,
                                         operands.isLowerOnly, offset);
        }
    }
}

// PERFORMANCE: the terms in ⌈k / KERNEL_INNER_BLOCK⌉ blocks of equal length (the last no longer), not in blocks of
// KERNEL_INNER_BLOCK and a remainder: each block loads and stores every element of C once, so a remainder of a few
// terms (k = 288 = 256 + 32, as the recursive LDLᵀ makes) cost a whole pass over C for little work. Where the blocks
// end decides only when a chain is stored and reloaded. The same rule as DenseKernels.cpp's innerBlockTerms.
// Measured (one thread, interleaved runs): the update of 600 with 288 terms 44-49 →
// 49-57 GFLOP/s, of 1 200 52 → 52-56; the LDLᵀ and the fronts within the noise.
std::size_t innerBlockTerms(std::size_t innerCount)
{
    const std::size_t blocks = (innerCount + KERNEL_INNER_BLOCK - 1) / KERNEL_INNER_BLOCK;
    return blocks == 0 ? 0 : (innerCount + blocks - 1) / blocks;
}

// One tile of C ← C − Â·Ŵᵀ (ProductTileOperands): for each block of terms, pack W and A of the tile, then its
// micro-tiles.
void subtractTile(const ProductTileOperands& operands, const PackingWorkspace& workspace)
{
    const std::size_t blockTerms = innerBlockTerms(operands.innerCount);
    for (std::size_t firstTerm = 0; firstTerm < operands.innerCount; firstTerm += blockTerms)
    {
        const std::size_t termCount = smaller(blockTerms, operands.innerCount - firstTerm);
        packW(operands, firstTerm, termCount, workspace.packedW);
        packA(operands, firstTerm, termCount, workspace.packedA);
        subtractPackedMicroTiles(operands, termCount, workspace.packedA, workspace.packedW);
    }
}

// The rows of the operands for one block of terms against W packed beforehand (packProductColumns): A packed here,
// then the micro-tiles; each element's chain continues from what C holds.
void subtractPackedProductRows(const ProductTileOperands& operands, std::size_t firstTerm, std::size_t termCount,
                               const double* packedW, double* packedA)
{
    packA(operands, firstTerm, termCount, packedA);
    subtractPackedMicroTiles(operands, termCount, packedA, packedW);
}

// ---------------------------------------------------------------------------------------------------------------
// Column products: y ← y − A·v

template <std::size_t VectorCount>
void subtractColumnProductsStrip(const ColumnProductOperands& operands, std::size_t firstRow)
{
    Vector accumulators[VectorCount];
    for (std::size_t r = 0; r < VectorCount; ++r)
        accumulators[r] = Simd::load(operands.y + firstRow + r * Simd::LANES);
    for (std::size_t p = 0; p < operands.innerCount; ++p)
    {
        const Vector v = Simd::broadcast(operands.v[p]);
        const double* aColumn = operands.a + firstRow + p * operands.lda;
        for (std::size_t r = 0; r < VectorCount; ++r)
            accumulators[r] = Simd::subtractProduct(Simd::load(aColumn + r * Simd::LANES), v, accumulators[r]);
    }
    for (std::size_t r = 0; r < VectorCount; ++r)
        Simd::store(operands.y + firstRow + r * Simd::LANES, accumulators[r]);
}

void subtractColumnProductsAll(const ColumnProductOperands& operands)
{
    constexpr std::size_t STRIP_VECTORS = 4;
    std::size_t row = 0;
    for (; row + STRIP_VECTORS * Simd::LANES <= operands.rowCount; row += STRIP_VECTORS * Simd::LANES)
        subtractColumnProductsStrip<STRIP_VECTORS>(operands, row);
    for (; row + Simd::LANES <= operands.rowCount; row += Simd::LANES)
        subtractColumnProductsStrip<1>(operands, row);
    for (; row < operands.rowCount; ++row)
    {
        double sum = operands.y[row];
        for (std::size_t p = 0; p < operands.innerCount; ++p)
            sum = Simd::subtractProduct(operands.a[row + p * operands.lda], operands.v[p], sum);
        operands.y[row] = sum;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Panel solve: P ← P·L⁻ᵀ·D⁻¹

bool startsTwoByTwo(const BlockDiagonalView& d, std::size_t p)
{
    return d.subdiagonal != nullptr && p + 1 < d.size && d.subdiagonal[p] != 0.0;
}

// The coefficients of the inverse of a 2 × 2 pivot (LAPACK dsytf2; the reference family has the same expressions).
struct TwoByTwoInverse
{
    double d11;
    double d22;
    double s;
};

TwoByTwoInverse invertTwoByTwo(double a, double b, double c)
{
    const double d11 = c / b;
    const double d22 = a / b;
    const double t = 1.0 / (d11 * d22 - 1.0);
    return {d11, d22, t / b};
}

// The panel solve on VectorCount·LANES rows starting at firstRow (VectorCount = 0: the single row firstRow, scalar).
template <std::size_t VectorCount>
void solvePanelRows(const PanelSolveOperands& operands, std::size_t firstRow)
{
    constexpr std::size_t REGISTERS = VectorCount == 0 ? 1 : VectorCount;
    double* rows = operands.p + firstRow;
    const auto column = [&](std::size_t j) { return rows + j * operands.ldp; };
    const auto w = [&](std::size_t j, std::size_t q) { return operands.w[j + q * operands.ldw]; };

    const BlockDiagonalView& d = operands.d;
    std::size_t j = 0;
    while (j < operands.columnCount)
    {
        const bool isPair = startsTwoByTwo(d, j);
        if constexpr (VectorCount == 0)
        {
            double r0 = column(j)[0];
            double r1 = isPair ? column(j + 1)[0] : 0.0;
            for (std::size_t q = 0; q < j; ++q)
            {
                const double solved = column(q)[0];
                r0 = Simd::subtractProduct(solved, w(j, q), r0);
                if (isPair)
                    r1 = Simd::subtractProduct(solved, w(j + 1, q), r1);
            }
            if (isPair)
            {
                const TwoByTwoInverse inverse = invertTwoByTwo(d.diagonal[j * d.diagonalStride], d.subdiagonal[j],
                                                               d.diagonal[(j + 1) * d.diagonalStride]);
                column(j)[0] = inverse.s * (inverse.d11 * r0 - r1);
                column(j + 1)[0] = inverse.s * (inverse.d22 * r1 - r0);
            }
            else
            {
                column(j)[0] = r0 / d.diagonal[j * d.diagonalStride];
            }
        }
        else
        {
            Vector r0[REGISTERS];
            Vector r1[REGISTERS];
            for (std::size_t r = 0; r < VectorCount; ++r)
            {
                r0[r] = Simd::load(column(j) + r * Simd::LANES);
                r1[r] = isPair ? Simd::load(column(j + 1) + r * Simd::LANES) : r0[r];
            }
            for (std::size_t q = 0; q < j; ++q)
            {
                const Vector w0 = Simd::broadcast(w(j, q));
                const double* solved = column(q);
                for (std::size_t r = 0; r < VectorCount; ++r)
                    r0[r] = Simd::subtractProduct(Simd::load(solved + r * Simd::LANES), w0, r0[r]);
                if (isPair)
                {
                    const Vector w1 = Simd::broadcast(w(j + 1, q));
                    for (std::size_t r = 0; r < VectorCount; ++r)
                        r1[r] = Simd::subtractProduct(Simd::load(solved + r * Simd::LANES), w1, r1[r]);
                }
            }
            if (isPair)
            {
                const TwoByTwoInverse inverse = invertTwoByTwo(d.diagonal[j * d.diagonalStride], d.subdiagonal[j],
                                                               d.diagonal[(j + 1) * d.diagonalStride]);
                const Vector s = Simd::broadcast(inverse.s);
                const Vector d11 = Simd::broadcast(inverse.d11);
                const Vector d22 = Simd::broadcast(inverse.d22);
                for (std::size_t r = 0; r < VectorCount; ++r)
                {
                    Simd::store(column(j) + r * Simd::LANES,
                                Simd::multiply(s, Simd::subtract(Simd::multiply(d11, r0[r]), r1[r])));
                    Simd::store(column(j + 1) + r * Simd::LANES,
                                Simd::multiply(s, Simd::subtract(Simd::multiply(d22, r1[r]), r0[r])));
                }
            }
            else
            {
                const Vector pivot = Simd::broadcast(d.diagonal[j * d.diagonalStride]);
                for (std::size_t r = 0; r < VectorCount; ++r)
                    Simd::store(column(j) + r * Simd::LANES, Simd::divide(r0[r], pivot));
            }
        }
        j += isPair ? 2 : 1;
    }
}

// PERFORMANCE: register blocking of the panel solve. A strip of MICRO_ROWS rows is solved MICRO_COLUMNS columns at a
// time: first the terms of the columns before the group, for the whole group at once (the micro-tile pattern: each
// vector of P loaded feeds MICRO_COLUMNS fused operations instead of one), then the terms inside the group, column by
// column, from the solved columns still in registers. Each element still receives its terms in increasing q, so the
// bits are those of solvePanelRows. A group never cuts a 2 × 2 block (it stops one column short instead).
// Measured (panel, 128 columns, m = 1 000-2 000, one thread): 33-48 GFLOP/s, against 20-21 GFLOP/s
// column by column.
void solvePanelStripGrouped(const PanelSolveOperands& operands, std::size_t firstRow)
{
    constexpr std::size_t ROWS = Simd::ROW_VECTORS;
    constexpr std::size_t GROUP = MICRO_COLUMNS;
    double* rows = operands.p + firstRow;
    const BlockDiagonalView& d = operands.d;
    const auto column = [&](std::size_t j) { return rows + j * operands.ldp; };
    const auto w = [&](std::size_t j, std::size_t q) { return operands.w[j + q * operands.ldw]; };

    std::size_t first = 0;
    while (first < operands.columnCount)
    {
        std::size_t width = smaller(GROUP, operands.columnCount - first);
        if (width > 1 && startsTwoByTwo(d, first + width - 1))
            --width;
        Vector accumulators[GROUP][ROWS];
        for (std::size_t c = 0; c < GROUP; ++c)
            if (c < width)
                for (std::size_t r = 0; r < ROWS; ++r)
                    accumulators[c][r] = Simd::load(column(first + c) + r * Simd::LANES);
        // The pairs of the group: column first + c starts a 2 × 2 block when isPairStart[c].
        bool isPairStart[GROUP];
        bool isPairEnd[GROUP];
        for (std::size_t c = 0; c < GROUP; ++c)
        {
            isPairStart[c] = c < width && startsTwoByTwo(d, first + c);
            isPairEnd[c] = c > 0 && isPairStart[c - 1];
        }

        // The columns before the group.
        for (std::size_t q = 0; q < first; ++q)
        {
            const double* solved = column(q);
            Vector values[ROWS];
            for (std::size_t r = 0; r < ROWS; ++r)
                values[r] = Simd::load(solved + r * Simd::LANES);
            for (std::size_t c = 0; c < GROUP; ++c)
            {
                if (c >= width)
                    continue;
                const Vector wValue = Simd::broadcast(w(first + c, q));
                for (std::size_t r = 0; r < ROWS; ++r)
                    accumulators[c][r] = Simd::subtractProduct(values[r], wValue, accumulators[c][r]);
            }
        }

        // The columns inside the group, in order; a 2 × 2 block (c, c+1) sums over the columns before c for both.
        for (std::size_t c = 0; c < GROUP; ++c)
        {
            if (c >= width || isPairEnd[c])
                continue;
            const std::size_t j = first + c;
            for (std::size_t q = 0; q < c; ++q)
            {
                const Vector w0 = Simd::broadcast(w(j, first + q));
                for (std::size_t r = 0; r < ROWS; ++r)
                    accumulators[c][r] = Simd::subtractProduct(accumulators[q][r], w0, accumulators[c][r]);
                if (isPairStart[c])
                {
                    const Vector w1 = Simd::broadcast(w(j + 1, first + q));
                    for (std::size_t r = 0; r < ROWS; ++r)
                        accumulators[c + 1][r] = Simd::subtractProduct(accumulators[q][r], w1, accumulators[c + 1][r]);
                }
            }
            if (isPairStart[c])
            {
                const TwoByTwoInverse inverse = invertTwoByTwo(d.diagonal[j * d.diagonalStride], d.subdiagonal[j],
                                                               d.diagonal[(j + 1) * d.diagonalStride]);
                const Vector s = Simd::broadcast(inverse.s);
                const Vector d11 = Simd::broadcast(inverse.d11);
                const Vector d22 = Simd::broadcast(inverse.d22);
                for (std::size_t r = 0; r < ROWS; ++r)
                {
                    const Vector r0 = accumulators[c][r];
                    const Vector r1 = accumulators[c + 1][r];
                    accumulators[c][r] = Simd::multiply(s, Simd::subtract(Simd::multiply(d11, r0), r1));
                    accumulators[c + 1][r] = Simd::multiply(s, Simd::subtract(Simd::multiply(d22, r1), r0));
                }
            }
            else
            {
                const Vector pivot = Simd::broadcast(d.diagonal[j * d.diagonalStride]);
                for (std::size_t r = 0; r < ROWS; ++r)
                    accumulators[c][r] = Simd::divide(accumulators[c][r], pivot);
            }
        }

        for (std::size_t c = 0; c < GROUP; ++c)
            if (c < width)
                for (std::size_t r = 0; r < ROWS; ++r)
                    Simd::store(column(first + c) + r * Simd::LANES, accumulators[c][r]);
        first += width;
    }
}

void solvePanelAll(const PanelSolveOperands& operands)
{
    constexpr std::size_t STRIP_VECTORS = Simd::ROW_VECTORS;
    std::size_t row = 0;
    for (; row + STRIP_VECTORS * Simd::LANES <= operands.rowCount; row += STRIP_VECTORS * Simd::LANES)
        solvePanelStripGrouped(operands, row);
    for (; row + Simd::LANES <= operands.rowCount; row += Simd::LANES)
        solvePanelRows<1>(operands, row);
    for (; row < operands.rowCount; ++row)
        solvePanelRows<0>(operands, row);
}

// ---------------------------------------------------------------------------------------------------------------
// Triangle solves with many right-hand sides

void solveUnitLowerBlockAll(const TriangleSolveOperands& operands)
{
    // Column sweep (as the reference): once x(q) is final, subtract l(i, q)·x(q) from the rows below, vectorised over
    // i; every x(i) receives its terms in increasing q.
    for (std::size_t c = 0; c < operands.rhsCount; ++c)
    {
        double* x = operands.b + c * operands.ldb;
        for (std::size_t q = 0; q < operands.size; ++q)
        {
            const double xq = x[q];
            const Vector xVector = Simd::broadcast(xq);
            const double* lColumn = operands.l + q * operands.ldl;
            std::size_t i = q + 1;
            for (; i + Simd::LANES <= operands.size; i += Simd::LANES)
                Simd::store(x + i, Simd::subtractProduct(Simd::load(lColumn + i), xVector, Simd::load(x + i)));
            for (; i < operands.size; ++i)
                x[i] = Simd::subtractProduct(lColumn[i], xq, x[i]);
        }
    }
}

// PERFORMANCE: the backward substitution of one right-hand side is a chain of dependent fused operations (one every
// 4 cycles); RHS_INTERLEAVE right-hand sides are solved together so that their independent chains overlap.
// Measured (solve, 64 right-hand sides, one thread): forward and backward 12.1, 28.0 and 35.3
// GFLOP/s at n = 256, 1 000, 2 000, against 7.8, 18.0 and 24.7 GFLOP/s without interleaving.
constexpr std::size_t RHS_INTERLEAVE = 4;

template <std::size_t Count>
void solveUnitLowerTransposedColumns(const TriangleSolveOperands& operands, std::size_t firstColumn)
{
    double* x[Count];
    for (std::size_t c = 0; c < Count; ++c)
        x[c] = operands.b + (firstColumn + c) * operands.ldb;
    for (std::size_t i = operands.size; i-- > 0;)
    {
        const double* lColumn = operands.l + i * operands.ldl;
        double sums[Count];
        for (std::size_t c = 0; c < Count; ++c)
            sums[c] = x[c][i];
        for (std::size_t q = operands.size; q-- > i + 1;)
        {
            const double lValue = lColumn[q];
            for (std::size_t c = 0; c < Count; ++c)
                sums[c] = Simd::subtractProduct(lValue, x[c][q], sums[c]);
        }
        for (std::size_t c = 0; c < Count; ++c)
            x[c][i] = sums[c];
    }
}

void solveUnitLowerTransposedBlockAll(const TriangleSolveOperands& operands)
{
    std::size_t column = 0;
    for (; column + RHS_INTERLEAVE <= operands.rhsCount; column += RHS_INTERLEAVE)
        solveUnitLowerTransposedColumns<RHS_INTERLEAVE>(operands, column);
    for (; column < operands.rhsCount; ++column)
        solveUnitLowerTransposedColumns<1>(operands, column);
}

// ---------------------------------------------------------------------------------------------------------------
// Rows of right-hand sides: the products and triangles of the supernodal solves (RowProductOperands)

// PERFORMANCE: register blocking of the solves on rows of right-hand sides. A micro-tile is ROW_TARGETS target rows ×
// ROW_STRIP_VECTORS vectors of columns (AVX2: 6 × 8, 12 accumulators; AVX-512: 8 × 24, 24 accumulators) held in
// registers while the terms are swept: per term, ROW_STRIP_VECTORS loads of the source row and ROW_TARGETS broadcasts
// of l feed ROW_TARGETS·ROW_STRIP_VECTORS fused operations (with the vectors along the right-hand sides). The panel is
// read in place, unpacked (ROW_TARGETS values of a column or of a row of it per term); the source rows of
// ROW_TERM_BLOCK terms stay in L2 while the strips of a group of targets reuse them, the accumulators being stored and
// reloaded between term blocks (the chains unchanged). The targets left by the groups are taken by four, two, then one.
// Measured: the AVX2 micro-tile 6 × 8 instead of 4 × 12 (a tile of 128 columns whole strips, no column left to
// one-vector strips) and the targets left by pairs: the solves of a design model 6.8 -> 6.3 s (mixed) and 19.9 -> 18.4
// s (2d) on six threads, three runs alternating; the transposed products of 6 to 128 targets 26-40 -> 36-50
// GFLOP/s on one thread.
constexpr std::size_t ROW_TARGETS = Simd::ROW_TARGETS;
constexpr std::size_t ROW_STRIP_VECTORS = Simd::ROW_STRIP_VECTORS;
constexpr std::size_t ROW_TERM_BLOCK = 128;

const double* rowOf(const double* x, std::size_t ldx, std::int32_t row)
{
    return x + static_cast<std::size_t>(row) * ldx;
}

double* rowOf(double* x, std::size_t ldx, std::int32_t row)
{
    return x + static_cast<std::size_t>(row) * ldx;
}

// Targets × Vectors elements of the rows targets[t] + column receive termCount terms: term p multiplies the source row
// sources[p·sourceStep] (at the same column) by l[t·targetStride + p·termStride].
template <std::size_t Targets, std::size_t Vectors>
void subtractRowMicroTile(double* const* targets, const double* x, std::size_t ldx, std::size_t column,
                          const std::int32_t* sources, std::ptrdiff_t sourceStep, std::size_t termCount,
                          const double* l, std::ptrdiff_t targetStride, std::ptrdiff_t termStride)
{
    Vector accumulators[Targets][Vectors];
    for (std::size_t t = 0; t < Targets; ++t)
        for (std::size_t v = 0; v < Vectors; ++v)
            accumulators[t][v] = Simd::load(targets[t] + column + v * Simd::LANES);
    for (std::size_t p = 0; p < termCount; ++p)
    {
        const double* source = rowOf(x, ldx, sources[static_cast<std::ptrdiff_t>(p) * sourceStep]) + column;
        Vector values[Vectors];
        for (std::size_t v = 0; v < Vectors; ++v)
            values[v] = Simd::load(source + v * Simd::LANES);
        const double* lTerm = l + static_cast<std::ptrdiff_t>(p) * termStride;
        for (std::size_t t = 0; t < Targets; ++t)
        {
            const Vector w = Simd::broadcast(lTerm[static_cast<std::ptrdiff_t>(t) * targetStride]);
            for (std::size_t v = 0; v < Vectors; ++v)
                accumulators[t][v] = Simd::subtractProduct(values[v], w, accumulators[t][v]);
        }
    }
    for (std::size_t t = 0; t < Targets; ++t)
        for (std::size_t v = 0; v < Vectors; ++v)
            Simd::store(targets[t] + column + v * Simd::LANES, accumulators[t][v]);
}

// Every strip of columns of Targets target rows, for termCount terms.
template <std::size_t Targets>
void subtractRowStrips(double* const* targets, const RowProductOperands& operands, const std::int32_t* sources,
                       std::size_t termCount, const double* l)
{
    constexpr std::size_t STRIP = ROW_STRIP_VECTORS * Simd::LANES;
    std::size_t column = 0;
    for (; column + STRIP <= operands.width; column += STRIP)
        subtractRowMicroTile<Targets, ROW_STRIP_VECTORS>(targets, operands.x, operands.ldx, column, sources,
                                                         operands.sourceStep, termCount, l, operands.targetStride,
                                                         operands.termStride);
    for (; column + Simd::LANES <= operands.width; column += Simd::LANES)
        subtractRowMicroTile<Targets, 1>(targets, operands.x, operands.ldx, column, sources, operands.sourceStep,
                                         termCount, l, operands.targetStride, operands.termStride);
}

void subtractRowProductsAll(const RowProductOperands& operands)
{
    for (std::size_t firstTerm = 0; firstTerm < operands.termCount; firstTerm += ROW_TERM_BLOCK)
    {
        const std::size_t termCount = smaller(ROW_TERM_BLOCK, operands.termCount - firstTerm);
        const std::int32_t* sources = operands.sources + static_cast<std::ptrdiff_t>(firstTerm) * operands.sourceStep;
        const double* lBlock = operands.l + static_cast<std::ptrdiff_t>(firstTerm) * operands.termStride;
        std::size_t first = 0;
        for (; first + ROW_TARGETS <= operands.targetCount; first += ROW_TARGETS)
        {
            double* targets[ROW_TARGETS];
            for (std::size_t t = 0; t < ROW_TARGETS; ++t)
                targets[t] = rowOf(operands.x, operands.ldx,
                                   operands.targets[static_cast<std::ptrdiff_t>(first + t) * operands.targetStep]);
            subtractRowStrips<ROW_TARGETS>(targets, operands, sources, termCount,
                                           lBlock + static_cast<std::ptrdiff_t>(first) * operands.targetStride);
        }
        if constexpr (ROW_TARGETS > 4)
            for (; first + 4 <= operands.targetCount; first += 4)
            {
                double* targets[4];
                for (std::size_t t = 0; t < 4; ++t)
                    targets[t] = rowOf(operands.x, operands.ldx,
                                       operands.targets[static_cast<std::ptrdiff_t>(first + t) * operands.targetStep]);
                subtractRowStrips<4>(targets, operands, sources, termCount,
                                     lBlock + static_cast<std::ptrdiff_t>(first) * operands.targetStride);
            }
        // The targets left (fewer than four) by pairs, then alone: alone, a target takes its terms at a third of the
        // rate.
        for (; first + 2 <= operands.targetCount; first += 2)
        {
            double* targets[2];
            for (std::size_t t = 0; t < 2; ++t)
                targets[t] = rowOf(operands.x, operands.ldx,
                                   operands.targets[static_cast<std::ptrdiff_t>(first + t) * operands.targetStep]);
            subtractRowStrips<2>(targets, operands, sources, termCount,
                                 lBlock + static_cast<std::ptrdiff_t>(first) * operands.targetStride);
        }
        for (; first < operands.targetCount; ++first)
        {
            double* target[1] = {rowOf(operands.x, operands.ldx,
                                       operands.targets[static_cast<std::ptrdiff_t>(first) * operands.targetStep])};
            subtractRowStrips<1>(target, operands, sources, termCount,
                                 lBlock + static_cast<std::ptrdiff_t>(first) * operands.targetStride);
        }
    }
}

// The triangle inside a group of `count` rows (at most ROW_TARGETS), on Vectors vectors of columns from `column`, after
// the terms of the rows outside the group: forward, row c receives the earlier rows q < c of the group, q increasing,
// with l(c, q) = l[c + q·ldl]; transposed, row c receives the later rows q > c, q decreasing, with l(q, c).
template <std::size_t Vectors>
void solveRowGroup(double* const* rows, std::size_t count, std::size_t column, const double* l, std::size_t ldl,
                   bool isTransposed)
{
    Vector values[ROW_TARGETS][Vectors];
    for (std::size_t c = 0; c < count; ++c)
        for (std::size_t v = 0; v < Vectors; ++v)
            values[c][v] = Simd::load(rows[c] + column + v * Simd::LANES);
    if (!isTransposed)
    {
        for (std::size_t c = 1; c < count; ++c)
            for (std::size_t q = 0; q < c; ++q)
            {
                const Vector w = Simd::broadcast(l[c + q * ldl]);
                for (std::size_t v = 0; v < Vectors; ++v)
                    values[c][v] = Simd::subtractProduct(values[q][v], w, values[c][v]);
            }
    }
    else
    {
        for (std::size_t c = count - 1; c-- > 0;)
            for (std::size_t q = count - 1; q > c; --q)
            {
                const Vector w = Simd::broadcast(l[q + c * ldl]);
                for (std::size_t v = 0; v < Vectors; ++v)
                    values[c][v] = Simd::subtractProduct(values[q][v], w, values[c][v]);
            }
    }
    for (std::size_t c = 0; c < count; ++c)
        for (std::size_t v = 0; v < Vectors; ++v)
            Simd::store(rows[c] + column + v * Simd::LANES, values[c][v]);
}

void solveRowGroupStrips(double* const* rows, std::size_t count, std::size_t width, const double* l, std::size_t ldl,
                         bool isTransposed)
{
    constexpr std::size_t STRIP = ROW_STRIP_VECTORS * Simd::LANES;
    std::size_t column = 0;
    for (; column + STRIP <= width; column += STRIP)
        solveRowGroup<ROW_STRIP_VECTORS>(rows, count, column, l, ldl, isTransposed);
    for (; column + Simd::LANES <= width; column += Simd::LANES)
        solveRowGroup<1>(rows, count, column, l, ldl, isTransposed);
}

// The row triangle by groups of ROW_TARGETS rows: each group first receives the terms of the rows already solved
// outside it (the row products, in the order of the chain), then solves its own small triangle; every element keeps
// the chain of RowTriangleOperands.
void solveRowTriangleAll(const RowTriangleOperands& operands)
{
    const std::size_t size = operands.size;
    RowProductOperands product;
    product.x = operands.x;
    product.ldx = operands.ldx;
    product.width = operands.width;
    double* rows[ROW_TARGETS];
    if (!operands.isTransposed)
    {
        for (std::size_t first = 0; first < size; first += ROW_TARGETS)
        {
            const std::size_t count = smaller(ROW_TARGETS, size - first);
            if (first > 0)
            {
                // Rows [first, first + count) receive the rows [0, first), q increasing: l(i, p) = l(first + i, p).
                product.targets = operands.rows + first;
                product.targetStep = 1;
                product.targetCount = count;
                product.sources = operands.rows;
                product.sourceStep = 1;
                product.termCount = first;
                product.l = operands.l + first;
                product.targetStride = 1;
                product.termStride = static_cast<std::ptrdiff_t>(operands.ldl);
                subtractRowProductsAll(product);
            }
            for (std::size_t c = 0; c < count; ++c)
                rows[c] = rowOf(operands.x, operands.ldx, operands.rows[first + c]);
            solveRowGroupStrips(rows, count, operands.width, operands.l + first + first * operands.ldl, operands.ldl,
                                false);
        }
        return;
    }
    std::size_t end = size;
    while (end > 0)
    {
        const std::size_t first = end > ROW_TARGETS ? end - ROW_TARGETS : 0;
        const std::size_t count = end - first;
        if (end < size)
        {
            // Rows [first, end) receive the rows [end, size), q decreasing: term p is row size − 1 − p and
            // l(i, p) = l(size − 1 − p, first + i).
            product.targets = operands.rows + first;
            product.targetStep = 1;
            product.targetCount = count;
            product.sources = operands.rows + (size - 1);
            product.sourceStep = -1;
            product.termCount = size - end;
            product.l = operands.l + (size - 1) + first * operands.ldl;
            product.targetStride = static_cast<std::ptrdiff_t>(operands.ldl);
            product.termStride = -1;
            subtractRowProductsAll(product);
        }
        for (std::size_t c = 0; c < count; ++c)
            rows[c] = rowOf(operands.x, operands.ldx, operands.rows[first + c]);
        solveRowGroupStrips(rows, count, operands.width, operands.l + first + first * operands.ldl, operands.ldl, true);
        end = first;
    }
}
