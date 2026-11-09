#pragma once

#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace oos::base
{

// Parallel loops whose results do not depend on the number of threads (parallel work shares the blocks of
// a result, never the terms of one sum; partial sums are combined by pieces of fixed size in a fixed order).
//
// [0, count) is split into pieces of `pieceSize` indices: piece p covers [p·pieceSize, min((p+1)·pieceSize, count)).
// The pieces depend on the size of the work and on the piece size the caller fixes, never on the number of threads;
// the pool only decides which thread runs a piece. A caller must therefore choose `pieceSize` from the work alone (a
// named constant, or a function of `count`), never from pool->threadCount().
//
// Cancellation: with a token, a piece that has not started when the token is cancelled is skipped, and the loop then
// throws OperationCancelledError. An interruption takes effect within the time of one piece per thread, so a piece
// must stay well under 0.1 s of work, or its body must call throwIfCancelled() itself.

// The number of pieces of [0, count): ⌈count / pieceSize⌉.
[[nodiscard]] constexpr std::size_t pieceCount(std::size_t count, std::size_t pieceSize)
{
    return pieceSize == 0 ? 0 : count / pieceSize + (count % pieceSize != 0 ? 1 : 0);
}

namespace detail
{

inline void requirePieceSize(std::size_t pieceSize)
{
    if (pieceSize == 0)
        throw std::invalid_argument("parallelFor: the piece size must be at least 1, found 0");
}

} // namespace detail

// Calls body(begin, end) for every piece of [0, count), on the pool (serially without one, or with one thread). The
// body of different pieces may run at the same time: it writes only to the part of the result its piece owns.
// Errors: the exception of the piece with the lowest index is rethrown (ThreadPool); with a cancelled token,
// OperationCancelledError.
template <typename Body>
void parallelFor(ThreadPool* pool, std::size_t count, std::size_t pieceSize, Body&& body,
                 const CancellationToken* cancellation = nullptr)
{
    detail::requirePieceSize(pieceSize);
    const std::size_t pieces = pieceCount(count, pieceSize);
    const std::function<void(std::size_t)> runPiece = [&](std::size_t piece)
    {
        if (cancellation != nullptr && cancellation->isCancelled())
            return;
        const std::size_t begin = piece * pieceSize;
        body(begin, std::min(begin + pieceSize, count));
    };
    if (pool == nullptr || pool->threadCount() == 1 || pieces <= 1)
    {
        for (std::size_t piece = 0; piece < pieces; ++piece)
            runPiece(piece);
    }
    else
    {
        pool->runBatch(pieces, runPiece);
    }
    if (cancellation != nullptr)
        cancellation->throwIfCancelled();
}

// The values fillOnPool() writes per piece: 8 MB of doubles, a few milliseconds of first touches.
inline constexpr std::size_t FILL_PIECE_VALUES = std::size_t{1} << 20;

// Writes `value` into every element of `values` on the pool, by pieces of FILL_PIECE_VALUES. The values do not depend
// on the pieces; the first touch of fresh pages (most of the time of zeroing a buffer just allocated) is spread over
// the threads.
template <typename T>
void fillOnPool(ThreadPool* pool, std::span<T> values, const T& value)
{
    parallelFor(pool, values.size(), FILL_PIECE_VALUES,
                [&](std::size_t begin, std::size_t end)
                {
                    std::fill(values.begin() + static_cast<std::ptrdiff_t>(begin),
                              values.begin() + static_cast<std::ptrdiff_t>(end), value);
                });
}

// Computes mapPiece(begin, end) for every piece, possibly in parallel, then combines the piece results **in piece
// order**, serially: combine(...combine(combine(identity, r0), r1)..., rN). The result has the same bits for any number
// of threads, because neither the pieces nor the order of the combination depend on it. T must be copyable; one T is
// kept per piece.
template <typename T, typename MapPiece, typename Combine>
[[nodiscard]] T parallelReduce(ThreadPool* pool, std::size_t count, std::size_t pieceSize, const T& identity,
                               MapPiece&& mapPiece, Combine&& combine, const CancellationToken* cancellation = nullptr)
{
    detail::requirePieceSize(pieceSize);
    const std::size_t pieces = pieceCount(count, pieceSize);
    std::vector<T> results(pieces, identity);
    parallelFor(
        pool, pieces, 1,
        [&](std::size_t piece, std::size_t)
        {
            const std::size_t begin = piece * pieceSize;
            results[piece] = mapPiece(begin, std::min(begin + pieceSize, count));
        },
        cancellation);
    T total = identity;
    for (const T& value : results)
        total = combine(total, value);
    return total;
}

} // namespace oos::base
