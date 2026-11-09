#pragma once

#include "oos/linalg/dense/DenseMatrix.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::design
{

// The mode tracking of an eigenvalue response (mode= and track=True: follow a mode from one design point to the next
// by its shape, with a modal assurance criterion; reported when the tracking is uncertain): at a new design point the
// modes may change their order (two frequencies cross as a thickness grows), so the response takes the mode whose
// shape is closest to the reference shape (the mode it took at the previous design cycle), not the root of the same
// rank.
//
// The modal assurance criterion of two shapes (Allemang, "The Modal Assurance Criterion – Twenty Years of Use and
// Abuse", Sound and Vibration 37(8), 2003):
//     MAC(a, b) = (aᵀ·b)² / ((aᵀ·a)·(bᵀ·b)),
// 1 for the same shape at any scale and sign, 0 for orthogonal ones. Taken on the g set of the two points (the same DOF
// map: a design point never changes the topology), unweighted: the mass of the new point would be needed for the
// cross-orthogonality, and the shapes of two points close enough to be tracked are told apart without it.
//
// The tracking is uncertain when the best match is below MODE_TRACKING_MIN_MAC, or when the next best is within
// MODE_TRACKING_AMBIGUITY of it (a repeated root splits a shape over its copies, a crossing in progress mixes two):
// the mode is still taken, and the evaluation reports it.
//
// Thread-safety: stateless. Ownership: none.
class ModeTracking
{
public:
    // A clean tracking keeps MAC above 0.99 (the modes of the tests moved by 10 % of a thickness); 0.9 still names one
    // mode, below it the shape has changed so much that "the same mode" is a matter of judgement.
    static constexpr double MODE_TRACKING_MIN_MAC = 0.9;
    // Two candidates within 0.1 of MAC of each other are not told apart with confidence.
    static constexpr double MODE_TRACKING_AMBIGUITY = 0.1;

    struct Match
    {
        std::size_t mode = 0; // the column of the best match (0-based)
        double mac = 0.0;
        std::size_t secondMode = 0; // the next best (the best itself when there is one column)
        double secondMac = 0.0;
        bool isUncertain = false;
    };

    // MAC(a, b); 0 when either shape is zero. Precondition: a and b have the same size.
    [[nodiscard]] static double modalAssuranceCriterion(std::span<const double> a, std::span<const double> b) noexcept;

    // The column of `shapes` (g set × modes) closest to `reference`, among the first `count` columns (every column when
    // 0). Precondition: the reference has shapes.rows() values and there is at least one column.
    [[nodiscard]] static Match match(std::span<const double> reference, const linalg::DenseMatrix& shapes,
                                     std::size_t count = 0);
};

} // namespace oos::design
