#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace oos::linalg
{

// One Sturm point: a shift and the number of eigenvalues below it (the negative pivots of K − σM).
struct SturmPoint
{
    double shift = 0.0;
    std::size_t countBelow = 0;
};

// An interval between two consecutive Sturm points whose count of eigenvalues differs from the count found in it.
struct SturmMismatch
{
    double lower = 0.0;
    double upper = 0.0;
    std::size_t expectedCount = 0; // countBelow(upper) − countBelow(lower)
    std::size_t foundCount = 0;
};

// What the Sturm counts say about a set of eigenvalues found: from the lowest Sturm point upward, the consecutive
// intervals [a, b) in which the number found equals the Sturm difference are verified; the first that does not is the
// mismatch.
struct SturmVerification
{
    // The highest Sturm point reached with every interval below it verified (the lowest point when none is).
    double verifiedUpTo = 0.0;
    // The eigenvalues found in [lowest point, verifiedUpTo): all those of the pencil there, no mode is missed.
    std::size_t verifiedCount = 0;
    // The first interval that does not agree, above verifiedUpTo, if any.
    bool hasMismatch = false;
    SturmMismatch mismatch;
};

// The Sturm sequence check of the eigenvalue solver: no mode of a range is missed.
//
// Method (Grimes, Lewis & Simon, "A shifted block Lanczos algorithm for solving sparse symmetric generalized
// eigenproblems", SIAM J. Matrix Anal. Appl. 15 (1994) §4.2, the trust intervals): by Sylvester's law of inertia, the
// number of negative pivots of LDLᵀ = K − σM is the number ν(σ) of eigenvalues of K φ = λ M φ below σ (M positive
// semi-definite and K − σM non-singular). The eigenvalues in [a, b) are then ν(b) − ν(a) in number; when the solver has
// found exactly that many there, it has found all of them. Every factorisation of the solver (a shift, a bound of the
// range) is a Sturm point; the check needs no factorisation of its own.
//
// An eigenvalue found within rounding of a Sturm point may be counted on the wrong side of it; ShiftStrategy keeps the
// shifts away from the eigenvalues found (SHIFT_CLEARANCE) so that this does not happen.
//
// Thread-safety: as a std::vector. Ownership: owns its points.
class SturmCheck
{
public:
    // Adds a point (kept sorted by shift). Throws SolverError when its count contradicts a point already there (a
    // count that decreases as the shift grows, or two counts for one shift): the factorisations are then not reliable.
    void addPoint(SturmPoint point);

    [[nodiscard]] const std::vector<SturmPoint>& points() const noexcept { return m_points; }
    [[nodiscard]] bool isEmpty() const noexcept { return m_points.empty(); }
    // ν at the shift of a point already added; throws std::invalid_argument when there is none at that shift.
    [[nodiscard]] std::size_t countBelow(double shift) const;

    // The verification of the eigenvalues found, given in increasing order. Precondition: at least one point.
    [[nodiscard]] SturmVerification verify(std::span<const double> sortedEigenvalues) const;

private:
    std::vector<SturmPoint> m_points; // bounded by the shifts of one analysis
};

} // namespace oos::linalg
