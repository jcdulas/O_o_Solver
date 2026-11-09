#pragma once

#include <cstddef>

namespace oos::linalg
{

// The inertia of a symmetric matrix: its numbers of positive, negative and zero eigenvalues. By Sylvester's law of
// inertia it equals the inertia of D in any factorisation P·A·Pᵀ = L·D·Lᵀ (Golub & Van Loan §8.1.5): the Sturm count
// of the eigenvalue solvers reads the number of negative pivots of K − σM.
struct Inertia
{
    std::size_t positive = 0;
    std::size_t negative = 0;
    std::size_t zero = 0;

    friend bool operator==(const Inertia&, const Inertia&) = default;
};

} // namespace oos::linalg
