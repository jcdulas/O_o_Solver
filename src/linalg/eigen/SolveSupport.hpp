#pragma once

// What the vibration and buckling solves of BlockLanczos share (internal to src/linalg/eigen).

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/linalg/eigen/SturmCheck.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace oos::linalg
{

// The moves of a shift that met an eigenvalue before the pencil is declared singular: a pencil singular at four
// distinct shifts is singular everywhere (a DOF with neither mass nor stiffness), not unlucky.
inline constexpr std::size_t MAX_SINGULAR_MOVES = 3;

// Mixes the run index into the seed (a Weyl step of the golden ratio): every run has its own random start.
inline constexpr std::uint64_t SEED_STEP = 0x9E3779B97F4A7C15ULL;

// Throws SolverError when an interval holds more eigenvalues found than its Sturm count allows: the factorisations are
// then not reliable (a root on a Sturm point, or a nearly singular pencil).
inline void requireNoExtraEigenvalue(const SturmVerification& verification)
{
    if (verification.hasMismatch && verification.mismatch.foundCount > verification.mismatch.expectedCount)
        throw base::SolverError(base::concatenate(
            {"eigenvalue analysis: ", std::to_string(verification.mismatch.foundCount), " eigenvalues found in [",
             std::to_string(verification.mismatch.lower), ", ", std::to_string(verification.mismatch.upper),
             ") where the Sturm counts allow ", std::to_string(verification.mismatch.expectedCount),
             "; the factorisations are not reliable there (an eigenvalue on a bound of the range, or a pencil "
             "close to singular): move the bounds of the range or check the model for massless mechanisms"}));
}

} // namespace oos::linalg
