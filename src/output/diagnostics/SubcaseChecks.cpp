#include "oos/output/diagnostics/SubcaseChecks.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"

#include <stdexcept>
#include <string>

namespace oos::output
{

LoadResultant LoadResultant::of(const assembly::DofMap& dofMap, std::span<const double> valuesG,
                                base::Vec3 referencePoint)
{
    if (valuesG.size() != static_cast<std::size_t>(dofMap.gSetSize()))
        throw std::invalid_argument(base::concatenate({"LoadResultant: expected ", std::to_string(dofMap.gSetSize()),
                                                       " values (the g set), found ", std::to_string(valuesG.size())}));
    LoadResultant resultant;
    constexpr std::size_t T1 = 0;
    constexpr std::size_t T2 = 1;
    constexpr std::size_t T3 = 2;
    constexpr std::size_t R1 = 3;
    constexpr std::size_t R2 = 4;
    constexpr std::size_t R3 = 5;
    for (std::int32_t grid = 0; grid < dofMap.gridCount(); ++grid)
    {
        base::Vec3 forceCd;
        base::Vec3 momentCd;
        for (int k = 0; k < 3; ++k)
        {
            forceCd[static_cast<std::size_t>(k)] = valuesG[static_cast<std::size_t>(dofMap.gridDof(grid, k + 1))];
            momentCd[static_cast<std::size_t>(k)] = valuesG[static_cast<std::size_t>(dofMap.gridDof(grid, k + 4))];
        }
        // PERFORMANCE: a grid point without load adds nothing; skipping it changes no bit (the sums start at +0, and
        // adding ±0 to a sum that started at +0 leaves it unchanged).
        // Measured: with the DofMap columns of the same merge, 600 cases of 197 640 DOFs from 11.1 s to 3.7 s
        // (DofMap.hpp).
        if (forceCd[0] == 0.0 && forceCd[1] == 0.0 && forceCd[2] == 0.0 && momentCd[0] == 0.0 && momentCd[1] == 0.0 &&
            momentCd[2] == 0.0)
            continue;
        // v_basic = R·v_cd, R the CD axes in basic.
        const base::Mat3 rotation = dofMap.cdToBasic(grid);
        const base::Vec3 f = rotation * forceCd;
        const base::Vec3 m = rotation * momentCd;
        const base::Vec3 r = dofMap.positionInBasic(grid) - referencePoint;
        // r × F split by the component of F that gives each term.
        resultant.rows[0][T1] += f[0];
        resultant.rows[0][R2] += r[2] * f[0];
        resultant.rows[0][R3] -= r[1] * f[0];
        resultant.rows[1][T2] += f[1];
        resultant.rows[1][R1] -= r[2] * f[1];
        resultant.rows[1][R3] += r[0] * f[1];
        resultant.rows[2][T3] += f[2];
        resultant.rows[2][R1] += r[1] * f[2];
        resultant.rows[2][R2] -= r[0] * f[2];
        resultant.rows[3][R1] += m[0];
        resultant.rows[4][R2] += m[1];
        resultant.rows[5][R3] += m[2];
    }
    for (std::size_t row = 0; row < 6; ++row)
        for (std::size_t column = 0; column < 6; ++column)
            resultant.totals[column] += resultant.rows[row][column];
    return resultant;
}

} // namespace oos::output
