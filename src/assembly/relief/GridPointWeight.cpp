#include "oos/assembly/relief/GridPointWeight.hpp"

#include "oos/assembly/singularities/SmallSymmetricEigen.hpp"

#include <algorithm>
#include <cmath>

namespace oos::assembly
{

namespace
{

// An off-diagonal term of Mt at most this fraction of its diagonal is zero: Mt is then diagonal and S the identity
// (the masses of the elements are isotropic; their sums differ from a diagonal matrix by round-off only).
constexpr double DIAGONAL_MASS_TOLERANCE = 1.0e-12;

base::Mat3 blockOf(const base::MatN<6>& m, std::size_t row, std::size_t column)
{
    base::Mat3 block;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            block(i, j) = m(row + i, column + j);
    return block;
}

// The eigenvectors of a symmetric 3 × 3 matrix as the rows of a rotation, each placed at the axis of its largest
// component and oriented with that component positive (on a conflict, the order of increasing eigenvalue is kept);
// the eigenvalues in the same order.
void principalRows(const base::Mat3& matrix, base::Mat3& axes, std::array<double, 3>& values)
{
    const SmallEigenDecomposition<3> decomposition = SmallSymmetricEigen<3>::decompose(matrix);
    std::array<std::size_t, 3> slotOf{};
    std::array<bool, 3> isTaken{};
    bool isConflict = false;
    for (std::size_t k = 0; k < 3; ++k)
    {
        std::size_t axis = 0;
        for (std::size_t i = 1; i < 3; ++i)
            if (std::abs(decomposition.vectors(i, k)) > std::abs(decomposition.vectors(axis, k)))
                axis = i;
        isConflict = isConflict || isTaken[axis];
        isTaken[axis] = true;
        slotOf[k] = axis;
    }
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t slot = isConflict ? k : slotOf[k];
        std::size_t largest = 0;
        for (std::size_t i = 1; i < 3; ++i)
            if (std::abs(decomposition.vectors(i, k)) > std::abs(decomposition.vectors(largest, k)))
                largest = i;
        const double sign = decomposition.vectors(largest, k) < 0.0 ? -1.0 : 1.0;
        for (std::size_t i = 0; i < 3; ++i)
            axes(slot, i) = sign * decomposition.vectors(i, k);
        values[slot] = decomposition.values[k];
    }
}

} // namespace

GridPointWeight GridPointWeight::fromRigidBodyMass(const base::MatN<6>& rigidBodyMass, std::int32_t referencePoint,
                                                   base::Vec3 referencePosition)
{
    GridPointWeight weight;
    weight.referencePoint = referencePoint;
    weight.referencePosition = referencePosition;
    weight.rigidBodyMass = rigidBodyMass;
    const base::Mat3 translational = blockOf(rigidBodyMass, 0, 0);
    bool isDiagonal = true;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (i != j &&
                std::abs(translational(i, j)) >
                    DIAGONAL_MASS_TOLERANCE * std::max(std::abs(translational(i, i)), std::abs(translational(j, j))))
                isDiagonal = false;
    std::array<double, 3> unused{};
    if (!isDiagonal)
        principalRows(translational, weight.massAxes, unused);
    const base::Mat3& s = weight.massAxes;
    const base::Mat3 st = s.transposed();
    const base::Mat3 mt = s * (translational * st);
    const base::Mat3 mtr = s * (blockOf(rigidBodyMass, 0, 3) * st);
    const base::Mat3 mrr = s * (blockOf(rigidBodyMass, 3, 3) * st);
    for (std::size_t i = 0; i < 3; ++i)
        weight.masses[i] = mt(i, i);
    const auto ratio = [](double value, double mass) { return mass != 0.0 ? value / mass : 0.0; };
    const double m1 = weight.masses[0];
    const double m2 = weight.masses[1];
    const double m3 = weight.masses[2];
    weight.centresOfGravity[0] = {0.0, -ratio(mtr(0, 2), m1), ratio(mtr(0, 1), m1)};
    weight.centresOfGravity[1] = {ratio(mtr(1, 2), m2), 0.0, -ratio(mtr(1, 0), m2)};
    weight.centresOfGravity[2] = {-ratio(mtr(2, 1), m3), ratio(mtr(2, 0), m3), 0.0};
    const base::Vec3& c1 = weight.centresOfGravity[0];
    const base::Vec3& c2 = weight.centresOfGravity[1];
    const base::Vec3& c3 = weight.centresOfGravity[2];
    base::Mat3 transfer;
    transfer(0, 0) = m2 * c2[2] * c2[2] + m3 * c3[1] * c3[1];
    transfer(1, 1) = m1 * c1[2] * c1[2] + m3 * c3[0] * c3[0];
    transfer(2, 2) = m1 * c1[1] * c1[1] + m2 * c2[0] * c2[0];
    transfer(0, 1) = transfer(1, 0) = -m3 * c3[0] * c3[1];
    transfer(0, 2) = transfer(2, 0) = -m2 * c2[0] * c2[2];
    transfer(1, 2) = transfer(2, 1) = -m1 * c1[1] * c1[2];
    weight.inertia = mrr;
    weight.inertia -= transfer;
    // The grid point weight generator's convention: I(S) holds the products of inertia as Σ m·x·y, the opposite sign
    // of the tensor terms of MO (MO(4,5) = −Σ m·x·y while I(S)(2,3) = +Σ m·y·z about the centre of gravity), and its
    // principal axes Q come from that matrix.
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (i != j)
                weight.inertia(i, j) = -weight.inertia(i, j);
    principalRows(weight.inertia, weight.principalAxes, weight.principalInertia);
    return weight;
}

base::Vec3 GridPointWeight::centreOfGravity() const noexcept
{
    const std::array<base::Vec3, 3>& c = centresOfGravity;
    return referencePosition +
           base::Vec3{0.5 * (c[1][0] + c[2][0]), 0.5 * (c[0][1] + c[2][1]), 0.5 * (c[0][2] + c[1][2])};
}

} // namespace oos::assembly
