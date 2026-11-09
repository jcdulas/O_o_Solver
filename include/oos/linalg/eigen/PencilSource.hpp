#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

// The matrices of an eigenvalue pencil (K, B) given front by front and by their products, none of them held whole
// (the pencils from the elements). B is the pencil's right-hand matrix: M for vibration,
// −K_d for buckling (SpectralTransform). The front sources take the order and the structure of one symbolic factor;
// the products and diagonals are on its equations.
//
// The sources of multilevel substructuring (MultilevelSubstructuring: K, B and, for buckling, M for its vibration
// basis) and of the factor out of core (OutOfCorePencil: K − σB at each shift, the products with B and with the inner
// product's matrix).
//
// A plug-in point: called once per front or per block of vectors. Thread-safety: the sources'
// addTerms() as FrontSource says (concurrently for distinct supernodes); the products and diagonals are const and may
// use a pool inside, called from one thread at a time. Ownership: the implementation owns its sources.
class PencilSource
{
public:
    PencilSource() = default;
    PencilSource(const PencilSource&) = delete;
    PencilSource& operator=(const PencilSource&) = delete;
    PencilSource(PencilSource&&) = delete;
    PencilSource& operator=(PencilSource&&) = delete;
    virtual ~PencilSource() = default;

    [[nodiscard]] virtual std::int32_t dimension() const noexcept = 0;
    // The front sources of K, of B and of the mass M (null when the pencil has no mass: a buckling model without one;
    // for vibration, B itself).
    [[nodiscard]] virtual FrontSource& stiffnessSource() = 0;
    [[nodiscard]] virtual FrontSource& rightHandSource() = 0;
    [[nodiscard]] virtual FrontSource* massSource() = 0;
    // The front source of K − σ·B; valid until the next call.
    [[nodiscard]] virtual FrontSource& shiftedSource(double shift) = 0;
    // Y ← K·X and Y ← B·X (X and Y dimension() × r, distinct storage).
    virtual void multiplyStiffness(ConstDenseMatrixView x, DenseMatrixView y) const = 0;
    virtual void multiplyRightHand(ConstDenseMatrixView x, DenseMatrixView y) const = 0;
    // The diagonals of K and B (dimension() values each).
    virtual void diagonals(std::span<double> stiffness, std::span<double> rightHand) const = 0;
};

} // namespace oos::linalg
