#pragma once

#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"

#include <cstdint>
#include <span>

namespace oos::linalg
{

// The front source of a matrix held whole (FrontSource): each front takes the matrix's entries of its
// supernode's columns (times a scale: −1 gives B = −K_d of a buckling pencil from K_d), in the order of each column's
// entries, as the multifrontal factorisation takes a matrix's
// entries (assembleFront); no right-hand side. The matrices of the full path given to multilevel substructuring, which
// takes its pencils from front sources so that the pencils of the elements take the same path.
//
// Thread-safety: addTerms() and diagonal() are safe concurrently (they read the matrix). Ownership: references the
// matrix and the symbolic factor, which must outlive it.
class MatrixFrontSource final : public FrontSource
{
public:
    // The matrix must be in the order of the symbolic factor (std::invalid_argument otherwise).
    MatrixFrontSource(const SymmetricCsc& matrix, const SymbolicFactor& symbolic, double scale = 1.0);

    void addTerms(FrontTerms& front) override;
    void diagonal(std::int32_t supernode, std::span<double> values) const override;

private:
    const SymmetricCsc& m_matrix;
    const SymbolicFactor& m_symbolic;
    double m_scale;
};

} // namespace oos::linalg
