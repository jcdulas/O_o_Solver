#include "oos/reduction/external/ExternalLoadSource.hpp"

namespace oos::reduction
{

void ExternalLoadSource::condensedLoadDerivatives(const ExternalUnit& /*unit*/,
                                                  linalg::ConstDenseMatrixView /*interiorLoads*/,
                                                  std::span<const linalg::DenseMatrixView> /*derivatives*/) const
{
}

} // namespace oos::reduction
