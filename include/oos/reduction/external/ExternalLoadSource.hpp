#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <span>

namespace oos::reduction
{

class ExternalUnit;

// The plug-in point of the condensed loads of an external unit (the condensed loads per basis vector, required when
// loads apply inside the zone; the loads of the zone are given to the plugin): given
// the loads of the zone's interior DOFs for some load vectors (the vectors of the load basis of a constraint set, or
// one case), the plugin's condensed loads f* on the unit's labels, and their derivatives with respect to the variables
// of the unit's derivatives. For a statically condensed local model, f* = −K_ai·K_ii⁻¹·P_i: the force its interior
// loads put on the contour when the contour is held (the loads of the contour points stay loads of the analysis).
//
// The interior DOFs are those of the zone (ExternalZone::interiorLabels): six per interior grid point, in increasing
// grid ID then component, in the displacement system of the grid point. The caller's plugins implement it.
//
// Thread-safety: called by the analysis on its own thread, one call at a time; an implementation that calls back into
// the caller hands the call over to the caller's thread. Ownership: none.
class ExternalLoadSource
{
public:
    ExternalLoadSource() = default;
    ExternalLoadSource(const ExternalLoadSource&) = delete;
    ExternalLoadSource& operator=(const ExternalLoadSource&) = delete;
    virtual ~ExternalLoadSource() = default;

    // f* of the vectors: interiorLoads (interior DOFs × vectors) → condensed (the unit's labels × vectors, zero on
    // entry). Throws what the plugin's failure becomes (a base error naming the zone).
    virtual void condensedLoads(const ExternalUnit& unit, linalg::ConstDenseMatrixView interiorLoads,
                                linalg::DenseMatrixView condensed) const = 0;

    // ∂f*/∂x_j of the vectors for each derivative d of unit.derivatives(), into derivatives[d] (labels × vectors, zero
    // on entry). By default nothing is written: condensed loads that do not depend on the variables.
    virtual void condensedLoadDerivatives(const ExternalUnit& unit, linalg::ConstDenseMatrixView interiorLoads,
                                          std::span<const linalg::DenseMatrixView> derivatives) const;
};

} // namespace oos::reduction
