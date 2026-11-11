#pragma once

#include "oos/elements/bars_beams/BeamFormulation.hpp"

namespace oos::elements
{

// Shared sizing facets of Bar2 and Beam2, at fixed grid geometry. `other == nullptr` selects a first derivative;
// otherwise the mixed partial in (field, other), including the same-field second derivative. Unknown fields return
// false so the caller retains its numerical path. Matrices and linear recovery columns use exact chain rules.
class BeamFieldDerivatives
{
public:
    static bool loads(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                      const DesignedField& field, const DesignedField* other, const ElementLoadContext& loads,
                      ElementVectorBlock& block);
    static bool stiffness(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                           const DesignedField& field, const DesignedField* other, ElementMatrixBlock& block);
    static bool mass(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                      const DesignedField& field, const DesignedField* other, const MassOptions& options,
                      TorsionalInertia torsion, ElementMatrixBlock& block);
    static bool recovery(const BeamData& beam, const ElementFamily& family, const ElementBatchView& view,
                          std::int64_t row, const DesignedField& field, const DesignedField* other,
                          ElementOutput output, const NodalDisplacements& displacements, ElementResultBlock& block);
    static bool slopeRates(const BeamData& beam, const ElementFamily& family, const ElementBatchView& view,
                            std::int64_t row, const DesignedField& field, ElementOutput output, RecoverySlopes& block);
    static bool differential(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                              const DesignedField& field, const DesignedField* other,
                              const NodalDisplacements& preload, ElementMatrixBlock& block);
};

} // namespace oos::elements
