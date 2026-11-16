#pragma once

#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"

#include <vector>

namespace oos::elements::mixedHybrid
{

// The map from the DOFs an element of the mixed hybrid formulation writes its matrices on (grid components in the
// displacement systems CD of the grid points, entity DOFs) to the local unknowns of its kernel (grid components in
// the basic system, each unknown with the sign the element sees it with): x_local = T u. On a grid block (the three
// components of one grid point, consecutive among the local unknowns) T = s R, R the CD-to-basic rotation of the grid
// point; s on an entity DOF. The element matrices are then Tᵀ K T, its loads Tᵀ f, its local unknowns T u.
//
// Thread-safety: a plain value.
struct LocalMap
{
    std::vector<ElementDof> dofs;
    Matrix toLocal; // local unknowns × dofs
};

// The map of the local unknowns `local` (MixedHybridModel::LocalDofs: their DOFs, the grid whose CD system each grid
// component is in the basic system of, their signs).
[[nodiscard]] LocalMap localMapOf(const ElementBatchView& view, const MixedHybridModel::LocalDofs& local);

// Tᵀ K T, row-major and complete (its symmetric part: K is symmetric up to round-off).
[[nodiscard]] std::vector<double> congruent(const Matrix& local, const LocalMap& map);

// Tᵀ f.
[[nodiscard]] std::vector<double> onDofs(const Vector& local, const LocalMap& map);

// The local unknowns from the model's displacements: x_local = T u.
[[nodiscard]] Vector localUnknowns(const LocalMap& map, const NodalDisplacements& displacements);

} // namespace oos::elements::mixedHybrid
