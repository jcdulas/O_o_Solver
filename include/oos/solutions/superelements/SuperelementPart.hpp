#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace oos::solutions
{

// The number of the residual structure (SEBULK, SUPER: superelement 0).
inline constexpr std::int32_t RESIDUAL_STRUCTURE = 0;

// A boundary point of a superelement (O_o_Reader's part_boundaries): the point in the superelement's own ID space and
// the point of the residual structure it is (an SuperelementConnection pair, the automatic search of
// SuperelementBulkOptions AUTO, an SuperelementExternalBoundary point of an external superelement).
struct SuperelementBoundaryPoint
{
    base::EntityId nodeId = 0;
    base::EntityId residualNodeId = 0;
};

// The matrices of an external superelement (SuperelementBulkOptions of the external matrix file type: the boundary
// stiffness KAA and mass MAA of the external matrix file its ASSIGN statement names), on its a set in this order: the
// points of its AnalysisDofSet, GeneralizedDofSet and SuperelementExternalBoundary cards by increasing ID, six
// components for a grid point and one for a scalar point.
struct ExternalSuperelementMatrices
{
    std::vector<reduction::UnitDofLabel> labels; // the rows, in the superelement's own IDs
    linalg::DenseMatrix stiffness;               // labels × labels
    std::optional<linalg::DenseMatrix> mass;     // labels × labels; nothing: not given
};

// One superelement upstream of the residual structure (deck superelements: each mapped to a condensation unit; a single
// level, the residual structure downstream of every one).
struct SuperelementPart
{
    std::int32_t superelementId = 0;
    // A part (BEGIN SUPER): its snapshot, taken with the Case Control tables of the residual structure (a part shares
    // the Case Control of the deck, BEGIN SUPER: its subcases select the set IDs of its own Bulk Data). Null for an
    // external superelement.
    std::shared_ptr<const input::ModelSnapshot> snapshot;
    // An external superelement: its matrices. Null for a part.
    std::shared_ptr<const ExternalSuperelementMatrices> external;
    std::vector<SuperelementBoundaryPoint> boundary; // sorted by nodeId
    // The q set of a part (QSET, GeneralizedDofSet1, SuperelementGeneralizedDofSet, SuperelementGeneralizedDofSet1 of
    // its scalar points): its generalised coordinates, the fixed-interface modes of its Craig-Bampton reduction in a
    // modal analysis, increasing. Empty: the part is reduced statically (Guyan) there.
    std::vector<base::EntityId> modalPoints;
};

// A subcase of the Case Control with the superelements its SUPER command selects (SUPER: a superelement, or a SET
// of them; without SUPER, every superelement processes every subcase).
struct SuperelementSubcase
{
    base::EntityId subcase = 1;
    std::vector<std::int32_t> superelements; // sorted; empty: every superelement

    [[nodiscard]] bool selects(std::int32_t superelementId) const noexcept;
};

} // namespace oos::solutions
