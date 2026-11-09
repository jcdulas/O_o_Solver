#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/loads/LoadSetIndex.hpp"
#include "oos/base/arrow/DictionaryView.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace oos::assembly
{

// One row of an inertial load set as the load basis describes it (InertialLoads::termsOf): the
// generator of its acceleration field and its coefficients, the field linear in them. A GravityLoad row is its
// acceleration vector on the uniform translations of every grid point along the basic axes (three generators every
// GravityLoad row of the model shares), an AccelNodeLoad row the same on its points (three per list of points), an
// ACCEL row the translations along the basic axes scaled by its table (three per CID, DIR, LOCi and VALi), an
// RotationalForceLoad row its own field times its factor (one per centre, axis, A and RACC). The generator is named by
// its kind and its parameters, two rows naming the same field by the same values; the case's inertial loads are Σ
// coefficients · Mgg · (the generators' unit fields).
struct InertialTerm
{
    std::vector<double> generator;    // its kind, then its parameters (the identity of the field)
    std::vector<double> coefficients; // 3 (the basic axes) or 1 (RFORCE)
};

// The inertial loads of a load set (the inertial family of the load basis): each is the mass matrix
// times a rigid acceleration field, P = Mgg·a (d'Alembert: the static load equivalent to the field of GravityLoad,
// ACCEL, AccelNodeLoad, RotationalForceLoad), with Mgg the assembled mass (PARAM WTMASS applied):
// - GravityLoad: a uniform translational acceleration A·N on every grid point;
// - AccelNodeLoad: the uniform acceleration A·N on the grid points the card lists only;
// - ACCEL: the acceleration N times a factor that varies linearly with the coordinate DIR of the grid point in the
//   rectangular system CID, interpolated in the card's (LOCi, VALi) table, the end values beyond its range;
// - RotationalForceLoad (METHOD = 1): the centrifugal and angular acceleration loads of a rotation about the axis R
// through grid
//   point G (the origin when 0), ω = 2π·A and α = 2π·RACC (A in revolutions per unit time): the acceleration of a point
//   at ρ from the axis point is ω×(ω×ρ) + α×ρ, and its rotations accelerate by α; the load is −Mgg·a (the force of
//   the field on the mass: outward for the centrifugal part).
// The rows come from O_o_Reader's inertialLoadsTable (the vectors in the basic system with the LOAD factors
// applied); without that table, from the GravityLoad rows of loadsTable (the acceleration in the basic system, scaled),
// and an RotationalForceLoad, ACCEL or AccelNodeLoad is refused naming the missing table.
//
// Refused with a diagnostic: RotationalForceLoad METHOD = 2 (the element-based form for coupled masses; METHOD = 1 is
// the one computed), an ACCEL in a cylindrical or spherical CID or with a DIR other than X, Y, Z, an ACCEL table of
// fewer than two points, an unknown grid point in AccelNodeLoad or RotationalForceLoad G, a coordinate system the model
// does not have.
//
// The rows of each set are found through LoadSetIndex (a deck of thousands of load sets reads each set several times),
// the card columns read once.
//
// Thread-safety: const methods are safe concurrently. Ownership: owns the index of inertialLoadsTable; borrows the
// snapshot, the DOF map and the index of loadsTable (LoadVectorBuilder's), which must outlive it.
class InertialLoads
{
public:
    // `loadRows` the rows of loadsTable by set, `loadCards` its card column (empty without load tables).
    InertialLoads(const input::ModelSnapshot& snapshot, const DofMap& dofMap, const LoadSetIndex& loadRows,
                  base::DictionaryView loadCards, base::MemoryBudget& budget);

    // Whether the set holds an inertial load (and so needs the mass matrix), through its terms
    // (input::ModelSnapshot::loadTermsOf).
    [[nodiscard]] bool hasLoads(base::EntityId loadSetId) const;
    // Whether the tables list inertial rows under the SID itself (its own cards, not those of its terms).
    [[nodiscard]] bool holdsOwnRows(base::EntityId setId) const;

    // Adds the inertial loads of the set to `loadsG` (one value per g-set DOF), every row in table order. Throws
    // std::invalid_argument when the size of `loadsG` is not the g-set size.
    void addLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix& mass, std::span<double> loadsG,
                  base::MemoryBudget& budget, base::DiagnosticSink& sink) const;

    // Adds the acceleration fields of the set, a = Σ_rows factor·field (one value per g-set DOF), the vector the loads
    // are Mgg·a of: an inertial load is linear in the mass, so the loads of a mass derivative are ∂P/∂v = (∂Mgg/∂v)·a,
    // formed element by element on the elements a designed field reaches (the design sensitivities). The
    // diagnostics and the size check are those of addLoads.
    void addAccelerations(base::EntityId loadSetId, std::span<double> accelerationsG, base::MemoryBudget& budget,
                          base::DiagnosticSink& sink) const;

    // The terms of the set's inertial rows (InertialTerm), in table order: the rows addLoads applies, with the same
    // diagnostics for those it does not.
    [[nodiscard]] std::vector<InertialTerm> termsOf(base::EntityId loadSetId, base::DiagnosticSink& sink) const;

private:
    // What is done with the field of one row: Mgg·field added to the loads, or the field itself (addAccelerations).
    using FieldApplier = std::function<void(std::span<const double> field, double factor)>;

    void addFromInertialTable(base::EntityId loadSetId, const FieldApplier& apply, std::size_t gSize,
                              base::MemoryBudget& budget, base::DiagnosticSink& sink) const;
    void addFromLoadTable(base::EntityId loadSetId, const FieldApplier& apply, std::size_t gSize,
                          base::MemoryBudget& budget) const;

    // One row read and checked: what its field is made of.
    struct RowSpec
    {
        enum class Kind : std::uint8_t
        {
            Gravity, // GravityLoad: `direction` on every grid point
            Region,  // AccelNodeLoad: `direction` on `grids`
            Varying, // ACCEL: `direction` times the table's factor at the coordinate along the axis in `frame`
            Rotation // RotationalForceLoad: ω×(ω×ρ) + α×ρ about `centre`, α on the rotations, times `factor`
        };
        Kind kind = Kind::Gravity;
        base::Vec3 direction{};          // the acceleration vector; RotationalForceLoad: its axis as given
        std::vector<std::int32_t> grids; // AccelNodeLoad: the grid rows (bounded by a card)
        input::CoordinateFrame frame;    // ACCEL: CID (the basic system by default)
        std::size_t axisIndex = 0;       // ACCEL: DIR
        std::span<const double> locations;
        std::span<const double> factors;
        base::Vec3 centre{};   // RotationalForceLoad
        double velocity = 0.0; // RotationalForceLoad: A (revolutions per unit time)
        double angularAcceleration = 0.0;
        double factor = 1.0; // the factor of the field (RFORCE: −the LOAD factors, the force on the mass)
    };
    // The set's rows of inertialLoadsTable (their problems reported, the rows not applied left out), or its GravityLoad
    // rows of loadsTable.
    [[nodiscard]] std::vector<RowSpec> inertialRowsOf(base::EntityId loadSetId, base::DiagnosticSink& sink) const;
    [[nodiscard]] std::vector<RowSpec> gravityRowsOf(base::EntityId loadSetId) const;
    // The field of a row (factor 1), added to `field` (the g set) in the CD systems.
    void writeField(const RowSpec& spec, std::span<double> field) const;

    const input::ModelSnapshot* m_snapshot;
    const DofMap* m_dofMap;
    const LoadSetIndex* m_loadRows;
    base::DictionaryView m_loadCards;
    LoadSetIndex m_inertialRows; // the rows of inertialLoadsTable by set (empty without it)
    base::DictionaryView m_inertialCards;
};

} // namespace oos::assembly
