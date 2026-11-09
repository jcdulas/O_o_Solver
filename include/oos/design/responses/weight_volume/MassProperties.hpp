#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string_view>

namespace oos::design
{

// Which elements a mass property takes: a predicate on the element's card, ID and property ID (0 without one); an
// empty predicate takes every element.
using ElementFilter = std::function<bool(std::string_view card, std::int32_t elementId, std::int32_t propId)>;
// Which element types a mass property takes, by card (an empty predicate takes every type): the batches it refuses
// are not computed at all.
using CardFilter = std::function<bool(std::string_view card)>;

// The rigid-body mass properties of the elements of a snapshot, for the WEIGHT, VOLUME and FRMASS responses.
//
// rigidBodyMass: MO = Σ_e D_eᵀ·M_e·D_e, the 6 × 6 rigid-body mass matrix about `reference` in the basic system, the
// sum of the element mass matrices M_e (the Mass facet, lumped or coupled as PARAM COUPMASS asks) carried by D_e, the
// rows of the rigid-body modes at the element's DOFs: for a translation component c of a grid point at x with the
// CD-to-basic rotation R (u_basic = R·u_CD), the row c of Rᵀ·[I, −[x − reference]×]; for a rotation, the row c of
// Rᵀ·[0, I]; a scalar point has a zero row. It equals the MO of the grid point weight generator (assembly/relief
// GridPointWeight: Dᵀ·Mgg·D) because Mgg is the sum of the same element matrices; summing element by element needs no
// assembly. The masses are in the units of the deck (before WTMASS, as the weight generator), without the NSM
// cards (as the static analysis, which takes no NSM set: a warning of StaticSolution says so).
//
// volume: Σ of the element volumes, the translational mass of the elements with every density 1 and every
// non-structural mass 0 (an overlay of the snapshot's "rho" and "nsm" columns), without the concentrated masses
// (CONM1, PointMassInertia, MassDofProperty-4: they have a mass and no volume).
//
// Determinism: the sums run batch by batch and element by element in the snapshot order, on one thread.
// Thread-safety: stateless functions. Ownership: none kept.
class MassProperties
{
public:
    // The cards of the concentrated masses, which have no volume.
    [[nodiscard]] static bool isConcentratedMass(std::string_view card) noexcept;

    // MO about `reference` (basic system) of the elements `filter` takes. Throws as the element families'
    // computeMass (a problem of an element) and MemoryBudgetError.
    [[nodiscard]] static base::MatN<6> rigidBodyMass(const input::ModelSnapshot& snapshot, base::Vec3 reference,
                                                     elements::MassFormulation formulation, const ElementFilter& filter,
                                                     base::MemoryBudget& budget);

    // The volume of the elements `filter` takes. `keepAlive` owns `snapshot` (the overlay of densities borrows it).
    [[nodiscard]] static double volume(const input::ModelSnapshot& snapshot,
                                       const std::shared_ptr<const void>& keepAlive, const ElementFilter& filter,
                                       base::MemoryBudget& budget);
    // The volume of each element `filter` takes, by element ID (the strain energy density of ESE item 4: the strain
    // energy table of O_o_Solver leaves its density column empty).
    [[nodiscard]] static std::map<std::int32_t, double> elementVolumes(const input::ModelSnapshot& snapshot,
                                                                       const std::shared_ptr<const void>& keepAlive,
                                                                       const ElementFilter& filter,
                                                                       base::MemoryBudget& budget);
    // The entry (row, column) of MO_e = D_eᵀ·M_e·D_e of each element `filter` takes, by element ID (the sensitivity map
    // of WEIGHT and FRMASS: every element's share at once).
    [[nodiscard]] static std::map<std::int32_t, double> elementRigidBodyMass(
        const input::ModelSnapshot& snapshot, base::Vec3 reference, elements::MassFormulation formulation,
        const ElementFilter& filter, std::size_t row, std::size_t column, base::MemoryBudget& budget);
    // The same, computed for the element types `takesCard` takes only (the shells of the linearity indicator).
    [[nodiscard]] static std::map<std::int32_t, double> elementVolumesOfCards(
        const input::ModelSnapshot& snapshot, const std::shared_ptr<const void>& keepAlive, const CardFilter& takesCard,
        base::MemoryBudget& budget);

    // One element of an assembly::ElementSet: its batch (an index into ElementSet::batches()) and its row.
    struct SetElement
    {
        std::size_t batch = 0;
        std::int64_t row = 0;
    };

    // MO about `reference` of the elements `elements` of `set`, given in increasing (batch, row): the sum rigidBodyMass
    // forms on the snapshot of `set` with a filter that takes exactly these elements, the same terms in the same order
    // (so the same bits), without computing the mass of the others. The families are those of `registry` (the standard
    // one, as rigidBodyMass), whatever registry `set` was made with; the batches of `set` are the snapshot's in order,
    // as rigidBodyMass walks them.
    // PERFORMANCE: the explicit part of WEIGHT or FRMASS for a designed field differences the mass of the elements the
    // field reaches on four perturbed snapshots; rigidBodyMass with a filter computes the mass of every element of the
    // model (and builds an index of the model) four times per field: O(model) per design variable.
    // Measured: the explicit parts of a WEIGHT response for 200 thickness variables on 250 000 DOFs, 1 625 s → 4.1 s,
    // the same bits.
    [[nodiscard]] static base::MatN<6> rigidBodyMassOf(const assembly::ElementSet& set,
                                                       std::span<const SetElement> elements,
                                                       const elements::ElementRegistry& registry, base::Vec3 reference,
                                                       elements::MassFormulation formulation);
    // The volume of the elements `elements` of `set` (increasing (batch, row)), concentrated masses left out: volume()
    // with a filter that takes exactly these elements, the same terms in the same order. `keepAlive` owns the snapshot
    // of `set`.
    [[nodiscard]] static double volumeOf(const assembly::ElementSet& set, const std::shared_ptr<const void>& keepAlive,
                                         std::span<const SetElement> elements,
                                         const elements::ElementRegistry& registry);
};

} // namespace oos::design
