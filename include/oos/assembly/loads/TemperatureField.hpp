#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/PointIndex.hpp"

#include <optional>

namespace oos::assembly
{

// The temperatures of a subcase with a thermal load (Case Control TEMPERATURE), gathered
// from temperatureTable of O_o_Reader for the set of TEMPERATURE(LOAD) and the set of TEMPERATURE(INITIAL):
// - TEMP: the temperature of a grid point; DefaultTemperatureLoad: the default of the set, for every grid point its
// TEMP cards leave
//   out; a grid point with neither has none (NaN);
// - PlateTemperatureLoad: the mean temperature and the gradient through the thickness of a shell (TEMPP1 TBAR, TPRIME);
//   LineElemTemperatureLoad: the temperature at end A of a bar and its change to end B (as temperatureTable gives
//   them).
// The elements read it through elements::ElementTemperatures (their checkThermalLoads, computeThermalLoads and the
// recovery).
//
// With the element deformations of the subcase's DEFORM (Case Control DEFORM, Bulk Data DEFORM), the field
// holds the subcase's initial strains: the (EID, D) pairs of the DEFORM cards of the set (the deform table of the
// snapshot), sorted by element, and a subcase with a DEFORM and no TEMPERATURE(LOAD) has no temperatures
// (ElementTemperatures::hasTemperatureLoad false). LoadVectorBuilder::checkThermalLoads checks the elements they name.
//
// A row the field cannot use is a Diagnostic: a TEMP on a point the model does not have ("load-unknown-point"), a
// card of temperatureTable it does not know ("load-unsupported"), a DEFORM set without a card ("load-unknown-set"),
// an element a DEFORM set names twice ("load-duplicate": which D applies would be ambiguous).
//
// Thread-safety: immutable once built. Ownership: owns its buffers (budgeted, one value per grid point).
class TemperatureField
{
public:
    // `loadSet` the set of TEMPERATURE(LOAD); `initialSet` the set of TEMPERATURE(INITIAL), if any.
    TemperatureField(const input::ModelSnapshot& snapshot, const input::PointIndex& points, base::EntityId loadSet,
                     std::optional<base::EntityId> initialSet, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    // The initial strains of a subcase: `loadSet` and `initialSet` its TEMPERATURE(LOAD) and (INITIAL), if
    // any (no load set: no temperatures, the initial set is not read), `deformationSet` its DEFORM, if any. Throws
    // std::invalid_argument when neither a load set nor a deformation set is given.
    TemperatureField(const input::ModelSnapshot& snapshot, const input::PointIndex& points,
                     std::optional<base::EntityId> loadSet, std::optional<base::EntityId> initialSet,
                     std::optional<base::EntityId> deformationSet, base::MemoryBudget& budget,
                     base::DiagnosticSink& sink);

    // The view the element facets read; valid as long as the field lives.
    [[nodiscard]] const elements::ElementTemperatures& temperatures() const noexcept { return m_view; }

private:
    struct Set
    {
        base::BudgetedBuffer<double> atGrid;
        base::BudgetedBuffer<elements::ElementTemperature> ofElements; // sorted by element ID
    };

    [[nodiscard]] static Set gather(const input::ModelSnapshot& snapshot, const input::PointIndex& points,
                                    base::EntityId setId, base::MemoryBudget& budget, base::DiagnosticSink& sink);
    [[nodiscard]] static base::BudgetedBuffer<elements::ElementDeformation> gatherDeformations(
        const input::ModelSnapshot& snapshot, base::EntityId setId, base::MemoryBudget& budget,
        base::DiagnosticSink& sink);

    Set m_load;
    Set m_initial;
    base::BudgetedBuffer<elements::ElementDeformation> m_deformations; // sorted by element ID, one per element
    elements::ElementTemperatures m_view;
};

} // namespace oos::assembly
