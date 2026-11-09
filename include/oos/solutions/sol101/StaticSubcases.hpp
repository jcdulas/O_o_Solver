#pragma once

#include "oos/assembly/loads/TemperatureField.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/PointIndex.hpp"

#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace oos::solutions
{

// One subcase of a static analysis (Case Control SUBCASE, SUBCOM): the ID its results carry, the row of
// subcasesTable its selections come from (0 for the global section of a deck without SUBCASE), its selections, and
// for a SUBCOM the subcases it combines with their coefficients (SUBSEQ).
struct StaticSubcase
{
    base::EntityId label = 1;
    base::EntityId tableId = 0;
    std::optional<base::EntityId> spc;
    std::optional<base::EntityId> mpc;
    std::optional<base::EntityId> load;
    std::optional<base::EntityId> temperatureLoad;
    std::optional<base::EntityId> temperatureInitial;
    // DEFORM: the element deformation set of the subcase (Case Control DEFORM), an initial strain as the
    // temperatures are (assembly::TemperatureField).
    std::optional<base::EntityId> deformation;
    std::optional<base::EntityId> contact; // BCONTACT: the contact table of the subcase, glued contact
    bool isCombination = false;
    std::vector<base::EntityId> combined; // bounded by the subcases of the deck
    std::vector<double> coefficients;

    // Whether the subcase has initial strains: a thermal load (TEMPERATURE(LOAD)) or element deformations (DEFORM).
    [[nodiscard]] bool hasInitialStrains() const noexcept
    {
        return temperatureLoad.has_value() || deformation.has_value();
    }
    // The selections that make its initial strains, no longer any (a case solved without its loads: a mode).
    void dropInitialStrains() noexcept
    {
        temperatureLoad.reset();
        temperatureInitial.reset();
        deformation.reset();
    }
};

// What makes the initial strains of a subcase: its TEMPERATURE(LOAD), TEMPERATURE(INITIAL) and DEFORM sets.
// Two subcases of the same key have the same initial strains (their thermal and deformation loads, the strains their
// recovery subtracts).
struct InitialStrainKey
{
    std::optional<base::EntityId> temperatureLoad;
    std::optional<base::EntityId> temperatureInitial;
    std::optional<base::EntityId> deformation;

    [[nodiscard]] bool isEmpty() const noexcept { return !temperatureLoad && !deformation; }
    auto operator<=>(const InitialStrainKey&) const = default;
};
[[nodiscard]] InitialStrainKey initialStrainKeyOf(const StaticSubcase& subcase);

// The initial strains of a key (assembly::TemperatureField: its temperatures and its element deformations), or null
// for an empty key. Its problems go to `sink`.
[[nodiscard]] std::unique_ptr<assembly::TemperatureField> initialStrainsOf(const input::ModelSnapshot& snapshot,
                                                                           const input::PointIndex& points,
                                                                           const InitialStrainKey& key,
                                                                           base::MemoryBudget& budget,
                                                                           base::DiagnosticSink& sink);
// The initial strains of a subcase (initialStrainsOf its key), or null without.
[[nodiscard]] std::unique_ptr<assembly::TemperatureField> initialStrainsOf(const input::ModelSnapshot& snapshot,
                                                                           const input::PointIndex& points,
                                                                           const StaticSubcase& subcase,
                                                                           base::MemoryBudget& budget,
                                                                           base::DiagnosticSink& sink);

// The subcases of a static analysis, in the order of the Case Control (SUBCASE, SUBCOM, SUBSEQ):
// - SUBCASE rows are solved (a deck without SUBCASE is solved as subcase 1, its selections those of the global
//   section, subcase 0 of O_o_Reader's subcasesTable);
// - a SUBCOM is the linear combination of the results of the subcases it names with its SUBSEQ coefficients
//   (combined_subcases and coefficients of subcasesTable, as O_o_Reader gives them): its displacements, SPC and MPC
//   forces and applied loads are combined, and its element results recovered from the combined displacements with the
//   temperatures of its own TEMPERATURE(LOAD) and the element deformations of its own DEFORM (in the
//   input format, the thermal strains and element deformations of a combination come from the SUBCOM's own commands,
//   none without them);
// - SYM, SYMCOM and REPCASE are refused with a diagnostic (symmetric half models and repeated output requests are not
//   supported).
// `asked` (the options of the run): the subcases to solve or combine, in that order; empty for every one. The subcases
// a SUBCOM asked for combines are solved with it. An ID the Case Control does not have throws std::invalid_argument.
[[nodiscard]] std::vector<StaticSubcase> staticSubcasesOf(const input::ModelSnapshot& snapshot,
                                                          const std::vector<base::EntityId>& asked,
                                                          base::DiagnosticSink& sink);

} // namespace oos::solutions
