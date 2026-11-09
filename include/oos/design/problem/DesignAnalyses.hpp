#pragma once

#include "oos/aero/trim/TrimCase.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace oos::design
{

// The analyses of the subcases of a design problem, several analysis types in one problem (the kind of each subcase, as
// the ANALYSIS command of a subcase gives it). Each list names the subcases of one kind; nothing: no subcase of that
// kind; an empty list: every subcase the solution of that kind takes by default (StaticSolution: every subcase;
// ModesSolution: every subcase; BucklingSolution: every subcase with a METHOD). The default is the statics of every
// subcase.
//
// A BUCK subcase has a static preload (its STATSUB, its own loads, else the static subcase before it:
// BucklingSolution): the static subcases of the deck are then solved by the buckling solution, with the outputs the
// responses read.
//
// The SAERO subcases are trimmed by TrimSolution, with the AeroTrimLink of the model (`links`: O_o_Reader gives
// the cards without their fields, the caller reads them, as for SOL 144).
struct DesignAnalyses
{
    std::optional<std::vector<std::int32_t>> statics = std::vector<std::int32_t>{};
    std::optional<std::vector<std::int32_t>> modes;
    std::optional<std::vector<std::int32_t>> buckling;
    std::optional<std::vector<std::int32_t>> aeroelastic;
    std::vector<aero::AeroLink> links;

    [[nodiscard]] bool hasEigenvalues() const noexcept { return modes.has_value() || buckling.has_value(); }
    [[nodiscard]] bool hasAeroelastic() const noexcept { return aeroelastic.has_value(); }
};

// The shape of a mode an eigenvalue response tracks (a mode followed from one design point to the next by its shape),
// by (response ID, subcase): one value per DOF of the g set, any scale (the modal
// assurance criterion is blind to it). The caller keeps those of the previous design cycle.
using ModeReferences = std::map<std::pair<std::int32_t, std::int32_t>, std::vector<double>>;

// The mode an eigenvalue response took in a subcase, and how sure the tracking was (ModeTracking).
struct TrackedMode
{
    std::int32_t responseId = 0;
    std::int32_t subcase = 0;
    std::int32_t requestedMode = 0; // ATTA
    std::int32_t mode = 0;          // the mode taken (1-based)
    // The modal assurance criterion of the mode taken and of the next best against the reference (1 and the largest
    // other when the response had no reference yet: the mode of ATTA).
    double mac = 1.0;
    double secondMac = 0.0;
    std::int32_t secondMode = 0;
    bool hasReference = false;
    bool isUncertain = false;
    // The shape of the mode taken, on the g set (the reference of the next point when this one is recorded).
    std::vector<double> shape;
};

} // namespace oos::design
