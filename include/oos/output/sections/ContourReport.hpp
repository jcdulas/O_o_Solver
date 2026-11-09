#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/output/sections/ZoneDefinition.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oos::output
{

// One contour point of a zone in its report: its grid point, the components through which the rest of the model acts
// on it (in its displacement system CD), whether it is a grid point of an element a fastener pierces, and the causes.
struct ContourPointReport
{
    std::int32_t grid = 0;
    ComponentMask components = 0;
    bool isPiercedPoint = false;
    std::vector<ContourCause> causes;
};

// What a zone did with one of its enclosed elements: "added" (a 0D element on one grid point), "held" (a rigid element
// whose grid points are all in the zone), "included" (enclosed="include"), "kept outside" (enclosed="warn": listed and
// warned), "excluded" (enclosed="exclude").
struct EnclosedDecision
{
    std::int32_t elementId = 0;
    std::string card;
    std::string decision;
};

// One constraint a zone holds: a grid point constrained in every subcase (interior: the local model is held there as
// the global one), with the components constrained in every subcase.
struct HeldConstraint
{
    std::int32_t grid = 0;
    ComponentMask components = 0;
};

// The contour report of a zone (also a result table, which a post-processor can show as it shows the mechanism
// shapes): per contour point its components and their causes, the enclosed elements and what was done with them, the
// counts of contour and interior grid points, the constraints it holds, and the warnings (enclosed elements kept
// outside).
//
// Thread-safety: immutable once built. Ownership: owns its rows.
struct ContourReport
{
    std::int32_t zoneId = 0;
    std::string name;
    std::int64_t unit = 0;
    std::vector<std::int32_t> elements;      // the members, sorted (elements and rigid elements)
    std::vector<ContourPointReport> contour; // by increasing grid point
    std::vector<std::int32_t> interior;      // increasing
    std::vector<EnclosedDecision> enclosed;
    std::vector<HeldConstraint> constraints;
    std::vector<base::Diagnostic> diagnostics;

    [[nodiscard]] std::size_t contourCount() const noexcept { return contour.size(); }
    [[nodiscard]] std::size_t interiorCount() const noexcept { return interior.size(); }
    // The report as text: a title, the counts, one line per contour point (grid, components, causes), the enclosed
    // elements and the warnings. The rows hold neutral card types ("ShellQuad4"); the text shows their deck keywords.
    [[nodiscard]] std::string text() const;
};

} // namespace oos::output
