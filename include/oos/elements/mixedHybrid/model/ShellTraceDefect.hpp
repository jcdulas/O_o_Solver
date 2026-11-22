#pragma once

#include "oos/elements/mixedHybrid/model/ShellPart.hpp"

namespace oos::elements
{
struct BodyAcceleration;
}

namespace oos::elements::mixedHybrid
{

// Angular defect of the hybrid trace at a cubic shell side (radians). The side's two
// incident shells can share the new rotation only if both are enriched.
struct ShellEdgeDefect
{
    std::size_t edge;
    double defect;
    double scale;
    std::array<double, 3> membraneWork{}; // force on the additional edge displacement mode, in basic
    double membraneScale = 0.0;
};

// Work of a local correction in director modes available at degree four.
// The reference energy sets a roundoff floor across the loaded part.
struct ShellInteriorDefect
{
    std::size_t shell = 0;
    double work = 0.0;
    double referenceEnergy = 0.0;
};

[[nodiscard]] ShellInteriorDefect directorEquilibrium(const ShellPart& part, std::size_t shell,
                                                      const ShellElementData& element, std::span<const double> x,
                                                      const ShellSolutionCoefficients& coefficients,
                                                      const BodyAcceleration* acceleration);

// The compatibility defect and the membrane work of a cubic solution on each side.
void appendTraceDefects(const ShellPart& part, std::size_t shell, const ShellElementData& element,
                        std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                        const TemperatureAtPosition& temperature, std::vector<ShellEdgeDefect>& defects);

} // namespace oos::elements::mixedHybrid
