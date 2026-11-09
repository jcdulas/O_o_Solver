#include "oos/output/sections/ContourReport.hpp"

#include <sstream>

namespace oos::output
{

std::string ContourReport::text() const
{
    std::ostringstream out;
    out << "Contour of zone " << (name.empty() ? std::to_string(zoneId) : name) << " (ID " << zoneId;
    if (unit != 0)
        out << ", unit " << unit;
    out << "): " << elements.size() << " elements, " << contour.size() << " contour grid points, " << interior.size()
        << " interior grid points\n";
    for (const ContourPointReport& point : contour)
    {
        out << "  grid " << point.grid << "  components " << componentDigitsOf(point.components)
            << (point.isPiercedPoint ? "  (grid point of a fastener's pierced element)" : "") << "  causes:";
        for (const ContourCause& cause : point.causes)
        {
            out << " " << cause.kind << " " << std::string(cause.card) << " " << cause.id;
            if (cause.components != 0)
                out << " (" << componentDigitsOf(cause.components) << ")";
            out << ";";
        }
        out << "\n";
    }
    for (const HeldConstraint& constraint : constraints)
        out << "  held constraint: grid " << constraint.grid << " components "
            << componentDigitsOf(constraint.components) << " (in every subcase)\n";
    for (const EnclosedDecision& enclosed : enclosed)
        out << "  enclosed " << std::string(enclosed.card) << " " << enclosed.elementId << ": " << enclosed.decision
            << "\n";
    for (const base::Diagnostic& diagnostic : diagnostics)
        out << "  " << base::formatDiagnostic(diagnostic) << "\n";
    return out.str();
}

} // namespace oos::output
