#include "oos/elements/bars_beams/BeamLineLoad.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

struct LoadTypeEntry
{
    std::string_view name;
    bool isMoment;
    bool isInElementSystem;
    std::size_t axis;
};

// The load types of the BeamDistributedLoad field TYPE.
constexpr std::array<LoadTypeEntry, 12> LOAD_TYPES{{{"ForceX", false, false, 0},
                                                    {"ForceY", false, false, 1},
                                                    {"ForceZ", false, false, 2},
                                                    {"ForceXAlongLength", false, true, 0},
                                                    {"ForceYAlongLength", false, true, 1},
                                                    {"ForceZAlongLength", false, true, 2},
                                                    {"MomentX", true, false, 0},
                                                    {"MomentY", true, false, 1},
                                                    {"MomentZ", true, false, 2},
                                                    {"MomentXAlongLength", true, true, 0},
                                                    {"MomentYAlongLength", true, true, 1},
                                                    {"MomentZAlongLength", true, true, 2}}};

base::Diagnostic loadProblem(const ElementLineLoad& load, std::string code, std::string field, std::string_view card,
                             std::int32_t elementId, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(load.card);
    diagnostic.id = load.setId;
    diagnostic.field = std::move(field);
    diagnostic.message = message.append(" (on ").append(card).append(" ").append(std::to_string(elementId)).append(")");
    return diagnostic;
}

// The value at ξ of the load varying linearly from P1 at X1 to P2 at X2.
double valueAt(const BeamLineLoad& load, double position)
{
    return load.startValue + (load.endValue - load.startValue) * (position - load.start) / (load.end - load.start);
}

} // namespace

std::variant<BeamLineLoad, base::Diagnostic> resolveLineLoad(const ElementLineLoad& load, const BeamGeometry& geometry,
                                                             std::string_view elementCard, std::int32_t elementId)
{
    const LoadTypeEntry* type = nullptr;
    for (const LoadTypeEntry& entry : LOAD_TYPES)
    {
        if (entry.name == load.loadType)
            type = &entry;
    }
    if (type == nullptr)
        return loadProblem(
            load, problem_code::INVALID_FIELD, "kind", elementCard, elementId,
            base::concatenate({"expected a load type ForceX, ForceY, ForceZ, ForceXAlongLength, ..., MomentZ, "
                               "MomentXAlongLength, ... (BeamDistributedLoad), found \"",
                               load.loadType, "\"; correct the card"}));
    const bool isProjected = load.scaleType == "ProjectedLength" || load.scaleType == "FractionalProjectedLength";
    const bool isFractional = load.scaleType == "FractionalLength" || load.scaleType == "FractionalProjectedLength";
    if (!isProjected && !isFractional && load.scaleType != "ActualLength")
        return loadProblem(
            load, problem_code::INVALID_FIELD, "scaleFactor", elementCard, elementId,
            base::concatenate({"expected a scale type ActualLength, FractionalLength, ProjectedLength or "
                               "FractionalProjectedLength (BeamDistributedLoad), found \"",
                               load.scaleType, "\"; correct the card"}));
    if (isProjected && type->isInElementSystem)
        return loadProblem(
            load, problem_code::INVALID_FIELD, "scaleFactor", elementCard, elementId,
            base::concatenate(
                {"expected ActualLength or FractionalLength with the element-system type ", load.loadType, ", found ",
                 load.scaleType,
                 " (BeamDistributedLoad: projected loads are along the basic axes only); use ActualLength "
                 "or FractionalLength, or a basic type"}));

    BeamLineLoad resolved;
    resolved.isMoment = type->isMoment;
    base::Vec3 axis;
    axis[type->axis] = 1.0;
    // The element axes are the columns of `axes`: a basic direction in the element system is axesᵀ·e.
    resolved.direction = type->isInElementSystem ? axis : geometry.axes.transposed() * axis;

    const double scale = isFractional ? geometry.length : 1.0;
    const double limit = isFractional ? 1.0 : geometry.length;
    const double tolerance = BEAM_LOAD_POSITION_TOLERANCE * limit;
    if (load.x1 < -tolerance || load.x1 > limit + tolerance)
        return loadProblem(
            load, problem_code::INVALID_FIELD, "xCoord1", elementCard, elementId,
            base::concatenate({"expected BeamDistributedLoad.xCoord1 between 0 and ", std::to_string(limit), " (",
                               load.scaleType, "), found ", std::to_string(load.x1), "; correct the position"}));
    if (load.x2 < load.x1 || load.x2 > limit + tolerance)
        return loadProblem(
            load, problem_code::INVALID_FIELD, "xCoord2", elementCard, elementId,
            base::concatenate({"expected BeamDistributedLoad.xCoord2 between BeamDistributedLoad.xCoord1 (",
                               std::to_string(load.x1), ") and ", std::to_string(limit), " (", load.scaleType,
                               "), found ", std::to_string(load.x2), "; correct the position"}));
    resolved.start = std::clamp(load.x1, 0.0, limit) * scale;
    resolved.end = std::clamp(load.x2, 0.0, limit) * scale;
    resolved.isConcentrated = load.x2 == load.x1;
    double factor = 1.0;
    if (isProjected)
    {
        // The load per unit projected length times |sin α| is the load per unit actual length.
        const base::Vec3 xAxis{1.0, 0.0, 0.0};
        factor = base::norm(base::cross(xAxis, resolved.direction));
    }
    resolved.startValue = load.p1 * factor;
    resolved.endValue = (resolved.isConcentrated ? load.p1 : load.p2) * factor;
    return resolved;
}

LoadResultant loadResultant(const BeamLineLoad& load, double from, double to, double about, bool includeEnds)
{
    const base::Vec3 xAxis{1.0, 0.0, 0.0};
    LoadResultant result;
    if (load.isConcentrated)
    {
        const bool isInside =
            includeEnds ? (load.start >= from && load.start <= to) : (load.start >= from && load.start < to);
        if (!isInside)
            return result;
        if (load.isMoment)
            result.moment = load.startValue * load.direction;
        else
        {
            result.force = load.startValue * load.direction;
            result.moment = ((load.start - about) * load.startValue) * base::cross(xAxis, load.direction);
        }
        return result;
    }
    const double a = std::max(from, load.start);
    const double b = std::min(to, load.end);
    if (b <= a)
        return result;
    // p linear on [a, b]: ∫p = (p(a) + p(b))·(b − a)/2; ∫(ξ − c)·p is a quadratic, integrated exactly by Simpson's
    // rule.
    const double pa = valueAt(load, a);
    const double pb = valueAt(load, b);
    const double middle = 0.5 * (a + b);
    const double pm = valueAt(load, middle);
    const double total = 0.5 * (pa + pb) * (b - a);
    const double firstMoment = (b - a) / 6.0 * ((a - about) * pa + 4.0 * (middle - about) * pm + (b - about) * pb);
    if (load.isMoment)
        result.moment = total * load.direction;
    else
    {
        result.force = total * load.direction;
        result.moment = firstMoment * base::cross(xAxis, load.direction);
    }
    return result;
}

LoadResultant loadResultant(std::span<const BeamLineLoad> loads, double from, double to, double about, bool includeEnds)
{
    LoadResultant sum;
    for (const BeamLineLoad& load : loads)
    {
        const LoadResultant part = loadResultant(load, from, to, about, includeEnds);
        sum.force = sum.force + part.force;
        sum.moment = sum.moment + part.moment;
    }
    return sum;
}

} // namespace oos::elements
