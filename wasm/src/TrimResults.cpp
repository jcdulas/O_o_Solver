#include "TrimResults.hpp"

#include <emscripten/bind.h>

#include <span>

namespace oos::wasm
{
namespace
{
using emscripten::val;

template <typename T>
val array(const char* type, std::span<const T> values)
{
    auto result = val::global(type).new_(double(values.size()));
    result.call<void>("set", val(emscripten::typed_memory_view(values.size(), values.data())));
    return result;
}

template <typename T>
val numbers(const T& values)
{
    return array<double>("Float64Array", {values.data(), values.size()});
}

val derivative(const solutions::StabilityDerivativeRow& row)
{
    auto result = val::object();
    result.set("label", row.label);
    result.set("rigidUnsplined", numbers(row.rigidUnsplined));
    result.set("rigidSplined", numbers(row.rigidSplined));
    result.set("elasticRestrained", numbers(row.elasticRestrained));
    result.set("elasticUnrestrained", numbers(row.elasticUnrestrained));
    result.set("inertialRestrained", numbers(row.inertialRestrained));
    result.set("inertialUnrestrained", numbers(row.inertialUnrestrained));
    return result;
}

val monitorDefinition(const std::string& name, const std::string& label, std::int32_t positionFrame,
                      std::int32_t displacementFrame, const base::Vec3& point)
{
    auto item = val::object();
    item.set("name", name);
    item.set("label", label);
    item.set("positionFrameId", positionFrame);
    item.set("displacementFrameId", displacementFrame);
    item.set("point", numbers(point.components));
    return item;
}
} // namespace

val trimResultsObject(const solutions::TrimResults& results)
{
    auto result = val::object();
    result.set("boxIds", array<std::int32_t>("Int32Array", results.boxIds));
    auto surfaces = val::array();
    for (const auto& name : results.surfaces)
        surfaces.call<void>("push", name);
    result.set("surfaces", surfaces);
    result.set("referenceChord", results.referenceChord);
    result.set("referenceSpan", results.referenceSpan);
    result.set("referenceArea", results.referenceArea);
    result.set("symmetryXz", results.symmetryXz);
    result.set("symmetryXy", results.symmetryXy);
    result.set("configuration", results.configuration);
    result.set("basicToReference", numbers(results.basicToReference.values));
    result.set("referenceOffset", numbers(results.referenceOffset.components));
    auto definitions = val::array();
    for (const auto& surface : results.surfaceDefinitions)
    {
        auto item = val::object();
        item.set("id", surface.id);
        item.set("label", surface.label);
        item.set("referenceChord", surface.referenceChord);
        item.set("referenceArea", surface.referenceArea);
        item.set("positionLower", surface.positionLower);
        item.set("positionUpper", surface.positionUpper);
        item.set("hingeMomentLower", surface.hingeMomentLower ? val(*surface.hingeMomentLower) : val::null());
        item.set("hingeMomentUpper", surface.hingeMomentUpper ? val(*surface.hingeMomentUpper) : val::null());
        definitions.call<void>("push", item);
    }
    result.set("surfaceDefinitions", definitions);
    auto monitorDefinitions = val::array();
    for (const auto& point : results.monitorPoints)
    {
        auto item = monitorDefinition(point.name, point.label, point.positionFrameId, point.displacementFrameId,
                                      point.pointInPositionFrame);
        item.set("component", point.component);
        item.set("isAerodynamic", point.isAerodynamic);
        item.set("pointClass", point.pointClass);
        monitorDefinitions.call<void>("push", item);
    }
    result.set("monitorPoints", monitorDefinitions);
    auto monitorDefinitions3 = val::array();
    for (const auto& point : results.monitorPoints3)
        monitorDefinitions3.call<void>(
            "push", monitorDefinition(point.name, point.label, point.positionFrameId, point.displacementFrameId,
                                      point.pointInPositionFrame));
    result.set("monitorPoints3", monitorDefinitions3);
    auto subcases = val::array();
    for (const auto& source : results.subcases)
    {
        auto item = val::object();
        item.set("subcase", source.subcase);
        item.set("trimSet", source.trimSet);
        item.set("mach", source.mach);
        item.set("dynamicPressure", source.dynamicPressure);
        item.set("hasUnrestrained", source.hasUnrestrained);
        auto variables = val::array();
        constexpr const char* KINDS[]{"rigidBody", "controlSurface", "user", "undefined"};
        constexpr const char* STATES[]{"fixed", "free", "linked", "scheduled"};
        for (const auto& variable : source.variables)
        {
            auto value = val::object();
            value.set("id", variable.id);
            value.set("label", variable.label);
            value.set("kind", std::string(KINDS[static_cast<unsigned>(variable.kind)]));
            value.set("state", std::string(STATES[static_cast<unsigned>(variable.state)]));
            value.set("value", variable.value);
            variables.call<void>("push", value);
        }
        item.set("variables", variables);
        auto derivatives = val::array();
        for (const auto& row : source.derivatives)
            derivatives.call<void>("push", derivative(row));
        item.set("derivatives", derivatives);
        item.set("reference", derivative(source.reference));
        auto hinges = val::array();
        for (const auto& hinge : source.hingeMomentDerivatives)
        {
            auto value = val::object();
            value.set("surface", hinge.surface);
            value.set("atReference", numbers(hinge.atReference));
            std::vector<double> entries;
            for (const auto& row : hinge.perVariable)
                entries.insert(entries.end(), row.begin(), row.end());
            value.set("perVariable", numbers(entries));
            hinges.call<void>("push", value);
        }
        item.set("hingeMomentDerivatives", hinges);
        item.set("pressures", numbers(source.pressures));
        item.set("boxForces", numbers(source.boxForces));
        item.set("boxMoments", numbers(source.boxMoments));
        item.set("hingeMoments", numbers(source.hingeMoments));
        item.set("supportAccelerations", numbers(source.supportAccelerations));
        auto monitors = val::array();
        for (std::size_t i = 0; i < source.monitorLoads.size(); ++i)
        {
            auto value = val::object();
            value.set("name", results.monitorPoints.at(i).name);
            std::vector<double> loads;
            for (const auto& row : source.monitorLoads[i])
                loads.insert(loads.end(), row.begin(), row.end());
            value.set("loads", numbers(loads));
            monitors.call<void>("push", value);
        }
        item.set("monitorLoads", monitors);
        auto monitor3 = val::array();
        for (std::size_t i = 0; i < source.monitor3Loads.size(); ++i)
        {
            auto value = val::object();
            value.set("name", results.monitorPoints3.at(i).name);
            value.set("loads", numbers(source.monitor3Loads[i]));
            monitor3.call<void>("push", value);
        }
        item.set("monitor3Loads", monitor3);
        subcases.call<void>("push", item);
    }
    result.set("subcases", subcases);
    return result;
}
} // namespace oos::wasm
