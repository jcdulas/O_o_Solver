#include "oos/api/Capabilities.hpp"

#include "oos/design/responses/ResponseDefinitions.hpp"

namespace oos::api
{
OptimizationCapabilities optimizationCapabilities(const input::ModelSnapshot* snapshot)
{
    OptimizationCapabilities result;
    result.analyses = {{"statics",
                        101,
                        true,
                        true,
                        true,
                        {"Second-order products use native gradients when static element blocks are unavailable.",
                         "Shape derivatives require supported derived geometry and constraints."}},
                       {"modes",
                        103,
                        true,
                        true,
                        true,
                        {"Native responses differentiate eigenvalues, frequencies and mode-selected displacement, "
                         "stress, strain and force fields.",
                         "Closed sizing formulations use bordered modal tangents and adjoint derivatives for "
                         "second-order products.",
                         "Other second-order products difference native gradients; usesModalTangents identifies "
                         "the analytic path.",
                         "Modal fields use the full vibration path and the extraction method's normalisation.",
                         "Repeated roots return one-sided coordinate derivatives; they are not a smooth Jacobian.",
                         "Individual modal field derivatives and Hessians are refused at repeated roots.",
                         "Use mode references and inspect tracking uncertainty across design points."}},
                       {"buckling",
                        105,
                        true,
                        true,
                        true,
                        {"Native responses differentiate load factors including the static preload.",
                         "Closed mechanical-preload sizing products use shape and preload tangents; "
                         "usesBucklingTangents identifies this path.",
                         "Other second-order products difference native gradients at tracked buckling points.",
                         "Repeated roots return one-sided coordinate derivatives; they are not a smooth Jacobian.",
                         "Use mode references and inspect tracking uncertainty across design points."}},
                       {"aeroelastic",
                        144,
                        true,
                        true,
                        true,
                        {"Native gradients differentiate structural sizing at fixed aerodynamic geometry.",
                         "Closed sizing products differentiate the coupled trim state and stability coefficients; "
                         "usesTrimTangents identifies this path.",
                         "Other second-order products difference supported native trim gradients.",
                         "Shape columns use complete trim differences to include moved splines and rigid-body modes.",
                         "Native trim linearisation includes applied inertial load mass derivatives on the full path.",
                         "A plan that condenses a driven unit uses the full trim path when gradients are retained; "
                         "applied loads still require the full path."}}};
    for (unsigned value = 0; value <= static_cast<unsigned>(design::ResponseType::StabilityDerivative); ++value)
        result.responses.emplace_back(design::responseTypeName(static_cast<design::ResponseType>(value)));
    if (snapshot == nullptr)
        return result;
    for (const auto& name : snapshot->tableNames())
    {
        const auto slash = name.find('/');
        if (slash == std::string::npos)
            continue;
        const auto prefix = name.substr(0, slash);
        const auto kind = prefix == "properties"  ? "property"
                          : prefix == "materials" ? "material"
                          : prefix == "elements"  ? "element"
                                                  : "";
        if (*kind == '\0')
            continue;
        const auto* reader = snapshot->findTable(name);
        for (const auto& field : reader->columnNames())
        {
            const auto type = reader->columnType(field);
            if (type == "float64" || type == "list<float64>" || type.starts_with("fixed_list<float64>") ||
                type.starts_with("list<fixed_list<float64>"))
                result.fields.push_back({kind, name.substr(slash + 1), field, type, type != "float64"});
        }
    }
    return result;
}
} // namespace oos::api
