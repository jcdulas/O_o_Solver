#include "oos/elements/mixedHybrid/families/MixedHybridShellFamily.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/elements/mixedHybrid/families/LocalMap.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"
#include "oos/elements/shells/ShellPressure.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellTemperature.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The model's shell of a batch row, which check() accepted (a row it refused is a bug of the caller).
const MixedShell& requiredShell(const MixedHybridModel& model, const ElementBatchView& view, std::int64_t row)
{
    const MixedShell* shell = model.shellOf(view.card(), row);
    if (shell == nullptr)
        throw std::logic_error(
            base::concatenate({"MixedHybridShellFamily: element ", std::to_string(view.elementId(row)), " of ",
                               view.card(), " was not accepted by check() (a bug of O_o_Solver: report it)"}));
    return *shell;
}

// What the facets ask of an element beside the operators of the stress recovery: the stiffness, the mass, or
// neither (the recovery, the thermal loads and the geometric stiffness).
constexpr ShellMatrixRequest WITH_STIFFNESS{true, false};
constexpr ShellMatrixRequest WITH_MASS{false, true};
constexpr ShellMatrixRequest RECOVERY_OPERATORS{false, false};

// The element of a shell with its matrices (those `request` asks).
ShellElementData shellElementOf(const MixedHybridModel& model, const MixedShell& shell,
                                const ShellMatrixRequest& request)
{
    std::variant<ShellElementData, std::string> element =
        model.shellPart()->elementWithMatrices(shell.partIndex, request);
    // check() builds the matrices of the elements the model keeps; in a larger model it checks the section and the
    // geometry of the others, and their matrices may refuse them here.
    if (const std::string* message = std::get_if<std::string>(&element))
        throw base::SolverError(
            base::concatenate({"shell element ", std::to_string(shell.elementId), " of ", MIXED_HYBRID_FORMULATION,
                               ": ", *message, "; its geometry or its section makes its matrices singular: check "
                                               "its grid points, its thickness and its material"}));
    return std::get<ShellElementData>(std::move(element));
}

// The map of the local unknowns of a shell to its DOFs.
LocalMap shellMapOf(const ElementBatchView& view, const MixedHybridModel& model, const MixedShell& shell)
{
    const ShellPart& part = *model.shellPart();
    return localMapOf(view, model.partDofs(part.localUnknowns(shell.partIndex), part.localSigns(shell.partIndex)));
}

// The iterations of the search of the point of the patch of a quadrilateral's corners closest to a position, and the
// step in the reference coordinates below which it stops (Gauss-Newton converges quadratically from the centre on an
// element that the formulation accepts: a few iterations).
constexpr int CLOSEST_POINT_ITERATIONS = 30;
constexpr double CLOSEST_POINT_STEP = 1.0e-13;

// The weights of the corners of a shell at the point of the patch of its corner positions X_k closest to the position
// x, for the temperatures the card's other families give at the corners (interpolated by their shape functions): on a
// triangle the barycentric coordinates of x projected on its plane; on a quadrilateral the bilinear functions
// N_k = (1 + r r_k)(1 + s s_k) / 4 at the (r, s) that minimise |Σ N_k X_k − x| (Gauss-Newton from the centre: the
// patch of a warped element is curved, and the formulation reads the temperatures at the points of its own surface).
std::array<double, 4> cornerWeightsAt(const std::array<SpaceVector, 4>& corners, std::size_t cornerCount,
                                      const SpaceVector& x)
{
    const auto dot = [](const SpaceVector& a, const SpaceVector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto minus = [](const SpaceVector& a, const SpaceVector& b)
    { return SpaceVector{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
    // The solution of the 2 × 2 normal equations [aa ab; ab bb] [u v] = [ra rb].
    const auto solve = [](double aa, double ab, double bb, double ra, double rb)
    {
        const double determinant = aa * bb - ab * ab;
        return std::array<double, 2>{(bb * ra - ab * rb) / determinant, (aa * rb - ab * ra) / determinant};
    };
    if (cornerCount == 3)
    {
        const SpaceVector e1 = minus(corners[1], corners[0]);
        const SpaceVector e2 = minus(corners[2], corners[0]);
        const SpaceVector d = minus(x, corners[0]);
        const std::array<double, 2> l = solve(dot(e1, e1), dot(e1, e2), dot(e2, e2), dot(e1, d), dot(e2, d));
        return {1.0 - l[0] - l[1], l[0], l[1], 0.0};
    }
    double r = 0.0;
    double s = 0.0;
    const auto weights = [&](double rr, double ss)
    {
        std::array<double, 4> n{};
        for (std::size_t k = 0; k < 4; ++k)
            n[k] = 0.25 * (1.0 + REFERENCE_SQUARE[k][0] * rr) * (1.0 + REFERENCE_SQUARE[k][1] * ss);
        return n;
    };
    for (int iteration = 0; iteration < CLOSEST_POINT_ITERATIONS; ++iteration)
    {
        SpaceVector residual{-x[0], -x[1], -x[2]};
        SpaceVector alongR{};
        SpaceVector alongS{};
        const std::array<double, 4> n = weights(r, s);
        for (std::size_t k = 0; k < 4; ++k)
        {
            const double dr = 0.25 * REFERENCE_SQUARE[k][0] * (1.0 + REFERENCE_SQUARE[k][1] * s);
            const double ds = 0.25 * REFERENCE_SQUARE[k][1] * (1.0 + REFERENCE_SQUARE[k][0] * r);
            for (std::size_t c = 0; c < 3; ++c)
            {
                residual[c] += n[k] * corners[k][c];
                alongR[c] += dr * corners[k][c];
                alongS[c] += ds * corners[k][c];
            }
        }
        const std::array<double, 2> step = solve(dot(alongR, alongR), dot(alongR, alongS), dot(alongS, alongS),
                                                 -dot(alongR, residual), -dot(alongS, residual));
        r += step[0];
        s += step[1];
        if (std::abs(step[0]) + std::abs(step[1]) < CLOSEST_POINT_STEP)
            break;
    }
    return weights(r, s);
}

// The corner positions of a shell (basic system).
std::array<SpaceVector, 4> cornerPositionsOf(const ShellPart& part, const MixedShell& shell)
{
    std::array<SpaceVector, 4> corners{};
    for (std::size_t k = 0; k < shell.cornerCount; ++k)
        corners[k] = part.input().grids[shell.part.corners.at(k)].position;
    return corners;
}

// The temperatures of a shell in a case as the formulation reads them (T0 the change at the reference surface, T1 the
// gradient through the thickness, at a position), from the change the card's other families read
// (resolveShellTemperature: uniform or at the corners), and the thermal terms of the element; empty without
// temperatures or with a zero change.
struct ShellCaseTemperatures
{
    TemperatureAtPosition at;
    std::optional<ShellThermalLoad> load;
};

ShellCaseTemperatures caseTemperaturesOf(const ElementBatchView& view, std::int64_t row, const ShellPart& part,
                                         const MixedShell& shell, const ShellElementData& element,
                                         const ElementTemperatures* temperatures)
{
    ShellCaseTemperatures result;
    if (temperatures == nullptr)
        return result;
    const std::span<const std::int32_t> cornerRows(shell.gridRows.data(), shell.cornerCount);
    const ShellTemperatureChange change =
        resolveShellTemperature(view, row, *temperatures, cornerRows, shell.classicSection).required();
    if (change.isZero())
        return result;
    if (change.hasCornerValues)
    {
        const std::array<SpaceVector, 4> corners = cornerPositionsOf(part, shell);
        const std::size_t count = shell.cornerCount;
        result.at = [corners, count, change](const SpaceVector& x)
        {
            const std::array<double, 4> weights = cornerWeightsAt(corners, count, x);
            double rise = 0.0;
            for (std::size_t k = 0; k < count; ++k)
                rise += weights[k] * change.atCorners[k];
            return std::array<double, 2>{rise, change.gradient};
        };
    }
    else
    {
        const std::array<double, 2> uniform{change.deltaT, change.gradient};
        result.at = [uniform](const SpaceVector&) { return uniform; };
    }
    result.load = part.thermalLoad(element, result.at);
    return result;
}

// The mass of a non-structural mass m per unit area of the surface of the grid points (the NSM cards): m ∫ uᵀ u dA, u
// the displacement of the local unknowns there (pointKinematics), on the rule of the element (the rule of its own
// mass).
Matrix addedMassOf(const ShellPart& part, const ShellElementData& element, double massPerArea)
{
    const ShellReferenceElement& re = part.referenceElement(element);
    const ShellElementDefinition definition = element.definition();
    const std::size_t n = element.matrices.unknownCount;
    Matrix mass(n, n);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(definition, re.h1Count, q);
        const Matrix u = pointKinematics(re, definition, q, g, n).first;
        const double weight = massPerArea * re.weights[q] * g.j;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                mass(i, j) += weight * (u(0, i) * u(0, j) + u(1, i) * u(1, j) + u(2, i) * u(2, j));
    }
    return mass;
}

// The volume of a shell, ∫ t μ0 dA on the rule of the element (the rule of its mass): t the thickness of the section
// at the point, μ0 the area of the reference surface of an offset section over that of the grid points (offsetArea; 1
// without an offset). It is what the translational inertia of the element is the density times (the I0 of
// sectionAtGeometry without the non-structural mass), and a sum over the points of a symmetric rule: the corner the
// element starts from changes only the order of its terms.
double shellVolumeOf(const ShellPart& part, const MixedShell& shell)
{
    const ShellElementData element = part.element(shell.partIndex);
    const ShellReferenceElement& re = part.referenceElement(element);
    const ShellElementDefinition definition = element.definition();
    double volume = 0.0;
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(definition, re.h1Count, q);
        const double thickness = definition.section->thicknessAt(re.points[q][0], re.points[q][1]);
        const double areaRatio = definition.offset == 0.0 ? 1.0 : offsetArea(g, definition.offset);
        volume += re.weights[q] * g.j * thickness * areaRatio;
    }
    return volume;
}

// The components (11, 22, 12) of a plane symmetric tensor in the frame turned by the angle φ (cosine c, sine s) about
// the normal: t'11 = c² t11 + s² t22 + 2cs t12, t'22 = s² t11 + c² t22 − 2cs t12, t'12 = cs (t22 − t11) + (c² − s²)
// t12. `engineering`: the third component is the engineering shear 2 t12.
std::array<double, 3> turnedTensor(const std::array<double, 3>& t, double c, double s, bool engineering)
{
    const double shear = engineering ? 0.5 * t[2] : t[2];
    const double t11 = c * c * t[0] + s * s * t[1] + 2.0 * c * s * shear;
    const double t22 = s * s * t[0] + c * c * t[1] - 2.0 * c * s * shear;
    const double t12 = c * s * (t[1] - t[0]) + (c * c - s * s) * shear;
    return {t11, t22, engineering ? 2.0 * t12 : t12};
}

// The state of a shell at one place in the output frame of the card's other families (spec section 10.2), and the
// angle φ of that frame from the local frame of the element (e1, e2) there: its x axis the side G1-G2 projected on
// the tangent plane, e1' = unit(side − (side · n) n) = cos φ e1 + sin φ e2. The formulation's curvature has the sign
// of ε(z) = e + z κ, the output's that of ε(z) = ε_m − z κ (ShellRecovery.hpp): κ_out = −κ. The moments are ∫σ z dz
// in both, the shear forces ∫τ dz on the faces of normal e1', e2'.
struct OutputPlace
{
    ShellPlaceState state;
    double angle = 0.0; // φ
};

OutputPlace outputPlaceOf(const ShellPointState& point, const SpaceVector& side, std::int32_t nodeId)
{
    const SpaceVector& n = point.normal;
    const double along = side[0] * n[0] + side[1] * n[1] + side[2] * n[2];
    const SpaceVector projected{side[0] - along * n[0], side[1] - along * n[1], side[2] - along * n[2]};
    const SpaceVector& e1 = point.frame[0];
    const SpaceVector& e2 = point.frame[1];
    const double x1 = projected[0] * e1[0] + projected[1] * e1[1] + projected[2] * e1[2];
    const double x2 = projected[0] * e2[0] + projected[1] * e2[1] + projected[2] * e2[2];
    const double angle = std::atan2(x2, x1);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const std::array<double, 8>& r = point.resultants;
    const std::array<double, 8>& e = point.strains;
    OutputPlace place;
    place.angle = angle;
    ShellPlaceState& state = place.state;
    state.nodeId = nodeId;
    state.thickness = point.thickness;
    state.membraneStrain = turnedTensor({e[0], e[1], e[2]}, c, s, true);
    const std::array<double, 3> curvature = turnedTensor({e[3], e[4], e[5]}, c, s, true);
    state.curvature = {-curvature[0], -curvature[1], -curvature[2]};
    state.membraneForce = turnedTensor({r[0], r[1], r[2]}, c, s, false);
    state.moment = turnedTensor({r[3], r[4], r[5]}, c, s, false);
    state.shearForce = {c * r[6] + s * r[7], -s * r[6] + c * r[7]};
    state.temperature.deltaT = point.thermal[0];
    state.temperature.gradient = point.thermal[1];
    return place;
}

// The places of a shell for its displacements (its centre, then its corners in the order of the card) in the output
// frame, and its section in that frame at the centre (the ply angles from the output x axis).
struct OutputPlaces
{
    std::vector<ShellPlaceState> states;
    elements::ShellSection section;
};

OutputPlaces outputPlacesOf(const ElementBatchView& view, const MixedHybridModel& model, const MixedShell& shell,
                            const NodalDisplacements& displacements)
{
    const ShellPart& part = *model.shellPart();
    // PERFORMANCE: the recovery reads the element at its centre and its corners, not at the points of its rule: a
    // case without temperatures (whose thermal load is integrated on the rule) takes the element as the part keeps
    // it, without building the tables of its points nor copying its matrices, once per case (each mode of an
    // eigenvalue analysis).
    const std::shared_ptr<const ShellElementData> kept =
        displacements.temperatures == nullptr ? part.keptElement(shell.partIndex) : nullptr;
    const std::optional<ShellElementData> built =
        kept ? std::nullopt : std::optional<ShellElementData>(shellElementOf(model, shell, RECOVERY_OPERATORS));
    const ShellElementData& element = kept ? *kept : *built;
    const Vector x = localUnknowns(shellMapOf(view, model, shell), displacements);
    const ShellCaseTemperatures temperatures =
        caseTemperaturesOf(view, shell.batchRow, part, shell, element, displacements.temperatures);
    const ShellRecovery recovery =
        part.recover(element, x, temperatures.load ? &temperatures.load->momentTerm : nullptr, temperatures.at, {});
    const SpaceVector& g1 = part.input().grids[shell.part.corners.at(0)].position;
    const SpaceVector& g2 = part.input().grids[shell.part.corners.at(1)].position;
    const SpaceVector side{g2[0] - g1[0], g2[1] - g1[1], g2[2] - g1[2]};
    OutputPlaces places;
    const OutputPlace centre = outputPlaceOf(recovery.centre.state, side, 0);
    places.states.push_back(centre.state);
    for (std::size_t c = 0; c < recovery.corners.size(); ++c)
        places.states.push_back(
            outputPlaceOf(recovery.corners[c].state, side, part.input().grids[shell.part.corners.at(c)].id).state);
    // ShellPointState::angle is the material axis from e1: from the output x axis it is angle − φ.
    places.section = shell.classicSection.rotatedToElementFrame(recovery.centre.state.angle - centre.angle);
    return places;
}

} // namespace

MixedHybridShellFamily::MixedHybridShellFamily(std::unique_ptr<ElementFamily> classic) : m_classic(std::move(classic))
{
}

void MixedHybridShellFamily::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    if (view.rowCount() == 0)
        return;
    if (view.entities() == nullptr)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "formulation-layout-missing";
        diagnostic.card = view.card();
        diagnostic.message = base::concatenate(
            {"expected the ", view.card(), " elements of ", MIXED_HYBRID_FORMULATION,
             " to be computed with the entities of the model's edges, found an element view without them (a path "
             "that does not carry the formulation yet: run the model on the full path)"});
        sink.report(std::move(diagnostic));
        return;
    }
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (base::Diagnostic& problem : model.elementProblems(view.card()))
        sink.report(std::move(problem));
    // Every row read: the ShellPart refused the whole mesh when it is not usable (the model's problems say why).
    if (model.shellPart() == nullptr || !model.shellPart()->usable())
        return;
}

void MixedHybridShellFamily::computeStiffness(const ElementBatchView& view, ElementRange range,
                                              ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedShell& shell = requiredShell(model, view, row);
        const ShellElementData element = shellElementOf(model, shell, WITH_STIFFNESS);
        const LocalMap map = shellMapOf(view, model, shell);
        block.append(row, map.dofs, congruent(element.matrices.stiffness, map));
    }
}

void MixedHybridShellFamily::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                         ElementMatrixBlock& block) const
{
    // The consistent mass whatever options.formulation asks (spec section 8), with the mass per area that the NSM
    // cards add (that of the property is in the section).
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedShell& shell = requiredShell(model, view, row);
        const ShellElementData element = shellElementOf(model, shell, WITH_MASS);
        if (!element.matrices.mass)
            throw std::logic_error("MixedHybridShellFamily: the ShellPart was built without the masses");
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        Matrix mass = *element.matrices.mass;
        if (added != 0.0)
            mass += addedMassOf(*model.shellPart(), element, added);
        const LocalMap map = shellMapOf(view, model, shell);
        block.append(row, map.dofs, congruent(mass, map));
    }
}

bool MixedHybridShellFamily::computeVolumes(const ElementBatchView& view, ElementRange range,
                                            std::span<double> volumes) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
        volumes[static_cast<std::size_t>(row - range.first)] =
            shellVolumeOf(*model.shellPart(), requiredShell(model, view, row));
    return true;
}

void MixedHybridShellFamily::checkPressureLoads(const ElementBatchView& view,
                                                std::span<const ElementPressure> pressures,
                                                base::DiagnosticSink& sink) const
{
    checkShellPressures(view, pressures, sink);
}

void MixedHybridShellFamily::computePressureLoads(const ElementBatchView& view,
                                                  std::span<const ElementPressure> pressures,
                                                  ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const ShellPart& part = *model.shellPart();
    for (const ElementPressure& pressure : pressures)
    {
        const MixedShell& shell = requiredShell(model, view, pressure.elementRow);
        // The load of a pressure reads the tables of the element, not its matrices (which take most of its build).
        const ShellElementData element = part.element(shell.partIndex);
        Vector load;
        if (!pressure.hasDirection)
            load = part.pressureLoad(element, pressure.cornerPressures);
        else
        {
            // The unit direction in the basic system (the element frame of pressureDirectionOf taken as the basic one).
            const PressureDirection direction = pressureDirectionOf(view, pressure, base::Mat3::identity());
            if (!direction.inElementFrame)
                throw std::invalid_argument(base::concatenate(
                    {"MixedHybridShellFamily: ", direction.problem, " (checkPressureLoads() refuses it)"}));
            const base::Vec3 unit = *direction.inElementFrame;
            const std::size_t count = shell.cornerCount;
            load = part.areaForceLoad(element,
                                      [&](const ShellLoadPoint& point)
                                      {
                                          const double p =
                                              cornerInterpolation(count, pressure.cornerPressures, point.xi, point.eta);
                                          return SpaceVector{p * unit[0], p * unit[1], p * unit[2]};
                                      });
        }
        const LocalMap map = shellMapOf(view, model, shell);
        block.append(pressure.elementRow, map.dofs, onDofs(load, map));
    }
}

void MixedHybridShellFamily::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                               base::DiagnosticSink& sink) const
{
    // The temperatures are read as the card's other families read them: their checks apply.
    m_classic->checkThermalLoads(view, temperatures, sink);
}

void MixedHybridShellFamily::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                                 const ElementTemperatures& temperatures,
                                                 ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const ShellPart& part = *model.shellPart();
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedShell& shell = requiredShell(model, view, row);
        const ShellElementData element = shellElementOf(model, shell, RECOVERY_OPERATORS);
        const ShellCaseTemperatures caseTemperatures =
            caseTemperaturesOf(view, row, part, shell, element, &temperatures);
        if (!caseTemperatures.load)
            continue;
        const LocalMap map = shellMapOf(view, model, shell);
        block.append(row, map.dofs, onDofs(caseTemperatures.load->forces, map));
    }
}

void MixedHybridShellFamily::checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    static_cast<void>(view);
    static_cast<void>(sink);
}

void MixedHybridShellFamily::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                          const NodalDisplacements& preload,
                                                          ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const ShellPart& part = *model.shellPart();
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedShell& shell = requiredShell(model, view, row);
        const ShellElementData element = shellElementOf(model, shell, RECOVERY_OPERATORS);
        const LocalMap map = shellMapOf(view, model, shell);
        const Vector x = localUnknowns(map, preload);
        const ShellCaseTemperatures temperatures =
            caseTemperaturesOf(view, row, part, shell, element, preload.temperatures);
        const Vector* momentTerm = temperatures.load ? &temperatures.load->momentTerm : nullptr;
        const Vector* membraneTerm = temperatures.load ? &temperatures.load->membraneTerm : nullptr;
        // The pressure of the preload does not follow the surface (spec section 7: phase 6).
        block.append(row, map.dofs,
                     congruent(part.geometricStiffness(element, x, momentTerm, membraneTerm, nullptr), map));
    }
}

bool MixedHybridShellFamily::differentialStiffnessPreloadGradient(const ElementBatchView& view, std::int64_t row,
                                                                  const NodalDisplacements& shape,
                                                                  ElementVectorBlock& block) const
{
    // K_e(u) = Tᵀ K(T u) T (computeDifferentialStiffness), so φ_eᵀ K_e(e_k) φ_e = (T e_k) · ∇_x(φᵀ K(x) φ) at
    // φ = T φ_e.
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const MixedShell& shell = requiredShell(model, view, row);
    const ShellElementData element = shellElementOf(model, shell, RECOVERY_OPERATORS);
    const LocalMap map = shellMapOf(view, model, shell);
    const Vector phi = localUnknowns(map, shape);
    block.append(row, map.dofs, onDofs(model.shellPart()->geometricStiffnessGradient(element, phi), map));
    return true;
}

void MixedHybridShellFamily::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                     const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    block.reset(outputColumns(output));
    if (block.columns().empty())
        return;
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const OutputPlaces places = outputPlacesOf(view, model, requiredShell(model, view, row), displacements);
        for (const ShellPlaceState& state : places.states)
            appendShellRows(block, row, view.elementId(row), state, places.section, output, composite);
    }
}

void MixedHybridShellFamily::recoverLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                           const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    block.reset(layerOutputColumns(output));
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    const CompositeOutput composite = compositeOutputOf(view.snapshot());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedShell& shell = requiredShell(model, view, row);
        if (!shell.classicSection.isLaminate())
            continue;
        const OutputPlaces places = outputPlacesOf(view, model, shell, displacements);
        appendLaminateRows(block, row, view.elementId(row), places.states.front(), places.section, output, composite);
    }
}

} // namespace oos::elements::mixedHybrid
