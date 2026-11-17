#include "oos/elements/mixedHybrid/families/MixedHybridLineFamily.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/BeamLineLoad.hpp"
#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"
#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"
#include "oos/elements/core/MarginOfSafety.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/families/LocalMap.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The model's line of a batch row, which check() accepted (a row it refused is a bug of the caller).
const MixedLine& requiredLine(const MixedHybridModel& model, const ElementBatchView& view, std::int64_t row)
{
    const MixedLine* line = model.lineOf(view.card(), row);
    if (line == nullptr)
        throw std::logic_error(
            base::concatenate({"MixedHybridLineFamily: element ", std::to_string(view.elementId(row)), " of ",
                               view.card(), " was not accepted by check() (a bug of O_o_Solver: report it)"}));
    return *line;
}

// The kernel's data of a line with the section `section`, with or without its mass: that of its edge alone, or that
// of a line on an edge of shells (its kinematics the trace of the first shell of the edge: ShellPart::lineData).
LineData lineDataWith(const MixedHybridModel& model, const MixedLine& line, const LineSection& section, bool withMass)
{
    std::variant<LineData, std::string> data =
        line.shellLine >= 0
            ? model.shellPart()->lineData(static_cast<std::size_t>(line.shellLine), section, withMass)
            : aloneLineData(*model.edge(line.edge).alone, static_cast<std::size_t>(line.edge), line.element, section,
                            {0, 3}, linePointCount(MIXED_HYBRID_DEGREE), withMass);
    if (const std::string* message = std::get_if<std::string>(&data))
        throw std::logic_error(base::concatenate({"MixedHybridLineFamily: element ", std::to_string(line.elementId),
                                                  ": ", *message, " (check() accepted it: a bug of O_o_Solver)"}));
    return std::get<LineData>(std::move(data));
}

// The kernel's data of a line with its own section and the mass per length `addedMass` of the NSM cards.
LineData lineDataOf(const MixedHybridModel& model, const MixedLine& line, bool withMass, double addedMass = 0.0)
{
    LineSection section = line.section;
    section.nonStructuralMass += addedMass;
    return lineDataWith(model, line, section, withMass);
}

// The section with a unit density and no non-structural mass: its mass per length is its area.
LineSection unitDensity(const LineSection& section)
{
    LineSection unit = section;
    unit.density = 1.0;
    unit.nonStructuralMass = 0.0;
    return unit;
}

// The first translation of the lower and of the upper grid of its edge among the local unknowns of a line alone
// (LineKinematics.hpp), the x component of the basic system.
constexpr std::array<std::size_t, 2> ALONE_GRID_X_TRANSLATIONS{0, 3};
// The unit translation along the x axis of the basic system among the six rigid motions of a shell
// (rigidMotionUnknowns).
constexpr std::size_t RIGID_X_TRANSLATION = 0;

// The local unknowns of a line (those of `data`) under a unit translation along the x axis of the basic system. A line
// alone: the x translations of the two grids of its edge (the functions between them vanish at the grids and take
// nothing of a constant). A line on an edge of shells has the local unknowns of the first shell of that edge, then
// the rotations at its two grids: the translation is that of the shell, whose edges and interior carry part of a
// constant (rigidMotionUnknowns), and the rotations stay zero.
Vector unitTranslationOf(const MixedHybridModel& model, const MixedLine& line, const LineData& data)
{
    Vector x(data.dofs.size(), 0.0);
    if (line.shellLine < 0)
    {
        for (const std::size_t translation : ALONE_GRID_X_TRANSLATIONS)
            x[translation] = 1.0;
        return x;
    }
    const ShellPart& part = *model.shellPart();
    const ShellElementData shell = part.element(data.source.element);
    const ShellReferenceElement& reference = part.referenceElement(shell.cornerCount);
    const std::array<Vector, 6> rigid = rigidMotionUnknowns(reference, shell.definition());
    const Vector& translation = rigid[RIGID_X_TRANSLATION];
    std::copy(translation.begin(), translation.end(), x.begin());
    return x;
}

// The volume of a line, ∫ A μ ds along its axis (μ the length of an offset axis over that of its edge): with a unit
// density and no non-structural mass, the momentum along x of its unit translation along x, Σ_i x_i · inertia(i, x)
// (LineData::inertia = ∫ (u + θ × o)ᵀ m μ ds with m = A, on the rule of the mass of the line).
double lineVolumeOf(const MixedHybridModel& model, const MixedLine& line)
{
    const LineData data = lineDataWith(model, line, unitDensity(line.section), false);
    const Vector x = unitTranslationOf(model, line, data);
    double volume = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        volume += x[i] * data.inertia(i, 0);
    return volume;
}

// The kinematics of a line at the fraction `fraction` of its chord from its grid A: those of its edge alone, or the
// trace of the first shell of its edge.
LineKinematics kinematicsAt(const MixedHybridModel& model, const MixedLine& line, double fraction)
{
    if (line.shellLine >= 0)
        return model.shellPart()->lineKinematics(static_cast<std::size_t>(line.shellLine), fraction);
    return aloneLineKinematics(*model.edge(line.edge).alone, line.forward, fraction);
}

// The map of a line with pin flags (LocalMap.hpp): on the grid block where a direction d of the element is released,
// P T with P = I − Σ d dᵀ (the grid point's component along d no longer reaches the line; the released directions of
// one end are axes of the element, orthogonal), and the line's own unknown λ of its end entity in a new column:
// x += d λ.
LocalMap withReleases(const LocalMap& gridMap, const MixedLine& line)
{
    const std::size_t n = gridMap.toLocal.rows();
    LocalMap map;
    map.dofs = gridMap.dofs;
    for (const LineRelease& release : line.releases)
        map.dofs.push_back(ElementDof::entity(release.entity, release.index));
    map.toLocal = Matrix(n, map.dofs.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < gridMap.dofs.size(); ++j)
            map.toLocal(i, j) = gridMap.toLocal(i, j);
    // The translations of the line's grid A and B among the local unknowns of a line alone (in the order of the edge,
    // its lower grid first), and its rotations Λ (in the order of the line), the last six.
    const std::size_t rotations = n - 6;
    const std::array<std::size_t, 2> translationOf =
        line.forward ? std::array<std::size_t, 2>{0, 3} : std::array<std::size_t, 2>{3, 0};
    for (std::size_t r = 0; r < line.releases.size(); ++r)
    {
        const LineRelease& release = line.releases[r];
        const std::size_t first = release.isRotation ? rotations + 3 * release.end : translationOf[release.end];
        const std::array<double, 3>& d = release.direction;
        for (std::size_t j = 0; j < gridMap.dofs.size(); ++j)
        {
            double along = 0.0;
            for (std::size_t b = 0; b < 3; ++b)
                along += d[b] * map.toLocal(first + b, j);
            for (std::size_t a = 0; a < 3; ++a)
                map.toLocal(first + a, j) -= d[a] * along;
        }
        for (std::size_t a = 0; a < 3; ++a)
            map.toLocal(first + a, n + r) = d[a];
    }
    return map;
}

// The map from the DOFs the element writes its matrices on to the local unknowns of its kernel (LocalMap.hpp): those
// of a line alone (with its pin flags), or the LineData::dofs of a line on an edge of shells (the shell's local
// unknowns, then the rotations Λ at its grids).
LocalMap lineMapOf(const ElementBatchView& view, const MixedHybridModel& model, const MixedLine& line,
                   const LineData& data)
{
    if (line.shellLine >= 0)
        return localMapOf(view, model.partDofs(data.dofs, data.signs));
    const LocalMap gridMap = localMapOf(view, model.aloneLineDofs(line));
    return line.releases.empty() ? gridMap : withReleases(gridMap, line);
}

// The section with a unit expansion coefficient: the kernel's thermal terms then take the free strain itself as the
// "temperature" (α ΔT + D/L, the DEFORM included, which no temperature alone gives).
LineSection unitExpansion(const LineSection& section)
{
    LineSection unit = section;
    unit.thermalExpansion = 1.0;
    return unit;
}

// A uniform free strain as the kernel's temperature field; empty for none.
LineTemperatureField uniformField(double freeStrain)
{
    if (freeStrain == 0.0)
        return {};
    return [freeStrain](const Vec3Of<double>&) { return freeStrain; };
}

// The load of one BeamDistributedLoad on the local unknowns of a line: the work of the force f and the couple m per
// length (basic system) at the axis, whose point is the edge's moved by the offset o of the axis:
//   δW = ∫ f · (u + θ × o) + m · θ ds = ∫ f · u + (m + o × f) · θ ds
// over the loaded part, by Gauss points (the integrand is a polynomial of the position on a line alone: exact; on a
// curved shell edge the chord fraction is not the edge's parameter and the rule is that of the loaded chord), or at
// its point for a concentrated load. The positions are distances from end A along the element (BeamLineLoad).
Vector lineLoadVector(const MixedHybridModel& model, const MixedLine& line, const BeamData& beam,
                      const BeamLineLoad& load, std::size_t unknownCount)
{
    const base::Mat3& axes = beam.geometry.axes;
    const base::Vec3 direction = axes * load.direction;
    const double length = beam.geometry.length;
    Vector vector(unknownCount, 0.0);
    const auto addAt = [&](double distance, double value, double weight)
    {
        const double fraction = distance / length;
        const LineKinematics kinematics = kinematicsAt(model, line, fraction);
        base::Vec3 force{};
        base::Vec3 couple{};
        (load.isMoment ? couple : force) = value * direction;
        // The offset of the axis at that point, linear between the ends.
        const Vec3Of<double>& a = line.element.offsets[0];
        const Vec3Of<double>& b = line.element.offsets[1];
        const base::Vec3 offset{a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]),
                                a[2] + fraction * (b[2] - a[2])};
        const base::Vec3 moment = couple + base::cross(offset, force);
        for (std::size_t j = 0; j < kinematics.displacement.columns(); ++j)
        {
            double work = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
                work += force[c] * kinematics.displacement(c, j) + moment[c] * kinematics.rotation(c, j);
            vector[j] += weight * work;
        }
    };
    if (load.isConcentrated)
    {
        addAt(load.start, load.startValue, 1.0);
        return vector;
    }
    const LineRule rule = gaussLegendre(linePointCount(MIXED_HYBRID_DEGREE));
    const double half = 0.5 * (load.end - load.start);
    for (std::size_t q = 0; q < rule.points.size(); ++q)
    {
        const double xi = rule.points[q];
        const double distance = load.start + half * (1.0 + xi);
        const double value = load.startValue + 0.5 * (1.0 + xi) * (load.endValue - load.startValue);
        addAt(distance, value, half * rule.weights[q]);
    }
    return vector;
}

// The sum of the loads of a line on its local unknowns: its free expansion and its line loads (a bar or beam only).
Vector lineLoadsOf(const MixedHybridModel& model, const MixedLine& line, const BeamData* beam, const LineData& data,
                   double freeStrain, std::span<const BeamLineLoad> loads)
{
    const std::size_t n = data.dofs.size();
    Vector total(n, 0.0);
    if (freeStrain != 0.0)
        total = lineThermalLoad(data, unitExpansion(line.section), uniformField(freeStrain));
    for (const BeamLineLoad& lineLoad : loads)
    {
        const Vector one = lineLoadVector(model, line, *beam, lineLoad, n);
        for (std::size_t i = 0; i < n; ++i)
            total[i] += one[i];
    }
    return total;
}

// The forces (N, V_y, V_z, T, M_y, M_z) on the face of normal +x of the section of a line at the fraction `fraction`
// of its chord from its grid A, the moments about its reference axis (the kernel's LineState).
std::array<double, 6> sectionStateForcesAt(const MixedHybridModel& model, const MixedLine& line, const LineData& data,
                                           std::span<const double> x, double fraction, double freeStrain)
{
    const LineKinematics kinematics = kinematicsAt(model, line, fraction);
    std::variant<LineState, std::string> state =
        lineStateOfCase(data, kinematics, line.element, unitExpansion(line.section), x, uniformField(freeStrain));
    if (const std::string* message = std::get_if<std::string>(&state))
        throw std::logic_error(
            base::concatenate({"MixedHybridLineFamily: element ", std::to_string(line.elementId), ": ", *message}));
    return std::get<LineState>(state).forces;
}

// The forces of the section of a line at the fraction `fraction` of its chord from its grid A, in the convention of
// the bar and beam output (BeamForces): moment1 = M_z, moment2 = −M_y about the neutral axis, shear1 = V_y,
// shear2 = V_z, axial = N, torque = T. The kernel gives the moments about the reference axis (the shear centre): the
// axial force acts at the neutral axis (y_n, z_n), so M_n = M − n × N e_x, M_y,n = M_y − z_n N, M_z,n = M_z + y_n N.
BeamForces sectionForcesAt(const MixedHybridModel& model, const MixedLine& line, const LineData& data,
                           std::span<const double> x, double fraction, double freeStrain)
{
    const std::array<double, 6> f = sectionStateForcesAt(model, line, data, x, fraction, freeStrain);
    const std::array<double, 2> neutral = line.section.at(fraction).neutralAxis;
    const double momentY = f[4] - neutral[1] * f[0];
    const double momentZ = f[5] + neutral[0] * f[0];
    return {momentZ, -momentY, f[1], f[2], f[0], f[3]};
}

// The end forces of a bar or beam alone, the forces the grid points exert on the element over its end DOFs in the
// element system ([F_A, M_A, F_B, M_B] at the end points: the input of the card's other family's BeamResponse), by the
// equilibrium of the element: r = K x − f, f its loads (the free expansion and the line loads), read at the
// translations of its grid points and at its rotations Λ. These are exact where the displacements of the grid points
// are (a load between the ends included), unlike the moments of the section at the ends, which the moment field of the
// element (a polynomial) only approximates under a load that is not one. The moments are carried from the grid points
// to the end points of the offset axis: M_end = M_grid − w × F.
std::array<double, 12> beamEndForces(const MixedHybridModel& model, const MixedLine& line, const BeamData& beam,
                                     const LineData& data, std::span<const double> x, double freeStrain,
                                     std::span<const BeamLineLoad> loads)
{
    const std::size_t n = x.size();
    const Vector load = lineLoadsOf(model, line, &beam, data, freeStrain, loads);
    Vector residual(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        double sum = -load[i];
        for (std::size_t j = 0; j < n; ++j)
            sum += data.stiffness(i, j) * x[j];
        residual[i] = sum;
    }
    // The translations come in the order of the edge (its lower grid first), the rotations Λ in that of the line.
    const std::size_t rotations = n - 6;
    const std::array<std::size_t, 2> translationOf =
        line.forward ? std::array<std::size_t, 2>{0, 3} : std::array<std::size_t, 2>{3, 0};
    const std::array<base::Vec3, 2> offsets{beam.geometry.offsetA, beam.geometry.offsetB};
    const base::Mat3 toElement = beam.geometry.axes.transposed();
    std::array<double, 12> endForces{};
    for (std::size_t end = 0; end < 2; ++end)
    {
        const base::Vec3 force{residual[translationOf[end]], residual[translationOf[end] + 1],
                               residual[translationOf[end] + 2]};
        const std::size_t r = rotations + 3 * end;
        const base::Vec3 moment =
            base::Vec3{residual[r], residual[r + 1], residual[r + 2]} - base::cross(offsets[end], force);
        const base::Vec3 forceInElement = toElement * force;
        const base::Vec3 momentInElement = toElement * moment;
        for (std::size_t k = 0; k < 3; ++k)
        {
            endForces[6 * end + k] = forceInElement[k];
            endForces[6 * end + 3 + k] = momentInElement[k];
        }
    }
    return endForces;
}

// The forces along a bar or beam of the formulation at a distance from its end A (BeamForces convention). A line
// alone: the response of the card's other family to the end forces of the formulation (beamEndForces, exact at the
// ends) and to its line loads between them. A line on an edge of shells: the section forces of the kernel, since the
// shells load it along the edge, which no end forces and line loads describe (under a concentrated line load these
// are the moments of the polynomial moment field of the element, an approximation).
//
// Ownership: borrows the model, the line, the beam, the data, the unknowns and the loads, which must outlive it.
class LineForces
{
public:
    LineForces(const MixedHybridModel& model, const MixedLine& line, const BeamData& beam, const LineData& data,
               std::span<const double> x, double freeStrain, std::span<const BeamLineLoad> loads)
        : m_model(&model), m_line(&line), m_beam(&beam), m_data(&data), m_x(x), m_freeStrain(freeStrain)
    {
        if (line.shellLine < 0)
            m_response.emplace(beam, beamEndForces(model, line, beam, data, x, freeStrain, loads), loads);
    }

    [[nodiscard]] BeamForces at(double distance) const
    {
        if (m_response)
            return m_response->forcesAt(distance);
        return sectionForcesAt(*m_model, *m_line, *m_data, m_x, distance / m_beam->geometry.length, m_freeStrain);
    }

    // The distance where the bar output reads its shear forces, axial force and torque (one value per element): end
    // A for a line alone (where the card's other family reads them: constant without line loads), the middle for a
    // line on an edge of shells (where they vary along the line: their mean for a linear variation).
    [[nodiscard]] double constantsAt() const { return m_response ? 0.0 : 0.5 * m_beam->geometry.length; }

private:
    const MixedHybridModel* m_model;
    const MixedLine* m_line;
    const BeamData* m_beam;
    const LineData* m_data;
    std::span<const double> m_x;
    double m_freeStrain;
    std::optional<BeamResponse> m_response;
};

// The row of a Bar2 (the layout of its rows without line loads): the forces at both ends, or the stresses (strains)
// of both ends, from the forces along the element.
void appendBarRow(const ElementBatchView& view, std::int64_t row, const BeamData& beam, const LineForces& forces,
                  ElementOutput output, ElementResultBlock& block)
{
    const BeamForces atA = forces.at(0.0);
    const BeamForces atB = forces.at(beam.geometry.length);
    const BeamForces constants = forces.at(forces.constantsAt());
    std::array<double, 15> values{};
    std::size_t count = 8;
    if (output == ElementOutput::Forces)
        values = {atA.moment1,      atA.moment2,      atB.moment1,     atB.moment2,
                  constants.shear1, constants.shear2, constants.axial, constants.torque};
    else
    {
        const StationStresses endA = stationStresses(beam.section.stations[0], atA);
        const StationStresses endB = stationStresses(beam.section.stations[beam.section.count - 1], atB);
        const double tension = tensionMargin(std::max(endA.maximum, endB.maximum), beam.material);
        const double compression = compressionMargin(std::min(endA.minimum, endB.minimum), beam.material);
        values = {endA.bending[0], endA.bending[1], endA.bending[2], endA.bending[3], endA.axial,
                  endA.maximum,    endA.minimum,    tension,         endB.bending[0], endB.bending[1],
                  endB.bending[2], endB.bending[3], endB.maximum,    endB.minimum,    compression};
        if (output == ElementOutput::Strains)
            for (const std::size_t k : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 8U, 9U, 10U, 11U, 12U, 13U})
                values[k] = uniaxialStrain(values[k], beam.material);
        count = 15;
    }
    block.append(row, view.elementId(row), std::span<const double>(values.data(), count));
}

} // namespace

MixedHybridLineFamily::MixedHybridLineFamily(std::unique_ptr<ElementFamily> classic) : m_classic(std::move(classic))
{
    const std::string_view card = m_classic->card();
    m_rows = card == "Bar2" ? Rows::Bar : card == "Beam2" ? Rows::Beam : Rows::Rod;
}

FacetSet MixedHybridLineFamily::facets() const noexcept
{
    return {Facet::Stiffness, Facet::Mass, Facet::Loads, Facet::Differential, Facet::Recovery, Facet::Derivatives};
}

void MixedHybridLineFamily::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
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
    for (base::Diagnostic& problem : MixedHybridModel::of(view.index()).elementProblems(view.card()))
        sink.report(std::move(problem));
}

void MixedHybridLineFamily::computeStiffness(const ElementBatchView& view, ElementRange range,
                                             ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedLine& line = requiredLine(model, view, row);
        const LineData data = lineDataOf(model, line, false);
        const LocalMap map = lineMapOf(view, model, line, data);
        block.append(row, map.dofs, congruent(data.stiffness, map));
    }
}

void MixedHybridLineFamily::computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                        ElementMatrixBlock& block) const
{
    // The consistent mass whatever options.formulation asks (spec section 8): a lumped form of the unknowns of the
    // edge has no meaning. The NSM cards add their mass per length.
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedLine& line = requiredLine(model, view, row);
        const double added = options.addedNonStructuralMass.empty()
                                 ? 0.0
                                 : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
        const LineData data = lineDataOf(model, line, true, added);
        const LocalMap map = lineMapOf(view, model, line, data);
        block.append(row, map.dofs, congruent(*data.mass, map));
    }
}

bool MixedHybridLineFamily::computeVolumes(const ElementBatchView& view, ElementRange range,
                                           std::span<double> volumes) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
        volumes[static_cast<std::size_t>(row - range.first)] = lineVolumeOf(model, requiredLine(model, view, row));
    return true;
}

std::vector<std::string> MixedHybridLineFamily::outputColumns(ElementOutput output) const
{
    return m_classic->outputColumns(output);
}

void MixedHybridLineFamily::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                                    const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    block.reset(outputColumns(output));
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedLine& line = requiredLine(model, view, row);
        const LineData data = lineDataOf(model, line, false);
        const Vector x = localUnknowns(lineMapOf(view, model, line, data), displacements);
        const double freeStrain = freeStrainOf(view, row, line, displacements.temperatures);
        if (m_rows == Rows::Rod)
        {
            // The axial force and the torque of a rod: constant along a rod alone, read at its middle (their mean
            // along a rod on an edge of shells).
            const BeamForces forces = sectionForcesAt(model, line, data, x, 0.5, freeStrain);
            const RodData& rod = std::get<RodData>(line.classic);
            const std::array<double, 4> values = rodOutputRow(rod, {forces.axial, forces.torque}, output);
            const std::size_t count = output == ElementOutput::Forces ? 2 : 4;
            block.append(row, view.elementId(row), std::span<const double>(values.data(), count));
            continue;
        }
        // A bar or a beam: its rows made by the card's other family from the forces along it (LineForces).
        const BeamData& beam = std::get<BeamData>(line.classic);
        const LineForces forces(model, line, beam, data, x, freeStrain, {});
        if (m_rows == Rows::Beam)
            Beam2Element::appendForcesRows(
                view, row, beam, [&forces](double distance) { return forces.at(distance); }, output, block);
        else
            appendBarRow(view, row, beam, forces, output, block);
    }
}

double MixedHybridLineFamily::freeStrainOf(const ElementBatchView& view, std::int64_t row, const MixedLine& data,
                                           const ElementTemperatures* temperatures)
{
    if (temperatures == nullptr)
        return 0.0;
    if (const RodData* rod = std::get_if<RodData>(&data.classic))
    {
        // α ΔT + δ/L, the free strain of the card's other family (rodThermalForce over E A).
        const RodInitialStrain initial = resolveRodInitialStrain(view, row, *rod, *temperatures).required();
        return rod->material.expansion * initial.deltaT + initial.elongation / rod->geometry.length;
    }
    NodalDisplacements onlyTemperatures;
    onlyTemperatures.temperatures = temperatures;
    return beamThermalStrain(view, row, std::get<BeamData>(data.classic), onlyTemperatures);
}

void MixedHybridLineFamily::checkThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                              base::DiagnosticSink& sink) const
{
    // The temperatures are read as the card's other family reads them: its checks apply.
    m_classic->checkThermalLoads(view, temperatures, sink);
}

void MixedHybridLineFamily::computeThermalLoads(const ElementBatchView& view, ElementRange range,
                                                const ElementTemperatures& temperatures,
                                                ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedLine& line = requiredLine(model, view, row);
        const double freeStrain = freeStrainOf(view, row, line, &temperatures);
        if (freeStrain == 0.0)
            continue;
        const LineData data = lineDataOf(model, line, false);
        const Vector load = lineThermalLoad(data, unitExpansion(line.section), uniformField(freeStrain));
        const LocalMap map = lineMapOf(view, model, line, data);
        block.append(row, map.dofs, onDofs(load, map));
    }
}

void MixedHybridLineFamily::checkDifferentialStiffness(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    // The offsets are in the section law of the formulation: nothing to refuse under OFFDEF ELMOFF (spec section 9).
    static_cast<void>(view);
    static_cast<void>(sink);
}

void MixedHybridLineFamily::computeDifferentialStiffness(const ElementBatchView& view, ElementRange range,
                                                         const NodalDisplacements& preload,
                                                         ElementMatrixBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const MixedLine& line = requiredLine(model, view, row);
        const LineData data = lineDataOf(model, line, false);
        const LocalMap map = lineMapOf(view, model, line, data);
        const Vector x = localUnknowns(map, preload);
        const double freeStrain = freeStrainOf(view, row, line, preload.temperatures);
        const std::vector<double> freeStrains =
            lineFreeStrains(data, unitExpansion(line.section), uniformField(freeStrain));
        block.append(row, map.dofs, congruent(lineGeometricStiffness(data, x, freeStrains), map));
    }
}

void MixedHybridLineFamily::computeLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                             ElementVectorBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    forEachLoadedElement(loads,
                         [&](std::int64_t row, std::span<const ElementLineLoad> elementLoads)
                         {
                             const MixedLine& line = requiredLine(model, view, row);
                             const BeamData& beam = std::get<BeamData>(line.classic);
                             const LineData data = lineDataOf(model, line, false);
                             const LocalMap map = lineMapOf(view, model, line, data);
                             const std::vector<BeamLineLoad> resolved =
                                 resolveLineLoadsOrThrow(view, elementLoads, beam.geometry);
                             const Vector total = lineLoadsOf(model, line, &beam, data, 0.0, resolved);
                             block.append(row, map.dofs, onDofs(total, map));
                         });
}

void MixedHybridLineFamily::recoverUnderLineLoads(const ElementBatchView& view, std::span<const ElementLineLoad> loads,
                                                  ElementOutput output, const NodalDisplacements& displacements,
                                                  ElementResultBlock& block) const
{
    const MixedHybridModel& model = MixedHybridModel::of(view.index());
    block.reset(outputColumnsUnderLineLoads(output));
    forEachLoadedElement(
        loads,
        [&](std::int64_t row, std::span<const ElementLineLoad> elementLoads)
        {
            const MixedLine& line = requiredLine(model, view, row);
            const BeamData& beam = std::get<BeamData>(line.classic);
            const LineData data = lineDataOf(model, line, false);
            const Vector x = localUnknowns(lineMapOf(view, model, line, data), displacements);
            const double freeStrain = freeStrainOf(view, row, line, displacements.temperatures);
            // The rows of the card's other family from the forces along the element (LineForces: the loads between
            // the ends in its equilibrium).
            const std::vector<BeamLineLoad> resolved = resolveLineLoadsOrThrow(view, elementLoads, beam.geometry);
            const LineForces forces(model, line, beam, data, x, freeStrain, resolved);
            if (m_rows == Rows::Beam)
            {
                Beam2Element::appendForcesRows(
                    view, row, beam, [&forces](double distance) { return forces.at(distance); }, output, block);
                return;
            }
            // A bar: one row at each end (the layout of the Bar2 under line loads).
            for (std::size_t end = 0; end < 2; ++end)
            {
                const double position = static_cast<double>(end);
                const BeamForces atEnd = forces.at(position * beam.geometry.length);
                const ResultPlace place{0, 1, position};
                if (output == ElementOutput::Forces)
                {
                    const std::array<double, 6> values{atEnd.moment1, atEnd.moment2, atEnd.shear1,
                                                       atEnd.shear2,  atEnd.axial,   atEnd.torque};
                    block.append(row, view.elementId(row), place, values);
                    continue;
                }
                const StationStresses stresses =
                    stationStresses(beam.section.stations[end == 0 ? 0 : beam.section.count - 1], atEnd);
                const double tension = tensionMargin(stresses.maximum, beam.material);
                const double compression = compressionMargin(stresses.minimum, beam.material);
                double margin = UNDEFINED_MARGIN;
                if (tension != UNDEFINED_MARGIN && compression != UNDEFINED_MARGIN)
                    margin = std::min(tension, compression);
                else if (tension != UNDEFINED_MARGIN)
                    margin = tension;
                else
                    margin = compression;
                std::array<double, 8> values{
                    stresses.bending[0], stresses.bending[1], stresses.bending[2], stresses.bending[3],
                    stresses.axial,      stresses.maximum,    stresses.minimum,    margin};
                if (output == ElementOutput::Strains)
                    for (std::size_t k = 0; k < 7; ++k)
                        values[k] = uniaxialStrain(values[k], beam.material);
                block.append(row, view.elementId(row), place, values);
            }
        });
}

} // namespace oos::elements::mixedHybrid
