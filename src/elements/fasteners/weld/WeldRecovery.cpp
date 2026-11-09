#include "oos/elements/bars_beams/BeamResponse.hpp"
#include "oos/elements/core/LinearRowRecovery.hpp"
#include "oos/elements/fasteners/weld/WeldElement.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace oos::elements
{

namespace
{

// The coefficients of the maximum shear stress of a solid circle of diameter D (WeldElement.hpp): the peak
// transverse shear stress 4V/(3A) = (16/(3π))·V/D² and the torsional stress at the edge 16T/(πD³).
constexpr double SHEAR_STRESS_COEFFICIENT = 16.0 / (3.0 * std::numbers::pi);
constexpr double TORSION_STRESS_COEFFICIENT = 16.0 / std::numbers::pi;
// The columns of a forces row (outputColumns(Forces)) and of a stress row.
constexpr std::size_t FORCE_COLUMNS = 8;
constexpr std::size_t STRESS_COLUMNS = 7;

// The section forces at a distance x from the beam end A' from the end forces f (the forces the points exert on the
// beam, end DOFs in the weld system), in the sign convention of the result tables (bars_beams/BeamResponse::forcesAt
// without loads).
BeamForces forcesAt(const std::array<double, 12>& endForces, double x)
{
    const base::Vec3 forceA{endForces[0], endForces[1], endForces[2]};
    const base::Vec3 momentA{endForces[3], endForces[4], endForces[5]};
    const base::Vec3 force = -forceA;
    const base::Vec3 moment = -momentA + base::Vec3{0.0, -x * forceA[2], x * forceA[1]};
    BeamForces forces;
    forces.axial = force[0];
    forces.shear1 = force[1];
    forces.shear2 = force[2];
    forces.torque = moment[0];
    forces.moment1 = moment[2];
    forces.moment2 = -moment[1];
    return forces;
}

// The forces row of a weld: the bending moments at A' and B', the shears, axial force and torque at A'.
std::array<double, FORCE_COLUMNS> forcesRow(const BeamForces& atA, const BeamForces& atB)
{
    return {atA.moment1, atA.moment2, atB.moment1, atB.moment2, atA.shear1, atA.shear2, atA.axial, atA.torque};
}

// The stress row of a weld from its forces row (WeldElement.hpp): N/A; N/A ± M·(D/2)/I at A' and B', M the
// resultant bending moment; 16V/(3πD²) + 16|T|/(πD³) and V/D², V the resultant shear at A'.
std::array<double, STRESS_COLUMNS> stressRow(std::span<const double> forces, const BeamStation& station,
                                             double diameter)
{
    const double d = diameter;
    const double axial = forces[6] / station.area;
    const double bendingA = std::hypot(forces[0], forces[1]) * (d / 2.0) / station.inertia1;
    const double bendingB = std::hypot(forces[2], forces[3]) * (d / 2.0) / station.inertia1;
    const double shear = std::hypot(forces[4], forces[5]);
    return {axial,
            axial + bendingA,
            axial - bendingA,
            axial + bendingB,
            axial - bendingB,
            SHEAR_STRESS_COEFFICIENT * shear / (d * d) + TORSION_STRESS_COEFFICIENT * std::abs(forces[7]) / (d * d * d),
            shear / (d * d)};
}

} // namespace

// forces.weld and stress.weld of the O_o_Reader data contract.
std::vector<std::string> WeldElement::outputColumns(ElementOutput output) const
{
    if (output == ElementOutput::Forces)
        return {"bendingMomentA1", "bendingMomentA2", "bendingMomentB1", "bendingMomentB2",
                "shear1",          "shear2",          "axial",           "torque"};
    if (output == ElementOutput::Stresses)
        return {"axial", "maxa", "mina", "maxb", "minb", "maxShear", "bearing"};
    return {};
}

// d = T·M·u, f = K_beam·d; the forces at A' (x = 0) and B' (x = beam length) and the stresses of WeldElement.hpp.
void WeldElement::recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                              const NodalDisplacements& displacements, ElementResultBlock& block) const
{
    requireOutput(output);
    block.reset(outputColumns(output));
    const Columns columns(view);
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        const Weld weld = resolve(view, columns, row).required();
        const std::array<double, 12> ends = weld.toBeamEnds * weld.connection.pointMotions(displacements);
        const std::array<double, 12> endForces = weld.beamStiffness * ends;
        const std::array<double, FORCE_COLUMNS> forces =
            forcesRow(forcesAt(endForces, 0.0), forcesAt(endForces, weld.beamLength));
        if (output == ElementOutput::Forces)
        {
            block.append(row, view.elementId(row), forces);
            continue;
        }
        block.append(row, view.elementId(row), stressRow(forces, weld.section.stations[0], weld.diameter));
    }
}

// The forces Linear; of the stresses the axial stress N/A, the others (resultants, an absolute value) not.
std::vector<ColumnForm> WeldElement::outputColumnForms(ElementOutput output) const
{
    if (output == ElementOutput::Forces)
        return linearColumnForms(FORCE_COLUMNS);
    if (output == ElementOutput::Stresses)
    {
        std::vector<ColumnForm> forms(STRESS_COLUMNS, ColumnForm::other());
        forms[0] = ColumnForm::linear();
        return forms;
    }
    return {};
}

bool WeldElement::recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                                     RecoverySlopes& slopes) const
{
    const std::array<RecoveryOutput, 1> outputs{RecoveryOutput::of(output)};
    return recoverySlopesOfOutputs(view, row, outputs, std::span<RecoverySlopes>(&slopes, 1));
}

// The weld resolved once for every output (its piercing points interpolated once): the end forces K_beam·T·M·u, so
// that the forces row of each DOF d of the pierced elements' grid points is forcesAt of the column
// P(·, d) = K_beam·T·M(·, d), and the axial stress its axial force over A (the other stress columns get no slopes).
// PERFORMANCE: each recovery of a case resolved the weld again (its piercing points' interpolation, its beam's
// stiffness); the slopes take one resolution for every case of a block, the stresses made from the forces' rows.
bool WeldElement::recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                              std::span<const RecoveryOutput> outputs,
                                              std::span<RecoverySlopes> slopes) const
{
    for (const RecoveryOutput& output : outputs)
        if (!output.isLayer)
            requireOutput(output.output);
    const Columns columns(view);
    return linearRowSlopesOfOutputs(
        outputs, slopes, [&] { return resolve(view, columns, row).required(); },
        [](const Weld& weld, ElementOutput output, RecoverySlopes& rowSlopes)
        {
            const FastenerConnection& connection = weld.connection;
            const std::size_t n = connection.dofCount();
            const base::MatN<12> endMap = weld.beamStiffness * weld.toBeamEnds;
            // The forces row of each DOF (n × 8).
            std::vector<double> rows(n * FORCE_COLUMNS); // bounded by the grid points of two pierced elements
            std::array<double, 12> motion{};
            for (std::size_t d = 0; d < n; ++d)
            {
                for (std::size_t i = 0; i < 12; ++i)
                    motion[i] = connection.map[i * n + d];
                const std::array<double, 12> endForces = endMap * motion;
                const std::array<double, FORCE_COLUMNS> forces =
                    forcesRow(forcesAt(endForces, 0.0), forcesAt(endForces, weld.beamLength));
                std::copy(forces.begin(), forces.end(), rows.begin() + static_cast<std::ptrdiff_t>(d * FORCE_COLUMNS));
            }
            const std::vector<ElementDof> dofs = connection.dofs();
            if (output == ElementOutput::Forces)
            {
                setLinearRowSlopes(
                    dofs, FORCE_COLUMNS, [&](std::size_t c, std::size_t d) { return rows[d * FORCE_COLUMNS + c]; },
                    rowSlopes);
                return;
            }
            const double area = weld.section.stations[0].area;
            setLinearRowSlopes(
                dofs, STRESS_COLUMNS, [&](std::size_t c, std::size_t d)
                { return c == 0 ? rows[d * FORCE_COLUMNS + 6] / area : 0.0; }, rowSlopes);
        });
}

std::size_t WeldElement::vectorisedRecoveryMinCases(ElementOutput output) const
{
    return outputColumns(output).empty() ? 0 : FASTENER_VECTORISED_RECOVERY_MIN_CASES;
}

// The forces are all Linear: nothing to complete. The stresses are made from the forces (vectorisedSource).
void WeldElement::completeRecoveredRows(const ElementBatchView& /*view*/, ElementOutput output,
                                            std::span<ElementResultBlock> /*blocks*/) const
{
    requireOutput(output);
    if (output != ElementOutput::Forces)
        throw std::invalid_argument(
            "WeldElement::completeRecoveredRows: the stresses are made from the forces "
            "(vectorisedSource); call completeRecoveredRowsFromSource for them");
}

ElementOutput WeldElement::vectorisedSource(ElementOutput output) const
{
    return output == ElementOutput::Stresses ? ElementOutput::Forces : output;
}

// The stress rows from the forces rows, by the function of recover(), the section of each element (its diameter, its
// area and inertia) resolved once for every case without its piercing points.
void WeldElement::completeRecoveredRowsFromSource(const ElementBatchView& view, ElementOutput output,
                                                      std::span<const ElementResultBlock> sources,
                                                      std::span<ElementResultBlock> blocks) const
{
    if (output != ElementOutput::Stresses)
        throw std::invalid_argument(
            "WeldElement::completeRecoveredRowsFromSource: expected the stresses (made from "
            "the forces), found another output");
    if (sources.empty())
        return;
    const Columns columns(view);
    const ElementResultBlock& structure = sources.front();
    for (std::size_t i = 0; i < structure.size(); ++i)
    {
        const double diameter = diameterOf(view, columns, structure.elementRow(i));
        const BeamStation station = weldStation(diameter);
        for (std::size_t c = 0; c < sources.size(); ++c)
            blocks[c].append(sources[c].elementRow(i), sources[c].elementId(i), sources[c].place(i),
                             stressRow(sources[c].values(i), station, diameter));
    }
}

} // namespace oos::elements
