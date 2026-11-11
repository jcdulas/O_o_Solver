#include "oos/elements/bars_beams/BeamFieldDerivatives.hpp"

#include "BeamSizing.hpp"
#include "oos/elements/bars_beams/BeamThermal.hpp"

#include <algorithm>
#include <limits>

namespace oos::elements
{
namespace
{

using beam_sizing::Direction;
using beam_sizing::Kernel;

struct Directions
{
    Direction first;
    Direction other;
};

std::optional<Directions> directionsOf(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                                       const DesignedField& field, const DesignedField* other)
{
    auto a = beam_sizing::directionOf(beam, view, row, field);
    auto b = other ? beam_sizing::directionOf(beam, view, row, *other) : beam_sizing::zeroDirection(beam);
    if (!a || !b)
        return std::nullopt;
    return Directions{std::move(*a), std::move(*b)};
}

double selected(const MixedScalar& value, bool mixed)
{
    return mixed ? value.mixed : value.first;
}

MixedScalar thermalStrainOf(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                           const Directions& directions, const ElementTemperatures* temperatures)
{
    if (temperatures == nullptr)
        return {};
    MixedScalar strain = beamDeformationStrain(view, row, beam, *temperatures);
    if (!temperatures->hasTemperatureLoad)
        return strain;
    const double change = resolveBeamTemperatureChange(view, row, beam, *temperatures).required();
    const bool usesReference = beamUsesReferenceTemperature(view, row, beam, *temperatures);
    const MixedScalar delta{change, usesReference ? -directions.first.material.referenceTemperature : 0.0,
                                    usesReference ? -directions.other.material.referenceTemperature : 0.0};
    const MixedScalar expansion{beam.material.expansion, directions.first.material.expansion,
                                                        directions.other.material.expansion};
    return strain + expansion * delta;
}

void append(const BeamData& beam, std::int64_t row, const beam_sizing::Matrix<12>& rates,
             bool mixed, ElementMatrixBlock& block)
{
    base::MatN<12> result;
    bool isZero = true;
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = i; j < 12; ++j)
        {
            const double value = 0.5 * (selected(rates(i, j), mixed) + selected(rates(j, i), mixed));
            result(i, j) = result(j, i) = value;
            isZero = isZero && value == 0.0;
        }
    if (!isZero)
        block.append(row, beam.geometry.dofs(), result.values);
}

std::array<double, 12> gridOf(const BeamData& beam, const NodalDisplacements& values)
{
    const auto dofs = beam.geometry.dofs();
    std::array<double, 12> result{};
    for (std::size_t i = 0; i < dofs.size(); ++i)
        result[i] = values.valueAt(dofs[i]);
    return result;
}

bool supports(ElementOutput output)
{
    return output == ElementOutput::Forces || output == ElementOutput::Stresses || output == ElementOutput::Strains;
}

void appendRecovery(const Kernel& kernel, const ElementFamily& family, const ElementBatchView& view,
                     std::int64_t row, ElementOutput output, std::span<const double> grid,
                     bool mixed, ElementResultBlock& block, MixedScalar thermalStrain = {},
                     std::span<const BeamLineLoad> lineLoads = {})
{
    const auto forms = family.outputColumnForms(output);
    block.reset(family.outputColumns(output));
    for (const auto& part : kernel.recover(view, output, grid, thermalStrain, lineLoads))
    {
        std::array<double, 15> values{};
        for (std::size_t c = 0; c < part.count; ++c)
            values[c] = forms[c].kind == ColumnForm::Kind::Linear ? selected(part.values[c], mixed)
                                                                : std::numeric_limits<double>::quiet_NaN();
        block.append(row, view.elementId(row), part.place, std::span(values).first(part.count));
    }
}

} // namespace

bool BeamFieldDerivatives::loads(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                                const DesignedField& field, const DesignedField* other,
                                const ElementLoadContext& loads, ElementVectorBlock& block)
{
    if (!loads.pressures.empty())
        return false;
    const auto directions = directionsOf(beam, view, row, field, other);
    if (!directions)
        return false;
    const auto lines = resolveLineLoadsOrThrow(view, loads.lineLoads, beam.geometry);
    const auto strain = thermalStrainOf(beam, view, row, *directions, loads.temperatures);
    const Kernel kernel(beam, directions->first, directions->other);
    const auto rates = kernel.loads(strain, lines, view.areLoadsOnOffsets());
    std::array<double, 12> values{};
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = selected(rates[i], other != nullptr);
    if (std::any_of(values.begin(), values.end(), [](double value) { return value != 0.0; }))
        block.append(row, beam.geometry.dofs(), values);
    return true;
}

bool BeamFieldDerivatives::stiffness(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                                    const DesignedField& field, const DesignedField* other, ElementMatrixBlock& block)
{
    const auto directions = directionsOf(beam, view, row, field, other);
    if (!directions)
        return false;
    const Kernel kernel(beam, directions->first, directions->other);
    append(beam, row, kernel.stiffness(), other != nullptr, block);
    return true;
}

bool BeamFieldDerivatives::mass(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                               const DesignedField& field, const DesignedField* other, const MassOptions& options,
                               TorsionalInertia torsion, ElementMatrixBlock& block)
{
    const auto directions = directionsOf(beam, view, row, field, other);
    if (!directions)
        return false;
    const double added = options.addedNonStructuralMass.empty() ? 0.0
                         : options.addedNonStructuralMass[static_cast<std::size_t>(row)];
    append(beam, row, beam_sizing::massOf(beam, directions->first, directions->other, added,
        options.formulation, torsion, view.isMassOnOffsets()), other != nullptr, block);
    return true;
}

bool BeamFieldDerivatives::recovery(const BeamData& beam, const ElementFamily& family, const ElementBatchView& view,
                                   std::int64_t row, const DesignedField& field, const DesignedField* other,
                                   ElementOutput output, const NodalDisplacements& displacements,
                                   ElementResultBlock& block)
{
    if (!supports(output))
        return false;
    const auto directions = directionsOf(beam, view, row, field, other);
    if (!directions)
        return false;
    const Kernel kernel(beam, directions->first, directions->other);
    const auto strain = thermalStrainOf(beam, view, row, *directions, displacements.temperatures);
    const auto loads = resolveLineLoadsOrThrow(view, displacements.lineLoads, beam.geometry, row);
    appendRecovery(kernel, family, view, row, output, gridOf(beam, displacements), other != nullptr,
                   block, strain, loads);
    return true;
}

bool BeamFieldDerivatives::slopeRates(const BeamData& beam, const ElementFamily& family,
                                     const ElementBatchView& view, std::int64_t row, const DesignedField& field,
                                     ElementOutput output, RecoverySlopes& block)
{
    if (!supports(output))
        return false;
    const auto directions = directionsOf(beam, view, row, field, nullptr);
    if (!directions)
        return false;
    const Kernel kernel(beam, directions->first, directions->other);
    const auto forms = family.outputColumnForms(output);
    const auto dofs = beam.geometry.dofs();
    block.reset(dofs, forms.size());
    std::array<double, 12> grid{};
    const auto rows = kernel.recover(view, output, grid);
    for (const auto& part : rows)
        block.appendRow(part.place);
    // Linear recovery is exactly linear in the nodal DOFs: its unit columns require no differencing.
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
        grid[d] = 1.0;
        const auto moved = kernel.recover(view, output, grid);
        grid[d] = 0.0;
        for (std::size_t r = 0; r < moved.size(); ++r)
            for (std::size_t c = 0; c < forms.size(); ++c)
                if (forms[c].kind == ColumnForm::Kind::Linear)
                    block.slopes(r, c)[d] = moved[r].values[c].first;
    }
    return true;
}

bool BeamFieldDerivatives::differential(const BeamData& beam, const ElementBatchView& view, std::int64_t row,
                                       const DesignedField& field, const DesignedField* other,
                                       const NodalDisplacements& preload, ElementMatrixBlock& block)
{
    const auto directions = directionsOf(beam, view, row, field, other);
    if (!directions)
        return false;
    const Kernel kernel(beam, directions->first, directions->other);
    const auto strain = thermalStrainOf(beam, view, row, *directions, preload.temperatures);
    append(beam, row, kernel.differential(gridOf(beam, preload), strain), other != nullptr, block);
    return true;
}

} // namespace oos::elements
