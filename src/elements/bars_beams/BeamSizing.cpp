#include "BeamSizing.hpp"

#include "oos/base/errors/SingularityError.hpp"
#include "oos/elements/bars_beams/BeamDifferential.hpp"
#include "oos/elements/bars_beams/BeamFlexibility.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements::beam_sizing
{
namespace
{

Matrix<6> compliance(const BeamData& beam, const Direction& first, const Direction& other, double position)
{
    const auto value = beam.section.at(position);
    const auto station = stationOf(value, first.section.at(position), other.section.at(position));
    const MixedScalar e{beam.material.youngModulus, first.material.youngModulus, other.material.youngModulus};
    const MixedScalar g{beam.material.shearModulus, first.material.shearModulus, other.material.shearModulus};
    const MixedScalar k1{beam.section.shearFactor1, first.section.shearFactor1, other.section.shearFactor1};
    const MixedScalar k2{beam.section.shearFactor2, first.section.shearFactor2, other.section.shearFactor2};
    const bool axial = e.value > 0.0 && value.area > 0.0;
    const bool plane1 = e.value > 0.0 && value.inertia1 > 0.0;
    const bool plane2 = e.value > 0.0 && value.inertia2 > 0.0;
    Matrix<6> result;
    if (axial) result(0, 0) = 1.0 / (e * station.area);
    if (g.value > 0.0 && value.torsionalConstant > 0.0)
        result(3, 3) = 1.0 / (g * station.torsion);
    if (plane1 && (k1 * station.area * g).value > 0.0)
        result(1, 1) = 1.0 / (k1 * station.area * g);
    if (plane2 && (k2 * station.area * g).value > 0.0)
        result(2, 2) = 1.0 / (k2 * station.area * g);
    if (plane1 && plane2)
    {
        // The full inverse retains the I12 derivative even at I12 == 0.
        const auto determinant = e * (station.inertia1 * station.inertia2 -
                                       station.productInertia * station.productInertia);
        result(4, 4) = station.inertia1 / determinant;
        result(4, 5) = result(5, 4) = station.productInertia / determinant;
        result(5, 5) = station.inertia2 / determinant;
    }
    else
    {
        if (plane2) result(4, 4) = 1.0 / (e * station.inertia2);
        if (plane1) result(5, 5) = 1.0 / (e * station.inertia1);
    }
    Matrix<6> toNeutral;
    for (std::size_t k = 0; k < 6; ++k)
        toNeutral(k, k) = 1.0;
    toNeutral(4, 0) = -station.neutral[1];
    toNeutral(5, 0) = station.neutral[0];
    return congruence(result, toNeutral);
}

Matrix<12> rawStiffness(const BeamData& beam, const Direction& first, const Direction& other,
                        const BeamFlexibility& flexibility, Matrix<6>& natural)
{
    const double length = beam.geometry.length;
    base::MatN<6> fp, fq, fpq;
    flexibility.visitIntegrationPoints([&](double x, double weight)
    {
        const auto c = compliance(beam, first, other, x / length);
        Matrix<6> resultants;
        for (std::size_t k = 0; k < 6; ++k)
            resultants(k, k) = 1.0;
        resultants(4, 2) = -(length - x);
        resultants(5, 1) = length - x;
        const auto part = congruence(c, resultants);
        for (std::size_t k = 0; k < 36; ++k)
        {
            fp.values[k] += weight * part.values[k].first;
            fq.values[k] += weight * part.values[k].other;
            fpq.values[k] += weight * part.values[k].mixed;
        }
    });
    const auto& k = flexibility.naturalStiffness();
    const auto p = -1.0 * (k * fp * k);
    const auto q = -1.0 * (k * fq * k);
    const auto pq = k * (fq * k * fp + fp * k * fq - fpq) * k;
    for (std::size_t index = 0; index < natural.values.size(); ++index)
        natural.values[index] = {k.values[index], p.values[index], q.values[index], pq.values[index]};
    const auto b = cantileverDeformation(length);
    const auto value = flexibility.stiffness();
    const auto firstEnd = base::congruence(p, b);
    const auto otherEnd = base::congruence(q, b);
    const auto mixedEnd = base::congruence(pq, b);
    Matrix<12> result;
    for (std::size_t index = 0; index < result.values.size(); ++index)
        result.values[index] = {value.values[index], firstEnd.values[index],
                                 otherEnd.values[index], mixedEnd.values[index]};
    return result;
}

// The released block is at most 12x12. Pivot decisions use its ordinary values; arithmetic propagates both
// derivatives of Krr^-1. The resolved element already passed the positive-definite pin check.
Matrix<12> releasedInverse(const Matrix<12>& raw, std::span<const std::size_t> released)
{
    const auto n = released.size();
    Matrix<12> work, inverse;
    for (std::size_t i = 0; i < n; ++i)
    {
        inverse(i, i) = 1.0;
        for (std::size_t j = 0; j < n; ++j)
            work(i, j) = raw(released[i], released[j]);
    }
    for (std::size_t column = 0; column < n; ++column)
    {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row)
            if (std::abs(work(row, column).value) > std::abs(work(pivot, column).value))
                pivot = row;
        if (work(pivot, column).value == 0.0)
            throw base::SingularityError("beam sizing derivatives found a singular released-end block");
        for (std::size_t j = 0; j < n; ++j)
        {
            std::swap(work(column, j), work(pivot, j));
            std::swap(inverse(column, j), inverse(pivot, j));
        }
        const auto diagonal = work(column, column);
        for (std::size_t j = 0; j < n; ++j)
        {
            work(column, j) /= diagonal;
            inverse(column, j) /= diagonal;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i == column)
                continue;
            const auto factor = work(i, column);
            for (std::size_t j = 0; j < n; ++j)
            {
                work(i, j) -= factor * work(column, j);
                inverse(i, j) -= factor * inverse(column, j);
            }
        }
    }
    return inverse;
}

Matrix<12> pointTransform(const Station& a, const Station& b, bool structural)
{
    Matrix<12> result;
    const std::array<Station, 2> ends{a, b};
    for (std::size_t end = 0; end < 2; ++end)
    {
        const auto& point = structural ? ends[end].neutral : ends[end].centre;
        const auto at = 6 * end;
        for (std::size_t k = 0; k < 6; ++k)
            result(at + k, at + k) = 1.0;
        result(at, at + 4) = point[1];
        result(at, at + 5) = -point[0];
        result(at + 1, at + 3) = -point[1];
        result(at + 2, at + 3) = point[0];
    }
    return result;
}

} // namespace

Kernel::Kernel(const BeamData& beam, const Direction& first, const Direction& other)
    : m_beam(beam), m_first(first), m_other(other),
      m_flexibility(beam.section, beam.material, beam.geometry.length)
{
    release(rawStiffness(beam, first, other, m_flexibility, m_natural));
}

void Kernel::release(const Matrix<12>& raw)
{
    std::array<std::size_t, 12> released{};
    std::size_t count = 0;
    for (std::size_t i = 0; i < 12; ++i)
        if (m_beam.geometry.isReleased[i]) released[count++] = i;
        else m_completion(i, i) = 1.0;
    const auto inverse = releasedInverse(raw, std::span(released).first(count));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < 12; ++j)
            if (!m_beam.geometry.isReleased[j])
                for (std::size_t k = 0; k < count; ++k)
                    m_completion(released[i], j) -= inverse(i, k) * raw(released[k], j);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < 12; ++j)
            if (!m_beam.geometry.isReleased[i] && !m_beam.geometry.isReleased[j])
            {
                m_end(i, j) = raw(i, j);
                for (std::size_t k = 0; k < count; ++k)
                    m_end(i, j) += raw(i, released[k]) * m_completion(released[k], j);
            }
}

Matrix<12> Kernel::stiffness() const
{
    return congruence(m_end, constant(m_beam.geometry.endTransform()));
}

Matrix<6> Kernel::complianceAt(double position) const
{
    return compliance(m_beam, m_first, m_other, position);
}

std::array<MixedScalar, 12> Kernel::endForces(std::span<const double> grid, MixedScalar thermalStrain,
                                           std::span<const BeamLineLoad> lineLoads) const
{
    std::array<double, 12> values{};
    std::copy(grid.begin(), grid.end(), values.begin());
    const auto ends = m_beam.geometry.endTransform() * values;
    std::array<MixedScalar, 12> result{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        for (std::size_t j = 0; j < 12; ++j)
            result[i] += m_end(i, j) * ends[j];
        // C^T K_raw e_thermal = K_condensed e_thermal: released columns cancel exactly.
        result[i] -= m_end(i, 6) * thermalStrain * m_beam.geometry.length;
    }
    if (!lineLoads.empty())
    {
        const auto raw = lineEndLoads(lineLoads);
        // K_condensed T u - C^T f_span: the release map also changes with the section/material fields.
        for (std::size_t i = 0; i < result.size(); ++i)
            for (std::size_t j = 0; j < raw.size(); ++j)
                result[i] -= m_completion(j, i) * raw[j];
    }
    return result;
}

Matrix<12> Kernel::differential(std::span<const double> grid, MixedScalar thermalStrain) const
{
    const auto force = endForces(grid, thermalStrain)[6];
    const auto unit = beamEndDifferentialStiffness(1.0, m_beam.geometry.length);
    Matrix<12> result;
    for (std::size_t k = 0; k < result.values.size(); ++k)
        result.values[k] = force * unit.values[k];
    return congruence(congruence(result, m_completion), constant(m_beam.geometry.endTransform()));
}

Station Kernel::at(double position) const
{
    return stationOf(m_beam.section.at(position), m_first.section.at(position), m_other.section.at(position));
}

Matrix<12> massOf(const BeamData& beam, const Direction& first, const Direction& other,
                  double addedMass, MassFormulation formulation, TorsionalInertia torsion, bool onOffsets)
{
    const auto length = beam.geometry.length;
    const auto withoutNsm = [](BeamSection section)
    {
        for (std::size_t k = 0; k < section.count; ++k)
            section.stations[k].nonStructuralMass = 0.0;
        return section;
    };
    // BeamMass is affine in the station properties at fixed density. Unit-density structural matrices and
    // zero-density NSM matrices give their coefficients exactly, without subtracting nearby sections.
    const auto structural = beamMass(withoutNsm(beam.section), 1.0, 0.0, length, formulation, torsion);
    const auto p = beamMass(withoutNsm(first.section), 1.0, 0.0, length, formulation, torsion);
    const auto q = beamMass(withoutNsm(other.section), 1.0, 0.0, length, formulation, torsion);
    const auto nsm = beamMass(beam.section, 0.0, addedMass, length, formulation, torsion);
    const auto nsmP = beamMass(first.section, 0.0, 0.0, length, formulation, torsion);
    const auto nsmQ = beamMass(other.section, 0.0, 0.0, length, formulation, torsion);
    const MixedScalar density{beam.material.density, first.material.density, other.material.density};
    Matrix<12> structuralRates, nsmRates;
    for (std::size_t k = 0; k < structural.values.size(); ++k)
    {
        structuralRates.values[k] = density * MixedScalar(structural.values[k], p.values[k], q.values[k]);
        nsmRates.values[k] = {nsm.values[k], nsmP.values[k], nsmQ.values[k]};
    }
    const auto last = beam.section.count - 1;
    const auto a = stationOf(beam.section.stations[0], first.section.stations[0], other.section.stations[0]);
    const auto b = stationOf(beam.section.stations[last], first.section.stations[last], other.section.stations[last]);
    auto result = congruence(structuralRates, pointTransform(a, b, true));
    const auto added = congruence(nsmRates, pointTransform(a, b, false));
    for (std::size_t k = 0; k < result.values.size(); ++k)
        result.values[k] += added.values[k];
    return congruence(result, constant(onOffsets ? beam.geometry.endTransform() : beam.geometry.rotationTransform()));
}

} // namespace oos::elements::beam_sizing
