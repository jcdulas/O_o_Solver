#include "oos/elements/mixedHybrid/shell/ShellElementLoads.hpp"

#include "ShellElementDetail.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

namespace oos::elements::mixedHybrid
{

Vector areaLoad(const ShellReferenceElement& re, const ShellElementDefinition& def, const AreaForceAt& force)
{
    const std::size_t n1 = re.h1Count;
    Vector out(shellUnknownCount(re), 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, n1, q);
        const SpaceVector v = force(g);
        const double s = re.weights[q] * g.j;
        for (std::size_t i = 0; i < n1; ++i)
        {
            const double n = def.volumeJets[q * n1 + i][0] * s;
            for (std::size_t c = 0; c < 3; ++c)
                out[3 * i + c] += n * v[c];
        }
    }
    return out;
}

ShellThermalLoad thermalLoad(const ShellReferenceElement& re, const ShellElementDefinition& def,
                             const ShellElementMatrices& matrices, const TemperaturesAt& temperatures)
{
    const std::size_t ns = re.momentCount;
    const std::size_t nr = re.reggeCount;
    Vector fr(nr, 0.0);
    Vector h(ns, 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, re.h1Count, q);
        const std::array<double, 2> t = temperatures(g);
        if (t[0] == 0.0 && t[1] == 0.0)
            continue;
        const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, re.points[q][0], re.points[q][1], g);
        const SectionLaws laws = sectionLaws(sp);
        const auto [e0, k0] = freeStrains(sp, g.curvature(), t);
        const double s = re.weights[q] * g.j;
        const Matrix rv =
            reggeLocal(re, [&](std::size_t m) { return re.regge[q * nr + m]; }, g, def.mapReferenceConstants);
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        const Voigt3 ae0 = times33(laws.abar, e0);
        const Voigt3 dbe0 = times33(laws.dinvBt, e0);
        const Voigt3 hk{k0[0] + dbe0[0], k0[1] + dbe0[1], k0[2] + dbe0[2]};
        for (std::size_t m = 0; m < nr; ++m)
            fr[m] += s * (rv(0, m) * ae0[0] + rv(1, m) * ae0[1] + rv(2, m) * ae0[2]);
        for (std::size_t j = 0; j < ns; ++j)
            h[j] += s * (sv(0, j) * hk[0] + sv(1, j) * hk[1] + sv(2, j) * hk[2]);
    }
    ShellThermalLoad load;
    load.forces = transposeTimes(matrices.reggeInterpolation, fr);
    const Vector pairedMoments = transposeTimes(matrices.pairing, matrices.complianceInverse * h);
    for (std::size_t i = 0; i < load.forces.size(); ++i)
        load.forces[i] += pairedMoments[i];
    load.momentTerm = std::move(h);
    load.membraneTerm = std::move(fr);
    return load;
}

Vector inertiaLoad(const ShellReferenceElement& re, const ShellElementDefinition& def, const SpaceVector& acceleration)
{
    const std::size_t nx = shellUnknownCount(re);
    Vector out(nx, 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, re.h1Count, q);
        const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, re.points[q][0], re.points[q][1], g);
        const auto [uu, bb] = pointKinematics(re, def, q, g, nx);
        const double s = re.weights[q] * g.j;
        for (std::size_t i = 0; i < nx; ++i)
        {
            double ua = 0.0;
            double ba = 0.0;
            for (std::size_t r = 0; r < 3; ++r)
            {
                ua += uu(r, i) * acceleration[r];
                ba += bb(r, i) * acceleration[r];
            }
            out[i] += (ua * sp.mass[0] + ba * sp.mass[1]) * s;
        }
    }
    return out;
}

Matrix followerPressure(const ShellReferenceElement& re, const ShellElementDefinition& def, const PressureAt& pressure)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nx = shellUnknownCount(re);
    Matrix out(nx, nx);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, n1, q);
        const double pw = pressure(g) * re.weights[q];
        if (pw == 0.0)
            continue;
        for (std::size_t i = 0; i < n1; ++i)
        {
            const ReferenceJet& a = def.volumeJets[q * n1 + i];
            for (std::size_t c = 0; c < 3; ++c)
            {
                // the change of φ_,1 × φ_,2 when φ moves by N_i e_c
                SpaceVector ec{};
                ec[c] = a[1];
                SpaceVector ed{};
                ed[c] = a[2];
                const SpaceVector v = add3(cross3(ec, g.a[1]), cross3(g.a[0], ed));
                for (std::size_t j = 0; j < n1; ++j)
                {
                    const double nj = def.volumeJets[q * n1 + j][0];
                    for (std::size_t d = 0; d < 3; ++d)
                        out(3 * j + d, 3 * i + c) += pw * v[d] * nj;
                }
            }
        }
    }
    shelldetail::symmetrise(out);
    return out;
}

} // namespace oos::elements::mixedHybrid
