#include "oos/elements/mixedHybrid/shell/ShellElementStresses.hpp"

#include "ShellElementDetail.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

// The moment coefficients M_σ⁻¹ ((G + C) x − h).
Vector momentCoefficients(const ShellElementMatrices& matrices, std::span<const double> x, const Vector* momentTerm)
{
    Vector rhs = matrices.pairing * x;
    if (momentTerm)
        for (std::size_t i = 0; i < rhs.size(); ++i)
            rhs[i] -= (*momentTerm)[i];
    return matrices.complianceInverse * rhs;
}

// N = Ā e + B D⁻¹ m.
Voigt3 membraneForce(const SectionLaws& laws, const Voigt3& strain, const Voigt3& moment)
{
    const Voigt3 ae = times33(laws.abar, strain);
    const Voigt3 bdm = times33(laws.b, times33(laws.dinv, moment));
    return {ae[0] + bdm[0], ae[1] + bdm[1], ae[2] + bdm[2]};
}

std::array<double, 3> evaluate3(const Field& f, double xi, double eta)
{
    return {f[0].evaluate(xi, eta), f[1].evaluate(xi, eta), f[2].evaluate(xi, eta)};
}

} // namespace

std::array<double, 8> resultants(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                 const ShellElementMatrices& matrices, std::span<const double> x,
                                 const Vector* momentTerm, const std::array<double, 2>& temperatures, double xi,
                                 double eta)
{
    const PointGeometry g = geometryAt(re, def, xi, eta);
    const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, xi, eta, g);
    const SectionLaws laws = sectionLaws(sp);
    const Vector mc = momentCoefficients(matrices, x, momentTerm);
    const Matrix sv = momentsLocal(
        re, [&](std::size_t j) { return evaluate3(re.momentBasis[j], xi, eta); }, g, def.mapReferenceConstants);
    const Voigt3 m = shelldetail::times3(sv, mc);
    const Vector rc = matrices.reggeInterpolation * x;
    const Matrix rv = reggeLocal(
        re, [&](std::size_t k) { return evaluate3(re.reggeBasis[k], xi, eta); }, g, def.mapReferenceConstants);
    Voigt3 e = shelldetail::times3(rv, rc);
    if (temperatures[0] != 0.0 || temperatures[1] != 0.0)
    {
        const Voigt3 e0 = freeStrains(sp, g.curvature(), temperatures).first;
        for (std::size_t i = 0; i < 3; ++i)
            e[i] -= e0[i];
    }
    const Voigt3 n = membraneForce(laws, e, m);
    const Matrix22 cv = g.cov();
    std::array<double, 2> gamma{};
    for (std::size_t i = 0; i < re.hcurlBasis.size(); ++i)
    {
        const Field& f = re.hcurlBasis[i];
        const double v[2] = {f[0].evaluate(xi, eta), f[1].evaluate(xi, eta)};
        const double xg = x[3 * re.h1Count + i];
        for (std::size_t ii = 0; ii < 2; ++ii)
            gamma[ii] += (cv[0][ii] * v[0] + cv[1][ii] * v[1]) * xg;
    }
    const double q0 = laws.h[0][0] * gamma[0] + laws.h[0][1] * gamma[1];
    const double q1 = laws.h[1][0] * gamma[0] + laws.h[1][1] * gamma[1];
    return {n[0], n[1], n[2], m[0], m[1], m[2], q0, q1};
}

std::vector<Voigt3> membraneForces(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                   const ShellElementMatrices& matrices, std::span<const double> x,
                                   const Vector* momentTerm, const TemperaturesAt& temperatures)
{
    const Vector mc = momentCoefficients(matrices, x, momentTerm);
    const Vector rc = matrices.reggeInterpolation * x;
    std::vector<Voigt3> out;
    out.reserve(re.points.size());
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry g = volumeGeometry(def, re.h1Count, q);
        const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, re.points[q][0], re.points[q][1], g);
        const SectionLaws laws = sectionLaws(sp);
        const Matrix sv = momentsLocal(
            re, [&](std::size_t jj) { return re.moments[q * re.momentCount + jj]; }, g, def.mapReferenceConstants);
        const Matrix rv = reggeLocal(
            re, [&](std::size_t m) { return re.regge[q * re.reggeCount + m]; }, g, def.mapReferenceConstants);
        const Voigt3 m = shelldetail::times3(sv, mc);
        Voigt3 e = shelldetail::times3(rv, rc);
        const std::array<double, 2> t = temperatures ? temperatures(g) : std::array<double, 2>{0.0, 0.0};
        if (t[0] != 0.0 || t[1] != 0.0)
        {
            const Voigt3 e0 = freeStrains(sp, g.curvature(), t).first;
            for (std::size_t i = 0; i < 3; ++i)
                e[i] -= e0[i];
        }
        out.push_back(membraneForce(laws, e, m));
    }
    return out;
}

Matrix geometricMembrane(const ShellReferenceElement& re, const ShellElementDefinition& def,
                         const ShellElementMatrices& matrices, std::span<const Voigt3> forces)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nr = re.reggeCount;
    const std::size_t nv = re.cornerCount;
    const std::size_t p = re.shearDegree;
    const std::size_t nx = shellUnknownCount(re);
    const std::vector<PointGeometry> geometries = shelldetail::volumeGeometries(re, def);
    // n_R,m = ∫ N : R_m dA; the weights of the functionals are D_R⁻ᵀ n_R
    Vector nrv(nr, 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const Matrix rv = reggeLocal(
            re, [&](std::size_t m) { return re.regge[q * nr + m]; }, geometries[q], def.mapReferenceConstants);
        const double s = re.weights[q] * geometries[q].j;
        for (std::size_t m = 0; m < nr; ++m)
            nrv[m] += s * (rv(0, m) * forces[q][0] + rv(1, m) * forces[q][1] + rv(2, m) * forces[q][2]);
    }
    const Vector wf = transposeTimes(matrices.reggeFunctionalInverse, nrv);
    Matrix ks(n1, n1);
    std::size_t f = 0;
    for (std::size_t k = 0; k < nv; ++k)
    {
        for (std::size_t l = 0; l <= p; ++l)
        {
            const double wl = wf[f++];
            if (wl == 0.0)
                continue;
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
            {
                const double tau = re.edgeParameters[gi];
                const auto tv = referenceEdgePointAndDirection(nv, k, tau).second;
                const double c = wl * re.edgeWeights[gi] * shelldetail::edgeLegendre(l, tau);
                std::vector<double> dt(n1);
                for (std::size_t i = 0; i < n1; ++i)
                {
                    const EdgeJet& h = def.edgeJets[k][gi * n1 + i];
                    dt[i] = h[1] * tv[0] + h[2] * tv[1];
                }
                for (std::size_t i = 0; i < n1; ++i)
                    for (std::size_t j = 0; j < n1; ++j)
                        ks(i, j) += c * dt[i] * dt[j];
            }
        }
    }
    const std::size_t nri = re.reggeInteriorMoments.size();
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const Matrix33 tm = covariantToVoigt(geometries[q].cov());
        // the weight of each covariant component of Δ, from the interior functionals
        std::array<double, 3> wc{};
        for (std::size_t m = 0; m < nri; ++m)
            wc[re.reggeInteriorMoments[m].component] += wf[f + m] * re.weights[q] * re.reggeTest[q * nri + m];
        for (std::size_t c = 0; c < 3; ++c)
        {
            if (def.mapReferenceConstants)
            {
                wc[c] += wf[f + nri + c] * re.weights[q];
            }
            else
            {
                const double s = wf[f + nri + c] * re.weights[q] * geometries[q].j;
                for (std::size_t comp = 0; comp < 3; ++comp)
                    wc[comp] += s * tm[c][comp];
            }
        }
        for (std::size_t i = 0; i < n1; ++i)
        {
            const ReferenceJet& a = def.volumeJets[q * n1 + i];
            for (std::size_t j = 0; j < n1; ++j)
            {
                const ReferenceJet& b = def.volumeJets[q * n1 + j];
                ks(i, j) += wc[0] * a[1] * b[1] + wc[1] * a[2] * b[2] + wc[2] * 0.5 * (a[1] * b[2] + a[2] * b[1]);
            }
        }
    }
    Matrix out(nx, nx);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t j = 0; j < n1; ++j)
            for (std::size_t c = 0; c < 3; ++c)
                out(3 * i + c, 3 * j + c) = ks(i, j);
    shelldetail::symmetrise(out);
    return out;
}

} // namespace oos::elements::mixedHybrid
