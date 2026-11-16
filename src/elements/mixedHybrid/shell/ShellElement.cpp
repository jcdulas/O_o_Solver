#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"

#include "ShellElementDetail.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <algorithm>

namespace oos::elements::mixedHybrid
{

using shelldetail::EdgePointGeometry;
using shelldetail::VOIGT_PAIRS;

namespace
{

// A point of the map whose Jacobian is at most this fraction of the largest makes the map not invertible.
constexpr double JACOBIAN_RATIO = 1e-10;

// The column c of T(−φ): the constant strain c of the frame of the constant states in local Voigt components.
Voigt3 constantStrain(double phi, std::size_t c)
{
    const Matrix33 t = strainTransformation(-phi);
    return {t[0][c], t[1][c], t[2][c]};
}

// The curvature of every unknown at surface point q in local Voigt components (3 × unknownCount):
//   κ_αβ = −ν · (∂_αβ u − Γ^λ_αβ ∂_λ u) for the displacement, ½ (g_α,β + g_β,α) − Γ^λ_αβ g_λ for the shear.
Matrix curvatureOperator(const ShellReferenceElement& re, const ShellElementDefinition& def, std::size_t q,
                         const PointGeometry& g, std::size_t unknownCount)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const Matrix33 tm = covariantToVoigt(g.cov());
    Matrix kappa(3, unknownCount);
    for (std::size_t i = 0; i < n1; ++i)
    {
        const ReferenceJet& h = def.volumeJets[q * n1 + i];
        const Matrix22 d2{{{h[3], h[4]}, {h[4], h[5]}}};
        Voigt3 kc{};
        for (std::size_t r = 0; r < 3; ++r)
        {
            const std::size_t al = VOIGT_PAIRS[r][0];
            const std::size_t be = VOIGT_PAIRS[r][1];
            kc[r] = d2[al][be] - g.christoffel[0][al][be] * h[1] - g.christoffel[1][al][be] * h[2];
        }
        const Voigt3 kv = times33(tm, kc);
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t r = 0; r < 3; ++r)
                kappa(r, 3 * i + c) = -g.nu[c] * kv[r];
    }
    for (std::size_t i = 0; i < nc; ++i)
    {
        const std::array<double, 6>& h = re.hcurl[q * nc + i];
        const Matrix22 dg{{{h[2], h[3]}, {h[4], h[5]}}};
        Voigt3 kc{};
        for (std::size_t r = 0; r < 3; ++r)
        {
            const std::size_t al = VOIGT_PAIRS[r][0];
            const std::size_t be = VOIGT_PAIRS[r][1];
            kc[r] = 0.5 * (dg[al][be] + dg[be][al]) - g.christoffel[0][al][be] * h[0] - g.christoffel[1][al][be] * h[1];
        }
        const Voigt3 kv = times33(tm, kc);
        for (std::size_t r = 0; r < 3; ++r)
            kappa(r, 3 * n1 + i) = kv[r];
    }
    return kappa;
}

// The transverse shear strain of every unknown at surface point q in local components (2 × unknownCount):
// γ_I = c[α][I] g_α.
Matrix shearOperator(const ShellReferenceElement& re, std::size_t q, const PointGeometry& g, std::size_t unknownCount)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const Matrix22 cv = g.cov();
    Matrix gl(2, unknownCount);
    for (std::size_t i = 0; i < nc; ++i)
    {
        const std::array<double, 6>& h = re.hcurl[q * nc + i];
        for (std::size_t ii = 0; ii < 2; ++ii)
            gl(ii, 3 * n1 + i) = cv[0][ii] * h[0] + cv[1][ii] * h[1];
    }
    return gl;
}

// The matrix of the membrane functionals on the element's membrane basis (reggeCount × reggeCount).
Matrix reggeFunctionalMatrix(const ShellReferenceElement& re, const ShellElementDefinition& def,
                             const std::vector<PointGeometry>& geometries,
                             const std::vector<std::vector<EdgePointGeometry>>& edges)
{
    const std::size_t nr = re.reggeCount;
    Matrix functionals(nr, nr);
    for (std::size_t m = 0; m < nr; ++m)
    {
        Vector column;
        if (m < nr - 3 || def.mapReferenceConstants)
        {
            column = reggeFunctionals(
                re, geometries, [&](std::size_t q) { return re.regge[q * nr + m]; },
                [&](std::size_t k, std::size_t gi)
                {
                    const std::array<double, 3>& r = re.reggeEdge[k][gi * nr + m];
                    const std::array<double, 2>& tv = edges[k][gi].tv;
                    return r[0] * tv[0] * tv[0] + r[1] * tv[1] * tv[1] + 2.0 * r[2] * tv[0] * tv[1];
                },
                def.mapReferenceConstants);
        }
        else
        {
            const std::size_t c = m - (nr - 3);
            column = reggeFunctionals(
                re, geometries, [&](std::size_t q)
                { return voigtToCovariant(geometries[q].con(), constantStrain(geometries[q].phi, c)); },
                [&](std::size_t k, std::size_t gi)
                {
                    const EdgePointGeometry& point = edges[k][gi];
                    const Voigt3 u = constantStrain(point.g.phi, c);
                    const double x1 = dot3(point.xt, point.g.e[0]);
                    const double x2 = dot3(point.xt, point.g.e[1]);
                    return u[0] * x1 * x1 + u[1] * x2 * x2 + u[2] * x1 * x2;
                },
                def.mapReferenceConstants);
        }
        functionals.setColumn(m, column);
    }
    return functionals;
}

// The membrane functionals of the strain of every displacement unknown (reggeCount × unknownCount):
// ε_αβ = ½ (a_α · ∂_β u + a_β · ∂_α u).
Matrix membraneStrainFunctionals(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                 const std::vector<PointGeometry>& geometries,
                                 const std::vector<std::vector<EdgePointGeometry>>& edges, std::size_t unknownCount)
{
    const std::size_t n1 = re.h1Count;
    Matrix out(re.reggeCount, unknownCount);
    for (std::size_t i = 0; i < n1; ++i)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            const Vector column = reggeFunctionals(
                re, geometries,
                [&](std::size_t q) -> std::array<double, 3>
                {
                    const ReferenceJet& h = def.volumeJets[q * n1 + i];
                    const PointGeometry& g = geometries[q];
                    return {g.a[0][c] * h[1], g.a[1][c] * h[2], 0.5 * (g.a[0][c] * h[2] + g.a[1][c] * h[1])};
                },
                [&](std::size_t k, std::size_t gi)
                {
                    const EdgeJet& h = def.edgeJets[k][gi * n1 + i];
                    const EdgePointGeometry& point = edges[k][gi];
                    return point.xt[c] * (h[1] * point.tv[0] + h[2] * point.tv[1]);
                },
                def.mapReferenceConstants);
            out.setColumn(3 * i + c, column);
        }
    }
    return out;
}

} // namespace

std::size_t shellUnknownCount(const ShellReferenceElement& re)
{
    return 3 * re.h1Count + re.hcurlCount + re.cornerCount * (re.shearDegree + 1);
}

std::pair<Matrix, Matrix> pointKinematics(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                          std::size_t q, const PointGeometry& g, std::size_t unknownCount)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    Matrix uu(3, unknownCount);
    Matrix bb(3, unknownCount);
    for (std::size_t i = 0; i < n1; ++i)
    {
        const ReferenceJet& t = def.volumeJets[q * n1 + i];
        const SpaceVector grad = add3(scl3(g.ac[0], t[1]), scl3(g.ac[1], t[2]));
        for (std::size_t c = 0; c < 3; ++c)
        {
            uu(c, 3 * i + c) = t[0];
            for (std::size_t r = 0; r < 3; ++r)
                bb(r, 3 * i + c) = -g.nu[c] * grad[r];
        }
    }
    for (std::size_t i = 0; i < nc; ++i)
    {
        const std::array<double, 6>& t = re.hcurl[q * nc + i];
        const SpaceVector v = add3(scl3(g.ac[0], t[0]), scl3(g.ac[1], t[1]));
        for (std::size_t r = 0; r < 3; ++r)
            bb(r, 3 * n1 + i) = v[r];
    }
    return {std::move(uu), std::move(bb)};
}

Matrix reggeLocal(const ShellReferenceElement& re, const ReferenceTensorValues& values, const PointGeometry& g,
                  bool mapReferenceConstants)
{
    const std::size_t nr = re.reggeCount;
    const Matrix33 tm = covariantToVoigt(g.cov());
    Matrix rv(3, nr);
    const std::size_t mapped = mapReferenceConstants ? nr : nr - 3;
    for (std::size_t m = 0; m < mapped; ++m)
    {
        const Voigt3 local = times33(tm, values(m));
        for (std::size_t r = 0; r < 3; ++r)
            rv(r, m) = local[r];
    }
    if (mapReferenceConstants)
        return rv;
    const Matrix33 t = strainTransformation(-g.phi);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t r = 0; r < 3; ++r)
            rv(r, nr - 3 + c) = t[r][c];
    return rv;
}

Matrix momentsLocal(const ShellReferenceElement& re, const ReferenceTensorValues& values, const PointGeometry& g,
                    bool mapReferenceConstants)
{
    const std::size_t ns = re.momentCount;
    const Matrix33 ts = contravariantToLocal(g.con(), g.j);
    Matrix sv(3, ns);
    const std::size_t mapped = mapReferenceConstants ? ns : ns - 3;
    for (std::size_t jj = 0; jj < mapped; ++jj)
    {
        const Voigt3 local = times33(ts, values(jj));
        for (std::size_t r = 0; r < 3; ++r)
            sv(r, jj) = local[r];
    }
    if (mapReferenceConstants)
        return sv;
    const Matrix33 t = strainTransformation(g.phi);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t r = 0; r < 3; ++r)
            sv(r, ns - 3 + c) = t[c][r];
    return sv;
}

Vector reggeFunctionals(const ShellReferenceElement& re, std::span<const PointGeometry> geometries,
                        const CovariantStrainAtPoint& volume, const TangentialStrainAtEdge& edge,
                        bool mapReferenceConstants)
{
    const std::size_t nv = re.cornerCount;
    const std::size_t p = re.shearDegree;
    const std::size_t nri = re.reggeInteriorMoments.size();
    Vector out(re.reggeCount, 0.0);
    std::size_t row = 0;
    for (std::size_t k = 0; k < nv; ++k)
    {
        for (std::size_t l = 0; l <= p; ++l)
        {
            double sum = 0.0;
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
                sum += re.edgeWeights[gi] * edge(k, gi) * shelldetail::edgeLegendre(l, re.edgeParameters[gi]);
            out[row++] = sum;
        }
    }
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const std::array<double, 3> e = volume(q);
        for (std::size_t m = 0; m < nri; ++m)
            out[row + m] += re.weights[q] * re.reggeTest[q * nri + m] * e[re.reggeInteriorMoments[m].component];
        if (mapReferenceConstants)
        {
            for (std::size_t c = 0; c < 3; ++c)
                out[row + nri + c] += re.weights[q] * e[c];
        }
        else
        {
            // the work of the three constant membrane forces of the frame of the constant states
            const Voigt3 v = times33(covariantToVoigt(geometries[q].cov()), e);
            const Matrix33 t = strainTransformation(geometries[q].phi);
            for (std::size_t c = 0; c < 3; ++c)
            {
                const double vf = t[c][0] * v[0] + t[c][1] * v[1] + t[c][2] * v[2];
                out[row + nri + c] += re.weights[q] * geometries[q].j * vf;
            }
        }
    }
    return out;
}

std::variant<ShellElementMatrices, std::string> buildShellElement(const ShellReferenceElement& re,
                                                                  const ShellElementDefinition& def, bool withMass)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const std::size_t ns = re.momentCount;
    const std::size_t nr = re.reggeCount;
    const std::size_t nv = re.cornerCount;
    const std::size_t p = re.shearDegree;
    const std::size_t nx = shellUnknownCount(re);
    const std::size_t nq = re.points.size();
    const auto ue = [&](std::size_t i, std::size_t c) { return 3 * i + c; };
    const auto ge = [&](std::size_t i) { return 3 * n1 + i; };
    const auto ae = [&](std::size_t k, std::size_t l) { return 3 * n1 + nc + k * (p + 1) + l; };

    const std::vector<PointGeometry> geometries = shelldetail::volumeGeometries(re, def);
    double jmax = 0.0;
    for (const PointGeometry& g : geometries)
        jmax = std::max(jmax, g.j);
    for (const PointGeometry& g : geometries)
        if (!(g.j > JACOBIAN_RATIO * jmax))
            return std::string("the map of the element is not invertible (its Jacobian vanishes inside)");
    const std::vector<std::vector<EdgePointGeometry>> edges = shelldetail::edgeGeometries(re, def);

    // ---- the Regge interpolation: the functionals of the element's membrane basis and of e(u)
    std::optional<Matrix> reggeInverse = equilibratedInverse(reggeFunctionalMatrix(re, def, geometries, edges));
    if (!reggeInverse)
        return std::string("membrane functionals not unisolvent on the element");
    const Matrix interpolation = *reggeInverse * membraneStrainFunctionals(re, def, geometries, edges, nx);

    // ---- the point loop: membrane, shear, moments, mass
    Matrix kr(nr, nr);
    Matrix ks(nx, nx);
    Matrix msig(ns, ns);
    Matrix gmat(ns, nx);
    Matrix cr(ns, nr);
    std::optional<Matrix> mass;
    if (withMass)
        mass = Matrix(nx, nx);
    bool coupled = false;
    for (std::size_t q = 0; q < nq; ++q)
    {
        const PointGeometry& g = geometries[q];
        const double wj = re.weights[q] * g.j;
        const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, re.points[q][0], re.points[q][1], g);
        const SectionLaws laws = sectionLaws(sp);
        for (const auto& row : laws.b)
            for (double v : row)
                if (v != 0.0)
                    coupled = true;
        const Matrix rv =
            reggeLocal(re, [&](std::size_t m) { return re.regge[q * nr + m]; }, g, def.mapReferenceConstants);
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        shelldetail::addCongruence(kr, rv, laws.abar, wj);
        shelldetail::addCongruence(msig, sv, laws.dinv, wj);
        if (coupled)
            shelldetail::addTransposeProduct33(cr, sv, laws.dinvBt, rv, wj);
        // the interior part of b_T: ∫ m : κ dA, κ in local Voigt components
        shelldetail::addTransposeProduct(gmat, sv, curvatureOperator(re, def, q, g, nx), wj);
        if (!def.kirchhoff)
            shelldetail::addCongruence2(ks, shearOperator(re, q, g, nx), laws.h, wj);
        if (mass)
        {
            const auto [uu, bb] = pointKinematics(re, def, q, g, nx);
            shelldetail::addTransposeProduct(*mass, uu, uu, sp.mass[0] * wj);
            shelldetail::addTransposeProduct(*mass, uu, bb, sp.mass[1] * wj);
            shelldetail::addTransposeProduct(*mass, bb, uu, sp.mass[1] * wj);
            shelldetail::addTransposeProduct(*mass, bb, bb, sp.mass[2] * wj);
        }
    }

    // ---- the boundary part of b_T: −∫ m_μμ (β · μ − a_T) ds
    for (std::size_t k = 0; k < nv; ++k)
    {
        const double sk = def.edgeSigns[k];
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const EdgePointGeometry& point = edges[k][gi];
            const PointGeometry& g = point.g;
            const double len = norm3(point.xt);
            const SpaceVector tangent = scl3(point.xt, 1.0 / len);
            const SpaceVector mu = cross3(tangent, g.nu);
            const double mc[2] = {dot3(mu, g.ac[0]), dot3(mu, g.ac[1])};
            const double w = re.edgeWeights[gi] * len;
            const std::vector<double> mmu = shelldetail::normalMoments(re, k, gi, g, mu, def.mapReferenceConstants);
            Vector column(nx, 0.0);
            for (std::size_t i = 0; i < n1; ++i)
            {
                const EdgeJet& h = def.edgeJets[k][gi * n1 + i];
                const double dmu = h[1] * mc[0] + h[2] * mc[1];
                for (std::size_t c = 0; c < 3; ++c)
                    column[ue(i, c)] = -g.nu[c] * dmu;
            }
            for (std::size_t i = 0; i < nc; ++i)
            {
                const std::array<double, 2>& h = re.hcurlEdge[k][gi * nc + i];
                column[ge(i)] = h[0] * mc[0] + h[1] * mc[1];
            }
            // a_T = s_k â / (|X_τ| W²): the pairing is in the reference measure over W², so that the rotation of a
            // rigid motion (W² θ · X_τ, of degree 2 on a conic edge, p on a polynomial one) is in the space
            const double tau = re.edgeParameters[gi];
            const double te = sk > 0.0 ? tau : 1.0 - tau;
            const double we = def.edgeWeights[k][gi];
            for (std::size_t l = 0; l <= p; ++l)
                column[ae(k, l)] = -sk * shelldetail::edgeLegendre(l, te) / (len * we * we);
            for (std::size_t j = 0; j < ns; ++j)
            {
                const double f = -w * mmu[j];
                if (f == 0.0)
                    continue;
                std::span<double> row = gmat.rowSpan(j);
                for (std::size_t i = 0; i < nx; ++i)
                    if (column[i] != 0.0)
                        row[i] += f * column[i];
            }
        }
    }

    // M_σ equilibrated (unit diagonal) and factorised: K = Yᵀ Y with Y = L⁻¹ S (G + C); the moments are
    // m = S (L Lᵀ)⁻¹ S (G x − h). The explicit inverse would lose the digits that the scales of the Piola map spread
    // over the diagonal (elongated elements).
    const std::optional<EquilibratedCholesky> factor = EquilibratedCholesky::factor(msig);
    if (!factor)
        return std::string("moment block of the element is not positive definite (degenerate element)");
    ShellElementMatrices out;
    out.unknownCount = nx;
    out.complianceInverse = factor->inverse();
    out.pairing = coupled ? gmat + cr * interpolation : std::move(gmat);
    const Matrix y = factor->halfSolve(out.pairing);
    Matrix stiffness = transposeTimes(interpolation, kr * interpolation);
    stiffness += ks;
    stiffness += transposeTimes(y, y);
    shelldetail::symmetrise(stiffness);
    out.stiffness = std::move(stiffness);
    out.mass = std::move(mass);
    out.reggeInterpolation = interpolation;
    out.reggeFunctionalInverse = std::move(*reggeInverse);
    return out;
}

} // namespace oos::elements::mixedHybrid
