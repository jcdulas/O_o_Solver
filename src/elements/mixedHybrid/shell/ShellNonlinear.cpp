#include "oos/elements/mixedHybrid/shell/ShellNonlinear.hpp"

#include "ShellElementDetail.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

namespace oos::elements::mixedHybrid
{

using shelldetail::VOIGT_PAIRS;

namespace
{

// The variables of a surface point: φ_,1, φ_,2, φ_,11, φ_,22, φ_,12, three components each.
constexpr std::size_t POINT_VARIABLES = 15;
// The variables of an edge point: φ_,1, φ_,2.
constexpr std::size_t EDGE_VARIABLES = 6;
// The order of the jets of a displacement function in the point variables: ∂ξ, ∂η, ∂ξξ, ∂ηη, ∂ξη.
constexpr std::size_t JET_OF_VARIABLE[5] = {1, 2, 3, 5, 4};

template <std::size_t N>
HessianVec3<N> constantVector(const SpaceVector& v)
{
    return {HessianScalar<N>::constant(v[0]), HessianScalar<N>::constant(v[1]), HessianScalar<N>::constant(v[2])};
}

// The values (φ_,1, φ_,2, φ_,11, φ_,22, φ_,12) at surface point q for the displacement unknowns x.
std::array<double, POINT_VARIABLES> pointVariables(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                   const NonlinearPreparation& pre, std::size_t q,
                                                   std::span<const double> x)
{
    const std::size_t n1 = re.h1Count;
    const PointGeometry& g = pre.geometries[q];
    std::array<double, POINT_VARIABLES> v{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        v[c] = g.a[0][c];
        v[3 + c] = g.a[1][c];
        v[6 + c] = pre.secondDerivatives[q][0][c];
        v[9 + c] = pre.secondDerivatives[q][1][c];
        v[12 + c] = pre.secondDerivatives[q][2][c];
    }
    for (std::size_t i = 0; i < n1; ++i)
    {
        const ReferenceJet& j = def.volumeJets[q * n1 + i];
        for (std::size_t c = 0; c < 3; ++c)
        {
            const double u = x[3 * i + c];
            for (std::size_t d = 0; d < 5; ++d)
                v[3 * d + c] += u * j[JET_OF_VARIABLE[d]];
        }
    }
    return v;
}

// The values (φ_,1, φ_,2) at point g of edge k for the displacement unknowns x.
std::array<double, EDGE_VARIABLES> edgeVariables(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                 const NonlinearEdgePoint& point, std::size_t k, std::size_t gi,
                                                 std::span<const double> x)
{
    const std::size_t n1 = re.h1Count;
    std::array<double, EDGE_VARIABLES> v{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        v[c] = point.g.a[0][c];
        v[3 + c] = point.g.a[1][c];
    }
    for (std::size_t i = 0; i < n1; ++i)
    {
        const EdgeJet& j = def.edgeJets[k][gi * n1 + i];
        for (std::size_t c = 0; c < 3; ++c)
        {
            v[c] += x[3 * i + c] * j[1];
            v[3 + c] += x[3 * i + c] * j[2];
        }
    }
    return v;
}

// The membrane functionals of the Green strain at the state of the points (phi1, edgePhi) and their derivatives with
// respect to the displacement unknowns.
struct GreenStrainFunctionals
{
    Vector values;
    Matrix derivatives;
};

GreenStrainFunctionals greenStrainFunctionals(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                              const NonlinearPreparation& pre, const std::vector<SurfaceFrame>& phi1,
                                              const std::vector<std::vector<SurfaceFrame>>& edgePhi, bool geometric,
                                              std::size_t unknownCount)
{
    const std::size_t n1 = re.h1Count;
    const auto edgeTangent = [&](std::size_t k, std::size_t gi)
    {
        const std::array<double, 2>& tv = pre.edges[k][gi].tv;
        const SurfaceFrame& f = edgePhi[k][gi];
        return add3(scl3(f[0], tv[0]), scl3(f[1], tv[1]));
    };
    const ReggeFunctionalTables tables =
        ReggeFunctionalTables::build(re, pre.geometries, def.mapReferenceConstants);
    GreenStrainFunctionals out;
    if (geometric)
    {
        out.values.assign(re.reggeCount, 0.0);
    }
    else
    {
        out.values = reggeFunctionals(
            re, tables,
            [&](std::size_t q) -> std::array<double, 3>
            {
                const SurfaceFrame& f = phi1[q];
                const SurfaceFrame& a = pre.geometries[q].a;
                const auto e = [&](std::size_t al, std::size_t be)
                { return 0.5 * (dot3(f[al], f[be]) - dot3(a[al], a[be])); };
                return {e(0, 0), e(1, 1), e(0, 1)};
            },
            [&](std::size_t k, std::size_t gi)
            {
                const SpaceVector ft = edgeTangent(k, gi);
                const SpaceVector& xt = pre.edges[k][gi].xt;
                return 0.5 * (dot3(ft, ft) - dot3(xt, xt));
            },
            def.mapReferenceConstants);
    }
    out.derivatives = Matrix(re.reggeCount, unknownCount);
    for (std::size_t i = 0; i < n1; ++i)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            const Vector column = reggeFunctionals(
                re, tables,
                [&](std::size_t q) -> std::array<double, 3>
                {
                    const ReferenceJet& hj = def.volumeJets[q * n1 + i];
                    const SurfaceFrame& f = phi1[q];
                    return {f[0][c] * hj[1], f[1][c] * hj[2], 0.5 * (f[0][c] * hj[2] + f[1][c] * hj[1])};
                },
                [&](std::size_t k, std::size_t gi)
                {
                    const std::array<double, 2>& tv = pre.edges[k][gi].tv;
                    const EdgeJet& hj = def.edgeJets[k][gi * n1 + i];
                    return edgeTangent(k, gi)[c] * (hj[1] * tv[0] + hj[2] * tv[1]);
                },
                def.mapReferenceConstants);
            out.derivatives.setColumn(3 * i + c, column);
        }
    }
    return out;
}

// Σ_f w_f Q_f: the second variation of the membrane functionals of the Green strain weighted by w (n1 × n1, the same
// on each Cartesian component).
Matrix membraneSecondVariation(const ShellReferenceElement& re, const ShellElementDefinition& def,
                               const NonlinearPreparation& pre, const Vector& wf)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nv = re.cornerCount;
    Matrix ks(n1, n1);
    std::size_t f = 0;
    std::vector<double> dt(n1);
    for (std::size_t k = 0; k < nv; ++k)
    {
        for (std::size_t l = 0; l <= re.shearDegree; ++l)
        {
            const double wl = wf[f++];
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
            {
                const std::array<double, 2>& tv = pre.edges[k][gi].tv;
                const double cf = wl * re.edgeWeights[gi] * shelldetail::edgeLegendre(l, re.edgeParameters[gi]);
                for (std::size_t i = 0; i < n1; ++i)
                {
                    const EdgeJet& hj = def.edgeJets[k][gi * n1 + i];
                    dt[i] = hj[1] * tv[0] + hj[2] * tv[1];
                }
                for (std::size_t i = 0; i < n1; ++i)
                    for (std::size_t j = 0; j < n1; ++j)
                        ks(i, j) += cf * dt[i] * dt[j];
            }
        }
    }
    const std::size_t nri = re.reggeInteriorMoments.size();
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        std::array<double, 3> wc{};
        for (std::size_t m = 0; m < nri; ++m)
            wc[re.reggeInteriorMoments[m].component] += wf[f + m] * re.weights[q] * re.reggeTest[q * nri + m];
        if (def.mapReferenceConstants)
        {
            for (std::size_t cc = 0; cc < 3; ++cc)
                wc[cc] += wf[f + nri + cc] * re.weights[q];
        }
        else
        {
            const Matrix33 tm = covariantToVoigt(g.cov());
            const Matrix33 tt = strainTransformation(g.phi);
            for (std::size_t cc = 0; cc < 3; ++cc)
            {
                const double s = wf[f + nri + cc] * re.weights[q] * g.j;
                for (std::size_t comp = 0; comp < 3; ++comp)
                {
                    const double tmc = tt[cc][0] * tm[0][comp] + tt[cc][1] * tm[1][comp] + tt[cc][2] * tm[2][comp];
                    wc[comp] += s * tmc;
                }
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
    return ks;
}

// ∫ m_loc · T · H(κ) dA: the second variation of the bending measures in the element weighted by its moments.
void addBendingSecondVariation(const ShellReferenceElement& re, const ShellElementDefinition& def,
                               const NonlinearPreparation& pre,
                               const std::vector<std::array<HessianScalar<POINT_VARIABLES>, 3>>& measures,
                               const Vector& m, Matrix& kgeo)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t ns = re.momentCount;
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        const Voigt3 ml = shelldetail::times3(sv, m);
        const Voigt3 wcov = transposeTimes33(covariantToVoigt(g.cov()), ml);
        std::array<std::array<double, POINT_VARIABLES>, POINT_VARIABLES> hh{};
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t a = 0; a < POINT_VARIABLES; ++a)
                for (std::size_t b = 0; b < POINT_VARIABLES; ++b)
                    hh[a][b] += wcov[r] * measures[q][r].hessian[a][b];
        const double w = re.weights[q] * g.j;
        for (std::size_t j = 0; j < n1; ++j)
        {
            const ReferenceJet& jj = def.volumeJets[q * n1 + j];
            // hj[a][cj] = Σ_e hh[a][3e + cj] jet_j[e]
            std::array<std::array<double, 3>, POINT_VARIABLES> hj{};
            for (std::size_t a = 0; a < POINT_VARIABLES; ++a)
                for (std::size_t cj = 0; cj < 3; ++cj)
                    for (std::size_t e = 0; e < 5; ++e)
                        hj[a][cj] += hh[a][3 * e + cj] * jj[JET_OF_VARIABLE[e]];
            for (std::size_t i = 0; i < n1; ++i)
            {
                const ReferenceJet& ji = def.volumeJets[q * n1 + i];
                for (std::size_t ci = 0; ci < 3; ++ci)
                {
                    for (std::size_t cj = 0; cj < 3; ++cj)
                    {
                        double s = 0.0;
                        for (std::size_t d = 0; d < 5; ++d)
                            s += ji[JET_OF_VARIABLE[d]] * hj[3 * d + ci][cj];
                        kgeo(3 * i + ci, 3 * j + cj) += w * s;
                    }
                }
            }
        }
    }
}

// −∫ m_μμ H(angle) ds: the second variation of the kink angles weighted by the normal moments of the edges.
void addKinkSecondVariation(const ShellReferenceElement& re, const ShellElementDefinition& def,
                            const NonlinearPreparation& pre,
                            const std::vector<std::vector<HessianScalar<EDGE_VARIABLES>>>& angles, const Vector& m,
                            Matrix& kgeo)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t ns = re.momentCount;
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        for (std::size_t gi = 0; gi < pre.edges[k].size(); ++gi)
        {
            const NonlinearEdgePoint& point = pre.edges[k][gi];
            double mmu = 0.0;
            for (std::size_t jj = 0; jj < ns; ++jj)
                mmu += point.normalMoments[jj] * m[jj];
            const double w = -re.edgeWeights[gi] * point.length * mmu;
            const auto& hh = angles[k][gi].hessian;
            for (std::size_t i = 0; i < n1; ++i)
            {
                const EdgeJet& ji = def.edgeJets[k][gi * n1 + i];
                for (std::size_t j = 0; j < n1; ++j)
                {
                    const EdgeJet& jj = def.edgeJets[k][gi * n1 + j];
                    for (std::size_t ci = 0; ci < 3; ++ci)
                    {
                        for (std::size_t cj = 0; cj < 3; ++cj)
                        {
                            const double s = ji[1] * (hh[ci][cj] * jj[1] + hh[ci][3 + cj] * jj[2]) +
                                             ji[2] * (hh[3 + ci][cj] * jj[1] + hh[3 + ci][3 + cj] * jj[2]);
                            kgeo(3 * i + ci, 3 * j + cj) += w * s;
                        }
                    }
                }
            }
        }
    }
}

Vector subtract(Vector a, const Vector& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] -= b[i];
    return a;
}

Vector add(Vector a, const Vector& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] += b[i];
    return a;
}

} // namespace

std::array<HessianScalar<15>, 3> bendingMeasure(const std::array<double, 15>& variables, const std::array<double, 3>& b)
{
    using H = HessianScalar<POINT_VARIABLES>;
    std::array<H, POINT_VARIABLES> x;
    for (std::size_t i = 0; i < POINT_VARIABLES; ++i)
        x[i] = H::variable(variables[i], i);
    const HessianVec3<POINT_VARIABLES> a1{x[0], x[1], x[2]};
    const HessianVec3<POINT_VARIABLES> a2{x[3], x[4], x[5]};
    const HessianVec3<POINT_VARIABLES> nu = hessianUnit(hessianCross(a1, a2));
    const auto d = [&](std::size_t o) { return HessianVec3<POINT_VARIABLES>{x[o], x[o + 1], x[o + 2]}; };
    return {H::constant(b[0]) - hessianDot(d(6), nu), H::constant(b[1]) - hessianDot(d(9), nu),
            H::constant(b[2]) - hessianDot(d(12), nu)};
}

HessianScalar<6> edgeAngle(const std::array<double, 6>& variables, const std::array<double, 2>& tv,
                           const SpaceVector& t, const SpaceVector& nu)
{
    using H = HessianScalar<EDGE_VARIABLES>;
    using V = HessianVec3<EDGE_VARIABLES>;
    std::array<H, EDGE_VARIABLES> x;
    for (std::size_t i = 0; i < EDGE_VARIABLES; ++i)
        x[i] = H::variable(variables[i], i);
    const V a1{x[0], x[1], x[2]};
    const V a2{x[3], x[4], x[5]};
    const V normal = hessianUnit(hessianCross(a1, a2));
    V xt;
    for (std::size_t c = 0; c < 3; ++c)
        xt[c] = scale(a1[c], tv[0]) + scale(a2[c], tv[1]);
    const V tangent = hessianUnit(xt);
    const V tc = constantVector<EDGE_VARIABLES>(t);
    const V nc = constantVector<EDGE_VARIABLES>(nu);
    V s;
    for (std::size_t c = 0; c < 3; ++c)
        s[c] = tc[c] + tangent[c];
    // the reference normal carried by the minimal rotation from t to t̂
    const H f = hessianDot(s, nc) * reciprocal(H::constant(1.0) + hessianDot(tc, tangent));
    V rn;
    for (std::size_t c = 0; c < 3; ++c)
        rn[c] = nc[c] - s[c] * f;
    return atan2(hessianDot(hessianCross(rn, normal), tangent), hessianDot(rn, normal));
}

NonlinearPreparation prepareNonlinear(const ShellReferenceElement& re, const ShellElementDefinition& def)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t ns = re.momentCount;
    const std::size_t nr = re.reggeCount;
    const std::size_t nv = re.cornerCount;
    const std::size_t nx = shellUnknownCount(re);
    const std::size_t nq = re.points.size();
    NonlinearPreparation pre;
    pre.geometries = shelldetail::volumeGeometries(re, def);
    for (std::size_t q = 0; q < nq; ++q)
    {
        std::array<SpaceVector, 3> d{};
        for (std::size_t i = 0; i < n1; ++i)
        {
            const ReferenceJet& h = def.volumeJets[q * n1 + i];
            for (std::size_t c = 0; c < 3; ++c)
            {
                d[0][c] += def.mapCoefficients[i][c] * h[3];
                d[1][c] += def.mapCoefficients[i][c] * h[5];
                d[2][c] += def.mapCoefficients[i][c] * h[4];
            }
        }
        const SpaceVector& nu = pre.geometries[q].nu;
        pre.referenceCurvature.push_back({dot3(d[0], nu), dot3(d[1], nu), dot3(d[2], nu)});
        pre.secondDerivatives.push_back(d);
    }
    pre.edges.resize(nv);
    for (std::size_t k = 0; k < nv; ++k)
    {
        const double sk = def.edgeSigns[k];
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const double tau = re.edgeParameters[gi];
            NonlinearEdgePoint point;
            point.tv = referenceEdgePointAndDirection(nv, k, tau).second;
            point.g = edgeGeometry(def, n1, k, gi);
            point.xt = add3(scl3(point.g.a[0], point.tv[0]), scl3(point.g.a[1], point.tv[1]));
            point.length = norm3(point.xt);
            point.tangent = scl3(point.xt, 1.0 / point.length);
            const SpaceVector mu = cross3(point.tangent, point.g.nu);
            point.mc = {dot3(mu, point.g.ac[0]), dot3(mu, point.g.ac[1])};
            point.normalMoments = shelldetail::normalMoments(re, k, gi, point.g, mu, def.mapReferenceConstants);
            point.canonicalParameter = sk > 0.0 ? tau : 1.0 - tau;
            point.weight = def.edgeWeights[k][gi];
            point.previousTangent = point.tangent;
            point.previousNormal = point.g.nu;
            pre.edges[k].push_back(std::move(point));
        }
    }
    pre.kr = Matrix(nr, nr);
    pre.cr = Matrix(ns, nr);
    pre.ks = Matrix(nx, nx);
    // a membrane-only section without offset has no transverse shear (ShellElement.hpp)
    const bool sheared = !def.kirchhoff && sectionCarriesMoments(def.section->section().parts, def.offset);
    for (std::size_t q = 0; q < nq; ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        const double w = re.weights[q] * g.j;
        const SectionPoint sp = sectionAtGeometry(*def.section, def.offset, re.points[q][0], re.points[q][1], g);
        const SectionLaws laws = sectionLaws(sp);
        const Matrix rv =
            reggeLocal(re, [&](std::size_t m) { return re.regge[q * nr + m]; }, g, def.mapReferenceConstants);
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        shelldetail::addCongruence(pre.kr, rv, laws.abar, w);
        shelldetail::addTransposeProduct33(pre.cr, sv, laws.dinvBt, rv, w);
        if (sheared)
        {
            const Matrix22 cv = g.cov();
            Matrix gl(2, nx);
            for (std::size_t i = 0; i < re.hcurlCount; ++i)
            {
                const std::array<double, 6>& h = re.hcurl[q * re.hcurlCount + i];
                for (std::size_t ii = 0; ii < 2; ++ii)
                    gl(ii, 3 * n1 + i) = cv[0][ii] * h[0] + cv[1][ii] * h[1];
            }
            shelldetail::addCongruence2(pre.ks, gl, laws.h, w);
        }
    }
    return pre;
}

NonlinearEvaluation evaluateNonlinear(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                      const ShellElementMatrices& matrices, const NonlinearPreparation& pre,
                                      std::span<const double> x, const Vector* momentTerm, const Vector* membraneTerm,
                                      bool geometric)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nc = re.hcurlCount;
    const std::size_t ns = re.momentCount;
    const std::size_t nr = re.reggeCount;
    const std::size_t nv = re.cornerCount;
    const std::size_t nx = shellUnknownCount(re);
    const std::size_t nq = re.points.size();
    // the state at which the measures are differentiated
    const Vector zero(nx, 0.0);
    const std::span<const double> xs = geometric ? std::span<const double>(zero) : x;

    // ---- bending: g, its derivative, the Hessians of the measures
    Vector gv(ns, 0.0);
    Matrix gd(ns, nx);
    std::vector<std::array<HessianScalar<POINT_VARIABLES>, 3>> volumeMeasures;
    volumeMeasures.reserve(nq);
    std::vector<SurfaceFrame> phi1;
    phi1.reserve(nq);
    for (std::size_t q = 0; q < nq; ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        const std::array<double, POINT_VARIABLES> v = pointVariables(re, def, pre, q, xs);
        phi1.push_back({SpaceVector{v[0], v[1], v[2]}, SpaceVector{v[3], v[4], v[5]}});
        const std::array<HessianScalar<POINT_VARIABLES>, 3> kb = bendingMeasure(v, pre.referenceCurvature[q]);
        // the shear part (linear in the shear unknowns)
        Voigt3 kg{};
        std::vector<Voigt3> kgd(nc);
        for (std::size_t i = 0; i < nc; ++i)
        {
            const std::array<double, 6>& hc = re.hcurl[q * nc + i];
            const Matrix22 dg{{{hc[2], hc[3]}, {hc[4], hc[5]}}};
            for (std::size_t r = 0; r < 3; ++r)
            {
                const std::size_t al = VOIGT_PAIRS[r][0];
                const std::size_t be = VOIGT_PAIRS[r][1];
                const double value = 0.5 * (dg[al][be] + dg[be][al]) - g.christoffel[0][al][be] * hc[0] -
                                     g.christoffel[1][al][be] * hc[1];
                kgd[i][r] = value;
                kg[r] += value * x[3 * n1 + i];
            }
        }
        const Matrix33 tm = covariantToVoigt(g.cov());
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        const double w = re.weights[q] * g.j;
        if (!geometric)
        {
            const Voigt3 kl = times33(tm, {kb[0].value + kg[0], kb[1].value + kg[1], kb[2].value + kg[2]});
            for (std::size_t j = 0; j < ns; ++j)
                gv[j] += (sv(0, j) * kl[0] + sv(1, j) * kl[1] + sv(2, j) * kl[2]) * w;
        }
        // the derivative of the local curvature with respect to the unknowns
        Matrix kd(3, nx);
        for (std::size_t i = 0; i < n1; ++i)
        {
            const ReferenceJet& j = def.volumeJets[q * n1 + i];
            for (std::size_t c = 0; c < 3; ++c)
            {
                Voigt3 column{};
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t d = 0; d < 5; ++d)
                        column[r] += kb[r].gradient[3 * d + c] * j[JET_OF_VARIABLE[d]];
                const Voigt3 lc = times33(tm, column);
                for (std::size_t r = 0; r < 3; ++r)
                    kd(r, 3 * i + c) = lc[r];
            }
        }
        for (std::size_t i = 0; i < nc; ++i)
        {
            const Voigt3 lc = times33(tm, kgd[i]);
            for (std::size_t r = 0; r < 3; ++r)
                kd(r, 3 * n1 + i) = lc[r];
        }
        shelldetail::addTransposeProduct(gd, sv, kd, w);
        volumeMeasures.push_back(kb);
    }

    // ---- the kinks: the angle of each side about the edge, γ · μ and the hybrid unknown
    std::vector<std::vector<HessianScalar<EDGE_VARIABLES>>> edgeAngles(nv);
    std::vector<std::vector<SurfaceFrame>> edgePhi(nv);
    for (std::size_t k = 0; k < nv; ++k)
    {
        const double sk = def.edgeSigns[k];
        for (std::size_t gi = 0; gi < pre.edges[k].size(); ++gi)
        {
            const NonlinearEdgePoint& point = pre.edges[k][gi];
            const std::array<double, EDGE_VARIABLES> v = edgeVariables(re, def, point, k, gi, xs);
            edgePhi[k].push_back({SpaceVector{v[0], v[1], v[2]}, SpaceVector{v[3], v[4], v[5]}});
            HessianScalar<EDGE_VARIABLES> angle =
                geometric ? edgeAngle(v, point.tv, point.tangent, point.g.nu)
                          : edgeAngle(v, point.tv, point.previousTangent, point.previousNormal);
            if (!geometric)
                angle.value += point.previousAngle;
            // the linear terms: γ · μ and the hybrid unknown a_T
            double gmu = 0.0;
            Vector column(nx, 0.0);
            for (std::size_t i = 0; i < nc; ++i)
            {
                const std::array<double, 2>& hc = re.hcurlEdge[k][gi * nc + i];
                const double value = hc[0] * point.mc[0] + hc[1] * point.mc[1];
                column[3 * n1 + i] += value;
                gmu += value * x[3 * n1 + i];
            }
            double aT = 0.0;
            for (std::size_t l = 0; l < re.rotationCount(k); ++l)
            {
                const std::size_t index = shellRotationUnknown(re, k, l);
                const double value = -sk * shelldetail::edgeLegendre(l, point.canonicalParameter) /
                                     (point.length * point.weight * point.weight);
                column[index] += value;
                aT -= value * x[index];
            }
            for (std::size_t i = 0; i < n1; ++i)
            {
                const EdgeJet& j = def.edgeJets[k][gi * n1 + i];
                for (std::size_t c = 0; c < 3; ++c)
                    column[3 * i + c] += angle.gradient[c] * j[1] + angle.gradient[3 + c] * j[2];
            }
            const double bv = angle.value + gmu - aT;
            const double w = re.edgeWeights[gi] * point.length;
            for (std::size_t jj = 0; jj < ns; ++jj)
            {
                const double f = -w * point.normalMoments[jj];
                if (!geometric)
                    gv[jj] += f * bv;
                std::span<double> row = gd.rowSpan(jj);
                for (std::size_t i = 0; i < nx; ++i)
                    if (column[i] != 0.0)
                        row[i] += f * column[i];
            }
            edgeAngles[k].push_back(angle);
        }
    }

    // ---- membrane: the membrane functionals of the Green strain and their derivatives
    const GreenStrainFunctionals green = greenStrainFunctionals(re, def, pre, phi1, edgePhi, geometric, nx);

    // ---- the stresses
    const Vector fr0 = membraneTerm ? *membraneTerm : Vector(nr, 0.0);
    Vector c;
    Matrix jm;
    if (geometric)
    {
        c = matrices.reggeInterpolation * x;
        jm = matrices.reggeInterpolation;
    }
    else
    {
        c = matrices.reggeFunctionalInverse * green.values;
        jm = matrices.reggeFunctionalInverse * green.derivatives;
    }
    Vector gt = geometric ? matrices.pairing * x : add(gv, pre.cr * c);
    if (momentTerm)
        gt = subtract(std::move(gt), *momentTerm);
    const Vector m = matrices.complianceInverse * gt;
    const Vector nmem = subtract(pre.kr * c, fr0);
    const Vector wf =
        re.rawMembraneWeights(transposeTimes(matrices.reggeFunctionalInverse, add(nmem, transposeTimes(pre.cr, m))));

    // ---- the second variations weighted by the stresses
    Matrix kgeo(nx, nx);
    const Matrix ks1 = membraneSecondVariation(re, def, pre, wf);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t j = 0; j < n1; ++j)
            for (std::size_t cc = 0; cc < 3; ++cc)
                kgeo(3 * i + cc, 3 * j + cc) += ks1(i, j);
    addBendingSecondVariation(re, def, pre, volumeMeasures, m, kgeo);
    addKinkSecondVariation(re, def, pre, edgeAngles, m, kgeo);
    shelldetail::symmetrise(kgeo);

    NonlinearEvaluation out;
    if (geometric)
    {
        out.forces.assign(nx, 0.0);
        out.tangent = std::move(kgeo);
        return out;
    }
    const Matrix gtd = gd + pre.cr * jm;
    const Vector ksx = pre.ks * x;
    out.energy = 0.5 * dot(c, pre.kr * c) - dot(c, fr0) + 0.5 * dot(gt, m) + 0.5 * dot(x, ksx);
    out.forces = add(add(transposeTimes(jm, nmem), transposeTimes(gtd, m)), ksx);
    Matrix tangent = transposeTimes(jm, pre.kr * jm);
    tangent += transposeTimes(gtd, matrices.complianceInverse * gtd);
    tangent += pre.ks;
    tangent += kgeo;
    shelldetail::symmetrise(tangent);
    out.tangent = std::move(tangent);
    return out;
}

namespace
{

// The variation of the point variables (φ_,1, φ_,2, φ_,11, φ_,22, φ_,12) at surface point q along the unknowns `phi`.
std::array<double, POINT_VARIABLES> pointVariation(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                   std::size_t q, std::span<const double> phi)
{
    const std::size_t n1 = re.h1Count;
    std::array<double, POINT_VARIABLES> v{};
    for (std::size_t i = 0; i < n1; ++i)
    {
        const ReferenceJet& j = def.volumeJets[q * n1 + i];
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t d = 0; d < 5; ++d)
                v[3 * d + c] += phi[3 * i + c] * j[JET_OF_VARIABLE[d]];
    }
    return v;
}

// The variation of the edge variables (φ_,1, φ_,2) at point gi of edge k along the unknowns `phi`.
std::array<double, EDGE_VARIABLES> edgeVariation(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                 std::size_t k, std::size_t gi, std::span<const double> phi)
{
    const std::size_t n1 = re.h1Count;
    std::array<double, EDGE_VARIABLES> v{};
    for (std::size_t i = 0; i < n1; ++i)
    {
        const EdgeJet& j = def.edgeJets[k][gi * n1 + i];
        for (std::size_t c = 0; c < 3; ++c)
        {
            v[c] += phi[3 * i + c] * j[1];
            v[3 + c] += phi[3 * i + c] * j[2];
        }
    }
    return v;
}

// vᵀ·H·v.
template <std::size_t N>
double quadraticForm(const std::array<std::array<double, N>, N>& h, const std::array<double, N>& v)
{
    double sum = 0.0;
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = 0; b < N; ++b)
            sum += v[a] * h[a][b] * v[b];
    return sum;
}

// a_f = φᵀ·(Q_f ⊗ I₃)·φ for each membrane functional f, Q_f the matrix membraneSecondVariation weighs by w_f: the
// same sums, each product of two functions replaced by the product of the variations of φ along them.
Vector membraneSecondVariationProducts(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                       const NonlinearPreparation& pre, std::span<const double> phi)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nv = re.cornerCount;
    Vector products(nv * (re.shearDegree + 1) + re.reggeInteriorMoments.size() + 3, 0.0); // the raw functionals
    std::size_t f = 0;
    for (std::size_t k = 0; k < nv; ++k)
    {
        for (std::size_t l = 0; l <= re.shearDegree; ++l)
        {
            double sum = 0.0;
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
            {
                const std::array<double, 2>& tv = pre.edges[k][gi].tv;
                // Σ_i (∂_τ N_i) φ_i: the derivative of φ along the edge.
                SpaceVector tangential{};
                for (std::size_t i = 0; i < n1; ++i)
                {
                    const EdgeJet& hj = def.edgeJets[k][gi * n1 + i];
                    const double dt = hj[1] * tv[0] + hj[2] * tv[1];
                    for (std::size_t c = 0; c < 3; ++c)
                        tangential[c] += dt * phi[3 * i + c];
                }
                sum += re.edgeWeights[gi] * shelldetail::edgeLegendre(l, re.edgeParameters[gi]) *
                       dot3(tangential, tangential);
            }
            products[f++] = sum;
        }
    }
    const std::size_t nri = re.reggeInteriorMoments.size();
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        const std::array<double, POINT_VARIABLES> v = pointVariation(re, def, q, phi);
        const SpaceVector d1{v[0], v[1], v[2]};
        const SpaceVector d2{v[3], v[4], v[5]};
        // The factors of wc[0], wc[1], wc[2] in membraneSecondVariation.
        const std::array<double, 3> pc{dot3(d1, d1), dot3(d2, d2), dot3(d1, d2)};
        for (std::size_t m = 0; m < nri; ++m)
            products[f + m] += re.weights[q] * re.reggeTest[q * nri + m] * pc[re.reggeInteriorMoments[m].component];
        if (def.mapReferenceConstants)
        {
            for (std::size_t cc = 0; cc < 3; ++cc)
                products[f + nri + cc] += re.weights[q] * pc[cc];
        }
        else
        {
            const Matrix33 tm = covariantToVoigt(g.cov());
            const Matrix33 tt = strainTransformation(g.phi);
            for (std::size_t cc = 0; cc < 3; ++cc)
            {
                double s = 0.0;
                for (std::size_t comp = 0; comp < 3; ++comp)
                    s += (tt[cc][0] * tm[0][comp] + tt[cc][1] * tm[1][comp] + tt[cc][2] * tm[2][comp]) * pc[comp];
                products[f + nri + cc] += re.weights[q] * g.j * s;
            }
        }
    }
    return products;
}

// b_j = φᵀ·(∂K/∂m_j)·φ for each moment function j: the bending (addBendingSecondVariation) and kink
// (addKinkSecondVariation) second variations of a unit moment j, contracted with φ.
Vector momentSecondVariationProducts(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                     const NonlinearPreparation& pre, std::span<const double> phi)
{
    const std::size_t ns = re.momentCount;
    const Vector zero(shellUnknownCount(re), 0.0);
    Vector products(ns, 0.0);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const PointGeometry& g = pre.geometries[q];
        const std::array<HessianScalar<POINT_VARIABLES>, 3> kb =
            bendingMeasure(pointVariables(re, def, pre, q, zero), pre.referenceCurvature[q]);
        const std::array<double, POINT_VARIABLES> v = pointVariation(re, def, q, phi);
        const Voigt3 h{quadraticForm(kb[0].hessian, v), quadraticForm(kb[1].hessian, v),
                       quadraticForm(kb[2].hessian, v)};
        // Σ_r wcov_r h_r with wcov = tmᵀ·(sv·m): the weight of m_j is Σ_s sv(s, j) (tm·h)_s.
        const Voigt3 th = times33(covariantToVoigt(g.cov()), h);
        const Matrix sv =
            momentsLocal(re, [&](std::size_t jj) { return re.moments[q * ns + jj]; }, g, def.mapReferenceConstants);
        const double w = re.weights[q] * g.j;
        for (std::size_t j = 0; j < ns; ++j)
            products[j] += w * (sv(0, j) * th[0] + sv(1, j) * th[1] + sv(2, j) * th[2]);
    }
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        for (std::size_t gi = 0; gi < pre.edges[k].size(); ++gi)
        {
            const NonlinearEdgePoint& point = pre.edges[k][gi];
            const HessianScalar<EDGE_VARIABLES> angle =
                edgeAngle(edgeVariables(re, def, point, k, gi, zero), point.tv, point.tangent, point.g.nu);
            const double h = quadraticForm(angle.hessian, edgeVariation(re, def, k, gi, phi));
            const double w = -re.edgeWeights[gi] * point.length * h;
            for (std::size_t j = 0; j < ns; ++j)
                products[j] += w * point.normalMoments[j];
        }
    }
    return products;
}

} // namespace

Vector geometricStiffnessGradient(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                  const ShellElementMatrices& matrices, const NonlinearPreparation& pre,
                                  std::span<const double> phi)
{
    // In evaluateNonlinear with `geometric`: K(x) = Σ_f w_f (Q_f ⊗ I₃) + Σ_j m_j B_j, with c = R x, m = C⁻¹ P x and
    // w = F⁻ᵀ (K_r c + C_rᵀ m) (R the Regge interpolation, P the pairing, C⁻¹ the inverse compliance, F⁻¹ the inverse
    // of the Regge functionals; the thermal terms are constant in x). With a_f = φᵀ (Q_f ⊗ I₃) φ, b_j = φᵀ B_j φ and
    // α = F⁻¹ a:
    //   φᵀ K(x) φ = w · a + m · b,   ∂/∂x = Rᵀ K_rᵀ α + Pᵀ C⁻ᵀ (C_r α + b).
    const Vector alpha =
        matrices.reggeFunctionalInverse * re.membraneFunctionals(membraneSecondVariationProducts(re, def, pre, phi));
    const Vector b = momentSecondVariationProducts(re, def, pre, phi);
    const Vector onMoments = transposeTimes(matrices.complianceInverse, add(pre.cr * alpha, b));
    return add(transposeTimes(matrices.reggeInterpolation, transposeTimes(pre.kr, alpha)),
               transposeTimes(matrices.pairing, onMoments));
}

void acceptNonlinearState(const ShellReferenceElement& re, const ShellElementDefinition& def,
                          NonlinearPreparation& preparation, std::span<const double> x)
{
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        for (std::size_t gi = 0; gi < preparation.edges[k].size(); ++gi)
        {
            NonlinearEdgePoint& point = preparation.edges[k][gi];
            const std::array<double, EDGE_VARIABLES> v = edgeVariables(re, def, point, k, gi, x);
            const HessianScalar<EDGE_VARIABLES> angle =
                edgeAngle(v, point.tv, point.previousTangent, point.previousNormal);
            const SpaceVector a1{v[0], v[1], v[2]};
            const SpaceVector a2{v[3], v[4], v[5]};
            const SpaceVector xt = add3(scl3(a1, point.tv[0]), scl3(a2, point.tv[1]));
            point.previousAngle += angle.value;
            point.previousTangent = unit3(xt);
            point.previousNormal = unit3(cross3(a1, a2));
        }
    }
}

} // namespace oos::elements::mixedHybrid
