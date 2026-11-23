#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"

#include "ShellElementDetail.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellNonlinear.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements::mixedHybrid
{

namespace
{

// The interior coefficients of a field of `components` components: G⁻¹ ∫ N_interior (f − Σ_i N_i x_i) dA on the
// reference measure, x the coefficients with their interior part at zero, f(q) the field at surface point q.
Matrix projectInterior(const ShellReferenceElement& re, const InteriorProjection& projection,
                       const Matrix& coefficients, const std::function<Vector(std::size_t)>& field)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t ni = projection.interior.size();
    const std::size_t components = coefficients.columns();
    Matrix rhs(ni, components);
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const Vector target = field(q);
        Vector boundary(components, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
        {
            const double n = re.h1[q * n1 + i][0];
            for (std::size_t c = 0; c < components; ++c)
                boundary[c] += n * coefficients(i, c);
        }
        for (std::size_t a = 0; a < ni; ++a)
        {
            const double n = re.weights[q] * re.h1[q * n1 + projection.interior[a]][0];
            for (std::size_t c = 0; c < components; ++c)
                rhs(a, c) += n * (target[c] - boundary[c]);
        }
    }
    return projection.gramInverse * rhs;
}

} // namespace

ShellElementDefinition ShellElementData::definition() const
{
    ShellElementDefinition def;
    def.mapCoefficients = mapCoefficients;
    def.mapOrigin = mapOrigin;
    def.weight = &weight;
    def.volumeJets = volumeJets;
    def.edgeJets = edgeJets;
    def.edgeWeights = edgeWeights;
    def.section = &section;
    def.offset = offset;
    def.axes = axes;
    def.edgeSigns = edgeSigns;
    def.kirchhoff = kirchhoff;
    def.mapReferenceConstants = mapReferenceConstants;
    return def;
}

Vector ShellElementData::local(std::span<const double> u) const
{
    Vector out(dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
        out[i] = signs[i] * u[dofs[i]];
    return out;
}

Matrix rotationRowsOf(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi, double eta)
{
    const std::size_t nx = shellUnknownCount(re);
    const PointGeometry g = geometryAt(re, def, xi, eta);
    // e_I = (e_I · a^α) a_α: the derivative along e_I is ∂_I u = (e_I · a^α) ∂_α u
    const auto onContravariant = [&](const SpaceVector& d)
    { return std::array<double, 2>{dot3(d, g.ac[0]), dot3(d, g.ac[1])}; };
    const std::array<double, 2> c0 = onContravariant(g.e[0]);
    const std::array<double, 2> c1 = onContravariant(g.e[1]);
    Matrix out(3, nx);
    const std::vector<ReferenceJet> jets = h1At(re, def, xi, eta);
    for (std::size_t i = 0; i < jets.size(); ++i)
    {
        const ReferenceJet& j = jets[i];
        for (std::size_t c = 0; c < 3; ++c)
        {
            SurfaceFrame du{};
            du[0][c] = j[1];
            du[1][c] = j[2];
            SpaceVector beta{};
            for (std::size_t al = 0; al < 2; ++al)
                beta = add3(beta, scl3(g.ac[al], -dot3(g.nu, du[al])));
            const SpaceVector d1u = add3(scl3(du[0], c0[0]), scl3(du[1], c0[1]));
            const SpaceVector d2u = add3(scl3(du[0], c1[0]), scl3(du[1], c1[1]));
            const double w = 0.5 * (dot3(d1u, g.e[1]) - dot3(d2u, g.e[0]));
            const SpaceVector theta = add3(cross3(g.nu, beta), scl3(g.nu, w));
            for (std::size_t r = 0; r < 3; ++r)
                out(r, 3 * i + c) = theta[r];
        }
    }
    for (std::size_t i = 0; i < re.hcurlBasis.size(); ++i)
    {
        const Field& f = re.hcurlBasis[i];
        const SpaceVector beta = add3(scl3(g.ac[0], f[0].evaluate(xi, eta)), scl3(g.ac[1], f[1].evaluate(xi, eta)));
        const SpaceVector theta = cross3(g.nu, beta);
        for (std::size_t r = 0; r < 3; ++r)
            out(r, 3 * re.h1Count + i) = theta[r];
    }
    return out;
}

SpaceVector rotationAt(const ShellReferenceElement& re, const ShellElementDefinition& def, double xi, double eta,
                       std::span<const double> x)
{
    return shelldetail::times3(rotationRowsOf(re, def, xi, eta), x);
}

InteriorProjection InteriorProjection::build(const ShellReferenceElement& re)
{
    InteriorProjection projection;
    for (std::size_t i = 0; i < re.h1Count; ++i)
        if (re.h1Kinds[i].isInterior())
            projection.interior.push_back(i);
    const std::size_t ni = projection.interior.size();
    if (ni == 0)
        return projection;
    Matrix gram(ni, ni);
    for (std::size_t q = 0; q < re.points.size(); ++q)
        for (std::size_t a = 0; a < ni; ++a)
            for (std::size_t b = 0; b < ni; ++b)
                gram(a, b) += re.weights[q] * re.h1[q * re.h1Count + projection.interior[a]][0] *
                              re.h1[q * re.h1Count + projection.interior[b]][0];
    std::optional<Matrix> gramInverse = inverse(gram);
    if (!gramInverse)
        throw base::InternalError("singular Gram matrix of the interior displacement functions");
    projection.gramInverse = std::move(*gramInverse);
    return projection;
}

std::vector<SpaceVector> edgeMapMoments(std::size_t cornerCount, std::size_t degree, std::size_t edge, bool forward,
                                        const SurfaceAt& surface, const Polynomial2& weight)
{
    std::vector<SpaceVector> moments(degree - 1, SpaceVector{});
    for (const LineRulePoint& point : unitIntervalRule(MAP_EDGE_POINTS))
    {
        const double te = point.point;
        const double tl = forward ? te : 1.0 - te;
        const ReferencePoint x = referenceEdgePointAndDirection(cornerCount, edge, tl).first;
        const SpaceVector xv = surface(x[0], x[1]);
        const double wv = weight.evaluate(x[0], x[1]);
        for (std::size_t l = 0; l + 1 < degree; ++l)
        {
            const double f = point.weight * shelldetail::edgeLegendre(l, te);
            for (std::size_t c = 0; c < 3; ++c)
                moments[l][c] += f * xv[c] * wv;
        }
    }
    return moments;
}

std::vector<SpaceVector> mapCoefficients(const ShellReferenceElement& re, const InteriorProjection& projection,
                                         std::span<const SpaceVector> corners,
                                         const std::vector<std::vector<SpaceVector>>& canonicalMoments,
                                         std::span<const double> edgeSigns, const SurfaceAt& surface,
                                         const Polynomial2& weight)
{
    const std::size_t n1 = re.h1Count;
    Matrix xg(n1, 3);
    for (std::size_t i = 0; i < n1; ++i)
    {
        const FunctionKind& kind = re.h1Kinds[i];
        if (kind.place == FunctionKind::Place::Vertex)
        {
            xg.setRow(i, corners[kind.entity]);
        }
        else if (kind.place == FunctionKind::Place::Edge)
        {
            // the moment of odd degree changes sign with the direction of the edge
            const double s = (edgeSigns[kind.entity] < 0.0 && kind.moment % 2 == 1) ? -1.0 : 1.0;
            const SpaceVector& m = canonicalMoments[kind.entity][kind.moment];
            xg.setRow(i, std::array<double, 3>{m[0] * s, m[1] * s, m[2] * s});
        }
    }
    if (!projection.interior.empty())
    {
        const Matrix interior = projectInterior(re, projection, xg,
                                                [&](std::size_t q)
                                                {
                                                    const double xi = re.points[q][0];
                                                    const double eta = re.points[q][1];
                                                    const double wv = weight.evaluate(xi, eta);
                                                    const SpaceVector xv = scl3(surface(xi, eta), wv);
                                                    return Vector{xv[0], xv[1], xv[2]};
                                                });
        for (std::size_t a = 0; a < projection.interior.size(); ++a)
            xg.setRow(projection.interior[a], interior.row(a));
    }
    std::vector<SpaceVector> out(n1);
    for (std::size_t i = 0; i < n1; ++i)
        out[i] = {xg(i, 0), xg(i, 1), xg(i, 2)};
    return out;
}

std::array<Vector, 6> rigidMotionUnknowns(const ShellReferenceElement& re, const ShellElementDefinition& def)
{
    return rigidMotionUnknowns(re, def, InteriorProjection::build(re));
}

std::array<Vector, 6> rigidMotionUnknowns(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                          const InteriorProjection& projection)
{
    const std::size_t n1 = re.h1Count;
    const std::size_t nv = re.cornerCount;
    const std::size_t nx = shellUnknownCount(re);
    // a constant field: the coefficients of W on the displacement basis (corner values 1, the edge moments of W along
    // the edge in the element's own direction, the interior by projection)
    Matrix constant(n1, 1);
    for (std::size_t i = 0; i < n1; ++i)
    {
        const FunctionKind& kind = re.h1Kinds[i];
        if (kind.place == FunctionKind::Place::Vertex)
        {
            constant(i, 0) = 1.0;
        }
        else if (kind.place == FunctionKind::Place::Edge)
        {
            double sum = 0.0;
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
                sum += re.edgeWeights[gi] * def.edgeWeights[kind.entity][gi] *
                       shelldetail::edgeLegendre(kind.moment, re.edgeParameters[gi]);
            constant(i, 0) = sum;
        }
    }
    if (!projection.interior.empty())
    {
        const auto weightAt = [&](std::size_t q)
        { return Vector{def.weight->evaluate(re.points[q][0], re.points[q][1])}; };
        const Matrix interior = projectInterior(re, projection, constant, weightAt);
        for (std::size_t a = 0; a < projection.interior.size(); ++a)
            constant(projection.interior[a], 0) = interior(a, 0);
    }
    std::array<Vector, 6> out;
    for (Vector& v : out)
        v.assign(nx, 0.0);
    for (std::size_t i = 0; i < n1; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            out[c][3 * i + c] = constant(i, 0);
    // the rotations: ω × X on the displacement, the rotation about the edges ω · X_τ in the measure of the edge
    // unknowns (coefficient on P_l: (2l + 1) times the moment)
    for (std::size_t c = 0; c < 3; ++c)
    {
        SpaceVector omega{};
        omega[c] = 1.0;
        Vector& v = out[3 + c];
        for (std::size_t i = 0; i < n1; ++i)
        {
            const SpaceVector w = cross3(omega, add3(def.mapCoefficients[i], scl3(def.mapOrigin, constant(i, 0))));
            for (std::size_t d = 0; d < 3; ++d)
                v[3 * i + d] = w[d];
        }
        for (std::size_t k = 0; k < nv; ++k)
        {
            const double sk = def.edgeSigns[k];
            const std::size_t rotations = re.rotationCount(k);
            std::vector<double> moments(rotations, 0.0);
            for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
            {
                const PointGeometry g = edgeGeometry(def, n1, k, gi);
                const auto tv = referenceEdgePointAndDirection(nv, k, re.edgeParameters[gi]).second;
                const SpaceVector xt = scl3(add3(scl3(g.a[0], tv[0]), scl3(g.a[1], tv[1])), sk);
                const double te = sk > 0.0 ? re.edgeParameters[gi] : 1.0 - re.edgeParameters[gi];
                const double we = def.edgeWeights[k][gi];
                for (std::size_t l = 0; l < rotations; ++l)
                    moments[l] += re.edgeWeights[gi] * we * we * dot3(omega, xt) * shelldetail::edgeLegendre(l, te);
            }
            for (std::size_t l = 0; l < rotations; ++l)
                v[shellRotationUnknown(re, k, l)] = static_cast<double>(2 * l + 1) * moments[l];
        }
    }
    return out;
}

Vector CondensedInterior::solve(const Vector& v) const
{
    if (!factor)
        return {};
    Vector scaled(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        scaled[i] = v[i] * scale[i];
    Vector x = factor->solve(std::move(scaled));
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] *= scale[i];
    return x;
}

Matrix CondensedInterior::condensedStiffness(const Matrix& stiffness) const
{
    Matrix schur(outer.size(), outer.size());
    for (std::size_t a = 0; a < outer.size(); ++a)
        for (std::size_t b = 0; b < outer.size(); ++b)
            schur(a, b) = stiffness(outer[a], outer[b]);
    if (factor)
    {
        // Y = L⁻¹ S K_io: K_oi K_ii⁻¹ K_io = Yᵀ Y
        Matrix y(inner.size(), outer.size());
        for (std::size_t a = 0; a < inner.size(); ++a)
            for (std::size_t b = 0; b < outer.size(); ++b)
                y(a, b) = koi(b, a) * scale[a];
        solveLower(factor->lower(), y);
        schur -= transposeTimes(y, y);
    }
    shelldetail::symmetrise(schur);
    return schur;
}

Vector CondensedInterior::recoverInterior(const Vector& interiorLoad, const Vector& outerUnknowns) const
{
    Vector rhs = interiorLoad;
    const Vector kio = transposeTimes(koi, outerUnknowns);
    for (std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] -= kio[i];
    return solve(rhs);
}

std::optional<CondensedInterior> condenseInterior(const Matrix& stiffness, const std::vector<bool>& isInterior)
{
    CondensedInterior condensed;
    for (std::size_t i = 0; i < isInterior.size(); ++i)
        (isInterior[i] ? condensed.inner : condensed.outer).push_back(i);
    const std::size_t ni = condensed.inner.size();
    Matrix kii(ni, ni);
    for (std::size_t a = 0; a < ni; ++a)
        for (std::size_t b = 0; b < ni; ++b)
            kii(a, b) = stiffness(condensed.inner[a], condensed.inner[b]);
    condensed.scale.resize(ni);
    for (std::size_t a = 0; a < ni; ++a)
        condensed.scale[a] = 1.0 / std::sqrt(std::max(std::fabs(kii(a, a)), 1e-300));
    for (std::size_t a = 0; a < ni; ++a)
        for (std::size_t b = 0; b < ni; ++b)
            kii(a, b) *= condensed.scale[a] * condensed.scale[b];
    if (ni > 0)
    {
        condensed.factor = Cholesky::factor(kii);
        if (!condensed.factor)
            return std::nullopt;
    }
    condensed.koi = Matrix(condensed.outer.size(), ni);
    for (std::size_t a = 0; a < condensed.outer.size(); ++a)
        for (std::size_t b = 0; b < ni; ++b)
            condensed.koi(a, b) = stiffness(condensed.outer[a], condensed.inner[b]);
    return condensed;
}

Matrix elementGeometricStiffness(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                 const ShellElementMatrices& matrices, std::span<const double> x,
                                 const Vector* momentTerm, const Vector* membraneTerm,
                                 const PressureAt* followerPressureAt)
{
    const NonlinearPreparation preparation = prepareNonlinear(re, def);
    Matrix kg = evaluateNonlinear(re, def, matrices, preparation, x, momentTerm, membraneTerm, true).tangent;
    if (followerPressureAt && *followerPressureAt)
        kg -= followerPressure(re, def, *followerPressureAt);
    return kg;
}

} // namespace oos::elements::mixedHybrid
