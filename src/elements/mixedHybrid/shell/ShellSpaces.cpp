#include "oos/elements/mixedHybrid/shell/ShellSpaces.hpp"

#include "ShellSpacesDetail.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

namespace oos::elements::mixedHybrid
{

namespace
{

// A generator is kept in an orthonormal basis of the span when its eigenvalue of the Gram matrix is above this
// fraction of the largest: the generators of a space may be dependent.
constexpr double SPAN_EIGENVALUE_RATIO = 1e-11;
// An eigenvalue of AᵀA below this fraction of the largest belongs to the null space of the edge functionals.
constexpr double NULL_EIGENVALUE_RATIO = 1e-10;
// Singular values of the functionals on the span at most this value are taken as zero in their pseudo-inverse.
constexpr double PSEUDO_INVERSE_TOLERANCE = 1e-12;
// The residual of the projection of a reference constant on a tensor space, relative to the area, that proves the
// constant is in the space.
constexpr double CONSTANT_RESIDUAL_RATIO = 1e-16;

// Values of the boundary functionals of a field.
using BoundaryFunctionals = std::function<std::vector<double>(const Field&)>;

std::vector<double> evaluateField(const Field& f, const ReferencePoint& x)
{
    std::vector<double> out;
    out.reserve(f.size());
    for (const Polynomial2& p : f)
        out.push_back(p.evaluate(x[0], x[1]));
    return out;
}

// The values of every field at every point of a rule: [field][point][component].
std::vector<std::vector<std::vector<double>>> fieldValues(const std::vector<Field>& fields,
                                                          const std::vector<SurfaceRulePoint>& rule)
{
    std::vector<std::vector<std::vector<double>>> values;
    values.reserve(fields.size());
    for (const Field& f : fields)
    {
        std::vector<std::vector<double>> atPoints;
        atPoints.reserve(rule.size());
        for (const SurfaceRulePoint& point : rule)
            atPoints.push_back(evaluateField(f, point.point));
        values.push_back(std::move(atPoints));
    }
    return values;
}

// The L2 Gram matrix of fields on a rule: ∫ Σ_c f_i,c f_j,c.
Matrix gramOnRule(const std::vector<Field>& fields, const std::vector<SurfaceRulePoint>& rule)
{
    const std::size_t n = fields.size();
    const auto values = fieldValues(fields, rule);
    Matrix gram(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (std::size_t q = 0; q < rule.size(); ++q)
            {
                double product = 0.0;
                for (std::size_t c = 0; c < values[i][q].size(); ++c)
                    product += values[i][q][c] * values[j][q][c];
                sum += rule[q].weight * product;
            }
            gram(i, j) = sum;
        }
    }
    return gram;
}

double largestValue(const Vector& values)
{
    double top = 0.0;
    for (double v : values)
        top = std::max(top, v);
    return top;
}

// The L2 Gram matrix of fields on the reference square: the domain does not matter for the rank, and the rule of
// BUILD_POINTS points is exact for the generators.
// Reference: tdnns/space.rs::gram_matrix
Matrix gramMatrix(const std::vector<Field>& generators)
{
    return gramOnRule(generators, referenceSurfaceRule(4, shellspaces::BUILD_POINTS));
}

// The basis dual to `boundaryCount` boundary functionals, completed by a basis of their common null space (the
// interior functions). The generators may be dependent: the work is done in an orthonormal basis of their span,
//   Q = V_kept Λ_kept^(−1/2)  (the eigenpairs of the Gram matrix `gram` above SPAN_EIGENVALUE_RATIO of the largest),
// where the functionals are the matrix A Q. The boundary functions are the columns of Q (A Q)⁺ (orthogonal to the
// null space in the inner product of `gram`), the interior functions Q times the eigenvectors of (A Q)ᵀ(A Q) of zero
// eigenvalue.
// Reference: tdnns/space.rs::dual_basis
std::vector<Field> dualBasis(const std::vector<Field>& generators, std::size_t boundaryCount,
                             const BoundaryFunctionals& functionals, const Matrix& gram)
{
    const std::size_t n = generators.size();
    Matrix a(boundaryCount, n);
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::vector<double> v = functionals(generators[j]);
        for (std::size_t i = 0; i < boundaryCount; ++i)
            a(i, j) = v[i];
    }
    const SymmetricEigen eigen = jacobiEigen(gram);
    const double top = largestValue(eigen.values);
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < n; ++i)
        if (eigen.values[i] > SPAN_EIGENVALUE_RATIO * top)
            keep.push_back(i);
    const std::size_t dimension = keep.size();
    Matrix q(n, dimension);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < dimension; ++j)
            q(i, j) = eigen.vectors(i, keep[j]) / std::sqrt(eigen.values[keep[j]]);
    const Matrix aq = a * q;
    // the null space of A Q: eigenvectors of (A Q)ᵀ(A Q) of zero eigenvalue (rank boundaryCount by construction)
    const SymmetricEigen nullEigen = jacobiEigen(transposeTimes(aq, aq));
    const double eigenMax = largestValue(nullEigen.values);
    std::vector<std::size_t> null;
    for (std::size_t i = 0; i < dimension; ++i)
        if (nullEigen.values[i] < NULL_EIGENVALUE_RATIO * eigenMax)
            null.push_back(i);
    if (null.size() + boundaryCount != dimension)
        throw base::InternalError("edge degrees of freedom are not independent on the space");
    const Matrix pinv = pseudoInverse(aq, PSEUDO_INVERSE_TOLERANCE); // dimension × boundaryCount
    std::vector<Field> basis;
    basis.reserve(dimension);
    for (std::size_t i = 0; i < boundaryCount; ++i)
        basis.push_back(shellspaces::combine(generators, q * pinv.column(i)));
    for (std::size_t m : null)
        basis.push_back(shellspaces::combine(generators, q * nullEigen.vectors.column(m)));
    return basis;
}

// The moment ∫_0^1 f(τ) P_l(2τ − 1) dτ of a function of the edge parameter, by the edge rule.
double edgeMoment(const std::vector<LineRulePoint>& rule, std::size_t l, const std::function<double(double)>& f)
{
    double sum = 0.0;
    for (const LineRulePoint& point : rule)
        sum += point.weight * f(point.point) * legendreValue(l, 2.0 * point.point - 1.0);
    return sum;
}

// ---- generators of the quadrilateral moment space (Airy-type fields; every term has two derivatives, so the span
// does not depend on the scale of the square)

// Reference: tdnns/space.rs::sigma_u_gradv
Field sigmaUGradV(const Polynomial2& u, const Polynomial2& v)
{
    const Polynomial2 ux = u.dXi();
    const Polynomial2 uy = u.dEta();
    const Polynomial2 vx = v.dXi();
    const Polynomial2 vy = v.dEta();
    const Polynomial2 xx = u * vy.dEta() + uy * vy;
    const Polynomial2 yy = u * vx.dXi() + ux * vx;
    const Polynomial2 xy = (u * vx.dEta()).scaled(-1.0) - (ux * vy + uy * vx).scaled(0.5);
    return {xx, yy, xy};
}

// Reference: tdnns/space.rs::sigma_grad
Field sigmaGrad(const Polynomial2& w)
{
    return {w.dEta().dEta(), w.dXi().dXi(), w.dXi().dEta().scaled(-1.0)};
}

// Reference: tdnns/space.rs::v_sigma_grad_u
Field vSigmaGradU(const Polynomial2& u, const Polynomial2& v)
{
    return {u.dEta().dEta() * v, u.dXi().dXi() * v, (u.dXi().dEta() * v).scaled(-1.0)};
}

// The bubble (1 − z²) z^i of ξ (axis 0) or η (axis 1).
// Reference: tdnns/space.rs::bubble
Polynomial2 bubble(std::size_t axis, std::size_t i)
{
    const Polynomial2 z = axis == 0 ? Polynomial2::monomial(1, 0) : Polynomial2::monomial(0, 1);
    Polynomial2 zi = Polynomial2::constant(1.0);
    for (std::size_t k = 0; k < i; ++k)
        zi = zi * z;
    return (Polynomial2::constant(1.0) - z * z) * zi;
}

Field unitComponent(std::size_t componentCount, std::size_t component, const Polynomial2& p)
{
    Field f(componentCount);
    f[component] = p;
    return f;
}

} // namespace

namespace shellspaces
{

// Reference: tdnns/space.rs::combine
Field combine(const std::vector<Field>& generators, const Vector& coefficients)
{
    const std::size_t componentCount = generators.front().size();
    Field f(componentCount);
    for (std::size_t j = 0; j < generators.size(); ++j)
    {
        const double cj = coefficients[j];
        if (cj == 0.0)
            continue;
        for (std::size_t k = 0; k < componentCount; ++k)
            f[k] = f[k] + generators[j][k].scaled(cj);
    }
    return f;
}

// Reference: tdnns/space.rs::orthonormal
std::vector<Field> orthonormal(const std::vector<Field>& generators, std::size_t cornerCount)
{
    const std::size_t n = generators.size();
    const SymmetricEigen eigen = jacobiEigen(gramOnRule(generators, referenceSurfaceRule(cornerCount, BUILD_POINTS)));
    const double top = largestValue(eigen.values);
    std::vector<Field> out;
    for (std::size_t k = 0; k < n; ++k)
    {
        const double lambda = eigen.values[k];
        if (!(lambda > SPAN_EIGENVALUE_RATIO * top))
            continue;
        Vector c(n);
        for (std::size_t i = 0; i < n; ++i)
            c[i] = eigen.vectors(i, k) / std::sqrt(lambda);
        out.push_back(combine(generators, c));
    }
    return out;
}

} // namespace shellspaces

SpaceBasis h1Space(std::size_t cornerCount, std::size_t degree)
{
    const std::size_t nv = cornerCount;
    const std::size_t k = degree;
    std::vector<Field> generators;
    for (std::size_t i = 0; i <= k; ++i)
        for (std::size_t j = 0; j <= k; ++j)
            if (nv == 4 || i + j <= k)
                generators.push_back({Polynomial2::monomial(i, j)});
    const std::vector<LineRulePoint> edgeRule = unitIntervalRule(shellspaces::BUILD_POINTS);
    const std::size_t perEdge = k - 1;
    const std::size_t boundaryCount = nv + nv * perEdge;
    const BoundaryFunctionals functionals = [&](const Field& f)
    {
        std::vector<double> out;
        out.reserve(boundaryCount);
        for (std::size_t v = 0; v < nv; ++v)
        {
            const ReferencePoint corner = referenceCorner(nv, v);
            out.push_back(f[0].evaluate(corner[0], corner[1]));
        }
        for (std::size_t e = 0; e < nv; ++e)
        {
            for (std::size_t l = 0; l < perEdge; ++l)
            {
                out.push_back(edgeMoment(edgeRule, l,
                                         [&](double t)
                                         {
                                             const ReferencePoint x = referenceEdgePointAndDirection(nv, e, t).first;
                                             return f[0].evaluate(x[0], x[1]);
                                         }));
            }
        }
        return out;
    };
    // The functions of the corners and of the edges are those orthogonal to the interior functions on the element
    // itself: its own reference shape, whose L2 product the maps that turn the shape onto itself keep. They are then
    // the same functions whatever the corner the element starts from, and so is the share of a distributed load
    // between the grid points, the edges and the interior. (The reference takes the product of the square for the
    // triangle too: the functions of its corners depend on the first one. The space, and the solution, are the same.)
    SpaceBasis basis;
    basis.functions = dualBasis(generators, boundaryCount, functionals,
                                gramOnRule(generators, referenceSurfaceRule(nv, shellspaces::BUILD_POINTS)));
    for (std::size_t v = 0; v < nv; ++v)
        basis.kinds.push_back(FunctionKind::vertex(v));
    for (std::size_t e = 0; e < nv; ++e)
        for (std::size_t l = 0; l < perEdge; ++l)
            basis.kinds.push_back(FunctionKind::edge(e, l));
    basis.kinds.resize(basis.functions.size(), FunctionKind::interior());
    return basis;
}

SpaceBasis hcurlSpace(std::size_t cornerCount, std::size_t degree)
{
    const std::size_t nv = cornerCount;
    const std::size_t p = degree;
    std::vector<Field> generators;
    for (std::size_t c = 0; c < 2; ++c)
    {
        for (std::size_t i = 0; i <= p + 1; ++i)
        {
            for (std::size_t j = 0; j <= p + 1; ++j)
            {
                const bool inSpace = nv == 4 ? ((c == 0 && i <= p) || (c == 1 && j <= p)) : i + j <= p;
                if (inSpace)
                    generators.push_back(unitComponent(2, c, Polynomial2::monomial(i, j)));
            }
        }
    }
    const std::vector<LineRulePoint> edgeRule = unitIntervalRule(shellspaces::BUILD_POINTS);
    const std::size_t boundaryCount = nv * (p + 1);
    const BoundaryFunctionals functionals = [&](const Field& f)
    {
        std::vector<double> out;
        out.reserve(boundaryCount);
        for (std::size_t e = 0; e < nv; ++e)
        {
            for (std::size_t l = 0; l <= p; ++l)
            {
                out.push_back(edgeMoment(edgeRule, l,
                                         [&](double t)
                                         {
                                             const auto [x, tv] = referenceEdgePointAndDirection(nv, e, t);
                                             return f[0].evaluate(x[0], x[1]) * tv[0] +
                                                    f[1].evaluate(x[0], x[1]) * tv[1];
                                         }));
            }
        }
        return out;
    };
    SpaceBasis basis;
    basis.functions = dualBasis(generators, boundaryCount, functionals, gramMatrix(generators));
    for (std::size_t e = 0; e < nv; ++e)
        for (std::size_t l = 0; l <= p; ++l)
            basis.kinds.push_back(FunctionKind::edge(e, l));
    basis.kinds.resize(basis.functions.size(), FunctionKind::interior());
    return basis;
}

std::vector<Field> hhjSpace(std::size_t cornerCount, std::size_t degree)
{
    const std::size_t p = degree;
    std::vector<Field> generators;
    if (cornerCount == 3)
    {
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t i = 0; i <= p; ++i)
                for (std::size_t j = 0; j <= p - i; ++j)
                    generators.push_back(unitComponent(3, c, Polynomial2::monomial(i, j)));
        return shellspaces::orthonormal(generators, cornerCount);
    }
    // the edges: the linear function of each side vanishes on the side opposite to it, its bubbles run along it
    struct Side
    {
        Polynomial2 linear;
        std::size_t axis;
    };
    const Side sides[] = {{Polynomial2::linearInEta(0.5, -0.5), 0},
                          {Polynomial2::linearInXi(0.5, 0.5), 1},
                          {Polynomial2::linearInEta(0.5, 0.5), 0},
                          {Polynomial2::linearInXi(0.5, -0.5), 1}};
    for (const Side& side : sides)
        for (std::size_t l = 0; l <= p; ++l)
            generators.push_back(sigmaUGradV(side.linear, bubble(side.axis, l)));
    if (p >= 1)
    {
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j < p; ++j)
                generators.push_back(sigmaGrad(bubble(0, i) * bubble(1, j)));
        for (std::size_t i = 0; i <= p + 1; ++i)
        {
            for (std::size_t j = 0; j < p; ++j)
            {
                generators.push_back(vSigmaGradU(bubble(0, i), bubble(1, j)));
                generators.push_back(vSigmaGradU(bubble(1, i), bubble(0, j)));
            }
        }
    }
    generators.push_back(sigmaUGradV(Polynomial2::linearInXi(0.5, -0.5), Polynomial2::linearInEta(0.5, -0.5)));
    for (std::size_t i = 0; i < p; ++i)
    {
        generators.push_back(sigmaUGradV(bubble(0, i), Polynomial2::linearInEta(0.5, -0.5)));
        generators.push_back(sigmaUGradV(bubble(1, i), Polynomial2::linearInXi(0.5, -0.5)));
    }
    return shellspaces::orthonormal(generators, cornerCount);
}

std::vector<Field> reggeSpace(std::size_t cornerCount, std::size_t degree)
{
    const std::size_t p = degree;
    std::vector<Field> generators;
    for (std::size_t c = 0; c < 3; ++c)
    {
        for (std::size_t i = 0; i <= p + 1; ++i)
        {
            for (std::size_t j = 0; j <= p + 1; ++j)
            {
                bool inSpace = i + j <= p;
                if (cornerCount == 4)
                    inSpace = c == 0 ? i <= p : (c == 1 ? j <= p : (i <= p && j <= p));
                if (inSpace)
                    generators.push_back(unitComponent(3, c, Polynomial2::monomial(i, j)));
            }
        }
    }
    return shellspaces::orthonormal(generators, cornerCount);
}

std::vector<Field> splitConstants(const std::vector<Field>& basis, std::size_t cornerCount)
{
    const std::vector<SurfaceRulePoint> rule = referenceSurfaceRule(cornerCount, shellspaces::BUILD_POINTS);
    std::vector<Field> constants;
    for (std::size_t c = 0; c < 3; ++c)
        constants.push_back(unitComponent(3, c, Polynomial2::constant(1.0)));
    const auto innerProduct = [&](const Field& a, const Field& b)
    {
        double sum = 0.0;
        for (const SurfaceRulePoint& point : rule)
        {
            const double xi = point.point[0];
            const double eta = point.point[1];
            double product = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
                product += a[c].evaluate(xi, eta) * b[c].evaluate(xi, eta);
            sum += point.weight * product;
        }
        return sum;
    };
    double area = 0.0;
    for (const SurfaceRulePoint& point : rule)
        area += point.weight;
    // the constants are in the space: their projection on the (orthonormal) basis reproduces them
    for (const Field& constant : constants)
    {
        Field projection(3);
        for (const Field& b : basis)
        {
            const double coefficient = innerProduct(b, constant);
            for (std::size_t k = 0; k < 3; ++k)
                projection[k] = projection[k] + b[k].scaled(coefficient);
        }
        Field residual(3);
        for (std::size_t k = 0; k < 3; ++k)
            residual[k] = projection[k] - constant[k];
        const double r2 = innerProduct(residual, residual);
        if (!(r2 < CONSTANT_RESIDUAL_RATIO * area))
            throw base::InternalError("the reference constants are not in the space (" + std::to_string(r2) + ")");
    }
    // the complement: the constant part of every basis function removed, then orthonormalised
    std::vector<Field> reduced;
    reduced.reserve(basis.size());
    for (const Field& b : basis)
    {
        Field f = b;
        for (const Field& constant : constants)
        {
            const double a = innerProduct(b, constant) / area;
            for (std::size_t k = 0; k < 3; ++k)
                f[k] = f[k] - constant[k].scaled(a);
        }
        reduced.push_back(std::move(f));
    }
    std::vector<Field> out = shellspaces::orthonormal(reduced, cornerCount);
    if (out.size() + 3 != basis.size())
        throw base::InternalError("the complement of the constants has the wrong dimension");
    out.insert(out.end(), constants.begin(), constants.end());
    return out;
}

std::vector<ReggeMoment> reggeInterior(std::size_t cornerCount, std::size_t degree)
{
    const std::size_t p = degree;
    std::vector<ReggeMoment> out;
    if (p == 0)
        return out;
    if (cornerCount == 3)
    {
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < p; ++i)
                for (std::size_t j = 0; j < p - i; ++j)
                    if (i + j > 0)
                        out.push_back({c, i, j});
        return out;
    }
    for (std::size_t i = 0; i <= p; ++i)
        for (std::size_t j = 0; j <= p; ++j)
            out.push_back({2, i, j});
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j <= p; ++j)
            out.push_back({1, i, j});
    for (std::size_t i = 0; i <= p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            out.push_back({0, i, j});
    std::erase_if(out, [](const ReggeMoment& m) { return m.i + m.j == 0; });
    return out;
}

} // namespace oos::elements::mixedHybrid
