#include "oos/elements/shells/MindlinShell.hpp"

#include "oos/elements/core/ShapeFunctions.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace oos::elements
{

namespace
{

// The element frame's x axis and normal n, in the element frame.
constexpr base::Vec3 ELEMENT_X{1.0, 0.0, 0.0};
constexpr base::Vec3 ELEMENT_NORMAL{0.0, 0.0, 1.0};

// The inverse of a small matrix by Gauss-Jordan elimination with partial pivoting. Precondition: not singular (the
// matrices of the tying points, constants of the element).
template <std::size_t M>
[[nodiscard]] base::MatN<M> inverseOf(base::MatN<M> a)
{
    base::MatN<M> inverse = base::MatN<M>::identity();
    for (std::size_t column = 0; column < M; ++column)
    {
        std::size_t pivot = column;
        for (std::size_t r = column + 1; r < M; ++r)
            if (std::abs(a(r, column)) > std::abs(a(pivot, column)))
                pivot = r;
        for (std::size_t c = 0; c < M; ++c)
        {
            std::swap(a(column, c), a(pivot, c));
            std::swap(inverse(column, c), inverse(pivot, c));
        }
        const double scale = 1.0 / a(column, column);
        for (std::size_t c = 0; c < M; ++c)
        {
            a(column, c) *= scale;
            inverse(column, c) *= scale;
        }
        for (std::size_t r = 0; r < M; ++r)
        {
            if (r == column)
                continue;
            const double factor = a(r, column);
            for (std::size_t c = 0; c < M; ++c)
            {
                a(r, c) -= factor * a(column, c);
                inverse(r, c) -= factor * inverse(column, c);
            }
        }
    }
    return inverse;
}

// ---- The tied strains of the six-node triangle (the class comment of MindlinShell), in the natural coordinates
// (ξ, η) of the element: side 1 is η = 0, side 2 is ξ = 0, side 3 is ξ + η = 1.

// The two Gauss points of a side, as fractions of its length.
const double SIDE_POINT_1 = 0.5 - 0.5 / std::sqrt(3.0);
const double SIDE_POINT_2 = 0.5 + 0.5 / std::sqrt(3.0);
// The inner tying point of the membrane strain along a side lies on the parallel to that side at this fraction of
// the way to the opposite corner, in the middle of the segment the element cuts on it.
const double INNER_LINE = 1.0 / std::sqrt(3.0);
const double INNER_MIDDLE = 0.5 * (1.0 - INNER_LINE);

// A tying point of the transverse shear: the covariant component cξ γ̃_ξ + cη γ̃_η at (ξ, η). The tangential shear at
// the two Gauss points of each side, then both shears at the centroid.
struct ShearTie
{
    double xi;
    double eta;
    double alongXi;
    double alongEta;
};
[[nodiscard]] std::array<ShearTie, 8> shearTies()
{
    constexpr double THIRD = 1.0 / 3.0;
    return {ShearTie{SIDE_POINT_1, 0.0, 1.0, 0.0},
            ShearTie{SIDE_POINT_2, 0.0, 1.0, 0.0},
            ShearTie{0.0, SIDE_POINT_1, 0.0, 1.0},
            ShearTie{0.0, SIDE_POINT_2, 0.0, 1.0},
            ShearTie{SIDE_POINT_2, 1.0 - SIDE_POINT_2, -1.0, 1.0},
            ShearTie{SIDE_POINT_1, 1.0 - SIDE_POINT_1, -1.0, 1.0},
            ShearTie{THIRD, THIRD, 1.0, 0.0},
            ShearTie{THIRD, THIRD, 0.0, 1.0}};
}

// The eight fields of the tied transverse shear at (ξ, η), each as (γ̃_ξ, γ̃_η): the linear fields and the two fields
// (η, −ξ) ξ and (η, −ξ) η, whose tangential component along each side is linear.
[[nodiscard]] std::array<std::array<double, 2>, 8> shearFieldsAt(double xi, double eta)
{
    return {{{1.0, 0.0}, {xi, 0.0}, {eta, 0.0}, {0.0, 1.0}, {0.0, xi}, {0.0, eta},
             {eta * xi, -xi * xi}, {eta * eta, -xi * eta}}};
}

// The weights of the eight tied values in the tied shears at (ξ, η): γ̃_α = Σ_i weights(α, i) · value_i. The tied
// field is the combination of the eight fields that takes the eight tied values: with A(i, j) the tied component of
// field j at tying point i, its coefficients are A⁻¹ · values.
[[nodiscard]] base::MatN<2, 8> shearTieWeightsAt(double xi, double eta)
{
    static const base::MatN<8> COEFFICIENTS = []
    {
        base::MatN<8> a;
        const std::array<ShearTie, 8> ties = shearTies();
        for (std::size_t i = 0; i < 8; ++i)
        {
            const std::array<std::array<double, 2>, 8> fields = shearFieldsAt(ties[i].xi, ties[i].eta);
            for (std::size_t j = 0; j < 8; ++j)
                a(i, j) = ties[i].alongXi * fields[j][0] + ties[i].alongEta * fields[j][1];
        }
        return inverseOf(a);
    }();
    const std::array<std::array<double, 2>, 8> fields = shearFieldsAt(xi, eta);
    base::MatN<2, 8> weights;
    for (std::size_t alpha = 0; alpha < 2; ++alpha)
        for (std::size_t i = 0; i < 8; ++i)
            for (std::size_t j = 0; j < 8; ++j)
                weights(alpha, i) += fields[j][alpha] * COEFFICIENTS(j, i);
    return weights;
}

// The three tying points of each tied membrane strain: ẽ_ξξ (along side 1), ẽ_ηη (along side 2) and ẽ_qq (along
// side 3: ẽ_ξξ + ẽ_ηη − 2 ẽ_ξη, the strain along g_η − g_ξ), each at the two Gauss points of its side and at its
// inner point.
using TiePoint = std::array<double, 2>;
[[nodiscard]] std::array<std::array<TiePoint, 3>, 3> membraneTies()
{
    return {{{{{SIDE_POINT_1, 0.0}, {SIDE_POINT_2, 0.0}, {INNER_MIDDLE, INNER_LINE}}},
             {{{0.0, SIDE_POINT_1}, {0.0, SIDE_POINT_2}, {INNER_LINE, INNER_MIDDLE}}},
             {{{SIDE_POINT_2, 1.0 - SIDE_POINT_2}, {SIDE_POINT_1, 1.0 - SIDE_POINT_1}, {INNER_MIDDLE, INNER_MIDDLE}}}}};
}

// The weights of the three tied values of strain `which` (0: ξξ, 1: ηη, 2: qq) in its tied value at (ξ, η): the
// linear field a + b ξ + c η through its three tying points.
[[nodiscard]] std::array<double, 3> membraneTieWeightsAt(std::size_t which, double xi, double eta)
{
    static const std::array<base::MatN<3>, 3> COEFFICIENTS = []
    {
        std::array<base::MatN<3>, 3> inverses;
        const std::array<std::array<TiePoint, 3>, 3> ties = membraneTies();
        for (std::size_t strain = 0; strain < 3; ++strain)
        {
            base::MatN<3> a;
            for (std::size_t i = 0; i < 3; ++i)
            {
                a(i, 0) = 1.0;
                a(i, 1) = ties[strain][i][0];
                a(i, 2) = ties[strain][i][1];
            }
            inverses[strain] = inverseOf(a);
        }
        return inverses;
    }();
    const base::MatN<3>& c = COEFFICIENTS[which];
    std::array<double, 3> weights{};
    for (std::size_t i = 0; i < 3; ++i)
        weights[i] = c(0, i) + xi * c(1, i) + eta * c(2, i);
    return weights;
}

// The symmetric correction of the bending strain by the membrane strain on a curved surface (the class comment of
// MindlinShell): κ += ½ (Bᵀε + εB), B_ij = e_i·∂d/∂s_j the curvature of the director field, ε the membrane strain
// tensor (ε12 = γxy/2). In the order (κx, κy, 2κxy) of the rows:
//   κx += B11 εx + ½ B21 γxy,   κy += B22 εy + ½ B12 γxy,   2κxy += B12 εx + B21 εy + ½ (B11 + B22) γxy.
// A uniform dilatation u0 = e X0 (k = e·sym B) then bends nothing. Zero on a flat element (B = 0).
template <std::size_t N, typename Point>
void addCurvatureOfStretching(ShellStrainRows<N>& rows, const Point& p)
{
    const base::Vec3& e1 = p.frame[0];
    const base::Vec3& e2 = p.frame[1];
    const double b11 = base::dot(e1, p.directorDX);
    const double b21 = base::dot(e2, p.directorDX);
    const double b12 = base::dot(e1, p.directorDY);
    const double b22 = base::dot(e2, p.directorDY);
    if (b11 == 0.0 && b21 == 0.0 && b12 == 0.0 && b22 == 0.0)
        return;
    for (std::size_t c = 0; c < 6 * N; ++c)
    {
        const double ex = rows.membrane(0, c);
        const double ey = rows.membrane(1, c);
        const double gxy = rows.membrane(2, c);
        rows.curvature(0, c) += b11 * ex + 0.5 * b21 * gxy;
        rows.curvature(1, c) += b22 * ey + 0.5 * b12 * gxy;
        rows.curvature(2, c) += b12 * ex + b21 * ey + 0.5 * (b11 + b22) * gxy;
    }
}

// The rows of the point `p` for the directors ν_a. For node a, with r_i = ν_a × e_i (so that e_i·(θ × ν_a) = r_i·θ),
// on its translations u_a and rotations θ_a:
//   εx     = ∂N/∂s1 e1·u                       εy = ∂N/∂s2 e2·u      γxy = (∂N/∂s2 e1 + ∂N/∂s1 e2)·u
//   −κx    = ∂N/∂s1 (∂d/∂s1·u + r1·θ)          −κy = ∂N/∂s2 (∂d/∂s2·u + r2·θ)
//   −2κxy  = ∂N/∂s2 (∂d/∂s1·u + r1·θ) + ∂N/∂s1 (∂d/∂s2·u + r2·θ)
// (then addCurvatureOfStretching)
//   γx     = ∂N/∂s1 d·u + N r1·θ               γy = ∂N/∂s2 d·u + N r2·θ
// On a flat element (d = n, ∂d/∂s = 0, e1 = x, e2 = y) these are the Mindlin plate's rows, bit for bit.
template <std::size_t N, typename Point>
[[nodiscard]] base::MatN<3, 6 * N> membraneRowsOf(const Point& p)
{
    base::MatN<3, 6 * N> rows;
    const base::Vec3& e1 = p.frame[0];
    const base::Vec3& e2 = p.frame[1];
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t k = 0; k < 3; ++k)
        {
            rows(0, 6 * a + k) = p.dX[a] * e1[k];
            rows(1, 6 * a + k) = p.dY[a] * e2[k];
            rows(2, 6 * a + k) = p.dY[a] * e1[k] + p.dX[a] * e2[k];
        }
    return rows;
}

template <std::size_t N, typename Point>
[[nodiscard]] base::MatN<2, 6 * N> shearRowsOf(const Point& p, const std::array<base::Vec3, N>& director)
{
    base::MatN<2, 6 * N> rows;
    for (std::size_t a = 0; a < N; ++a)
    {
        const std::size_t u = 6 * a;
        const std::size_t theta = u + 3;
        const double n = p.shape.n[a];
        const base::Vec3 r1 = base::cross(director[a], p.frame[0]);
        const base::Vec3 r2 = base::cross(director[a], p.frame[1]);
        for (std::size_t k = 0; k < 3; ++k)
        {
            rows(0, u + k) = p.dX[a] * p.director[k];
            rows(0, theta + k) = n * r1[k];
            rows(1, u + k) = p.dY[a] * p.director[k];
            rows(1, theta + k) = n * r2[k];
        }
    }
    return rows;
}

template <std::size_t N, typename Point>
[[nodiscard]] ShellStrainRows<N> shellRows(const Point& p, const std::array<base::Vec3, N>& director)
{
    ShellStrainRows<N> rows;
    rows.membrane = membraneRowsOf<N>(p);
    rows.shear = shearRowsOf<N>(p, director);
    const base::Vec3& e1 = p.frame[0];
    const base::Vec3& e2 = p.frame[1];
    for (std::size_t a = 0; a < N; ++a)
    {
        const std::size_t u = 6 * a;
        const std::size_t theta = u + 3;
        const double dX = p.dX[a];
        const double dY = p.dY[a];
        const base::Vec3 r1 = base::cross(director[a], e1);
        const base::Vec3 r2 = base::cross(director[a], e2);
        for (std::size_t k = 0; k < 3; ++k)
        {
            rows.curvature(0, u + k) = -(dX * p.directorDX[k]);
            rows.curvature(0, theta + k) = -(dX * r1[k]);
            rows.curvature(1, u + k) = -(dY * p.directorDY[k]);
            rows.curvature(1, theta + k) = -(dY * r2[k]);
            rows.curvature(2, u + k) = -(dY * p.directorDX[k] + dX * p.directorDY[k]);
            rows.curvature(2, theta + k) = -(dY * r1[k] + dX * r2[k]);
        }
    }
    addCurvatureOfStretching<N>(rows, p);
    return rows;
}

// The strains at a point of rows `rows` of the homothety u = ε x of the element (the positions on the translations,
// no rotation), less (ε, ε, 0) for the membrane: zero on a flat element whose directors are its normal. Where the
// directors turn across the element (a curved element, directors shared with the neighbours) the homothety, which
// moves each fibre without stretching it, has a curvature and a transverse shear. A free thermal dilatation ε is
// that homothety, without stress (a uniform dilatation bends nothing: J. L. Sanders, An improved first-approximation
// theory for thin shells, NASA TR R-24, 1959).
struct DilatationStrains
{
    std::array<double, 3> membrane{};
    std::array<double, 3> curvature{};
    std::array<double, 2> shear{};
};

template <std::size_t N>
[[nodiscard]] DilatationStrains dilatationStrainsOf(const base::MatN<3, 6 * N>& membrane,
                                                    const base::MatN<3, 6 * N>& curvature,
                                                    const base::MatN<2, 6 * N>& shear,
                                                    const std::array<base::Vec3, N>& position, double dilatation)
{
    DilatationStrains strains;
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double u = dilatation * position[a][k];
            for (std::size_t r = 0; r < 3; ++r)
            {
                strains.membrane[r] += membrane(r, 6 * a + k) * u;
                strains.curvature[r] += curvature(r, 6 * a + k) * u;
            }
            for (std::size_t r = 0; r < 2; ++r)
                strains.shear[r] += shear(r, 6 * a + k) * u;
        }
    strains.membrane[0] -= dilatation;
    strains.membrane[1] -= dilatation;
    return strains;
}

// The tangents ∂X0/∂ξ and ∂X0/∂η of the surface X0 = Σ N_a X_a for the shape values `s`.
template <std::size_t N>
[[nodiscard]] std::array<base::Vec3, 2> surfaceTangents(const ShapeValues<N>& s,
                                                        const std::array<base::Vec3, N>& position)
{
    base::Vec3 alongXi{0.0, 0.0, 0.0};
    base::Vec3 alongEta{0.0, 0.0, 0.0};
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t k = 0; k < 3; ++k)
        {
            alongXi[k] += s.dXi[a] * position[a][k];
            alongEta[k] += s.dEta[a] * position[a][k];
        }
    return {alongXi, alongEta};
}

// The unit normal of the surface at (ξ, η).
template <typename Shape>
[[nodiscard]] base::Vec3 surfaceNormalAt(const std::array<base::Vec3, Shape::NODES>& position, double xi, double eta)
{
    const std::array<base::Vec3, 2> tangents = surfaceTangents<Shape::NODES>(Shape::at(xi, eta), position);
    return base::normalized(base::cross(tangents[0], tangents[1]));
}

} // namespace

template <typename Shape>
MindlinShell<Shape>::MindlinShell(const std::array<base::Vec3, N>& position, const std::array<base::Vec3, N>& director,
                                  const ShellSection& section, const std::array<double, C>& thickness,
                                  double drillingFactor)
    : m_position(position), m_director(director), m_section(section), m_drillingFactor(drillingFactor)
{
    // The edge node e lies between the corners e and e + 1 (HigherOrderShapes): the mean of their thicknesses, so
    // that the interpolation of the N nodes reproduces the bilinear (linear) field of the corners.
    for (std::size_t i = 0; i < C; ++i)
        m_nodeThickness[i] = thickness[i];
    for (std::size_t e = 0; e < N - C; ++e)
        m_nodeThickness[C + e] = 0.5 * (thickness[e] + thickness[(e + 1) % C]);
    for (const QuadraturePoint& point : Shape::rule())
        m_area += point.weight * pointAt(point.xi, point.eta).determinant;
}

template <typename Shape>
bool MindlinShell<Shape>::isValidShape(const std::array<double, N>& x, const std::array<double, N>& y)
{
    const auto determinantAt = [&](double xi, double eta)
    {
        const ShapeValues<N> s = Shape::at(xi, eta);
        double j00 = 0.0, j01 = 0.0, j10 = 0.0, j11 = 0.0;
        for (std::size_t i = 0; i < N; ++i)
        {
            j00 += s.dXi[i] * x[i];
            j01 += s.dXi[i] * y[i];
            j10 += s.dEta[i] * x[i];
            j11 += s.dEta[i] * y[i];
        }
        return j00 * j11 - j01 * j10;
    };
    for (const QuadraturePoint& point : Shape::rule())
        if (!(determinantAt(point.xi, point.eta) > 0.0))
            return false;
    for (const QuadraturePoint& point : Shape::massRule())
        if (!(determinantAt(point.xi, point.eta) > 0.0))
            return false;
    for (std::size_t i = 0; i < N; ++i)
        if (!(determinantAt(Shape::XI[i], Shape::ETA[i]) > 0.0))
            return false;
    return true;
}

template <typename Shape>
auto MindlinShell<Shape>::nodalDirectors(const std::array<base::Vec3, N>& position) -> std::array<base::Vec3, N>
{
    std::array<base::Vec3, N> directors{};
    for (std::size_t a = 0; a < N; ++a)
        directors[a] = surfaceNormalAt<Shape>(position, Shape::XI[a], Shape::ETA[a]);
    return directors;
}

template <typename Shape>
double MindlinShell<Shape>::smallestNormalCosine(const std::array<base::Vec3, N>& position)
{
    double smallest = 1.0;
    const auto visit = [&](double xi, double eta)
    { smallest = std::min(smallest, base::dot(surfaceNormalAt<Shape>(position, xi, eta), ELEMENT_NORMAL)); };
    for (std::size_t a = 0; a < N; ++a)
        visit(Shape::XI[a], Shape::ETA[a]);
    for (const QuadraturePoint& point : Shape::rule())
        visit(point.xi, point.eta);
    for (const QuadraturePoint& point : Shape::massRule())
        visit(point.xi, point.eta);
    return smallest;
}

template <typename Shape>
auto MindlinShell<Shape>::pointAt(double xi, double eta) const -> SurfacePoint
{
    SurfacePoint p;
    p.xi = xi;
    p.eta = eta;
    p.shape = Shape::at(xi, eta);
    const std::array<base::Vec3, 2> tangents = surfaceTangents<N>(p.shape, m_position);
    // The tangent frame: e3 the unit normal, e1 the element x axis projected on the tangent plane, e2 = e3 × e1 (on a
    // flat element exactly the element frame: x − 0·n = x).
    const base::Vec3 normal = base::normalized(base::cross(tangents[0], tangents[1]));
    const base::Vec3 e1 = base::normalized(ELEMENT_X - base::dot(ELEMENT_X, normal) * normal);
    const base::Vec3 e2 = base::cross(normal, e1);
    p.frame = {e1, e2, normal};
    // J = [[∂s1/∂ξ, ∂s2/∂ξ], [∂s1/∂η, ∂s2/∂η]] (the tangents' components along e1, e2), [∂/∂s1; ∂/∂s2] = J⁻¹ [∂/∂ξ;
    // ∂/∂η] (the convention of PlaneJacobian); det J = (∂X0/∂ξ × ∂X0/∂η)·e3 = dA / (dξ dη).
    const double j00 = base::dot(e1, tangents[0]);
    const double j01 = base::dot(e2, tangents[0]);
    const double j10 = base::dot(e1, tangents[1]);
    const double j11 = base::dot(e2, tangents[1]);
    p.determinant = j00 * j11 - j01 * j10;
    p.jacobian = {j00, j01, j10, j11};
    for (std::size_t i = 0; i < N; ++i)
    {
        p.dX[i] = (j11 * p.shape.dXi[i] - j01 * p.shape.dEta[i]) / p.determinant;
        p.dY[i] = (-j10 * p.shape.dXi[i] + j00 * p.shape.dEta[i]) / p.determinant;
    }
    // d = n + Σ N_a (ν_a − n) and ∂d/∂s = Σ ∂N_a/∂s (ν_a − n): Σ N_a = 1 and Σ ∂N_a/∂s = 0, so this is Σ N_a ν_a,
    // written so that a flat element (every ν_a = n) gets d = n and ∂d/∂s = 0 exactly.
    base::Vec3 offset{0.0, 0.0, 0.0};
    for (std::size_t a = 0; a < N; ++a)
    {
        const base::Vec3 deviation = m_director[a] - ELEMENT_NORMAL;
        for (std::size_t k = 0; k < 3; ++k)
        {
            offset[k] += p.shape.n[a] * deviation[k];
            p.directorDX[k] += p.dX[a] * deviation[k];
            p.directorDY[k] += p.dY[a] * deviation[k];
        }
    }
    p.director = ELEMENT_NORMAL + offset;
    return p;
}

template <typename Shape>
double MindlinShell<Shape>::thicknessAt(const ShapeValues<N>& shape) const
{
    double t = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        t += shape.n[i] * m_nodeThickness[i];
    return t;
}

template <typename Shape>
double MindlinShell<Shape>::deltaTAt(const ShellTemperatureChange& temperature, const ShapeValues<N>& shape) const
{
    if (!temperature.hasCornerValues)
        return temperature.deltaT;
    double value = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        value += shape.n[i] * temperature.atCorners[i];
    return value;
}

template <typename Shape>
base::MatN<2> MindlinShell<Shape>::shearStiffness(double t) const
{
    // Along the material axes, then in the element frame: S = R diag(S₁, S₂) Rᵀ, R the rotation by the material angle
    // (the shear strain and force transform as vectors).
    std::array<double, 2> principal{};
    for (std::size_t k = 0; k < 2; ++k)
    {
        const double stiffness = m_section.transverseShearStiffness[k];
        // MID3 blank (or a rigid direction): the shear of the bending material (see the class comment).
        principal[k] = t * (stiffness > 0.0 ? stiffness : m_section.shearRatio * m_section.bendingMatrix(2, 2));
    }
    const double c = std::cos(m_section.frameAngle);
    const double sine = std::sin(m_section.frameAngle);
    base::MatN<2> s;
    s(0, 0) = c * c * principal[0] + sine * sine * principal[1];
    s(1, 1) = sine * sine * principal[0] + c * c * principal[1];
    s(0, 1) = c * sine * (principal[0] - principal[1]);
    s(1, 0) = s(0, 1);
    return s;
}

template <typename Shape>
auto MindlinShell<Shape>::stiffness() const -> base::MatN<DOFS>
{
    base::MatN<DOFS> k;
    const TiedStrains tied = tiedStrains();
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const StrainRows rows = rowsAt(p, tied);
        const double t = thicknessAt(p.shape);
        const double scale = point.weight * p.determinant;
        if (m_section.hasMembrane)
            k += (scale * t) * base::congruence(m_section.membraneMatrix, rows.membrane);
        if (m_section.hasBending)
        {
            k += scale * base::congruence(m_section.bendingStiffness(t), rows.curvature);
            k += scale * base::congruence(shearStiffness(t), rows.shear);
        }
        if (m_section.hasCoupling && m_section.hasMembrane && m_section.hasBending)
        {
            const base::MatN<DOFS> half =
                (SHELL_COUPLING_SIGN * scale) *
                (rows.membrane.transposed() * (m_section.couplingStiffness(t) * rows.curvature));
            k += half + half.transposed();
        }
    }
    if (hasDrilling())
        addDrilling(k, m_section.membraneMatrix(2, 2));
    return k;
}

template <typename Shape>
auto MindlinShell<Shape>::tiedStrains() const -> TiedStrains
{
    TiedStrains tied;
    if constexpr (Shape::HAS_TIED_STRAINS)
    {
        // The covariant shear along a g_ξ + b g_η from the local shears: with d = (a ∂X0/∂ξ + b ∂X0/∂η) in the
        // tangent axes, γ̃ = d1 γx + d2 γy.
        const std::array<ShearTie, SHEAR_TIES> shears = shearTies();
        for (std::size_t i = 0; i < SHEAR_TIES; ++i)
        {
            const SurfacePoint p = pointAt(shears[i].xi, shears[i].eta);
            const base::MatN<2, DOFS> local = shearRowsOf<N>(p, m_director);
            const double d1 = shears[i].alongXi * p.jacobian[0] + shears[i].alongEta * p.jacobian[2];
            const double d2 = shears[i].alongXi * p.jacobian[1] + shears[i].alongEta * p.jacobian[3];
            for (std::size_t c = 0; c < DOFS; ++c)
                tied.shear(i, c) = d1 * local(0, c) + d2 * local(1, c);
        }
        // The covariant membrane strain along the same vector: ẽ = d1² εx + d2² εy + d1 d2 γxy; along g_ξ, along g_η
        // and along g_η − g_ξ.
        constexpr std::array<std::array<double, 2>, 3> ALONG{{{1.0, 0.0}, {0.0, 1.0}, {-1.0, 1.0}}};
        const std::array<std::array<TiePoint, 3>, 3> membranes = membraneTies();
        for (std::size_t strain = 0; strain < 3; ++strain)
            for (std::size_t i = 0; i < 3; ++i)
            {
                const SurfacePoint p = pointAt(membranes[strain][i][0], membranes[strain][i][1]);
                const base::MatN<3, DOFS> local = membraneRowsOf<N>(p);
                const double d1 = ALONG[strain][0] * p.jacobian[0] + ALONG[strain][1] * p.jacobian[2];
                const double d2 = ALONG[strain][0] * p.jacobian[1] + ALONG[strain][1] * p.jacobian[3];
                for (std::size_t c = 0; c < DOFS; ++c)
                    tied.membrane(3 * strain + i, c) =
                        d1 * d1 * local(0, c) + d2 * d2 * local(1, c) + d1 * d2 * local(2, c);
            }
    }
    return tied;
}

template <typename Shape>
auto MindlinShell<Shape>::rowsAt(const SurfacePoint& p, const TiedStrains& tied) const -> StrainRows
{
    StrainRows rows = shellRows<N>(p, m_director);
    if constexpr (Shape::HAS_TIED_STRAINS)
    {
        // The local components from the covariant ones: c(α, i) = g^α·e_i, the inverse of the tangents' components
        // along the tangent axes; γ_i = Σ c(α, i) γ̃_α and ε_ij = Σ c(α, i) c(β, j) ẽ_αβ.
        const double xiOnX = p.jacobian[3] / p.determinant;
        const double etaOnX = -p.jacobian[1] / p.determinant;
        const double xiOnY = -p.jacobian[2] / p.determinant;
        const double etaOnY = p.jacobian[0] / p.determinant;
        const base::MatN<2, SHEAR_TIES> shearWeights = shearTieWeightsAt(p.xi, p.eta);
        const std::array<double, 3> alongXi = membraneTieWeightsAt(0, p.xi, p.eta);
        const std::array<double, 3> alongEta = membraneTieWeightsAt(1, p.xi, p.eta);
        const std::array<double, 3> alongSide3 = membraneTieWeightsAt(2, p.xi, p.eta);
        for (std::size_t c = 0; c < DOFS; ++c)
        {
            double shearXi = 0.0;
            double shearEta = 0.0;
            for (std::size_t i = 0; i < SHEAR_TIES; ++i)
            {
                shearXi += shearWeights(0, i) * tied.shear(i, c);
                shearEta += shearWeights(1, i) * tied.shear(i, c);
            }
            rows.shear(0, c) = xiOnX * shearXi + etaOnX * shearEta;
            rows.shear(1, c) = xiOnY * shearXi + etaOnY * shearEta;
            double xiXi = 0.0;
            double etaEta = 0.0;
            double side3 = 0.0;
            for (std::size_t i = 0; i < 3; ++i)
            {
                xiXi += alongXi[i] * tied.membrane(i, c);
                etaEta += alongEta[i] * tied.membrane(3 + i, c);
                side3 += alongSide3[i] * tied.membrane(6 + i, c);
            }
            const double xiEta = 0.5 * (xiXi + etaEta - side3); // ẽ_qq = ẽ_ξξ + ẽ_ηη − 2 ẽ_ξη
            rows.membrane(0, c) = xiOnX * xiOnX * xiXi + etaOnX * etaOnX * etaEta + 2.0 * xiOnX * etaOnX * xiEta;
            rows.membrane(1, c) = xiOnY * xiOnY * xiXi + etaOnY * etaOnY * etaEta + 2.0 * xiOnY * etaOnY * xiEta;
            rows.membrane(2, c) = 2.0 * (xiOnX * xiOnY * xiXi + etaOnX * etaOnY * etaEta +
                                         (xiOnX * etaOnY + etaOnX * xiOnY) * xiEta);
        }
    }
    return rows;
}

template <typename Shape>
auto MindlinShell<Shape>::drillingRowAt(const SurfacePoint& p) const -> std::array<double, DOFS>
{
    // r = ω − θ_n (docs/shell-quad4-formulation.md, section 8): ω = ½ (e2·∂u0/∂s1 − e1·∂u0/∂s2) the rotation of the
    // surface about its normal, θ_n = Σ N_a (θ_a − θ̄)·ν_a + θ̄·e3 that of the nodes about their directors, θ̄ the
    // mean rotation of the nodes. Zero for a rigid motion.
    std::array<double, DOFS> row{};
    const base::Vec3& e1 = p.frame[0];
    const base::Vec3& e2 = p.frame[1];
    const base::Vec3 onMean = p.frame[2] - p.director; // the coefficient of θ̄ in θ_n: e3 − Σ N_a ν_a
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t k = 0; k < 3; ++k)
        {
            row[6 * a + k] = 0.5 * (p.dX[a] * e2[k] - p.dY[a] * e1[k]);
            row[6 * a + 3 + k] = -(p.shape.n[a] * m_director[a][k] + onMean[k] / static_cast<double>(N));
        }
    return row;
}

template <typename Shape>
void MindlinShell<Shape>::addDrilling(base::MatN<DOFS>& k, double shearModulus) const
{
    // The full rule: the reduced rule of the quadrilateral would leave the rotations ξ² − η² of the edge nodes about
    // their directors, zero at its four points, without stiffness across a whole mesh.
    for (const QuadraturePoint& point : Shape::massRule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const std::array<double, DOFS> row = drillingRowAt(p);
        const double scale = m_drillingFactor * point.weight * p.determinant * thicknessAt(p.shape) * shearModulus;
        for (std::size_t i = 0; i < DOFS; ++i)
        {
            if (row[i] == 0.0)
                continue;
            const double left = scale * row[i];
            for (std::size_t j = 0; j < DOFS; ++j)
                k(i, j) += left * row[j];
        }
    }
}

template <typename Shape>
double MindlinShell<Shape>::dilatationOf(const ShellTemperatureChange& temperature) const
{
    if (!m_section.hasMembrane || temperature.isZero())
        return 0.0;
    // The free membrane strain of a point: (t G1)⁻¹ N_T.
    const base::Mat3 compliance = base::inverse(m_section.membraneMatrix);
    double sum = 0.0;
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const double t = thicknessAt(p.shape);
        const ThermalResultants thermal =
            m_section.thermalResultants(deltaTAt(temperature, p.shape), temperature.gradient, t);
        const std::array<double, 3> strain = compliance * thermal.force;
        sum += point.weight * p.determinant * 0.5 * (strain[0] + strain[1]) / t;
    }
    return sum / m_area;
}

template <typename Shape>
auto MindlinShell<Shape>::mass(MassFormulation formulation, double areaMass) const -> base::MatN<DOFS>
{
    // M_ij = ∫ m N_i N_j dA, m = ρ t + NSM, and the rotary inertia ∫ ρ t³/12 N_i N_j dA, with the full rule of the
    // shape: the stiffness rule is too low for N_i N_j (rank 4 of 8 for the ShellQuad8, 3 of 6 for the ShellTria6), and
    // its diagonal is not the one HRZ lumping is defined on.
    const double nonStructural = m_section.nonStructuralMass + areaMass;
    base::MatN<N> translational;
    base::MatN<N> rotary;
    double total = 0.0;
    for (const QuadraturePoint& point : Shape::massRule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const double t = thicknessAt(p.shape);
        const double scale = point.weight * p.determinant;
        const double areal = m_section.density * t + nonStructural;
        total += scale * areal;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
            {
                translational(i, j) += scale * areal * p.shape.n[i] * p.shape.n[j];
                rotary(i, j) += scale * m_section.density * t * t * t / 12.0 * p.shape.n[i] * p.shape.n[j];
            }
    }
    base::MatN<DOFS> m;
    if (formulation == MassFormulation::Lumped)
    {
        // HRZ lumping (Hinton, Rock, Zienkiewicz 1976): the diagonal of the consistent translational mass scaled
        // so that each direction carries the element's whole mass; no rotary inertia.
        double diagonalSum = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            diagonalSum += translational(i, i);
        // A massless element has a zero diagonal, not 0/0.
        const double scale = diagonalSum > 0.0 ? total / diagonalSum : 0.0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(6 * i + d, 6 * i + d) = translational(i, i) * scale;
        return m;
    }
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            for (std::size_t d = 0; d < 3; ++d)
                m(6 * i + d, 6 * j + d) = translational(i, j);
            // The fibre rotation b = Σ N_a (θ_a × ν_a): b·b gives θ_iᵀ ((ν_i·ν_j) I − ν_j ν_iᵀ) θ_j N_i N_j
            // ([ν_i]×ᵀ [ν_j]×), diag(1, 1, 0) on a flat element.
            const base::Vec3& ni = m_director[i];
            const base::Vec3& nj = m_director[j];
            const double cosine = base::dot(ni, nj);
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    m(6 * i + 3 + r, 6 * j + 3 + c) = rotary(i, j) * ((r == c ? cosine : 0.0) - nj[r] * ni[c]);
        }
    return m;
}

template <typename Shape>
auto MindlinShell<Shape>::pressureLoads(const std::array<double, C>& cornerPressures,
                                        const std::optional<base::Vec3>& direction) const -> std::array<double, DOFS>
{
    // The corner pressures interpolated by the corner shape functions of the linear (bilinear) element under the
    // edge nodes: p(ξ, η) = Σ_corners M_i p_i, with M the 3-node or 4-node shapes; F_i = ∫ N_i p e3 dA (the normal of
    // the surface at each point; e3 dA = ∂X0/∂ξ × ∂X0/∂η dξ dη) or F_i = (∫ N_i p dA) v along a given direction v,
    // with the full rule of the shape (exact on a flat element of straight edges: the consistent loads).
    std::array<double, N> along{};       // ∫ N_i p dA, for a given direction
    std::array<base::Vec3, N> normal{};  // ∫ N_i p e3 dA
    for (const QuadraturePoint& point : Shape::massRule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        double pressure = 0.0;
        if constexpr (C == 4)
        {
            const BilinearShape corners = bilinearShapeAt(point.xi, point.eta);
            for (std::size_t i = 0; i < 4; ++i)
                pressure += corners.n[i] * cornerPressures[i];
        }
        else
        {
            const std::array<double, 3> corners{1.0 - point.xi - point.eta, point.xi, point.eta};
            for (std::size_t i = 0; i < 3; ++i)
                pressure += corners[i] * cornerPressures[i];
        }
        const base::Vec3& e3 = p.frame[2];
        for (std::size_t i = 0; i < N; ++i)
        {
            const double share = point.weight * p.determinant * pressure * p.shape.n[i];
            along[i] += share;
            for (std::size_t d = 0; d < 3; ++d)
                normal[i][d] += share * e3[d];
        }
    }
    std::array<double, DOFS> loads{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t d = 0; d < 3; ++d)
            loads[6 * i + d] = direction ? along[i] * (*direction)[d] : normal[i][d];
    return loads;
}

template <typename Shape>
auto MindlinShell<Shape>::thermalLoads(const ShellTemperatureChange& temperature) const -> std::array<double, DOFS>
{
    // f = Σ_g w dA (B_mᵀ N_T − B_bᵀ M_T): the initial strains α ΔT of the membrane and the initial curvatures
    // κ_T = −D⁻¹ M_T of the bending (the sign of the shell curvature), with the stiffness rule: the loads of a
    // thermal strain are then those K gives that strain, and a free expansion is stress-free. The dilatation ε of
    // the free strain (dilatationOf) is taken apart: its loads are K q for the homothety q = ε x of the element, so
    // that a curved element, or one whose directors are the normals of its grid points, expands freely under a
    // uniform temperature rise as a flat one does. Written with the strains (dm, dk, dg) the homothety has off
    // those of a flat element (dilatationStrainsOf):
    //   f = Σ_g w dA (B_mᵀ (N_T + t G1 dm + s [B] dk) − B_bᵀ (M_T − D dk − s [B] dm) + B_sᵀ S dg).
    std::array<double, DOFS> loads{};
    if (temperature.isZero())
        return loads;
    const TiedStrains tied = tiedStrains();
    const double dilatation = dilatationOf(temperature);
    const bool isCoupled = m_section.hasCoupling && m_section.hasMembrane && m_section.hasBending;
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const StrainRows rows = rowsAt(p, tied);
        const double t = thicknessAt(p.shape);
        ThermalResultants thermal =
            m_section.thermalResultants(deltaTAt(temperature, p.shape), temperature.gradient, t);
        std::array<double, 2> shearForce{};
        if (dilatation != 0.0)
        {
            const DilatationStrains free =
                dilatationStrainsOf<N>(rows.membrane, rows.curvature, rows.shear, m_position, dilatation);
            const std::array<double, 3> force = m_section.membraneMatrix * free.membrane;
            for (std::size_t c = 0; c < 3; ++c)
                thermal.force[c] += t * force[c];
            if (m_section.hasBending)
            {
                const std::array<double, 3> moment = m_section.bendingStiffness(t) * free.curvature;
                for (std::size_t c = 0; c < 3; ++c)
                    thermal.moment[c] -= moment[c];
                shearForce = shearStiffness(t) * free.shear;
            }
            if (isCoupled)
            {
                const base::Mat3 coupling = m_section.couplingStiffness(t);
                const std::array<double, 3> onForce = coupling * free.curvature;
                const std::array<double, 3> onMoment = coupling * free.membrane;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    thermal.force[c] += SHELL_COUPLING_SIGN * onForce[c];
                    thermal.moment[c] -= SHELL_COUPLING_SIGN * onMoment[c];
                }
            }
        }
        const double scale = point.weight * p.determinant;
        const std::array<double, DOFS> membrane = rows.membrane.transposed() * thermal.force;
        const std::array<double, DOFS> bending = rows.curvature.transposed() * thermal.moment;
        const std::array<double, DOFS> shear = rows.shear.transposed() * shearForce;
        for (std::size_t c = 0; c < DOFS; ++c)
            loads[c] += scale * ((m_section.hasMembrane ? membrane[c] : 0.0) -
                                 (m_section.hasBending ? bending[c] - shear[c] : 0.0));
    }
    return loads;
}

template <typename Shape>
auto MindlinShell<Shape>::placeStates(const std::array<double, DOFS>& q, const std::array<std::int32_t, N>& gridIds,
                                      const ShellTemperatureChange& temperature) const
    -> std::array<ShellPlaceState, C + 1>
{
    const PlaceOperators operators = placeOperators();
    return placeStates(operators, placeThermal(operators, temperature), q, gridIds);
}

template <typename Shape>
auto MindlinShell<Shape>::placeOperators() const -> PlaceOperators
{
    PlaceOperators operators;
    const bool isCoupled = m_section.hasCoupling && m_section.hasMembrane && m_section.hasBending;
    const TiedStrains tied = tiedStrains();
    // The quadrilateral: the strains of the displacements are the element's at the points of its reduced rule (the
    // points of best accuracy of the reduced integration: J. Barlow, Optimal stress locations in finite element
    // models, IJNME 10 (1976) 243-251); the rows of a place are carried from these four points by the bilinear
    // functions through them, (1 + 3 ξ ξ_g)(1 + 3 η η_g) / 4. The triangle: its tied strains at the place.
    const auto rule = Shape::rule();
    std::array<StrainRows, std::tuple_size<decltype(rule)>::value> atRule;
    if constexpr (!Shape::HAS_TIED_STRAINS)
        for (std::size_t g = 0; g < rule.size(); ++g)
            atRule[g] = rowsAt(pointAt(rule[g].xi, rule[g].eta), tied);
    for (std::size_t place = 0; place <= C; ++place)
    {
        const bool isCentre = place == 0;
        const double xi = isCentre ? Shape::CENTRE_XI : Shape::XI[place - 1];
        const double eta = isCentre ? Shape::CENTRE_ETA : Shape::ETA[place - 1];
        const SurfacePoint p = pointAt(xi, eta);
        StrainRows rows;
        if constexpr (Shape::HAS_TIED_STRAINS)
        {
            rows = rowsAt(p, tied);
        }
        else
        {
            for (std::size_t g = 0; g < rule.size(); ++g)
            {
                const double weight = 0.25 * (1.0 + 3.0 * xi * rule[g].xi) * (1.0 + 3.0 * eta * rule[g].eta);
                rows.membrane += weight * atRule[g].membrane;
                rows.curvature += weight * atRule[g].curvature;
                rows.shear += weight * atRule[g].shear;
            }
        }
        operators.shape[place] = p.shape;
        operators.membrane[place] = rows.membrane;
        operators.curvature[place] = rows.curvature;
        operators.shear[place] = rows.shear;
        const double t = thicknessAt(p.shape);
        operators.thickness[place] = t;
        operators.coupling[place] = isCoupled ? m_section.couplingStiffness(t) : base::Mat3{};
        if (m_section.hasBending)
        {
            operators.bending[place] = m_section.bendingStiffness(t);
            operators.shearStiffness[place] = shearStiffness(t);
        }
    }
    return operators;
}

template <typename Shape>
auto MindlinShell<Shape>::placeThermal(const PlaceOperators& operators, const ShellTemperatureChange& temperature) const
    -> PlaceThermal
{
    PlaceThermal thermal;
    const double dilatation = dilatationOf(temperature);
    for (std::size_t place = 0; place <= C; ++place)
    {
        thermal.temperature[place] = temperature;
        thermal.temperature[place].deltaT = deltaTAt(temperature, operators.shape[place]);
        thermal.resultants[place] = m_section.thermalResultants(thermal.temperature[place].deltaT, temperature.gradient,
                                                                operators.thickness[place]);
        if (dilatation != 0.0)
        {
            const DilatationStrains free = dilatationStrainsOf<N>(
                operators.membrane[place], operators.curvature[place], operators.shear[place], m_position, dilatation);
            thermal.membraneOfDilatation[place] = free.membrane;
            thermal.curvatureOfDilatation[place] = free.curvature;
            thermal.shearOfDilatation[place] = free.shear;
        }
    }
    return thermal;
}

// PERFORMANCE: the state of the places from the parts formed once (PlaceOperators, PlaceThermal): the recovery slopes
// evaluate 1 + 6N states per element, each of which formed the shape functions, the Jacobian, the strain rows and the
// stiffnesses of every place again; the operations on the displacements are those of before, in the same order,
// so their bits.
// Measured: the recovery of ShellQuad8 and ShellTria6 plates.
template <typename Shape>
auto MindlinShell<Shape>::placeStates(const PlaceOperators& operators, const PlaceThermal& thermal,
                                      const std::array<double, DOFS>& q,
                                      const std::array<std::int32_t, N>& gridIds) const
    -> std::array<ShellPlaceState, C + 1>
{
    std::array<ShellPlaceState, C + 1> states{};
    for (std::size_t place = 0; place <= C; ++place)
    {
        const bool isCentre = place == 0;
        ShellPlaceState& state = states[place];
        state.nodeId = isCentre ? 0 : gridIds[place - 1];
        state.thickness = operators.thickness[place];
        state.temperature = thermal.temperature[place];
        // The strains of the displacements, less what the strains of the free dilatation of the element are off
        // those of a flat element (thermalLoads): the stresses then follow as on a flat element.
        state.membraneStrain = operators.membrane[place] * q;
        state.curvature = operators.curvature[place] * q;
        std::array<double, 2> shear = operators.shear[place] * q;
        for (std::size_t c = 0; c < 3; ++c)
        {
            state.membraneStrain[c] -= thermal.membraneOfDilatation[place][c];
            state.curvature[c] -= thermal.curvatureOfDilatation[place][c];
        }
        for (std::size_t c = 0; c < 2; ++c)
            shear[c] -= thermal.shearOfDilatation[place][c];
        const ThermalResultants& resultants = thermal.resultants[place];
        const base::Mat3& b = operators.coupling[place];
        if (m_section.hasMembrane)
        {
            const std::array<double, 3> force = m_section.membraneMatrix * state.membraneStrain;
            const std::array<double, 3> coupled = b * state.curvature;
            for (std::size_t c = 0; c < 3; ++c)
                state.membraneForce[c] =
                    state.thickness * force[c] + SHELL_COUPLING_SIGN * coupled[c] - resultants.force[c];
        }
        if (m_section.hasBending)
        {
            const std::array<double, 3> moment = operators.bending[place] * state.curvature;
            const std::array<double, 3> coupled = b * state.membraneStrain;
            for (std::size_t c = 0; c < 3; ++c)
                state.moment[c] = -(moment[c] + SHELL_COUPLING_SIGN * coupled[c] + resultants.moment[c]);
            state.shearForce = operators.shearStiffness[place] * shear;
        }
    }
    return states;
}

template <typename Shape>
base::MatN<2> MindlinShell<Shape>::shearStiffnessRate(double t, const ShellSectionRate& rate) const
{
    // shearStiffness differentiated: each principal value t·κG (flexible) or t·TS/T·G2_33 (MID3 blank, rigid).
    std::array<double, 2> principal{};
    for (std::size_t k = 0; k < 2; ++k)
        principal[k] =
            t * (m_section.transverseShearStiffness[k] > 0.0 ? rate.transverseShearStiffness[k]
                                                             : m_section.shearRatio * rate.bendingMatrix(2, 2));
    const double c = std::cos(m_section.frameAngle);
    const double sine = std::sin(m_section.frameAngle);
    base::MatN<2> s;
    s(0, 0) = c * c * principal[0] + sine * sine * principal[1];
    s(1, 1) = sine * sine * principal[0] + c * c * principal[1];
    s(0, 1) = c * sine * (principal[0] - principal[1]);
    s(1, 0) = s(0, 1);
    return s;
}

template <typename Shape>
auto MindlinShell<Shape>::stiffnessSectionRate(const ShellSectionRate& rate) const -> base::MatN<DOFS>
{
    base::MatN<DOFS> k;
    const TiedStrains tied = tiedStrains();
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const StrainRows rows = rowsAt(p, tied);
        const double t = thicknessAt(p.shape);
        const double scale = point.weight * p.determinant;
        if (m_section.hasMembrane && rate.movesMembrane)
            k += (scale * t) * base::congruence(rate.membraneMatrix, rows.membrane);
        if (m_section.hasBending)
        {
            if (rate.movesBending)
                k += scale *
                     base::congruence(m_section.bendingRatio * t * t * t / 12.0 * rate.bendingMatrix, rows.curvature);
            if (rate.movesBending || rate.movesShear)
                k += scale * base::congruence(shearStiffnessRate(t, rate), rows.shear);
        }
        // K_mb is linear in [B]: the rate's coupling, the coupling zero at the point or not.
        if (m_section.hasMembrane && m_section.hasBending && rate.movesCoupling)
        {
            const base::MatN<DOFS> half =
                (SHELL_COUPLING_SIGN * scale) *
                (rows.membrane.transposed() * (((t * t) * rate.couplingMatrix) * rows.curvature));
            k += half + half.transposed();
        }
    }
    // The drilling term is linear in the shear modulus of the membrane.
    if (hasDrilling() && rate.movesMembrane)
        addDrilling(k, rate.membraneMatrix(2, 2));
    return k;
}

template <typename Shape>
auto MindlinShell<Shape>::placeStateSectionRates(const std::array<ShellPlaceState, C + 1>& states,
                                                 const std::array<double, DOFS>& q, const ShellSectionRate& rate) const
    -> std::array<ShellPlaceState, C + 1>
{
    const PlaceOperators operators = placeOperators();
    // A coupling zero at the point moved by the rate counts: N and M are linear in [B].
    const bool isCoupled =
        (m_section.hasCoupling || rate.movesCoupling) && m_section.hasMembrane && m_section.hasBending;
    std::array<ShellPlaceState, C + 1> rates{};
    for (std::size_t place = 0; place <= C; ++place)
    {
        const ShellPlaceState& state = states[place];
        ShellPlaceState& out = rates[place];
        out.nodeId = state.nodeId;
        const double t = operators.thickness[place];
        const base::Mat3 couplingRate = isCoupled ? (t * t) * rate.couplingMatrix : base::Mat3{};
        if (m_section.hasMembrane)
        {
            const std::array<double, 3> force = rate.membraneMatrix * state.membraneStrain;
            const std::array<double, 3> coupled = couplingRate * state.curvature;
            for (std::size_t c = 0; c < 3; ++c)
                out.membraneForce[c] = t * force[c] + SHELL_COUPLING_SIGN * coupled[c];
        }
        if (m_section.hasBending)
        {
            const std::array<double, 3> moment =
                (m_section.bendingRatio * t * t * t / 12.0 * rate.bendingMatrix) * state.curvature;
            const std::array<double, 3> coupled = couplingRate * state.membraneStrain;
            for (std::size_t c = 0; c < 3; ++c)
                out.moment[c] = -(moment[c] + SHELL_COUPLING_SIGN * coupled[c]);
            out.shearForce = shearStiffnessRate(t, rate) * (operators.shear[place] * q);
        }
    }
    return rates;
}

template <typename Shape>
auto MindlinShell<Shape>::differentialStiffness(const std::array<double, 3>& membraneForce) const -> base::MatN<DOFS>
{
    // ShellDifferential on the derivatives along the tangent axes of each point: u_k,s_i·u_k,s_j summed over the three
    // translations of the element frame is the same in any Cartesian frame, so the matrix is that of a flat element
    // laid on the tangent plane, the resultants of the centre taken in each point's tangent frame.
    base::MatN<DOFS> k;
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        addMembraneDifferentialStiffness<N, 6>(k, p.dX, p.dY, point.weight * p.determinant, membraneForce);
    }
    return k;
}

template class MindlinShell<SerendipityQuadShape>;
template class MindlinShell<QuadraticTriangleShape>;

} // namespace oos::elements
