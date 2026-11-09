#include "oos/elements/shells/MindlinShell.hpp"

#include "oos/elements/core/ShapeFunctions.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements
{

namespace
{

// The element frame's x axis and normal n, in the element frame.
constexpr base::Vec3 ELEMENT_X{1.0, 0.0, 0.0};
constexpr base::Vec3 ELEMENT_NORMAL{0.0, 0.0, 1.0};

// The strain-displacement rows over the 6N local DOFs at a point (the class comment of MindlinShell): the membrane
// strains (εx, εy, γxy), the shell curvatures (κx, κy, 2κxy) and the transverse shear strains (γx, γy), in the tangent
// frame of the point.
template <std::size_t N>
struct ShellRows
{
    base::MatN<3, 6 * N> membrane;
    base::MatN<3, 6 * N> curvature;
    base::MatN<2, 6 * N> shear;
};

// The symmetric correction of the bending strain by the membrane strain on a curved surface (the class comment of
// MindlinShell): κ += ½ (Bᵀε + εB), B_ij = e_i·∂d/∂s_j the curvature of the director field, ε the membrane strain
// tensor (ε12 = γxy/2). In the order (κx, κy, 2κxy) of the rows:
//   κx += B11 εx + ½ B21 γxy,   κy += B22 εy + ½ B12 γxy,   2κxy += B12 εx + B21 εy + ½ (B11 + B22) γxy.
// A uniform dilatation u0 = e X0 (k = e·sym B) then bends nothing. Zero on a flat element (B = 0).
template <std::size_t N, typename Point>
void addCurvatureOfStretching(ShellRows<N>& rows, const Point& p)
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
[[nodiscard]] ShellRows<N> shellRows(const Point& p, const std::array<base::Vec3, N>& director)
{
    ShellRows<N> rows;
    const base::Vec3& e1 = p.frame[0];
    const base::Vec3& e2 = p.frame[1];
    for (std::size_t a = 0; a < N; ++a)
    {
        const std::size_t u = 6 * a;
        const std::size_t theta = u + 3;
        const double dX = p.dX[a];
        const double dY = p.dY[a];
        const double n = p.shape.n[a];
        const base::Vec3 r1 = base::cross(director[a], e1);
        const base::Vec3 r2 = base::cross(director[a], e2);
        for (std::size_t k = 0; k < 3; ++k)
        {
            rows.membrane(0, u + k) = dX * e1[k];
            rows.membrane(1, u + k) = dY * e2[k];
            rows.membrane(2, u + k) = dY * e1[k] + dX * e2[k];
            rows.curvature(0, u + k) = -(dX * p.directorDX[k]);
            rows.curvature(0, theta + k) = -(dX * r1[k]);
            rows.curvature(1, u + k) = -(dY * p.directorDY[k]);
            rows.curvature(1, theta + k) = -(dY * r2[k]);
            rows.curvature(2, u + k) = -(dY * p.directorDX[k] + dX * p.directorDY[k]);
            rows.curvature(2, theta + k) = -(dY * r1[k] + dX * r2[k]);
            rows.shear(0, u + k) = dX * p.director[k];
            rows.shear(0, theta + k) = n * r1[k];
            rows.shear(1, u + k) = dY * p.director[k];
            rows.shear(1, theta + k) = n * r2[k];
        }
    }
    addCurvatureOfStretching<N>(rows, p);
    return rows;
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
                                  const ShellSection& section, const std::array<double, C>& thickness)
    : m_position(position), m_director(director), m_section(section)
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
    // No term on the rotation about a director: the element has no drilling stiffness (the class comment).
    base::MatN<DOFS> k;
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const ShellRows<N> rows = shellRows<N>(p, m_director);
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
    return k;
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
    // thermal strain are then those K gives that strain, and a free expansion is stress-free.
    std::array<double, DOFS> loads{};
    if (temperature.isZero())
        return loads;
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const ShellRows<N> rows = shellRows<N>(p, m_director);
        const double t = thicknessAt(p.shape);
        const ThermalResultants thermal =
            m_section.thermalResultants(deltaTAt(temperature, p.shape), temperature.gradient, t);
        const double scale = point.weight * p.determinant;
        const std::array<double, DOFS> membrane = rows.membrane.transposed() * thermal.force;
        const std::array<double, DOFS> bending = rows.curvature.transposed() * thermal.moment;
        for (std::size_t c = 0; c < DOFS; ++c)
            loads[c] +=
                scale * ((m_section.hasMembrane ? membrane[c] : 0.0) - (m_section.hasBending ? bending[c] : 0.0));
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
    for (std::size_t place = 0; place <= C; ++place)
    {
        const bool isCentre = place == 0;
        const double xi = isCentre ? Shape::CENTRE_XI : Shape::XI[place - 1];
        const double eta = isCentre ? Shape::CENTRE_ETA : Shape::ETA[place - 1];
        const SurfacePoint p = pointAt(xi, eta);
        const ShellRows<N> rows = shellRows<N>(p, m_director);
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
    for (std::size_t place = 0; place <= C; ++place)
    {
        thermal.temperature[place] = temperature;
        thermal.temperature[place].deltaT = deltaTAt(temperature, operators.shape[place]);
        thermal.resultants[place] = m_section.thermalResultants(thermal.temperature[place].deltaT, temperature.gradient,
                                                                operators.thickness[place]);
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
        state.membraneStrain = operators.membrane[place] * q;
        state.curvature = operators.curvature[place] * q;
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
            state.shearForce = operators.shearStiffness[place] * (operators.shear[place] * q);
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
    for (const QuadraturePoint& point : Shape::rule())
    {
        const SurfacePoint p = pointAt(point.xi, point.eta);
        const ShellRows<N> rows = shellRows<N>(p, m_director);
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
