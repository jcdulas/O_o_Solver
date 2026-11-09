#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingPlate.hpp"

#include "oos/elements/core/MatrixKernels.hpp"
#include "oos/elements/shells/ShellDifferential.hpp"

#include <cmath>

namespace oos::elements
{

namespace
{

constexpr std::size_t NODES = 3;

// The plate takes no membrane-bending coupling (the family refuses one): the resultant rates carry none.
constexpr bool IS_COUPLED = false;

// The constant of the residual bending compliance L²/(24 D) of the TRIA3 shear (NASTRAN-95 tria3d.f: TSM = 1/(2·12·I)).
constexpr double TRIA_BENDING_COMPLIANCE_FACTOR = 24.0;

// The local DOF of corner i: membrane (u, v) at 0, 1, bending (w, θx, θy) at 2, 3, 4, θn at 5.
[[nodiscard]] constexpr std::size_t local(std::size_t node, std::size_t dof)
{
    return 6 * node + dof;
}

// The local DOF of DOF a of the bending (w, θx, θy per corner).
[[nodiscard]] constexpr std::size_t bendingDof(std::size_t a)
{
    return local(a / 3, 2 + a % 3);
}

// Adds a matrix over the bending DOFs (9 × 9) into the 18 local DOFs.
void addOnBendingDofs(base::MatN<18>& k, const base::MatN<9>& bending)
{
    for (std::size_t a = 0; a < 9; ++a)
        for (std::size_t b = 0; b < 9; ++b)
            k(bendingDof(a), bendingDof(b)) += bending(a, b);
}

// The edges G1→G2, G2→G3, G3→G1: unit vectors and lengths in the element plane.
struct TriangleEdges
{
    std::array<double, 3> ux{};
    std::array<double, 3> uy{};
    std::array<double, 3> length{};
};

[[nodiscard]] TriangleEdges edgesOf(const ShellTria3DrillingGeometry& g)
{
    TriangleEdges edges;
    for (std::size_t e = 0; e < NODES; ++e)
    {
        const std::size_t b = (e + 1) % NODES;
        const double dx = g.x[b] - g.x[e];
        const double dy = g.y[b] - g.y[e];
        edges.length[e] = std::hypot(dx, dy);
        edges.ux[e] = dx / edges.length[e];
        edges.uy[e] = dy / edges.length[e];
    }
    return edges;
}

// The local DOF of DOF a of the membrane (u, v, θn per corner).
[[nodiscard]] constexpr std::size_t membraneDof(std::size_t a)
{
    constexpr std::array<std::size_t, 3> OF_NODE{0, 1, 5};
    return local(a / 3, OF_NODE[a % 3]);
}

// The three points of the quadratic rule in area coordinates (ShellTria3DrillingMembrane), weight A_e/3 each.
constexpr std::array<std::array<double, 3>, 3> DRILLING_RULE{
    {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}}};

// The exact integral over a triangle of area A of L_a L_b L_c (area coordinates): 2A·α!β!γ!/(α + β + γ + 2)!, the
// multiplicities of the three indices; for three indices 2A·{6, 2, 1}/120 when all, two or none coincide.
[[nodiscard]] double cubicIntegral(std::size_t a, std::size_t b, std::size_t c, double area)
{
    std::array<int, 3> multiplicity{};
    ++multiplicity[a];
    ++multiplicity[b];
    ++multiplicity[c];
    double factorials = 1.0;
    for (const int m : multiplicity)
        for (int f = 2; f <= m; ++f)
            factorials *= f;
    return 2.0 * area * factorials / 120.0;
}

} // namespace

ShellTria3DrillingPlate::ShellTria3DrillingPlate(const ShellTria3DrillingGeometry& geometry, const ShellSection& section,
                               const std::array<double, 3>& thickness)
    : m_geometry(geometry), m_section(section), m_thickness(thickness)
{
    m_centreThickness = (thickness[0] + thickness[1] + thickness[2]) / 3.0;
    if (section.hasMembrane)
        m_membrane.emplace(geometry, section.membraneMatrix, m_centreThickness, section.membraneShearModulus);
    if (section.hasBending)
        setEdgeShear(section.bendingStiffness(m_centreThickness));
}

void ShellTria3DrillingPlate::setEdgeShear(const base::Mat3& d)
{
    const TriangleEdges edges = edgesOf(m_geometry);
    // C (3 × 9): the tangential shear at the middle of edge e, γ_e = (w_b − w_a)/L_e − ½ u_e·(β_a + β_b), β = (−θy, θx)
    // (the edge shear of ShellQuad4DrillingBending).
    base::MatN<3, 9> tangential;
    for (std::size_t e = 0; e < NODES; ++e)
    {
        const std::size_t b = (e + 1) % NODES;
        for (const std::size_t node : {e, b})
        {
            tangential(e, 3 * node + 1) = -0.5 * edges.uy[e];
            tangential(e, 3 * node + 2) = 0.5 * edges.ux[e];
        }
        tangential(e, 3 * e) = -1.0 / edges.length[e];
        tangential(e, 3 * b) = 1.0 / edges.length[e];
    }
    // The frame of NASTRAN-95's t3setd: x along the median from the corner opposite the shortest edge to that edge's
    // middle. Of edges of one length (TRIA3_EDGE_TIE_TOLERANCE: an isosceles triangle), the one whose middle comes
    // first in the basic system (x, then y, then z), not the first in the card's order: the stiffness then does not
    // depend on the order of the grid points (t3setd takes the first of its loop, so that the bending of a triangle of
    // a gear bay model moved by 3 % with a cyclic order of its nodes).
    double least = edges.length[0];
    for (std::size_t e = 1; e < NODES; ++e)
        least = std::min(least, edges.length[e]);
    std::size_t shortest = NODES;
    for (std::size_t e = 0; e < NODES; ++e)
    {
        if (edges.length[e] > least * (1.0 + TRIA_EDGE_TIE_TOLERANCE))
            continue;
        if (shortest == NODES)
        {
            shortest = e;
            continue;
        }
        // The edges' middles doubled (a + b, the same sum whichever end comes first).
        const base::Vec3 middle = m_geometry.positions[e] + m_geometry.positions[(e + 1) % NODES];
        const base::Vec3 chosen = m_geometry.positions[shortest] + m_geometry.positions[(shortest + 1) % NODES];
        if (std::lexicographical_compare(middle.components.begin(), middle.components.end(), chosen.components.begin(),
                                         chosen.components.end()))
            shortest = e;
    }
    const std::size_t opposite = (shortest + 2) % NODES;
    const std::size_t next = (shortest + 1) % NODES;
    const double medianX = 0.5 * (m_geometry.x[shortest] + m_geometry.x[next]) - m_geometry.x[opposite];
    const double medianY = 0.5 * (m_geometry.y[shortest] + m_geometry.y[next]) - m_geometry.y[opposite];
    const double lengthX = std::hypot(medianX, medianY);
    const double c = medianX / lengthX;
    const double s = medianY / lengthX;
    const double lengthY = 2.0 * m_geometry.area / lengthX;
    // M: columns (sin_e, cos_e, 1) of the edge directions in that frame (t3gemd.f); g = M⁻ᵀ γ.
    base::Mat3 m;
    for (std::size_t e = 0; e < NODES; ++e)
    {
        m(0, e) = -s * edges.ux[e] + c * edges.uy[e];
        m(1, e) = c * edges.ux[e] + s * edges.uy[e];
        m(2, e) = 1.0;
    }
    const base::MatN<3, 9> centre = inverted(m).transposed() * tangential;
    // D'11, D'22: the bending stiffness along the frame's axes, vᵀ D v with v = (c², s², 2cs) and (s², c², −2cs) on
    // (κx, κy, 2κxy).
    const base::Vec3 alongX{c * c, s * s, 2.0 * c * s};
    const base::Vec3 alongY{s * s, c * c, -2.0 * c * s};
    const double dX = base::dot(alongX, d * alongX);
    const double dY = base::dot(alongY, d * alongY);
    // tria3d.f: the residual bending compliance in series with the transverse shear compliance of MID3 (zero when MID3
    // is blank or the direction rigid), its element-frame tensor 1/(t κ G) carried into the median frame, C' = Rᵀ C R
    // with R the rotation by the median's angle (c, s).
    const std::array<double, 3> perThickness = m_section.transverseShearCompliance();
    const double shearXX = perThickness[0] / m_centreThickness;
    const double shearYY = perThickness[1] / m_centreThickness;
    const double shearXY = perThickness[2] / m_centreThickness;
    const double complianceXX = c * c * shearXX + 2.0 * c * s * shearXY + s * s * shearYY;
    const double complianceYY = s * s * shearXX - 2.0 * c * s * shearXY + c * c * shearYY;
    const double complianceXY = c * s * (shearYY - shearXX) + (c * c - s * s) * shearXY;
    base::Mat3 z;
    z(0, 0) = complianceYY + lengthY * lengthY / (TRIA_BENDING_COMPLIANCE_FACTOR * dY);
    z(1, 1) = complianceXX + lengthX * lengthX / (TRIA_BENDING_COMPLIANCE_FACTOR * dX);
    z(0, 1) = complianceXY;
    z(1, 0) = complianceXY;
    z(2, 2) = 0.5 * (complianceXX + complianceYY) + m_geometry.area / (2.0 * std::sqrt(dX * dY));
    const base::Mat3 rigidity = inverted(z);
    m_edgeShear = m_geometry.area * base::congruence(rigidity, centre);
    m_edgeCentre = centre;
    m_edgeRigidity = rigidity;
    m_shearCompliance(0, 0) = complianceYY;
    m_shearCompliance(1, 1) = complianceXX;
    m_shearCompliance(0, 1) = complianceXY;
    m_shearCompliance(1, 0) = complianceXY;
    m_shearCompliance(2, 2) = 0.5 * (complianceXX + complianceYY);
    m_bendingCompliance(0, 0) = lengthY * lengthY / (TRIA_BENDING_COMPLIANCE_FACTOR * dY);
    m_bendingCompliance(1, 1) = lengthX * lengthX / (TRIA_BENDING_COMPLIANCE_FACTOR * dX);
    m_bendingCompliance(2, 2) = m_geometry.area / (2.0 * std::sqrt(dX * dY));
    m_medianCos = c;
    m_medianSin = s;
    m_medianLengthX = lengthX;
    m_medianLengthY = lengthY;
    m_bendingAlongX = dX;
    m_bendingAlongY = dY;
    // (Q_yz, Q_xz) of the median frame, then Qx = c Q_xz − s Q_yz, Qy = s Q_xz + c Q_yz.
    const base::MatN<3, 9> forces = rigidity * centre;
    for (std::size_t j = 0; j < 9; ++j)
    {
        m_shearForce(0, j) = c * forces(1, j) - s * forces(0, j);
        m_shearForce(1, j) = s * forces(1, j) + c * forces(0, j);
    }
}

base::MatN<3, 6> ShellTria3DrillingPlate::linearStrainMatrix() const
{
    base::MatN<3, 6> b;
    for (std::size_t i = 0; i < NODES; ++i)
    {
        b(0, 2 * i) = m_geometry.dX[i];
        b(1, 2 * i + 1) = m_geometry.dY[i];
        b(2, 2 * i) = m_geometry.dY[i];
        b(2, 2 * i + 1) = m_geometry.dX[i];
    }
    return b;
}

base::MatN<3, 9> ShellTria3DrillingPlate::curvatureMatrix() const
{
    // The curvatures on linear rotations: κx = −∂θy/∂x, κy = ∂θx/∂y, 2κxy = ∂θx/∂x − ∂θy/∂y.
    base::MatN<3, 9> b;
    for (std::size_t i = 0; i < NODES; ++i)
    {
        b(0, 3 * i + 2) = -m_geometry.dX[i];
        b(1, 3 * i + 1) = m_geometry.dY[i];
        b(2, 3 * i + 1) = m_geometry.dX[i];
        b(2, 3 * i + 2) = -m_geometry.dY[i];
    }
    return b;
}

ShellTria3DrillingPlate::SectionRate ShellTria3DrillingPlate::sectionRate(const ShellSectionRate& rate) const
{
    SectionRate out;
    const double t = m_centreThickness;
    if (m_membrane && rate.movesMembrane)
    {
        // K = A_e/3 Σ t (Bᵀ C B + γ hᵀ h), γ = G: linear in (C, G) together.
        const ShellTria3DrillingMembrane membrane(m_geometry, rate.membraneMatrix, t, rate.membraneShearModulus);
        const base::MatN<9>& stiffness = membrane.stiffness();
        for (std::size_t a = 0; a < 9; ++a)
            for (std::size_t b = 0; b < 9; ++b)
                out.stiffness(membraneDof(a), membraneDof(b)) += stiffness(a, b);
    }
    if (m_section.hasBending && (rate.movesBending || rate.movesShear))
    {
        const base::Mat3 stiffnessRate = m_section.bendingRatio * t * t * t / 12.0 * rate.bendingMatrix;
        // dZ (setEdgeShear's compliance) in the median frame (c, s).
        const double c = m_medianCos;
        const double s = m_medianSin;
        const base::Vec3 alongX{c * c, s * s, 2.0 * c * s};
        const base::Vec3 alongY{s * s, c * c, -2.0 * c * s};
        const double dX = m_bendingAlongX;
        const double dY = m_bendingAlongY;
        const double dXRate = base::dot(alongX, stiffnessRate * alongX);
        const double dYRate = base::dot(alongY, stiffnessRate * alongY);
        const std::array<double, 3> perThickness = transverseShearComplianceRate(m_section, rate);
        const double shearXX = perThickness[0] / t;
        const double shearYY = perThickness[1] / t;
        const double shearXY = perThickness[2] / t;
        const double complianceXX = c * c * shearXX + 2.0 * c * s * shearXY + s * s * shearYY;
        const double complianceYY = s * s * shearXX - 2.0 * c * s * shearXY + c * c * shearYY;
        const double complianceXY = c * s * (shearYY - shearXX) + (c * c - s * s) * shearXY;
        const double lx = m_medianLengthX;
        const double ly = m_medianLengthY;
        base::Mat3 z;
        z(0, 0) = complianceYY - ly * ly * dYRate / (TRIA_BENDING_COMPLIANCE_FACTOR * dY * dY);
        z(1, 1) = complianceXX - lx * lx * dXRate / (TRIA_BENDING_COMPLIANCE_FACTOR * dX * dX);
        z(0, 1) = complianceXY;
        z(1, 0) = complianceXY;
        // d(A_e/(2 √(dX dY))) = −A_e (dX' dY + dX dY')/(4 (dX dY)^(3/2)).
        const double product = dX * dY;
        z(2, 2) = 0.5 * (complianceXX + complianceYY) -
                  m_geometry.area * (dXRate * dY + dX * dYRate) / (4.0 * product * std::sqrt(product));
        out.rigidity = -1.0 * (m_edgeRigidity * (z * m_edgeRigidity));
        const base::MatN<9> bending = m_geometry.area * base::congruence(stiffnessRate, curvatureMatrix()) +
                                      m_geometry.area * base::congruence(out.rigidity, m_edgeCentre);
        addOnBendingDofs(out.stiffness, bending);
    }
    return out;
}

ShellPlaceState ShellTria3DrillingPlate::centreStateSectionRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                        const ShellSectionRate& rate,
                                                        const SectionRate& plateRate) const
{
    ShellPlaceState out;
    out.nodeId = state.nodeId;
    const double t = m_centreThickness;
    if (m_section.hasMembrane)
    {
        const std::array<double, 3> force = rate.membraneMatrix * state.membraneStrain;
        for (std::size_t c = 0; c < 3; ++c)
            out.membraneForce[c] = t * force[c];
    }
    if (m_section.hasBending)
    {
        const std::array<double, 3> moment =
            (m_section.bendingRatio * t * t * t / 12.0 * rate.bendingMatrix) * state.curvature;
        for (std::size_t c = 0; c < 3; ++c)
            out.moment[c] = -moment[c];
        std::array<double, 9> bendingDofs{};
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                bendingDofs[3 * i + d] = q[local(i, 2 + d)];
        // (Q_yz, Q_xz) of the median frame, then Qx = c Q_xz − s Q_yz, Qy = s Q_xz + c Q_yz (setEdgeShear).
        const std::array<double, 3> forces = plateRate.rigidity * (m_edgeCentre * bendingDofs);
        out.shearForce = {m_medianCos * forces[1] - m_medianSin * forces[0],
                          m_medianSin * forces[1] + m_medianCos * forces[0]};
    }
    return out;
}

base::MatN<18> ShellTria3DrillingPlate::stiffness() const
{
    base::MatN<18> k;
    if (m_membrane)
    {
        const base::MatN<9>& membrane = m_membrane->stiffness();
        for (std::size_t a = 0; a < 9; ++a)
            for (std::size_t b = 0; b < 9; ++b)
                k(membraneDof(a), membraneDof(b)) += membrane(a, b);
    }
    if (m_section.hasBending)
    {
        const base::MatN<3, 9> b = curvatureMatrix();
        const base::MatN<9> bending =
            m_geometry.area * base::congruence(m_section.bendingStiffness(m_centreThickness), b) + m_edgeShear;
        addOnBendingDofs(k, bending);
    }
    return k;
}

base::MatN<18> ShellTria3DrillingPlate::thicknessRate(double thickness) const
{
    base::MatN<18> rate;
    if (m_membrane)
    {
        const base::MatN<9>& membrane = m_membrane->stiffness();
        for (std::size_t a = 0; a < 9; ++a)
            for (std::size_t b = 0; b < 9; ++b)
                rate(membraneDof(a), membraneDof(b)) += membrane(a, b) / thickness;
    }
    if (m_section.hasBending)
    {
        const base::MatN<3, 9> b = curvatureMatrix();
        base::Mat3 compliance = m_shearCompliance;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                compliance(r, c) += 3.0 * m_bendingCompliance(r, c);
        const base::Mat3 rigidityRate = (1.0 / thickness) * (m_edgeRigidity * (compliance * m_edgeRigidity));
        const base::MatN<9> bending =
            (3.0 * m_geometry.area / thickness) * base::congruence(m_section.bendingStiffness(m_centreThickness), b) +
            m_geometry.area * base::congruence(rigidityRate, m_edgeCentre);
        addOnBendingDofs(rate, bending);
    }
    return rate;
}

base::Mat3 ShellTria3DrillingPlate::rigiditySecondRate(double thickness) const
{
    // Z = Z_s + Z_b with Z_s ∝ 1/T, Z_b ∝ 1/T³: Z' = −Y/T, Y = Z_s + 3·Z_b, Y' = −(Z_s + 9·Z_b)/T; W' = W·Y·W/T and
    // W'' = W·(2·Y·W·Y − (Z_s + 9·Z_b) − Y)·W/T².
    base::Mat3 y = m_shearCompliance;
    base::Mat3 inner;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            y(r, c) += 3.0 * m_bendingCompliance(r, c);
    const base::Mat3 ywy = y * (m_edgeRigidity * y);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            inner(r, c) = 2.0 * ywy(r, c) - 2.0 * m_shearCompliance(r, c) - 12.0 * m_bendingCompliance(r, c);
    return (1.0 / (thickness * thickness)) * (m_edgeRigidity * (inner * m_edgeRigidity));
}

base::MatN<18> ShellTria3DrillingPlate::thicknessSecondRate(double thickness) const
{
    base::MatN<18> rate;
    const double squared = thickness * thickness;
    if (m_section.hasBending)
    {
        const base::MatN<3, 9> b = curvatureMatrix();
        const base::MatN<9> bending =
            (6.0 * m_geometry.area / squared) * base::congruence(m_section.bendingStiffness(m_centreThickness), b) +
            m_geometry.area * base::congruence(rigiditySecondRate(thickness), m_edgeCentre);
        addOnBendingDofs(rate, bending);
    }
    return rate;
}

ShellPlaceState ShellTria3DrillingPlate::centreStateSecondRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                       double thickness) const
{
    std::array<double, 2> shearSecondRates{};
    if (m_section.hasBending)
    {
        std::array<double, 9> bendingDofs{};
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                bendingDofs[3 * i + d] = q[local(i, 2 + d)];
        const std::array<double, 3> forces = rigiditySecondRate(thickness) * (m_edgeCentre * bendingDofs);
        shearSecondRates = {m_medianCos * forces[1] - m_medianSin * forces[0],
                            m_medianSin * forces[1] + m_medianCos * forces[0]};
    }
    return shellResultantSecondRates(state, m_section, IS_COUPLED, shearSecondRates, thickness);
}

base::MatN<18> ShellTria3DrillingPlate::massThicknessSecondRate(MassFormulation formulation, double areaMass,
                                                       double thickness) const
{
    const double area = m_geometry.area;
    base::MatN<18> m;
    if (formulation == MassFormulation::Lumped)
    {
        // The HRZ lumping of massThicknessRate: m_ii = M_ii·s with M_ii, total and Σ affine in T.
        const double nonStructural = m_section.nonStructuralMass + areaMass;
        std::array<double, 3> diagonal{};
        std::array<double, 3> diagonalRate{};
        for (std::size_t i = 0; i < NODES; ++i)
        {
            diagonal[i] = nonStructural * area * 2.0 / 12.0;
            for (std::size_t k = 0; k < NODES; ++k)
            {
                diagonal[i] += m_section.density * m_thickness[k] * cubicIntegral(i, i, k, area);
                diagonalRate[i] += m_section.density * m_thickness[k] / thickness * cubicIntegral(i, i, k, area);
            }
        }
        const double total = (m_section.density * m_centreThickness + nonStructural) * area;
        const double totalRate = m_section.density * m_centreThickness / thickness * area;
        const double diagonalSum = diagonal[0] + diagonal[1] + diagonal[2];
        const double diagonalSumRate = diagonalRate[0] + diagonalRate[1] + diagonalRate[2];
        if (!(diagonalSum > 0.0))
            return m;
        const double scaleRate = (totalRate * diagonalSum - total * diagonalSumRate) / (diagonalSum * diagonalSum);
        const double scaleSecondRate = -2.0 * scaleRate * diagonalSumRate / diagonalSum;
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(i, d)) = 2.0 * diagonalRate[i] * scaleRate + diagonal[i] * scaleSecondRate;
        return m;
    }
    const double inertiaSecondRate =
        m_section.density * m_centreThickness * m_centreThickness * m_centreThickness / (2.0 * thickness * thickness);
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            const double rotary = inertiaSecondRate * area * (i == j ? 2.0 : 1.0) / 12.0;
            m(local(i, 3), local(j, 3)) = rotary;
            m(local(i, 4), local(j, 4)) = rotary;
        }
    return m;
}

ShellPlaceState ShellTria3DrillingPlate::centreStateRates(const ShellPlaceState& state, const std::array<double, 18>& q,
                                                 double thickness) const
{
    std::array<double, 2> shearRates{};
    if (m_section.hasBending)
    {
        std::array<double, 9> bendingDofs{};
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                bendingDofs[3 * i + d] = q[local(i, 2 + d)];
        base::Mat3 compliance = m_shearCompliance;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                compliance(r, c) += 3.0 * m_bendingCompliance(r, c);
        const base::Mat3 rigidityRate = (1.0 / thickness) * (m_edgeRigidity * (compliance * m_edgeRigidity));
        // (Q_yz, Q_xz) of the median frame, then Qx = c Q_xz − s Q_yz, Qy = s Q_xz + c Q_yz (setEdgeShear).
        const std::array<double, 3> forces = rigidityRate * (m_edgeCentre * bendingDofs);
        shearRates = {m_medianCos * forces[1] - m_medianSin * forces[0],
                      m_medianSin * forces[1] + m_medianCos * forces[0]};
    }
    return shellResultantRates(state, m_section, IS_COUPLED, shearRates, thickness);
}

base::MatN<18> ShellTria3DrillingPlate::massThicknessRate(MassFormulation formulation, double areaMass, double thickness) const
{
    // The parts of mass() that follow T (t_k = r_k·T): the rate of ρ·Σ_k t_k·∫ N_i N_j N_k dA is ρ·Σ_k (t_k/T)·∫ ...
    const double area = m_geometry.area;
    const double nonStructural = m_section.nonStructuralMass + areaMass;
    base::MatN<3> translational;
    base::MatN<3> translationalRate;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            double value = nonStructural * area * (i == j ? 2.0 : 1.0) / 12.0;
            double rate = 0.0;
            for (std::size_t k = 0; k < NODES; ++k)
            {
                value += m_section.density * m_thickness[k] * cubicIntegral(i, j, k, area);
                rate += m_section.density * m_thickness[k] / thickness * cubicIntegral(i, j, k, area);
            }
            translational(i, j) = value;
            translationalRate(i, j) = rate;
        }
    base::MatN<18> m;
    if (formulation == MassFormulation::Lumped)
    {
        // m_ii = M_ii·s, s = total/Σ_k M_kk: dm_ii = dM_ii·s + M_ii·ds, ds = (d total·Σ − total·dΣ)/Σ².
        const double total = (m_section.density * m_centreThickness + nonStructural) * area;
        const double totalRate = m_section.density * m_centreThickness / thickness * area;
        const double diagonalSum = translational(0, 0) + translational(1, 1) + translational(2, 2);
        const double diagonalRate = translationalRate(0, 0) + translationalRate(1, 1) + translationalRate(2, 2);
        if (!(diagonalSum > 0.0))
            return m;
        const double scale = total / diagonalSum;
        const double scaleRate = (totalRate * diagonalSum - total * diagonalRate) / (diagonalSum * diagonalSum);
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(i, d)) = translationalRate(i, i) * scale + translational(i, i) * scaleRate;
        return m;
    }
    const double inertiaRate =
        m_section.density * m_centreThickness * m_centreThickness * m_centreThickness / (4.0 * thickness);
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(j, d)) = translationalRate(i, j);
            const double rotary = inertiaRate * area * (i == j ? 2.0 : 1.0) / 12.0;
            m(local(i, 3), local(j, 3)) = rotary;
            m(local(i, 4), local(j, 4)) = rotary;
        }
    return m;
}

base::MatN<18> ShellTria3DrillingPlate::massDensityRate(MassFormulation formulation, double areaMass, double densityRate) const
{
    // mass() differentiated in ρ at fixed thicknesses: ρ·Σ_k t_k·∫ N_i N_j N_k dA moves by dρ·Σ_k t_k·∫ ...
    const double area = m_geometry.area;
    const double nonStructural = m_section.nonStructuralMass + areaMass;
    base::MatN<3> translational;
    base::MatN<3> translationalRate;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            double value = nonStructural * area * (i == j ? 2.0 : 1.0) / 12.0;
            double rate = 0.0;
            for (std::size_t k = 0; k < NODES; ++k)
            {
                value += m_section.density * m_thickness[k] * cubicIntegral(i, j, k, area);
                rate += densityRate * m_thickness[k] * cubicIntegral(i, j, k, area);
            }
            translational(i, j) = value;
            translationalRate(i, j) = rate;
        }
    base::MatN<18> m;
    if (formulation == MassFormulation::Lumped)
    {
        // m_ii = M_ii·s, s = total/Σ_k M_kk, as massThicknessRate.
        const double total = (m_section.density * m_centreThickness + nonStructural) * area;
        const double totalRate = densityRate * m_centreThickness * area;
        const double diagonalSum = translational(0, 0) + translational(1, 1) + translational(2, 2);
        const double diagonalRate = translationalRate(0, 0) + translationalRate(1, 1) + translationalRate(2, 2);
        if (!(diagonalSum > 0.0))
            return m;
        const double scale = total / diagonalSum;
        const double scaleRate = (totalRate * diagonalSum - total * diagonalRate) / (diagonalSum * diagonalSum);
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(i, d)) = translationalRate(i, i) * scale + translational(i, i) * scaleRate;
        return m;
    }
    const double inertiaRate = densityRate * m_centreThickness * m_centreThickness * m_centreThickness / 12.0;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(j, d)) = translationalRate(i, j);
            const double rotary = inertiaRate * area * (i == j ? 2.0 : 1.0) / 12.0;
            m(local(i, 3), local(j, 3)) = rotary;
            m(local(i, 4), local(j, 4)) = rotary;
        }
    return m;
}

std::array<double, 18> ShellTria3DrillingPlate::thermalLoads(const ShellTemperatureChange& temperature) const
{
    std::array<double, 18> loads{};
    if (temperature.isZero())
        return loads;
    const ThermalResultants thermal =
        m_section.thermalResultants(temperature.deltaT, temperature.gradient, m_centreThickness);
    if (m_membrane)
    {
        // f = A_e Bᵀ N_T, B averaged over the three points of the rule (exact for the linear strains).
        for (const std::array<double, 3>& l : DRILLING_RULE)
        {
            const std::array<double, 9> nodal = m_membrane->strainMatrix(l).transposed() * thermal.force;
            for (std::size_t a = 0; a < 9; ++a)
                loads[membraneDof(a)] += m_geometry.area / 3.0 * nodal[a];
        }
    }
    if (m_section.hasBending)
    {
        const std::array<double, 9> bending = curvatureMatrix().transposed() * thermal.moment;
        for (std::size_t b = 0; b < 9; ++b)
            loads[bendingDof(b)] -= m_geometry.area * bending[b];
    }
    return loads;
}

ShellPlaceState ShellTria3DrillingPlate::centreState(const std::array<double, 18>& q,
                                            const ShellTemperatureChange& temperature) const
{
    return centreState(centreOperators(), centreThermal(temperature), q, temperature);
}

ShellTria3DrillingPlate::CentreOperators ShellTria3DrillingPlate::centreOperators() const
{
    CentreOperators operators;
    if (m_membrane)
        operators.membrane = m_membrane->strainMatrix({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
    else
        operators.linearStrain = linearStrainMatrix();
    operators.curvature = curvatureMatrix();
    if (m_section.hasBending)
        operators.bending = m_section.bendingStiffness(m_centreThickness);
    return operators;
}

ThermalResultants ShellTria3DrillingPlate::centreThermal(const ShellTemperatureChange& temperature) const
{
    return m_section.thermalResultants(temperature.deltaT, temperature.gradient, m_centreThickness);
}

// PERFORMANCE: the centre state from the parts formed once (CentreOperators): the recovery slopes evaluate 19 states
// per element, each of which formed the strain matrices and the stiffnesses again; the operations on the
// displacements are those of before, in the same order, so their bits.
// Measured on the recovery of a plate meshed with this element.
ShellPlaceState ShellTria3DrillingPlate::centreState(const CentreOperators& operators, const ThermalResultants& thermal,
                                            const std::array<double, 18>& q,
                                            const ShellTemperatureChange& temperature) const
{
    std::array<double, 9> bendingDofs{};
    for (std::size_t b = 0; b < 9; ++b)
        bendingDofs[b] = q[bendingDof(b)];
    ShellPlaceState state;
    state.thickness = m_centreThickness;
    state.temperature = temperature;
    if (m_membrane)
    {
        std::array<double, 9> membraneDofs{};
        for (std::size_t a = 0; a < 9; ++a)
            membraneDofs[a] = q[membraneDof(a)];
        state.membraneStrain = operators.membrane * membraneDofs;
    }
    else
    {
        // No membrane stiffness: the strains of the linear (u, v) field, for the strains at the fibres.
        std::array<double, 6> inPlaneDofs{};
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 2; ++d)
                inPlaneDofs[2 * i + d] = q[local(i, d)];
        state.membraneStrain = operators.linearStrain * inPlaneDofs;
    }
    state.curvature = operators.curvature * bendingDofs;
    if (m_section.hasMembrane)
    {
        // N = t G1 ε − N_T.
        const std::array<double, 3> force = m_section.membraneMatrix * state.membraneStrain;
        for (std::size_t c = 0; c < 3; ++c)
            state.membraneForce[c] = m_centreThickness * force[c] - thermal.force[c];
    }
    if (m_section.hasBending)
    {
        const std::array<double, 3> moment = operators.bending * state.curvature;
        // The resultant ∫σ·z dz is −(D κ + M_T) with the curvature sign of curvatureMatrix (see ShellRecovery.hpp).
        for (std::size_t c = 0; c < 3; ++c)
            state.moment[c] = -(moment[c] + thermal.moment[c]);
        state.shearForce = m_shearForce * bendingDofs;
    }
    return state;
}

base::MatN<18> ShellTria3DrillingPlate::mass(MassFormulation formulation, double areaMass) const
{
    // M_ij = ∫ m N_i N_j dA, m = ρ Σ_k t_k N_k + NSM, integrated exactly (area coordinates).
    const double area = m_geometry.area;
    const double nonStructural = m_section.nonStructuralMass + areaMass;
    base::MatN<3> translational;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            double value = nonStructural * area * (i == j ? 2.0 : 1.0) / 12.0;
            for (std::size_t k = 0; k < NODES; ++k)
                value += m_section.density * m_thickness[k] * cubicIntegral(i, j, k, area);
            translational(i, j) = value;
        }
    const double total = (m_section.density * m_centreThickness + nonStructural) * area;
    base::MatN<18> m;
    if (formulation == MassFormulation::Lumped)
    {
        // HRZ lumping: the diagonal of the consistent translational mass scaled to the total.
        const double diagonalSum = translational(0, 0) + translational(1, 1) + translational(2, 2);
        // A massless element has a zero diagonal, not 0/0 (ShellQuad4DrillingPlate::mass).
        const double scale = diagonalSum > 0.0 ? total / diagonalSum : 0.0;
        for (std::size_t i = 0; i < NODES; ++i)
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(i, d)) = translational(i, i) * scale;
        return m;
    }
    const double inertia = m_section.density * m_centreThickness * m_centreThickness * m_centreThickness / 12.0;
    for (std::size_t i = 0; i < NODES; ++i)
        for (std::size_t j = 0; j < NODES; ++j)
        {
            for (std::size_t d = 0; d < 3; ++d)
                m(local(i, d), local(j, d)) = translational(i, j);
            const double rotary = inertia * area * (i == j ? 2.0 : 1.0) / 12.0;
            m(local(i, 3), local(j, 3)) = rotary;
            m(local(i, 4), local(j, 4)) = rotary;
        }
    return m;
}

std::array<double, 18> ShellTria3DrillingPlate::pressureLoads(const std::array<double, 3>& cornerPressures,
                                                     const std::optional<base::Vec3>& direction) const
{
    std::array<double, 18> loads{};
    const base::Vec3 along = direction.value_or(base::Vec3{0.0, 0.0, 1.0});
    const double sum = cornerPressures[0] + cornerPressures[1] + cornerPressures[2];
    for (std::size_t i = 0; i < NODES; ++i)
    {
        const double force = m_geometry.area * (cornerPressures[i] + sum) / 12.0;
        for (std::size_t d = 0; d < 3; ++d)
            loads[local(i, d)] = force * along[d];
    }
    return loads;
}

base::MatN<18> ShellTria3DrillingPlate::differentialStiffness(const std::array<double, 3>& membraneForce) const
{
    base::MatN<18> k;
    addMembraneDifferentialStiffness<3, 6>(k, m_geometry.dX, m_geometry.dY, m_geometry.area, membraneForce);
    return k;
}

} // namespace oos::elements
