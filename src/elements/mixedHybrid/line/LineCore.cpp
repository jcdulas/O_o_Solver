#include "oos/elements/mixedHybrid/line/LineCore.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"

#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The step of the central difference of the section axes along a curved edge, as a fraction of the inverse of the
// rate of turn of its tangent and normal (the derivative of an eccentric offset, offsetAndMetric).
constexpr double AXES_DIFFERENCE_STEP = 1e-5;

// The indices of (Γ_x, K_x, K_y, K_z) among the six strains of a section.
constexpr std::array<std::size_t, 4> AXIAL_AND_BENDING{0, 3, 4, 5};

using Rows3 = std::array<std::array<double, 3>, 3>;

// R v: the components of v along the section axes (R has the axes as its rows).
Vec3Of<double> inSectionAxes(const SectionAxes& axes, const Vec3Of<double>& v)
{
    return {dot3(axes[0], v), dot3(axes[1], v), dot3(axes[2], v)};
}

Vec3Of<double> columnOf(const Matrix& rows, std::size_t column)
{
    return {rows(0, column), rows(1, column), rows(2, column)};
}

Vec3Of<double> times(const Rows3& m, const Vec3Of<double>& v)
{
    return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}

// aᵀ M b.
double bilinear(const Vec3Of<double>& a, const Rows3& m, const Vec3Of<double>& b)
{
    Vec3Of<double> row{};
    for (std::size_t j = 0; j < 3; ++j)
        row[j] = a[0] * m[0][j] + a[1] * m[1][j] + a[2] * m[2][j];
    return dot3(row, b);
}

// The inverse of a matrix of order 0 to 3 by its adjugate (Cramer's rule): A⁻¹ = adj(A) / det A, the determinant
// expanded along the first row. Empty when the determinant is zero.
std::optional<Matrix> adjugateInverse(const Matrix& a)
{
    const std::size_t n = a.rows();
    OOS_ASSERT(n <= 3 && a.columns() == n);
    Matrix out(n, n);
    if (n == 1)
    {
        if (a(0, 0) == 0.0)
            return std::nullopt;
        out(0, 0) = 1.0 / a(0, 0);
    }
    else if (n == 2)
    {
        const double det = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
        if (det == 0.0)
            return std::nullopt;
        out(0, 0) = a(1, 1) / det;
        out(0, 1) = -a(0, 1) / det;
        out(1, 0) = -a(1, 0) / det;
        out(1, 1) = a(0, 0) / det;
    }
    else if (n == 3)
    {
        const double minor00 = a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2);
        const double minor01 = a(1, 0) * a(2, 2) - a(2, 0) * a(1, 2);
        const double minor02 = a(1, 0) * a(2, 1) - a(2, 0) * a(1, 1);
        const double det = a(0, 0) * minor00 - a(0, 1) * minor01 + a(0, 2) * minor02;
        if (det == 0.0)
            return std::nullopt;
        out(0, 0) = minor00 / det;
        out(0, 1) = (a(0, 2) * a(2, 1) - a(2, 2) * a(0, 1)) / det;
        out(0, 2) = (a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2)) / det;
        out(1, 0) = -minor01 / det;
        out(1, 1) = (a(0, 0) * a(2, 2) - a(2, 0) * a(0, 2)) / det;
        out(1, 2) = (a(0, 2) * a(1, 0) - a(1, 2) * a(0, 0)) / det;
        out(2, 0) = minor02 / det;
        out(2, 1) = (a(0, 1) * a(2, 0) - a(2, 1) * a(0, 0)) / det;
        out(2, 2) = (a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1)) / det;
    }
    return out;
}

// The offset of the axis from the edge at a point of a line, and the metric of the axis: its length per unit length
// of the edge, μ = |t + do/ds| (an eccentric ring, a stiffener on a curved edge). The offset is the end offsets
// (global, linear along the line) plus the eccentricity along the section axes; the derivative of the latter follows
// the turn of the section axes, taken by a central difference of sectionAxes along the edge.
// Reference: src/tdnns/line.rs::offset_metric
std::variant<std::pair<Vec3Of<double>, double>, std::string> offsetAndMetric(const LineElement& line,
                                                                             const LineSection& section,
                                                                             const LineKinematics& kin,
                                                                             const SectionAxes& axes)
{
    const double lam = kin.fraction;
    const Vec3Of<double> offset = add3(add3(scl3(line.offsets[0], 1.0 - lam), scl3(line.offsets[1], lam)),
                                       add3(scl3(axes[1], line.eccentricity[0]), scl3(axes[2], line.eccentricity[1])));
    Vec3Of<double> offsetRate = scl3(sub3(line.offsets[1], line.offsets[0]), kin.fractionRate);
    const double rate = norm3(kin.tangentRate) + norm3(kin.normalRate);
    if (rate > 0.0 && (line.eccentricity[0] != 0.0 || line.eccentricity[1] != 0.0))
    {
        const double h = AXES_DIFFERENCE_STEP / rate;
        const auto axesAt = [&](double s)
        {
            std::optional<Vec3Of<double>> normal;
            if (kin.normal)
                normal = add3(*kin.normal, scl3(kin.normalRate, s));
            return sectionAxes(line, section, unit3(add3(kin.tangent, scl3(kin.tangentRate, s))), normal);
        };
        auto plus = axesAt(h);
        if (const std::string* problem = std::get_if<std::string>(&plus))
            return *problem;
        auto minus = axesAt(-h);
        if (const std::string* problem = std::get_if<std::string>(&minus))
            return *problem;
        const SectionAxes& forward = std::get<SectionAxes>(plus);
        const SectionAxes& backward = std::get<SectionAxes>(minus);
        for (std::size_t i = 1; i < 3; ++i)
            offsetRate = add3(offsetRate, scl3(sub3(forward[i], backward[i]), 0.5 * line.eccentricity[i - 1] / h));
    }
    return std::pair{offset, norm3(add3(kin.tangent, offsetRate))};
}

// The section law of a line about its edge, on (Γ_x, K_x, K_y, K_z), with the offset `offset` of the axis in section
// axes and the metric μ of the axis: the strains of the axis are (Γ_edge + (K × o) · x, K) / μ and its length is μ
// times that of the edge, so that the law on the strains of the edge is TᵀDT / μ,
//   T = [[1, 0, o_z, −o_y], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]   ((K × o) · x = K_y o_z − K_z o_y);
// and the matrices of the partially mixed form: with TᵀDT / μ = [[a0, b], [bᵀ, D_m]],
//   D⁻¹ (the components of D_m with a stiffness), D⁻¹ bᵀ, Ā = a0 − b D⁻¹ bᵀ.
// Reference: src/tdnns/line.rs::Law
struct SectionLaw
{
    // The law at the axis, on (Γ_x, K_x, K_y, K_z).
    std::array<std::array<double, 4>, 4> axisLaw{};
    Vec3Of<double> offset{};
    double metric = 1.0;
    Vec3Of<double> coupling{};
    Rows3 bendingInverse{};
    Vec3Of<double> inverseCoupling{};
    double membrane = 0.0;
    // The components of the moment that have a stiffness.
    std::array<bool, 3> active{};
    // The shear stiffnesses of the planes (x, y) and (x, z), over μ.
    std::array<double, 2> shearStiffness{};
};

// Reference: src/tdnns/line.rs::Law::new
SectionLaw makeSectionLaw(const LineSectionPoint& point, const Vec3Of<double>& offset, double metric)
{
    SectionLaw law;
    law.offset = offset;
    law.metric = metric;
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 4; ++b)
            law.axisLaw[a][b] = point.stiffness[AXIAL_AND_BENDING[a]][AXIAL_AND_BENDING[b]];
    const std::array<std::array<double, 4>, 4> t{
        {{1.0, 0.0, offset[2], -offset[1]}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}}};
    std::array<std::array<double, 4>, 4> law4{};
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 4; ++b)
            for (std::size_t c = 0; c < 4; ++c)
                for (std::size_t e = 0; e < 4; ++e)
                    law4[a][b] += t[c][a] * law.axisLaw[c][e] * t[e][b] / metric;
    const double a0 = law4[0][0];
    law.coupling = {law4[0][1], law4[0][2], law4[0][3]};
    // Components without stiffness carry no moment: they are left out of the inverse.
    std::vector<std::size_t> keep;
    for (std::size_t c = 0; c < 3; ++c)
    {
        law.active[c] = law4[1 + c][1 + c] > 0.0;
        if (law.active[c])
            keep.push_back(c);
    }
    Matrix kept(keep.size(), keep.size());
    for (std::size_t i = 0; i < keep.size(); ++i)
        for (std::size_t j = 0; j < keep.size(); ++j)
            kept(i, j) = law4[1 + keep[i]][1 + keep[j]];
    const Matrix keptInverse = adjugateInverse(kept).value_or(Matrix(keep.size(), keep.size()));
    for (std::size_t i = 0; i < keep.size(); ++i)
        for (std::size_t j = 0; j < keep.size(); ++j)
            law.bendingInverse[keep[i]][keep[j]] = keptInverse(i, j);
    law.inverseCoupling = times(law.bendingInverse, law.coupling);
    law.membrane = a0 - dot3(law.coupling, law.inverseCoupling);
    law.shearStiffness = {point.stiffness[1][1] / metric, point.stiffness[2][2] / metric};
    return law;
}

// The Legendre polynomials P_l(2λ − 1), l ≤ degree, and their derivatives d/ds = P'_l · 2 dλ/ds.
std::vector<std::pair<double, double>> momentPolynomials(std::size_t degree, double lam, double fractionRate)
{
    std::vector<std::pair<double, double>> out;
    out.reserve(degree + 1);
    for (std::size_t l = 0; l <= degree; ++l)
    {
        const auto [v, dv] = legendreWithDerivative(l, 2.0 * lam - 1.0);
        out.emplace_back(v, dv * 2.0 * fractionRate);
    }
    return out;
}

// The moment coefficients that are kept: those of the components with a stiffness somewhere along the line.
std::vector<std::size_t> keptMomentRows(std::size_t momentCount, const std::array<bool, 3>& active)
{
    std::vector<std::size_t> rows;
    for (std::size_t r = 0; r < momentCount; ++r)
        if (active[r % 3])
            rows.push_back(r);
    return rows;
}

// The index of the point nearest the middle of the line (the first of equals).
std::size_t middlePoint(std::span<const WeightedLineKinematics> points)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
        if (std::fabs(points[i].kinematics.fraction - 0.5) < std::fabs(points[best].kinematics.fraction - 0.5))
            best = i;
    return best;
}

const char* const SINGULAR_COMPLIANCE = "line element: singular section compliance";

} // namespace

Vector LineData::local(std::span<const double> modelUnknowns) const
{
    Vector x(dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
        x[i] = signs[i] * modelUnknowns[dofs[i]];
    return x;
}

Vector LineData::thermalMoments(std::span<const double> freeStrains) const
{
    Vector g0(moments.rows(), 0.0);
    for (std::size_t q = 0; q < axial.size() && q < freeStrains.size(); ++q)
    {
        const double factor = axial[q].weight * freeStrains[q];
        for (std::size_t i = 0; i < g0.size(); ++i)
            g0[i] += axial[q].momentRow[i] * factor;
    }
    return g0;
}

Vector LineData::thermalLoad(std::span<const double> freeStrains) const
{
    Vector f(dofs.size(), 0.0);
    for (std::size_t q = 0; q < axial.size() && q < freeStrains.size(); ++q)
    {
        const LineAxialPoint& p = axial[q];
        const double factor = p.axialStiffness * p.weight * freeStrains[q];
        for (std::size_t i = 0; i < f.size(); ++i)
            f[i] += p.strainRow[i] * factor;
    }
    const Vector fromMoments = transposeTimes(moments, thermalMoments(freeStrains));
    for (std::size_t i = 0; i < f.size(); ++i)
        f[i] += fromMoments[i];
    return f;
}

std::vector<double> LineData::edgeFreeStrains(std::span<const double> freeStrains,
                                              std::span<const LineFreeCurvature> freeCurvatures) const
{
    std::vector<double> edge(axial.size(), 0.0);
    for (std::size_t q = 0; q < axial.size(); ++q)
    {
        const std::array<double, 2>& ofCurvature = axial[q].freeCurvatureAxial;
        const double given = q < freeStrains.size() ? freeStrains[q] : 0.0;
        edge[q] = given + ofCurvature[0] * freeCurvatures[q].y + ofCurvature[1] * freeCurvatures[q].z;
    }
    return edge;
}

Vector LineData::thermalMoments(std::span<const double> freeStrains,
                                std::span<const LineFreeCurvature> freeCurvatures) const
{
    // g0 = Σ w M_sᵀ (D⁻¹ Bᵀ Γ0 + K0): the term of the free axial strain of the edge, then that of the curvatures.
    Vector g0 = thermalMoments(edgeFreeStrains(freeStrains, freeCurvatures));
    for (std::size_t q = 0; q < axial.size(); ++q)
    {
        const LineAxialPoint& p = axial[q];
        for (std::size_t i = 0; i < g0.size(); ++i)
            g0[i] += p.weight * (p.freeCurvatureRows[0][i] * freeCurvatures[q].y +
                                 p.freeCurvatureRows[1][i] * freeCurvatures[q].z);
    }
    return g0;
}

Vector LineData::thermalLoad(std::span<const double> freeStrains,
                             std::span<const LineFreeCurvature> freeCurvatures) const
{
    const std::vector<double> edge = edgeFreeStrains(freeStrains, freeCurvatures);
    Vector f(dofs.size(), 0.0);
    for (std::size_t q = 0; q < axial.size(); ++q)
    {
        const LineAxialPoint& p = axial[q];
        const double factor = p.axialStiffness * p.weight * edge[q];
        for (std::size_t i = 0; i < f.size(); ++i)
            f[i] += p.strainRow[i] * factor;
    }
    const Vector fromMoments = transposeTimes(moments, thermalMoments(freeStrains, freeCurvatures));
    for (std::size_t i = 0; i < f.size(); ++i)
        f[i] += fromMoments[i];
    return f;
}

std::variant<LineData, std::string> lineCore(const LineSource& source, std::span<const WeightedLineKinematics> points,
                                             std::size_t kinematicsColumns, std::size_t momentDegree, bool shear,
                                             const LineElement& line, const LineSection& section,
                                             std::span<const std::size_t> kinematicsDofs,
                                             std::span<const double> kinematicsSigns,
                                             const std::array<std::size_t, 2>& rotationDofs, bool withMass)
{
    OOS_ASSERT(!points.empty());
    const std::size_t nxk = kinematicsColumns;
    const std::size_t nl = nxk + 6;
    std::vector<SectionAxes> frames;
    frames.reserve(points.size());
    for (const WeightedLineKinematics& p : points)
    {
        auto axes = sectionAxes(line, section, p.kinematics.tangent, p.kinematics.normal);
        if (const std::string* problem = std::get_if<std::string>(&axes))
            return *problem;
        frames.push_back(std::get<SectionAxes>(axes));
    }
    // The components of the moments: the section axes at the middle of the line.
    const SectionAxes basis = frames[middlePoint(points)];
    const std::size_t pm = momentDegree;
    const std::size_t nm = 3 * (pm + 1);
    Matrix pairing(nm, nl);      // G: −∫ dM/ds · θ and the end terms
    Matrix compliance(nm, nm);   // M_σ
    Matrix couplingRows(nm, nl); // C: ∫ M_sᵀ D⁻¹ Bᵀ Γ_x
    Matrix axialStiffness(nl, nl);
    std::optional<Matrix> mass;
    if (withMass)
        mass = Matrix(nl, nl);
    Matrix inertia(nl, 3);
    std::vector<LineAxialPoint> axial;
    axial.reserve(points.size());
    std::vector<Vec3Of<double>> axisPoints;
    axisPoints.reserve(points.size());
    std::array<bool, 3> active{};
    const bool shearOn = shear && section.bending;

    for (std::size_t q = 0; q < points.size(); ++q)
    {
        const LineKinematics& kin = points[q].kinematics;
        const double w = points[q].weight;
        const SectionAxes& axes = frames[q];
        const Vec3Of<double>& tl = axes[0];
        const double lam = kin.fraction;
        // The offset of the axis: element offsets (global, linear) and eccentricity (section axes).
        auto offsetMetric = offsetAndMetric(line, section, kin, axes);
        if (const std::string* problem = std::get_if<std::string>(&offsetMetric))
            return *problem;
        const auto [offset, mu] = std::get<std::pair<Vec3Of<double>, double>>(offsetMetric);
        const LineSectionPoint sp = section.at(lam);
        const SectionLaw law = makeSectionLaw(sp, inSectionAxes(axes, offset), mu);
        for (std::size_t c = 0; c < 3; ++c)
            active[c] = active[c] || law.active[c];

        // The row of the axial strain t · du/ds.
        Vector gx(nl, 0.0);
        for (std::size_t col = 0; col < nxk; ++col)
            gx[col] = tl[0] * kin.displacementRate(0, col) + tl[1] * kin.displacementRate(1, col) +
                      tl[2] * kin.displacementRate(2, col);
        const double axialFactor = law.membrane * w;
        for (std::size_t i = 0; i < nl; ++i)
            for (std::size_t j = 0; j < nl; ++j)
                axialStiffness(i, j) += gx[i] * gx[j] * axialFactor;

        // The moment basis E_c P_l(2λ − 1): its value and d/ds, in section axes (M_s), and the shear force of its
        // derivative, t × dM/ds (V_s).
        const std::vector<std::pair<double, double>> pl = momentPolynomials(pm, lam, kin.fractionRate);
        const auto momentOf = [&](std::size_t c, std::size_t l)
        { return scl3(inSectionAxes(axes, basis[c]), pl[l].first); };
        const auto shearOf = [&](std::size_t c, std::size_t l)
        { return scl3(inSectionAxes(axes, cross3(tl, basis[c])), pl[l].second); };
        for (std::size_t c = 0; c < 3; ++c)
        {
            for (std::size_t l = 0; l <= pm; ++l)
            {
                const std::size_t row = 3 * l + c;
                const double dv = pl[l].second;
                // −dM/ds · θ
                for (std::size_t col = 0; col < nxk; ++col)
                {
                    const double e = basis[c][0] * kin.rotation(0, col) + basis[c][1] * kin.rotation(1, col) +
                                     basis[c][2] * kin.rotation(2, col);
                    pairing(row, col) -= w * dv * e;
                }
                const Vec3Of<double> ms = momentOf(c, l);
                // The coupling M_sᵀ D⁻¹ Bᵀ Γ_x.
                const double cc = dot3(ms, law.inverseCoupling);
                for (std::size_t col = 0; col < nxk; ++col)
                    couplingRows(row, col) += w * cc * gx[col];
                const Vec3Of<double> qs = shearOf(c, l);
                for (std::size_t c2 = 0; c2 < 3; ++c2)
                {
                    for (std::size_t l2 = 0; l2 <= pm; ++l2)
                    {
                        double v = bilinear(ms, law.bendingInverse, momentOf(c2, l2));
                        if (shearOn)
                        {
                            // The complementary energy of the shear force, V_s · C_s V_s.
                            const Vec3Of<double> qs2 = shearOf(c2, l2);
                            for (std::size_t i = 0; i < 2; ++i)
                                if (law.shearStiffness[i] > 0.0)
                                    v += qs[1 + i] * qs2[1 + i] / law.shearStiffness[i];
                        }
                        compliance(row, 3 * l2 + c2) += w * v;
                    }
                }
            }
        }

        // The translation of the axis, u + θ × o.
        Matrix axisDisplacement(3, nl);
        for (std::size_t col = 0; col < nxk; ++col)
        {
            const Vec3Of<double> x = cross3(columnOf(kin.rotation, col), offset);
            for (std::size_t r = 0; r < 3; ++r)
                axisDisplacement(r, col) = kin.displacement(r, col) + x[r];
        }
        inertia += axisDisplacement.transposed() * (sp.massPerLength * mu * w);
        if (mass)
        {
            *mass += transposeTimes(axisDisplacement, axisDisplacement) * (sp.massPerLength * mu * w);
            // The rotary inertia of the section, in its axes.
            Matrix rotationInAxes(3, nl);
            for (std::size_t col = 0; col < nxk; ++col)
            {
                const Vec3Of<double> v = inSectionAxes(axes, columnOf(kin.rotation, col));
                for (std::size_t r = 0; r < 3; ++r)
                    rotationInAxes(r, col) = v[r];
            }
            const auto [jx, jy, jz, jyz] = sp.inertia;
            Matrix rotary(3, 3);
            rotary(0, 0) = jx;
            rotary(1, 1) = jy;
            rotary(1, 2) = -jyz;
            rotary(2, 1) = -jyz;
            rotary(2, 2) = jz;
            *mass += transposeTimes(rotationInAxes, rotary) * rotationInAxes * (mu * w);
        }

        // The geometric stiffness: N (du⊥ · du⊥), du⊥ the part of du/ds across t.
        Matrix transverse(3, nl);
        for (std::size_t col = 0; col < nxk; ++col)
        {
            const Vec3Of<double> d = columnOf(kin.displacementRate, col);
            const Vec3Of<double> across = sub3(d, scl3(tl, dot3(d, tl)));
            for (std::size_t r = 0; r < 3; ++r)
                transverse(r, col) = across[r];
        }
        // The moment part of the axial force: (D⁻¹ Bᵀ) · M_s on the moment coefficients.
        Vector momentRow(nm, 0.0);
        // The rows of a unit free curvature K_y and K_z of the axis: μ M_s · e_y and μ M_s · e_z.
        std::array<Vector, 2> freeCurvatureRows{Vector(nm, 0.0), Vector(nm, 0.0)};
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t l = 0; l <= pm; ++l)
            {
                const Vec3Of<double> ms = momentOf(c, l);
                momentRow[3 * l + c] = dot3(ms, law.inverseCoupling);
                freeCurvatureRows[0][3 * l + c] = mu * ms[1];
                freeCurvatureRows[1][3 * l + c] = mu * ms[2];
            }
        // The axial strain of the edge in a unit free curvature of the axis, at the offset o from it.
        const std::array<double, 2> freeCurvatureAxial{-mu * law.offset[2], mu * law.offset[1]};
        axial.push_back({std::move(gx), law.membrane, std::move(transverse), w, std::move(momentRow), lam,
                         std::move(freeCurvatureRows), freeCurvatureAxial});
        axisPoints.push_back(add3(kin.point, offset));
    }

    // The end terms + M(B) · Λ_B − M(A) · Λ_A, with P_l(1) = 1 and P_l(−1) = (−1)^l.
    for (std::size_t c = 0; c < 3; ++c)
    {
        for (std::size_t l = 0; l <= pm; ++l)
        {
            const std::size_t row = 3 * l + c;
            const double atA = l % 2 == 0 ? 1.0 : -1.0;
            for (std::size_t r = 0; r < 3; ++r)
            {
                pairing(row, nxk + 3 + r) += basis[c][r];
                pairing(row, nxk + r) -= atA * basis[c][r];
            }
        }
    }

    // The components of the moment without stiffness are dropped.
    const std::vector<std::size_t> rows = keptMomentRows(nm, active);
    Matrix moments(nm, nl);
    Matrix complianceInverse(nm, nm);
    Matrix stiffness = axialStiffness;
    if (!rows.empty())
    {
        const std::size_t nk = rows.size();
        Matrix keptCompliance(nk, nk);
        Matrix coupled(nk, nl); // G + C on the kept rows
        for (std::size_t i = 0; i < nk; ++i)
        {
            for (std::size_t j = 0; j < nk; ++j)
                keptCompliance(i, j) = compliance(rows[i], rows[j]);
            for (std::size_t j = 0; j < nl; ++j)
                coupled(i, j) = pairing(rows[i], j) + couplingRows(rows[i], j);
        }
        const std::optional<Cholesky> factor = Cholesky::factor(keptCompliance);
        if (!factor)
            return std::string(SINGULAR_COMPLIANCE);
        const Matrix solution = factor->solve(coupled);
        const Matrix inverse = factor->inverse();
        for (std::size_t i = 0; i < nk; ++i)
        {
            moments.setRow(rows[i], solution.rowSpan(i));
            for (std::size_t j = 0; j < nk; ++j)
                complianceInverse(rows[i], rows[j]) = inverse(i, j);
        }
        // K = K_axial + Yᵀ Y, Y = L⁻¹ (G + C).
        Matrix y = coupled;
        solveLower(factor->lower(), y);
        stiffness += transposeTimes(y, y);
    }

    LineData data;
    data.source = source;
    data.dofs.assign(kinematicsDofs.begin(), kinematicsDofs.end());
    data.signs.assign(kinematicsSigns.begin(), kinematicsSigns.end());
    for (const std::size_t first : rotationDofs)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            data.dofs.push_back(first + c);
            data.signs.push_back(1.0);
        }
    }
    data.stiffness = symmetricPart(stiffness);
    if (mass)
        data.mass = symmetricPart(*mass);
    data.axial = std::move(axial);
    data.moments = std::move(moments);
    data.complianceInverse = std::move(complianceInverse);
    data.basis = basis;
    data.momentDegree = pm;
    data.shear = shear;
    data.inertia = std::move(inertia);
    data.axisPoints = std::move(axisPoints);
    return data;
}

std::variant<LineState, std::string> lineState(const LineData& data, const LineKinematics& kinematics,
                                               const LineElement& line, const LineSection& section,
                                               std::span<const double> x, double temperature, double freeStrain,
                                               std::span<const double> freeMoments,
                                               const LineFreeCurvature& freeCurvature)
{
    const LineKinematics& kin = kinematics;
    const std::size_t nxk = data.dofs.size() - 6;
    auto axesOrProblem = sectionAxes(line, section, kin.tangent, kin.normal);
    if (const std::string* problem = std::get_if<std::string>(&axesOrProblem))
        return *problem;
    const SectionAxes axes = std::get<SectionAxes>(axesOrProblem);
    const Vec3Of<double>& tl = axes[0];
    const double lam = kin.fraction;
    auto offsetMetric = offsetAndMetric(line, section, kin, axes);
    if (const std::string* problem = std::get_if<std::string>(&offsetMetric))
        return *problem;
    const Vec3Of<double> offset = std::get<std::pair<Vec3Of<double>, double>>(offsetMetric).first;
    const double metric = std::get<std::pair<Vec3Of<double>, double>>(offsetMetric).second;
    const LineSectionPoint sp = section.at(lam);
    const SectionLaw law = makeSectionLaw(sp, inSectionAxes(axes, offset), metric);

    // The axial strain of the edge, t · du/ds.
    double gamma = 0.0;
    for (std::size_t col = 0; col < nxk; ++col)
        gamma += (tl[0] * kin.displacementRate(0, col) + tl[1] * kin.displacementRate(1, col) +
                  tl[2] * kin.displacementRate(2, col)) *
                 x[col];
    // The moment coefficients, less the part of the free strain.
    Vector coefficients = data.moments * x;
    if (!freeMoments.empty())
    {
        const Vector thermal = data.complianceInverse * freeMoments;
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            coefficients[i] -= thermal[i];
    }
    Vec3Of<double> m{};
    Vec3Of<double> dm{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        for (std::size_t l = 0; l <= data.momentDegree; ++l)
        {
            const auto [v, dv] = legendreWithDerivative(l, 2.0 * lam - 1.0);
            const double coefficient = coefficients[3 * l + c];
            m = add3(m, scl3(data.basis[c], coefficient * v));
            dm = add3(dm, scl3(data.basis[c], coefficient * dv * 2.0 * kin.fractionRate));
        }
    }
    const Vec3Of<double>& os = law.offset;
    const double mu = law.metric;
    // The free strains of the edge: the free axial strain given, and under free curvatures K0a of the axis the
    // curvatures μ K0a and the axial strain μ (−o_z K0a_y + o_y K0a_z) of the edge, at −o from the axis. The
    // sums are left out without free curvatures: the same bits as before them.
    const bool hasFreeCurvature = !freeCurvature.isZero();
    const Vec3Of<double> freeKappaEdge{0.0, mu * freeCurvature.y, mu * freeCurvature.z};
    const double freeEdge =
        hasFreeCurvature ? freeStrain - freeKappaEdge[1] * os[2] + freeKappaEdge[2] * os[1] : freeStrain;
    // The elastic curvatures of the edge, D⁻¹ (M_s − bᵀ (Γ_x − Γ0)), and its curvatures, with the free ones.
    const Vec3Of<double> ms = inSectionAxes(axes, m);
    const double elasticAxial = gamma - freeEdge;
    const Vec3Of<double> kappaEdge = times(law.bendingInverse, sub3(ms, scl3(law.coupling, elasticAxial)));
    const Vec3Of<double> totalKappaEdge = hasFreeCurvature ? add3(kappaEdge, freeKappaEdge) : kappaEdge;
    // The strains of the axis: those seen from the edge over the metric.
    const double gammaAxis = (gamma + totalKappaEdge[1] * os[2] - totalKappaEdge[2] * os[1]) / mu;
    const Vec3Of<double> kappa{kappaEdge[0] / mu, kappaEdge[1] / mu, kappaEdge[2] / mu};
    const Vec3Of<double> totalKappa{totalKappaEdge[0] / mu, totalKappaEdge[1] / mu, totalKappaEdge[2] / mu};
    const double freeAxis = freeStrain / mu;
    const std::array<double, 4> strains4{gammaAxis - freeAxis, kappa[0], kappa[1], kappa[2]};
    std::array<double, 4> forces4{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        double sum = 0.0;
        for (std::size_t b = 0; b < 4; ++b)
            sum += law.axisLaw[a][b] * strains4[b];
        forces4[a] = sum;
    }
    // The shear force V = t × dM/ds, in section axes, and its strain where the section has a shear flexibility.
    const Vec3Of<double> vs = inSectionAxes(axes, cross3(tl, dm));
    const auto shearStrain = [&](std::size_t i)
    {
        if (data.shear && section.bending && law.shearStiffness[i] > 0.0)
            return vs[1 + i] / (law.shearStiffness[i] * mu);
        return 0.0;
    };
    LineState state;
    state.point = add3(kin.point, offset);
    state.axes = axes;
    state.forces = {forces4[0], vs[1], vs[2], forces4[1], forces4[2], forces4[3]};
    state.strains = {gammaAxis, shearStrain(0), shearStrain(1), totalKappa[0], totalKappa[1], totalKappa[2]};
    state.temperature = temperature;
    state.freeStrain = freeAxis;
    state.freeCurvature = freeCurvature;
    return state;
}

} // namespace oos::elements::mixedHybrid
