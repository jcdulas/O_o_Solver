#include "oos/elements/mixedHybrid/line/LineCore.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The step of the central difference of the section axes along a curved edge, as a fraction of the inverse of the
// rate of turn of its tangent and normal (the derivative of an eccentric offset, offsetAndMetric).
constexpr double AXES_DIFFERENCE_STEP = 1e-5;

// R v: the components of v along the section axes (R has the axes as its rows).
Vec3Of<double> inSectionAxes(const SectionAxes& axes, const Vec3Of<double>& v)
{
    return {dot3(axes[0], v), dot3(axes[1], v), dot3(axes[2], v)};
}

Vec3Of<double> columnOf(const Matrix& rows, std::size_t column)
{
    return {rows(0, column), rows(1, column), rows(2, column)};
}

// The offset of the axis from the edge at a point of a line, and the metric of the axis: its length per unit length
// of the edge, μ = |t + do/ds| (an eccentric ring, a stiffener on a curved edge). The offset is the end offsets
// (global, linear along the line) plus the eccentricity along the section axes; the derivative of the latter follows
// the turn of the section axes, taken by a central difference of sectionAxes along the edge.
// Reference: src/tdnns/line.rs::offset_metric
std::variant<std::pair<Vec3Of<double>, Vec3Of<double>>, std::string> offsetAndMetric(const LineElement& line,
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
    return std::pair{offset, add3(kin.tangent, offsetRate)};
}

// Forces and moments at the physical section, expressed in its axes. The equilibrium unknown is the
// moment about the mesh edge. Transport its wrench before evaluating complementary energy.
struct Wrench
{
    Vec3Of<double> force{}, moment{};
};

Wrench addAxial(const Wrench& w, const Wrench& axial, double n)
{
    return {add3(w.force, scl3(axial.force, n)), add3(w.moment, scl3(axial.moment, n))};
}

struct SectionLaw
{
    SectionAxes axes{};
    Vec3Of<double> offset{};
    double metric = 0.0, membrane = 0.0;
    Wrench axial{};
    std::array<double, 6> inverse{};
    double bendingCross = 0.0;
    std::array<double, 2> neutral{};
    std::array<bool, 3> active{};

    Wrench wrench(const Vec3Of<double>& force, const Vec3Of<double>& moment) const
    {
        return {inSectionAxes(axes, force), inSectionAxes(axes, sub3(moment, cross3(offset, force)))};
    }

    Wrench strain(const Wrench& w) const
    {
        const double my = w.moment[1] - neutral[1] * w.force[0];
        const double mz = w.moment[2] + neutral[0] * w.force[0];
        const double ky = inverse[4] * my + bendingCross * mz;
        const double kz = bendingCross * my + inverse[5] * mz;
        return {{inverse[0] * w.force[0] - neutral[1] * ky + neutral[0] * kz, inverse[1] * w.force[1],
                 inverse[2] * w.force[2]},
                {inverse[3] * w.moment[0], ky, kz}};
    }

    double work(const Wrench& a, const Wrench& strainB) const
    {
        return metric * (dot3(a.force, strainB.force) + dot3(a.moment, strainB.moment));
    }
    double coupling(const Wrench& w) const { return -membrane * work(axial, strain(w)); }
};

std::variant<SectionLaw, std::string> sectionLaw(const LineElement& line, const LineSection& section,
                                                 const LineKinematics& kin, const SectionAxes& edgeAxes, bool shear)
{
    const auto geometry = offsetAndMetric(line, section, kin, edgeAxes);
    if (const auto* error = std::get_if<std::string>(&geometry))
        return *error;
    const auto& [offset, derivative] = std::get<std::pair<Vec3Of<double>, Vec3Of<double>>>(geometry);
    SectionLaw law;
    law.metric = norm3(derivative);
    if (!(law.metric > 0.0))
        return std::string("the offset beam axis has zero tangent");
    const auto frame = sectionAxes(line, section, scl3(derivative, 1.0 / law.metric), kin.normal);
    if (const auto* error = std::get_if<std::string>(&frame))
        return *error;
    law.axes = std::get<SectionAxes>(frame);
    law.offset = offset;
    law.axial = law.wrench(kin.tangent, {});
    const auto point = section.at(kin.fraction);
    law.neutral = point.neutralAxis;
    const auto& k = point.stiffness;
    for (std::size_t i = 0; i < 4; ++i)
        if (k[i][i] > 0.0 && (i == 0 || i == 3 || shear))
            law.inverse[i] = 1.0 / k[i][i];
    const auto [yy, zz, yz] = point.centroidalBending;
    const double det = yy * zz - yz * yz;
    if (det > 0.0)
    {
        law.inverse[4] = zz / det;
        law.inverse[5] = yy / det;
        law.bendingCross = -yz / det;
    }
    else
    {
        if (yy > 0.0)
            law.inverse[4] = 1.0 / yy;
        if (zz > 0.0)
            law.inverse[5] = 1.0 / zz;
    }
    law.active = {k[3][3] > 0.0, yy > 0.0, zz > 0.0};
    const double compliance = law.work(law.axial, law.strain(law.axial));
    if (compliance > 0.0)
        law.membrane = 1.0 / compliance;
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

// Curvature on the edge, in global axes, from the condensed moment coefficients and the axial strain.
Matrix curvatureRows(const LineData& data, const LineKinematics& kin, const SectionLaw& law)
{
    const auto pl = momentPolynomials(data.momentDegree, kin.fraction, kin.fractionRate);
    Matrix coefficients(3, data.moments.rows());
    for (std::size_t l = 0; l <= data.momentDegree; ++l)
        for (std::size_t c = 0; c < 3; ++c)
        {
            const auto v = scl3(cross3(kin.tangent, data.basis[c]), pl[l].second);
            const auto w = law.wrench(v, scl3(data.basis[c], pl[l].first));
            const auto curvature = law.strain(addAxial(w, law.axial, law.coupling(w))).moment;
            for (std::size_t k = 0; k < 3; ++k)
                coefficients(k, 3 * l + c) = curvature[k];
        }
    const Matrix local = coefficients * data.moments;
    const auto axialCurvature = law.strain(law.axial).moment;
    Matrix global(3, data.dofs.size());
    for (std::size_t j = 0; j < global.columns(); ++j)
    {
        const double gx =
            j < kin.displacementRate.columns() ? dot3(kin.tangent, columnOf(kin.displacementRate, j)) : 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t c = 0; c < 3; ++c)
                global(c, j) += law.metric * law.axes[k][c] * (local(k, j) + axialCurvature[k] * law.membrane * gx);
    }
    return global;
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
            g0[i] += axial[q].freeAxialRow[i] * factor;
    }
    return g0;
}

Vector LineData::thermalLoad(std::span<const double> freeStrains) const
{
    Vector f(dofs.size(), 0.0);
    for (std::size_t q = 0; q < axial.size() && q < freeStrains.size(); ++q)
    {
        const LineAxialPoint& p = axial[q];
        const double factor = p.axialStiffness * p.weight * p.freeAxialScale * freeStrains[q];
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
        edge[q] = axial[q].freeAxialScale * given + ofCurvature[0] * freeCurvatures[q].y +
                  ofCurvature[1] * freeCurvatures[q].z;
    }
    return edge;
}

Vector LineData::thermalMoments(std::span<const double> freeStrains,
                                std::span<const LineFreeCurvature> freeCurvatures) const
{
    // g0 = Σ w M_sᵀ (D⁻¹ Bᵀ Γ0 + K0): the term of the free axial strain of the edge, then that of the curvatures.
    Vector g0 = thermalMoments(freeStrains);
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
    std::vector<SectionLaw> laws;
    laws.reserve(points.size());
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
        const auto result = sectionLaw(line, section, kin, axes, shearOn);
        if (const auto* error = std::get_if<std::string>(&result))
            return *error;
        const SectionLaw& law = std::get<SectionLaw>(result);
        const double mu = law.metric;
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
        std::vector<Wrench> condensed(nm), strains(nm);
        Vector momentRow(nm), freeAxialRow(nm);
        std::array<Vector, 2> freeCurvatureRows{Vector(nm), Vector(nm)};
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t l = 0; l <= pm; ++l)
            {
                const auto i = 3 * l + c;
                const auto mode = law.wrench(scl3(cross3(tl, basis[c]), pl[l].second), scl3(basis[c], pl[l].first));
                momentRow[i] = law.coupling(mode);
                condensed[i] = addAxial(mode, law.axial, momentRow[i]);
                strains[i] = law.strain(condensed[i]);
                freeAxialRow[i] = mu * condensed[i].force[0];
                freeCurvatureRows[0][i] = mu * condensed[i].moment[1];
                freeCurvatureRows[1][i] = mu * condensed[i].moment[2];
            }
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
                for (std::size_t col = 0; col < nxk; ++col)
                    couplingRows(row, col) += w * momentRow[row] * gx[col];
                for (std::size_t j = 0; j < nm; ++j)
                    compliance(row, j) += w * law.work(condensed[row], strains[j]);
            }
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
        const std::array<double, 2> freeCurvatureAxial{mu * law.axial.moment[1], mu * law.axial.moment[2]};
        axial.push_back({std::move(gx), law.membrane, std::move(transverse), w, std::move(momentRow), lam,
                         std::move(freeCurvatureRows), freeCurvatureAxial, mu * law.axial.force[0],
                         std::move(freeAxialRow)});
        laws.push_back(law);
        axisPoints.push_back(add3(kin.point, law.offset));
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
    data.axial = std::move(axial);
    data.moments = std::move(moments);
    data.complianceInverse = std::move(complianceInverse);
    data.basis = basis;
    data.momentDegree = pm;
    data.shear = shear;
    if (section.bending && ((shear && section.shearFactors) || norm3(line.offsets[0]) > 0.0 ||
                            norm3(line.offsets[1]) > 0.0 || line.eccentricity[0] != 0.0 || line.eccentricity[1] != 0.0))
    {
        // Project d(theta)/d(lambda) onto Legendre modes. The mean is fixed by the two hybrid end rotations.
        // Integrating mode l gives (P_(l+1) - P_(l-1)) / (2 (2l+1)), zero at either end.
        data.rotationBubbles.assign(pm, Matrix(3, nl));
        for (std::size_t q = 0; q < points.size(); ++q)
        {
            const LineKinematics& kin = points[q].kinematics;
            const Matrix curvature = curvatureRows(data, kin, laws[q]);
            for (std::size_t l = 1; l <= pm; ++l)
                data.rotationBubbles[l - 1] +=
                    curvature * (0.5 * points[q].weight * legendreValue(l, 2.0 * kin.fraction - 1.0));
        }
    }
    for (std::size_t q = 0; q < points.size(); ++q)
    {
        const LineKinematics& kin = points[q].kinematics;
        const SectionLaw& law = laws[q];
        const SectionAxes& axes = law.axes;
        const double mu = law.metric;
        const auto& offset = law.offset;
        const LineSectionPoint sp = section.at(kin.fraction);
        const double w = points[q].weight;
        const Matrix rotation = data.sectionRotation(kin);
        const auto addMassAt = [&](double perLength, const std::array<double, 2>& centre)
        {
            if (perLength == 0.0)
                return;
            const auto arm = add3(offset, add3(scl3(axes[1], centre[0]), scl3(axes[2], centre[1])));
            Matrix displacement(3, nl);
            for (std::size_t col = 0; col < nl; ++col)
            {
                const auto spin = cross3(columnOf(rotation, col), arm);
                for (std::size_t r = 0; r < 3; ++r)
                    displacement(r, col) = (col < nxk ? kin.displacement(r, col) : 0.0) + spin[r];
            }
            inertia += displacement.transposed() * (perLength * mu * w);
            if (mass)
                *mass += transposeTimes(displacement, displacement) * (perLength * mu * w);
        };
        if (sp.neutralAxis == sp.massCentre)
            addMassAt(sp.massPerLength, sp.neutralAxis);
        else
        {
            addMassAt(sp.structuralMassPerLength, sp.neutralAxis);
            addMassAt(sp.nonStructuralMassPerLength, sp.massCentre);
        }
        if (mass)
        {
            // The rotary inertia of the section, in its axes.
            Matrix rotationInAxes(3, nl);
            for (std::size_t col = 0; col < nl; ++col)
            {
                const Vec3Of<double> v = inSectionAxes(axes, columnOf(rotation, col));
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
    }
    if (mass)
        data.mass = symmetricPart(*mass);
    data.inertia = std::move(inertia);
    data.axisPoints = std::move(axisPoints);
    return data;
}

Matrix LineData::sectionRotation(const LineKinematics& kin) const
{
    const std::size_t n = dofs.size();
    Matrix rotation(3, n);
    if (rotationBubbles.empty())
    {
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t j = 0; j < kin.rotation.columns(); ++j)
                rotation(c, j) = kin.rotation(c, j);
        return rotation;
    }
    for (std::size_t c = 0; c < 3; ++c)
    {
        rotation(c, n - 6 + c) = 1.0 - kin.fraction;
        rotation(c, n - 3 + c) = kin.fraction;
    }
    const double u = 2.0 * kin.fraction - 1.0;
    for (std::size_t l = 1; l <= rotationBubbles.size(); ++l)
        rotation += rotationBubbles[l - 1] * (legendreValue(l + 1, u) - legendreValue(l - 1, u));
    return rotation;
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
    const auto result = sectionLaw(line, section, kin, axes, data.shear && section.bending);
    if (const auto* error = std::get_if<std::string>(&result))
        return *error;
    const auto& law = std::get<SectionLaw>(result);
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
    const auto w = law.wrench(cross3(tl, dm), m);
    const double freeEdge = law.metric * (law.axial.force[0] * freeStrain + law.axial.moment[1] * freeCurvature.y +
                                          law.axial.moment[2] * freeCurvature.z);
    const auto total = addAxial(w, law.axial, law.membrane * (gamma - freeEdge) + law.coupling(w));
    const auto elastic = law.strain(total);
    LineState state;
    state.point = add3(kin.point, law.offset);
    state.axes = law.axes;
    state.forces = {total.force[0], total.force[1], total.force[2], total.moment[0], total.moment[1], total.moment[2]};
    state.strains = {elastic.force[0] + freeStrain,
                     elastic.force[1],
                     elastic.force[2],
                     elastic.moment[0],
                     elastic.moment[1] + freeCurvature.y,
                     elastic.moment[2] + freeCurvature.z};
    state.temperature = temperature;
    state.freeStrain = freeStrain;
    state.freeCurvature = freeCurvature;
    return state;
}

} // namespace oos::elements::mixedHybrid
