#include "oos/reduction/external/ExternalUnitChecks.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/reduction/external/SemiDefiniteness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace oos::reduction
{

namespace
{

// The labels a refusal names at most (the others are counted).
constexpr std::size_t NAMED_LABELS = 5;
// The six rigid-body motions, as their refusal names them.
constexpr std::array<const char*, 6> MOTION_NAMES{"the translation along basic x", "the translation along basic y",
                                                  "the translation along basic z", "the rotation about basic x",
                                                  "the rotation about basic y",    "the rotation about basic z"};

std::string labelText(const UnitDofLabel& label)
{
    return base::concatenate(
        {"(grid ", std::to_string(label.id), ", component ", std::to_string(label.component), ")"});
}

std::string number(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.3e", value);
    return text.data();
}

base::Diagnostic problem(const ExternalZone& zone, std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = "ZONE";
    diagnostic.id = zone.id;
    diagnostic.message = base::concatenate({"external unit of ", zone.displayName(), ": ", message});
    return diagnostic;
}

// The labels of `labels` as a list for a message: the first NAMED_LABELS, then how many more.
std::string labelList(const std::vector<UnitDofLabel>& labels)
{
    std::string text;
    for (std::size_t k = 0; k < labels.size() && k < NAMED_LABELS; ++k)
        text.append(k == 0 ? "" : ", ").append(labelText(labels[k]));
    if (labels.size() > NAMED_LABELS)
        text.append(" and ").append(std::to_string(labels.size() - NAMED_LABELS)).append(" more");
    return text;
}

// max |a_ij − a_ji| / max |a_ij| and the worst pair (i > j).
struct Asymmetry
{
    double ratio = 0.0;
    std::size_t row = 0;
    std::size_t column = 0;
};

Asymmetry asymmetryOf(const linalg::DenseMatrix& a)
{
    const std::size_t n = a.rows();
    double largest = 0.0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            largest = std::max(largest, std::abs(a(i, j)));
    Asymmetry worst;
    if (largest == 0.0)
        return worst;
    double difference = 0.0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            if (const double d = std::abs(a(i, j) - a(j, i)); d > difference)
            {
                difference = d;
                worst.row = i;
                worst.column = j;
            }
    worst.ratio = difference / largest;
    return worst;
}

// ½·(A + Aᵀ), lower triangle and upper alike.
linalg::DenseMatrix symmetrised(const linalg::DenseMatrix& a, base::MemoryBudget& budget, const char* label)
{
    const std::size_t n = a.rows();
    linalg::DenseMatrix s(budget, n, n, label);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            s(i, j) = 0.5 * (a(i, j) + a(j, i));
    return s;
}

} // namespace

ExternalUnitChecks::ExternalUnitChecks(const ExternalUnit& unit, const ExternalZone& zone,
                                       const assembly::DofMap& dofMap, base::MemoryBudget& budget,
                                       base::DiagnosticSink& sink)
{
    const std::size_t n = unit.size();
    bool isAccepted = true;

    // 1. The labels: exactly the contour DOFs of the zone.
    std::vector<UnitDofLabel> missing; // bounded by the contour
    std::vector<base::DofIndex> dofOfRow(n, -1);
    for (std::size_t c = 0; c < zone.contourLabels.size(); ++c)
    {
        const std::optional<std::size_t> row = unit.rowOf(zone.contourLabels[c]);
        if (!row)
            missing.push_back(zone.contourLabels[c]);
        else
            dofOfRow[*row] = zone.contourDofs[c];
    }
    std::vector<UnitDofLabel> extra; // bounded by the unit's labels
    for (std::size_t r = 0; r < n; ++r)
        if (dofOfRow[r] < 0)
            extra.push_back(unit.labels()[r]);
    if (!missing.empty() || !extra.empty())
    {
        std::string message = base::concatenate({"its labels are not the contour DOFs of the zone (",
                                                 std::to_string(zone.contourLabels.size()), " DOFs, ",
                                                 std::to_string(n), " labels given)"});
        if (!missing.empty())
            message.append("; missing: ").append(labelList(missing));
        if (!extra.empty())
            message.append("; not on the contour: ").append(labelList(extra));
        message.append(
            "; give K* on the contour DOFs of the zone report, in the displacement systems of the grid "
            "points");
        sink.report(problem(zone, "external-unit-labels", std::move(message)));
        return;
    }
    m_report.areLabelsAccepted = true;

    // 2. Symmetry.
    const auto checkSymmetry = [&](const linalg::DenseMatrix& matrix, const char* what) -> double
    {
        const Asymmetry asymmetry = asymmetryOf(matrix);
        if (asymmetry.ratio > EXTERNAL_UNIT_SYMMETRY_TOLERANCE)
        {
            isAccepted = false;
            sink.report(problem(
                zone, "external-unit-symmetry",
                base::concatenate({what, " is not symmetric: its terms at ", labelText(unit.labels()[asymmetry.row]),
                                   " x ", labelText(unit.labels()[asymmetry.column]),
                                   " and the transposed one differ by ", number(asymmetry.ratio),
                                   " of its largest term, above the tolerance ",
                                   number(EXTERNAL_UNIT_SYMMETRY_TOLERANCE),
                                   " (EXTERNAL_UNIT_SYMMETRY_TOLERANCE); give a symmetric matrix"})));
        }
        return asymmetry.ratio;
    };
    m_report.asymmetry = checkSymmetry(unit.stiffness(), "the stiffness K*");
    if (unit.hasMass())
        m_report.massAsymmetry = checkSymmetry(unit.mass(), "the mass M*");
    for (const ExternalUnitDerivative& derivative : unit.derivatives())
    {
        const std::string stiffnessWhat =
            base::concatenate({"the derivative of K* with respect to variable ", std::to_string(derivative.variable)});
        m_report.derivativeAsymmetry =
            std::max(m_report.derivativeAsymmetry, checkSymmetry(derivative.stiffness, stiffnessWhat.c_str()));
        if (derivative.mass)
        {
            const std::string massWhat = base::concatenate(
                {"the derivative of M* with respect to variable ", std::to_string(derivative.variable)});
            m_report.derivativeAsymmetry =
                std::max(m_report.derivativeAsymmetry, checkSymmetry(*derivative.mass, massWhat.c_str()));
        }
    }
    const linalg::DenseMatrix stiffness = symmetrised(unit.stiffness(), budget, "external unit K* check");

    // 3. The rigid-body motions of the contour, unless the zone holds constraints.
    if (!zone.holdsConstraints)
    {
        m_report.isRigidBodyChecked = true;
        // The centroid of the contour points: the lever arms of the rotations stay of the size of the zone.
        base::Vec3 centroid{};
        std::vector<std::int32_t> gridRows; // bounded by the contour
        for (std::size_t r = 0; r < n; ++r)
            gridRows.push_back(dofMap.labelOf(dofOfRow[r]).row);
        std::vector<std::int32_t> points = gridRows;
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        for (const std::int32_t row : points)
            centroid = centroid + dofMap.positionInBasic(row);
        centroid = centroid * (1.0 / static_cast<double>(points.size()));
        std::vector<double> motion(n);
        double largestTerm = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
                largestTerm = std::max(largestTerm, std::abs(stiffness(i, j)));
        double worstResidual = 0.0;
        std::size_t worstMotion = 0;
        std::size_t worstRow = 0;
        for (std::size_t m = 0; m < MOTION_NAMES.size(); ++m)
        {
            for (std::size_t r = 0; r < n; ++r)
            {
                const std::int32_t gridRow = gridRows[r];
                base::Vec3 translation{};
                base::Vec3 rotation{};
                if (m < 3)
                    translation[m] = 1.0;
                else
                    rotation[m - 3] = 1.0;
                const base::Vec3 arm = dofMap.positionInBasic(gridRow) - centroid;
                const base::Mat3 toBasic = dofMap.cdToBasic(gridRow);
                const base::Vec3 displacement = toBasic.transposed() * (translation + base::cross(rotation, arm));
                const base::Vec3 turn = toBasic.transposed() * rotation;
                const int component = unit.labels()[r].component;
                motion[r] = component <= 3 ? displacement[static_cast<std::size_t>(component - 1)]
                                           : turn[static_cast<std::size_t>(component - 4)];
            }
            double largestMotion = 0.0;
            for (const double value : motion)
                largestMotion = std::max(largestMotion, std::abs(value));
            const double scale = largestTerm * largestMotion;
            if (scale == 0.0)
                continue; // the motion moves no label, or K* is zero
            for (std::size_t i = 0; i < n; ++i)
            {
                double force = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    force += stiffness(i, j) * motion[j];
                if (const double residual = std::abs(force) / scale; residual > worstResidual)
                {
                    worstResidual = residual;
                    worstMotion = m;
                    worstRow = i;
                }
            }
        }
        m_report.rigidBodyResidual = worstResidual;
        if (worstResidual > EXTERNAL_UNIT_RIGID_BODY_TOLERANCE)
        {
            isAccepted = false;
            sink.report(problem(
                zone, "external-unit-rigid-body",
                base::concatenate({"the stiffness K* misses a rigid-body motion of the contour: ",
                                   MOTION_NAMES[worstMotion], " meets the stiffness ", number(worstResidual),
                                   " (relative to the largest term of K*) at ", labelText(unit.labels()[worstRow]),
                                   ", above the tolerance ", number(EXTERNAL_UNIT_RIGID_BODY_TOLERANCE),
                                   " (EXTERNAL_UNIT_RIGID_BODY_TOLERANCE); a free local model has no stiffness "
                                   "against a rigid motion (look for a grounded spring or a non-invariant element), "
                                   "or hold its constraints in the zone"})));
        }
    }

    // 4. Semi-definiteness, by the factorisation.
    const auto checkDefiniteness = [&](const linalg::DenseMatrix& matrix, const char* what) -> SemiDefiniteResult
    {
        const SemiDefiniteResult result = testSemiDefinite(matrix.view(), EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE, budget);
        if (!result.isSemiDefinite)
        {
            isAccepted = false;
            sink.report(
                problem(zone, "external-unit-semidefinite",
                        base::concatenate({what, " has a negative eigenvalue: its factorisation meets the pivot ",
                                           number(result.smallestPivot), " (relative to its largest term) at ",
                                           labelText(unit.labels()[result.row]), ", below −",
                                           number(EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE),
                                           " (EXTERNAL_UNIT_SEMIDEFINITE_TOLERANCE); a stiffness or a mass is positive "
                                           "semi-definite: look for a wrong sign"})));
        }
        return result;
    };
    const SemiDefiniteResult definiteness = checkDefiniteness(stiffness, "the stiffness K*");
    m_report.rank = definiteness.rank;
    m_report.smallestPivot = definiteness.smallestPivot;
    if (unit.hasMass())
    {
        const linalg::DenseMatrix mass = symmetrised(unit.mass(), budget, "external unit M* check");
        m_report.smallestMassPivot = checkDefiniteness(mass, "the mass M*").smallestPivot;
    }
    m_report.isAccepted = isAccepted;
}

} // namespace oos::reduction
