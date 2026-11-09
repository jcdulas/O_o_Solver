#include "oos/assembly/constraints/RigidElementEquations.hpp"

#include "oos/assembly/singularities/SmallSymmetricEigen.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/ids/ComponentCode.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::assembly
{

namespace
{

using Row6 = std::array<double, 6>;

// One grid point row of a rigid element in the table.
struct PointRow
{
    base::EntityId nodeId = 0;
    std::int32_t gridRow = 0;
    base::ComponentCode components;
    bool isIndependent = true;
    double weight = 0.0;
};

// One DOF of a rigid element: the grid point row and the component 1 to 6.
struct ElementDofRef
{
    std::int32_t gridRow = 0;
    int component = 1;
};

base::Diagnostic rigidDiagnostic(std::string code, std::string_view card, base::EntityId elementId, std::string field,
                                 std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = elementId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The row, over the rigid motion (t, θ) in the basic system at x0, of component k (1 to 6) of a grid point at
// x0 + r whose CD-to-basic rotation is `rotation`: the small-rotation rigid transfer t_p = Rᵀ·(t − [r×]·θ),
// θ_p = Rᵀ·θ. The rotation part is multiplied by `rotationScale`.
Row6 rigidRow(const base::Mat3& rotation, int component, base::Vec3 r, double rotationScale)
{
    Row6 row{};
    const std::size_t k = static_cast<std::size_t>((component - 1) % 3);
    const base::Vec3 axis{rotation(0, k), rotation(1, k), rotation(2, k)}; // the CD direction k in basic
    if (component <= 3)
    {
        // axis·(t − r × θ) = axis·t + θ·(r × axis), by the scalar triple product.
        const base::Vec3 rotationPart = base::cross(r, axis);
        for (std::size_t i = 0; i < 3; ++i)
        {
            row[i] = axis[i];
            row[3 + i] = rotationPart[i] * rotationScale;
        }
    }
    else
    {
        for (std::size_t i = 0; i < 3; ++i)
            row[3 + i] = axis[i] * rotationScale;
    }
    return row;
}

// The row of a DOF of an independent point over the CD DOFs of a reference point: rigidRow · blockdiag(R_ref, R_ref).
Row6 transferredRow(const Row6& basicRow, const base::Mat3& referenceRotation)
{
    Row6 row{};
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 3; ++i)
        {
            row[j] += basicRow[i] * referenceRotation(i, j);
            row[3 + j] += basicRow[3 + i] * referenceRotation(i, j);
        }
    return row;
}

// Inverts the n × n row-major `a` in place by Gauss-Jordan elimination with partial pivoting; returns the smallest
// pivot magnitude met (0 when a pivot is exactly zero, then `a` is left undefined).
double invertInPlace(std::vector<double>& a, std::size_t n)
{
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        inverse[i * n + i] = 1.0;
    double smallestPivot = HUGE_VAL;
    for (std::size_t k = 0; k < n; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(a[i * n + k]) > std::abs(a[pivot * n + k]))
                pivot = i;
        const double pivotValue = a[pivot * n + k];
        smallestPivot = std::min(smallestPivot, std::abs(pivotValue));
        if (pivotValue == 0.0)
            return 0.0;
        if (pivot != k)
            for (std::size_t j = 0; j < n; ++j)
            {
                std::swap(a[k * n + j], a[pivot * n + j]);
                std::swap(inverse[k * n + j], inverse[pivot * n + j]);
            }
        for (std::size_t j = 0; j < n; ++j)
        {
            a[k * n + j] /= pivotValue;
            inverse[k * n + j] /= pivotValue;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i == k || a[i * n + k] == 0.0)
                continue;
            const double factor = a[i * n + k];
            for (std::size_t j = 0; j < n; ++j)
            {
                a[i * n + j] -= factor * a[k * n + j];
                inverse[i * n + j] -= factor * inverse[k * n + j];
            }
        }
    }
    a = std::move(inverse);
    return smallestPivot;
}

std::string pointText(base::EntityId nodeId, int component)
{
    return base::concatenate({"component ", std::to_string(component), " of grid point ", std::to_string(nodeId)});
}

// The element-local logic, one function per formulation, sharing the parsed rows.
class ElementBuilder
{
public:
    ElementBuilder(const DofMap& dofMap, std::string_view card, base::EntityId elementId, std::vector<PointRow> points,
                   LocalConstraintRows& rows, base::DiagnosticSink& sink)
        : m_dofMap(dofMap),
          m_card(card),
          m_elementId(elementId),
          m_points(std::move(points)),
          m_rows(rows),
          m_sink(sink)
    {
    }

    bool buildRigidBody(ConstraintSource source);
    bool buildRigidRod();
    bool buildRigidInterpolation();

private:
    void report(std::string code, std::string field, std::string message)
    {
        m_sink.report(rigidDiagnostic(std::move(code), m_card, m_elementId, std::move(field), std::move(message)));
    }
    [[nodiscard]] base::DofIndex dofOf(const ElementDofRef& dof) const
    {
        return m_dofMap.gridDof(dof.gridRow, dof.component);
    }
    [[nodiscard]] base::EntityId idOf(std::int32_t gridRow) const { return m_dofMap.gridId(gridRow); }
    // The DOFs of the rows of one role, in the order of the table and of the components.
    [[nodiscard]] std::vector<ElementDofRef> dofsOf(bool isIndependent) const;
    // Reports a DOF that is both independent and dependent; true when there is none.
    bool checkRoles(const std::vector<ElementDofRef>& independent, const std::vector<ElementDofRef>& dependent);

    const DofMap& m_dofMap;
    std::string_view m_card;
    base::EntityId m_elementId;
    std::vector<PointRow> m_points;
    LocalConstraintRows& m_rows;
    base::DiagnosticSink& m_sink;
};

std::vector<ElementDofRef> ElementBuilder::dofsOf(bool isIndependent) const
{
    std::vector<ElementDofRef> dofs;
    for (const PointRow& point : m_points)
    {
        if (point.isIndependent != isIndependent)
            continue;
        for (int component = 1; component <= 6; ++component)
            if (point.components.contains(component))
                dofs.push_back({point.gridRow, component});
    }
    return dofs;
}

bool ElementBuilder::checkRoles(const std::vector<ElementDofRef>& independent,
                                const std::vector<ElementDofRef>& dependent)
{
    for (const ElementDofRef& d : dependent)
        for (const ElementDofRef& i : independent)
            if (d.gridRow == i.gridRow && d.component == i.component)
            {
                report("rigid-dof-both-roles", "",
                       base::concatenate({pointText(idOf(d.gridRow), d.component),
                                          " is both independent and dependent in the element; make it one or the "
                                          "other"}));
                return false;
            }
    return true;
}

bool ElementBuilder::buildRigidBody(ConstraintSource source)
{
    const std::vector<ElementDofRef> independent = dofsOf(true);
    const std::vector<ElementDofRef> dependent = dofsOf(false);
    if (independent.size() != 6)
    {
        const char* field = source == ConstraintSource::RigidBar         ? "independentComponentsA"
                            : source == ConstraintSource::RigidBodyMulti ? "independentComponents1"
                                                                         : "independentNode";
        report("rigid-independent-count", field,
               base::concatenate({"a rigid ", m_card, " needs exactly 6 independent components (", m_card,
                                  ": they define the rigid motion of the element), found ",
                                  std::to_string(independent.size()), "; correct its independent components"}));
        return false;
    }
    if (!checkRoles(independent, dependent))
        return false;
    // The rigid motion q is taken at the first independent grid point; its rotation is scaled by the element size so
    // that the columns of A are of the same order (a length times a rotation is a displacement).
    const base::Vec3 origin = m_dofMap.positionInBasic(independent.front().gridRow);
    double size = 0.0;
    for (const PointRow& point : m_points)
        size = std::max(size, base::norm(m_dofMap.positionInBasic(point.gridRow) - origin));
    const double rotationScale = size > 0.0 ? 1.0 / size : 1.0; // θ = θ'/size: the rotation part scales by 1/size
    const auto rowOf = [&](const ElementDofRef& dof)
    {
        return rigidRow(m_dofMap.cdToBasic(dof.gridRow), dof.component, m_dofMap.positionInBasic(dof.gridRow) - origin,
                        rotationScale);
    };
    // A' = D·A with each row normalised to a largest term of 1: the pivots of A' are then comparable to 1.
    std::vector<double> a(36);
    std::array<double, 6> rowScale{};
    for (std::size_t i = 0; i < 6; ++i)
    {
        const Row6 row = rowOf(independent[i]);
        double largest = 0.0;
        for (const double value : row)
            largest = std::max(largest, std::abs(value));
        rowScale[i] = largest > 0.0 ? largest : 1.0;
        for (std::size_t j = 0; j < 6; ++j)
            a[i * 6 + j] = row[j] / rowScale[i];
    }
    if (invertInPlace(a, 6) < RigidElementEquations::RIGID_PIVOT_TOLERANCE)
    {
        report("rigid-independent-singular", source == ConstraintSource::RigidBar ? "independentComponentsA" : "",
               base::concatenate({"the 6 independent components of the ", m_card,
                                  " cannot describe every rigid motion of the element (", m_card,
                                  "): choose components that fix the three translations and the three rotations (for "
                                  "a bar, the rotation about its axis needs a rotational component)"}));
        return false;
    }
    // A⁻¹ = A'⁻¹·D: column j of A'⁻¹ divided by the scale of row j.
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            a[i * 6 + j] /= rowScale[j];
    std::array<ConstraintTerm, 6> terms{};
    for (const ElementDofRef& dof : dependent)
    {
        const Row6 b = rowOf(dof);
        for (std::size_t j = 0; j < 6; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < 6; ++k)
                sum += b[k] * a[k * 6 + j];
            terms[j] = {dofOf(independent[j]), sum};
        }
        m_rows.addRow(dofOf(dof), terms);
    }
    return true;
}

bool ElementBuilder::buildRigidRod()
{
    if (m_points.size() != 2)
    {
        report("rigid-rod-bad-component", "componentsAtA",
               base::concatenate({"an RigidRod connects two grid points, found ", std::to_string(m_points.size()),
                                  " rows; give one dependent translation, RigidRod.componentsAtA or "
                                  "RigidRod.componentsAtB"}));
        return false;
    }
    const PointRow& a = m_points[0];
    const PointRow& b = m_points[1];
    const PointRow& dependentPoint = a.isIndependent ? b : a;
    const PointRow& independentPoint = a.isIndependent ? a : b;
    const int count = dependentPoint.components.count();
    int dependentComponent = 0;
    for (int component = 1; component <= 3; ++component)
        if (dependentPoint.components.contains(component))
            dependentComponent = component;
    if (dependentPoint.isIndependent || count != 1 || dependentComponent == 0 || !independentPoint.components.isEmpty())
    {
        report("rigid-rod-bad-component", "componentsAtA",
               "an RigidRod has exactly one dependent translational component (1, 2 or 3), at RigidRod.nodeA "
               "(RigidRod.componentsAtA) or at RigidRod.nodeB (RigidRod.componentsAtB); "
               "correct RigidRod.componentsAtA and RigidRod.componentsAtB");
        return false;
    }
    const base::Vec3 span = m_dofMap.positionInBasic(b.gridRow) - m_dofMap.positionInBasic(a.gridRow);
    const double length = base::norm(span);
    if (length == 0.0)
    {
        report("rigid-rod-zero-length", "nodeB",
               "the two grid points of the RigidRod coincide, so it has no axis; move one of them or use an "
               "RigidBodyStar");
        return false;
    }
    const base::Vec3 axis = span / length;
    // RigidRod: e·(t_B − t_A) = 0; the coefficient of translation k of a point is ±(Rᵀ·e)_k.
    std::array<ConstraintTerm, 6> equation{};
    std::size_t next = 0;
    for (const PointRow* point : {&a, &b})
    {
        const double sign = point == &a ? -1.0 : 1.0;
        const base::Mat3 rotation = m_dofMap.cdToBasic(point->gridRow);
        for (int component = 1; component <= 3; ++component)
        {
            const auto k = static_cast<std::size_t>(component - 1);
            const double coefficient =
                sign * (rotation(0, k) * axis[0] + rotation(1, k) * axis[1] + rotation(2, k) * axis[2]);
            equation[next++] = {m_dofMap.gridDof(point->gridRow, component), coefficient};
        }
    }
    const base::DofIndex dependentDof = m_dofMap.gridDof(dependentPoint.gridRow, dependentComponent);
    double dependentCoefficient = 0.0;
    for (const ConstraintTerm& term : equation)
        if (term.dof == dependentDof)
            dependentCoefficient = term.coefficient;
    if (std::abs(dependentCoefficient) < RigidElementEquations::RIGID_ROD_MIN_AXIS_COSINE)
    {
        report(
            "rigid-rod-dependent-perpendicular",
            dependentPoint.gridRow == a.gridRow ? "componentsAtA" : "componentsAtB",
            base::concatenate({"the dependent ", pointText(dependentPoint.nodeId, dependentComponent),
                               " is perpendicular to the RigidRod axis, so the element cannot be solved for it; choose "
                               "the translation most nearly along the axis"}));
        return false;
    }
    std::array<ConstraintTerm, 5> terms{};
    next = 0;
    for (const ConstraintTerm& term : equation)
        if (term.dof != dependentDof)
            terms[next++] = {term.dof, -term.coefficient / dependentCoefficient};
    m_rows.addRow(dependentDof, terms);
    return true;
}

bool ElementBuilder::buildRigidInterpolation()
{
    // The rows: the weighted independent points, then the reference point (REFC), then the UM DOFs.
    std::vector<const PointRow*> independents;
    std::vector<const PointRow*> dependents;
    for (const PointRow& point : m_points)
        (point.isIndependent ? independents : dependents).push_back(&point);
    if (dependents.empty() || dependents.front()->components.isEmpty() || independents.empty())
    {
        report("rigid-interpolation-bad-reference", "referenceComponents",
               "an RigidInterpolation needs a reference grid point with its components "
               "RigidInterpolation.referenceComponents "
               "and at least one weighted "
               "grid point; "
               "complete the card");
        return false;
    }
    const PointRow& reference = *dependents.front();
    for (const PointRow* point : independents)
    {
        if (!(point->weight >= 0.0) || point->components.isEmpty())
        {
            report("rigid-interpolation-bad-weight", "weight",
                   base::concatenate({"grid point ", std::to_string(point->nodeId),
                                      " has a weight that is negative or not a number, or no components; give each "
                                      "group a weight RigidInterpolation.weighti >= 0 and its components "
                                      "RigidInterpolation.componentsi"}));
            return false;
        }
    }
    // The independent DOFs, each once (a grid point may be in several groups).
    std::vector<ElementDofRef> columns;
    const auto columnOf = [&](const ElementDofRef& dof)
    {
        for (std::size_t c = 0; c < columns.size(); ++c)
            if (columns[c].gridRow == dof.gridRow && columns[c].component == dof.component)
                return c;
        columns.push_back(dof);
        return columns.size() - 1;
    };
    std::vector<ElementDofRef> referenceDofs;
    for (int component = 1; component <= 6; ++component)
        if (reference.components.contains(component))
            referenceDofs.push_back({reference.gridRow, component});
    for (const PointRow* point : independents)
        for (int component = 1; component <= 6; ++component)
            if (point->components.contains(component))
                (void)columnOf({point->gridRow, component});
    if (!checkRoles(columns, referenceDofs))
        return false;

    // Lc, the mean distance of the independent grid points (each once) to the reference point. A point within the
    // model's coincidence distance of the reference point is on it: no lever arm, in Lc and in its rows (its
    // arm of round-off made a single leg's rotation rows ill-conditioned: the rows are scaled by 1/Lc, Lc that
    // round-off, which turned the eigensolution's 1e-16 into spurious terms of 1e-16/Lc).
    const base::Vec3 referencePosition = m_dofMap.positionInBasic(reference.gridRow);
    const double coincidence = m_dofMap.coincidenceDistance();
    const auto armOf = [&](std::int32_t gridRow)
    {
        const base::Vec3 arm = m_dofMap.positionInBasic(gridRow) - referencePosition;
        return base::norm(arm) <= coincidence ? base::Vec3{} : arm;
    };
    double distanceSum = 0.0;
    std::size_t distinctCount = 0;
    for (std::size_t i = 0; i < independents.size(); ++i)
    {
        bool isRepeated = false;
        for (std::size_t j = 0; j < i; ++j)
            isRepeated = isRepeated || independents[j]->gridRow == independents[i]->gridRow;
        if (isRepeated)
            continue;
        distanceSum += base::norm(armOf(independents[i]->gridRow));
        ++distinctCount;
    }
    const double characteristicLength = distanceSum > 0.0 ? distanceSum / static_cast<double>(distinctCount) : 1.0;

    // Normal equations in the scaled reference DOFs u' = D·u, D = diag(1, 1, 1, Lc, Lc, Lc): a row s over u becomes
    // s' = s·D⁻¹, and with the rotational weights scaled by Lc² every term of M' = Σ w·s'ᵀ·s' is of the order of the
    // weights.
    const base::Mat3 referenceRotation = m_dofMap.cdToBasic(reference.gridRow);
    base::MatN<6> normal;
    std::vector<Row6> rightHandSide(columns.size(), Row6{}); // H' column by column
    for (const PointRow* point : independents)
    {
        const base::Mat3 rotation = m_dofMap.cdToBasic(point->gridRow);
        const base::Vec3 r = armOf(point->gridRow);
        for (int component = 1; component <= 6; ++component)
        {
            if (!point->components.contains(component))
                continue;
            Row6 s = transferredRow(rigidRow(rotation, component, r, 1.0), referenceRotation);
            for (std::size_t j = 3; j < 6; ++j)
                s[j] /= characteristicLength;
            const double weight =
                component <= 3 ? point->weight : point->weight * characteristicLength * characteristicLength;
            Row6& h = rightHandSide[columnOf({point->gridRow, component})];
            for (std::size_t i = 0; i < 6; ++i)
            {
                h[i] += weight * s[i];
                for (std::size_t j = 0; j < 6; ++j)
                    normal(i, j) += weight * s[i] * s[j];
            }
        }
    }
    const SmallEigenDecomposition<6> eigen = SmallSymmetricEigen<6>::decompose(normal);
    const double largest = eigen.values[5];
    std::array<bool, 6> isNull{};
    for (std::size_t j = 0; j < 6; ++j)
        isNull[j] =
            !(largest > 0.0) || eigen.values[j] <= RigidElementEquations::RIGID_INTERPOLATION_RANK_TOLERANCE * largest;
    // A REFC component is determined when no null vector z of M (z = D⁻¹·z' in the unscaled DOFs) has a part on it.
    for (const ElementDofRef& dof : referenceDofs)
    {
        const auto c = static_cast<std::size_t>(dof.component - 1);
        for (std::size_t j = 0; j < 6; ++j)
        {
            if (!isNull[j])
                continue;
            double length = 0.0;
            for (std::size_t i = 0; i < 6; ++i)
            {
                const double zi = eigen.vectors(i, j) / (i < 3 ? 1.0 : characteristicLength);
                length += zi * zi;
            }
            const double part = std::abs(eigen.vectors(c, j) / (c < 3 ? 1.0 : characteristicLength));
            if (part > RigidElementEquations::RIGID_INTERPOLATION_NULL_COMPONENT_TOLERANCE * std::sqrt(length))
            {
                report("rigid-interpolation-reference-undetermined", "referenceComponents",
                       base::concatenate(
                           {pointText(reference.nodeId, dof.component),
                            " (RigidInterpolation.referenceComponents) is not determined by the weighted "
                            "components of "
                            "the RigidInterpolation (the "
                            "independent DOFs must be able to resist every rigid motion of the "
                            "reference components); add components RigidInterpolation.componentsi or grid points, "
                            "or remove it from RigidInterpolation.referenceComponents"}));
                return false;
            }
        }
    }
    // G = D⁻¹·M'⁺·H', the rows of the REFC components only.
    std::vector<std::vector<double>> g(referenceDofs.size(), std::vector<double>(columns.size(), 0.0));
    for (std::size_t r = 0; r < referenceDofs.size(); ++r)
    {
        const auto c = static_cast<std::size_t>(referenceDofs[r].component - 1);
        const double unscale = c < 3 ? 1.0 : 1.0 / characteristicLength;
        for (std::size_t col = 0; col < columns.size(); ++col)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < 6; ++j)
            {
                if (isNull[j])
                    continue;
                double projection = 0.0;
                for (std::size_t i = 0; i < 6; ++i)
                    projection += eigen.vectors(i, j) * rightHandSide[col][i];
                sum += eigen.vectors(c, j) * projection / eigen.values[j];
            }
            g[r][col] = unscale * sum;
        }
    }

    const std::size_t equationCount = referenceDofs.size();
    std::vector<ConstraintTerm> terms;
    if (dependents.size() == 1)
    {
        // Default: the REFC DOFs are dependent, u_ref,c = G_c·u_ind.
        for (std::size_t r = 0; r < equationCount; ++r)
        {
            terms.clear();
            for (std::size_t col = 0; col < columns.size(); ++col)
                terms.push_back({dofOf(columns[col]), g[r][col]});
            m_rows.addRow(dofOf(referenceDofs[r]), terms);
        }
        return true;
    }
    // UM: the equations u_ref,c − G_c·u_ind = 0 over the element DOFs [REFC..., independent...], solved for the UM
    // DOFs instead.
    std::vector<ElementDofRef> elementDofs = referenceDofs;
    elementDofs.insert(elementDofs.end(), columns.begin(), columns.end());
    const std::size_t dofCount = elementDofs.size();
    std::vector<double> equations(equationCount * dofCount, 0.0);
    for (std::size_t r = 0; r < equationCount; ++r)
    {
        equations[r * dofCount + r] = 1.0;
        for (std::size_t col = 0; col < columns.size(); ++col)
            equations[r * dofCount + equationCount + col] = -g[r][col];
    }
    std::vector<std::size_t> mPositions;
    for (std::size_t d = 1; d < dependents.size(); ++d)
        for (int component = 1; component <= 6; ++component)
        {
            if (!dependents[d]->components.contains(component))
                continue;
            std::size_t position = dofCount;
            for (std::size_t e = 0; e < dofCount; ++e)
                if (elementDofs[e].gridRow == dependents[d]->gridRow && elementDofs[e].component == component)
                    position = e;
            if (position == dofCount || std::find(mPositions.begin(), mPositions.end(), position) != mPositions.end())
            {
                report("rigid-interpolation-dependent-invalid", "dependentMarker",
                       base::concatenate(
                           {pointText(dependents[d]->nodeId, component),
                            " is listed after RigidInterpolation.dependentMarker but is not a DOF of the element, "
                            "or "
                            "is listed twice (RigidInterpolation: the RigidInterpolation.dependentMarker DOFs are "
                            "DOFs of "
                            "RigidInterpolation.referenceNode/RigidInterpolation.referenceComponents "
                            "or of the weighted points); correct the RigidInterpolation.dependentMarker list"}));
                return false;
            }
            mPositions.push_back(position);
        }
    if (mPositions.size() != equationCount)
    {
        report(
            "rigid-interpolation-dependent-invalid", "dependentMarker",
            base::concatenate({"the RigidInterpolation.dependentMarker list gives ", std::to_string(mPositions.size()),
                               " DOFs where RigidInterpolation.referenceComponents has ", std::to_string(equationCount),
                               " (RigidInterpolation: as many DOFs in RigidInterpolation.dependentMarker as in "
                               "RigidInterpolation.referenceComponents); correct the "
                               "RigidInterpolation.dependentMarker list"}));
        return false;
    }
    std::vector<double> mBlock(equationCount * equationCount);
    double largestTerm = 0.0;
    for (std::size_t r = 0; r < equationCount; ++r)
        for (std::size_t k = 0; k < equationCount; ++k)
        {
            mBlock[r * equationCount + k] = equations[r * dofCount + mPositions[k]];
            largestTerm = std::max(largestTerm, std::abs(mBlock[r * equationCount + k]));
        }
    if (invertInPlace(mBlock, equationCount) <=
        RigidElementEquations::RIGID_INTERPOLATION_DEPENDENT_PIVOT_TOLERANCE * largestTerm)
    {
        report("rigid-interpolation-dependent-singular", "dependentMarker",
               "the RigidInterpolation equations cannot be solved for the DOFs listed after "
               "RigidInterpolation.dependentMarker (they do not take part in independent equations); choose other "
               "RigidInterpolation.dependentMarker DOFs");
        return false;
    }
    for (std::size_t k = 0; k < equationCount; ++k)
    {
        terms.clear();
        for (std::size_t e = 0; e < dofCount; ++e)
        {
            if (std::find(mPositions.begin(), mPositions.end(), e) != mPositions.end())
                continue;
            double sum = 0.0;
            for (std::size_t r = 0; r < equationCount; ++r)
                sum += mBlock[k * equationCount + r] * equations[r * dofCount + e];
            terms.push_back({dofOf(elementDofs[e]), -sum});
        }
        m_rows.addRow(dofOf(elementDofs[mPositions[k]]), terms);
    }
    return true;
}

} // namespace

std::string_view cardNameOf(ConstraintSource source) noexcept
{
    switch (source)
    {
    case ConstraintSource::RigidBodyMulti:
        return "RigidBodyMulti";
    case ConstraintSource::RigidBodyStar:
        return "RigidBodyStar";
    case ConstraintSource::RigidInterpolation:
        return "RigidInterpolation";
    case ConstraintSource::RigidBar:
        return "RigidBar";
    case ConstraintSource::RigidRod:
        return "RigidRod";
    case ConstraintSource::Mpc:
        return "MultiPointConstr";
    case ConstraintSource::Fastener:
        return "Fastener";
    case ConstraintSource::Weld:
        return "Weld";
    case ConstraintSource::ContactTable:
        return "ContactTable";
    case ConstraintSource::ContactPair:
        return "ContactPair";
    }
    return "?";
}

void LocalConstraintRows::addRow(base::DofIndex dependent, std::span<const ConstraintTerm> rowTerms)
{
    dependents.push_back(dependent);
    for (const ConstraintTerm& term : rowTerms)
        if (term.coefficient != 0.0)
            terms.push_back(term);
    offsets.push_back(terms.size());
}

std::optional<ConstraintSource> RigidElementEquations::findSource(std::string_view type) noexcept
{
    for (const ConstraintSource source :
         {ConstraintSource::RigidBodyMulti, ConstraintSource::RigidBodyStar, ConstraintSource::RigidInterpolation,
          ConstraintSource::RigidBar, ConstraintSource::RigidRod})
        if (type == cardNameOf(source))
            return source;
    return std::nullopt;
}

bool RigidElementEquations::build(const input::RigidTable& table, std::int64_t beginRow, std::int64_t endRow,
                                  LocalConstraintRows& rows, base::DiagnosticSink& sink) const
{
    rows.clear();
    const base::DictionaryView types = table.types();
    const base::DictionaryView roles = table.roles();
    const std::span<const std::int32_t> elementIds = table.ids();
    const std::span<const std::int32_t> nodeIds = table.nodeIds();
    const base::Utf8View components = table.components();
    const std::span<const double> weights = table.weights();
    const std::string_view card = types[beginRow];
    const base::EntityId elementId = elementIds[static_cast<std::size_t>(beginRow)];
    const std::optional<ConstraintSource> source = findSource(card);
    if (!source)
    {
        sink.report(
            rigidDiagnostic("unsupported-rigid-element", card, elementId, "",
                            base::concatenate({"the rigid element ", card,
                                               " is not supported by this version of O_o_Solver (RigidBodyMulti, "
                                               "RigidBodyStar, RigidInterpolation, RigidBar and RigidRod are); "
                                               "replace it by one of them"})));
        return false;
    }
    std::vector<PointRow> points;
    bool isValid = true;
    for (std::int64_t row = beginRow; row < endRow; ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        const std::optional<input::PointRef> point = m_dofMap->points().find(nodeIds[index]);
        if (!point || point->kind != input::PointKind::Grid)
        {
            sink.report(rigidDiagnostic(
                point ? "rigid-scalar-point" : "rigid-unknown-point", card, elementId, "",
                base::concatenate(
                    {"the element connects point ", std::to_string(nodeIds[index]),
                     point ? ", a scalar point: rigid elements connect grid points (use an MultiPointConstr)"
                           : ", which the model does not define: add the Node or correct the card"})));
            isValid = false;
            continue;
        }
        PointRow parsed;
        parsed.nodeId = nodeIds[index];
        parsed.gridRow = point->row;
        parsed.isIndependent = roles[row] == "independent";
        parsed.weight = weights[index];
        bool isParsed = true;
        try
        {
            parsed.components = base::ComponentCode::parse(components[row]);
        }
        catch (const std::invalid_argument&)
        {
            isParsed = false;
        }
        if (!isParsed || parsed.components.isScalar())
        {
            sink.report(rigidDiagnostic(
                "rigid-bad-component", card, elementId, "",
                base::concatenate({"grid point ", std::to_string(nodeIds[index]), " has the components '",
                                   components[row], "'; expected digits 1 to 6"})));
            isValid = false;
            continue;
        }
        points.push_back(parsed);
    }
    if (!isValid)
        return false;
    ElementBuilder builder(*m_dofMap, card, elementId, std::move(points), rows, sink);
    bool isBuilt = false;
    switch (*source)
    {
    case ConstraintSource::RigidBodyMulti:
    case ConstraintSource::RigidBodyStar:
    case ConstraintSource::RigidBar:
        isBuilt = builder.buildRigidBody(*source);
        break;
    case ConstraintSource::RigidRod:
        isBuilt = builder.buildRigidRod();
        break;
    case ConstraintSource::RigidInterpolation:
        isBuilt = builder.buildRigidInterpolation();
        break;
    case ConstraintSource::Mpc:
    case ConstraintSource::Fastener:
    case ConstraintSource::Weld:
    case ConstraintSource::ContactTable:
    case ConstraintSource::ContactPair:
        break; // not cards of rigidElementsTable (findSource never gives them)
    }
    if (!isBuilt)
        rows.clear();
    return isBuilt;
}

} // namespace oos::assembly
