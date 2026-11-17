#include "oos/elements/mixedHybrid/solid/VolumeKernel.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

using DualNumber = Dual<double>;
using DualPoint = std::array<DualNumber, 3>;

constexpr std::size_t MIN_TRACE_DEGREE = 1;
constexpr std::size_t MAX_TRACE_DEGREE = 4;
// The rule has d + 4 points per direction: exact on an affine tetrahedron for the mass of the bubbles (document 28,
// section 3.2); the integrands of a curved element are rational, and the results do not move with more points.
constexpr std::size_t RULE_EXTRA_POINTS = 4;
// The numbers of interior fields of a modal analysis: none, the answer to a uniform acceleration, to a linear one.
constexpr std::size_t NO_INTERIOR_FIELDS = 0;
constexpr std::size_t UNIFORM_INTERIOR_FIELDS = 3;
constexpr std::size_t LINEAR_INTERIOR_FIELDS = 12;
constexpr std::size_t COMPONENTS = 3;
constexpr std::size_t STRESS_COMPONENTS = 6;
// Engineering strains of the field "component c of a scalar function u": the direct strain c is u,c; the two shear
// strains that have c (γxy at 3, γyz at 4, γzx at 5) take the other two derivatives.
constexpr std::array<std::array<std::size_t, 2>, 3> SHEAR_ROWS{{{3, 5}, {3, 4}, {4, 5}}};
constexpr std::array<std::array<std::size_t, 2>, 3> SHEAR_DERIVATIVES{{{1, 2}, {0, 2}, {1, 0}}};
// The Voigt index of the stress σ_ab.
constexpr std::array<std::array<std::size_t, 3>, 3> VOIGT_INDEX{{{0, 3, 5}, {3, 1, 4}, {5, 4, 2}}};

// The points of the rule on the element: reference coordinates, weight (with |det J|), position, and the inverse of
// the Jacobian, d r_k / d x_m at (k, m).
// Reference: volume::Points
struct RulePoints
{
    std::vector<SolidPoint> reference;
    Vector weights;
    std::vector<SolidPoint> positions;
    std::vector<base::Mat3> inverseJacobians;
    double volume = 0.0;
};

// The point X(r) and its Jacobian J(m, k) = d x_m / d r_k.
struct PointJacobian
{
    SolidPoint position{};
    base::Mat3 derivatives;
};

// The derivatives along r0 and r1 come from one evaluation with two dual directions, along r2 from a second one.
// Reference: volume::jacobian
PointJacobian jacobianAt(const VolumeGeometry& geometry, const SolidPoint& r)
{
    const DualPoint a =
        geometry.at(DualPoint{DualNumber::variable(r[0], 0), DualNumber::variable(r[1], 1), DualNumber(r[2])});
    const DualPoint b = geometry.at(DualPoint{DualNumber(r[0]), DualNumber(r[1]), DualNumber::variable(r[2], 0)});
    PointJacobian out;
    for (std::size_t m = 0; m < COMPONENTS; ++m)
    {
        out.position[m] = a[m].value;
        out.derivatives(m, 0) = a[m].derivatives[0];
        out.derivatives(m, 1) = a[m].derivatives[1];
        out.derivatives(m, 2) = b[m].derivatives[0];
    }
    return out;
}

// Values (index 0) and derivatives (index 1 + k) at the points of a rule of the n functions that `evaluate` gives at a
// reference point: points × n each. tabulateReference gives the derivatives d/dr_k along the reference coordinates,
// toBasicGradients turns them into the gradients d/dx_m in the basic axes.
// Reference: volume::tabulate
using Tabulation = std::array<Matrix, 4>;

template <class Evaluate>
Tabulation tabulateReference(std::span<const VolumeRulePoint> rule, std::size_t n, const Evaluate& evaluate)
{
    const std::size_t count = rule.size();
    Tabulation out{Matrix(count, n), Matrix(count, n), Matrix(count, n), Matrix(count, n)};
    for (std::size_t g = 0; g < count; ++g)
    {
        const SolidPoint& r = rule[g].point;
        const std::vector<DualNumber> a =
            evaluate(DualPoint{DualNumber::variable(r[0], 0), DualNumber::variable(r[1], 1), DualNumber(r[2])});
        const std::vector<DualNumber> b =
            evaluate(DualPoint{DualNumber(r[0]), DualNumber(r[1]), DualNumber::variable(r[2], 0)});
        for (std::size_t j = 0; j < n; ++j)
        {
            out[0](g, j) = a[j].value;
            out[1](g, j) = a[j].derivatives[0];
            out[2](g, j) = a[j].derivatives[1];
            out[3](g, j) = b[j].derivatives[0];
        }
    }
    return out;
}

// The gradient d f / d x_m = Σ_k (d f / d r_k)(d r_k / d x_m) of every function at every point, in place.
void toBasicGradients(Tabulation& tabulation, const std::vector<base::Mat3>& inverseJacobians)
{
    for (std::size_t g = 0; g < inverseJacobians.size(); ++g)
    {
        const base::Mat3& inverse = inverseJacobians[g];
        for (std::size_t j = 0; j < tabulation[0].columns(); ++j)
        {
            const std::array<double, 3> dr{tabulation[1](g, j), tabulation[2](g, j), tabulation[3](g, j)};
            for (std::size_t m = 0; m < COMPONENTS; ++m)
                tabulation[1 + m](g, j) = dr[0] * inverse(0, m) + dr[1] * inverse(1, m) + dr[2] * inverse(2, m);
        }
    }
}

// The points of the rule on the element from the tabulation of its functions along the reference coordinates (before
// toBasicGradients): X(r) = Σ_j N_j(r) X_j and J(m, k) = Σ_j (d N_j / d r_k) X_j(m), the sums that
// VolumeGeometry::at makes, in the same order (so the same values as jacobianAt, without evaluating the functions
// again).
// Reference: volume::points
SolidResult<RulePoints> rulePoints(const VolumeGeometry& geometry, std::span<const VolumeRulePoint> rule,
                                   const Tabulation& functions)
{
    const std::string name = shapeName(geometry.shape);
    RulePoints points;
    double first = 0.0;
    for (std::size_t g = 0; g < rule.size(); ++g)
    {
        PointJacobian jacobian;
        for (std::size_t j = 0; j < functions[0].columns(); ++j)
            for (std::size_t m = 0; m < COMPONENTS; ++m)
            {
                const double c = geometry.coordinate(j, m);
                if (c == 0.0)
                    continue;
                jacobian.position[m] += functions[0](g, j) * c;
                for (std::size_t k = 0; k < COMPONENTS; ++k)
                    jacobian.derivatives(m, k) += functions[1 + k](g, j) * c;
            }
        const double det = base::determinant(jacobian.derivatives);
        if (!(std::fabs(det) > 0.0))
            return SolidResult<RulePoints>::refused("flat " + name);
        if (first == 0.0)
            first = det;
        if (det * first < 0.0)
            return SolidResult<RulePoints>::refused(
                name + ": the element folds (its corners in a wrong order, or its edges curved too far)");
        points.reference.push_back(rule[g].point);
        points.weights.push_back(rule[g].weight * std::fabs(det));
        points.positions.push_back(jacobian.position);
        points.inverseJacobians.push_back(base::inverse(jacobian.derivatives));
        points.volume += rule[g].weight * std::fabs(det);
    }
    return points;
}

// The scalar fields of the element at the points of its rule: its functions and its bubbles, values and gradients.
// Reference: volume::Fields
struct Fields
{
    RulePoints points;
    std::size_t functionCount = 0;       // ns
    std::size_t bubbleFunctionCount = 0; // nbs
    Tabulation functions;                // sh
    Tabulation bubbles;                  // bu
};

// Reference: volume::fields
// PERFORMANCE: the functions N = B·C are tabulated as four matrix products, B the basis of the space of degree p and
// its derivatives at the points (two evaluations with dual numbers per point), C their coefficients on it
// (VolumeGeometry::functionCoefficients): the sums VolumeGeometry::functions makes, in the same order. The Jacobian is
// made from their derivatives (rulePoints) instead of evaluating them again through VolumeGeometry::at. The same
// values. Measured: a kernel of a hexahedron of degree 4 (512 points), native, 51 → 39 ms; of a tetrahedron 9.2 →
// 6.5 ms.
SolidResult<Fields> tabulateFields(const VolumeGeometry& geometry, std::size_t d, std::size_t n)
{
    const VolumeShape shape = geometry.shape;
    const std::vector<VolumeRulePoint> rule = volumeRule(shape, n);
    Fields fields;
    fields.functionCount = geometry.functionCount();
    fields.bubbleFunctionCount = volumeSpaceDimension(shape, d - 1);
    const Tabulation basis = tabulateReference(rule, geometry.functionCoefficients.rows(), [&](const DualPoint& r)
                                               { return volumeSpace(shape, geometry.traceDegree, r); });
    for (std::size_t i = 0; i < basis.size(); ++i)
        fields.functions[i] = basis[i] * geometry.functionCoefficients;
    SolidResult<RulePoints> points = rulePoints(geometry, rule, fields.functions);
    if (!points.ok())
        return SolidResult<Fields>::refused(points.refusal());
    fields.points = std::move(points.value());
    toBasicGradients(fields.functions, fields.points.inverseJacobians);
    fields.bubbles = tabulateReference(rule, fields.bubbleFunctionCount,
                                       [&](const DualPoint& r)
                                       {
                                           const DualNumber factor = volumeBubble(shape, r);
                                           std::vector<DualNumber> values = volumeSpace(shape, d - 1, r);
                                           for (DualNumber& value : values)
                                               value = value * factor;
                                           return values;
                                       });
    toBasicGradients(fields.bubbles, fields.points.inverseJacobians);
    return fields;
}

// The values and gradients of the functions and of the bubbles side by side: points × (ns + nbs).
Matrix sideBySide(const Matrix& functions, const Matrix& bubbles)
{
    Matrix out(functions.rows(), functions.columns() + bubbles.columns());
    out.setBlock(0, 0, functions);
    out.setBlock(0, functions.columns(), bubbles);
    return out;
}

// The derivative a of the scalar functions (the functions first, the bubbles after them) projected on the stress
// basis: P_a = ∫ φᵀ ∂_a N (N × the scalars), weightedPhi the basis times the weights of the points.
std::array<Matrix, COMPONENTS> projectedGradients(const Fields& fields, const Matrix& weightedPhi)
{
    std::array<Matrix, COMPONENTS> projected;
    for (std::size_t a = 0; a < COMPONENTS; ++a)
        projected[a] = transposeTimes(weightedPhi, sideBySide(fields.functions[1 + a], fields.bubbles[1 + a]));
    return projected;
}

// Whether the engineering strain c reads the derivative a of the displacement component m: ε_m = ∂_m u_m, and each
// shear strain SHEAR_ROWS[m][s] reads ∂_SHEAR_DERIVATIVES[m][s] u_m.
constexpr bool strainReads(std::size_t c, std::size_t m, std::size_t a)
{
    if (c == m)
        return a == m;
    for (std::size_t s = 0; s < 2; ++s)
        if (SHEAR_ROWS[m][s] == c)
            return SHEAR_DERIVATIVES[m][s] == a;
    return false;
}

// E_c = ∫ φᵀ ε_c on every field (field 3 j + m is component m of scalar j, the functions first, the bubbles after
// them: N × 3 scalars each): the strain c of field (j, m) is the derivative a of the scalar j that it reads, so its
// projection is column j of P_a. Copies, no product: the six strains of the three components share three projections.
std::array<Matrix, STRESS_COMPONENTS> projectedStrains(const std::array<Matrix, COMPONENTS>& gradients)
{
    const std::size_t rows = gradients[0].rows();
    const std::size_t scalars = gradients[0].columns();
    std::array<Matrix, STRESS_COMPONENTS> projected;
    for (Matrix& strain : projected)
        strain = Matrix(rows, COMPONENTS * scalars);
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
        for (std::size_t m = 0; m < COMPONENTS; ++m)
            for (std::size_t a = 0; a < COMPONENTS; ++a)
            {
                if (!strainReads(c, m, a))
                    continue;
                for (std::size_t j = 0; j < scalars; ++j)
                    for (std::size_t i = 0; i < rows; ++i)
                        projected[c](i, COMPONENTS * j + m) = gradients[a](i, j);
            }
    return projected;
}

// The stiffness Σ_cc' D_cc' E_cᵀ E_c' on every field, from the Gram blocks of the projected gradients G_ab = P_aᵀ P_b
// (scalars × scalars, G_ba = G_abᵀ):
//   K(3 i + m, 3 j + n) = Σ_ab C(m, a; n, b) G_ab(i, j),  C(m, a; n, b) = Σ_cc' D_cc' [c reads a of m] [c' reads b of
//   n],
// which is E_cᵀ D E_c' term by term: nine products of scalars × N × scalars instead of thirty-six of three times that
// width on each side.
Matrix fieldStiffness(const std::array<Matrix, COMPONENTS>& gradients, const SolidMaterial& material)
{
    const std::size_t scalars = gradients[0].columns();
    std::array<std::array<Matrix, COMPONENTS>, COMPONENTS> gram;
    for (std::size_t a = 0; a < COMPONENTS; ++a)
        for (std::size_t b = a; b < COMPONENTS; ++b)
        {
            gram[a][b] = transposeTimes(gradients[a], gradients[b]);
            if (b != a)
                gram[b][a] = gram[a][b].transposed();
        }
    Matrix stiffness(COMPONENTS * scalars, COMPONENTS * scalars);
    for (std::size_t m = 0; m < COMPONENTS; ++m)
        for (std::size_t a = 0; a < COMPONENTS; ++a)
            for (std::size_t n = 0; n < COMPONENTS; ++n)
                for (std::size_t b = 0; b < COMPONENTS; ++b)
                {
                    double coefficient = 0.0;
                    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
                        for (std::size_t c2 = 0; c2 < STRESS_COMPONENTS; ++c2)
                            if (strainReads(c, m, a) && strainReads(c2, n, b))
                                coefficient += material.stiffness[c][c2];
                    if (coefficient == 0.0)
                        continue;
                    const Matrix& g = gram[a][b];
                    for (std::size_t i = 0; i < scalars; ++i)
                        for (std::size_t j = 0; j < scalars; ++j)
                            stiffness(COMPONENTS * i + m, COMPONENTS * j + n) += coefficient * g(i, j);
                }
    return stiffness;
}

// The stress basis at the points made orthonormal on the element: φ ← φ L⁻ᵀ, L Lᵀ its Gram matrix ∫ φᵀ φ dV.
struct StressBasis
{
    Matrix values;     // points × N
    Matrix gramFactor; // L
};

SolidResult<StressBasis> orthonormalStressBasis(const RulePoints& points, VolumeShape shape, std::size_t d)
{
    const std::size_t count = points.reference.size();
    const std::size_t n = volumeSpaceDimension(shape, d);
    Matrix phi(count, n);
    for (std::size_t g = 0; g < count; ++g)
        phi.setRow(g, volumeSpace(shape, d, points.reference[g]));
    const Matrix gram = transposeTimes(phi, weightedRows(phi, points.weights));
    std::optional<Matrix> lower = choleskyLower(symmetricPart(gram));
    if (!lower)
        return SolidResult<StressBasis>::refused("solid: the stress functions are not independent");
    Matrix t = phi.transposed();
    solveLower(*lower, t);
    return StressBasis{t.transposed(), std::move(*lower)};
}

// The work of the stress components D_c· ε: Σ_c' D_cc' E_c', the zero terms of the law skipped.
std::array<Matrix, STRESS_COMPONENTS> stressesOf(const SolidMaterial& material,
                                                 const std::array<Matrix, STRESS_COMPONENTS>& projected)
{
    std::array<Matrix, STRESS_COMPONENTS> out;
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
    {
        out[c] = Matrix(projected[0].rows(), projected[0].columns());
        for (std::size_t c2 = 0; c2 < STRESS_COMPONENTS; ++c2)
            if (material.stiffness[c][c2] != 0.0)
                out[c] += projected[c2] * material.stiffness[c][c2];
    }
    return out;
}

// Σ_{c' < 3} D_cc': the stress c of a unit free strain on the three direct strains.
double thermalStress(const SolidMaterial& material, std::size_t c)
{
    double sum = 0.0;
    for (std::size_t c2 = 0; c2 < COMPONENTS; ++c2)
        sum += material.stiffness[c][c2];
    return sum;
}

// The columns or rows [first, first + count) of m.
Matrix columnsOf(const Matrix& m, std::size_t first, std::size_t count)
{
    return m.block(0, first, m.rows(), count);
}
Matrix rowsOf(const Matrix& m, std::size_t first, std::size_t count)
{
    return m.block(first, 0, count, m.columns());
}

// What the elimination of the bubbles needs: the factor of K_bb and T = −K_bb⁻¹ K_bq.
struct BubbleElimination
{
    EquilibratedCholesky factor;
    Matrix operatorT; // n_bubble × nq
};

// m [I; T]: the columns of a matrix on all the fields reduced to the parameters, the bubbles following them.
Matrix reduceColumns(const Matrix& m, std::size_t nq, const Matrix& t)
{
    return columnsOf(m, 0, nq) + columnsOf(m, nq, t.rows()) * t;
}

// [I; T]ᵀ m: the rows reduced in the same way.
Matrix reduceRows(const Matrix& m, std::size_t nq, const Matrix& t)
{
    return rowsOf(m, 0, nq) + transposeTimes(t, rowsOf(m, nq, t.rows()));
}

// The load of one SolidLoad on all the fields (functions then bubbles), and the stress coefficients of its free strain
// (zero for a body force).
struct FieldLoad
{
    Vector onFields;   // nd
    Vector freeStress; // N: ∫ φ α ΔT dV
};

FieldLoad loadOnFields(const SolidLoad& load, const RulePoints& points, const Matrix& values, const Matrix& phi,
                       const std::array<Matrix, STRESS_COMPONENTS>& projected, const SolidMaterial& material)
{
    const std::size_t count = points.reference.size();
    FieldLoad out{Vector(COMPONENTS * values.columns(), 0.0), Vector(phi.columns(), 0.0)};
    if (load.kind == SolidLoad::Kind::Body)
    {
        for (std::size_t g = 0; g < count; ++g)
        {
            const SolidPoint force = load.force(points.positions[g]);
            for (std::size_t j = 0; j < values.columns(); ++j)
                for (std::size_t c = 0; c < COMPONENTS; ++c)
                    out.onFields[COMPONENTS * j + c] += points.weights[g] * force[c] * values(g, j);
        }
        return out;
    }
    for (std::size_t g = 0; g < count; ++g)
    {
        const double freeStrain = load.freeStrain(points.positions[g]);
        for (std::size_t i = 0; i < phi.columns(); ++i)
            out.freeStress[i] += points.weights[g] * freeStrain * phi(g, i);
    }
    // the work of the stresses of the free strain (1, 1, 1, 0, 0, 0) α ΔT in every field
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
    {
        const double s = thermalStress(material, c);
        if (s == 0.0)
            continue;
        const Vector work = transposeTimes(projected[c], out.freeStress);
        for (std::size_t k = 0; k < work.size(); ++k)
            out.onFields[k] += work[k] * s;
    }
    return out;
}

// The load on the parameters and the particular stresses: the bubbles take their part of the load, a = K_bb⁻¹ f_b,
// f = f_q + Tᵀ f_b, β = σ_b a − (D (1,1,1,0,0,0)) ∫ φ α ΔT.
SolidLoadResult reduceLoad(const FieldLoad& load, std::size_t nq, const BubbleElimination& bubbles,
                           const std::array<Matrix, STRESS_COMPONENTS>& stresses, const SolidMaterial& material)
{
    const std::size_t nbub = bubbles.operatorT.rows();
    const std::size_t n = load.freeStress.size();
    const Vector onBubbles(load.onFields.begin() + static_cast<std::ptrdiff_t>(nq), load.onFields.end());
    const Vector amplitudes = bubbles.factor.solve(onBubbles);
    SolidLoadResult out;
    out.force.assign(load.onFields.begin(), load.onFields.begin() + static_cast<std::ptrdiff_t>(nq));
    const Vector carried = transposeTimes(bubbles.operatorT, onBubbles);
    for (std::size_t i = 0; i < nq; ++i)
        out.force[i] += carried[i];
    out.particularStresses.assign(STRESS_COMPONENTS * n, 0.0);
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
    {
        const Vector part = columnsOf(stresses[c], nq, nbub) * amplitudes;
        const double s = thermalStress(material, c);
        for (std::size_t i = 0; i < n; ++i)
            out.particularStresses[c * n + i] = part[i] - s * load.freeStress[i];
    }
    return out;
}

// The mass of the fields: the scalar mass ρ ∫ N_i N_j dV, the same for the three components (nd × nd).
Matrix fieldMass(const Matrix& values, const Vector& weights, double density)
{
    const Matrix scalarMass = transposeTimes(values, weightedRows(values, weights)) * density;
    const std::size_t scalars = values.columns();
    Matrix out(COMPONENTS * scalars, COMPONENTS * scalars);
    for (std::size_t i = 0; i < scalars; ++i)
        for (std::size_t j = 0; j < scalars; ++j)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                out(COMPONENTS * i + c, COMPONENTS * j + c) = scalarMass(i, j);
    return out;
}

// The parameters of the displacements u_c = 1 (fields 0 … 2) and, for twelve fields, u_c = (x_m − x̄_m) / h (field
// 3 + 3 m + c): those of the geometry itself, the corners about their mean x̄, h the cube root of the volume.
Matrix interiorFieldLoads(const VolumeGeometry& geometry, std::size_t fieldCount, double volume)
{
    const std::size_t functionCount = geometry.functionCount();
    const std::size_t cornerCount = geometry.corners.size();
    const double h = std::cbrt(volume);
    SolidPoint mean{};
    for (std::size_t m = 0; m < COMPONENTS; ++m)
    {
        double sum = 0.0;
        for (std::size_t k = 0; k < cornerCount; ++k)
            sum += geometry.corners[k][m];
        mean[m] = sum / static_cast<double>(cornerCount);
    }
    Matrix out(COMPONENTS * functionCount, fieldCount);
    for (std::size_t j = 0; j < functionCount; ++j)
    {
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            if (j < cornerCount)
                out(COMPONENTS * j + c, c) = 1.0;
            if (fieldCount <= UNIFORM_INTERIOR_FIELDS)
                continue;
            for (std::size_t m = 0; m < COMPONENTS; ++m)
            {
                const double v = j < cornerCount ? geometry.coordinate(j, m) - mean[m] : geometry.coordinate(j, m);
                out(COMPONENTS * j + c, COMPONENTS + COMPONENTS * m + c) = v / h;
            }
        }
    }
    return out;
}

// The interior fields of a modal analysis (document 28, section 3.5). What the static elimination of the bubbles
// leaves out is their answer to the inertia forces. The fields kept are that answer, boundary held, when the element
// moves as a uniform displacement (3 fields) or as a linear one (12): K_bb v = (M_bq + M_bb T) q, q the parameters of
// these displacements. They are made orthonormal for the mass with a diagonal stiffness: with M_v = L Lᵀ,
// L⁻¹ K_v L⁻ᵀ = Z Λ Zᵀ (jacobiEigen) and the fields v L⁻ᵀ Z.
SolidResult<SolidInteriorFields> modalInteriorFields(const Matrix& reducedMassRows, const Matrix& fieldMassMatrix,
                                                     const Matrix& fieldStiffness, const BubbleElimination& bubbles,
                                                     const Matrix& loads, std::size_t nq)
{
    const std::size_t nbub = bubbles.operatorT.rows();
    const Matrix massBq = rowsOf(reducedMassRows, nq, nbub); // M_bq + M_bb T
    Matrix v = bubbles.factor.solve(massBq * loads);
    const Matrix massBb = fieldMassMatrix.block(nq, nq, nbub, nbub);
    const Matrix stiffnessBb = fieldStiffness.block(nq, nq, nbub, nbub);
    const Matrix initialMass = transposeTimes(v, massBb) * v;
    for (std::size_t c = 0; c < v.columns(); ++c)
        v.scaleColumn(c, 1.0 / std::sqrt(initialMass(c, c)));
    const Matrix massV = transposeTimes(v, massBb) * v;
    const Matrix stiffnessV = transposeTimes(v, stiffnessBb) * v;
    const std::optional<Matrix> lower = choleskyLower(symmetricPart(massV));
    if (!lower)
        return SolidResult<SolidInteriorFields>::refused("solid: the interior fields are not independent");
    Matrix a = stiffnessV;
    solveLower(*lower, a);
    a = a.transposed();
    solveLower(*lower, a);
    SymmetricEigen eigen = jacobiEigen(symmetricPart(a));
    solveLowerTransposed(*lower, eigen.vectors);
    const Matrix y = v * eigen.vectors;
    return SolidInteriorFields{std::move(eigen.values), transposeTimes(massBq, y)};
}

// The degrees and options a kernel accepts; an empty message when they are accepted.
std::string optionsRefusal(const VolumeGeometry& geometry, const SolidOptions& options)
{
    const std::size_t p = options.traceDegree;
    const std::size_t d = options.stressDegree;
    const std::size_t pf = std::min(options.faceDegree, options.traceDegree);
    if (p < MIN_TRACE_DEGREE || p > MAX_TRACE_DEGREE || d < p || d == 0)
        return "solid: trace degree 1 to 4, stress degree at least the trace degree";
    if (geometry.traceDegree != p || geometry.faceDegree != pf)
        return "solid: the geometry was built for other degrees";
    const std::size_t fields = options.interiorFieldCount;
    const bool knownCount =
        fields == NO_INTERIOR_FIELDS || fields == UNIFORM_INTERIOR_FIELDS || fields == LINEAR_INTERIOR_FIELDS;
    if (!knownCount || fields > COMPONENTS * volumeSpaceDimension(geometry.shape, d - 1))
        return "solid: 0, 3 or 12 interior fields, 12 with a stress degree of 2 at least";
    return {};
}

// The area vector of a face point: a1 × a2 times the weight, outward.
SolidPoint areaVector(const VolumeFacePoint& point)
{
    return scl3(cross3(point.tangentS, point.tangentT), point.weight * point.outward);
}

// True when every corner of `corners` is a corner of `face`.
bool hasCorners(const ShapeFace& face, std::span<const std::size_t> corners)
{
    const std::span<const std::size_t> own = face.cornerSpan();
    return std::all_of(corners.begin(), corners.end(),
                       [&](std::size_t k) { return std::find(own.begin(), own.end(), k) != own.end(); });
}

} // namespace

SolidResult<VolumeKernel> VolumeKernel::build(const VolumeGeometry& geometry, std::span<const base::EntityId> order,
                                              const SolidMaterial& material, const SolidOptions& options, bool withMass,
                                              std::span<const SolidLoad> loads)
{
    const std::string refusal = optionsRefusal(geometry, options);
    if (!refusal.empty())
        return SolidResult<VolumeKernel>::refused(refusal);
    const VolumeShape shape = geometry.shape;
    VolumeKernel kernel;
    kernel.m_traceDegree = options.traceDegree;
    kernel.m_faceDegree = std::min(options.faceDegree, options.traceDegree);
    kernel.m_stressDegree = options.stressDegree;
    kernel.m_pointsPerDirection = options.pointsPerDirection.value_or(options.stressDegree + RULE_EXTRA_POINTS);
    SolidResult<Fields> tabulated = tabulateFields(geometry, kernel.m_stressDegree, kernel.m_pointsPerDirection);
    if (!tabulated.ok())
        return SolidResult<VolumeKernel>::refused(tabulated.refusal());
    const Fields& fields = tabulated.value();
    const RulePoints& points = fields.points;
    const std::size_t nq = COMPONENTS * fields.functionCount;
    const std::size_t nbub = COMPONENTS * fields.bubbleFunctionCount;
    kernel.m_volume = points.volume;
    kernel.m_parameterCount = nq;
    kernel.m_bubbleCount = nbub;

    // the stress basis, made orthonormal on the element
    SolidResult<StressBasis> basis = orthonormalStressBasis(points, shape, kernel.m_stressDegree);
    if (!basis.ok())
        return SolidResult<VolumeKernel>::refused(basis.refusal());
    const Matrix& phi = basis.value().values;
    kernel.m_stressFunctionCount = phi.columns();
    kernel.m_stressGramFactor = basis.value().gramFactor;

    // E_c = ∫ φ ε_c, the stresses D E and the stiffness Eᵀ D E on all the fields
    const Matrix weightedPhi = weightedRows(phi, points.weights);
    const std::array<Matrix, COMPONENTS> gradients = projectedGradients(fields, weightedPhi);
    const std::array<Matrix, STRESS_COMPONENTS> projected = projectedStrains(gradients);
    const std::array<Matrix, STRESS_COMPONENTS> stresses = stressesOf(material, projected);
    const Matrix fullStiffness = symmetricPart(fieldStiffness(gradients, material));

    // the bubbles: K_bb = S⁻¹ L Lᵀ S⁻¹, T = −K_bb⁻¹ K_bq
    std::optional<EquilibratedCholesky> factor = EquilibratedCholesky::factor(fullStiffness.block(nq, nq, nbub, nbub));
    if (!factor)
        return SolidResult<VolumeKernel>::refused("solid: the bubbles are not independent");
    const Matrix stiffnessBq = fullStiffness.block(nq, 0, nbub, nq);
    BubbleElimination bubbles{std::move(*factor), Matrix()};
    bubbles.operatorT = -bubbles.factor.solve(stiffnessBq);
    kernel.m_stiffness =
        symmetricPart(fullStiffness.block(0, 0, nq, nq) + transposeTimes(stiffnessBq, bubbles.operatorT));
    kernel.m_stressOperator = Matrix(STRESS_COMPONENTS * kernel.m_stressFunctionCount, nq);
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
        kernel.m_stressOperator.setBlock(c * kernel.m_stressFunctionCount, 0,
                                         reduceColumns(stresses[c], nq, bubbles.operatorT));

    // loads
    const Matrix values = sideBySide(fields.functions[0], fields.bubbles[0]);
    for (const SolidLoad& load : loads)
    {
        const FieldLoad onFields = loadOnFields(load, points, values, phi, projected, material);
        kernel.m_loads.push_back(reduceLoad(onFields, nq, bubbles, stresses, material));
    }

    // the mass of the fields, the bubbles following statically, and the interior fields of a modal analysis
    if (withMass)
    {
        const Matrix massMatrix = fieldMass(values, points.weights, material.density);
        const Matrix reducedColumns = reduceColumns(massMatrix, nq, bubbles.operatorT);
        kernel.m_mass = symmetricPart(reduceRows(reducedColumns, nq, bubbles.operatorT));
        const std::size_t fieldCount = options.interiorFieldCount;
        if (fieldCount > 0 && material.density > 0.0)
        {
            const Matrix fieldLoads = interiorFieldLoads(geometry, fieldCount, kernel.m_volume);
            SolidResult<SolidInteriorFields> interior =
                modalInteriorFields(reducedColumns, massMatrix, fullStiffness, bubbles, fieldLoads, nq);
            if (!interior.ok())
                return SolidResult<VolumeKernel>::refused(interior.refusal());
            kernel.m_interiorFields = std::move(interior.value());
        }
    }
    kernel.m_bubbleOperator = std::move(bubbles.operatorT);
    kernel.m_geometry = geometry;
    kernel.m_order.assign(order.begin(), order.end());
    return kernel;
}

SolidResult<VolumeKernel> VolumeKernel::withMass(double density) const
{
    SolidResult<Fields> tabulated = tabulateFields(m_geometry, m_stressDegree, m_pointsPerDirection);
    if (!tabulated.ok())
        return SolidResult<VolumeKernel>::refused(tabulated.refusal());
    const Fields& fields = tabulated.value();
    // the steps of build() with the mass, on the same fields and the same T
    const Matrix values = sideBySide(fields.functions[0], fields.bubbles[0]);
    const Matrix massMatrix = fieldMass(values, fields.points.weights, density);
    const Matrix reducedColumns = reduceColumns(massMatrix, m_parameterCount, m_bubbleOperator);
    VolumeKernel out = *this;
    out.m_mass = symmetricPart(reduceRows(reducedColumns, m_parameterCount, m_bubbleOperator));
    return out;
}

Vector VolumeKernel::stressCoefficients(std::span<const double> q, std::span<const std::size_t> loadIndices) const
{
    Vector beta = m_stressOperator * q;
    for (std::size_t l : loadIndices)
        for (std::size_t i = 0; i < beta.size(); ++i)
            beta[i] += m_loads[l].particularStresses[i];
    return beta;
}

VoigtVector VolumeKernel::stressAt(std::span<const double> beta, const SolidPoint& r) const
{
    Vector y = volumeSpace(m_geometry.shape, m_stressDegree, r);
    const std::size_t n = y.size();
    solveLower(m_stressGramFactor, y);
    VoigtVector out{};
    for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum += beta[c * n + i] * y[i];
        out[c] = sum;
    }
    return out;
}

VoigtVector VolumeKernel::stressAtBarycentric(std::span<const double> beta, const std::array<double, 4>& l) const
{
    OOS_ASSERT(m_geometry.shape == VolumeShape::Tetrahedron);
    return stressAt(beta, {2.0 * l[1] - 1.0, 2.0 * l[2] - 1.0, 2.0 * l[3] - 1.0});
}

std::vector<double> VolumeKernel::functionsAt(const SolidPoint& r) const
{
    return m_geometry.functions(r);
}

std::vector<SolidPoint> VolumeKernel::gradients(const SolidPoint& r) const
{
    const base::Mat3 inverse = base::inverse(jacobianAt(m_geometry, r).derivatives);
    const std::vector<DualNumber> a =
        m_geometry.functions(DualPoint{DualNumber::variable(r[0], 0), DualNumber::variable(r[1], 1), DualNumber(r[2])});
    const std::vector<DualNumber> b =
        m_geometry.functions(DualPoint{DualNumber(r[0]), DualNumber(r[1]), DualNumber::variable(r[2], 0)});
    std::vector<SolidPoint> out(a.size());
    for (std::size_t j = 0; j < a.size(); ++j)
    {
        const std::array<double, 3> dr{a[j].derivatives[0], a[j].derivatives[1], b[j].derivatives[0]};
        for (std::size_t m = 0; m < COMPONENTS; ++m)
            out[j][m] = dr[0] * inverse(0, m) + dr[1] * inverse(1, m) + dr[2] * inverse(2, m);
    }
    return out;
}

SolidResult<Matrix> VolumeKernel::geometric(std::span<const double> beta) const
{
    SolidResult<Fields> tabulated = tabulateFields(m_geometry, m_stressDegree, m_pointsPerDirection);
    if (!tabulated.ok())
        return SolidResult<Matrix>::refused(tabulated.refusal());
    const Fields& fields = tabulated.value();
    const RulePoints& points = fields.points;
    const std::size_t count = points.reference.size();
    const std::size_t scalars = fields.functionCount + fields.bubbleFunctionCount;
    // the gradients of the scalar fields, functions then bubbles
    std::array<Matrix, COMPONENTS> gradient;
    for (std::size_t m = 0; m < COMPONENTS; ++m)
        gradient[m] = sideBySide(fields.functions[1 + m], fields.bubbles[1 + m]);
    std::vector<VoigtVector> stress(count);
    for (std::size_t g = 0; g < count; ++g)
        stress[g] = stressAt(beta, points.reference[g]);
    // G_ij = Σ_ab ∫ σ_ab N_i,a N_j,b, the same for the three components
    Matrix scalarG(scalars, scalars);
    Vector weights(count);
    for (std::size_t a = 0; a < COMPONENTS; ++a)
    {
        for (std::size_t b = 0; b < COMPONENTS; ++b)
        {
            for (std::size_t g = 0; g < count; ++g)
                weights[g] = points.weights[g] * stress[g][VOIGT_INDEX[a][b]];
            scalarG += transposeTimes(gradient[a], weightedRows(gradient[b], weights));
        }
    }
    const std::size_t nq = m_parameterCount;
    Matrix full(nq + m_bubbleCount, nq + m_bubbleCount);
    for (std::size_t i = 0; i < scalars; ++i)
        for (std::size_t j = 0; j < scalars; ++j)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                full(COMPONENTS * i + c, COMPONENTS * j + c) = scalarG(i, j);
    const Matrix reduced = reduceColumns(full, nq, m_bubbleOperator);
    return symmetricPart(reduceRows(reduced, nq, m_bubbleOperator));
}

SolidResult<Vector> VolumeKernel::geometricGradient(std::span<const double> phi) const
{
    OOS_ASSERT(phi.size() >= m_parameterCount);
    SolidResult<Fields> tabulated = tabulateFields(m_geometry, m_stressDegree, m_pointsPerDirection);
    if (!tabulated.ok())
        return SolidResult<Vector>::refused(tabulated.refusal());
    const Fields& fields = tabulated.value();
    const RulePoints& points = fields.points;
    const std::size_t count = points.reference.size();
    const std::size_t scalars = fields.functionCount + fields.bubbleFunctionCount;
    // ψ = [I; T] φ: the parameters, then the bubbles they carry (as in geometric()).
    Vector psi(phi.begin(), phi.begin() + static_cast<std::ptrdiff_t>(m_parameterCount));
    const Vector bubbles = m_bubbleOperator * phi.first(m_parameterCount);
    psi.insert(psi.end(), bubbles.begin(), bubbles.end());
    // the gradients of the scalar fields, functions then bubbles
    std::array<Matrix, COMPONENTS> gradient;
    for (std::size_t a = 0; a < COMPONENTS; ++a)
        gradient[a] = sideBySide(fields.functions[1 + a], fields.bubbles[1 + a]);
    // d[c][a] at each point: ∂ψ_c/∂x_a = Σ_i N_i,a ψ[3i + c].
    std::array<std::array<Vector, COMPONENTS>, COMPONENTS> d;
    for (std::size_t c = 0; c < COMPONENTS; ++c)
    {
        Vector component(scalars);
        for (std::size_t i = 0; i < scalars; ++i)
            component[i] = psi[COMPONENTS * i + c];
        for (std::size_t a = 0; a < COMPONENTS; ++a)
            d[c][a] = gradient[a] * component;
    }
    // φᵀ K_g(β) φ = Σ_g w_g Σ_ab σ_ab(g) Σ_c d[c][a] d[c][b], σ_v(g) = Σ_i β[v N + i] y_i(g) (stressAt): its derivative
    // with respect to β[v N + i] is Σ_g w_g y_i(g) Σ_{(a,b) of v} Σ_c d[c][a] d[c][b].
    const std::size_t n = volumeSpaceDimension(m_geometry.shape, m_stressDegree);
    Vector onStresses(STRESS_COMPONENTS * n, 0.0);
    for (std::size_t g = 0; g < count; ++g)
    {
        Vector y = volumeSpace(m_geometry.shape, m_stressDegree, points.reference[g]);
        solveLower(m_stressGramFactor, y);
        VoigtVector products{};
        for (std::size_t a = 0; a < COMPONENTS; ++a)
            for (std::size_t b = 0; b < COMPONENTS; ++b)
                for (std::size_t c = 0; c < COMPONENTS; ++c)
                    products[VOIGT_INDEX[a][b]] += d[c][a][g] * d[c][b][g];
        for (std::size_t v = 0; v < STRESS_COMPONENTS; ++v)
            for (std::size_t i = 0; i < n; ++i)
                onStresses[v * n + i] += points.weights[g] * products[v] * y[i];
    }
    // β = stressOperator q (the particular stresses of the loads do not depend on q).
    return transposeTimes(m_stressOperator, onStresses);
}

std::optional<std::size_t> VolumeKernel::faceOf(std::span<const std::size_t> corners) const
{
    const std::span<const ShapeFace> faces = shapeFaces(m_geometry.shape);
    for (std::size_t lf = 0; lf < faces.size(); ++lf)
        if (hasCorners(faces[lf], corners))
            return lf;
    return std::nullopt;
}

std::vector<VolumeFaceAreaPoint> VolumeKernel::facePoints(const std::array<std::size_t, 3>& corners,
                                                          std::size_t n) const
{
    std::vector<VolumeFaceAreaPoint> out;
    const std::optional<std::size_t> face = faceOf(corners);
    if (!face)
        return out;
    for (const VolumeFacePoint& point : facePointsOf(*face, n))
        out.push_back({point.reference, point.position,
                       scl3(cross3(point.tangentS, point.tangentT), point.weight * point.outward)});
    return out;
}

std::vector<VolumeFacePoint> VolumeKernel::facePointsOf(std::size_t face, std::size_t n) const
{
    const VolumeShape shape = m_geometry.shape;
    const std::span<const std::size_t> corners = shapeFaces(shape)[face].cornerSpan();
    const SolidPoint inside = m_geometry.at(shapeCentre(shape));
    std::vector<VolumeFacePoint> out;
    for (const SurfaceRulePoint& rule : faceRule(corners.size(), n))
    {
        const DualPoint r = facePoint(
            shape, corners,
            std::array<DualNumber, 2>{DualNumber::variable(rule.point[0], 0), DualNumber::variable(rule.point[1], 1)});
        const DualPoint x = m_geometry.at(r);
        VolumeFacePoint point;
        for (std::size_t m = 0; m < COMPONENTS; ++m)
        {
            point.reference[m] = r[m].value;
            point.position[m] = x[m].value;
            point.tangentS[m] = x[m].derivatives[0];
            point.tangentT[m] = x[m].derivatives[1];
        }
        point.weight = rule.weight;
        point.outward = dot3(cross3(point.tangentS, point.tangentT), sub3(point.position, inside)) < 0.0 ? -1.0 : 1.0;
        out.push_back(point);
    }
    return out;
}

std::vector<Dual<double>> VolumeKernel::faceFunctionsWithDerivatives(std::size_t face,
                                                                     const std::array<double, 2>& c) const
{
    const VolumeShape shape = m_geometry.shape;
    const std::span<const std::size_t> corners = shapeFaces(shape)[face].cornerSpan();
    const std::array<DualNumber, 2> faceCoordinates{DualNumber::variable(c[0], 0), DualNumber::variable(c[1], 1)};
    return m_geometry.functions(facePoint(shape, corners, faceCoordinates));
}

Vector VolumeKernel::faceLoad(std::size_t face, std::size_t n,
                              const std::function<SolidPoint(const VolumeFacePoint&)>& tractionAt) const
{
    Vector out(m_parameterCount, 0.0);
    for (const VolumeFacePoint& point : facePointsOf(face, n))
    {
        const SolidPoint t = tractionAt(point);
        const std::vector<double> values = functionsAt(point.reference);
        for (std::size_t j = 0; j < values.size(); ++j)
        {
            if (values[j] == 0.0)
                continue;
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                out[COMPONENTS * j + c] += values[j] * t[c];
        }
    }
    return out;
}

Vector VolumeKernel::faceTractionLoad(std::size_t face, const TractionField& traction, std::size_t n) const
{
    return faceLoad(face, n, [&](const VolumeFacePoint& point)
                    { return scl3(traction(point.position), norm3(areaVector(point))); });
}

Vector VolumeKernel::faceStressLoad(std::size_t face, const StressField& stress, std::size_t n) const
{
    return faceLoad(face, n,
                    [&](const VolumeFacePoint& point)
                    {
                        const VoigtVector s = stress(point.position);
                        const SolidPoint a = areaVector(point);
                        return SolidPoint{s[0] * a[0] + s[3] * a[1] + s[5] * a[2],
                                          s[3] * a[0] + s[1] * a[1] + s[4] * a[2],
                                          s[5] * a[0] + s[4] * a[1] + s[2] * a[2]};
                    });
}

Vector VolumeKernel::facePressureLoad(std::size_t face, const PressureField& pressure, std::size_t n) const
{
    return faceLoad(face, n,
                    [&](const VolumeFacePoint& point) { return scl3(areaVector(point), -pressure(point.position)); });
}

Matrix VolumeKernel::followerPressureStiffness(std::size_t face, const PressureField& pressure, std::size_t n) const
{
    const VolumeShape shape = m_geometry.shape;
    const std::span<const std::size_t> corners = shapeFaces(shape)[face].cornerSpan();
    const SolidPoint inside = m_geometry.at(shapeCentre(shape));
    Matrix loadStiffness(m_parameterCount, m_parameterCount);
    for (const SurfaceRulePoint& rule : faceRule(corners.size(), n))
    {
        const DualPoint r = facePoint(
            shape, corners,
            std::array<DualNumber, 2>{DualNumber::variable(rule.point[0], 0), DualNumber::variable(rule.point[1], 1)});
        const DualPoint x = m_geometry.at(r);
        SolidPoint a1{};
        SolidPoint a2{};
        SolidPoint position{};
        for (std::size_t m = 0; m < COMPONENTS; ++m)
        {
            a1[m] = x[m].derivatives[0];
            a2[m] = x[m].derivatives[1];
            position[m] = x[m].value;
        }
        const double outward = dot3(cross3(a1, a2), sub3(position, inside)) < 0.0 ? -1.0 : 1.0;
        const std::vector<DualNumber> functions = m_geometry.functions(r);
        const double amplitude = rule.weight * outward * pressure(position);
        for (std::size_t j = 0; j < functions.size(); ++j)
        {
            if (functions[j].value == 0.0)
                continue;
            for (std::size_t k = 0; k < functions.size(); ++k)
            {
                for (std::size_t d = 0; d < COMPONENTS; ++d)
                {
                    SolidPoint e{};
                    e[d] = 1.0;
                    // the change of a1 × a2 when the face moves by N_k e_d: N_k,1 e × a2 + N_k,2 a1 × e
                    const SolidPoint v = add3(scl3(cross3(e, a2), functions[k].derivatives[0]),
                                              scl3(cross3(a1, e), functions[k].derivatives[1]));
                    for (std::size_t c = 0; c < COMPONENTS; ++c)
                        loadStiffness(COMPONENTS * j + c, COMPONENTS * k + d) += amplitude * functions[j].value * v[c];
                }
            }
        }
    }
    return symmetricPart(loadStiffness);
}

} // namespace oos::elements::mixedHybrid
