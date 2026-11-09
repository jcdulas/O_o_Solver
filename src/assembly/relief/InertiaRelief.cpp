#include "oos/assembly/relief/InertiaRelief.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/ids/ComponentCode.hpp"
#include "oos/base/math/Mat3.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace oos::assembly
{

namespace
{

// A Cholesky pivot of Dᵀ·M·D at most this fraction of its largest diagonal is zero: a support DOF whose rigid motion
// moves (almost) no mass, so that the acceleration of the supports is not defined.
constexpr double MASSLESS_SUPPORT_RATIO = 1.0e-12;
// A row of D whose remaining norm is at most this fraction of the largest row norm adds no direction (the automatic
// supports of INREL = −2).
constexpr double DEPENDENT_SUPPORT_RATIO = 1.0e-8;

base::Diagnostic supportProblem(std::string code, std::string_view card, base::EntityId id, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = id;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// A Gauss-Jordan pivot of Φ_S at most this fraction of the largest entry is zero (rigidSupportCoefficients).
constexpr double SINGULAR_SUPPORT_RATIO = 1.0e-12;

} // namespace

std::optional<std::array<double, MAX_SUPPORT_DOFS * MAX_SUPPORT_DOFS>> rigidSupportCoefficients(
    std::span<const double> phiAtSupports, std::size_t supportCount)
{
    constexpr std::size_t N = MAX_SUPPORT_DOFS;
    if (supportCount != N || phiAtSupports.size() != N * N)
        return std::nullopt;
    // Φ_S·C = I: [Φ_S | I] reduced to [I | C] (rows j the supports, columns a the rigid motions).
    std::array<double, N * 2 * N> augmented{};
    double largest = 0.0;
    for (std::size_t j = 0; j < N; ++j)
    {
        for (std::size_t a = 0; a < N; ++a)
        {
            augmented[j * 2 * N + a] = phiAtSupports[j * N + a];
            largest = std::max(largest, std::abs(phiAtSupports[j * N + a]));
        }
        augmented[j * 2 * N + N + j] = 1.0;
    }
    if (!(largest > 0.0))
        return std::nullopt;
    for (std::size_t column = 0; column < N; ++column)
    {
        std::size_t pivotRow = column;
        for (std::size_t row = column + 1; row < N; ++row)
            if (std::abs(augmented[row * 2 * N + column]) > std::abs(augmented[pivotRow * 2 * N + column]))
                pivotRow = row;
        const double pivot = augmented[pivotRow * 2 * N + column];
        if (std::abs(pivot) <= SINGULAR_SUPPORT_RATIO * largest)
            return std::nullopt;
        for (std::size_t k = 0; k < 2 * N; ++k)
            std::swap(augmented[column * 2 * N + k], augmented[pivotRow * 2 * N + k]);
        for (std::size_t k = 0; k < 2 * N; ++k)
            augmented[column * 2 * N + k] /= pivot;
        for (std::size_t row = 0; row < N; ++row)
        {
            if (row == column)
                continue;
            const double factor = augmented[row * 2 * N + column];
            for (std::size_t k = 0; k < 2 * N; ++k)
                augmented[row * 2 * N + k] -= factor * augmented[column * 2 * N + k];
        }
    }
    // The rows of the reduced system are the rigid motions a, its right half's columns the supports j: C(a, j).
    std::array<double, N * N> coefficients{};
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t a = 0; a < N; ++a)
            coefficients[j * N + a] = augmented[a * 2 * N + N + j];
    return coefficients;
}

std::vector<base::DofIndex> supportDofsOf(const input::ModelSnapshot& snapshot, const DofMap& dofMap,
                                          base::EntityId subcase, base::DiagnosticSink& sink)
{
    std::vector<base::DofIndex> dofs; // bounded by the FreeBodySupport points (a few)
    if (!snapshot.hasTable(input::TableRole::Supports))
        return dofs;
    const base::ContractReader& table = snapshot.supports();
    const base::DictionaryView cards = table.dictionaryTexts("card");
    const std::span<const std::int32_t> nodeIds = table.values<std::int32_t>("nodeId");
    const base::Utf8View components = table.texts("dof");
    // dofSetsTable: subcases is null for a FreeBodySupport (every subcase takes it).
    const base::ListView<std::int32_t> selections = table.lists<std::int32_t>("subcases", base::NullPolicy::Allow);
    const base::ValidityView hasSelection = table.validity("subcases");
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto index = static_cast<std::size_t>(row);
        const std::string_view card = cards[row];
        if (card == "FreeBodySupportSet")
        {
            const std::span<const std::int32_t> subcases =
                hasSelection.isValid(row) ? selections[row] : std::span<const std::int32_t>();
            if (std::find(subcases.begin(), subcases.end(), subcase) == subcases.end())
                continue;
        }
        else if (card != "FreeBodySupport")
        {
            continue; // the other DOF set cards of dofSetsTable (ASET, OmittedDofSet, ...): not the r set
        }
        const std::optional<input::PointRef> point = dofMap.points().find(nodeIds[index]);
        base::ComponentCode code;
        try
        {
            code = base::ComponentCode::parse(components[row]);
        }
        catch (const std::invalid_argument&)
        {
            code = base::ComponentCode();
        }
        if (!point)
        {
            sink.report(
                supportProblem("support-unknown-point", card, nodeIds[index],
                               base::concatenate({"expected a grid or scalar point of the model, found ID ",
                                                  std::to_string(nodeIds[index]), "; correct the ", card, " card"})));
            continue;
        }
        if (point->kind == input::PointKind::Scalar)
        {
            dofs.push_back(dofMap.scalarDof(point->row));
            continue;
        }
        if (code.isEmpty())
        {
            sink.report(supportProblem(
                "support-bad-component", card, nodeIds[index],
                base::concatenate({"expected components 1 to 6 of grid point ", std::to_string(nodeIds[index]),
                                   ", found '", components[row], "'; correct the ", card, " card"})));
            continue;
        }
        for (int component = 1; component <= 6; ++component)
            if (code.contains(component))
                dofs.push_back(dofMap.gridDof(point->row, component));
    }
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    return dofs;
}

std::vector<base::DofIndex> chooseAutomaticSupports(const RigidBodyModes& modes, const DofSets& sets)
{
    const linalg::DenseMatrix& d = modes.modes();
    constexpr std::size_t N = RigidBodyModes::MODE_COUNT;
    std::vector<std::array<double, N>> basis; // the orthonormal directions of the rows taken: at most six
    std::vector<base::DofIndex> chosen;       // at most six
    double largestNorm = 0.0;
    for (std::size_t dof = 0; dof < d.rows(); ++dof)
    {
        double norm = 0.0;
        for (std::size_t k = 0; k < N; ++k)
            norm += d(dof, k) * d(dof, k);
        largestNorm = std::max(largestNorm, std::sqrt(norm));
    }
    for (std::size_t step = 0; step < N; ++step)
    {
        double best = DEPENDENT_SUPPORT_RATIO * largestNorm;
        std::size_t bestDof = d.rows();
        std::array<double, N> bestRow{};
        for (std::size_t dof = 0; dof < d.rows(); ++dof)
        {
            if (sets.exclusiveSetOf(static_cast<base::DofIndex>(dof)) != ExclusiveSet::B)
                continue;
            std::array<double, N> row{};
            for (std::size_t k = 0; k < N; ++k)
                row[k] = d(dof, k);
            for (const std::array<double, N>& q : basis)
            {
                double dot = 0.0;
                for (std::size_t k = 0; k < N; ++k)
                    dot += q[k] * row[k];
                for (std::size_t k = 0; k < N; ++k)
                    row[k] -= dot * q[k];
            }
            double norm = 0.0;
            for (std::size_t k = 0; k < N; ++k)
                norm += row[k] * row[k];
            norm = std::sqrt(norm);
            if (norm > best)
            {
                best = norm;
                bestDof = dof;
                bestRow = row;
            }
        }
        if (bestDof == d.rows())
            break;
        for (double& value : bestRow)
            value /= best;
        basis.push_back(bestRow);
        chosen.push_back(static_cast<base::DofIndex>(bestDof));
    }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

std::vector<base::DofIndex> chooseConditionedSupports(const DofMap& dofMap, std::span<const std::uint8_t> isCandidate)
{
    constexpr std::size_t N = RigidBodyModes::MODE_COUNT;
    constexpr std::size_t TRANSLATIONS = 3;
    if (isCandidate.size() != static_cast<std::size_t>(dofMap.gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"chooseConditionedSupports: expected ", std::to_string(dofMap.gSetSize()),
                               " candidate flags, found ", std::to_string(isCandidate.size())}));
    // The candidates: the translations of grid points.
    std::vector<base::DofIndex> candidates; // bounded by the g set
    base::Vec3 centroid{};
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
    {
        bool hasCandidate = false;
        for (int component = 1; component <= static_cast<int>(TRANSLATIONS); ++component)
        {
            const base::DofIndex dof = dofMap.gridDof(row, component);
            if (isCandidate[static_cast<std::size_t>(dof)] == 0)
                continue;
            candidates.push_back(dof);
            hasCandidate = true;
        }
        if (hasCandidate)
            centroid = centroid + dofMap.positionInBasic(row);
    }
    if (candidates.empty())
        return {};
    std::size_t pointCount = 0;
    std::int64_t lastRow = -1;
    for (const base::DofIndex dof : candidates)
        if (dof / 6 != lastRow)
        {
            lastRow = dof / 6;
            ++pointCount;
        }
    centroid = centroid * (1.0 / static_cast<double>(pointCount));
    double spread = 0.0;
    lastRow = -1;
    for (const base::DofIndex dof : candidates)
        if (dof / 6 != lastRow)
        {
            lastRow = dof / 6;
            const base::Vec3 offset = dofMap.positionInBasic(static_cast<std::int32_t>(lastRow)) - centroid;
            spread += base::dot(offset, offset);
        }
    spread = std::sqrt(spread / static_cast<double>(pointCount));
    const double length = spread > 0.0 ? spread : 1.0;
    // The rows: u_cd = Rᵀ·(t + θ × (x − c)/L) of the translational component of each candidate.
    std::vector<std::array<double, N>> rows(candidates.size()); // bounded by the g set
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        const auto row = static_cast<std::int32_t>(candidates[i] / 6);
        const auto component = static_cast<std::size_t>(candidates[i] % 6);
        const base::Mat3 toBasic = dofMap.cdToBasic(row);
        const base::Vec3 offset = (dofMap.positionInBasic(row) - centroid) * (1.0 / length);
        for (std::size_t mode = 0; mode < N; ++mode)
        {
            base::Vec3 motion{};
            if (mode < TRANSLATIONS)
                motion[mode] = 1.0;
            else
            {
                base::Vec3 rotation{};
                rotation[mode - TRANSLATIONS] = 1.0;
                motion = base::cross(rotation, offset);
            }
            // (Rᵀ·motion)_component = Σ_k R(k, component)·motion_k.
            double value = 0.0;
            for (std::size_t k = 0; k < TRANSLATIONS; ++k)
                value += toBasic(k, component) * motion[k];
            rows[i][mode] = value;
        }
    }
    double largestNorm = 0.0;
    for (const std::array<double, N>& row : rows)
    {
        double norm = 0.0;
        for (const double value : row)
            norm += value * value;
        largestNorm = std::max(largestNorm, std::sqrt(norm));
    }
    std::vector<base::DofIndex> chosen; // at most six
    for (std::size_t step = 0; step < N; ++step)
    {
        double best = DEPENDENT_SUPPORT_RATIO * largestNorm;
        std::size_t bestIndex = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            double norm = 0.0;
            for (const double value : rows[i])
                norm += value * value;
            norm = std::sqrt(norm);
            if (norm > best)
            {
                best = norm;
                bestIndex = i;
            }
        }
        if (bestIndex == rows.size())
            break;
        std::array<double, N> direction = rows[bestIndex];
        for (double& value : direction)
            value /= best;
        // The rows less their part along the direction taken (the next step compares what is left).
        for (std::array<double, N>& row : rows)
        {
            double dot = 0.0;
            for (std::size_t k = 0; k < N; ++k)
                dot += direction[k] * row[k];
            for (std::size_t k = 0; k < N; ++k)
                row[k] -= dot * direction[k];
        }
        chosen.push_back(candidates[bestIndex]);
    }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

namespace
{

// The rigid motion of `amplitudes` added at the grid point of row `row` (addRigidMotion's arithmetic, one point).
void addRigidMotionOfRow(const DofMap& dofMap, const std::array<double, RigidBodyModes::MODE_COUNT>& amplitudes,
                         std::span<double> valuesG, std::int32_t row, std::span<const std::uint8_t> isHeld)
{
    const base::Vec3 translation{amplitudes[0], amplitudes[1], amplitudes[2]};
    const base::Vec3 rotation{amplitudes[3], amplitudes[4], amplitudes[5]};
    const base::Mat3 toCd = dofMap.cdToBasic(row).transposed();
    const base::Vec3 u = toCd * (translation + base::cross(rotation, dofMap.positionInBasic(row)));
    const base::Vec3 theta = toCd * rotation;
    for (int k = 0; k < 3; ++k)
    {
        const auto translationDof = static_cast<std::size_t>(dofMap.gridDof(row, k + 1));
        const auto rotationDof = static_cast<std::size_t>(dofMap.gridDof(row, k + 4));
        if (isHeld.empty() || isHeld[translationDof] == 0)
            valuesG[translationDof] += u[static_cast<std::size_t>(k)];
        if (isHeld.empty() || isHeld[rotationDof] == 0)
            valuesG[rotationDof] += theta[static_cast<std::size_t>(k)];
    }
}

} // namespace

void addRigidMotion(const DofMap& dofMap, const std::array<double, RigidBodyModes::MODE_COUNT>& amplitudes,
                    std::span<double> valuesG, std::span<const std::uint8_t> isHeld)
{
    if (valuesG.size() != static_cast<std::size_t>(dofMap.gSetSize()) ||
        (!isHeld.empty() && isHeld.size() != valuesG.size()))
        throw std::invalid_argument(base::concatenate({"addRigidMotion: expected ", std::to_string(dofMap.gSetSize()),
                                                       " values (the g set), found ", std::to_string(valuesG.size())}));
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
        addRigidMotionOfRow(dofMap, amplitudes, valuesG, row, isHeld);
}

void addRigidMotionAt(const DofMap& dofMap, const std::array<double, RigidBodyModes::MODE_COUNT>& amplitudes,
                      std::span<double> valuesG, std::span<const std::int32_t> gridRows,
                      std::span<const std::uint8_t> isHeld)
{
    if (valuesG.size() != static_cast<std::size_t>(dofMap.gSetSize()) ||
        (!isHeld.empty() && isHeld.size() != valuesG.size()))
        throw std::invalid_argument(base::concatenate({"addRigidMotionAt: expected ", std::to_string(dofMap.gSetSize()),
                                                       " values (the g set), found ", std::to_string(valuesG.size())}));
    for (const std::int32_t row : gridRows)
    {
        if (row < 0 || row >= dofMap.gridCount())
            throw std::invalid_argument(
                base::concatenate({"addRigidMotionAt: expected grid rows below ", std::to_string(dofMap.gridCount()),
                                   ", found ", std::to_string(row)}));
        addRigidMotionOfRow(dofMap, amplitudes, valuesG, row, isHeld);
    }
}

std::array<double, RigidBodyModes::MODE_COUNT> rigidResultantOf(const DofMap& dofMap, std::span<const double> valuesG)
{
    if (valuesG.size() != static_cast<std::size_t>(dofMap.gSetSize()))
        throw std::invalid_argument(base::concatenate({"rigidResultantOf: expected ", std::to_string(dofMap.gSetSize()),
                                                       " values (the g set), found ", std::to_string(valuesG.size())}));
    std::array<double, RigidBodyModes::MODE_COUNT> resultant{};
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
    {
        base::Vec3 force{};
        base::Vec3 moment{};
        for (int k = 0; k < 3; ++k)
        {
            force[static_cast<std::size_t>(k)] = valuesG[static_cast<std::size_t>(dofMap.gridDof(row, k + 1))];
            moment[static_cast<std::size_t>(k)] = valuesG[static_cast<std::size_t>(dofMap.gridDof(row, k + 4))];
        }
        if (force == base::Vec3{} && moment == base::Vec3{})
            continue;
        const base::Mat3 toBasic = dofMap.cdToBasic(row);
        const base::Vec3 basicForce = toBasic * force;
        const base::Vec3 basicMoment = base::cross(dofMap.positionInBasic(row), basicForce) + toBasic * moment;
        for (std::size_t k = 0; k < 3; ++k)
        {
            resultant[k] += basicForce[k];
            resultant[3 + k] += basicMoment[k];
        }
    }
    return resultant;
}

InertiaRelief::InertiaRelief(linalg::DenseMatrix supportModes, const linalg::BlockSparseMatrix& mass,
                             base::MemoryBudget& budget)
    : m_mass(&mass), m_modes(std::move(supportModes))
{
    const std::size_t r = m_modes.columns();
    const std::size_t g = m_modes.rows();
    if (r > MAX_SUPPORT_DOFS)
        throw std::invalid_argument(
            base::concatenate({"InertiaRelief: expected at most ", std::to_string(MAX_SUPPORT_DOFS),
                               " support DOFs, found ", std::to_string(r)}));
    m_massModes = linalg::DenseMatrix(budget, g, r, "inertia relief M D");
    base::BudgetedBuffer<double> column(budget, g, "inertia relief mode");
    base::BudgetedBuffer<double> product(budget, g, "inertia relief M mode");
    for (std::size_t j = 0; j < r; ++j)
    {
        for (std::size_t dof = 0; dof < g; ++dof)
            column[dof] = m_modes(dof, j);
        mass.multiply(column.span(), product.span());
        for (std::size_t dof = 0; dof < g; ++dof)
            m_massModes(dof, j) = product[dof];
    }
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < r; ++j)
        {
            double sum = 0.0;
            for (std::size_t dof = 0; dof < g; ++dof)
                sum += m_modes(dof, i) * m_massModes(dof, j);
            m_reducedMass(i, j) = sum;
        }
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = i + 1; j < r; ++j)
            m_reducedMass(i, j) = m_reducedMass(j, i) = 0.5 * (m_reducedMass(i, j) + m_reducedMass(j, i));
    // Cholesky of the symmetric positive definite Dᵀ·M·D (Golub and Van Loan, Algorithm 4.2.1).
    double largest = 0.0;
    for (std::size_t i = 0; i < r; ++i)
        largest = std::max(largest, m_reducedMass(i, i));
    for (std::size_t j = 0; j < r; ++j)
    {
        double pivot = m_reducedMass(j, j);
        for (std::size_t k = 0; k < j; ++k)
            pivot -= m_cholesky(j, k) * m_cholesky(j, k);
        if (!(pivot > MASSLESS_SUPPORT_RATIO * largest))
            throw base::SolverError(base::concatenate(
                {"inertia relief: the rigid-body motion of support DOF ", std::to_string(j + 1), " of ",
                 std::to_string(r),
                 " moves no mass (Dᵀ·M·D is singular), so the acceleration of the supports is not defined; give the "
                 "model mass in that motion (density, PointMassInertia) or support only the DOFs of the rigid-body motions "
                 "that "
                 "carry mass"}));
        m_cholesky(j, j) = std::sqrt(pivot);
        for (std::size_t i = j + 1; i < r; ++i)
        {
            double sum = m_reducedMass(i, j);
            for (std::size_t k = 0; k < j; ++k)
                sum -= m_cholesky(i, k) * m_cholesky(j, k);
            m_cholesky(i, j) = sum / m_cholesky(j, j);
        }
    }
}

std::array<double, MAX_SUPPORT_DOFS> InertiaRelief::solveReduced(std::array<double, MAX_SUPPORT_DOFS> b) const
{
    const std::size_t r = m_modes.columns();
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t k = 0; k < i; ++k)
            b[i] -= m_cholesky(i, k) * b[k];
        b[i] /= m_cholesky(i, i);
    }
    for (std::size_t i = r; i-- > 0;)
    {
        for (std::size_t k = i + 1; k < r; ++k)
            b[i] -= m_cholesky(k, i) * b[k];
        b[i] /= m_cholesky(i, i);
    }
    return b;
}

std::array<double, MAX_SUPPORT_DOFS> InertiaRelief::accelerationsOf(std::span<const double> loadsG) const
{
    if (loadsG.size() != m_modes.rows())
        throw std::invalid_argument(base::concatenate({"InertiaRelief: expected ", std::to_string(m_modes.rows()),
                                                       " loads (the g set), found ", std::to_string(loadsG.size())}));
    std::array<double, MAX_SUPPORT_DOFS> generalised{};
    for (std::size_t j = 0; j < m_modes.columns(); ++j)
    {
        double sum = 0.0;
        for (std::size_t dof = 0; dof < loadsG.size(); ++dof)
            sum += m_modes(dof, j) * loadsG[dof];
        generalised[j] = sum;
    }
    return solveReduced(generalised);
}

void InertiaRelief::relieve(std::span<double> loadsG) const
{
    const std::array<double, MAX_SUPPORT_DOFS> acceleration = accelerationsOf(loadsG);
    for (std::size_t j = 0; j < m_modes.columns(); ++j)
        for (std::size_t dof = 0; dof < loadsG.size(); ++dof)
            loadsG[dof] -= m_massModes(dof, j) * acceleration[j];
}

std::array<double, MAX_SUPPORT_DOFS> InertiaRelief::rigidMotionOf(std::span<const double> displacementsG) const
{
    if (displacementsG.size() != m_modes.rows())
        throw std::invalid_argument(
            base::concatenate({"InertiaRelief: expected ", std::to_string(m_modes.rows()),
                               " displacements (the g set), found ", std::to_string(displacementsG.size())}));
    std::array<double, MAX_SUPPORT_DOFS> generalised{};
    for (std::size_t j = 0; j < m_modes.columns(); ++j)
    {
        double sum = 0.0;
        for (std::size_t dof = 0; dof < displacementsG.size(); ++dof)
            sum += m_massModes(dof, j) * displacementsG[dof];
        generalised[j] = sum;
    }
    return solveReduced(generalised);
}

void InertiaRelief::removeRigidMotion(std::span<double> displacementsG, base::MemoryBudget& budget) const
{
    if (displacementsG.size() != m_modes.rows())
        throw std::invalid_argument(
            base::concatenate({"InertiaRelief: expected ", std::to_string(m_modes.rows()),
                               " displacements (the g set), found ", std::to_string(displacementsG.size())}));
    base::BudgetedBuffer<double> massTimesU(budget, displacementsG.size(), "inertia relief M u");
    m_mass->multiply(displacementsG, massTimesU.span());
    std::array<double, MAX_SUPPORT_DOFS> generalised{};
    for (std::size_t j = 0; j < m_modes.columns(); ++j)
    {
        double sum = 0.0;
        for (std::size_t dof = 0; dof < displacementsG.size(); ++dof)
            sum += m_modes(dof, j) * massTimesU[dof];
        generalised[j] = sum;
    }
    const std::array<double, MAX_SUPPORT_DOFS> amplitude = solveReduced(generalised);
    for (std::size_t j = 0; j < m_modes.columns(); ++j)
        for (std::size_t dof = 0; dof < displacementsG.size(); ++dof)
            displacementsG[dof] -= m_modes(dof, j) * amplitude[j];
}

} // namespace oos::assembly
