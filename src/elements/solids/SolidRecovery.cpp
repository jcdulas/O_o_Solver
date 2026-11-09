#include "oos/elements/solids/SolidRecovery.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

// The Jacobi sweeps stop when the off-diagonal terms fall below this fraction of the tensor's norm (round-off of a
// double), or after MAX_JACOBI_SWEEPS (a 3 × 3 tensor converges quadratically in 4 or 5).
constexpr double JACOBI_TOLERANCE = 1.0e-15;
constexpr int MAX_JACOBI_SWEEPS = 50;
// The factor √2/3 of the equivalent strain of the solids.
const double EQUIVALENT_STRAIN_FACTOR = std::sqrt(2.0) / 3.0;
// The values of a stress or strain row (solidOutputColumns) and its Voigt components, the first ones.
constexpr std::size_t SOLID_ROW_VALUES = 20;
constexpr std::size_t VOIGT_COMPONENTS = 6;

[[nodiscard]] base::Mat3 tensorOf(const Voigt6& v, double shearScale)
{
    return base::matrixFromRows({v[0], shearScale * v[3], shearScale * v[5]},
                                {shearScale * v[3], v[1], shearScale * v[4]},
                                {shearScale * v[5], shearScale * v[4], v[2]});
}

// The index of the component of largest magnitude, the last one when two tie.
[[nodiscard]] std::size_t largestComponent(base::Vec3 v)
{
    std::size_t largest = 0;
    for (std::size_t k = 1; k < 3; ++k)
        if (std::abs(v[k]) >= std::abs(v[largest]))
            largest = k;
    return largest;
}

// The Jacobi sweeps of W symmetric tensors side by side (lanes), each with the operations of the sweeps of one tensor
// alone, so its bits whatever W and whatever the other lanes: a[3·i + j][l] and v[3·i + j][l] the tensor and the
// rotations of lane l. A lane that converged (off-diagonal sum within JACOBI_TOLERANCE of its norm at the start of a
// sweep) or whose term (p, q) is zero is not rotated: its rotation is computed and discarded (selected away).
//
// One rotation R (the identity but R(p, p) = R(q, q) = c, R(p, q) = s, R(q, p) = −s, p < q, r the third index) is
// a ← (Rᵀ·a)·R, then a(p, q) = a(q, p) = 0, and v ← v·R, with the bits of those full 3 × 3 products (base::MatN: each
// term the sum from +0 of its three products, k increasing) for finite values: a sum from +0 never holds −0 before its
// last nonzero term, so a product by an exact 0 adds nothing to it and a product by 1 leaves its factor, but for the
// sign of a zero: Σ = 0.0 + x for one term, (0.0 + x) + y for two, in the order of k (p before q).
//
// PERFORMANCE: the principal values of the solids' rows: the full products spent most of their 54 + 27 products
// on R's zeros and ones, and the sweeps of one tensor wait on the latency of their divisions and square roots; here
// only the terms of c and s, and the rows of a block's cases JACOBI_LANES at a time, so that their sweeps overlap.
template <std::size_t W>
struct JacobiLanes
{
    std::array<std::array<double, W>, 9> a{};
    std::array<std::array<double, W>, 9> v{};
};

template <std::size_t W>
void sweepJacobi(JacobiLanes<W>& x)
{
    const auto at = [](std::size_t i, std::size_t j) { return 3 * i + j; };
    std::array<double, W> norm{};
    for (std::size_t l = 0; l < W; ++l)
    {
        double sum = 0.0;
        for (std::size_t e = 0; e < 9; ++e)
            sum += x.a[e][l] * x.a[e][l];
        norm[l] = std::sqrt(sum);
    }
    std::array<bool, W> isDone{};
    // Each step of every lane in one loop over the lanes, so that the lanes' divisions and square roots follow one
    // another (their latencies overlap).
    std::array<double, W> c{};
    std::array<double, W> s{};
    std::array<bool, W> isRotated{};
    std::array<std::array<double, W>, 9> rotated{};
    for (int sweep = 0; sweep < MAX_JACOBI_SWEEPS; ++sweep)
    {
        bool isAnyActive = false;
        for (std::size_t l = 0; l < W; ++l)
        {
            const double off = std::abs(x.a[at(0, 1)][l]) + std::abs(x.a[at(0, 2)][l]) + std::abs(x.a[at(1, 2)][l]);
            isDone[l] = isDone[l] || off <= JACOBI_TOLERANCE * norm[l];
            isAnyActive = isAnyActive || !isDone[l];
        }
        if (!isAnyActive)
            break;
        for (std::size_t p = 0; p < 2; ++p)
            for (std::size_t q = p + 1; q < 3; ++q)
            {
                const std::size_t r = 3 - p - q;
                // The rotation that zeroes a(p, q): t = sign(θ)/(|θ| + √(θ² + 1)), θ = (a_qq − a_pp)/(2 a_pq).
                for (std::size_t l = 0; l < W; ++l)
                {
                    const double apq = x.a[at(p, q)][l];
                    isRotated[l] = !isDone[l] && apq != 0.0;
                    const double theta = (x.a[at(q, q)][l] - x.a[at(p, p)][l]) / (2.0 * apq);
                    const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    c[l] = 1.0 / std::sqrt(t * t + 1.0);
                    s[l] = t * c[l];
                }
                // T = Rᵀ·a: T(i, j) = Σ_k R(k, i)·a(k, j).
                for (std::size_t j = 0; j < 3; ++j)
                    for (std::size_t l = 0; l < W; ++l)
                    {
                        const double apj = x.a[at(p, j)][l];
                        const double aqj = x.a[at(q, j)][l];
                        rotated[at(p, j)][l] = (0.0 + c[l] * apj) + (-s[l]) * aqj;
                        rotated[at(q, j)][l] = (0.0 + s[l] * apj) + c[l] * aqj;
                        rotated[at(r, j)][l] = 0.0 + x.a[at(r, j)][l];
                    }
                // a = T·R and v = v·R: X(i, p) = (0 + X(i, p)·c) + X(i, q)·(−s), X(i, q) = (0 + X(i, p)·s) +
                // X(i, q)·c, X(i, r) = 0 + X(i, r); then a(p, q) = a(q, p) = 0.
                for (std::size_t i = 0; i < 3; ++i)
                    for (std::size_t l = 0; l < W; ++l)
                    {
                        const double tp = rotated[at(i, p)][l];
                        const double tq = rotated[at(i, q)][l];
                        const double vp = x.v[at(i, p)][l];
                        const double vq = x.v[at(i, q)][l];
                        const double newP = i == q ? 0.0 : (0.0 + tp * c[l]) + tq * (-s[l]);
                        const double newQ = i == p ? 0.0 : (0.0 + tp * s[l]) + tq * c[l];
                        const double newR = 0.0 + rotated[at(i, r)][l];
                        const double newVp = (0.0 + vp * c[l]) + vq * (-s[l]);
                        const double newVq = (0.0 + vp * s[l]) + vq * c[l];
                        const double newVr = 0.0 + x.v[at(i, r)][l];
                        x.a[at(i, p)][l] = isRotated[l] ? newP : x.a[at(i, p)][l];
                        x.a[at(i, q)][l] = isRotated[l] ? newQ : x.a[at(i, q)][l];
                        x.a[at(i, r)][l] = isRotated[l] ? newR : x.a[at(i, r)][l];
                        x.v[at(i, p)][l] = isRotated[l] ? newVp : vp;
                        x.v[at(i, q)][l] = isRotated[l] ? newVq : vq;
                        x.v[at(i, r)][l] = isRotated[l] ? newVr : x.v[at(i, r)][l];
                    }
            }
    }
}

// The principal state of lane l of swept tensors: the values as (largest, smallest, middle), each with its column of
// v, the direction of the first two with their largest component positive, of the third negative.
template <std::size_t W>
PrincipalState principalStateOfLane(const JacobiLanes<W>& x, std::size_t l)
{
    const auto diagonal = [&](std::size_t i) { return x.a[4 * i][l]; };
    std::array<std::size_t, 3> order{0, 1, 2};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = i + 1; j < 3; ++j)
            if (diagonal(order[j]) > diagonal(order[i]))
                std::swap(order[i], order[j]);
    const std::array<std::size_t, 3> outputOrder{order[0], order[2], order[1]};
    PrincipalState state;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t column = outputOrder[k];
        state.values[k] = diagonal(column);
        base::Vec3 direction{x.v[column][l], x.v[3 + column][l], x.v[6 + column][l]};
        const double wanted = k < 2 ? 1.0 : -1.0;
        if (direction[largestComponent(direction)] * wanted < 0.0)
            direction = -direction;
        state.directions[k] = direction;
    }
    return state;
}

// The tensors of a row and of its lanes loaded: a the tensor, v the identity.
template <std::size_t W>
void loadLane(JacobiLanes<W>& x, std::size_t l, const base::Mat3& tensor)
{
    for (std::size_t e = 0; e < 9; ++e)
    {
        x.a[e][l] = tensor.values[e];
        x.v[e][l] = e % 4 == 0 ? 1.0 : 0.0;
    }
}

// The rows side by side in the sweeps of the principal values (JacobiLanes).
constexpr std::size_t JACOBI_LANES = 4;

void fillPrincipal(const PrincipalState& state, std::array<double, 20>& row)
{
    for (std::size_t k = 0; k < 3; ++k)
    {
        row[6 + k] = state.values[k];
        for (std::size_t i = 0; i < 3; ++i)
            row[11 + 3 * k + i] = state.directions[k][i];
    }
}

// A stress row from its components and their principal state (solidStressRow).
std::array<double, 20> solidStressRowOf(const Voigt6& s, const PrincipalState& state)
{
    std::array<double, 20> row{};
    for (std::size_t k = 0; k < 6; ++k)
        row[k] = s[k];
    fillPrincipal(state, row);
    const double dxy = s[0] - s[1];
    const double dyz = s[1] - s[2];
    const double dzx = s[2] - s[0];
    row[9] = std::sqrt(0.5 * (dxy * dxy + dyz * dyz + dzx * dzx) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
    row[10] = -(s[0] + s[1] + s[2]) / 3.0;
    return row;
}

// A strain row from its components and the principal state of its tensor (shears halved: solidStrainRow).
std::array<double, 20> solidStrainRowOf(const Voigt6& e, const PrincipalState& state)
{
    std::array<double, 20> row{};
    for (std::size_t k = 0; k < 6; ++k)
        row[k] = e[k];
    fillPrincipal(state, row);
    const double d12 = state.values[0] - state.values[1];
    const double d23 = state.values[1] - state.values[2];
    const double d31 = state.values[2] - state.values[0];
    row[9] = EQUIVALENT_STRAIN_FACTOR * std::sqrt(d12 * d12 + d23 * d23 + d31 * d31);
    row[10] = -(e[0] + e[1] + e[2]) / 3.0;
    return row;
}

} // namespace

PrincipalState principalStateOf(const base::Mat3& tensor)
{
    JacobiLanes<1> x;
    loadLane(x, 0, tensor);
    sweepJacobi(x);
    return principalStateOfLane(x, 0);
}

void principalStatesOf(std::span<const base::Mat3> tensors, std::span<PrincipalState> states)
{
    if (states.size() != tensors.size())
        throw std::invalid_argument(base::concatenate(
            {"principalStatesOf: expected a state for each tensor, found ", std::to_string(tensors.size()),
             " tensors and ", std::to_string(states.size()), " states; pass spans of one size"}));
    std::size_t first = 0;
    for (; first + JACOBI_LANES <= tensors.size(); first += JACOBI_LANES)
    {
        JacobiLanes<JACOBI_LANES> x;
        for (std::size_t l = 0; l < JACOBI_LANES; ++l)
            loadLane(x, l, tensors[first + l]);
        sweepJacobi(x);
        for (std::size_t l = 0; l < JACOBI_LANES; ++l)
            states[first + l] = principalStateOfLane(x, l);
    }
    for (; first < tensors.size(); ++first)
        states[first] = principalStateOf(tensors[first]);
}

std::vector<std::string> solidOutputColumns(ElementOutput output)
{
    switch (output)
    {
    case ElementOutput::Stresses:
        return {"oxx",      "oyy", "ozz", "txy", "tyz", "txz", "o1",  "o2",  "o3",  "vonMises",
                "pressure", "o1x", "o1y", "o1z", "o2x", "o2y", "o2z", "o3x", "o3y", "o3z"};
    case ElementOutput::Strains:
        return {"exx",      "eyy", "ezz", "exy", "eyz", "exz", "e1",  "e2",  "e3",  "vonMises",
                "pressure", "o1x", "o1y", "o1z", "o2x", "o2y", "o2z", "o3x", "o3y", "o3z"};
    default:
        return {};
    }
}

std::vector<ColumnForm> solidOutputColumnForms(ElementOutput output)
{
    if (solidOutputColumns(output).empty())
        return {};
    std::vector<ColumnForm> forms(SOLID_ROW_VALUES, ColumnForm::other());
    for (std::size_t k = 0; k < VOIGT_COMPONENTS; ++k)
        forms[k] = ColumnForm::linear();
    return forms;
}

void completeSolidRows(ElementOutput output, std::span<ElementResultBlock> blocks)
{
    if (output != ElementOutput::Stresses && output != ElementOutput::Strains)
        throw std::invalid_argument("completeSolidRows: expected stresses or strains, found another output kind");
    const bool isStress = output == ElementOutput::Stresses;
    // The rows of every case of the block, JACOBI_LANES at a time (their sweeps side by side), each made as
    // solidStressRow or solidStrainRow makes it.
    std::array<Voigt6, JACOBI_LANES> components{};
    std::array<std::span<double>, JACOBI_LANES> targets{};
    std::size_t filled = 0;
    const auto flush = [&]()
    {
        std::array<base::Mat3, JACOBI_LANES> tensors{};
        std::array<PrincipalState, JACOBI_LANES> states{};
        for (std::size_t l = 0; l < filled; ++l)
            tensors[l] = tensorOf(components[l], isStress ? 1.0 : 0.5);
        principalStatesOf(std::span<const base::Mat3>(tensors.data(), filled),
                          std::span<PrincipalState>(states.data(), filled));
        for (std::size_t l = 0; l < filled; ++l)
        {
            const std::array<double, SOLID_ROW_VALUES> row =
                isStress ? solidStressRowOf(components[l], states[l]) : solidStrainRowOf(components[l], states[l]);
            std::copy(row.begin(), row.end(), targets[l].begin());
        }
        filled = 0;
    };
    for (ElementResultBlock& block : blocks)
        for (std::size_t i = 0; i < block.size(); ++i)
        {
            const std::span<double> values = block.mutableValues(i);
            for (std::size_t k = 0; k < VOIGT_COMPONENTS; ++k)
                components[filled][k] = values[k];
            targets[filled] = values;
            if (++filled == JACOBI_LANES)
                flush();
        }
    if (filled > 0)
        flush();
}

std::array<double, 20> solidStressRow(const Voigt6& s)
{
    return solidStressRowOf(s, principalStateOf(tensorOf(s, 1.0)));
}

std::array<double, 20> solidStrainRow(const Voigt6& e)
{
    return solidStrainRowOf(e, principalStateOf(tensorOf(e, 0.5)));
}

void appendSolidRows(ElementResultBlock& block, std::int64_t row, std::int32_t elementId,
                     std::span<const std::int32_t> cornerIds, const SolidResponse& response,
                     std::span<const double> displacements, ElementOutput output,
                     std::span<const Voigt6> thermalStresses, std::span<const Voigt6> thermalModeStrains)
{
    if (output != ElementOutput::Stresses && output != ElementOutput::Strains)
        throw std::invalid_argument("solid recovery: expected stresses or strains, found another output kind");
    for (std::size_t point = 0; point < response.pointCount; ++point)
    {
        ResultPlace place;
        place.nodeId = point == 0 ? 0 : cornerIds[point - 1];
        std::array<double, 20> values{};
        if (output == ElementOutput::Stresses)
        {
            Voigt6 stress = response.stress[point].apply(displacements);
            if (!thermalStresses.empty())
                for (std::size_t k = 0; k < 6; ++k)
                    stress[k] -= thermalStresses[point][k];
            values = solidStressRow(stress);
        }
        else
        {
            Voigt6 strain = response.strain[point].apply(displacements);
            if (!thermalModeStrains.empty())
                for (std::size_t k = 0; k < 6; ++k)
                    strain[k] += thermalModeStrains[point][k];
            values = solidStrainRow(strain);
        }
        block.append(row, elementId, place, values);
    }
}

} // namespace oos::elements
