#pragma once

#include "oos/base/ids/EntityId.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace oos::design
{

// A sparse g-set vector: (DOF, value) by increasing DOF, each DOF once.
using SparseVector = std::vector<std::pair<base::DofIndex, double>>;

// Sorts by DOF and sums the values of a DOF given more than once, in the order they were given (deterministic), and
// drops the exact zeros.
void compressSparse(SparseVector& vector);
// a·x + y as a sparse vector (both compressed).
[[nodiscard]] SparseVector addScaledSparse(const SparseVector& y, double a, const SparseVector& x);
// x·v, v dense.
[[nodiscard]] double dotSparse(const SparseVector& x, std::span<const double> v) noexcept;

// A sparse row over the independent design variables: (column, value) by increasing column, each column once. The
// explicit part of an item is one: an element item depends at fixed displacements on the few variables that drive the
// fields of its element, so a dense row per item would hold items × variables values (the full gradient mode streams
// G precisely to never hold that much).
using VariableRow = std::vector<std::pair<std::size_t, double>>;

// Sorts by column and sums the values of a column given more than once, in the order they were given, and drops the
// exact zeros: the same additions in the same order as accumulating into a dense row from 0, so the same bits.
void compressRow(VariableRow& row);
// a·x + y as a sparse row (both compressed): y_j + a·x_j where both have column j, y_j or a·x_j where one has it.
[[nodiscard]] VariableRow addScaledRow(const VariableRow& y, double a, const VariableRow& x);
// dense[column] += value for each entry of the row (a compressed row; `dense` holds every column).
void addRowTo(const VariableRow& row, std::span<double> dense) noexcept;
// The entries of a compressed row in the columns [first, first + count), as a dense block (overwritten).
void rowSlice(const VariableRow& row, std::size_t first, std::span<double> block) noexcept;

// One response item of an evaluation: the response's position in the evaluation's values and the item's index there.
struct ItemKey
{
    std::size_t response = 0;
    std::size_t item = 0;

    friend bool operator==(const ItemKey&, const ItemKey&) = default;
    friend auto operator<=>(const ItemKey&, const ItemKey&) = default;
};

// The first-order description of one response item at a design point: r(u, x) linear in its
// displacement weights at the current state, c = ∂r/∂u_g (`weights`, on the g set of the subcase's displacements), and
// its explicit part ∂r/∂x at fixed displacements (`explicitPart`, a sparse row over the independent variables: the
// designed fields' own effect on the item, through d(field)/dx; a column it does not hold is 0). The total gradient
// is dr/dx_j = ∂r/∂x_j + c·du/dx_j.
//
// An equation item e = h(r_1, ..., r_p, x) (DRESP2, DesignResp3) also keeps its own partials: ∂h/∂x through its
// variable and field arguments (`ownExplicit`) and the weights ∂h/∂r_k of its response arguments (`arguments`), from
// which its chained weights c_e = Σ_k ∂h/∂r_k·c_k and explicit part ∂h/∂x + Σ_k ∂h/∂r_k·∂r_k/∂x are formed.
struct ItemLinearisation
{
    std::int32_t subcase = 0; // 0: a global item (WEIGHT, VOLUME, FRMASS, a global equation): no displacement weights
    SparseVector weights;
    VariableRow explicitPart;
    bool isEquation = false;
    VariableRow ownExplicit;
    std::vector<std::pair<ItemKey, double>> arguments;
    // Whether the item is linearised at all: false for an item an item mask leaves out (GradientOptions::itemMasks)
    // that no wanted equation item reads; such an item gets no weights, no solve and a NaN gradient.
    bool isWanted = true;

    [[nodiscard]] bool readsDisplacements() const noexcept { return !weights.empty(); }
};

// The linearisations of the items of an evaluation, by response (the evaluation's order) and item, and their gradients
// (items × variables each, dense), which the solves of StaticSensitivity fill: every gradient starts as the item's
// explicit part. The streamed full mode (GradientStream) leaves `gradients` empty and forms G block by block.
struct LinearisedItems
{
    std::vector<std::vector<ItemLinearisation>> items;
    std::vector<std::vector<std::vector<double>>> gradients;

    [[nodiscard]] ItemLinearisation& at(const ItemKey& key) { return items[key.response][key.item]; }
    [[nodiscard]] const ItemLinearisation& at(const ItemKey& key) const { return items[key.response][key.item]; }
    [[nodiscard]] std::vector<double>& gradientOf(const ItemKey& key) { return gradients[key.response][key.item]; }
};

} // namespace oos::design
