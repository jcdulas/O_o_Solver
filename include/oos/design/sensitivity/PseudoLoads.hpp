#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/design/model/DesignRelations.hpp"
#include "oos/design/sensitivity/ResponseLinearisation.hpp"
#include "oos/solutions/common/KeptRelief.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace oos::design
{

// The pseudo-loads of a design point: per designed field v_r and subcase s,
//     q_rs = ∂P_s/∂v_r − ∂K/∂v_r · u_s,
// formed element by element on the elements the field reaches (sparse: the DOFs of those elements), and per design
// variable x_j the pseudo-load of the subcase by the chain rule through the relations,
//     b_js = Σ_r (∂v_r/∂x_j) · q_rs,
// the right-hand side of the direct method (K·du_s/dx_j = b_js) and the vector the adjoint vectors are dotted with
// (dr/dx_j = ∂r/∂x_j + λᵀ·b_js). The enforced displacements and the constraints do not depend on x (SensitivitySystem),
// so b_js holds every effect of the variable on the equilibrium of the subcase.
//
// An external unit adds its own term to b_js directly, not through a field (its pseudo-loads are
// b_j = ∂f*/∂x_j − ∂K*/∂x_j·u_a): addVariableLoad, summed after the fields' terms.
//
// Also kept: u_sᵀ·(∂K/∂v_r)·u_s, twice the explicit derivative of the total strain energy (ESE item 3); and, when asked
// (`keepsLoadParts`, for the self-adjoint responses), the load part ∂P_s/∂v_r apart: every load
// (loadPartOf), and the applied loads alone (appliedLoadPartOf: the distributed and inertial loads that addDenseLoad
// takes, without the thermal loads that addLoad takes element by element; OLOAD has no thermal load).
//
// Inertia relief (PARAM INREL = −1; solutions::KeptRelief): the relieved loads of a subcase are Q·P,
// Q = I − M·D·(Dᵀ·M·D)⁻¹·Dᵀ, and with rigid support motions their derivative is Q·(∂P/∂v − ∂M/∂v·D·ü_r): the caller
// adds −∂M/∂v·D·ü_r as an applied load of the field (addAppliedLoad), and after finish() projectRelief() keeps the
// coefficients c_rs = (Dᵀ·M·D)⁻¹·Dᵀ·(the field's load part) of the low-rank term −M·D·c_rs, which assemble(), dotAll(),
// dotRange() and markSupport() add (the load parts are kept for that: construct with keepsLoadParts). The term is
// dense on the DOFs with mass, kept as r ≤ 6 coefficients per field.
//
// Thread-safety: none while being filled; const methods are safe concurrently. Ownership: owns its vectors (bounded
// by the DOFs of the designed elements times the subcases).
class PseudoLoads
{
public:
    // `fields`: the field values of the design point (their gradients d(v_r)/dx), `variableCount` the size of x.
    PseudoLoads(const std::vector<FieldValue>& fields, std::size_t variableCount, std::size_t subcaseCount,
                bool keepsLoadParts = false);

    // −dK_e·u_e on the element's DOFs (g-set indices `dofs`, dK_e row-major over them), and u_eᵀ·dK_e·u_e.
    void addStiffness(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                      std::span<const double> stiffnessDerivative, std::span<const double> displacementsG);
    // What addStiffness adds, formed apart (on any thread: it reads only its arguments) and added later by
    // addStiffnessTerms, so that the elements of a field can be differenced in parallel and their terms added in
    // element order: the same bits as addStiffness.
    struct StiffnessTerms
    {
        std::vector<double> loads; // −dK_e·u_e, one per DOF of `dofs`
        double energy = 0.0;       // u_eᵀ·dK_e·u_e
    };
    [[nodiscard]] static StiffnessTerms stiffnessTerms(std::span<const base::DofIndex> dofs,
                                                       std::span<const double> stiffnessDerivative,
                                                       std::span<const double> displacementsG);
    // `loads` and `energy`: those of stiffnessTerms().
    void addStiffnessTerms(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                           std::span<const double> loads, double energy);
    // +dP on DOFs (an element's load derivative), or a whole g-set vector (dense, its non-zeros taken).
    void addLoad(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                 std::span<const double> values);
    // +dP on DOFs of an applied load (an element's share of the inertial loads): a load part and an applied load part,
    // as addDenseLoad keeps them, where addLoad (a thermal load) is no applied load.
    void addAppliedLoad(std::size_t field, std::size_t subcase, std::span<const base::DofIndex> dofs,
                        std::span<const double> values);
    void addDenseLoad(std::size_t field, std::size_t subcase, std::span<const double> valuesG);
    // Compresses the vectors of a field: call once its elements are all added.
    void finish(std::size_t field);
    // A SUBCOM: adds Σ a_k times the load parts of its terms (subcase, a_k) to the field's pseudo-load of
    // `subcase` and to its load parts, after finish(field) and before projectRelief(field) (which then projects them as
    // its terms'). The SUBCOM's own terms are its stiffness term: so b_c = −∂K/∂v·u_c + Σ a_k·∂P'_k/∂v.
    // std::logic_error without the load parts kept.
    void addCombination(std::size_t field, std::size_t subcase, std::span<const std::pair<std::size_t, double>> terms);
    // The inertia relief of a subcase (null: none; borrowed, it must outlive this object). Call before projectRelief.
    void setRelief(std::size_t subcase, const solutions::KeptRelief* relief);
    // After finish(field): the relief coefficients of the field in every relieved subcase. std::logic_error without
    // the load parts kept.
    void projectRelief(std::size_t field);
    // +b on DOFs for variable j itself (an external unit: ∂f*/∂x_j − ∂K*/∂x_j·u_a on its contour DOFs), then
    // finishVariableLoads() once every unit is added.
    void addVariableLoad(std::size_t variable, std::size_t subcase, std::span<const base::DofIndex> dofs,
                         std::span<const double> values);
    void finishVariableLoads();
    // The direct term of variable j in subcase s (empty without one).
    [[nodiscard]] const SparseVector& variableLoadOf(std::size_t variable, std::size_t subcase) const
    {
        return m_variableLoads.empty() ? m_empty : m_variableLoads[variable][subcase];
    }

    [[nodiscard]] const SparseVector& of(std::size_t field, std::size_t subcase) const
    {
        return m_loads[field][subcase];
    }
    [[nodiscard]] double energyRate(std::size_t field, std::size_t subcase) const
    {
        return m_energyRates[field][subcase];
    }
    // ∂P_s/∂v_r: every load, and the applied loads only (empty unless the loads parts are kept).
    [[nodiscard]] const SparseVector& loadPartOf(std::size_t field, std::size_t subcase) const;
    [[nodiscard]] const SparseVector& appliedLoadPartOf(std::size_t field, std::size_t subcase) const;
    // The fields that drive variable j and their ∂v_r/∂x_j.
    [[nodiscard]] const std::vector<std::pair<std::size_t, double>>& fieldsOf(std::size_t variable) const
    {
        return m_fieldsOfVariable[variable];
    }
    // Whether b_js is non-zero (a field of x_j has a pseudo-load in the subcase).
    [[nodiscard]] bool hasLoad(std::size_t variable, std::size_t subcase) const;
    // b_js on the g set, into `valuesG` (overwritten).
    void assemble(std::size_t variable, std::size_t subcase, std::span<double> valuesG) const;
    // Σ_j v_j·b_js on the g set, into `valuesG` (overwritten), v one value per variable: each field's q_rs once, with
    // Σ_j v_j·∂v_r/∂x_j (the δu of a Hessian-vector product), where Σ_j v_j·assemble(j) is O(x·g).
    void combine(std::size_t subcase, std::span<const double> v, std::span<double> valuesG) const;
    // b_jsᵀ·v for every variable j, v a g-set vector (an expanded adjoint vector), into `products` (variableCount()).
    void dotAll(std::size_t subcase, std::span<const double> vectorG, std::span<double> products) const;
    // b_jsᵀ·v for the variables j = first ... first + products.size() − 1 only (a column block of the streamed full
    // mode, GradientStream): the same sums as dotAll, so the same bits.
    void dotRange(std::size_t subcase, std::span<const double> vectorG, std::size_t first,
                  std::span<double> products) const;
    [[nodiscard]] std::size_t variableCount() const noexcept { return m_fieldsOfVariable.size(); }
    // Sets mask[dof] for every g-set DOF a pseudo-load of the subcase is non-zero at (every field's, every variable's
    // own term): the DOFs an adjoint vector is read at (the units the condensed path recovers).
    void markSupport(std::size_t subcase, std::span<std::uint8_t> mask) const;
    // Sets mask[dof] for every g-set DOF dotAll() reads of a vector in the subcase: every stored entry of the fields'
    // and the variables' own pseudo-loads, zero or not (so that a value dotAll() multiplies by zero is read too).
    // False, the mask untouched, when dotAll() reads every DOF (inertia relief: its term reads the whole vector). The
    // DOFs at which the full path expands an adjoint vector for the products.
    [[nodiscard]] bool markReads(std::size_t subcase, std::span<std::uint8_t> mask) const;

    // The entries of a subcase's pseudo-loads by tiles of `tileDofs` consecutive g-set DOFs (the products
    // of many adjoint vectors at once, AdjointColumns): tile t covers the DOFs [t·tileDofs, (t + 1)·tileDofs), and its
    // segments are segments[tileStart[t]] ... segments[tileStart[t + 1] − 1]. A segment is the run of one source's
    // entries in the tile, entries [begin, end) of that source: the source a field r (0 ≤ source < fieldCount()) or
    // the own load of a variable j (source fieldCount() + j, variableLoadOf). Each source's entries are sorted by DOF,
    // so its segments in tile order are its entries in order; in a tile the segments follow the sources' order.
    struct LoadSegment
    {
        std::uint32_t source = 0;
        std::uint32_t begin = 0;
        std::uint32_t end = 0;
    };
    struct LoadSegments
    {
        base::BudgetedBuffer<std::size_t> tileStart; // tiles + 1
        base::BudgetedBuffer<LoadSegment> segments;
    };
    [[nodiscard]] LoadSegments segments(std::size_t subcase, std::size_t tileDofs, std::size_t gSize,
                                        base::MemoryBudget& budget) const;
    // The entries of source `source` of segments() in a subcase.
    [[nodiscard]] const SparseVector& sourceOf(std::size_t source, std::size_t subcase) const
    {
        return source < m_loads.size() ? m_loads[source][subcase] : variableLoadOf(source - m_loads.size(), subcase);
    }
    [[nodiscard]] std::size_t fieldCount() const noexcept { return m_loads.size(); }
    // Whether the variables' own loads exist (an external unit's, addVariableLoad): segments() then numbers them as
    // sources after the fields.
    [[nodiscard]] bool hasVariableLoads() const noexcept { return !m_variableLoads.empty(); }

private:
    std::vector<std::vector<SparseVector>> m_loads; // [field][subcase]
    std::vector<std::vector<double>> m_energyRates; // [field][subcase]
    std::vector<std::vector<std::pair<std::size_t, double>>> m_fieldsOfVariable;
    bool m_keepsLoadParts = false;
    std::vector<std::vector<SparseVector>> m_loadParts;        // [field][subcase], when kept
    std::vector<std::vector<SparseVector>> m_appliedLoadParts; // [field][subcase], when kept
    std::size_t m_subcaseCount = 0;
    std::vector<std::vector<SparseVector>> m_variableLoads; // [variable][subcase], made at the first variable load
    SparseVector m_empty;
    // Inertia relief: per subcase its relief (null: none), and per field and subcase the coefficients c_rs.
    std::vector<const solutions::KeptRelief*> m_reliefs;
    std::vector<std::vector<std::array<double, assembly::MAX_SUPPORT_DOFS>>> m_reliefCoefficients;

    // Σ_r (∂v_r/∂x_j)·c_rs of variable j in a relieved subcase.
    [[nodiscard]] std::array<double, assembly::MAX_SUPPORT_DOFS> reliefOf(std::size_t variable,
                                                                          std::size_t subcase) const;
};

} // namespace oos::design
