#include "oos/reduction/scheduling/KeptInteriorFactors.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/platform/MemoryInfo.hpp"

#include <algorithm>
#include <bit>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

template <typename T>
base::BudgetedBuffer<T> copyOf(std::span<const T> values, base::MemoryBudget& budget, const std::string& label)
{
    base::BudgetedBuffer<T> copy(budget, values.size(), label);
    std::copy(values.begin(), values.end(), copy.span().begin());
    return copy;
}

// Whether two spans hold the same bits (doubles compared by their bits: −0 and 0 are told apart).
bool isSameBits(std::span<const double> a, std::span<const double> b) noexcept
{
    if (a.size() != b.size())
        return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k]))
            return false;
    return true;
}

// Whether `kept` was made from the equations of `system` (the comparison of KeptInteriorFactors::take): the boundary
// labels, the interior count and the matrix compared in full, the symbolic shape and the factor's own check.
bool isMadeFrom(const KeptInteriorFactor& kept, const UnitSystem& system, linalg::KernelKind kernel,
                const linalg::PivotPolicy& policy)
{
    const linalg::SymmetricCsc& matrix = system.matrix();
    const linalg::SymbolicFactor& symbolic = system.symbolic();
    const std::span<const UnitDofLabel> labels = system.boundaryLabels();
    const std::span<const UnitDofLabel> keptLabels = kept.boundaryLabels.span();
    const std::span<const std::int64_t> offsets = matrix.columnOffsets();
    const std::span<const std::int32_t> rows = matrix.rowIndices();
    const std::span<const std::int64_t> keptOffsets = kept.columnOffsets.span();
    const std::span<const std::int32_t> keptRows = kept.rowIndices.span();
    return kept.interiorDofCount == system.interiorDofCount() &&
           std::equal(labels.begin(), labels.end(), keptLabels.begin(), keptLabels.end()) &&
           std::equal(offsets.begin(), offsets.end(), keptOffsets.begin(), keptOffsets.end()) &&
           std::equal(rows.begin(), rows.end(), keptRows.begin(), keptRows.end()) &&
           isSameBits(matrix.values(), kept.values.span()) && kept.supernodeCount == symbolic.supernodeCount() &&
           kept.panelValueCount == symbolic.panelValueCount() &&
           kept.factor.isFactorisationOf(symbolic, matrix, kernel, policy);
}

} // namespace

KeptInteriorFactor& KeptInteriorFactor::operator=(KeptInteriorFactor&& other) noexcept
{
    if (this == &other)
        return *this;
    {
        // What it holds, freed here in the order of destruction (the buffers, then their budgets); it is left with
        // moved-from members, whose assignments below free nothing.
        const KeptInteriorFactor held(std::move(*this));
    }
    factorMemoryOwner = std::move(other.factorMemoryOwner);
    equationsBudget = std::move(other.equationsBudget);
    unitIdentity = other.unitIdentity;
    unitId = other.unitId;
    boundaryLabels = std::move(other.boundaryLabels);
    interiorDofCount = other.interiorDofCount;
    columnOffsets = std::move(other.columnOffsets);
    rowIndices = std::move(other.rowIndices);
    values = std::move(other.values);
    supernodeCount = other.supernodeCount;
    panelValueCount = other.panelValueCount;
    factor = std::move(other.factor);
    bytes = other.bytes;
    return *this;
}

KeptInteriorFactor keepInteriorFactor(std::uint64_t unitIdentity, std::int64_t unitId, const UnitSystem& system,
                                      linalg::PartialFactor factor, std::uint64_t factorBytes,
                                      std::shared_ptr<base::MemoryBudget> factorMemoryOwner)
{
    if (!system.hasEquations() || !factor.isInteriorFactorKept() || factorMemoryOwner == nullptr)
        throw std::invalid_argument(base::concatenate(
            {"keepInteriorFactor (unit ", std::to_string(unitId),
             "): a kept factor needs the unit's equations, its interior factor and the budget it is reserved in; "
             "offer the factor of a unit condensed with its equations, before they are released"}));
    KeptInteriorFactor kept;
    kept.factorMemoryOwner = std::move(factorMemoryOwner);
    kept.equationsBudget = std::make_unique<base::MemoryBudget>(base::MemoryBudget::UNLIMITED);
    base::MemoryBudget& budget = *kept.equationsBudget;
    kept.unitIdentity = unitIdentity;
    kept.unitId = unitId;
    kept.boundaryLabels = copyOf(system.boundaryLabels(), budget, "kept factor boundary labels");
    kept.interiorDofCount = system.interiorDofCount();
    const linalg::SymmetricCsc& matrix = system.matrix();
    kept.columnOffsets = copyOf(matrix.columnOffsets(), budget, "kept factor matrix offsets");
    kept.rowIndices = copyOf(matrix.rowIndices(), budget, "kept factor matrix rows");
    kept.values = copyOf(matrix.values(), budget, "kept factor matrix values");
    kept.supernodeCount = system.symbolic().supernodeCount();
    kept.panelValueCount = system.symbolic().panelValueCount();
    kept.factor = std::move(factor);
    kept.bytes = factorBytes + budget.used();
    return kept;
}

std::uint64_t KeptInteriorFactors::defaultLimitBytes()
{
    return base::MemoryInfo::query().totalBytes / DEFAULT_LIMIT_DIVISOR;
}

void KeptInteriorFactors::offer(KeptInteriorFactor factor)
{
    std::vector<KeptInteriorFactor> dropped; // freed outside the lock
    const std::lock_guard<std::mutex> lock(m_mutex);
    ++m_counts.offered;
    // One factor per unit: the earlier one (of other equations, or of the same) gives way.
    const auto same = std::find_if(m_kept.begin(), m_kept.end(), [&](const KeptInteriorFactor& kept)
                                   { return kept.unitIdentity == factor.unitIdentity; });
    if (same != m_kept.end())
    {
        m_bytes -= same->bytes;
        dropped.push_back(std::move(*same));
        m_kept.erase(same);
        ++m_counts.replaced;
    }
    if (factor.bytes > m_limitBytes)
    {
        ++m_counts.refused;
        return;
    }
    m_bytes += factor.bytes;
    m_kept.push_back(std::move(factor));
    evictBeyondLimit();
}

std::optional<KeptInteriorFactor> KeptInteriorFactors::take(std::uint64_t unitIdentity, const UnitSystem& system,
                                                            linalg::KernelKind kernel,
                                                            const linalg::PivotPolicy& policy)
{
    if (!system.hasEquations())
        return std::nullopt;
    std::optional<KeptInteriorFactor> taken;
    const std::lock_guard<std::mutex> lock(m_mutex);
    const auto found = std::find_if(m_kept.begin(), m_kept.end(),
                                    [&](const KeptInteriorFactor& kept) { return kept.unitIdentity == unitIdentity; });
    if (found == m_kept.end())
        return std::nullopt;
    // A factor of the unit made from other equations is dropped: the unit changed since (its stiffness, its
    // constraints, its boundary), and a later analysis of the old equations is unlikely.
    const bool isSame = isMadeFrom(*found, system, kernel, policy);
    if (isSame)
        taken = std::move(*found);
    m_bytes -= found->bytes;
    m_kept.erase(found);
    ++(isSame ? m_counts.taken : m_counts.mismatched);
    return taken;
}

std::uint64_t KeptInteriorFactors::limitBytes() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_limitBytes;
}

void KeptInteriorFactors::setLimitBytes(std::uint64_t limitBytes)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_limitBytes = limitBytes;
    evictBeyondLimit();
}

std::uint64_t KeptInteriorFactors::bytes() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_bytes;
}

std::size_t KeptInteriorFactors::count() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_kept.size();
}

KeptInteriorFactorCounts KeptInteriorFactors::counts() const
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_counts;
}

void KeptInteriorFactors::clear()
{
    std::vector<KeptInteriorFactor> dropped; // freed outside the lock
    const std::lock_guard<std::mutex> lock(m_mutex);
    dropped.swap(m_kept);
    m_bytes = 0;
    m_counts = KeptInteriorFactorCounts{};
}

void KeptInteriorFactors::evictBeyondLimit()
{
    std::size_t first = 0;
    while (m_bytes > m_limitBytes && first < m_kept.size())
    {
        m_bytes -= m_kept[first].bytes;
        ++first;
        ++m_counts.evicted;
    }
    m_kept.erase(m_kept.begin(), m_kept.begin() + static_cast<std::ptrdiff_t>(first));
}

} // namespace oos::reduction
