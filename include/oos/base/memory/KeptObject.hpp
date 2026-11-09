#pragma once

#include "oos/base/memory/MemoryBudget.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace oos::base
{

// An object kept beyond the analysis that built it (the structures a StructureCache keeps between design points, step
// 10.13): its buffers are reserved in a memory budget of its own, which the object keeps alive, so that what it holds
// is that budget's use once built (bytes()); an analysis that takes it borrows it with those bytes reserved in its own
// budget for as long as the borrowed pointer, or a copy, lives (lend()), so that the analysis counts what it uses
// whoever built it.
//
// Thread-safety: immutable once built; lend() may run concurrently (the budgets are thread-safe). Ownership: owns the
// object and its budget (declared first: destroyed after the object); shared. T may be incomplete where a KeptObject
// is only held (its type is complete where it is built).
template <typename T>
class KeptObject : public std::enable_shared_from_this<KeptObject<T>>
{
public:
    // The object `make(budget)` returns, built with the fresh budget it is kept in. Throws what make throws.
    template <typename Make>
    [[nodiscard]] static std::shared_ptr<const KeptObject> build(Make&& make)
    {
        auto kept = std::shared_ptr<KeptObject>(new KeptObject());
        kept->m_object = std::make_unique<T>(std::forward<Make>(make)(*kept->m_budget));
        kept->m_bytes = kept->m_budget->used();
        return kept;
    }

    KeptObject(const KeptObject&) = delete;
    KeptObject& operator=(const KeptObject&) = delete;
    KeptObject(KeptObject&&) = delete;
    KeptObject& operator=(KeptObject&&) = delete;
    ~KeptObject() = default;

    [[nodiscard]] const T& object() const noexcept { return *m_object; }
    // The bytes its buffers hold (its budget's use once built).
    [[nodiscard]] std::uint64_t bytes() const noexcept { return m_bytes; }

    // The object, with bytes() reserved in `budget` under `label` while the pointer or a copy of it lives. Throws
    // MemoryBudgetError when the budget cannot hold them.
    [[nodiscard]] std::shared_ptr<const T> lend(MemoryBudget& budget, const std::string& label) const
    {
        struct Loan
        {
            MemoryReservation reservation; // declared first: given back after the object is released
            std::shared_ptr<const KeptObject> kept;
        };
        auto loan = std::make_shared<Loan>(Loan{budget.reserve(m_bytes, label), this->shared_from_this()});
        const T* object = &loan->kept->object();
        return std::shared_ptr<const T>(std::move(loan), object);
    }

private:
    KeptObject() : m_budget(std::make_unique<MemoryBudget>(MemoryBudget::UNLIMITED)) {}

    std::unique_ptr<MemoryBudget> m_budget; // declared first: outlives the object reserved in it
    std::unique_ptr<T> m_object;
    std::uint64_t m_bytes = 0;
};

} // namespace oos::base
