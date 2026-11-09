#pragma once

// Internal to the assembly module: an array that grows by appends, its capacity reserved in the memory budget before
// it is allocated (the BudgetedBuffer rule for an array whose final size is only known at the end:
// the node groups of the elements, whose count of nodes per element the facets give).

#include "oos/base/memory/MemoryBudget.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace oos::assembly
{

// Thread-safety: none. Ownership: owns its values and its reservation; the budget must outlive it.
template <typename T>
class BudgetedGrowingArray
{
public:
    static constexpr std::size_t INITIAL_CAPACITY = 1024;

    BudgetedGrowingArray(base::MemoryBudget& budget, std::string label) : m_budget(&budget), m_label(std::move(label))
    {
    }

    // Throws MemoryBudgetError when the larger capacity does not fit (nothing is allocated then).
    void append(std::span<const T> values)
    {
        const std::size_t needed = m_values.size() + values.size();
        if (needed > m_values.capacity())
            grow(needed);
        m_values.insert(m_values.end(), values.begin(), values.end());
    }
    void push(const T& value) { append(std::span<const T>(&value, 1)); }

    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }
    [[nodiscard]] std::span<const T> span() const noexcept { return m_values; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return m_values[i]; }

private:
    void grow(std::size_t needed)
    {
        const std::size_t capacity = std::max({needed, 2 * m_values.capacity(), INITIAL_CAPACITY});
        base::MemoryReservation larger = m_budget->reserve(capacity * sizeof(T), m_label);
        m_values.reserve(capacity);
        m_reservation = std::move(larger);
    }

    base::MemoryBudget* m_budget;
    std::string m_label;
    base::MemoryReservation m_reservation;
    std::vector<T> m_values;
};

} // namespace oos::assembly
