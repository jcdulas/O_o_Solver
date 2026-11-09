#pragma once

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/memory/AlignedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace oos::base
{

// A buffer whose size grows with the model (numerical code): its size is reserved in the budget before
// the allocation, and given back after the memory is freed. A buffer that does not fit is refused with
// MemoryBudgetError before anything is allocated. 64-byte aligned and zero-initialised, as AlignedBuffer.
//
// Thread-safety: as a std::vector (concurrent reads, or writes to distinct elements). Ownership: owns its memory and
// its reservation; movable, not copyable. The budget must outlive the buffer.
template <typename T>
    requires std::is_trivially_copyable_v<T>
class BudgetedBuffer
{
public:
    BudgetedBuffer() = default;
    BudgetedBuffer(MemoryBudget& budget, std::size_t size, const std::string& label)
        : m_reservation(budget.reserve(byteCount(budget, size, label), label)), m_buffer(size)
    {
    }
    // The values are left uninitialised (AlignedBuffer's FOR_OVERWRITE): the caller writes each one before reading it.
    BudgetedBuffer(MemoryBudget& budget, std::size_t size, const std::string& label, ForOverwrite)
        : m_reservation(budget.reserve(byteCount(budget, size, label), label)), m_buffer(size, FOR_OVERWRITE)
    {
    }
    // Takes the values of `buffer` as they are (memory already in hand, given back by an earlier user: a factor's
    // panels reused by the next factorisation of its structure), its size reserved in the budget first, as
    // a new allocation would be. When the reservation is refused (MemoryBudgetError), `buffer` is left untouched.
    BudgetedBuffer(MemoryBudget& budget, AlignedBuffer<T>&& buffer, const std::string& label)
        : m_reservation(budget.reserve(byteCount(budget, buffer.size(), label), label)), m_buffer(std::move(buffer))
    {
    }

    [[nodiscard]] T* data() noexcept { return m_buffer.data(); }
    [[nodiscard]] const T* data() const noexcept { return m_buffer.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return m_buffer.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_buffer.empty(); }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return m_buffer[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return m_buffer[i]; }
    [[nodiscard]] std::span<T> span() noexcept { return m_buffer.span(); }
    [[nodiscard]] std::span<const T> span() const noexcept { return m_buffer.span(); }
    [[nodiscard]] std::uint64_t reservedBytes() const noexcept { return m_reservation.bytes(); }

    // Hands the values over and gives the reservation back: the memory then belongs to the caller, outside the budget,
    // and this buffer is empty.
    [[nodiscard]] AlignedBuffer<T> takeValues() noexcept
    {
        AlignedBuffer<T> values = std::move(m_buffer);
        m_reservation.release();
        return values;
    }

private:
    // The size in bytes, refused as a budget overflow when it does not even fit 64 bits.
    [[nodiscard]] static std::uint64_t byteCount(const MemoryBudget& budget, std::size_t size, const std::string& label)
    {
        if (size > std::numeric_limits<std::uint64_t>::max() / sizeof(T))
            throw MemoryBudgetError(
                concatenate({"memory budget exceeded: '", label, "' asks for ", std::to_string(size), " values of ",
                             std::to_string(sizeof(T)),
                             " bytes, more than 2^64 bytes; the size is wrong or the model too "
                             "large for one piece: condense more units"}),
                MemoryShortfall{std::numeric_limits<std::uint64_t>::max(), budget.available(), budget.limit(), {}});
        return static_cast<std::uint64_t>(size) * sizeof(T);
    }

    MemoryReservation m_reservation; // declared first: reserved before the allocation, given back after the release
    AlignedBuffer<T> m_buffer;
};

} // namespace oos::base
