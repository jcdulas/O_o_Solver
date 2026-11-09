#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace oos::base
{

// The alignment of every large numeric buffer: one cache line, and the width of an AVX-512 register, so that the SIMD
// kernels may use aligned loads on the first element of a column.
inline constexpr std::size_t BUFFER_ALIGNMENT = 64;

// The tag of the constructors that leave the values uninitialised, for a buffer its user overwrites in whole before
// reading it (as std::make_unique_for_overwrite).
struct ForOverwrite
{
};
inline constexpr ForOverwrite FOR_OVERWRITE{};

// A contiguous, 64-byte aligned, zero-initialised array of trivially copyable values, indexed in 64 bits (a dense K*
// above 46 340 DOF has more than 2³¹ terms). A buffer whose size grows with the model is a BudgetedBuffer,
// which wraps this one.
//
// Thread-safety: as a std::vector. Ownership: owns its memory; movable, not copyable (a copy of a large buffer is
// always explicit: copyFrom).
template <typename T>
    requires std::is_trivially_copyable_v<T>
class AlignedBuffer
{
public:
    AlignedBuffer() = default;
    // Throws std::bad_array_new_length when size·sizeof(T) does not fit a std::size_t, std::bad_alloc when the system
    // has not the memory.
    explicit AlignedBuffer(std::size_t size) : AlignedBuffer(size, FOR_OVERWRITE) { std::fill_n(m_data, size, T{}); }
    // The values are left uninitialised: the caller writes each one before reading it.
    //
    // PERFORMANCE: a zero fill is one serial pass over the buffer, and the pages it touches are first touched by the
    // calling thread; the factor panels of a large model (5.7 GB at 1 M DOF) and the fronts are overwritten in whole by
    // the factorisation, so the fill is pure cost there.
    // Measured: shell model of 999 120 DOF, 6 threads, AVX2, three alternating runs each: factorisation
    // 4.17-4.44 s against 4.85-5.01 s with the zero fill (-12 %), whole run 5.20-5.47 s against 5.92-6.21 s.
    AlignedBuffer(std::size_t size, ForOverwrite) : m_size(size)
    {
        if (size == 0)
            return;
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        m_data = static_cast<T*>(::operator new(size * sizeof(T), std::align_val_t{BUFFER_ALIGNMENT}));
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : m_data(std::exchange(other.m_data, nullptr)), m_size(std::exchange(other.m_size, 0))
    {
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_data = std::exchange(other.m_data, nullptr);
            m_size = std::exchange(other.m_size, 0);
        }
        return *this;
    }
    ~AlignedBuffer() { release(); }

    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] std::size_t size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return m_data[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return m_data[i]; }
    [[nodiscard]] std::span<T> span() noexcept { return {m_data, m_size}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {m_data, m_size}; }

    // Copies `source`, which must have size() values (std::invalid_argument otherwise).
    void copyFrom(std::span<const T> source)
    {
        if (source.size() != m_size)
            throw std::invalid_argument(std::string("AlignedBuffer::copyFrom: expected ")
                                            .append(std::to_string(m_size))
                                            .append(" values, the source has ")
                                            .append(std::to_string(source.size())));
        std::copy_n(source.data(), m_size, m_data);
    }

private:
    void release() noexcept
    {
        if (m_data != nullptr)
            ::operator delete(m_data, std::align_val_t{BUFFER_ALIGNMENT});
        m_data = nullptr;
        m_size = 0;
    }

    T* m_data = nullptr;
    std::size_t m_size = 0;
};

} // namespace oos::base
