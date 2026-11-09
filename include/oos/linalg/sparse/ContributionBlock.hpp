#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::linalg
{

// One DOF of a node of a SparsePattern: the node index and the component, counted from 0 within the node (0 to 5 for
// a grid point, 0 for a scalar point). The assembly module maps the DOFs of an element (grid point rows and the
// components 1 to 6 of the input format) to these before it hands the element matrices to the SparseAssembler; linalg
// knows no element.
struct NodeDof
{
    std::int32_t node = 0;
    std::int32_t component = 0;

    friend constexpr bool operator==(const NodeDof&, const NodeDof&) = default;
};

// The dense symmetric contributions (element matrices) of a range of elements, as a ContributionSource writes them for
// the SparseAssembler: per contribution its DOFs and its n × n matrix over them, row-major and complete (both
// triangles; the assembler reads the triangle that falls in the lower triangle of the assembled matrix). The order of
// the contributions is the order in which they are added to every term.
//
// Storage is contiguous for the whole range, so that a source fills it without an allocation per element once the
// block has grown; its size is bounded by the range the assembler chooses times the largest element (std::vector is
// the right container).
//
// Thread-safety: none for writes; one block per piece. Ownership: owns its values.
class ContributionBlock
{
public:
    // Forgets the contributions, keeping the memory.
    void clear() noexcept;

    // Appends a contribution. Precondition: matrix.size() is dofs.size()² (std::invalid_argument otherwise).
    void append(std::span<const NodeDof> dofs, std::span<const double> matrix);

    [[nodiscard]] std::size_t size() const noexcept { return m_dofOffsets.size() - 1; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] std::span<const NodeDof> dofs(std::size_t i) const noexcept
    {
        return {m_dofs.data() + m_dofOffsets[i], m_dofOffsets[i + 1] - m_dofOffsets[i]};
    }
    // The n × n matrix of contribution i, row-major: term (r, c) at r·n + c, n = dofs(i).size().
    [[nodiscard]] std::span<const double> matrix(std::size_t i) const noexcept
    {
        const std::size_t n = m_dofOffsets[i + 1] - m_dofOffsets[i];
        return {m_values.data() + m_valueOffsets[i], n * n};
    }

private:
    std::vector<std::size_t> m_dofOffsets{0};
    std::vector<NodeDof> m_dofs;
    std::vector<std::size_t> m_valueOffsets{0};
    std::vector<double> m_values;
};

} // namespace oos::linalg
