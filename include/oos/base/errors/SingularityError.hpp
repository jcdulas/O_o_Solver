#pragma once

#include "oos/base/errors/SolverError.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oos::base
{

// One degree of freedom as the user reads it in messages: a grid or scalar point and a component 1..6 (0 for a scalar
// point).
struct GridComponent
{
    std::int32_t grid = 0;
    int component = 0;

    friend bool operator==(const GridComponent&, const GridComponent&) = default;
};

// A mechanism: a motion of the structure that costs no strain energy, as the DOFs it moves most and their amplitudes
// (normalised so that the largest is 1).
struct MechanismShape
{
    std::vector<GridComponent> dofs;
    std::vector<double> amplitudes; // one per entry of dofs
};

// The stiffness matrix is singular or too ill-conditioned to be factorised (a pivot below the threshold, a MAXRATIO
// above its limit); the message names the grid points and components, the error also carries them and the mechanism
// shapes found, so that the caller can show them.
class SingularityError : public SolverError
{
public:
    explicit SingularityError(const std::string& message, std::vector<GridComponent> dofs = {},
                              std::vector<MechanismShape> mechanisms = {})
        : SolverError(message), m_dofs(std::move(dofs)), m_mechanisms(std::move(mechanisms))
    {
    }

    // The singular or ill-conditioned DOFs, in the order the factorisation met them.
    [[nodiscard]] const std::vector<GridComponent>& dofs() const noexcept { return m_dofs; }
    [[nodiscard]] const std::vector<MechanismShape>& mechanisms() const noexcept { return m_mechanisms; }

private:
    std::vector<GridComponent> m_dofs;
    std::vector<MechanismShape> m_mechanisms;
};

} // namespace oos::base
