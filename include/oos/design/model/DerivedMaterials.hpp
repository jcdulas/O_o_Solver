#pragma once
#include "oos/design/model/DesignModel.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"

#include <array>
#include <optional>

namespace oos::design
{
// The original blank E, G or nu of an isotropic material remains dependent until a relation explicitly drives it.
// The same Hooke-law identities are applied to the material card and the summary read by different element families.
class DerivedMaterials
{
public:
    void bind(const DesignModel& model, const input::ModelSnapshot& snapshot);
    [[nodiscard]] bool reaches(std::size_t relation) const noexcept;
    void addCells(input::ColumnOverlay& overlay, std::span<const double> fields,
                  std::optional<std::size_t> relation = {}) const;

private:
    struct Group
    {
        std::int64_t cardRow = -1;
        std::int64_t summaryRow = -1;
        std::uint8_t blank = 0;
        std::array<double, 3> base{};
        std::array<std::optional<std::size_t>, 3> relations{};
    };
    std::vector<Group> m_groups;
    std::vector<std::optional<std::size_t>> m_relationGroups;
};
} // namespace oos::design
