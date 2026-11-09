#include "oos/design/model/DesignStamps.hpp"

#include <bit>
#include <cstddef>
#include <string>

namespace oos::design
{

namespace
{

// FNV-1a, 64 bits (Fowler, Noll, Vo): offset basis and prime of the 64-bit variant.
constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
// What a shape variable's part of a stamp starts with, so that it never reads as the identity of a field.
const std::string SHAPE_IDENTITY = "DesignNodeMove";

class Fnv1a
{
public:
    void mixByte(std::uint8_t byte) noexcept
    {
        m_hash ^= byte;
        m_hash *= FNV_PRIME;
    }
    void mixWord(std::uint64_t word) noexcept
    {
        for (int shift = 0; shift < 64; shift += 8)
            mixByte(static_cast<std::uint8_t>(word >> shift));
    }
    void mixText(const std::string& text) noexcept
    {
        mixWord(text.size());
        for (const char c : text)
            mixByte(static_cast<std::uint8_t>(c));
    }
    [[nodiscard]] std::uint64_t value() const noexcept { return m_hash; }

private:
    std::uint64_t m_hash = FNV_OFFSET_BASIS;
};

} // namespace

std::vector<std::uint64_t> computeDesignStamps(const DesignModel& model, const FieldTargets& targets,
                                               const std::vector<FieldValue>& fields, std::span<const double> shapes)
{
    const std::size_t unitCount = targets.unitIds().size();
    std::vector<Fnv1a> hashes(unitCount);
    std::vector<bool> isDriven(unitCount, false);
    for (std::size_t index = 0; index < model.relations().size(); ++index)
    {
        const DesignRelation& relation = model.relations()[index];
        for (const std::size_t unit : targets[index].units)
        {
            Fnv1a& hash = hashes[unit];
            isDriven[unit] = true;
            hash.mixWord(static_cast<std::uint64_t>(relation.targetKind));
            hash.mixText(relation.targetCard);
            hash.mixWord(static_cast<std::uint64_t>(static_cast<std::uint32_t>(relation.targetId)));
            hash.mixText(relation.field);
            hash.mixWord(static_cast<std::uint64_t>(static_cast<std::int64_t>(relation.fieldItem)));
            hash.mixWord(std::bit_cast<std::uint64_t>(fields[index].value));
        }
    }
    const std::vector<ShapeTarget>& shapeTargets = targets.shapeTargets();
    if (!shapeTargets.empty() && unitCount > 0)
    {
        const std::vector<double> positions = targets.movedPositions(model.shapes(), shapes);
        const std::vector<ShapeBasis>& bases = model.shapes().bases();
        for (std::size_t b = 0; b < shapeTargets.size(); ++b)
            for (const std::size_t unit : shapeTargets[b].units)
            {
                Fnv1a& hash = hashes[unit];
                isDriven[unit] = true;
                hash.mixText(SHAPE_IDENTITY);
                for (std::size_t m = 0; m < shapeTargets[b].points.size(); ++m)
                {
                    hash.mixWord(static_cast<std::uint64_t>(static_cast<std::uint32_t>(bases[b].moves[m].gridId)));
                    for (std::size_t k = 0; k < 3; ++k)
                        hash.mixWord(std::bit_cast<std::uint64_t>(positions[3 * shapeTargets[b].points[m] + k]));
                }
            }
    }
    std::vector<std::uint64_t> stamps(unitCount, EMPTY_DESIGN_STAMP);
    for (std::size_t unit = 0; unit < unitCount; ++unit)
    {
        if (!isDriven[unit])
            continue;
        const std::uint64_t value = hashes[unit].value();
        stamps[unit] = value == EMPTY_DESIGN_STAMP ? 1 : value;
    }
    return stamps;
}

} // namespace oos::design
