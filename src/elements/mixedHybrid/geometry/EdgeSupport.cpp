#include "oos/elements/mixedHybrid/geometry/EdgeSupport.hpp"

namespace oos::elements::mixedHybrid
{

EdgeSupport EdgeSupport::simple(std::size_t across)
{
    EdgeSupport s = pinned();
    s.rotations[across] = true;
    return s;
}

EdgeSupport EdgeSupport::symmetry(std::size_t axis)
{
    EdgeSupport s{{false, false, false}, {true, true, true}};
    s.translations[axis] = true;
    s.rotations[axis] = false;
    return s;
}

EdgeSupport EdgeSupport::antisymmetry(std::size_t axis)
{
    EdgeSupport s{{true, true, true}, {false, false, false}};
    s.translations[axis] = false;
    s.rotations[axis] = true;
    return s;
}

std::vector<int> EdgeSupport::components() const
{
    std::vector<int> out;
    for (int c = 0; c < 3; ++c)
        if (translations[static_cast<std::size_t>(c)])
            out.push_back(c);
    for (int c = 0; c < 3; ++c)
        if (rotations[static_cast<std::size_t>(c)])
            out.push_back(c + 3);
    return out;
}

void addEdgeSupport(std::map<EdgeKey, EdgeSupport>& edgeSupports, std::size_t a, std::size_t b,
                    const EdgeSupport& support)
{
    EdgeSupport& all = edgeSupports[EdgeKey::of(a, b)];
    for (std::size_t c = 0; c < 3; ++c)
    {
        all.translations[c] = all.translations[c] || support.translations[c];
        all.rotations[c] = all.rotations[c] || support.rotations[c];
    }
}

EdgeSupport edgeHeld(const std::map<EdgeKey, EdgeSupport>& statedOnEdges, const EdgeKey& edge, std::size_t shellCount,
                     const std::array<bool, 6>& heldAtLower, const std::array<bool, 6>& heldAtHigher,
                     bool readGridSupportsAlongEdges)
{
    if (!statedOnEdges.empty())
    {
        const auto it = statedOnEdges.find(edge);
        return it == statedOnEdges.end() ? EdgeSupport{} : it->second;
    }
    EdgeSupport s;
    const bool external = shellCount == 1;
    if (external && readGridSupportsAlongEdges)
        for (std::size_t c = 0; c < 3; ++c)
        {
            s.translations[c] = heldAtLower[c] && heldAtHigher[c];
            s.rotations[c] = heldAtLower[c + 3] && heldAtHigher[c + 3];
        }
    return s;
}

} // namespace oos::elements::mixedHybrid
