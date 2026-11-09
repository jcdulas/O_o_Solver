#include "oos/reduction/absorption/SpiderCatalog.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/units/StampHasher.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>

namespace oos::reduction
{

namespace
{

constexpr base::NullPolicy ALLOW = base::NullPolicy::Allow;

// The number of the point of a fastener side, which the stamp of the side mixes in before its grid points: every side
// has one point, its piercing point, numbered 0. Kept in the stamp so that the stamps, and the condensation caches
// keyed by them, stay those of the earlier builds.
constexpr std::int64_t PIERCING_POINT_NUMBER = 0;

template <typename T>
void sortAndMakeUnique(std::vector<T>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

SpiderKind kindOfCard(std::string_view card)
{
    if (card == "RigidInterpolation")
        return SpiderKind::RigidInterpolation;
    if (card == "RigidBodyStar")
        return SpiderKind::RigidBodyStar;
    throw base::ContractError(base::concatenate(
        {"spiders: expected the card RigidBodyStar or RigidInterpolation, found '", card,
         "'; the table does not follow the data contract: take the snapshot again with a O_o_Reader that follows "
         "it"}));
}

// The rigid spiders of O_o_Reader's spiders table.
void readRigidSpiders(const base::ContractReader& table, std::vector<Spider>& spiders)
{
    const std::span<const std::int64_t> components = table.values<std::int64_t>("componentId", ALLOW);
    const base::ValidityView hasComponent = table.validity("componentId");
    const base::DictionaryView cards = table.dictionaryTexts("card");
    const std::span<const std::int32_t> elementIds = table.values<std::int32_t>("elementId");
    const std::span<const std::int32_t> references = table.values<std::int32_t>("referenceNodeId", ALLOW);
    const base::Utf8View referenceDofs = table.texts("referenceDofs", ALLOW);
    const base::ValidityView hasReferenceDofs = table.validity("referenceDofs");
    const base::ListView<std::int32_t> legs = table.lists<std::int32_t>("legNodeIds", ALLOW);
    const base::TextListView legDofs = table.textLists("legDofs", ALLOW);
    const base::ValidityView hasLegDofs = table.validity("legDofs");
    const base::ValidityView hasReference = table.validity("referenceNodeId");
    const base::ListView<double> legWeights = table.lists<double>("legWeights", ALLOW);
    const base::ValidityView hasLegWeights = table.validity("legWeights");
    const base::BooleanView isSpider = table.booleans("isSpider", ALLOW);
    const base::ValidityView hasIsSpider = table.validity("isSpider");
    const base::DictionaryView reasons = table.dictionaryTexts("reason", ALLOW);
    const base::ValidityView hasReason = table.validity("reason");
    const base::BooleanView isEligible = table.booleans("isEligible", ALLOW);
    const base::ValidityView hasIsEligible = table.validity("isEligible");
    const base::ListView<std::int32_t> blockingNodeIds = table.lists<std::int32_t>("blockingNodeIds", ALLOW);
    const base::ListView<std::int32_t> blockingConnections = table.lists<std::int32_t>("blockingConnections", ALLOW);
    const std::span<const std::int64_t> revisions = table.values<std::int64_t>("revision");

    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        Spider spider;
        spider.kind = kindOfCard(cards[row]);
        spider.key = SpiderKey{elementIds[r], FastenerSide::None};
        spider.card = std::string(cards[row]);
        spider.referenceNodeId = hasReference.isValid(row) ? references[r] : 0;
        spider.referenceDofs = hasReferenceDofs.isValid(row) ? std::string(referenceDofs[row]) : std::string();
        const std::span<const std::int32_t> legIds = legs[row];
        // The leg components (and the weights of an RigidInterpolation) follow the legs one to one (O_o_Reader: "as the
        // card gives them"); a table where they do not is not the contract's.
        if ((hasLegDofs.isValid(row) && legDofs.count(row) != static_cast<std::int64_t>(legIds.size())) ||
            (spider.kind == SpiderKind::RigidInterpolation && hasLegWeights.isValid(row) &&
             legWeights[row].size() != legIds.size()))
            throw base::ContractError(base::concatenate(
                {"spiders: expected one leg component and (RigidInterpolation) one weight per leg of element ",
                 std::to_string(elementIds[r]), ", found ", std::to_string(legIds.size()), " legs, ",
                 std::to_string(legDofs.count(row)), " components and ", std::to_string(legWeights[row].size()),
                 " weights; the table does not follow the data contract: take the snapshot again with a O_o_Reader "
                 "that follows it"}));
        spider.legNodeIds.assign(legIds.begin(), legIds.end());
        sortAndMakeUnique(spider.legNodeIds);
        if (hasComponent.isValid(row))
            spider.componentId = components[r];
        spider.isSpider = hasIsSpider.isValid(row) && isSpider[row];
        if (!spider.isSpider)
            spider.notSpiderReason = hasReason.isValid(row) ? std::string(reasons[row]) : std::string("not a spider");
        spider.hasEligibility = hasIsEligible.isValid(row);
        spider.isEligible = spider.hasEligibility && isEligible[row];
        const std::span<const std::int32_t> blockingLegs = blockingNodeIds[row];
        spider.blockingNodeIds.assign(blockingLegs.begin(), blockingLegs.end());
        sortAndMakeUnique(spider.blockingNodeIds);
        const std::span<const std::int32_t> blocking = blockingConnections[row];
        spider.blockingConnections.assign(blocking.begin(), blocking.end());
        sortAndMakeUnique(spider.blockingConnections);
        spider.revision = revisions[r];
        spiders.push_back(std::move(spider));
    }
}

// The eligibility of each fastener side in O_o_Reader's own rows (one per fastener, side and point): a side is
// eligible when each of its rows is, blocked by the union of their connections; its component when its rows agree.
struct SideEligibility
{
    bool isEligible = true;
    std::vector<base::EntityId> blockingConnections;
    std::optional<std::int64_t> componentId;
    bool hasMixedComponents = false;
};

std::map<SpiderKey, SideEligibility> sideEligibility(const base::ContractReader& table, bool& hasEligibility)
{
    std::map<SpiderKey, SideEligibility> sides;
    hasEligibility = table.hasColumn("isEligible");
    const std::span<const std::int32_t> elementIds = table.values<std::int32_t>("elementId");
    const base::DictionaryView sideNames = table.dictionaryTexts("side");
    const bool hasComponents = table.hasColumn("componentId");
    const bool hasBlocking = table.hasColumn("blockingConnections");
    std::optional<base::BooleanView> isEligible;
    std::optional<base::ValidityView> isEligibilityKnown;
    if (hasEligibility)
    {
        isEligible = table.booleans("isEligible", ALLOW);
        isEligibilityKnown = table.validity("isEligible");
    }
    std::optional<base::ListView<std::int32_t>> blocking;
    if (hasBlocking)
        blocking = table.lists<std::int32_t>("blockingConnections", ALLOW);
    std::span<const std::int64_t> components;
    std::optional<base::ValidityView> hasComponent;
    if (hasComponents)
    {
        components = table.values<std::int64_t>("componentId", ALLOW);
        hasComponent = table.validity("componentId");
    }
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        const auto r = static_cast<std::size_t>(row);
        const FastenerSide side = sideNames[row] == "A" ? FastenerSide::A : FastenerSide::B;
        SideEligibility& entry = sides[SpiderKey{elementIds[r], side}];
        if (isEligible)
            entry.isEligible = entry.isEligible && isEligibilityKnown->isValid(row) && (*isEligible)[row];
        if (blocking)
        {
            const std::span<const std::int32_t> connections = (*blocking)[row];
            entry.blockingConnections.insert(entry.blockingConnections.end(), connections.begin(), connections.end());
        }
        if (hasComponent && hasComponent->isValid(row))
        {
            if (entry.componentId && *entry.componentId != components[r])
                entry.hasMixedComponents = true;
            entry.componentId = components[r];
        }
    }
    for (auto& [key, entry] : sides)
    {
        sortAndMakeUnique(entry.blockingConnections);
        if (entry.hasMixedComponents)
            entry.componentId.reset();
    }
    return sides;
}

// The fastener sides of the solver's piercing layout (FastenerPiercingTable), with O_o_Reader's eligibility.
void readFastenerSides(const input::FastenerPiercingTable& piercings, std::vector<Spider>& spiders)
{
    bool hasEligibility = false;
    const std::map<SpiderKey, SideEligibility> eligibility = sideEligibility(piercings.reader(), hasEligibility);
    const std::span<const std::int32_t> elementIds = piercings.fastenerIds();
    const base::ListView<std::int32_t> grids = piercings.grids();
    const base::ListView<double> weights = piercings.weights();
    std::int64_t first = 0;
    while (first < piercings.rowCount())
    {
        std::int64_t last = first + 1;
        const std::int32_t elementId = elementIds[static_cast<std::size_t>(first)];
        while (last < piercings.rowCount() && elementIds[static_cast<std::size_t>(last)] == elementId &&
               piercings.isSideA(last) == piercings.isSideA(first))
            ++last;
        Spider spider;
        spider.kind = SpiderKind::FastenerSide;
        spider.key = SpiderKey{elementId, piercings.isSideA(first) ? FastenerSide::A : FastenerSide::B};
        spider.card = std::string(piercings.card(first));
        spider.referenceNodeId =
            piercings.isPointGridGiven().isValid(first) ? piercings.pointGrids()[static_cast<std::size_t>(first)] : 0;
        // The stamp of the side: what its interpolation depends on, in table order (the points by number, each with
        // its grid points and the bits of its weights): equal content gives equal stamps.
        StampHasher stamp;
        stamp.mix(static_cast<std::int64_t>(spider.key.elementId));
        stamp.mix(static_cast<std::int64_t>(spider.key.side));
        for (std::int64_t row = first; row < last; ++row)
        {
            stamp.mix(PIERCING_POINT_NUMBER);
            for (const std::int32_t grid : grids[row])
            {
                spider.legNodeIds.push_back(grid);
                stamp.mix(static_cast<std::int64_t>(grid));
            }
            for (const double weight : weights[row])
            {
                std::uint64_t bits = 0;
                std::memcpy(&bits, &weight, sizeof(bits));
                stamp.mix(bits);
            }
        }
        sortAndMakeUnique(spider.legNodeIds);
        spider.revision = stamp.stamp();
        spider.isSpider = !spider.legNodeIds.empty();
        if (!spider.isSpider)
            spider.notSpiderReason = "noPiercedElement";
        if (const auto found = eligibility.find(spider.key); found != eligibility.end())
        {
            spider.hasEligibility = hasEligibility;
            spider.isEligible = hasEligibility && found->second.isEligible;
            spider.blockingConnections = found->second.blockingConnections;
            spider.componentId = found->second.componentId;
        }
        spiders.push_back(std::move(spider));
        first = last;
    }
}

} // namespace

SpiderCatalog::SpiderCatalog(const base::ContractReader* spiders, const input::FastenerPiercingTable* piercings)
    : m_hasRigidSpiders(spiders != nullptr)
{
    if (spiders != nullptr)
        readRigidSpiders(*spiders, m_spiders);
    if (piercings != nullptr)
        readFastenerSides(*piercings, m_spiders);
    std::sort(m_spiders.begin(), m_spiders.end(), [](const Spider& a, const Spider& b) { return a.key < b.key; });
    for (std::size_t k = 1; k < m_spiders.size(); ++k)
        if (m_spiders[k].key == m_spiders[k - 1].key)
            throw base::ContractError(base::concatenate(
                {"spiders: expected each RigidBodyStar, RigidInterpolation and fastener side once, found element ",
                 std::to_string(m_spiders[k].key.elementId),
                 " twice (the spiders and fastener piercing tables name an element ID once; "
                 "take the spiders table without groups=)"}));
}

std::optional<std::size_t> SpiderCatalog::indexOf(const SpiderKey& key) const noexcept
{
    const auto found = std::lower_bound(m_spiders.begin(), m_spiders.end(), key,
                                        [](const Spider& spider, const SpiderKey& k) { return spider.key < k; });
    if (found == m_spiders.end() || found->key != key)
        return std::nullopt;
    return static_cast<std::size_t>(found - m_spiders.begin());
}

} // namespace oos::reduction
