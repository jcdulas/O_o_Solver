#include "oos/reduction/external/ReplacedElements.hpp"

#include "oos/reduction/external/ReplacedElementFamily.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace oos::reduction
{

ReplacedElements::ReplacedElements(const elements::ElementRegistry& base, std::vector<base::EntityId> replaced)
    : m_ids(std::move(replaced))
{
    std::sort(m_ids.begin(), m_ids.end());
    m_ids.erase(std::unique(m_ids.begin(), m_ids.end()), m_ids.end());
    for (const elements::ElementFamily* family : base.families())
        m_registry.add(std::make_unique<ReplacedElementFamily>(*family, m_ids));
}

bool ReplacedElements::isReplaced(base::EntityId elementId) const noexcept
{
    return std::binary_search(m_ids.begin(), m_ids.end(), elementId);
}

} // namespace oos::reduction
