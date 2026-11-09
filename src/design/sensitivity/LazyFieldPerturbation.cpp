#include "oos/design/sensitivity/LazyFieldPerturbation.hpp"

#include <utility>

namespace oos::design
{

LazyFieldPerturbation::LazyFieldPerturbation(FieldPerturbation built) : m_value(std::move(built))
{
    std::call_once(m_once, [] {});
}

LazyFieldPerturbation::LazyFieldPerturbation(std::function<FieldPerturbation()> make) : m_make(std::move(make)) {}

const FieldPerturbation& LazyFieldPerturbation::get() const
{
    std::call_once(m_once, [this] { m_value.emplace(m_make()); });
    return *m_value;
}

} // namespace oos::design
