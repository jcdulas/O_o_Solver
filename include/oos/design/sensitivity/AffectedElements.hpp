#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/design/model/DesignModel.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oos::design
{

// One element of an element set: its batch (index into ElementSet::batches()) and its row.
struct ElementRef
{
    std::size_t batch = 0;
    std::int64_t row = 0;

    friend bool operator==(const ElementRef&, const ElementRef&) = default;
    friend auto operator<=>(const ElementRef&, const ElementRef&) = default;
};

// The elements each designed field reaches (b_j = ∂f/∂x_j − ∂K/∂x_j·u is formed element by element on the
// elements a variable drives): a property field the elements of that property (the property card and ID, as
// ElementCatalog finds them), and for a shell property (PSHELL, CompProp, CompPropGlobal) the Weld whose sides read
// its thickness (elements::weldSideProperty: the thickness may set the weld's beam length); a material field the
// elements whose property or own card names the material in a column "matId", "matId1", "membraneMatId" ... (the plies
// of laminatePliesTable included); an element field that element. The pseudo-loads and the explicit parts of the
// responses are formed on these elements only; every other element has a zero derivative.
//
// An element set of any snapshot of the model may be used: an overlay changes values, never the rows or the batches.
//
// Thread-safety: immutable once built. Ownership: owns its lists (bounded by the designed elements).
class AffectedElements
{
public:
    AffectedElements(const DesignModel& model, const assembly::ElementSet& set);

    // The elements of a relation, in increasing (batch, row).
    [[nodiscard]] const std::vector<ElementRef>& of(std::size_t relation) const { return m_elements[relation]; }

private:
    std::vector<std::vector<ElementRef>> m_elements;
};

} // namespace oos::design
