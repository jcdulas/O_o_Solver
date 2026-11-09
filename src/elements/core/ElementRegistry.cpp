#include "oos/elements/core/ElementRegistry.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"
#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/fasteners/fastener/FastenerElement.hpp"
#include "oos/elements/fasteners/weld/WeldElement.hpp"
#include "oos/elements/masses/massDofProperty/MassDofPropertyElement.hpp"
#include "oos/elements/masses/massDofInline/MassDofInlineElement.hpp"
#include "oos/elements/masses/massScalarProperty/MassScalarPropertyElement.hpp"
#include "oos/elements/masses/massScalarInline/MassScalarInlineElement.hpp"
#include "oos/elements/masses/pointMassMatrix/PointMassMatrixElement.hpp"
#include "oos/elements/masses/pointMassInertia/PointMassInertiaElement.hpp"
#include "oos/elements/rods/rod2/Rod2Element.hpp"
#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"
#include "oos/elements/rods/tube2/Tube2Element.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4Element.hpp"
#include "oos/elements/shells/shellQuad8/ShellQuad8Element.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingElement.hpp"
#include "oos/elements/shells/shearPanel4/ShearPanel4Element.hpp"
#include "oos/elements/shells/shellTria3/ShellTria3Element.hpp"
#include "oos/elements/shells/shellTria6/ShellTria6Element.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingElement.hpp"
#include "oos/elements/solids/solidHexa/SolidHexaElement.hpp"
#include "oos/elements/solids/solidPenta/SolidPentaElement.hpp"
#include "oos/elements/solids/solidPyramid/SolidPyramidElement.hpp"
#include "oos/elements/solids/solidTetra/SolidTetraElement.hpp"
#include "oos/elements/springs/bushing/BushingElement.hpp"
#include "oos/elements/springs/bushing1D/Bushing1DElement.hpp"
#include "oos/elements/springs/gap2/Gap2Element.hpp"
#include "oos/elements/springs/springDofProperty/SpringDofPropertyElement.hpp"
#include "oos/elements/springs/springDofInline/SpringDofInlineElement.hpp"
#include "oos/elements/springs/springScalarProperty/SpringScalarPropertyElement.hpp"
#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

// The formulations of a card as a message lists them: "ClassicFormulation" or "ClassicFormulation, Other".
std::string formulationList(const std::vector<std::string_view>& names)
{
    std::string list;
    for (const std::string_view name : names)
    {
        if (!list.empty())
            list.append(", ");
        list.append(name);
    }
    return list;
}

base::Diagnostic selectionError(std::string card, std::string field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = "unknown-formulation";
    diagnostic.card = std::move(card);
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

} // namespace

ElementRegistry ElementRegistry::standard()
{
    ElementRegistry registry;
    // Springs.
    registry.add(std::make_unique<SpringDofPropertyElement>());
    registry.add(std::make_unique<SpringDofInlineElement>());
    registry.add(std::make_unique<SpringScalarPropertyElement>());
    registry.add(std::make_unique<SpringScalarInlineElement>());
    registry.add(std::make_unique<BushingElement>());
    registry.add(std::make_unique<Bushing1DElement>());
    registry.add(std::make_unique<Gap2Element>());
    // Masses.
    registry.add(std::make_unique<PointMassMatrixElement>());
    registry.add(std::make_unique<PointMassInertiaElement>());
    registry.add(std::make_unique<MassDofPropertyElement>());
    registry.add(std::make_unique<MassDofInlineElement>());
    registry.add(std::make_unique<MassScalarPropertyElement>());
    registry.add(std::make_unique<MassScalarInlineElement>());
    // Rods.
    registry.add(std::make_unique<Rod2Element>());
    registry.add(std::make_unique<Rod2InlineSectionElement>());
    registry.add(std::make_unique<Tube2Element>());
    // Bars and beams.
    registry.add(std::make_unique<Bar2Element>());
    registry.add(std::make_unique<Beam2Element>());
    // Fasteners.
    registry.add(std::make_unique<FastenerElement>());
    registry.add(std::make_unique<WeldElement>());
    // Shells.
    registry.add(std::make_unique<ShellQuad4Element>());
    registry.add(std::make_unique<ShellTria3Element>());
    registry.add(std::make_unique<ShearPanel4Element>());
    registry.add(std::make_unique<ShellQuad8Element>());
    registry.add(std::make_unique<ShellTria6Element>());
    registry.add(std::make_unique<ShellQuad4DrillingElement>());
    registry.add(std::make_unique<ShellTria3DrillingElement>());
    // Solids.
    registry.add(std::make_unique<SolidHexaElement>());
    registry.add(std::make_unique<SolidPentaElement>());
    registry.add(std::make_unique<SolidTetraElement>());
    registry.add(std::make_unique<SolidPyramidElement>());
    return registry;
}

void ElementRegistry::add(std::unique_ptr<ElementFamily> family)
{
    const std::string_view card = family->card();
    const std::string_view formulation = family->formulation();
    for (CardFamilies& entry : m_cards)
    {
        if (entry.card != card)
            continue;
        for (const std::unique_ptr<ElementFamily>& known : entry.formulations)
        {
            if (known->formulation() == formulation)
                throw std::invalid_argument(
                    base::concatenate({"element registry: the card ", card, " has a family of the formulation ",
                                       formulation, " already; register each formulation of a card once"}));
        }
        entry.formulations.push_back(std::move(family));
        rebuildChosen();
        return;
    }
    CardFamilies entry;
    entry.card = std::string(card);
    entry.formulations.push_back(std::move(family));
    m_cards.push_back(std::move(entry));
    rebuildChosen();
}

ElementRegistry ElementRegistry::forModel(const input::ModelSnapshot& snapshot)
{
    ElementRegistry registry = standard();
    for (const CardFamilies& card : registry.m_cards)
        registry.selectFromModel(snapshot, card);
    // A choice the snapshot gives for a card without a family is refused too: it would otherwise go unused unnoticed.
    for (const input::ElementFormulationChoice& choice : snapshot.elementFormulations())
    {
        if (registry.findCard(choice.card) == nullptr)
            registry.m_problems.push_back(selectionError(
                std::string(choice.card), "",
                base::concatenate({"expected an element card of O_o_Solver for the formulation ", choice.formulation,
                                   ", found ", std::string(choice.card),
                                   ", which has no element family; give the formulation of a supported card"})));
    }
    return registry;
}

void ElementRegistry::selectFromModel(const input::ModelSnapshot& snapshot, const CardFamilies& card)
{
    // Formulation parameters use the exact neutral card name, such as ShellQuad4Formulation.

    const std::string parameterName = std::string(card.card) + "Formulation";
    const std::string paramCard = base::concatenate({"Parameter ", parameterName});
    const std::string cardName = std::string(card.card);
    const input::ParamTable* params = snapshot.hasTable(input::TableRole::Params) ? &snapshot.params() : nullptr;
    // A formulation holds for the whole model: a subcase cannot change it.
    if (params != nullptr)
    {
        if (const std::optional<std::int32_t> subcase = params->findSubcaseGiving(parameterName))
            m_problems.push_back(selectionError(
                paramCard, parameterName,
                base::concatenate({"expected Parameter ", parameterName,
                                   " in the model records or in the global control section (the formulation of the ",
                                   cardName, " holds for the whole model), found it in the control section of subcase ",
                                   std::to_string(*subcase), "; move it above the first SubcaseStart"})));
    }

    // The program that built the snapshot first, then the model parameters.
    std::string chosen;
    std::string source;
    std::string field;
    for (const input::ElementFormulationChoice& choice : snapshot.elementFormulations())
    {
        if (choice.card == card.card)
        {
            chosen = choice.formulation;
            source = base::concatenate({"the formulation chosen for the ", cardName});
        }
    }
    if (chosen.empty() && params != nullptr)
    {
        if (const std::optional<input::ParamValue> value = params->find(parameterName))
        {
            if (value->kind != input::ParamValue::Kind::Text)
            {
                m_problems.push_back(selectionError(
                    paramCard, parameterName,
                    base::concatenate({"expected Parameter ", parameterName, " to name a formulation of the ", cardName,
                                       " (", formulationList(formulationsOf(card.card)),
                                       "), found a number; give one of these names"})));
                return;
            }
            chosen = value->text;
            source = paramCard;
            field = parameterName;
        }
    }
    if (chosen.empty())
        return;
    if (!select(card.card, chosen))
        m_problems.push_back(selectionError(field.empty() ? cardName : paramCard, field,
                                            base::concatenate({"expected ", source, " to be one of its formulations (",
                                                               formulationList(formulationsOf(card.card)), "), found ",
                                                               chosen, "; give one of these names"})));
}

bool ElementRegistry::select(std::string_view card, std::string_view formulation)
{
    // A formulation is a neutral name of the model format: compared exactly.
    const std::string_view name = formulation;
    for (CardFamilies& entry : m_cards)
    {
        if (entry.card != card)
            continue;
        for (std::size_t i = 0; i < entry.formulations.size(); ++i)
        {
            if (entry.formulations[i]->formulation() == name)
            {
                entry.chosen = i;
                rebuildChosen();
                return true;
            }
        }
        return false;
    }
    return false;
}

const ElementRegistry::CardFamilies* ElementRegistry::findCard(std::string_view card) const noexcept
{
    for (const CardFamilies& entry : m_cards)
    {
        if (entry.card == card)
            return &entry;
    }
    return nullptr;
}

const ElementFamily* ElementRegistry::find(std::string_view card) const noexcept
{
    const CardFamilies* entry = findCard(card);
    return entry != nullptr ? entry->formulations[entry->chosen].get() : nullptr;
}

std::vector<std::string_view> ElementRegistry::formulationsOf(std::string_view card) const
{
    std::vector<std::string_view> names;
    if (const CardFamilies* entry = findCard(card))
    {
        for (const std::unique_ptr<ElementFamily>& family : entry->formulations)
            names.push_back(family->formulation());
    }
    return names;
}

void ElementRegistry::rebuildChosen()
{
    m_chosen.clear();
    for (const CardFamilies& entry : m_cards)
        m_chosen.push_back(entry.formulations[entry.chosen].get());
}

void ElementRegistry::addSupportTo(input::SupportCatalog& catalog) const
{
    for (const ElementFamily* family : m_chosen)
    {
        catalog.addCard(std::string(family->card()));
        for (std::string& card : family->propertyCards())
            catalog.addCard(std::move(card));
        for (input::FieldRule& rule : family->fieldRules())
            catalog.addFieldRule(std::move(rule));
    }
}

} // namespace oos::elements
