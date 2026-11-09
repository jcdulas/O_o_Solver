#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/snapshot/SupportCatalog.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::elements
{

// The element families of this build of O_o_Solver, by card and formulation: the assembler asks it for the family of
// each element batch of a snapshot. standard() registers every family written so far (ElementRegistry.cpp holds the
// list: adding an element type, or a formulation of one, adds one line there).
//
// A card may have several families, one per formulation (ElementFamily::formulation, ElementFormulation.hpp); the
// first one registered for a card is its default. The registry answers for one formulation per card, the chosen one:
// the default, or the one select() or forModel() chose. Every lookup (find, families, addSupportTo) sees only the
// chosen families.
//
// The registry also tells the support catalog what the element library supports (addSupportTo): the element cards,
// the property and material cards they read, and their field rules. input::SupportCatalog::current() cannot do it
// itself, as input lies below elements in the layering of the modules: the layer that builds the catalog for a run
// asks the registry.
//
// Thread-safety: immutable once built; concurrent lookups are safe. Ownership: owns its families.
class ElementRegistry
{
public:
    // Every family written so far, each card in its default formulation.
    [[nodiscard]] static ElementRegistry standard();
    // standard() in the formulations the model chooses: those the program that built the snapshot gave
    // (ModelSnapshot::elementFormulations), then the deck's PARAM F<card>, then the defaults. A choice the registry
    // cannot follow leaves the card in its default formulation and is reported by selectionProblems(), which the
    // support check of every run turns into refusals: nothing is computed with a formulation other than the chosen one.
    [[nodiscard]] static ElementRegistry forModel(const input::ModelSnapshot& snapshot);

    // Registers a family; the first family of a card is the card's default formulation, and the chosen one until
    // select() chooses another. Throws std::invalid_argument when a family of the same card and formulation is
    // registered already.
    void add(std::unique_ptr<ElementFamily> family);

    // Chooses the formulation of a card (compared in upper case). False, and nothing changed, when the card has no
    // family of that formulation.
    [[nodiscard]] bool select(std::string_view card, std::string_view formulation);

    // The family of a card in its chosen formulation, or nullptr when the card has none (the support catalog refuses
    // such a card).
    [[nodiscard]] const ElementFamily* find(std::string_view card) const noexcept;
    // The chosen family of every card, in the order the cards were first registered.
    [[nodiscard]] std::span<const ElementFamily* const> families() const noexcept { return m_chosen; }
    // The formulations a card has, its default first (empty for a card without a family).
    [[nodiscard]] std::vector<std::string_view> formulationsOf(std::string_view card) const;

    // The choices of forModel() it could not follow, one error each: a formulation the card does not have, a PARAM
    // F<card> whose value is not a name, a PARAM F<card> given in the Case Control of one subcase.
    [[nodiscard]] const std::vector<base::Diagnostic>& selectionProblems() const noexcept { return m_problems; }

    // Adds every element card, the property and material cards they read, and their field rules to `catalog`, those
    // of the chosen formulation of each card.
    void addSupportTo(input::SupportCatalog& catalog) const;

private:
    // The families of one card: its formulations in the order they were registered (the default first), and the index
    // of the chosen one.
    struct CardFamilies
    {
        std::string card;
        std::vector<std::unique_ptr<ElementFamily>> formulations;
        std::size_t chosen = 0;
    };

    [[nodiscard]] const CardFamilies* findCard(std::string_view card) const noexcept;
    // The choice of one card from the snapshot (its own choice, then its PARAM), reported in m_problems when it cannot
    // be followed.
    void selectFromModel(const input::ModelSnapshot& snapshot, const CardFamilies& card);
    void rebuildChosen();

    std::vector<CardFamilies> m_cards;
    std::vector<const ElementFamily*> m_chosen; // the chosen family of each card of m_cards
    std::vector<base::Diagnostic> m_problems;
};

} // namespace oos::elements
