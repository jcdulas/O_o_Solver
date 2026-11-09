#include "oos/api/ModelInput.hpp"

#include "oos/api/Analyses.hpp"
#include "oos/base/arrow/TablePackage.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/input/contract/TableRole.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace oos::api
{

void ModelInput::setVersions(std::int64_t structureVersion, std::int64_t contentVersion)
{
    m_builder.setVersions(structureVersion, contentVersion);
}

void ModelInput::setSourceCards(std::vector<SourceCard> cards)
{
    std::vector<input::SourceCard> source;
    source.reserve(cards.size());
    for (SourceCard& card : cards)
        source.push_back({std::move(card.card), card.count});
    m_builder.setSourceCards(std::move(source));
}

void ModelInput::setTable(const std::string& name, std::vector<std::uint8_t> package, std::vector<AddedColumn> added)
{
    const std::optional<input::TableRole> role = input::findTableRole(name);
    if (!role)
        throw base::ContractError("setTable: '" + name +
                                  "' is not a table of the model snapshot; expected a name such as nodes, "
                                  "singlePointConstrTable or subcasesTable (input/contract/TableRole.cpp lists them)");
    base::ImportedTable table = base::importTablePackage(std::move(package));
    if (!added.empty())
    {
        // The added columns are appended to the table, whose own columns are shared: the source table is kept alive
        // by the new one.
        std::vector<input::AddedRealColumn> columns;
        columns.reserve(added.size());
        for (AddedColumn& column : added)
            columns.push_back({std::move(column.name), std::move(column.values)});
        auto source = std::make_shared<base::ImportedTable>(std::move(table));
        table = input::appendRealColumns(name, *source, source, std::move(columns));
    }
    m_builder.setTable(*role, std::move(table));
}

void ModelInput::addElementTable(const std::string& card, std::vector<std::uint8_t> package)
{
    m_builder.addElementBatch(card, base::importTablePackage(std::move(package)));
}

void ModelInput::addPropertyTable(const std::string& card, std::vector<std::uint8_t> package)
{
    m_builder.addPropertyTable(card, base::importTablePackage(std::move(package)));
}

void ModelInput::addMethodTable(const std::string& card, const std::string& idColumn, std::vector<std::uint8_t> package)
{
    m_builder.addMethodTable(card, idColumn, base::importTablePackage(std::move(package)));
}

void ModelInput::addMaterialTable(const std::string& card, std::vector<std::uint8_t> package)
{
    m_builder.addMaterialCardTable(card, base::importTablePackage(std::move(package)));
}

void ModelInput::addLoadCardTable(const std::string& card, std::vector<std::uint8_t> package)
{
    m_builder.addLoadCardTable(card, base::importTablePackage(std::move(package)));
}

void ModelInput::addAeroCardTable(const std::string& card, const std::string& idColumn,
                                  std::vector<std::uint8_t> package)
{
    m_builder.addAeroCardTable(card, idColumn, base::importTablePackage(std::move(package)));
}

void ModelInput::setDisplacementRotations(std::vector<double> rotations)
{
    // The snapshot borrows the values; the shared vector is their owner and lives as long as the snapshot.
    auto owner = std::make_shared<const std::vector<double>>(std::move(rotations));
    m_builder.setCdRotations({std::span<const double>(owner->data(), owner->size()), owner});
}

void ModelInput::setOffsetDefinition(const std::string& value)
{
    m_builder.setOffsetDefinitionValue(value);
}

void ModelInput::setElementFormulation(const std::string& card, const std::string& formulation)
{
    // Only the neutral card types the element library computes, as elementFormulations() and the result tables name
    // them; checked here already so that the error comes at the call rather than at build().
    const std::vector<ElementFormulations> known = elementFormulations();
    const bool isKnown =
        std::any_of(known.begin(), known.end(), [&](const ElementFormulations& entry) { return entry.card == card; });
    if (!isKnown)
        throw std::invalid_argument("setElementFormulation: '" + card +
                                    "' is not an element card type of the solver; give its neutral name, as "
                                    "elementFormulations() lists them (for example ShellQuad4)");
    m_builder.setElementFormulation(card, formulation);
}

std::shared_ptr<const input::ModelSnapshot> ModelInput::build()
{
    return std::make_shared<const input::ModelSnapshot>(m_builder.build());
}

} // namespace oos::api
