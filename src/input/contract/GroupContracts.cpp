#include "oos/input/contract/GroupContracts.hpp"

#include <string>
#include <utility>

namespace oos::input
{

namespace
{

base::ColumnContract required(std::string name, std::string type)
{
    return {std::move(name), std::move(type), false, {}};
}

} // namespace

std::string_view groupTableName(GroupTable table) noexcept
{
    switch (table)
    {
    case GroupTable::Elements:
        return "groupElements";
    case GroupTable::Nodes:
        return "groupNodes";
    case GroupTable::Boundaries:
        return "componentBoundaries";
    case GroupTable::Summary:
        return "groupRevisions";
    }
    return "";
}

base::TableContract groupTableContract(GroupTable table)
{
    const std::string name(groupTableName(table));
    switch (table)
    {
    case GroupTable::Elements:
        return {name, {required("group", "utf8"), required("elementId", "int32")}};
    case GroupTable::Nodes:
        return {name, {required("group", "utf8"), required("nodeId", "int32")}};
    case GroupTable::Boundaries:
        return {name, {required("group", "utf8"), required("nodeId", "int32"), required("isInterface", "bool")}};
    case GroupTable::Summary:
        return {name, {required("group", "utf8"), required("revision", "int64")}};
    }
    return {};
}

base::ContractReader readGroupTable(GroupTable kind, base::ImportedTable table)
{
    return {std::move(table), groupTableContract(kind)};
}

} // namespace oos::input
