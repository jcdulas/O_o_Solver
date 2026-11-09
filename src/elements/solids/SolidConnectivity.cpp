#include "oos/elements/solids/SolidConnectivity.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"

#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader gives an absent mid-side node as ID 0 (and row −1 in nodesRows).
constexpr std::int32_t ABSENT_NODE_ID = 0;

[[nodiscard]] std::string fieldOf(std::size_t node)
{
    return std::string("node").append(std::to_string(node + 1));
}

} // namespace

SolidConnectivity::SolidConnectivity(const ElementBatchView& view)
    : rows(view.batch().nodeRows()), ids(view.batch().nodes()), valid(view.columns().validity("nodesRows"))
{
}

Resolution<SolidNodes> resolveSolidNodes(const ElementBatchView& view, const SolidConnectivity& connectivity,
                                         std::int64_t row, SolidCardShapes shapes)
{
    using Result = Resolution<SolidNodes>;
    const std::span<const std::int32_t> ids = connectivity.ids[row];
    const std::span<const std::int32_t> rows = connectivity.rows[row];
    const std::size_t cornerCount = solidCornerCount(shapes.linear);
    const std::size_t fullCount = solidNodeCount(shapes.quadratic);

    // Which mid-side nodes are given: all or none (the element comment says why).
    std::size_t middleCount = 0;
    std::size_t firstAbsent = fullCount;
    for (std::size_t i = cornerCount; i < fullCount; ++i)
    {
        if (i < ids.size() && ids[i] != ABSENT_NODE_ID)
            ++middleCount;
        else if (firstAbsent == fullCount)
            firstAbsent = i;
    }
    const std::size_t edgeCount = fullCount - cornerCount;
    if (middleCount != 0 && middleCount != edgeCount)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, fieldOf(firstAbsent),
            base::concatenate({"expected all ", std::to_string(edgeCount), " mid-side nodes or none, found ",
                               std::to_string(middleCount),
                               "; an element with some of its mid-side nodes is not supported yet: give all of them "
                               "or none"})));

    SolidNodes nodes;
    nodes.shape = middleCount == 0 ? shapes.linear : shapes.quadratic;
    nodes.count = solidNodeCount(nodes.shape);
    const std::int64_t nodeTableRows = view.snapshot().nodes().rowCount();
    const bool isRowValid = connectivity.valid.isValid(row);
    for (std::size_t i = 0; i < nodes.count; ++i)
    {
        const std::int32_t id = i < ids.size() ? ids[i] : ABSENT_NODE_ID;
        // A null row of nodesRows does not say which grid point is unknown: the IDs tell it.
        const bool isKnown = isRowValid && i < rows.size() ? rows[i] >= 0 && rows[i] < nodeTableRows
                                                           : view.index().points().findGrid(id).has_value();
        if (id == ABSENT_NODE_ID || !isKnown)
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, fieldOf(i),
                base::concatenate({"expected the ID of a grid point, found ",
                                   id == ABSENT_NODE_ID ? std::string("blank") : std::to_string(id),
                                   ", which the model does not define; define it or correct the card"})));
        for (std::size_t j = 0; j < i; ++j)
            if (nodes.ids[j] == id)
                return Result::failure(
                    elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, fieldOf(i),
                                   base::concatenate({"expected distinct grid points, found ", std::to_string(id),
                                                      " as ", fieldOf(j), " and ", fieldOf(i), "; correct the card"})));
        nodes.ids[i] = id;
    }
    if (!isRowValid)
        throw base::ContractError(base::concatenate(
            {view.card(), " ", std::to_string(view.elementId(row)),
             ": nodesRows is null although every grid point exists; the element table and the node table come from "
             "different snapshots: take every table of the snapshot from one model"}));
    for (std::size_t i = 0; i < nodes.count; ++i)
    {
        const std::int32_t gridRow = checkedGridRow(view, rows[i], nodes.ids[i]);
        nodes.rows[i] = gridRow;
        nodes.positions[i] = view.positionInBasic(gridRow);
        nodes.cdFrames[i] = view.displacementFrame(gridRow);
    }
    return Result::success(nodes);
}

} // namespace oos::elements
